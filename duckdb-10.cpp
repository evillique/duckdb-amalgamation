// See https://raw.githubusercontent.com/duckdb/duckdb/master/LICENSE for licensing information

#include "duckdb.hpp"
#include "duckdb-internal.hpp"
#ifndef DUCKDB_AMALGAMATION
#error header mismatch
#endif












namespace duckdb {

typedef struct {
	uint32_t dict_size;
	uint32_t dict_end;
	uint32_t bitpacking_width;
	uint32_t fsst_symbol_table_offset;
} fsst_compression_header_t;

// Counts and offsets used during scanning/fetching
//                                         |               ColumnSegment to be scanned / fetched from				 |
//                                         | untouched | bp align | unused d-values | to scan | bp align | untouched |
typedef struct BPDeltaDecodeOffsets {
	idx_t delta_decode_start_row;      //                         X
	idx_t bitunpack_alignment_offset;  //			   <--------->
	idx_t bitunpack_start_row;         //	           X
	idx_t unused_delta_decoded_values; //						  <----------------->
	idx_t scan_offset;                 //			   <---------------------------->
	idx_t total_delta_decode_count;    //					      <-------------------------->
	idx_t total_bitunpack_count;       //              <------------------------------------------------>
} bp_delta_offsets_t;

struct FSSTStorage {
	static constexpr size_t COMPACTION_FLUSH_LIMIT = (size_t)Storage::BLOCK_SIZE / 5 * 4;
	static constexpr double MINIMUM_COMPRESSION_RATIO = 1.2;
	static constexpr double ANALYSIS_SAMPLE_SIZE = 0.25;

	static unique_ptr<AnalyzeState> StringInitAnalyze(ColumnData &col_data, PhysicalType type);
	static bool StringAnalyze(AnalyzeState &state_p, Vector &input, idx_t count);
	static idx_t StringFinalAnalyze(AnalyzeState &state_p);

	static unique_ptr<CompressionState> InitCompression(ColumnDataCheckpointer &checkpointer,
	                                                    unique_ptr<AnalyzeState> analyze_state_p);
	static void Compress(CompressionState &state_p, Vector &scan_vector, idx_t count);
	static void FinalizeCompress(CompressionState &state_p);

	static unique_ptr<SegmentScanState> StringInitScan(ColumnSegment &segment);
	template <bool ALLOW_FSST_VECTORS = false>
	static void StringScanPartial(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count, Vector &result,
	                              idx_t result_offset);
	static void StringScan(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count, Vector &result);
	static void StringFetchRow(ColumnSegment &segment, ColumnFetchState &state, row_t row_id, Vector &result,
	                           idx_t result_idx);

	static void SetDictionary(ColumnSegment &segment, BufferHandle &handle, StringDictionaryContainer container);
	static StringDictionaryContainer GetDictionary(ColumnSegment &segment, BufferHandle &handle);

	static char *FetchStringPointer(StringDictionaryContainer dict, data_ptr_t baseptr, int32_t dict_offset);
	static bp_delta_offsets_t CalculateBpDeltaOffsets(int64_t last_known_row, idx_t start, idx_t scan_count);
	static bool ParseFSSTSegmentHeader(data_ptr_t base_ptr, duckdb_fsst_decoder_t *decoder_out,
	                                   bitpacking_width_t *width_out);
};

//===--------------------------------------------------------------------===//
// Analyze
//===--------------------------------------------------------------------===//
struct FSSTAnalyzeState : public AnalyzeState {
	FSSTAnalyzeState() : count(0), fsst_string_total_size(0), empty_strings(0) {
	}

	~FSSTAnalyzeState() override {
		if (fsst_encoder) {
			duckdb_fsst_destroy(fsst_encoder);
		}
	}

	duckdb_fsst_encoder_t *fsst_encoder = nullptr;
	idx_t count;

	StringHeap fsst_string_heap;
	std::vector<string_t> fsst_strings;
	size_t fsst_string_total_size;

	RandomEngine random_engine;
	bool have_valid_row = false;

	idx_t empty_strings;
};

unique_ptr<AnalyzeState> FSSTStorage::StringInitAnalyze(ColumnData &col_data, PhysicalType type) {
	return make_unique<FSSTAnalyzeState>();
}

bool FSSTStorage::StringAnalyze(AnalyzeState &state_p, Vector &input, idx_t count) {
	auto &state = (FSSTAnalyzeState &)state_p;
	UnifiedVectorFormat vdata;
	input.ToUnifiedFormat(count, vdata);

	state.count += count;
	auto data = (string_t *)vdata.data;

	// Note that we ignore the sampling in case we have not found any valid strings yet, this solves the issue of
	// not having seen any valid strings here leading to an empty fsst symbol table.
	bool sample_selected = !state.have_valid_row || state.random_engine.NextRandom() < ANALYSIS_SAMPLE_SIZE;

	for (idx_t i = 0; i < count; i++) {
		auto idx = vdata.sel->get_index(i);

		if (!vdata.validity.RowIsValid(idx)) {
			continue;
		}

		// We need to check all strings for this, otherwise we run in to trouble during compression if we miss ones
		auto string_size = data[idx].GetSize();
		if (string_size >= StringUncompressed::STRING_BLOCK_LIMIT) {
			return false;
		}

		if (!sample_selected) {
			continue;
		}

		if (string_size > 0) {
			state.have_valid_row = true;
			if (data[idx].IsInlined()) {
				state.fsst_strings.push_back(data[idx]);
			} else {
				state.fsst_strings.emplace_back(state.fsst_string_heap.AddBlob(data[idx]));
			}
			state.fsst_string_total_size += string_size;
		} else {
			state.empty_strings++;
		}
	}
	return true;
}

idx_t FSSTStorage::StringFinalAnalyze(AnalyzeState &state_p) {
	auto &state = (FSSTAnalyzeState &)state_p;

	size_t compressed_dict_size = 0;
	size_t max_compressed_string_length = 0;

	auto string_count = state.fsst_strings.size();

	if (!string_count) {
		return DConstants::INVALID_INDEX;
	}

	size_t output_buffer_size = 7 + 2 * state.fsst_string_total_size; // size as specified in fsst.h

	std::vector<size_t> fsst_string_sizes;
	std::vector<unsigned char *> fsst_string_ptrs;
	for (auto &str : state.fsst_strings) {
		fsst_string_sizes.push_back(str.GetSize());
		fsst_string_ptrs.push_back((unsigned char *)str.GetDataUnsafe());
	}

	state.fsst_encoder = duckdb_fsst_create(string_count, &fsst_string_sizes[0], &fsst_string_ptrs[0], 0);

	// TODO: do we really need to encode to get a size estimate?
	auto compressed_ptrs = std::vector<unsigned char *>(string_count, nullptr);
	auto compressed_sizes = std::vector<size_t>(string_count, 0);
	unique_ptr<unsigned char[]> compressed_buffer(new unsigned char[output_buffer_size]);

	auto res =
	    duckdb_fsst_compress(state.fsst_encoder, string_count, &fsst_string_sizes[0], &fsst_string_ptrs[0],
	                         output_buffer_size, compressed_buffer.get(), &compressed_sizes[0], &compressed_ptrs[0]);

	if (string_count != res) {
		throw std::runtime_error("FSST output buffer is too small unexpectedly");
	}

	// Sum and and Max compressed lengths
	for (auto &size : compressed_sizes) {
		compressed_dict_size += size;
		max_compressed_string_length = MaxValue(max_compressed_string_length, size);
	}
	D_ASSERT(compressed_dict_size == (compressed_ptrs[res - 1] - compressed_ptrs[0]) + compressed_sizes[res - 1]);

	auto minimum_width = BitpackingPrimitives::MinimumBitWidth(max_compressed_string_length);
	auto bitpacked_offsets_size =
	    BitpackingPrimitives::GetRequiredSize(string_count + state.empty_strings, minimum_width);

	auto estimated_base_size = (bitpacked_offsets_size + compressed_dict_size) * (1 / ANALYSIS_SAMPLE_SIZE);
	auto num_blocks = estimated_base_size / (Storage::BLOCK_SIZE - sizeof(duckdb_fsst_decoder_t));
	auto symtable_size = num_blocks * sizeof(duckdb_fsst_decoder_t);

	auto estimated_size = estimated_base_size + symtable_size;

	return estimated_size * MINIMUM_COMPRESSION_RATIO;
}

//===--------------------------------------------------------------------===//
// Compress
//===--------------------------------------------------------------------===//

class FSSTCompressionState : public CompressionState {
public:
	explicit FSSTCompressionState(ColumnDataCheckpointer &checkpointer) : checkpointer(checkpointer) {
		auto &db = checkpointer.GetDatabase();
		auto &config = DBConfig::GetConfig(db);
		function = config.GetCompressionFunction(CompressionType::COMPRESSION_FSST, PhysicalType::VARCHAR);
		CreateEmptySegment(checkpointer.GetRowGroup().start);
	}

	~FSSTCompressionState() override {
		if (fsst_encoder) {
			duckdb_fsst_destroy(fsst_encoder);
		}
	}

	void Reset() {
		index_buffer.clear();
		current_width = 0;
		max_compressed_string_length = 0;
		last_fitting_size = 0;

		// Reset the pointers into the current segment
		auto &buffer_manager = BufferManager::GetBufferManager(current_segment->db);
		current_handle = buffer_manager.Pin(current_segment->block);
		current_dictionary = FSSTStorage::GetDictionary(*current_segment, current_handle);
		current_end_ptr = current_handle.Ptr() + current_dictionary.end;
	}

	void CreateEmptySegment(idx_t row_start) {
		auto &db = checkpointer.GetDatabase();
		auto &type = checkpointer.GetType();
		auto compressed_segment = ColumnSegment::CreateTransientSegment(db, type, row_start);
		current_segment = std::move(compressed_segment);
		current_segment->function = function;
		Reset();
	}

	void UpdateState(string_t uncompressed_string, unsigned char *compressed_string, size_t compressed_string_len) {
		if (!HasEnoughSpace(compressed_string_len)) {
			Flush();
			if (!HasEnoughSpace(compressed_string_len)) {
				throw InternalException("FSST string compression failed due to insufficient space in empty block");
			};
		}

		UncompressedStringStorage::UpdateStringStats(current_segment->stats, uncompressed_string);

		// Write string into dictionary
		current_dictionary.size += compressed_string_len;
		auto dict_pos = current_end_ptr - current_dictionary.size;
		memcpy(dict_pos, compressed_string, compressed_string_len);
		current_dictionary.Verify();

		// We just push the string length to effectively delta encode the strings
		index_buffer.push_back(compressed_string_len);

		max_compressed_string_length = MaxValue(max_compressed_string_length, compressed_string_len);

		current_width = BitpackingPrimitives::MinimumBitWidth(max_compressed_string_length);
		current_segment->count++;
	}

	void AddNull() {
		if (!HasEnoughSpace(0)) {
			Flush();
			if (!HasEnoughSpace(0)) {
				throw InternalException("FSST string compression failed due to insufficient space in empty block");
			};
		}
		index_buffer.push_back(0);
		current_segment->count++;
	}

	void AddEmptyString() {
		AddNull();
		UncompressedStringStorage::UpdateStringStats(current_segment->stats, "");
	}

	size_t GetRequiredSize(size_t string_len) {
		bitpacking_width_t required_minimum_width;
		if (string_len > max_compressed_string_length) {
			required_minimum_width = BitpackingPrimitives::MinimumBitWidth(string_len);
		} else {
			required_minimum_width = current_width;
		}

		size_t current_dict_size = current_dictionary.size;
		idx_t current_string_count = index_buffer.size();

		size_t dict_offsets_size =
		    BitpackingPrimitives::GetRequiredSize(current_string_count + 1, required_minimum_width);

		// TODO switch to a symbol table per RowGroup, saves a bit of space
		return sizeof(fsst_compression_header_t) + current_dict_size + dict_offsets_size + string_len +
		       fsst_serialized_symbol_table_size;
	}

	// Checks if there is enough space, if there is, sets last_fitting_size
	bool HasEnoughSpace(size_t string_len) {
		auto required_size = GetRequiredSize(string_len);

		if (required_size <= Storage::BLOCK_SIZE) {
			last_fitting_size = required_size;
			return true;
		}
		return false;
	}

	void Flush(bool final = false) {
		auto next_start = current_segment->start + current_segment->count;

		auto segment_size = Finalize();
		auto &state = checkpointer.GetCheckpointState();
		state.FlushSegment(std::move(current_segment), segment_size);

		if (!final) {
			CreateEmptySegment(next_start);
		}
	}

	idx_t Finalize() {
		auto &buffer_manager = BufferManager::GetBufferManager(current_segment->db);
		auto handle = buffer_manager.Pin(current_segment->block);
		D_ASSERT(current_dictionary.end == Storage::BLOCK_SIZE);

		// calculate sizes
		auto compressed_index_buffer_size =
		    BitpackingPrimitives::GetRequiredSize(current_segment->count, current_width);
		auto total_size = sizeof(fsst_compression_header_t) + compressed_index_buffer_size + current_dictionary.size +
		                  fsst_serialized_symbol_table_size;

		if (total_size != last_fitting_size) {
			throw InternalException("FSST string compression failed due to incorrect size calculation");
		}

		// calculate ptr and offsets
		auto base_ptr = handle.Ptr();
		auto header_ptr = (fsst_compression_header_t *)base_ptr;
		auto compressed_index_buffer_offset = sizeof(fsst_compression_header_t);
		auto symbol_table_offset = compressed_index_buffer_offset + compressed_index_buffer_size;

		D_ASSERT(current_segment->count == index_buffer.size());
		BitpackingPrimitives::PackBuffer<sel_t, false>(base_ptr + compressed_index_buffer_offset,
		                                               (uint32_t *)(index_buffer.data()), current_segment->count,
		                                               current_width);

		// Write the fsst symbol table or nothing
		if (fsst_encoder != nullptr) {
			memcpy(base_ptr + symbol_table_offset, &fsst_serialized_symbol_table[0], fsst_serialized_symbol_table_size);
		} else {
			memset(base_ptr + symbol_table_offset, 0, fsst_serialized_symbol_table_size);
		}

		Store<uint32_t>(symbol_table_offset, (data_ptr_t)&header_ptr->fsst_symbol_table_offset);
		Store<uint32_t>((uint32_t)current_width, (data_ptr_t)&header_ptr->bitpacking_width);

		if (total_size >= FSSTStorage::COMPACTION_FLUSH_LIMIT) {
			// the block is full enough, don't bother moving around the dictionary
			return Storage::BLOCK_SIZE;
		}
		// the block has space left: figure out how much space we can save
		auto move_amount = Storage::BLOCK_SIZE - total_size;
		// move the dictionary so it lines up exactly with the offsets
		auto new_dictionary_offset = symbol_table_offset + fsst_serialized_symbol_table_size;
		memmove(base_ptr + new_dictionary_offset, base_ptr + current_dictionary.end - current_dictionary.size,
		        current_dictionary.size);
		current_dictionary.end -= move_amount;
		D_ASSERT(current_dictionary.end == total_size);
		// write the new dictionary (with the updated "end")
		FSSTStorage::SetDictionary(*current_segment, handle, current_dictionary);

		return total_size;
	}

	ColumnDataCheckpointer &checkpointer;
	CompressionFunction *function;

	// State regarding current segment
	unique_ptr<ColumnSegment> current_segment;
	BufferHandle current_handle;
	StringDictionaryContainer current_dictionary;
	data_ptr_t current_end_ptr;

	// Buffers and map for current segment
	std::vector<uint32_t> index_buffer;

	size_t max_compressed_string_length;
	bitpacking_width_t current_width;
	idx_t last_fitting_size;

	duckdb_fsst_encoder_t *fsst_encoder = nullptr;
	unsigned char fsst_serialized_symbol_table[sizeof(duckdb_fsst_decoder_t)];
	size_t fsst_serialized_symbol_table_size = sizeof(duckdb_fsst_decoder_t);
};

unique_ptr<CompressionState> FSSTStorage::InitCompression(ColumnDataCheckpointer &checkpointer,
                                                          unique_ptr<AnalyzeState> analyze_state_p) {
	auto analyze_state = static_cast<FSSTAnalyzeState *>(analyze_state_p.get());
	auto compression_state = make_unique<FSSTCompressionState>(checkpointer);

	if (analyze_state->fsst_encoder == nullptr) {
		throw InternalException("No encoder found during FSST compression");
	}

	compression_state->fsst_encoder = analyze_state->fsst_encoder;
	compression_state->fsst_serialized_symbol_table_size =
	    duckdb_fsst_export(compression_state->fsst_encoder, &compression_state->fsst_serialized_symbol_table[0]);
	analyze_state->fsst_encoder = nullptr;

	return std::move(compression_state);
}

void FSSTStorage::Compress(CompressionState &state_p, Vector &scan_vector, idx_t count) {
	auto &state = (FSSTCompressionState &)state_p;

	// Get vector data
	UnifiedVectorFormat vdata;
	scan_vector.ToUnifiedFormat(count, vdata);
	auto data = (string_t *)vdata.data;

	// Collect pointers to strings to compress
	vector<size_t> sizes_in;
	vector<unsigned char *> strings_in;
	size_t total_size = 0;
	idx_t total_count = 0;
	for (idx_t i = 0; i < count; i++) {
		auto idx = vdata.sel->get_index(i);

		// Note: we treat nulls and empty strings the same
		if (!vdata.validity.RowIsValid(idx) || data[idx].GetSize() == 0) {
			continue;
		}

		total_count++;
		total_size += data[idx].GetSize();
		sizes_in.push_back(data[idx].GetSize());
		strings_in.push_back((unsigned char *)data[idx].GetDataUnsafe());
	}

	// Only Nulls or empty strings in this vector, nothing to compress
	if (total_count == 0) {
		for (idx_t i = 0; i < count; i++) {
			auto idx = vdata.sel->get_index(i);
			if (!vdata.validity.RowIsValid(idx)) {
				state.AddNull();
			} else if (data[idx].GetSize() == 0) {
				state.AddEmptyString();
			} else {
				throw FatalException("FSST: no encoder found even though there are values to encode");
			}
		}
		return;
	}

	// Compress buffers
	size_t compress_buffer_size = MaxValue<size_t>(total_size * 2 + 7, 1);
	vector<unsigned char *> strings_out(total_count, nullptr);
	vector<size_t> sizes_out(total_count, 0);
	vector<unsigned char> compress_buffer(compress_buffer_size, 0);

	auto res = duckdb_fsst_compress(
	    state.fsst_encoder,   /* IN: encoder obtained from duckdb_fsst_create(). */
	    total_count,          /* IN: number of strings in batch to compress. */
	    &sizes_in[0],         /* IN: byte-lengths of the inputs */
	    &strings_in[0],       /* IN: input string start pointers. */
	    compress_buffer_size, /* IN: byte-length of output buffer. */
	    &compress_buffer[0],  /* OUT: memorxy buffer to put the compressed strings in (one after the other). */
	    &sizes_out[0],        /* OUT: byte-lengths of the compressed strings. */
	    &strings_out[0]       /* OUT: output string start pointers. Will all point into [output,output+size). */
	);

	if (res != total_count) {
		throw FatalException("FSST compression failed to compress all strings");
	}

	// Push the compressed strings to the compression state one by one
	idx_t compressed_idx = 0;
	for (idx_t i = 0; i < count; i++) {
		auto idx = vdata.sel->get_index(i);
		if (!vdata.validity.RowIsValid(idx)) {
			state.AddNull();
		} else if (data[idx].GetSize() == 0) {
			state.AddEmptyString();
		} else {
			state.UpdateState(data[idx], strings_out[compressed_idx], sizes_out[compressed_idx]);
			compressed_idx++;
		}
	}
}

void FSSTStorage::FinalizeCompress(CompressionState &state_p) {
	auto &state = (FSSTCompressionState &)state_p;
	state.Flush(true);
}

//===--------------------------------------------------------------------===//
// Scan
//===--------------------------------------------------------------------===//
struct FSSTScanState : public StringScanState {
	FSSTScanState() {
		ResetStoredDelta();
	}

	buffer_ptr<void> duckdb_fsst_decoder;
	bitpacking_width_t current_width;

	// To speed up delta decoding we store the last index
	uint32_t last_known_index;
	int64_t last_known_row;

	void StoreLastDelta(uint32_t value, int64_t row) {
		last_known_index = value;
		last_known_row = row;
	}
	void ResetStoredDelta() {
		last_known_index = 0;
		last_known_row = -1;
	}
};

unique_ptr<SegmentScanState> FSSTStorage::StringInitScan(ColumnSegment &segment) {
	auto state = make_unique<FSSTScanState>();
	auto &buffer_manager = BufferManager::GetBufferManager(segment.db);
	state->handle = buffer_manager.Pin(segment.block);
	auto base_ptr = state->handle.Ptr() + segment.GetBlockOffset();

	state->duckdb_fsst_decoder = make_buffer<duckdb_fsst_decoder_t>();
	auto retval = ParseFSSTSegmentHeader(base_ptr, (duckdb_fsst_decoder_t *)state->duckdb_fsst_decoder.get(),
	                                     &state->current_width);
	if (!retval) {
		state->duckdb_fsst_decoder = nullptr;
	}

	return std::move(state);
}

void DeltaDecodeIndices(uint32_t *buffer_in, uint32_t *buffer_out, idx_t decode_count, uint32_t last_known_value) {
	buffer_out[0] = buffer_in[0];
	buffer_out[0] += last_known_value;
	for (idx_t i = 1; i < decode_count; i++) {
		buffer_out[i] = buffer_in[i] + buffer_out[i - 1];
	}
}

void BitUnpackRange(data_ptr_t src_ptr, data_ptr_t dst_ptr, idx_t count, idx_t row, bitpacking_width_t width) {
	auto bitunpack_src_ptr = &src_ptr[(row * width) / 8];
	BitpackingPrimitives::UnPackBuffer<uint32_t>(dst_ptr, bitunpack_src_ptr, count, width);
}

//===--------------------------------------------------------------------===//
// Scan base data
//===--------------------------------------------------------------------===//
template <bool ALLOW_FSST_VECTORS>
void FSSTStorage::StringScanPartial(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count, Vector &result,
                                    idx_t result_offset) {

	auto &scan_state = (FSSTScanState &)*state.scan_state;
	auto start = segment.GetRelativeIndex(state.row_index);

	bool enable_fsst_vectors;
	if (ALLOW_FSST_VECTORS) {
		auto &config = DBConfig::GetConfig(segment.db);
		enable_fsst_vectors = config.options.enable_fsst_vectors;
	} else {
		enable_fsst_vectors = false;
	}

	auto baseptr = scan_state.handle.Ptr() + segment.GetBlockOffset();
	auto dict = GetDictionary(segment, scan_state.handle);
	auto base_data = (data_ptr_t)(baseptr + sizeof(fsst_compression_header_t));
	string_t *result_data;

	if (scan_count == 0) {
		return;
	}

	if (enable_fsst_vectors) {
		D_ASSERT(result_offset == 0);
		if (scan_state.duckdb_fsst_decoder) {
			D_ASSERT(result_offset == 0 || result.GetVectorType() == VectorType::FSST_VECTOR);
			result.SetVectorType(VectorType::FSST_VECTOR);
			FSSTVector::RegisterDecoder(result, scan_state.duckdb_fsst_decoder);
			result_data = FSSTVector::GetCompressedData<string_t>(result);
		} else {
			D_ASSERT(result.GetVectorType() == VectorType::FLAT_VECTOR);
			result_data = FlatVector::GetData<string_t>(result);
		}
	} else {
		D_ASSERT(result.GetVectorType() == VectorType::FLAT_VECTOR);
		result_data = FlatVector::GetData<string_t>(result);
	}

	if (start == 0 || scan_state.last_known_row >= (int64_t)start) {
		scan_state.ResetStoredDelta();
	}

	auto offsets = CalculateBpDeltaOffsets(scan_state.last_known_row, start, scan_count);

	auto bitunpack_buffer = unique_ptr<uint32_t[]>(new uint32_t[offsets.total_bitunpack_count]);
	BitUnpackRange(base_data, (data_ptr_t)bitunpack_buffer.get(), offsets.total_bitunpack_count,
	               offsets.bitunpack_start_row, scan_state.current_width);
	auto delta_decode_buffer = unique_ptr<uint32_t[]>(new uint32_t[offsets.total_delta_decode_count]);
	DeltaDecodeIndices(bitunpack_buffer.get() + offsets.bitunpack_alignment_offset, delta_decode_buffer.get(),
	                   offsets.total_delta_decode_count, scan_state.last_known_index);

	if (enable_fsst_vectors) {
		// Lookup decompressed offsets in dict
		for (idx_t i = 0; i < scan_count; i++) {
			uint32_t string_length = bitunpack_buffer[i + offsets.scan_offset];
			result_data[i] = UncompressedStringStorage::FetchStringFromDict(
			    segment, dict, result, baseptr, delta_decode_buffer[i + offsets.unused_delta_decoded_values],
			    string_length);
			FSSTVector::SetCount(result, scan_count);
		}
	} else {
		// Just decompress
		for (idx_t i = 0; i < scan_count; i++) {
			uint32_t str_len = bitunpack_buffer[i + offsets.scan_offset];
			auto str_ptr = FSSTStorage::FetchStringPointer(
			    dict, baseptr, delta_decode_buffer[i + offsets.unused_delta_decoded_values]);

			if (str_len > 0) {
				result_data[i + result_offset] = FSSTPrimitives::DecompressValue(
				    scan_state.duckdb_fsst_decoder.get(), result, (unsigned char *)str_ptr, str_len);
			} else {
				result_data[i + result_offset] = string_t(nullptr, 0);
			}
		}
	}

	scan_state.StoreLastDelta(delta_decode_buffer[scan_count + offsets.unused_delta_decoded_values - 1],
	                          start + scan_count - 1);
}

void FSSTStorage::StringScan(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count, Vector &result) {
	StringScanPartial<true>(segment, state, scan_count, result, 0);
}

//===--------------------------------------------------------------------===//
// Fetch
//===--------------------------------------------------------------------===//
void FSSTStorage::StringFetchRow(ColumnSegment &segment, ColumnFetchState &state, row_t row_id, Vector &result,
                                 idx_t result_idx) {

	auto &buffer_manager = BufferManager::GetBufferManager(segment.db);
	auto handle = buffer_manager.Pin(segment.block);
	auto base_ptr = handle.Ptr() + segment.GetBlockOffset();
	auto base_data = (data_ptr_t)(base_ptr + sizeof(fsst_compression_header_t));
	auto dict = GetDictionary(segment, handle);

	duckdb_fsst_decoder_t decoder;
	bitpacking_width_t width;
	auto have_symbol_table = ParseFSSTSegmentHeader(base_ptr, &decoder, &width);

	auto result_data = FlatVector::GetData<string_t>(result);

	if (have_symbol_table) {
		// We basically just do a scan of 1 which is kinda expensive as we need to repeatedly delta decode until we
		// reach the row we want, we could consider a more clever caching trick if this is slow
		auto offsets = CalculateBpDeltaOffsets(-1, row_id, 1);

		auto bitunpack_buffer = unique_ptr<uint32_t[]>(new uint32_t[offsets.total_bitunpack_count]);
		BitUnpackRange(base_data, (data_ptr_t)bitunpack_buffer.get(), offsets.total_bitunpack_count,
		               offsets.bitunpack_start_row, width);
		auto delta_decode_buffer = unique_ptr<uint32_t[]>(new uint32_t[offsets.total_delta_decode_count]);
		DeltaDecodeIndices(bitunpack_buffer.get() + offsets.bitunpack_alignment_offset, delta_decode_buffer.get(),
		                   offsets.total_delta_decode_count, 0);

		uint32_t string_length = bitunpack_buffer[offsets.scan_offset];

		string_t compressed_string = UncompressedStringStorage::FetchStringFromDict(
		    segment, dict, result, base_ptr, delta_decode_buffer[offsets.unused_delta_decoded_values], string_length);

		result_data[result_idx] = FSSTPrimitives::DecompressValue(
		    (void *)&decoder, result, (unsigned char *)compressed_string.GetDataUnsafe(), compressed_string.GetSize());
	} else {
		// There's no fsst symtable, this only happens for empty strings or nulls, we can just emit an empty string
		result_data[result_idx] = string_t(nullptr, 0);
	}
}

//===--------------------------------------------------------------------===//
// Get Function
//===--------------------------------------------------------------------===//
CompressionFunction FSSTFun::GetFunction(PhysicalType data_type) {
	D_ASSERT(data_type == PhysicalType::VARCHAR);
	return CompressionFunction(
	    CompressionType::COMPRESSION_FSST, data_type, FSSTStorage::StringInitAnalyze, FSSTStorage::StringAnalyze,
	    FSSTStorage::StringFinalAnalyze, FSSTStorage::InitCompression, FSSTStorage::Compress,
	    FSSTStorage::FinalizeCompress, FSSTStorage::StringInitScan, FSSTStorage::StringScan,
	    FSSTStorage::StringScanPartial<false>, FSSTStorage::StringFetchRow, UncompressedFunctions::EmptySkip);
}

bool FSSTFun::TypeIsSupported(PhysicalType type) {
	return type == PhysicalType::VARCHAR;
}

//===--------------------------------------------------------------------===//
// Helper Functions
//===--------------------------------------------------------------------===//
void FSSTStorage::SetDictionary(ColumnSegment &segment, BufferHandle &handle, StringDictionaryContainer container) {
	auto header_ptr = (fsst_compression_header_t *)(handle.Ptr() + segment.GetBlockOffset());
	Store<uint32_t>(container.size, (data_ptr_t)&header_ptr->dict_size);
	Store<uint32_t>(container.end, (data_ptr_t)&header_ptr->dict_end);
}

StringDictionaryContainer FSSTStorage::GetDictionary(ColumnSegment &segment, BufferHandle &handle) {
	auto header_ptr = (fsst_compression_header_t *)(handle.Ptr() + segment.GetBlockOffset());
	StringDictionaryContainer container;
	container.size = Load<uint32_t>((data_ptr_t)&header_ptr->dict_size);
	container.end = Load<uint32_t>((data_ptr_t)&header_ptr->dict_end);
	return container;
}

char *FSSTStorage::FetchStringPointer(StringDictionaryContainer dict, data_ptr_t baseptr, int32_t dict_offset) {
	if (dict_offset == 0) {
		return nullptr;
	}

	auto dict_end = baseptr + dict.end;
	auto dict_pos = dict_end - dict_offset;
	return (char *)(dict_pos);
}

// Returns false if no symbol table was found. This means all strings are either empty or null
bool FSSTStorage::ParseFSSTSegmentHeader(data_ptr_t base_ptr, duckdb_fsst_decoder_t *decoder_out,
                                         bitpacking_width_t *width_out) {
	auto header_ptr = (fsst_compression_header_t *)base_ptr;
	auto fsst_symbol_table_offset = Load<uint32_t>((data_ptr_t)&header_ptr->fsst_symbol_table_offset);
	*width_out = (bitpacking_width_t)(Load<uint32_t>((data_ptr_t)&header_ptr->bitpacking_width));
	return duckdb_fsst_import(decoder_out, base_ptr + fsst_symbol_table_offset);
}

// The calculation of offsets and counts while scanning or fetching is a bit tricky, for two reasons:
// - bitunpacking needs to be aligned to BITPACKING_ALGORITHM_GROUP_SIZE
// - delta decoding needs to decode from the last known value.
bp_delta_offsets_t FSSTStorage::CalculateBpDeltaOffsets(int64_t last_known_row, idx_t start, idx_t scan_count) {
	D_ASSERT((idx_t)(last_known_row + 1) <= start);
	bp_delta_offsets_t result;

	result.delta_decode_start_row = (idx_t)(last_known_row + 1);
	result.bitunpack_alignment_offset =
	    result.delta_decode_start_row % BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE;
	result.bitunpack_start_row = result.delta_decode_start_row - result.bitunpack_alignment_offset;
	result.unused_delta_decoded_values = start - result.delta_decode_start_row;
	result.scan_offset = result.bitunpack_alignment_offset + result.unused_delta_decoded_values;
	result.total_delta_decode_count = scan_count + result.unused_delta_decoded_values;
	result.total_bitunpack_count =
	    BitpackingPrimitives::RoundUpToAlgorithmGroupSize<idx_t>(scan_count + result.scan_offset);

	D_ASSERT(result.total_delta_decode_count + result.bitunpack_alignment_offset <= result.total_bitunpack_count);
	return result;
}

} // namespace duckdb









namespace duckdb {

//===--------------------------------------------------------------------===//
// Scan
//===--------------------------------------------------------------------===//
unique_ptr<SegmentScanState> ConstantInitScan(ColumnSegment &segment) {
	return nullptr;
}

//===--------------------------------------------------------------------===//
// Scan Partial
//===--------------------------------------------------------------------===//
void ConstantFillFunctionValidity(ColumnSegment &segment, Vector &result, idx_t start_idx, idx_t count) {
	auto &validity = (ValidityStatistics &)*segment.stats.statistics;
	if (validity.has_null) {
		auto &mask = FlatVector::Validity(result);
		for (idx_t i = 0; i < count; i++) {
			mask.SetInvalid(start_idx + i);
		}
	}
}

template <class T>
void ConstantFillFunction(ColumnSegment &segment, Vector &result, idx_t start_idx, idx_t count) {
	auto &nstats = (NumericStatistics &)*segment.stats.statistics;

	auto data = FlatVector::GetData<T>(result);
	auto constant_value = nstats.min.GetValueUnsafe<T>();
	for (idx_t i = 0; i < count; i++) {
		data[start_idx + i] = constant_value;
	}
}

void ConstantScanPartialValidity(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count, Vector &result,
                                 idx_t result_offset) {
	ConstantFillFunctionValidity(segment, result, result_offset, scan_count);
}

template <class T>
void ConstantScanPartial(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count, Vector &result,
                         idx_t result_offset) {
	ConstantFillFunction<T>(segment, result, result_offset, scan_count);
}

//===--------------------------------------------------------------------===//
// Scan base data
//===--------------------------------------------------------------------===//
void ConstantScanFunctionValidity(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count, Vector &result) {
	auto &validity = (ValidityStatistics &)*segment.stats.statistics;
	if (validity.has_null) {
		if (result.GetVectorType() == VectorType::CONSTANT_VECTOR) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
		} else {
			result.Flatten(scan_count);
			ConstantFillFunctionValidity(segment, result, 0, scan_count);
		}
	}
}

template <class T>
void ConstantScanFunction(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count, Vector &result) {
	auto &nstats = (NumericStatistics &)*segment.stats.statistics;

	auto data = FlatVector::GetData<T>(result);
	data[0] = nstats.min.GetValueUnsafe<T>();
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
}

//===--------------------------------------------------------------------===//
// Fetch
//===--------------------------------------------------------------------===//
void ConstantFetchRowValidity(ColumnSegment &segment, ColumnFetchState &state, row_t row_id, Vector &result,
                              idx_t result_idx) {
	ConstantFillFunctionValidity(segment, result, result_idx, 1);
}

template <class T>
void ConstantFetchRow(ColumnSegment &segment, ColumnFetchState &state, row_t row_id, Vector &result, idx_t result_idx) {
	ConstantFillFunction<T>(segment, result, result_idx, 1);
}

//===--------------------------------------------------------------------===//
// Get Function
//===--------------------------------------------------------------------===//
CompressionFunction ConstantGetFunctionValidity(PhysicalType data_type) {
	D_ASSERT(data_type == PhysicalType::BIT);
	return CompressionFunction(CompressionType::COMPRESSION_CONSTANT, data_type, nullptr, nullptr, nullptr, nullptr,
	                           nullptr, nullptr, ConstantInitScan, ConstantScanFunctionValidity,
	                           ConstantScanPartialValidity, ConstantFetchRowValidity, UncompressedFunctions::EmptySkip);
}

template <class T>
CompressionFunction ConstantGetFunction(PhysicalType data_type) {
	return CompressionFunction(CompressionType::COMPRESSION_CONSTANT, data_type, nullptr, nullptr, nullptr, nullptr,
	                           nullptr, nullptr, ConstantInitScan, ConstantScanFunction<T>, ConstantScanPartial<T>,
	                           ConstantFetchRow<T>, UncompressedFunctions::EmptySkip);
}

CompressionFunction ConstantFun::GetFunction(PhysicalType data_type) {
	switch (data_type) {
	case PhysicalType::BIT:
		return ConstantGetFunctionValidity(data_type);
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		return ConstantGetFunction<int8_t>(data_type);
	case PhysicalType::INT16:
		return ConstantGetFunction<int16_t>(data_type);
	case PhysicalType::INT32:
		return ConstantGetFunction<int32_t>(data_type);
	case PhysicalType::INT64:
		return ConstantGetFunction<int64_t>(data_type);
	case PhysicalType::UINT8:
		return ConstantGetFunction<uint8_t>(data_type);
	case PhysicalType::UINT16:
		return ConstantGetFunction<uint16_t>(data_type);
	case PhysicalType::UINT32:
		return ConstantGetFunction<uint32_t>(data_type);
	case PhysicalType::UINT64:
		return ConstantGetFunction<uint64_t>(data_type);
	case PhysicalType::INT128:
		return ConstantGetFunction<hugeint_t>(data_type);
	case PhysicalType::FLOAT:
		return ConstantGetFunction<float>(data_type);
	case PhysicalType::DOUBLE:
		return ConstantGetFunction<double>(data_type);
	default:
		throw InternalException("Unsupported type for ConstantUncompressed::GetFunction");
	}
}

bool ConstantFun::TypeIsSupported(PhysicalType type) {
	switch (type) {
	case PhysicalType::BIT:
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
	case PhysicalType::INT16:
	case PhysicalType::INT32:
	case PhysicalType::INT64:
	case PhysicalType::UINT8:
	case PhysicalType::UINT16:
	case PhysicalType::UINT32:
	case PhysicalType::UINT64:
	case PhysicalType::INT128:
	case PhysicalType::FLOAT:
	case PhysicalType::DOUBLE:
		return true;
	default:
		throw InternalException("Unsupported type for constant function");
	}
}

} // namespace duckdb

















#include <functional>

namespace duckdb {

template <class T>
CompressionFunction GetPatasFunction(PhysicalType data_type) {
	throw NotImplementedException("GetPatasFunction not implemented for the given datatype");
}

template <>
CompressionFunction GetPatasFunction<float>(PhysicalType data_type) {
	return CompressionFunction(CompressionType::COMPRESSION_PATAS, data_type, PatasInitAnalyze<float>,
	                           PatasAnalyze<float>, PatasFinalAnalyze<float>, PatasInitCompression<float>,
	                           PatasCompress<float>, PatasFinalizeCompress<float>, PatasInitScan<float>,
	                           PatasScan<float>, PatasScanPartial<float>, PatasFetchRow<float>, PatasSkip<float>);
}

template <>
CompressionFunction GetPatasFunction<double>(PhysicalType data_type) {
	return CompressionFunction(CompressionType::COMPRESSION_PATAS, data_type, PatasInitAnalyze<double>,
	                           PatasAnalyze<double>, PatasFinalAnalyze<double>, PatasInitCompression<double>,
	                           PatasCompress<double>, PatasFinalizeCompress<double>, PatasInitScan<double>,
	                           PatasScan<double>, PatasScanPartial<double>, PatasFetchRow<double>, PatasSkip<double>);
}

CompressionFunction PatasCompressionFun::GetFunction(PhysicalType type) {
	switch (type) {
	case PhysicalType::FLOAT:
		return GetPatasFunction<float>(type);
	case PhysicalType::DOUBLE:
		return GetPatasFunction<double>(type);
	default:
		throw InternalException("Unsupported type for Patas");
	}
}

bool PatasCompressionFun::TypeIsSupported(PhysicalType type) {
	switch (type) {
	case PhysicalType::FLOAT:
	case PhysicalType::DOUBLE:
		return true;
	default:
		return false;
	}
}

} // namespace duckdb








#include <functional>

namespace duckdb {

using rle_count_t = uint16_t;

//===--------------------------------------------------------------------===//
// Analyze
//===--------------------------------------------------------------------===//
struct EmptyRLEWriter {
	template <class VALUE_TYPE>
	static void Operation(VALUE_TYPE value, rle_count_t count, void *dataptr, bool is_null) {
	}
};

template <class T>
struct RLEState {
	RLEState() : seen_count(0), last_value(NullValue<T>()), last_seen_count(0), dataptr(nullptr) {
	}

	idx_t seen_count;
	T last_value;
	rle_count_t last_seen_count;
	void *dataptr;
	bool all_null = true;

public:
	template <class OP>
	void Flush() {
		OP::template Operation<T>(last_value, last_seen_count, dataptr, all_null);
	}

	template <class OP = EmptyRLEWriter>
	void Update(T *data, ValidityMask &validity, idx_t idx) {
		if (validity.RowIsValid(idx)) {
			if (all_null) {
				// no value seen yet
				// assign the current value, and increment the seen_count
				// note that we increment last_seen_count rather than setting it to 1
				// this is intentional: this is the first VALID value we see
				// but it might not be the first value in case of nulls!
				last_value = data[idx];
				seen_count++;
				last_seen_count++;
				all_null = false;
			} else if (last_value == data[idx]) {
				// the last value is identical to this value: increment the last_seen_count
				last_seen_count++;
			} else {
				// the values are different
				// issue the callback on the last value
				Flush<OP>();

				// increment the seen_count and put the new value into the RLE slot
				last_value = data[idx];
				seen_count++;
				last_seen_count = 1;
			}
		} else {
			// NULL value: we merely increment the last_seen_count
			last_seen_count++;
		}
		if (last_seen_count == NumericLimits<rle_count_t>::Maximum()) {
			// we have seen the same value so many times in a row we are at the limit of what fits in our count
			// write away the value and move to the next value
			Flush<OP>();
			last_seen_count = 0;
			seen_count++;
		}
	}
};

template <class T>
struct RLEAnalyzeState : public AnalyzeState {
	RLEAnalyzeState() {
	}

	RLEState<T> state;
};

template <class T>
unique_ptr<AnalyzeState> RLEInitAnalyze(ColumnData &col_data, PhysicalType type) {
	return make_unique<RLEAnalyzeState<T>>();
}

template <class T>
bool RLEAnalyze(AnalyzeState &state, Vector &input, idx_t count) {
	auto &rle_state = (RLEAnalyzeState<T> &)state;
	UnifiedVectorFormat vdata;
	input.ToUnifiedFormat(count, vdata);

	auto data = (T *)vdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto idx = vdata.sel->get_index(i);
		rle_state.state.Update(data, vdata.validity, idx);
	}
	return true;
}

template <class T>
idx_t RLEFinalAnalyze(AnalyzeState &state) {
	auto &rle_state = (RLEAnalyzeState<T> &)state;
	return (sizeof(rle_count_t) + sizeof(T)) * rle_state.state.seen_count;
}

//===--------------------------------------------------------------------===//
// Compress
//===--------------------------------------------------------------------===//
struct RLEConstants {
	static constexpr const idx_t RLE_HEADER_SIZE = sizeof(uint64_t);
};

template <class T>
struct RLECompressState : public CompressionState {
	struct RLEWriter {
		template <class VALUE_TYPE>
		static void Operation(VALUE_TYPE value, rle_count_t count, void *dataptr, bool is_null) {
			auto state = (RLECompressState<T> *)dataptr;
			state->WriteValue(value, count, is_null);
		}
	};

	static idx_t MaxRLECount() {
		auto entry_size = sizeof(T) + sizeof(rle_count_t);
		auto entry_count = (Storage::BLOCK_SIZE - RLEConstants::RLE_HEADER_SIZE) / entry_size;
		auto max_vector_count = entry_count / STANDARD_VECTOR_SIZE;
		return max_vector_count * STANDARD_VECTOR_SIZE;
	}

	explicit RLECompressState(ColumnDataCheckpointer &checkpointer_p) : checkpointer(checkpointer_p) {
		auto &db = checkpointer.GetDatabase();
		auto &type = checkpointer.GetType();
		auto &config = DBConfig::GetConfig(db);
		function = config.GetCompressionFunction(CompressionType::COMPRESSION_RLE, type.InternalType());
		CreateEmptySegment(checkpointer.GetRowGroup().start);

		state.dataptr = (void *)this;
		max_rle_count = MaxRLECount();
	}

	void CreateEmptySegment(idx_t row_start) {
		auto &db = checkpointer.GetDatabase();
		auto &type = checkpointer.GetType();
		auto column_segment = ColumnSegment::CreateTransientSegment(db, type, row_start);
		column_segment->function = function;
		current_segment = std::move(column_segment);
		auto &buffer_manager = BufferManager::GetBufferManager(db);
		handle = buffer_manager.Pin(current_segment->block);
	}

	void Append(UnifiedVectorFormat &vdata, idx_t count) {
		auto data = (T *)vdata.data;
		for (idx_t i = 0; i < count; i++) {
			auto idx = vdata.sel->get_index(i);
			state.template Update<RLECompressState<T>::RLEWriter>(data, vdata.validity, idx);
		}
	}

	void WriteValue(T value, rle_count_t count, bool is_null) {
		// write the RLE entry
		auto handle_ptr = handle.Ptr() + RLEConstants::RLE_HEADER_SIZE;
		auto data_pointer = (T *)handle_ptr;
		auto index_pointer = (rle_count_t *)(handle_ptr + max_rle_count * sizeof(T));
		data_pointer[entry_count] = value;
		index_pointer[entry_count] = count;
		entry_count++;

		// update meta data
		if (!is_null) {
			NumericStatistics::Update<T>(current_segment->stats, value);
		}
		current_segment->count += count;

		if (entry_count == max_rle_count) {
			// we have finished writing this segment: flush it and create a new segment
			auto row_start = current_segment->start + current_segment->count;
			FlushSegment();
			CreateEmptySegment(row_start);
			entry_count = 0;
		}
	}

	void FlushSegment() {
		// flush the segment
		// we compact the segment by moving the counts so they are directly next to the values
		idx_t counts_size = sizeof(rle_count_t) * entry_count;
		idx_t original_rle_offset = RLEConstants::RLE_HEADER_SIZE + max_rle_count * sizeof(T);
		idx_t minimal_rle_offset = AlignValue(RLEConstants::RLE_HEADER_SIZE + sizeof(T) * entry_count);
		idx_t total_segment_size = minimal_rle_offset + counts_size;
		auto data_ptr = handle.Ptr();
		memmove(data_ptr + minimal_rle_offset, data_ptr + original_rle_offset, counts_size);
		// store the final RLE offset within the segment
		Store<uint64_t>(minimal_rle_offset, data_ptr);
		handle.Destroy();

		auto &state = checkpointer.GetCheckpointState();
		state.FlushSegment(std::move(current_segment), total_segment_size);
	}

	void Finalize() {
		state.template Flush<RLECompressState<T>::RLEWriter>();

		FlushSegment();
		current_segment.reset();
	}

	ColumnDataCheckpointer &checkpointer;
	CompressionFunction *function;
	unique_ptr<ColumnSegment> current_segment;
	BufferHandle handle;

	RLEState<T> state;
	idx_t entry_count = 0;
	idx_t max_rle_count;
};

template <class T>
unique_ptr<CompressionState> RLEInitCompression(ColumnDataCheckpointer &checkpointer, unique_ptr<AnalyzeState> state) {
	return make_unique<RLECompressState<T>>(checkpointer);
}

template <class T>
void RLECompress(CompressionState &state_p, Vector &scan_vector, idx_t count) {
	auto &state = (RLECompressState<T> &)state_p;
	UnifiedVectorFormat vdata;
	scan_vector.ToUnifiedFormat(count, vdata);

	state.Append(vdata, count);
}

template <class T>
void RLEFinalizeCompress(CompressionState &state_p) {
	auto &state = (RLECompressState<T> &)state_p;
	state.Finalize();
}

//===--------------------------------------------------------------------===//
// Scan
//===--------------------------------------------------------------------===//
template <class T>
struct RLEScanState : public SegmentScanState {
	explicit RLEScanState(ColumnSegment &segment) {
		auto &buffer_manager = BufferManager::GetBufferManager(segment.db);
		handle = buffer_manager.Pin(segment.block);
		entry_pos = 0;
		position_in_entry = 0;
		rle_count_offset = Load<uint64_t>(handle.Ptr() + segment.GetBlockOffset());
		D_ASSERT(rle_count_offset <= Storage::BLOCK_SIZE);
	}

	void Skip(ColumnSegment &segment, idx_t skip_count) {
		auto data = handle.Ptr() + segment.GetBlockOffset();
		auto index_pointer = (rle_count_t *)(data + rle_count_offset);

		for (idx_t i = 0; i < skip_count; i++) {
			// assign the current value
			position_in_entry++;
			if (position_in_entry >= index_pointer[entry_pos]) {
				// handled all entries in this RLE value
				// move to the next entry
				entry_pos++;
				position_in_entry = 0;
			}
		}
	}

	BufferHandle handle;
	uint32_t rle_offset;
	idx_t entry_pos;
	idx_t position_in_entry;
	uint32_t rle_count_offset;
};

template <class T>
unique_ptr<SegmentScanState> RLEInitScan(ColumnSegment &segment) {
	auto result = make_unique<RLEScanState<T>>(segment);
	return std::move(result);
}

//===--------------------------------------------------------------------===//
// Scan base data
//===--------------------------------------------------------------------===//
template <class T>
void RLESkip(ColumnSegment &segment, ColumnScanState &state, idx_t skip_count) {
	auto &scan_state = (RLEScanState<T> &)*state.scan_state;
	scan_state.Skip(segment, skip_count);
}

template <class T>
void RLEScanPartial(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count, Vector &result,
                    idx_t result_offset) {
	auto &scan_state = (RLEScanState<T> &)*state.scan_state;

	auto data = scan_state.handle.Ptr() + segment.GetBlockOffset();
	auto data_pointer = (T *)(data + RLEConstants::RLE_HEADER_SIZE);
	auto index_pointer = (rle_count_t *)(data + scan_state.rle_count_offset);

	auto result_data = FlatVector::GetData<T>(result);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	for (idx_t i = 0; i < scan_count; i++) {
		// assign the current value
		result_data[result_offset + i] = data_pointer[scan_state.entry_pos];
		scan_state.position_in_entry++;
		if (scan_state.position_in_entry >= index_pointer[scan_state.entry_pos]) {
			// handled all entries in this RLE value
			// move to the next entry
			scan_state.entry_pos++;
			scan_state.position_in_entry = 0;
		}
	}
}

template <class T>
void RLEScan(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count, Vector &result) {
	// FIXME: emit constant vector if repetition of single value is >= scan_count
	RLEScanPartial<T>(segment, state, scan_count, result, 0);
}

//===--------------------------------------------------------------------===//
// Fetch
//===--------------------------------------------------------------------===//
template <class T>
void RLEFetchRow(ColumnSegment &segment, ColumnFetchState &state, row_t row_id, Vector &result, idx_t result_idx) {
	RLEScanState<T> scan_state(segment);
	scan_state.Skip(segment, row_id);

	auto data = scan_state.handle.Ptr() + segment.GetBlockOffset();
	auto data_pointer = (T *)(data + RLEConstants::RLE_HEADER_SIZE);
	auto result_data = FlatVector::GetData<T>(result);
	result_data[result_idx] = data_pointer[scan_state.entry_pos];
}

//===--------------------------------------------------------------------===//
// Get Function
//===--------------------------------------------------------------------===//
template <class T>
CompressionFunction GetRLEFunction(PhysicalType data_type) {
	return CompressionFunction(CompressionType::COMPRESSION_RLE, data_type, RLEInitAnalyze<T>, RLEAnalyze<T>,
	                           RLEFinalAnalyze<T>, RLEInitCompression<T>, RLECompress<T>, RLEFinalizeCompress<T>,
	                           RLEInitScan<T>, RLEScan<T>, RLEScanPartial<T>, RLEFetchRow<T>, RLESkip<T>);
}

CompressionFunction RLEFun::GetFunction(PhysicalType type) {
	switch (type) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		return GetRLEFunction<int8_t>(type);
	case PhysicalType::INT16:
		return GetRLEFunction<int16_t>(type);
	case PhysicalType::INT32:
		return GetRLEFunction<int32_t>(type);
	case PhysicalType::INT64:
		return GetRLEFunction<int64_t>(type);
	case PhysicalType::INT128:
		return GetRLEFunction<hugeint_t>(type);
	case PhysicalType::UINT8:
		return GetRLEFunction<uint8_t>(type);
	case PhysicalType::UINT16:
		return GetRLEFunction<uint16_t>(type);
	case PhysicalType::UINT32:
		return GetRLEFunction<uint32_t>(type);
	case PhysicalType::UINT64:
		return GetRLEFunction<uint64_t>(type);
	case PhysicalType::FLOAT:
		return GetRLEFunction<float>(type);
	case PhysicalType::DOUBLE:
		return GetRLEFunction<double>(type);
	default:
		throw InternalException("Unsupported type for RLE");
	}
}

bool RLEFun::TypeIsSupported(PhysicalType type) {
	switch (type) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
	case PhysicalType::INT16:
	case PhysicalType::INT32:
	case PhysicalType::INT64:
	case PhysicalType::INT128:
	case PhysicalType::UINT8:
	case PhysicalType::UINT16:
	case PhysicalType::UINT32:
	case PhysicalType::UINT64:
	case PhysicalType::FLOAT:
	case PhysicalType::DOUBLE:
		return true;
	default:
		return false;
	}
}

} // namespace duckdb






namespace duckdb {

//===--------------------------------------------------------------------===//
// Storage Class
//===--------------------------------------------------------------------===//
UncompressedStringSegmentState::~UncompressedStringSegmentState() {
	while (head) {
		// prevent deep recursion here
		head = std::move(head->next);
	}
}

//===--------------------------------------------------------------------===//
// Analyze
//===--------------------------------------------------------------------===//
struct StringAnalyzeState : public AnalyzeState {
	StringAnalyzeState() : count(0), total_string_size(0), overflow_strings(0) {
	}

	idx_t count;
	idx_t total_string_size;
	idx_t overflow_strings;
};

unique_ptr<AnalyzeState> UncompressedStringStorage::StringInitAnalyze(ColumnData &col_data, PhysicalType type) {
	return make_unique<StringAnalyzeState>();
}

bool UncompressedStringStorage::StringAnalyze(AnalyzeState &state_p, Vector &input, idx_t count) {
	auto &state = (StringAnalyzeState &)state_p;
	UnifiedVectorFormat vdata;
	input.ToUnifiedFormat(count, vdata);

	state.count += count;
	auto data = (string_t *)vdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto idx = vdata.sel->get_index(i);
		if (vdata.validity.RowIsValid(idx)) {
			auto string_size = data[idx].GetSize();
			state.total_string_size += string_size;
			if (string_size >= StringUncompressed::STRING_BLOCK_LIMIT) {
				state.overflow_strings++;
			}
		}
	}
	return true;
}

idx_t UncompressedStringStorage::StringFinalAnalyze(AnalyzeState &state_p) {
	auto &state = (StringAnalyzeState &)state_p;
	return state.count * sizeof(int32_t) + state.total_string_size + state.overflow_strings * BIG_STRING_MARKER_SIZE;
}

//===--------------------------------------------------------------------===//
// Scan
//===--------------------------------------------------------------------===//
unique_ptr<SegmentScanState> UncompressedStringStorage::StringInitScan(ColumnSegment &segment) {
	auto result = make_unique<StringScanState>();
	auto &buffer_manager = BufferManager::GetBufferManager(segment.db);
	result->handle = buffer_manager.Pin(segment.block);
	return std::move(result);
}

//===--------------------------------------------------------------------===//
// Scan base data
//===--------------------------------------------------------------------===//
void UncompressedStringStorage::StringScanPartial(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count,
                                                  Vector &result, idx_t result_offset) {
	// clear any previously locked buffers and get the primary buffer handle
	auto &scan_state = (StringScanState &)*state.scan_state;
	auto start = segment.GetRelativeIndex(state.row_index);

	auto baseptr = scan_state.handle.Ptr() + segment.GetBlockOffset();
	auto dict = GetDictionary(segment, scan_state.handle);
	auto base_data = (int32_t *)(baseptr + DICTIONARY_HEADER_SIZE);
	auto result_data = FlatVector::GetData<string_t>(result);

	int32_t previous_offset = start > 0 ? base_data[start - 1] : 0;

	for (idx_t i = 0; i < scan_count; i++) {
		// std::abs used since offsets can be negative to indicate big strings
		uint32_t string_length = std::abs(base_data[start + i]) - std::abs(previous_offset);
		result_data[result_offset + i] =
		    FetchStringFromDict(segment, dict, result, baseptr, base_data[start + i], string_length);
		previous_offset = base_data[start + i];
	}
}

void UncompressedStringStorage::StringScan(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count,
                                           Vector &result) {
	StringScanPartial(segment, state, scan_count, result, 0);
}

//===--------------------------------------------------------------------===//
// Fetch
//===--------------------------------------------------------------------===//
BufferHandle &ColumnFetchState::GetOrInsertHandle(ColumnSegment &segment) {
	auto primary_id = segment.block->BlockId();

	auto entry = handles.find(primary_id);
	if (entry == handles.end()) {
		// not pinned yet: pin it
		auto &buffer_manager = BufferManager::GetBufferManager(segment.db);
		auto handle = buffer_manager.Pin(segment.block);
		auto entry = handles.insert(make_pair(primary_id, std::move(handle)));
		return entry.first->second;
	} else {
		// already pinned: use the pinned handle
		return entry->second;
	}
}

void UncompressedStringStorage::StringFetchRow(ColumnSegment &segment, ColumnFetchState &state, row_t row_id,
                                               Vector &result, idx_t result_idx) {
	// fetch a single row from the string segment
	// first pin the main buffer if it is not already pinned
	auto &handle = state.GetOrInsertHandle(segment);

	auto baseptr = handle.Ptr() + segment.GetBlockOffset();
	auto dict = GetDictionary(segment, handle);
	auto base_data = (int32_t *)(baseptr + DICTIONARY_HEADER_SIZE);
	auto result_data = FlatVector::GetData<string_t>(result);

	auto dict_offset = base_data[row_id];
	uint32_t string_length;
	if ((idx_t)row_id == 0) {
		// edge case where this is the first string in the dict
		string_length = std::abs(dict_offset);
	} else {
		string_length = std::abs(dict_offset) - std::abs(base_data[row_id - 1]);
	}
	result_data[result_idx] = FetchStringFromDict(segment, dict, result, baseptr, dict_offset, string_length);
}

//===--------------------------------------------------------------------===//
// Append
//===--------------------------------------------------------------------===//

unique_ptr<CompressedSegmentState> UncompressedStringStorage::StringInitSegment(ColumnSegment &segment,
                                                                                block_id_t block_id) {
	auto &buffer_manager = BufferManager::GetBufferManager(segment.db);
	if (block_id == INVALID_BLOCK) {
		auto handle = buffer_manager.Pin(segment.block);
		StringDictionaryContainer dictionary;
		dictionary.size = 0;
		dictionary.end = segment.SegmentSize();
		SetDictionary(segment, handle, dictionary);
	}
	return make_unique<UncompressedStringSegmentState>();
}

idx_t UncompressedStringStorage::FinalizeAppend(ColumnSegment &segment, SegmentStatistics &stats) {
	auto &buffer_manager = BufferManager::GetBufferManager(segment.db);
	auto handle = buffer_manager.Pin(segment.block);
	auto dict = GetDictionary(segment, handle);
	D_ASSERT(dict.end == segment.SegmentSize());
	// compute the total size required to store this segment
	auto offset_size = DICTIONARY_HEADER_SIZE + segment.count * sizeof(int32_t);
	auto total_size = offset_size + dict.size;
	if (total_size >= COMPACTION_FLUSH_LIMIT) {
		// the block is full enough, don't bother moving around the dictionary
		return segment.SegmentSize();
	}
	// the block has space left: figure out how much space we can save
	auto move_amount = segment.SegmentSize() - total_size;
	// move the dictionary so it lines up exactly with the offsets
	auto dataptr = handle.Ptr();
	memmove(dataptr + offset_size, dataptr + dict.end - dict.size, dict.size);
	dict.end -= move_amount;
	D_ASSERT(dict.end == total_size);
	// write the new dictionary (with the updated "end")
	SetDictionary(segment, handle, dict);
	return total_size;
}

//===--------------------------------------------------------------------===//
// Get Function
//===--------------------------------------------------------------------===//
CompressionFunction StringUncompressed::GetFunction(PhysicalType data_type) {
	D_ASSERT(data_type == PhysicalType::VARCHAR);
	return CompressionFunction(CompressionType::COMPRESSION_UNCOMPRESSED, data_type,
	                           UncompressedStringStorage::StringInitAnalyze, UncompressedStringStorage::StringAnalyze,
	                           UncompressedStringStorage::StringFinalAnalyze, UncompressedFunctions::InitCompression,
	                           UncompressedFunctions::Compress, UncompressedFunctions::FinalizeCompress,
	                           UncompressedStringStorage::StringInitScan, UncompressedStringStorage::StringScan,
	                           UncompressedStringStorage::StringScanPartial, UncompressedStringStorage::StringFetchRow,
	                           UncompressedFunctions::EmptySkip, UncompressedStringStorage::StringInitSegment,
	                           UncompressedStringStorage::StringInitAppend, UncompressedStringStorage::StringAppend,
	                           UncompressedStringStorage::FinalizeAppend);
}

//===--------------------------------------------------------------------===//
// Helper Functions
//===--------------------------------------------------------------------===//
void UncompressedStringStorage::SetDictionary(ColumnSegment &segment, BufferHandle &handle,
                                              StringDictionaryContainer container) {
	auto startptr = handle.Ptr() + segment.GetBlockOffset();
	Store<uint32_t>(container.size, startptr);
	Store<uint32_t>(container.end, startptr + sizeof(uint32_t));
}

StringDictionaryContainer UncompressedStringStorage::GetDictionary(ColumnSegment &segment, BufferHandle &handle) {
	auto startptr = handle.Ptr() + segment.GetBlockOffset();
	StringDictionaryContainer container;
	container.size = Load<uint32_t>(startptr);
	container.end = Load<uint32_t>(startptr + sizeof(uint32_t));
	return container;
}

idx_t UncompressedStringStorage::RemainingSpace(ColumnSegment &segment, BufferHandle &handle) {
	auto dictionary = GetDictionary(segment, handle);
	D_ASSERT(dictionary.end == segment.SegmentSize());
	idx_t used_space = dictionary.size + segment.count * sizeof(int32_t) + DICTIONARY_HEADER_SIZE;
	D_ASSERT(segment.SegmentSize() >= used_space);
	return segment.SegmentSize() - used_space;
}

void UncompressedStringStorage::WriteString(ColumnSegment &segment, string_t string, block_id_t &result_block,
                                            int32_t &result_offset) {
	auto &state = (UncompressedStringSegmentState &)*segment.GetSegmentState();
	if (state.overflow_writer) {
		// overflow writer is set: write string there
		state.overflow_writer->WriteString(string, result_block, result_offset);
	} else {
		// default overflow behavior: use in-memory buffer to store the overflow string
		WriteStringMemory(segment, string, result_block, result_offset);
	}
}

void UncompressedStringStorage::WriteStringMemory(ColumnSegment &segment, string_t string, block_id_t &result_block,
                                                  int32_t &result_offset) {
	uint32_t total_length = string.GetSize() + sizeof(uint32_t);
	shared_ptr<BlockHandle> block;
	BufferHandle handle;

	auto &buffer_manager = BufferManager::GetBufferManager(segment.db);
	auto &state = (UncompressedStringSegmentState &)*segment.GetSegmentState();
	// check if the string fits in the current block
	if (!state.head || state.head->offset + total_length >= state.head->size) {
		// string does not fit, allocate space for it
		// create a new string block
		idx_t alloc_size = MaxValue<idx_t>(total_length, Storage::BLOCK_SIZE);
		auto new_block = make_unique<StringBlock>();
		new_block->offset = 0;
		new_block->size = alloc_size;
		// allocate an in-memory buffer for it
		handle = buffer_manager.Allocate(alloc_size, false, &block);
		state.overflow_blocks[block->BlockId()] = new_block.get();
		new_block->block = std::move(block);
		new_block->next = std::move(state.head);
		state.head = std::move(new_block);
	} else {
		// string fits, copy it into the current block
		handle = buffer_manager.Pin(state.head->block);
	}

	result_block = state.head->block->BlockId();
	result_offset = state.head->offset;

	// copy the string and the length there
	auto ptr = handle.Ptr() + state.head->offset;
	Store<uint32_t>(string.GetSize(), ptr);
	ptr += sizeof(uint32_t);
	memcpy(ptr, string.GetDataUnsafe(), string.GetSize());
	state.head->offset += total_length;
}

string_t UncompressedStringStorage::ReadOverflowString(ColumnSegment &segment, Vector &result, block_id_t block,
                                                       int32_t offset) {
	D_ASSERT(block != INVALID_BLOCK);
	D_ASSERT(offset < Storage::BLOCK_SIZE);

	auto &block_manager = segment.GetBlockManager();
	auto &buffer_manager = block_manager.buffer_manager;
	auto &state = (UncompressedStringSegmentState &)*segment.GetSegmentState();
	if (block < MAXIMUM_BLOCK) {
		// read the overflow string from disk
		// pin the initial handle and read the length
		auto block_handle = block_manager.RegisterBlock(block);
		auto handle = buffer_manager.Pin(block_handle);

		// read header
		uint32_t compressed_size = Load<uint32_t>(handle.Ptr() + offset);
		uint32_t uncompressed_size = Load<uint32_t>(handle.Ptr() + offset + sizeof(uint32_t));
		uint32_t remaining = compressed_size;
		offset += 2 * sizeof(uint32_t);

		data_ptr_t decompression_ptr;
		std::unique_ptr<data_t[]> decompression_buffer;

		// If string is in single block we decompress straight from it, else we copy first
		if (remaining <= Storage::BLOCK_SIZE - sizeof(block_id_t) - offset) {
			decompression_ptr = handle.Ptr() + offset;
		} else {
			decompression_buffer = std::unique_ptr<data_t[]>(new data_t[compressed_size]);
			auto target_ptr = decompression_buffer.get();

			// now append the string to the single buffer
			while (remaining > 0) {
				idx_t to_write = MinValue<idx_t>(remaining, Storage::BLOCK_SIZE - sizeof(block_id_t) - offset);
				memcpy(target_ptr, handle.Ptr() + offset, to_write);

				remaining -= to_write;
				offset += to_write;
				target_ptr += to_write;
				if (remaining > 0) {
					// read the next block
					block_id_t next_block = Load<block_id_t>(handle.Ptr() + offset);
					block_handle = block_manager.RegisterBlock(next_block);
					handle = buffer_manager.Pin(block_handle);
					offset = 0;
				}
			}
			decompression_ptr = decompression_buffer.get();
		}

		// overflow strings on disk are gzipped, decompress here
		auto decompressed_target_handle =
		    buffer_manager.Allocate(MaxValue<idx_t>(Storage::BLOCK_SIZE, uncompressed_size));
		auto decompressed_target_ptr = decompressed_target_handle.Ptr();
		MiniZStream s;
		s.Decompress((const char *)decompression_ptr, compressed_size, (char *)decompressed_target_ptr,
		             uncompressed_size);

		auto final_buffer = decompressed_target_handle.Ptr();
		StringVector::AddHandle(result, std::move(decompressed_target_handle));
		return ReadString(final_buffer, 0, uncompressed_size);
	} else {
		// read the overflow string from memory
		// first pin the handle, if it is not pinned yet
		auto entry = state.overflow_blocks.find(block);
		D_ASSERT(entry != state.overflow_blocks.end());
		auto handle = buffer_manager.Pin(entry->second->block);
		auto final_buffer = handle.Ptr();
		StringVector::AddHandle(result, std::move(handle));
		return ReadStringWithLength(final_buffer, offset);
	}
}

string_t UncompressedStringStorage::ReadString(data_ptr_t target, int32_t offset, uint32_t string_length) {
	auto ptr = target + offset;
	auto str_ptr = (char *)(ptr);
	return string_t(str_ptr, string_length);
}

string_t UncompressedStringStorage::ReadStringWithLength(data_ptr_t target, int32_t offset) {
	auto ptr = target + offset;
	auto str_length = Load<uint32_t>(ptr);
	auto str_ptr = (char *)(ptr + sizeof(uint32_t));
	return string_t(str_ptr, str_length);
}

void UncompressedStringStorage::WriteStringMarker(data_ptr_t target, block_id_t block_id, int32_t offset) {
	memcpy(target, &block_id, sizeof(block_id_t));
	target += sizeof(block_id_t);
	memcpy(target, &offset, sizeof(int32_t));
}

void UncompressedStringStorage::ReadStringMarker(data_ptr_t target, block_id_t &block_id, int32_t &offset) {
	memcpy(&block_id, target, sizeof(block_id_t));
	target += sizeof(block_id_t);
	memcpy(&offset, target, sizeof(int32_t));
}

string_location_t UncompressedStringStorage::FetchStringLocation(StringDictionaryContainer dict, data_ptr_t baseptr,
                                                                 int32_t dict_offset) {
	D_ASSERT(dict_offset >= -1 * Storage::BLOCK_SIZE && dict_offset <= Storage::BLOCK_SIZE);
	if (dict_offset < 0) {
		string_location_t result;
		ReadStringMarker(baseptr + dict.end - (-1 * dict_offset), result.block_id, result.offset);
		return result;
	} else {
		return string_location_t(INVALID_BLOCK, dict_offset);
	}
}

string_t UncompressedStringStorage::FetchStringFromDict(ColumnSegment &segment, StringDictionaryContainer dict,
                                                        Vector &result, data_ptr_t baseptr, int32_t dict_offset,
                                                        uint32_t string_length) {
	// fetch base data
	D_ASSERT(dict_offset <= Storage::BLOCK_SIZE);
	string_location_t location = FetchStringLocation(dict, baseptr, dict_offset);
	return FetchString(segment, dict, result, baseptr, location, string_length);
}

string_t UncompressedStringStorage::FetchString(ColumnSegment &segment, StringDictionaryContainer dict, Vector &result,
                                                data_ptr_t baseptr, string_location_t location,
                                                uint32_t string_length) {
	if (location.block_id != INVALID_BLOCK) {
		// big string marker: read from separate block
		return ReadOverflowString(segment, result, location.block_id, location.offset);
	} else {
		if (location.offset == 0) {
			return string_t(nullptr, 0);
		}
		// normal string: read string from this block
		auto dict_end = baseptr + dict.end;
		auto dict_pos = dict_end - location.offset;

		auto str_ptr = (char *)(dict_pos);
		return string_t(str_ptr, string_length);
	}
}

} // namespace duckdb



namespace duckdb {

CompressionFunction UncompressedFun::GetFunction(PhysicalType type) {
	switch (type) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
	case PhysicalType::INT16:
	case PhysicalType::INT32:
	case PhysicalType::INT64:
	case PhysicalType::INT128:
	case PhysicalType::UINT8:
	case PhysicalType::UINT16:
	case PhysicalType::UINT32:
	case PhysicalType::UINT64:
	case PhysicalType::FLOAT:
	case PhysicalType::DOUBLE:
	case PhysicalType::LIST:
	case PhysicalType::INTERVAL:
		return FixedSizeUncompressed::GetFunction(type);
	case PhysicalType::BIT:
		return ValidityUncompressed::GetFunction(type);
	case PhysicalType::VARCHAR:
		return StringUncompressed::GetFunction(type);
	default:
		throw InternalException("Unsupported type for Uncompressed");
	}
}

bool UncompressedFun::TypeIsSupported(PhysicalType type) {
	return true;
}

} // namespace duckdb










namespace duckdb {

//===--------------------------------------------------------------------===//
// Mask constants
//===--------------------------------------------------------------------===//
// LOWER_MASKS contains masks with all the lower bits set until a specific value
// LOWER_MASKS[0] has the 0 lowest bits set, i.e.:
// 0b0000000000000000000000000000000000000000000000000000000000000000,
// LOWER_MASKS[10] has the 10 lowest bits set, i.e.:
// 0b0000000000000000000000000000000000000000000000000000000111111111,
// etc...
// 0b0000000000000000000000000000000000000001111111111111111111111111,
// ...
// 0b0000000000000000000001111111111111111111111111111111111111111111,
// until LOWER_MASKS[64], which has all bits set:
// 0b1111111111111111111111111111111111111111111111111111111111111111
// generated with this python snippet:
// for i in range(65):
//   print(hex(int((64 - i) * '0' + i * '1', 2)) + ",")
const validity_t ValidityUncompressed::LOWER_MASKS[] = {0x0,
                                                        0x1,
                                                        0x3,
                                                        0x7,
                                                        0xf,
                                                        0x1f,
                                                        0x3f,
                                                        0x7f,
                                                        0xff,
                                                        0x1ff,
                                                        0x3ff,
                                                        0x7ff,
                                                        0xfff,
                                                        0x1fff,
                                                        0x3fff,
                                                        0x7fff,
                                                        0xffff,
                                                        0x1ffff,
                                                        0x3ffff,
                                                        0x7ffff,
                                                        0xfffff,
                                                        0x1fffff,
                                                        0x3fffff,
                                                        0x7fffff,
                                                        0xffffff,
                                                        0x1ffffff,
                                                        0x3ffffff,
                                                        0x7ffffff,
                                                        0xfffffff,
                                                        0x1fffffff,
                                                        0x3fffffff,
                                                        0x7fffffff,
                                                        0xffffffff,
                                                        0x1ffffffff,
                                                        0x3ffffffff,
                                                        0x7ffffffff,
                                                        0xfffffffff,
                                                        0x1fffffffff,
                                                        0x3fffffffff,
                                                        0x7fffffffff,
                                                        0xffffffffff,
                                                        0x1ffffffffff,
                                                        0x3ffffffffff,
                                                        0x7ffffffffff,
                                                        0xfffffffffff,
                                                        0x1fffffffffff,
                                                        0x3fffffffffff,
                                                        0x7fffffffffff,
                                                        0xffffffffffff,
                                                        0x1ffffffffffff,
                                                        0x3ffffffffffff,
                                                        0x7ffffffffffff,
                                                        0xfffffffffffff,
                                                        0x1fffffffffffff,
                                                        0x3fffffffffffff,
                                                        0x7fffffffffffff,
                                                        0xffffffffffffff,
                                                        0x1ffffffffffffff,
                                                        0x3ffffffffffffff,
                                                        0x7ffffffffffffff,
                                                        0xfffffffffffffff,
                                                        0x1fffffffffffffff,
                                                        0x3fffffffffffffff,
                                                        0x7fffffffffffffff,
                                                        0xffffffffffffffff};

// UPPER_MASKS contains masks with all the highest bits set until a specific value
// UPPER_MASKS[0] has the 0 highest bits set, i.e.:
// 0b0000000000000000000000000000000000000000000000000000000000000000,
// UPPER_MASKS[10] has the 10 highest bits set, i.e.:
// 0b1111111111110000000000000000000000000000000000000000000000000000,
// etc...
// 0b1111111111111111111111110000000000000000000000000000000000000000,
// ...
// 0b1111111111111111111111111111111111111110000000000000000000000000,
// until UPPER_MASKS[64], which has all bits set:
// 0b1111111111111111111111111111111111111111111111111111111111111111
// generated with this python snippet:
// for i in range(65):
//   print(hex(int(i * '1' + (64 - i) * '0', 2)) + ",")
const validity_t ValidityUncompressed::UPPER_MASKS[] = {0x0,
                                                        0x8000000000000000,
                                                        0xc000000000000000,
                                                        0xe000000000000000,
                                                        0xf000000000000000,
                                                        0xf800000000000000,
                                                        0xfc00000000000000,
                                                        0xfe00000000000000,
                                                        0xff00000000000000,
                                                        0xff80000000000000,
                                                        0xffc0000000000000,
                                                        0xffe0000000000000,
                                                        0xfff0000000000000,
                                                        0xfff8000000000000,
                                                        0xfffc000000000000,
                                                        0xfffe000000000000,
                                                        0xffff000000000000,
                                                        0xffff800000000000,
                                                        0xffffc00000000000,
                                                        0xffffe00000000000,
                                                        0xfffff00000000000,
                                                        0xfffff80000000000,
                                                        0xfffffc0000000000,
                                                        0xfffffe0000000000,
                                                        0xffffff0000000000,
                                                        0xffffff8000000000,
                                                        0xffffffc000000000,
                                                        0xffffffe000000000,
                                                        0xfffffff000000000,
                                                        0xfffffff800000000,
                                                        0xfffffffc00000000,
                                                        0xfffffffe00000000,
                                                        0xffffffff00000000,
                                                        0xffffffff80000000,
                                                        0xffffffffc0000000,
                                                        0xffffffffe0000000,
                                                        0xfffffffff0000000,
                                                        0xfffffffff8000000,
                                                        0xfffffffffc000000,
                                                        0xfffffffffe000000,
                                                        0xffffffffff000000,
                                                        0xffffffffff800000,
                                                        0xffffffffffc00000,
                                                        0xffffffffffe00000,
                                                        0xfffffffffff00000,
                                                        0xfffffffffff80000,
                                                        0xfffffffffffc0000,
                                                        0xfffffffffffe0000,
                                                        0xffffffffffff0000,
                                                        0xffffffffffff8000,
                                                        0xffffffffffffc000,
                                                        0xffffffffffffe000,
                                                        0xfffffffffffff000,
                                                        0xfffffffffffff800,
                                                        0xfffffffffffffc00,
                                                        0xfffffffffffffe00,
                                                        0xffffffffffffff00,
                                                        0xffffffffffffff80,
                                                        0xffffffffffffffc0,
                                                        0xffffffffffffffe0,
                                                        0xfffffffffffffff0,
                                                        0xfffffffffffffff8,
                                                        0xfffffffffffffffc,
                                                        0xfffffffffffffffe,
                                                        0xffffffffffffffff};

//===--------------------------------------------------------------------===//
// Analyze
//===--------------------------------------------------------------------===//
struct ValidityAnalyzeState : public AnalyzeState {
	ValidityAnalyzeState() : count(0) {
	}

	idx_t count;
};

unique_ptr<AnalyzeState> ValidityInitAnalyze(ColumnData &col_data, PhysicalType type) {
	return make_unique<ValidityAnalyzeState>();
}

bool ValidityAnalyze(AnalyzeState &state_p, Vector &input, idx_t count) {
	auto &state = (ValidityAnalyzeState &)state_p;
	state.count += count;
	return true;
}

idx_t ValidityFinalAnalyze(AnalyzeState &state_p) {
	auto &state = (ValidityAnalyzeState &)state_p;
	return (state.count + 7) / 8;
}

//===--------------------------------------------------------------------===//
// Scan
//===--------------------------------------------------------------------===//
struct ValidityScanState : public SegmentScanState {
	BufferHandle handle;
	block_id_t block_id;
};

unique_ptr<SegmentScanState> ValidityInitScan(ColumnSegment &segment) {
	auto result = make_unique<ValidityScanState>();
	auto &buffer_manager = BufferManager::GetBufferManager(segment.db);
	result->handle = buffer_manager.Pin(segment.block);
	result->block_id = segment.block->BlockId();
	return std::move(result);
}

//===--------------------------------------------------------------------===//
// Scan base data
//===--------------------------------------------------------------------===//
void ValidityScanPartial(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count, Vector &result,
                         idx_t result_offset) {
	auto start = segment.GetRelativeIndex(state.row_index);

	static_assert(sizeof(validity_t) == sizeof(uint64_t), "validity_t should be 64-bit");
	auto &scan_state = (ValidityScanState &)*state.scan_state;

	auto &result_mask = FlatVector::Validity(result);
	auto buffer_ptr = scan_state.handle.Ptr() + segment.GetBlockOffset();
	D_ASSERT(scan_state.block_id == segment.block->BlockId());
	auto input_data = (validity_t *)buffer_ptr;

#ifdef DEBUG
	// this method relies on all the bits we are going to write to being set to valid
	for (idx_t i = 0; i < scan_count; i++) {
		D_ASSERT(result_mask.RowIsValid(result_offset + i));
	}
#endif
#if STANDARD_VECTOR_SIZE < 128
	// fallback for tiny vector sizes
	// the bitwise ops we use below don't work if the vector size is too small
	ValidityMask source_mask(input_data);
	for (idx_t i = 0; i < scan_count; i++) {
		if (!source_mask.RowIsValid(start + i)) {
			if (result_mask.AllValid()) {
				result_mask.Initialize(MaxValue<idx_t>(STANDARD_VECTOR_SIZE, result_offset + scan_count));
			}
			result_mask.SetInvalid(result_offset + i);
		}
	}
#else
	// the code below does what the fallback code above states, but using bitwise ops:
	auto result_data = (validity_t *)result_mask.GetData();

	// set up the initial positions
	// we need to find the validity_entry to modify, together with the bit-index WITHIN the validity entry
	idx_t result_entry = result_offset / ValidityMask::BITS_PER_VALUE;
	idx_t result_idx = result_offset - result_entry * ValidityMask::BITS_PER_VALUE;

	// same for the input: find the validity_entry we are pulling from, together with the bit-index WITHIN that entry
	idx_t input_entry = start / ValidityMask::BITS_PER_VALUE;
	idx_t input_idx = start - input_entry * ValidityMask::BITS_PER_VALUE;

	// now start the bit games
	idx_t pos = 0;
	while (pos < scan_count) {
		// these are the current validity entries we are dealing with
		idx_t current_result_idx = result_entry;
		idx_t offset;
		validity_t input_mask = input_data[input_entry];

		// construct the mask to AND together with the result
		if (result_idx < input_idx) {
			// we have to shift the input RIGHT if the result_idx is smaller than the input_idx
			auto shift_amount = input_idx - result_idx;
			D_ASSERT(shift_amount > 0 && shift_amount <= ValidityMask::BITS_PER_VALUE);

			input_mask = input_mask >> shift_amount;

			// now the upper "shift_amount" bits are set to 0
			// we need them to be set to 1
			// otherwise the subsequent bitwise & will modify values outside of the range of values we want to alter
			input_mask |= ValidityUncompressed::UPPER_MASKS[shift_amount];

			// after this, we move to the next input_entry
			offset = ValidityMask::BITS_PER_VALUE - input_idx;
			input_entry++;
			input_idx = 0;
			result_idx += offset;
		} else if (result_idx > input_idx) {
			// we have to shift the input LEFT if the result_idx is bigger than the input_idx
			auto shift_amount = result_idx - input_idx;
			D_ASSERT(shift_amount > 0 && shift_amount <= ValidityMask::BITS_PER_VALUE);

			// to avoid overflows, we set the upper "shift_amount" values to 0 first
			input_mask = (input_mask & ~ValidityUncompressed::UPPER_MASKS[shift_amount]) << shift_amount;

			// now the lower "shift_amount" bits are set to 0
			// we need them to be set to 1
			// otherwise the subsequent bitwise & will modify values outside of the range of values we want to alter
			input_mask |= ValidityUncompressed::LOWER_MASKS[shift_amount];

			// after this, we move to the next result_entry
			offset = ValidityMask::BITS_PER_VALUE - result_idx;
			result_entry++;
			result_idx = 0;
			input_idx += offset;
		} else {
			// if the input_idx is equal to result_idx they are already aligned
			// we just move to the next entry for both after this
			offset = ValidityMask::BITS_PER_VALUE - result_idx;
			input_entry++;
			result_entry++;
			result_idx = input_idx = 0;
		}
		// now we need to check if we should include the ENTIRE mask
		// OR if we need to mask from the right side
		pos += offset;
		if (pos > scan_count) {
			// we need to set any bits that are past the scan_count on the right-side to 1
			// this is required so we don't influence any bits that are not part of the scan
			input_mask |= ValidityUncompressed::UPPER_MASKS[pos - scan_count];
		}
		// now finally we can merge the input mask with the result mask
		if (input_mask != ValidityMask::ValidityBuffer::MAX_ENTRY) {
			if (!result_data) {
				result_mask.Initialize(MaxValue<idx_t>(STANDARD_VECTOR_SIZE, result_offset + scan_count));
				result_data = (validity_t *)result_mask.GetData();
			}
			result_data[current_result_idx] &= input_mask;
		}
	}
#endif

#ifdef DEBUG
	// verify that we actually accomplished the bitwise ops equivalent that we wanted to do
	ValidityMask input_mask(input_data);
	for (idx_t i = 0; i < scan_count; i++) {
		D_ASSERT(result_mask.RowIsValid(result_offset + i) == input_mask.RowIsValid(start + i));
	}
#endif
}

void ValidityScan(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count, Vector &result) {
	result.Flatten(scan_count);

	auto start = segment.GetRelativeIndex(state.row_index);
	if (start % ValidityMask::BITS_PER_VALUE == 0) {
		auto &scan_state = (ValidityScanState &)*state.scan_state;

		// aligned scan: no need to do anything fancy
		// note: this is only an optimization which avoids having to do messy bitshifting in the common case
		// it is not required for correctness
		auto &result_mask = FlatVector::Validity(result);
		auto buffer_ptr = scan_state.handle.Ptr() + segment.GetBlockOffset();
		D_ASSERT(scan_state.block_id == segment.block->BlockId());
		auto input_data = (validity_t *)buffer_ptr;
		auto result_data = (validity_t *)result_mask.GetData();
		idx_t start_offset = start / ValidityMask::BITS_PER_VALUE;
		idx_t entry_scan_count = (scan_count + ValidityMask::BITS_PER_VALUE - 1) / ValidityMask::BITS_PER_VALUE;
		for (idx_t i = 0; i < entry_scan_count; i++) {
			auto input_entry = input_data[start_offset + i];
			if (!result_data && input_entry == ValidityMask::ValidityBuffer::MAX_ENTRY) {
				continue;
			}
			if (!result_data) {
				result_mask.Initialize(MaxValue<idx_t>(STANDARD_VECTOR_SIZE, scan_count));
				result_data = (validity_t *)result_mask.GetData();
			}
			result_data[i] = input_entry;
		}
	} else {
		// unaligned scan: fall back to scan_partial which does bitshift tricks
		ValidityScanPartial(segment, state, scan_count, result, 0);
	}
}

//===--------------------------------------------------------------------===//
// Fetch
//===--------------------------------------------------------------------===//
void ValidityFetchRow(ColumnSegment &segment, ColumnFetchState &state, row_t row_id, Vector &result, idx_t result_idx) {
	D_ASSERT(row_id >= 0 && row_id < row_t(segment.count));
	auto &buffer_manager = BufferManager::GetBufferManager(segment.db);
	auto handle = buffer_manager.Pin(segment.block);
	auto dataptr = handle.Ptr() + segment.GetBlockOffset();
	ValidityMask mask((validity_t *)dataptr);
	auto &result_mask = FlatVector::Validity(result);
	if (!mask.RowIsValidUnsafe(row_id)) {
		result_mask.SetInvalid(result_idx);
	}
}

//===--------------------------------------------------------------------===//
// Append
//===--------------------------------------------------------------------===//
static unique_ptr<CompressionAppendState> ValidityInitAppend(ColumnSegment &segment) {
	auto &buffer_manager = BufferManager::GetBufferManager(segment.db);
	auto handle = buffer_manager.Pin(segment.block);
	return make_unique<CompressionAppendState>(std::move(handle));
}

unique_ptr<CompressedSegmentState> ValidityInitSegment(ColumnSegment &segment, block_id_t block_id) {
	auto &buffer_manager = BufferManager::GetBufferManager(segment.db);
	if (block_id == INVALID_BLOCK) {
		auto handle = buffer_manager.Pin(segment.block);
		memset(handle.Ptr(), 0xFF, segment.SegmentSize());
	}
	return nullptr;
}

idx_t ValidityAppend(CompressionAppendState &append_state, ColumnSegment &segment, SegmentStatistics &stats,
                     UnifiedVectorFormat &data, idx_t offset, idx_t vcount) {
	D_ASSERT(segment.GetBlockOffset() == 0);
	auto &validity_stats = (ValidityStatistics &)*stats.statistics;

	auto max_tuples = segment.SegmentSize() / ValidityMask::STANDARD_MASK_SIZE * STANDARD_VECTOR_SIZE;
	idx_t append_count = MinValue<idx_t>(vcount, max_tuples - segment.count);
	if (data.validity.AllValid()) {
		// no null values: skip append
		segment.count += append_count;
		validity_stats.has_no_null = true;
		return append_count;
	}

	ValidityMask mask((validity_t *)append_state.handle.Ptr());
	for (idx_t i = 0; i < append_count; i++) {
		auto idx = data.sel->get_index(offset + i);
		if (!data.validity.RowIsValidUnsafe(idx)) {
			mask.SetInvalidUnsafe(segment.count + i);
			validity_stats.has_null = true;
		} else {
			validity_stats.has_no_null = true;
		}
	}
	segment.count += append_count;
	return append_count;
}

idx_t ValidityFinalizeAppend(ColumnSegment &segment, SegmentStatistics &stats) {
	return ((segment.count + STANDARD_VECTOR_SIZE - 1) / STANDARD_VECTOR_SIZE) * ValidityMask::STANDARD_MASK_SIZE;
}

void ValidityRevertAppend(ColumnSegment &segment, idx_t start_row) {
	idx_t start_bit = start_row - segment.start;

	auto &buffer_manager = BufferManager::GetBufferManager(segment.db);
	auto handle = buffer_manager.Pin(segment.block);
	idx_t revert_start;
	if (start_bit % 8 != 0) {
		// handle sub-bit stuff (yay)
		idx_t byte_pos = start_bit / 8;
		idx_t bit_start = byte_pos * 8;
		idx_t bit_end = (byte_pos + 1) * 8;
		ValidityMask mask((validity_t *)handle.Ptr() + byte_pos);
		for (idx_t i = start_bit; i < bit_end; i++) {
			mask.SetValid(i - bit_start);
		}
		revert_start = bit_end / 8;
	} else {
		revert_start = start_bit / 8;
	}
	// for the rest, we just memset
	memset(handle.Ptr() + revert_start, 0xFF, segment.SegmentSize() - revert_start);
}

//===--------------------------------------------------------------------===//
// Get Function
//===--------------------------------------------------------------------===//
CompressionFunction ValidityUncompressed::GetFunction(PhysicalType data_type) {
	D_ASSERT(data_type == PhysicalType::BIT);
	return CompressionFunction(CompressionType::COMPRESSION_UNCOMPRESSED, data_type, ValidityInitAnalyze,
	                           ValidityAnalyze, ValidityFinalAnalyze, UncompressedFunctions::InitCompression,
	                           UncompressedFunctions::Compress, UncompressedFunctions::FinalizeCompress,
	                           ValidityInitScan, ValidityScan, ValidityScanPartial, ValidityFetchRow,
	                           UncompressedFunctions::EmptySkip, ValidityInitSegment, ValidityInitAppend,
	                           ValidityAppend, ValidityFinalizeAppend, ValidityRevertAppend);
}

} // namespace duckdb


























namespace duckdb {

DataTableInfo::DataTableInfo(AttachedDatabase &db, shared_ptr<TableIOManager> table_io_manager_p, string schema,
                             string table)
    : db(db), table_io_manager(std::move(table_io_manager_p)), cardinality(0), schema(std::move(schema)),
      table(std::move(table)) {
}

bool DataTableInfo::IsTemporary() const {
	return db.IsTemporary();
}

DataTable::DataTable(AttachedDatabase &db, shared_ptr<TableIOManager> table_io_manager_p, const string &schema,
                     const string &table, vector<ColumnDefinition> column_definitions_p,
                     unique_ptr<PersistentTableData> data)
    : info(make_shared<DataTableInfo>(db, std::move(table_io_manager_p), schema, table)),
      column_definitions(std::move(column_definitions_p)), db(db), is_root(true) {
	// initialize the table with the existing data from disk, if any
	auto types = GetTypes();
	this->row_groups =
	    make_shared<RowGroupCollection>(info, TableIOManager::Get(*this).GetBlockManagerForRowData(), types, 0);
	if (data && !data->row_groups.empty()) {
		this->row_groups->Initialize(*data);
	} else {
		this->row_groups->InitializeEmpty();
		D_ASSERT(row_groups->GetTotalRows() == 0);
	}
	row_groups->Verify();
}

DataTable::DataTable(ClientContext &context, DataTable &parent, ColumnDefinition &new_column, Expression *default_value)
    : info(parent.info), db(parent.db), is_root(true) {
	// add the column definitions from this DataTable
	for (auto &column_def : parent.column_definitions) {
		column_definitions.emplace_back(column_def.Copy());
	}
	column_definitions.emplace_back(new_column.Copy());
	// prevent any new tuples from being added to the parent
	lock_guard<mutex> parent_lock(parent.append_lock);

	this->row_groups = parent.row_groups->AddColumn(context, new_column, default_value);

	// also add this column to client local storage
	auto &local_storage = LocalStorage::Get(context, db);
	local_storage.AddColumn(&parent, this, new_column, default_value);

	// this table replaces the previous table, hence the parent is no longer the root DataTable
	parent.is_root = false;
}

DataTable::DataTable(ClientContext &context, DataTable &parent, idx_t removed_column)
    : info(parent.info), db(parent.db), is_root(true) {
	// prevent any new tuples from being added to the parent
	lock_guard<mutex> parent_lock(parent.append_lock);

	for (auto &column_def : parent.column_definitions) {
		column_definitions.emplace_back(column_def.Copy());
	}
	// first check if there are any indexes that exist that point to the removed column
	info->indexes.Scan([&](Index &index) {
		for (auto &column_id : index.column_ids) {
			if (column_id == removed_column) {
				throw CatalogException("Cannot drop this column: an index depends on it!");
			} else if (column_id > removed_column) {
				throw CatalogException("Cannot drop this column: an index depends on a column after it!");
			}
		}
		return false;
	});

	// erase the column definitions from this DataTable
	D_ASSERT(removed_column < column_definitions.size());
	column_definitions.erase(column_definitions.begin() + removed_column);

	storage_t storage_idx = 0;
	for (idx_t i = 0; i < column_definitions.size(); i++) {
		auto &col = column_definitions[i];
		col.SetOid(i);
		if (col.Generated()) {
			continue;
		}
		col.SetStorageOid(storage_idx++);
	}

	// alter the row_groups and remove the column from each of them
	this->row_groups = parent.row_groups->RemoveColumn(removed_column);

	// scan the original table, and fill the new column with the transformed value
	auto &local_storage = LocalStorage::Get(context, db);
	local_storage.DropColumn(&parent, this, removed_column);

	// this table replaces the previous table, hence the parent is no longer the root DataTable
	parent.is_root = false;
}

// Alter column to add new constraint
DataTable::DataTable(ClientContext &context, DataTable &parent, unique_ptr<BoundConstraint> constraint)
    : info(parent.info), db(parent.db), row_groups(parent.row_groups), is_root(true) {

	lock_guard<mutex> parent_lock(parent.append_lock);
	for (auto &column_def : parent.column_definitions) {
		column_definitions.emplace_back(column_def.Copy());
	}

	// Verify the new constraint against current persistent/local data
	VerifyNewConstraint(context, parent, constraint.get());

	// Get the local data ownership from old dt
	auto &local_storage = LocalStorage::Get(context, db);
	local_storage.MoveStorage(&parent, this);
	// this table replaces the previous table, hence the parent is no longer the root DataTable
	parent.is_root = false;
}

DataTable::DataTable(ClientContext &context, DataTable &parent, idx_t changed_idx, const LogicalType &target_type,
                     const vector<column_t> &bound_columns, Expression &cast_expr)
    : info(parent.info), db(parent.db), is_root(true) {
	// prevent any tuples from being added to the parent
	lock_guard<mutex> lock(append_lock);
	for (auto &column_def : parent.column_definitions) {
		column_definitions.emplace_back(column_def.Copy());
	}
	// first check if there are any indexes that exist that point to the changed column
	info->indexes.Scan([&](Index &index) {
		for (auto &column_id : index.column_ids) {
			if (column_id == changed_idx) {
				throw CatalogException("Cannot change the type of this column: an index depends on it!");
			}
		}
		return false;
	});

	// change the type in this DataTable
	column_definitions[changed_idx].SetType(target_type);

	// set up the statistics for the table
	// the column that had its type changed will have the new statistics computed during conversion
	this->row_groups = parent.row_groups->AlterType(context, changed_idx, target_type, bound_columns, cast_expr);

	// scan the original table, and fill the new column with the transformed value
	auto &local_storage = LocalStorage::Get(context, db);
	local_storage.ChangeType(&parent, this, changed_idx, target_type, bound_columns, cast_expr);

	// this table replaces the previous table, hence the parent is no longer the root DataTable
	parent.is_root = false;
}

vector<LogicalType> DataTable::GetTypes() {
	vector<LogicalType> types;
	for (auto &it : column_definitions) {
		types.push_back(it.Type());
	}
	return types;
}

TableIOManager &TableIOManager::Get(DataTable &table) {
	return *table.info->table_io_manager;
}

//===--------------------------------------------------------------------===//
// Scan
//===--------------------------------------------------------------------===//
void DataTable::InitializeScan(TableScanState &state, const vector<column_t> &column_ids,
                               TableFilterSet *table_filters) {
	state.Initialize(column_ids, table_filters);
	row_groups->InitializeScan(state.table_state, column_ids, table_filters);
}

void DataTable::InitializeScan(DuckTransaction &transaction, TableScanState &state, const vector<column_t> &column_ids,
                               TableFilterSet *table_filters) {
	InitializeScan(state, column_ids, table_filters);
	auto &local_storage = LocalStorage::Get(transaction);
	local_storage.InitializeScan(this, state.local_state, table_filters);
}

void DataTable::InitializeScanWithOffset(TableScanState &state, const vector<column_t> &column_ids, idx_t start_row,
                                         idx_t end_row) {
	state.Initialize(column_ids);
	row_groups->InitializeScanWithOffset(state.table_state, column_ids, start_row, end_row);
}

idx_t DataTable::MaxThreads(ClientContext &context) {
	idx_t parallel_scan_vector_count = RowGroup::ROW_GROUP_VECTOR_COUNT;
	if (ClientConfig::GetConfig(context).verify_parallelism) {
		parallel_scan_vector_count = 1;
	}
	idx_t parallel_scan_tuple_count = STANDARD_VECTOR_SIZE * parallel_scan_vector_count;
	return GetTotalRows() / parallel_scan_tuple_count + 1;
}

void DataTable::InitializeParallelScan(ClientContext &context, ParallelTableScanState &state) {
	row_groups->InitializeParallelScan(state.scan_state);

	auto &local_storage = LocalStorage::Get(context, db);
	local_storage.InitializeParallelScan(this, state.local_state);
}

bool DataTable::NextParallelScan(ClientContext &context, ParallelTableScanState &state, TableScanState &scan_state) {
	if (row_groups->NextParallelScan(context, state.scan_state, scan_state.table_state)) {
		return true;
	}
	scan_state.table_state.batch_index = state.scan_state.batch_index;
	auto &local_storage = LocalStorage::Get(context, db);
	if (local_storage.NextParallelScan(context, this, state.local_state, scan_state.local_state)) {
		return true;
	} else {
		// finished all scans: no more scans remaining
		return false;
	}
}

void DataTable::Scan(DuckTransaction &transaction, DataChunk &result, TableScanState &state) {
	// scan the persistent segments
	if (state.table_state.Scan(transaction, result)) {
		D_ASSERT(result.size() > 0);
		return;
	}

	// scan the transaction-local segments
	auto &local_storage = LocalStorage::Get(transaction);
	local_storage.Scan(state.local_state, state.GetColumnIds(), result);
}

bool DataTable::CreateIndexScan(TableScanState &state, DataChunk &result, TableScanType type) {
	return state.table_state.ScanCommitted(result, type);
}

//===--------------------------------------------------------------------===//
// Fetch
//===--------------------------------------------------------------------===//
void DataTable::Fetch(DuckTransaction &transaction, DataChunk &result, const vector<column_t> &column_ids,
                      const Vector &row_identifiers, idx_t fetch_count, ColumnFetchState &state) {
	row_groups->Fetch(transaction, result, column_ids, row_identifiers, fetch_count, state);
}

//===--------------------------------------------------------------------===//
// Append
//===--------------------------------------------------------------------===//
static void VerifyNotNullConstraint(TableCatalogEntry &table, Vector &vector, idx_t count, const string &col_name) {
	if (!VectorOperations::HasNull(vector, count)) {
		return;
	}

	throw ConstraintException("NOT NULL constraint failed: %s.%s", table.name, col_name);
}

// To avoid throwing an error at SELECT, instead this moves the error detection to INSERT
static void VerifyGeneratedExpressionSuccess(ClientContext &context, TableCatalogEntry &table, DataChunk &chunk,
                                             Expression &expr, column_t index) {
	auto &col = table.GetColumn(LogicalIndex(index));
	D_ASSERT(col.Generated());
	ExpressionExecutor executor(context, expr);
	Vector result(col.Type());
	try {
		executor.ExecuteExpression(chunk, result);
	} catch (InternalException &ex) {
		throw;
	} catch (std::exception &ex) {
		throw ConstraintException("Incorrect value for generated column \"%s %s AS (%s)\" : %s", col.Name(),
		                          col.Type().ToString(), col.GeneratedExpression().ToString(), ex.what());
	}
}

static void VerifyCheckConstraint(ClientContext &context, TableCatalogEntry &table, Expression &expr,
                                  DataChunk &chunk) {
	ExpressionExecutor executor(context, expr);
	Vector result(LogicalType::INTEGER);
	try {
		executor.ExecuteExpression(chunk, result);
	} catch (std::exception &ex) {
		throw ConstraintException("CHECK constraint failed: %s (Error: %s)", table.name, ex.what());
	} catch (...) { // LCOV_EXCL_START
		throw ConstraintException("CHECK constraint failed: %s (Unknown Error)", table.name);
	} // LCOV_EXCL_STOP
	UnifiedVectorFormat vdata;
	result.ToUnifiedFormat(chunk.size(), vdata);

	auto dataptr = (int32_t *)vdata.data;
	for (idx_t i = 0; i < chunk.size(); i++) {
		auto idx = vdata.sel->get_index(i);
		if (vdata.validity.RowIsValid(idx) && dataptr[idx] == 0) {
			throw ConstraintException("CHECK constraint failed: %s", table.name);
		}
	}
}

bool DataTable::IsForeignKeyIndex(const vector<PhysicalIndex> &fk_keys, Index &index, ForeignKeyType fk_type) {
	if (fk_type == ForeignKeyType::FK_TYPE_PRIMARY_KEY_TABLE ? !index.IsUnique() : !index.IsForeign()) {
		return false;
	}
	if (fk_keys.size() != index.column_ids.size()) {
		return false;
	}
	for (auto &fk_key : fk_keys) {
		bool is_found = false;
		for (auto &index_key : index.column_ids) {
			if (fk_key.index == index_key) {
				is_found = true;
				break;
			}
		}
		if (!is_found) {
			return false;
		}
	}
	return true;
}

// Find the first index that is not null, and did not find a match
static idx_t FirstMissingMatch(const ManagedSelection &matches) {
	idx_t match_idx = 0;

	for (idx_t i = 0; i < matches.Size(); i++) {
		auto match = matches.IndexMapsToLocation(match_idx, i);
		match_idx += match;
		if (!match) {
			// This index is missing in the matches vector
			return i;
		}
	}
	return DConstants::INVALID_INDEX;
}

idx_t LocateErrorIndex(bool is_append, const ManagedSelection &matches) {
	idx_t failed_index = DConstants::INVALID_INDEX;
	if (!is_append) {
		// We expected to find nothing, so the first error is the first match
		failed_index = matches[0];
	} else {
		// We expected to find matches for all of them, so the first missing match is the first error
		return FirstMissingMatch(matches);
	}
	return failed_index;
}

[[noreturn]] static void ThrowForeignKeyConstraintError(idx_t failed_index, bool is_append, Index *index,
                                                        DataChunk &input) {
	auto verify_type = is_append ? VerifyExistenceType::APPEND_FK : VerifyExistenceType::DELETE_FK;

	D_ASSERT(failed_index != DConstants::INVALID_INDEX);
	D_ASSERT(index->type == IndexType::ART);
	auto &art_index = (ART &)*index;
	auto key_name = art_index.GenerateErrorKeyName(input, failed_index);
	auto exception_msg = art_index.GenerateConstraintErrorMessage(verify_type, key_name);
	throw ConstraintException(exception_msg);
}

bool IsForeignKeyConstraintError(bool is_append, idx_t input_count, const ManagedSelection &matches) {
	if (is_append) {
		// We need to find a match for all of the values
		return matches.Count() != input_count;
	} else {
		// We should not find any matches
		return matches.Count() != 0;
	}
}

static bool IsAppend(VerifyExistenceType verify_type) {
	return verify_type == VerifyExistenceType::APPEND_FK;
}

void DataTable::VerifyForeignKeyConstraint(const BoundForeignKeyConstraint &bfk, ClientContext &context,
                                           DataChunk &chunk, VerifyExistenceType verify_type) {
	const vector<PhysicalIndex> *src_keys_ptr = &bfk.info.fk_keys;
	const vector<PhysicalIndex> *dst_keys_ptr = &bfk.info.pk_keys;

	bool is_append = IsAppend(verify_type);
	if (!is_append) {
		src_keys_ptr = &bfk.info.pk_keys;
		dst_keys_ptr = &bfk.info.fk_keys;
	}

	auto table_entry_ptr =
	    Catalog::GetEntry<TableCatalogEntry>(context, INVALID_CATALOG, bfk.info.schema, bfk.info.table);
	if (table_entry_ptr == nullptr) {
		throw InternalException("Can't find table \"%s\" in foreign key constraint", bfk.info.table);
	}
	// make the data chunk to check
	vector<LogicalType> types;
	for (auto &col : table_entry_ptr->GetColumns().Physical()) {
		types.emplace_back(col.Type());
	}
	DataChunk dst_chunk;
	dst_chunk.InitializeEmpty(types);
	for (idx_t i = 0; i < src_keys_ptr->size(); i++) {
		dst_chunk.data[(*dst_keys_ptr)[i].index].Reference(chunk.data[(*src_keys_ptr)[i].index]);
	}
	dst_chunk.SetCardinality(chunk.size());
	auto data_table = table_entry_ptr->GetStoragePtr();

	idx_t count = dst_chunk.size();
	if (count <= 0) {
		return;
	}

	// Set up a way to record conflicts, rather than directly throw on them
	unordered_set<column_t> empty_column_list;
	ConflictInfo empty_conflict_info(empty_column_list, false);
	ConflictManager regular_conflicts(verify_type, count, &empty_conflict_info);
	ConflictManager transaction_conflicts(verify_type, count, &empty_conflict_info);
	regular_conflicts.SetMode(ConflictManagerMode::SCAN);
	transaction_conflicts.SetMode(ConflictManagerMode::SCAN);

	data_table->info->indexes.VerifyForeignKey(*dst_keys_ptr, dst_chunk, regular_conflicts);
	regular_conflicts.Finalize();
	auto &regular_matches = regular_conflicts.Conflicts();
	// check whether or not the chunk can be inserted or deleted into the referenced table' transaction local storage
	auto &local_storage = LocalStorage::Get(context, db);

	bool error = IsForeignKeyConstraintError(is_append, count, regular_matches);
	bool transaction_error = false;

	bool transaction_check = local_storage.Find(data_table);
	if (transaction_check) {
		auto &transact_index = local_storage.GetIndexes(data_table);
		transact_index.VerifyForeignKey(*dst_keys_ptr, dst_chunk, transaction_conflicts);
		transaction_conflicts.Finalize();
		auto &transaction_matches = transaction_conflicts.Conflicts();
		transaction_error = IsForeignKeyConstraintError(is_append, count, transaction_matches);
	}

	if (!transaction_error && !error) {
		// No error occurred;
		return;
	}

	// Some error occurred, and we likely want to throw
	Index *index;
	Index *transaction_index;

	auto fk_type = is_append ? ForeignKeyType::FK_TYPE_PRIMARY_KEY_TABLE : ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE;
	// check whether or not the chunk can be inserted or deleted into the referenced table' storage
	index = data_table->info->indexes.FindForeignKeyIndex(*dst_keys_ptr, fk_type);
	if (transaction_check) {
		auto &transact_index = local_storage.GetIndexes(data_table);
		// check whether or not the chunk can be inserted or deleted into the referenced table' storage
		transaction_index = transact_index.FindForeignKeyIndex(*dst_keys_ptr, fk_type);
	}

	if (!transaction_check) {
		// Only local state is checked, throw the error
		D_ASSERT(error);
		auto failed_index = LocateErrorIndex(is_append, regular_matches);
		D_ASSERT(failed_index != DConstants::INVALID_INDEX);
		ThrowForeignKeyConstraintError(failed_index, is_append, index, dst_chunk);
	}
	if (transaction_error && error && is_append) {
		// When we want to do an append, we only throw if the foreign key does not exist in both transaction and local
		// storage
		auto &transaction_matches = transaction_conflicts.Conflicts();
		idx_t failed_index = DConstants::INVALID_INDEX;
		idx_t regular_idx = 0;
		idx_t transaction_idx = 0;
		for (idx_t i = 0; i < count; i++) {
			bool in_regular = regular_matches.IndexMapsToLocation(regular_idx, i);
			regular_idx += in_regular;
			bool in_transaction = transaction_matches.IndexMapsToLocation(transaction_idx, i);
			transaction_idx += in_transaction;

			if (!in_regular && !in_transaction) {
				// We need to find a match for all of the input values
				// The failed index is i, it does not show up in either regular or transaction storage
				failed_index = i;
				break;
			}
		}
		if (failed_index == DConstants::INVALID_INDEX) {
			// We don't throw, every value was present in either regular or transaction storage
			return;
		}
		ThrowForeignKeyConstraintError(failed_index, true, index, dst_chunk);
	}
	if (!is_append && transaction_check) {
		auto &transaction_matches = transaction_conflicts.Conflicts();
		if (error) {
			auto failed_index = LocateErrorIndex(false, regular_matches);
			D_ASSERT(failed_index != DConstants::INVALID_INDEX);
			ThrowForeignKeyConstraintError(failed_index, false, index, dst_chunk);
		} else {
			D_ASSERT(transaction_error);
			D_ASSERT(transaction_matches.Count() != DConstants::INVALID_INDEX);
			auto failed_index = LocateErrorIndex(false, transaction_matches);
			D_ASSERT(failed_index != DConstants::INVALID_INDEX);
			ThrowForeignKeyConstraintError(failed_index, false, transaction_index, dst_chunk);
		}
	}
}

void DataTable::VerifyAppendForeignKeyConstraint(const BoundForeignKeyConstraint &bfk, ClientContext &context,
                                                 DataChunk &chunk) {
	VerifyForeignKeyConstraint(bfk, context, chunk, VerifyExistenceType::APPEND_FK);
}

void DataTable::VerifyDeleteForeignKeyConstraint(const BoundForeignKeyConstraint &bfk, ClientContext &context,
                                                 DataChunk &chunk) {
	VerifyForeignKeyConstraint(bfk, context, chunk, VerifyExistenceType::DELETE_FK);
}

void DataTable::VerifyNewConstraint(ClientContext &context, DataTable &parent, const BoundConstraint *constraint) {
	if (constraint->type != ConstraintType::NOT_NULL) {
		throw NotImplementedException("FIXME: ALTER COLUMN with such constraint is not supported yet");
	}

	parent.row_groups->VerifyNewConstraint(parent, *constraint);
	auto &local_storage = LocalStorage::Get(context, db);
	local_storage.VerifyNewConstraint(parent, *constraint);
}

void DataTable::VerifyAppendConstraints(TableCatalogEntry &table, ClientContext &context, DataChunk &chunk,
                                        ConflictManager *conflict_manager) {
	if (table.HasGeneratedColumns()) {
		// Verify that the generated columns expression work with the inserted values
		auto binder = Binder::CreateBinder(context);
		physical_index_set_t bound_columns;
		CheckBinder generated_check_binder(*binder, context, table.name, table.GetColumns(), bound_columns);
		for (auto &col : table.GetColumns().Logical()) {
			if (!col.Generated()) {
				continue;
			}
			D_ASSERT(col.Type().id() != LogicalTypeId::ANY);
			generated_check_binder.target_type = col.Type();
			auto to_be_bound_expression = col.GeneratedExpression().Copy();
			auto bound_expression = generated_check_binder.Bind(to_be_bound_expression);
			VerifyGeneratedExpressionSuccess(context, table, chunk, *bound_expression, col.Oid());
		}
	}
	auto &constraints = table.GetConstraints();
	auto &bound_constraints = table.GetBoundConstraints();
	for (idx_t i = 0; i < bound_constraints.size(); i++) {
		auto &base_constraint = constraints[i];
		auto &constraint = bound_constraints[i];
		switch (base_constraint->type) {
		case ConstraintType::NOT_NULL: {
			auto &bound_not_null = *reinterpret_cast<BoundNotNullConstraint *>(constraint.get());
			auto &not_null = *reinterpret_cast<NotNullConstraint *>(base_constraint.get());
			auto &col = table.GetColumns().GetColumn(LogicalIndex(not_null.index));
			VerifyNotNullConstraint(table, chunk.data[bound_not_null.index.index], chunk.size(), col.Name());
			break;
		}
		case ConstraintType::CHECK: {
			auto &check = *reinterpret_cast<BoundCheckConstraint *>(constraint.get());
			VerifyCheckConstraint(context, table, *check.expression, chunk);
			break;
		}
		case ConstraintType::UNIQUE: {
			//! check whether or not the chunk can be inserted into the indexes
			if (conflict_manager) {
				// This is only provided when a ON CONFLICT clause was provided
				idx_t matching_indexes = 0;
				auto &conflict_info = conflict_manager->GetConflictInfo();
				// First we figure out how many indexes match our conflict target
				// So we can optimize accordingly
				info->indexes.Scan([&](Index &index) {
					matching_indexes += conflict_info.ConflictTargetMatches(index);
					return false;
				});
				conflict_manager->SetMode(ConflictManagerMode::SCAN);
				conflict_manager->SetIndexCount(matching_indexes);
				// First we verify only the indexes that match our conflict target
				info->indexes.Scan([&](Index &index) {
					if (!index.IsUnique()) {
						return false;
					}
					if (conflict_info.ConflictTargetMatches(index)) {
						index.VerifyAppend(chunk, *conflict_manager);
					}
					return false;
				});

				conflict_manager->SetMode(ConflictManagerMode::THROW);
				// Then we scan the other indexes, throwing if they cause conflicts on tuples that were not found during
				// the scan
				info->indexes.Scan([&](Index &index) {
					if (!index.IsUnique()) {
						return false;
					}
					index.VerifyAppend(chunk, *conflict_manager);
					return false;
				});
			} else {
				// Only need to verify that no unique constraints are violated
				info->indexes.Scan([&](Index &index) {
					if (!index.IsUnique()) {
						return false;
					}
					index.VerifyAppend(chunk);
					return false;
				});
			}
			break;
		}
		case ConstraintType::FOREIGN_KEY: {
			auto &bfk = *reinterpret_cast<BoundForeignKeyConstraint *>(constraint.get());
			if (bfk.info.type == ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE ||
			    bfk.info.type == ForeignKeyType::FK_TYPE_SELF_REFERENCE_TABLE) {
				VerifyAppendForeignKeyConstraint(bfk, context, chunk);
			}
			break;
		}
		default:
			throw NotImplementedException("Constraint type not implemented!");
		}
	}
}

void DataTable::InitializeLocalAppend(LocalAppendState &state, ClientContext &context) {
	if (!is_root) {
		throw TransactionException("Transaction conflict: adding entries to a table that has been altered!");
	}
	auto &local_storage = LocalStorage::Get(context, db);
	local_storage.InitializeAppend(state, this);
}

void DataTable::LocalAppend(LocalAppendState &state, TableCatalogEntry &table, ClientContext &context, DataChunk &chunk,
                            bool unsafe) {
	if (chunk.size() == 0) {
		return;
	}
	D_ASSERT(chunk.ColumnCount() == table.GetColumns().PhysicalColumnCount());
	if (!is_root) {
		throw TransactionException("Transaction conflict: adding entries to a table that has been altered!");
	}

	chunk.Verify();

	// verify any constraints on the new chunk
	if (!unsafe) {
		VerifyAppendConstraints(table, context, chunk);
	}

	// append to the transaction local data
	LocalStorage::Append(state, chunk);
}

void DataTable::FinalizeLocalAppend(LocalAppendState &state) {
	LocalStorage::FinalizeAppend(state);
}

OptimisticDataWriter *DataTable::CreateOptimisticWriter(ClientContext &context) {
	auto &local_storage = LocalStorage::Get(context, db);
	return local_storage.CreateOptimisticWriter(this);
}

void DataTable::LocalMerge(ClientContext &context, RowGroupCollection &collection) {
	auto &local_storage = LocalStorage::Get(context, db);
	local_storage.LocalMerge(this, collection);
}

void DataTable::LocalAppend(TableCatalogEntry &table, ClientContext &context, DataChunk &chunk) {
	LocalAppendState append_state;
	auto &storage = table.GetStorage();
	storage.InitializeLocalAppend(append_state, context);
	storage.LocalAppend(append_state, table, context, chunk);
	storage.FinalizeLocalAppend(append_state);
}

void DataTable::LocalAppend(TableCatalogEntry &table, ClientContext &context, ColumnDataCollection &collection) {
	LocalAppendState append_state;
	auto &storage = table.GetStorage();
	storage.InitializeLocalAppend(append_state, context);
	for (auto &chunk : collection.Chunks()) {
		storage.LocalAppend(append_state, table, context, chunk);
	}
	storage.FinalizeLocalAppend(append_state);
}

void DataTable::AppendLock(TableAppendState &state) {
	state.append_lock = unique_lock<mutex>(append_lock);
	if (!is_root) {
		throw TransactionException("Transaction conflict: adding entries to a table that has been altered!");
	}
	state.row_start = row_groups->GetTotalRows();
	state.current_row = state.row_start;
}

void DataTable::InitializeAppend(DuckTransaction &transaction, TableAppendState &state, idx_t append_count) {
	// obtain the append lock for this table
	if (!state.append_lock) {
		throw InternalException("DataTable::AppendLock should be called before DataTable::InitializeAppend");
	}
	row_groups->InitializeAppend(transaction, state, append_count);
}

void DataTable::Append(DataChunk &chunk, TableAppendState &state) {
	D_ASSERT(is_root);
	row_groups->Append(chunk, state);
}

void DataTable::ScanTableSegment(idx_t row_start, idx_t count, const std::function<void(DataChunk &chunk)> &function) {
	idx_t end = row_start + count;

	vector<column_t> column_ids;
	vector<LogicalType> types;
	for (idx_t i = 0; i < this->column_definitions.size(); i++) {
		auto &col = this->column_definitions[i];
		column_ids.push_back(i);
		types.push_back(col.Type());
	}
	DataChunk chunk;
	chunk.Initialize(Allocator::Get(db), types);

	CreateIndexScanState state;

	InitializeScanWithOffset(state, column_ids, row_start, row_start + count);
	auto row_start_aligned = state.table_state.row_group_state.row_group->start +
	                         state.table_state.row_group_state.vector_index * STANDARD_VECTOR_SIZE;

	idx_t current_row = row_start_aligned;
	while (current_row < end) {
		state.table_state.ScanCommitted(chunk, TableScanType::TABLE_SCAN_COMMITTED_ROWS);
		if (chunk.size() == 0) {
			break;
		}
		idx_t end_row = current_row + chunk.size();
		// start of chunk is current_row
		// end of chunk is end_row
		// figure out if we need to write the entire chunk or just part of it
		idx_t chunk_start = MaxValue<idx_t>(current_row, row_start);
		idx_t chunk_end = MinValue<idx_t>(end_row, end);
		D_ASSERT(chunk_start < chunk_end);
		idx_t chunk_count = chunk_end - chunk_start;
		if (chunk_count != chunk.size()) {
			D_ASSERT(chunk_count <= chunk.size());
			// need to slice the chunk before insert
			idx_t start_in_chunk;
			if (current_row >= row_start) {
				start_in_chunk = 0;
			} else {
				start_in_chunk = row_start - current_row;
			}
			SelectionVector sel(start_in_chunk, chunk_count);
			chunk.Slice(sel, chunk_count);
			chunk.Verify();
		}
		function(chunk);
		chunk.Reset();
		current_row = end_row;
	}
}

void DataTable::MergeStorage(RowGroupCollection &data, TableIndexList &indexes) {
	row_groups->MergeStorage(data);
	row_groups->Verify();
}

void DataTable::WriteToLog(WriteAheadLog &log, idx_t row_start, idx_t count) {
	if (log.skip_writing) {
		return;
	}
	log.WriteSetTable(info->schema, info->table);
	ScanTableSegment(row_start, count, [&](DataChunk &chunk) { log.WriteInsert(chunk); });
}

void DataTable::CommitAppend(transaction_t commit_id, idx_t row_start, idx_t count) {
	lock_guard<mutex> lock(append_lock);
	row_groups->CommitAppend(commit_id, row_start, count);
	info->cardinality += count;
}

void DataTable::RevertAppendInternal(idx_t start_row, idx_t count) {
	if (count == 0) {
		// nothing to revert!
		return;
	}
	// adjust the cardinality
	info->cardinality = start_row;
	D_ASSERT(is_root);
	// revert appends made to row_groups
	row_groups->RevertAppendInternal(start_row, count);
}

void DataTable::RevertAppend(idx_t start_row, idx_t count) {
	lock_guard<mutex> lock(append_lock);

	if (!info->indexes.Empty()) {
		idx_t current_row_base = start_row;
		row_t row_data[STANDARD_VECTOR_SIZE];
		Vector row_identifiers(LogicalType::ROW_TYPE, (data_ptr_t)row_data);
		ScanTableSegment(start_row, count, [&](DataChunk &chunk) {
			for (idx_t i = 0; i < chunk.size(); i++) {
				row_data[i] = current_row_base + i;
			}
			info->indexes.Scan([&](Index &index) {
				index.Delete(chunk, row_identifiers);
				return false;
			});
			current_row_base += chunk.size();
		});
	}
	RevertAppendInternal(start_row, count);
}

//===--------------------------------------------------------------------===//
// Indexes
//===--------------------------------------------------------------------===//
bool DataTable::AppendToIndexes(TableIndexList &indexes, DataChunk &chunk, row_t row_start) {
	if (indexes.Empty()) {
		return true;
	}
	// first generate the vector of row identifiers
	Vector row_identifiers(LogicalType::ROW_TYPE);
	VectorOperations::GenerateSequence(row_identifiers, chunk.size(), row_start, 1);

	vector<Index *> already_appended;
	bool append_failed = false;
	// now append the entries to the indices
	indexes.Scan([&](Index &index) {
		try {
			if (!index.Append(chunk, row_identifiers)) {
				append_failed = true;
				return true;
			}
		} catch (...) {
			append_failed = true;
			return true;
		}
		already_appended.push_back(&index);
		return false;
	});

	if (append_failed) {
		// constraint violation!
		// remove any appended entries from previous indexes (if any)
		for (auto *index : already_appended) {
			index->Delete(chunk, row_identifiers);
		}
		return false;
	}
	return true;
}

bool DataTable::AppendToIndexes(DataChunk &chunk, row_t row_start) {
	D_ASSERT(is_root);
	return AppendToIndexes(info->indexes, chunk, row_start);
}

void DataTable::RemoveFromIndexes(TableAppendState &state, DataChunk &chunk, row_t row_start) {
	D_ASSERT(is_root);
	if (info->indexes.Empty()) {
		return;
	}
	// first generate the vector of row identifiers
	Vector row_identifiers(LogicalType::ROW_TYPE);
	VectorOperations::GenerateSequence(row_identifiers, chunk.size(), row_start, 1);

	// now remove the entries from the indices
	RemoveFromIndexes(state, chunk, row_identifiers);
}

void DataTable::RemoveFromIndexes(TableAppendState &state, DataChunk &chunk, Vector &row_identifiers) {
	D_ASSERT(is_root);
	info->indexes.Scan([&](Index &index) {
		index.Delete(chunk, row_identifiers);
		return false;
	});
}

void DataTable::RemoveFromIndexes(Vector &row_identifiers, idx_t count) {
	D_ASSERT(is_root);
	row_groups->RemoveFromIndexes(info->indexes, row_identifiers, count);
}

//===--------------------------------------------------------------------===//
// Delete
//===--------------------------------------------------------------------===//
static bool TableHasDeleteConstraints(TableCatalogEntry &table) {
	auto &bound_constraints = table.GetBoundConstraints();
	for (auto &constraint : bound_constraints) {
		switch (constraint->type) {
		case ConstraintType::NOT_NULL:
		case ConstraintType::CHECK:
		case ConstraintType::UNIQUE:
			break;
		case ConstraintType::FOREIGN_KEY: {
			auto &bfk = *reinterpret_cast<BoundForeignKeyConstraint *>(constraint.get());
			if (bfk.info.type == ForeignKeyType::FK_TYPE_PRIMARY_KEY_TABLE ||
			    bfk.info.type == ForeignKeyType::FK_TYPE_SELF_REFERENCE_TABLE) {
				return true;
			}
			break;
		}
		default:
			throw NotImplementedException("Constraint type not implemented!");
		}
	}
	return false;
}

void DataTable::VerifyDeleteConstraints(TableCatalogEntry &table, ClientContext &context, DataChunk &chunk) {
	auto &bound_constraints = table.GetBoundConstraints();
	for (auto &constraint : bound_constraints) {
		switch (constraint->type) {
		case ConstraintType::NOT_NULL:
		case ConstraintType::CHECK:
		case ConstraintType::UNIQUE:
			break;
		case ConstraintType::FOREIGN_KEY: {
			auto &bfk = *reinterpret_cast<BoundForeignKeyConstraint *>(constraint.get());
			if (bfk.info.type == ForeignKeyType::FK_TYPE_PRIMARY_KEY_TABLE ||
			    bfk.info.type == ForeignKeyType::FK_TYPE_SELF_REFERENCE_TABLE) {
				VerifyDeleteForeignKeyConstraint(bfk, context, chunk);
			}
			break;
		}
		default:
			throw NotImplementedException("Constraint type not implemented!");
		}
	}
}

idx_t DataTable::Delete(TableCatalogEntry &table, ClientContext &context, Vector &row_identifiers, idx_t count) {
	D_ASSERT(row_identifiers.GetType().InternalType() == ROW_TYPE);
	if (count == 0) {
		return 0;
	}

	auto &transaction = DuckTransaction::Get(context, db);
	auto &local_storage = LocalStorage::Get(transaction);
	bool has_delete_constraints = TableHasDeleteConstraints(table);

	row_identifiers.Flatten(count);
	auto ids = FlatVector::GetData<row_t>(row_identifiers);

	DataChunk verify_chunk;
	vector<column_t> col_ids;
	vector<LogicalType> types;
	ColumnFetchState fetch_state;
	if (has_delete_constraints) {
		// initialize the chunk if there are any constraints to verify
		for (idx_t i = 0; i < column_definitions.size(); i++) {
			col_ids.push_back(column_definitions[i].StorageOid());
			types.emplace_back(column_definitions[i].Type());
		}
		verify_chunk.Initialize(Allocator::Get(context), types);
	}
	idx_t pos = 0;
	idx_t delete_count = 0;
	while (pos < count) {
		idx_t start = pos;
		bool is_transaction_delete = ids[pos] >= MAX_ROW_ID;
		// figure out which batch of rows to delete now
		for (pos++; pos < count; pos++) {
			bool row_is_transaction_delete = ids[pos] >= MAX_ROW_ID;
			if (row_is_transaction_delete != is_transaction_delete) {
				break;
			}
		}
		idx_t current_offset = start;
		idx_t current_count = pos - start;

		Vector offset_ids(row_identifiers, current_offset, pos);
		if (is_transaction_delete) {
			// transaction-local delete
			if (has_delete_constraints) {
				// perform the constraint verification
				local_storage.FetchChunk(this, offset_ids, current_count, col_ids, verify_chunk, fetch_state);
				VerifyDeleteConstraints(table, context, verify_chunk);
			}
			delete_count += local_storage.Delete(this, offset_ids, current_count);
		} else {
			// regular table delete
			if (has_delete_constraints) {
				// perform the constraint verification
				Fetch(transaction, verify_chunk, col_ids, offset_ids, current_count, fetch_state);
				VerifyDeleteConstraints(table, context, verify_chunk);
			}
			delete_count += row_groups->Delete(transaction, this, ids + current_offset, current_count);
		}
	}
	return delete_count;
}

//===--------------------------------------------------------------------===//
// Update
//===--------------------------------------------------------------------===//
static void CreateMockChunk(vector<LogicalType> &types, const vector<PhysicalIndex> &column_ids, DataChunk &chunk,
                            DataChunk &mock_chunk) {
	// construct a mock DataChunk
	mock_chunk.InitializeEmpty(types);
	for (column_t i = 0; i < column_ids.size(); i++) {
		mock_chunk.data[column_ids[i].index].Reference(chunk.data[i]);
	}
	mock_chunk.SetCardinality(chunk.size());
}

static bool CreateMockChunk(TableCatalogEntry &table, const vector<PhysicalIndex> &column_ids,
                            physical_index_set_t &desired_column_ids, DataChunk &chunk, DataChunk &mock_chunk) {
	idx_t found_columns = 0;
	// check whether the desired columns are present in the UPDATE clause
	for (column_t i = 0; i < column_ids.size(); i++) {
		if (desired_column_ids.find(column_ids[i]) != desired_column_ids.end()) {
			found_columns++;
		}
	}
	if (found_columns == 0) {
		// no columns were found: no need to check the constraint again
		return false;
	}
	if (found_columns != desired_column_ids.size()) {
		// not all columns in UPDATE clause are present!
		// this should not be triggered at all as the binder should add these columns
		throw InternalException("Not all columns required for the CHECK constraint are present in the UPDATED chunk!");
	}
	// construct a mock DataChunk
	auto types = table.GetTypes();
	CreateMockChunk(types, column_ids, chunk, mock_chunk);
	return true;
}

void DataTable::VerifyUpdateConstraints(ClientContext &context, TableCatalogEntry &table, DataChunk &chunk,
                                        const vector<PhysicalIndex> &column_ids) {
	auto &constraints = table.GetConstraints();
	auto &bound_constraints = table.GetBoundConstraints();
	for (idx_t i = 0; i < bound_constraints.size(); i++) {
		auto &base_constraint = constraints[i];
		auto &constraint = bound_constraints[i];
		switch (constraint->type) {
		case ConstraintType::NOT_NULL: {
			auto &bound_not_null = *reinterpret_cast<BoundNotNullConstraint *>(constraint.get());
			auto &not_null = *reinterpret_cast<NotNullConstraint *>(base_constraint.get());
			// check if the constraint is in the list of column_ids
			// FIXME: double usage of 'i'?
			for (idx_t i = 0; i < column_ids.size(); i++) {
				if (column_ids[i] == bound_not_null.index) {
					// found the column id: check the data in
					auto &col = table.GetColumn(LogicalIndex(not_null.index));
					VerifyNotNullConstraint(table, chunk.data[i], chunk.size(), col.Name());
					break;
				}
			}
			break;
		}
		case ConstraintType::CHECK: {
			auto &check = *reinterpret_cast<BoundCheckConstraint *>(constraint.get());

			DataChunk mock_chunk;
			if (CreateMockChunk(table, column_ids, check.bound_columns, chunk, mock_chunk)) {
				VerifyCheckConstraint(context, table, *check.expression, mock_chunk);
			}
			break;
		}
		case ConstraintType::UNIQUE:
		case ConstraintType::FOREIGN_KEY:
			break;
		default:
			throw NotImplementedException("Constraint type not implemented!");
		}
	}
	// update should not be called for indexed columns!
	// instead update should have been rewritten to delete + update on higher layer
#ifdef DEBUG
	info->indexes.Scan([&](Index &index) {
		D_ASSERT(!index.IndexIsUpdated(column_ids));
		return false;
	});

#endif
}

void DataTable::Update(TableCatalogEntry &table, ClientContext &context, Vector &row_ids,
                       const vector<PhysicalIndex> &column_ids, DataChunk &updates) {
	D_ASSERT(row_ids.GetType().InternalType() == ROW_TYPE);

	D_ASSERT(column_ids.size() == updates.ColumnCount());
	auto count = updates.size();
	updates.Verify();
	if (count == 0) {
		return;
	}

	if (!is_root) {
		throw TransactionException("Transaction conflict: cannot update a table that has been altered!");
	}

	// first verify that no constraints are violated
	VerifyUpdateConstraints(context, table, updates, column_ids);

	// now perform the actual update
	auto &transaction = DuckTransaction::Get(context, db);

	updates.Flatten();
	row_ids.Flatten(count);
	auto ids = FlatVector::GetData<row_t>(row_ids);
	auto first_id = FlatVector::GetValue<row_t>(row_ids, 0);
	if (first_id >= MAX_ROW_ID) {
		// update is in transaction-local storage: push update into local storage
		auto &local_storage = LocalStorage::Get(context, db);
		local_storage.Update(this, row_ids, column_ids, updates);
		return;
	}

	// update is in the row groups
	// we need to figure out for each id to which row group it belongs
	// usually all (or many) ids belong to the same row group
	// we iterate over the ids and check for every id if it belongs to the same row group as their predecessor
	row_groups->Update(transaction, ids, column_ids, updates);
}

void DataTable::UpdateColumn(TableCatalogEntry &table, ClientContext &context, Vector &row_ids,
                             const vector<column_t> &column_path, DataChunk &updates) {
	D_ASSERT(row_ids.GetType().InternalType() == ROW_TYPE);
	D_ASSERT(updates.ColumnCount() == 1);
	updates.Verify();
	if (updates.size() == 0) {
		return;
	}

	if (!is_root) {
		throw TransactionException("Transaction conflict: cannot update a table that has been altered!");
	}

	// now perform the actual update
	auto &transaction = DuckTransaction::Get(context, db);

	updates.Flatten();
	row_ids.Flatten(updates.size());
	row_groups->UpdateColumn(transaction, row_ids, column_path, updates);
}

//===--------------------------------------------------------------------===//
// Create Index Scan
//===--------------------------------------------------------------------===//
void DataTable::InitializeCreateIndexScan(CreateIndexScanState &state, const vector<column_t> &column_ids) {
	// we grab the append lock to make sure nothing is appended until AFTER we finish the index scan
	state.append_lock = std::unique_lock<mutex>(append_lock);
	row_groups->InitializeCreateIndexScan(state);
	InitializeScan(state, column_ids);
}

unique_ptr<BaseStatistics> DataTable::GetStatistics(ClientContext &context, column_t column_id) {
	if (column_id == COLUMN_IDENTIFIER_ROW_ID) {
		return nullptr;
	}
	return row_groups->CopyStats(column_id);
}

void DataTable::SetStatistics(column_t column_id, const std::function<void(BaseStatistics &)> &set_fun) {
	D_ASSERT(column_id != COLUMN_IDENTIFIER_ROW_ID);
	row_groups->SetStatistics(column_id, set_fun);
}

//===--------------------------------------------------------------------===//
// Checkpoint
//===--------------------------------------------------------------------===//
void DataTable::Checkpoint(TableDataWriter &writer) {
	// checkpoint each individual row group
	// FIXME: we might want to combine adjacent row groups in case they have had deletions...
	vector<unique_ptr<BaseStatistics>> global_stats;
	for (idx_t i = 0; i < column_definitions.size(); i++) {
		global_stats.push_back(row_groups->CopyStats(i));
	}

	row_groups->Checkpoint(writer, global_stats);

	// The rowgroup payload data has been written. Now write:
	//   column stats
	//   row-group pointers
	//   table pointer
	//   index data
	writer.FinalizeTable(std::move(global_stats), info.get());
}

void DataTable::CommitDropColumn(idx_t index) {
	row_groups->CommitDropColumn(index);
}

idx_t DataTable::GetTotalRows() {
	return row_groups->GetTotalRows();
}

void DataTable::CommitDropTable() {
	// commit a drop of this table: mark all blocks as modified so they can be reclaimed later on
	row_groups->CommitDropTable();
}

//===--------------------------------------------------------------------===//
// GetStorageInfo
//===--------------------------------------------------------------------===//
void DataTable::GetStorageInfo(TableStorageInfo &result) {
	row_groups->GetStorageInfo(result);
}

} // namespace duckdb








namespace duckdb {

Index::Index(AttachedDatabase &db, IndexType type, TableIOManager &table_io_manager,
             const vector<column_t> &column_ids_p, const vector<unique_ptr<Expression>> &unbound_expressions,
             IndexConstraintType constraint_type_p, bool track_memory)

    : type(type), table_io_manager(table_io_manager), column_ids(column_ids_p), constraint_type(constraint_type_p),
      db(db), buffer_manager(BufferManager::GetBufferManager(db)), memory_size(0), track_memory(track_memory) {

	for (auto &expr : unbound_expressions) {
		types.push_back(expr->return_type.InternalType());
		logical_types.push_back(expr->return_type);
		auto unbound_expression = expr->Copy();
		bound_expressions.push_back(BindExpression(unbound_expression->Copy()));
		this->unbound_expressions.emplace_back(std::move(unbound_expression));
	}
	for (auto &bound_expr : bound_expressions) {
		executor.AddExpression(*bound_expr);
	}

	// create the column id set
	for (auto column_id : column_ids) {
		column_id_set.insert(column_id);
	}
}

void Index::InitializeLock(IndexLock &state) {
	state.index_lock = unique_lock<mutex>(lock);
}

bool Index::Append(DataChunk &entries, Vector &row_identifiers) {
	IndexLock state;
	InitializeLock(state);
	return Append(state, entries, row_identifiers);
}

void Index::Delete(DataChunk &entries, Vector &row_identifiers) {
	IndexLock state;
	InitializeLock(state);
	Delete(state, entries, row_identifiers);
}

bool Index::MergeIndexes(Index *other_index) {

	IndexLock state;
	InitializeLock(state);

	switch (this->type) {
	case IndexType::ART: {
		auto art = (ART *)this;
		return art->MergeIndexes(state, other_index);
	}
	default:
		throw InternalException("Unimplemented index type for merge");
	}
}

void Index::ExecuteExpressions(DataChunk &input, DataChunk &result) {
	executor.Execute(input, result);
}

unique_ptr<Expression> Index::BindExpression(unique_ptr<Expression> expr) {
	if (expr->type == ExpressionType::BOUND_COLUMN_REF) {
		auto &bound_colref = (BoundColumnRefExpression &)*expr;
		return make_unique<BoundReferenceExpression>(expr->return_type, column_ids[bound_colref.binding.column_index]);
	}
	ExpressionIterator::EnumerateChildren(
	    *expr, [this](unique_ptr<Expression> &expr) { expr = BindExpression(std::move(expr)); });
	return expr;
}

bool Index::IndexIsUpdated(const vector<PhysicalIndex> &column_ids) const {
	for (auto &column : column_ids) {
		if (column_id_set.find(column.index) != column_id_set.end()) {
			return true;
		}
	}
	return false;
}

BlockPointer Index::Serialize(duckdb::MetaBlockWriter &writer) {
	throw NotImplementedException("The implementation of this index serialization does not exist.");
}

} // namespace duckdb














namespace duckdb {

//===--------------------------------------------------------------------===//
// OptimisticDataWriter
//===--------------------------------------------------------------------===//
OptimisticDataWriter::OptimisticDataWriter(DataTable *table) : table(table) {
}

OptimisticDataWriter::OptimisticDataWriter(DataTable *table, OptimisticDataWriter &parent)
    : table(table), partial_manager(std::move(parent.partial_manager)),
      written_blocks(std::move(parent.written_blocks)) {
	if (partial_manager) {
		partial_manager->FlushPartialBlocks();
	}
}

OptimisticDataWriter::~OptimisticDataWriter() {
}

bool OptimisticDataWriter::PrepareWrite() {
	// check if we should pre-emptively write the table to disk
	if (table->info->IsTemporary() || StorageManager::Get(table->info->db).InMemory()) {
		return false;
	}
	// we should! write the second-to-last row group to disk
	// allocate the partial block-manager if none is allocated yet
	if (!partial_manager) {
		auto &block_manager = table->info->table_io_manager->GetBlockManagerForRowData();
		partial_manager = make_unique<PartialBlockManager>(block_manager);
	}
	return true;
}

void OptimisticDataWriter::CheckFlushToDisk(RowGroupCollection &row_groups) {
	// we finished writing a complete row group
	if (!PrepareWrite()) {
		return;
	}
	// flush second-to-last row group
	auto row_group = row_groups.GetRowGroup(-2);
	FlushToDisk(row_group);
}

void OptimisticDataWriter::FlushToDisk(RowGroup *row_group) {
	// flush the specified row group
	D_ASSERT(row_group);
	//! The set of column compression types (if any)
	vector<CompressionType> compression_types;
	D_ASSERT(compression_types.empty());
	for (auto &column : table->column_definitions) {
		compression_types.push_back(column.CompressionType());
	}
	auto row_group_pointer = row_group->WriteToDisk(*partial_manager, compression_types);

	// update the set of written blocks
	for (idx_t col_idx = 0; col_idx < row_group_pointer.statistics.size(); col_idx++) {
		row_group_pointer.states[col_idx]->GetBlockIds(written_blocks);
	}
}

void OptimisticDataWriter::FlushToDisk(RowGroupCollection &row_groups, bool force) {
	if (!partial_manager) {
		if (!force) {
			// no partial manager - nothing to flush
			return;
		}
		if (!PrepareWrite()) {
			return;
		}
	}
	// flush the last row group
	FlushToDisk(row_groups.GetRowGroup(-1));
}

void OptimisticDataWriter::FinalFlush() {
	if (!partial_manager) {
		return;
	}
	// then flush the partial manager
	partial_manager->FlushPartialBlocks();
	partial_manager.reset();
}

void OptimisticDataWriter::Rollback() {
	if (partial_manager) {
		partial_manager->Clear();
		partial_manager.reset();
	}
	if (!written_blocks.empty()) {
		auto &block_manager = table->info->table_io_manager->GetBlockManagerForRowData();
		for (auto block_id : written_blocks) {
			block_manager.MarkBlockAsFree(block_id);
		}
	}
}

//===--------------------------------------------------------------------===//
// Local Table Storage
//===--------------------------------------------------------------------===//
LocalTableStorage::LocalTableStorage(DataTable &table)
    : table(&table), allocator(Allocator::Get(table.db)), deleted_rows(0), optimistic_writer(&table) {
	auto types = table.GetTypes();
	row_groups = make_shared<RowGroupCollection>(table.info, TableIOManager::Get(table).GetBlockManagerForRowData(),
	                                             types, MAX_ROW_ID, 0);
	row_groups->InitializeEmpty();
	table.info->indexes.Scan([&](Index &index) {
		D_ASSERT(index.type == IndexType::ART);
		auto &art = (ART &)index;
		if (art.constraint_type != IndexConstraintType::NONE) {
			// unique index: create a local ART index that maintains the same unique constraint
			vector<unique_ptr<Expression>> unbound_expressions;
			for (auto &expr : art.unbound_expressions) {
				unbound_expressions.push_back(expr->Copy());
			}
			indexes.AddIndex(make_unique<ART>(art.column_ids, art.table_io_manager, std::move(unbound_expressions),
			                                  art.constraint_type, art.db, false));
		}
		return false;
	});
}

LocalTableStorage::LocalTableStorage(ClientContext &context, DataTable &new_dt, LocalTableStorage &parent,
                                     idx_t changed_idx, const LogicalType &target_type,
                                     const vector<column_t> &bound_columns, Expression &cast_expr)
    : table(&new_dt), allocator(Allocator::Get(table->db)), deleted_rows(parent.deleted_rows),
      optimistic_writer(table, parent.optimistic_writer), optimistic_writers(std::move(parent.optimistic_writers)) {
	row_groups = parent.row_groups->AlterType(context, changed_idx, target_type, bound_columns, cast_expr);
	parent.row_groups.reset();
	indexes.Move(parent.indexes);
}

LocalTableStorage::LocalTableStorage(DataTable &new_dt, LocalTableStorage &parent, idx_t drop_idx)
    : table(&new_dt), allocator(Allocator::Get(table->db)), deleted_rows(parent.deleted_rows),
      optimistic_writer(table, parent.optimistic_writer), optimistic_writers(std::move(parent.optimistic_writers)) {
	row_groups = parent.row_groups->RemoveColumn(drop_idx);
	parent.row_groups.reset();
	indexes.Move(parent.indexes);
}

LocalTableStorage::LocalTableStorage(ClientContext &context, DataTable &new_dt, LocalTableStorage &parent,
                                     ColumnDefinition &new_column, Expression *default_value)
    : table(&new_dt), allocator(Allocator::Get(table->db)), deleted_rows(parent.deleted_rows),
      optimistic_writer(table, parent.optimistic_writer), optimistic_writers(std::move(parent.optimistic_writers)) {
	row_groups = parent.row_groups->AddColumn(context, new_column, default_value);
	parent.row_groups.reset();
	indexes.Move(parent.indexes);
}

LocalTableStorage::~LocalTableStorage() {
}

void LocalTableStorage::InitializeScan(CollectionScanState &state, TableFilterSet *table_filters) {
	if (row_groups->GetTotalRows() == 0) {
		// nothing to scan
		return;
	}
	row_groups->InitializeScan(state, state.GetColumnIds(), table_filters);
}

idx_t LocalTableStorage::EstimatedSize() {
	idx_t appended_rows = row_groups->GetTotalRows() - deleted_rows;
	if (appended_rows == 0) {
		return 0;
	}
	idx_t row_size = 0;
	auto &types = row_groups->GetTypes();
	for (auto &type : types) {
		row_size += GetTypeIdSize(type.InternalType());
	}
	return appended_rows * row_size;
}

void LocalTableStorage::CheckFlushToDisk() {
	if (deleted_rows != 0) {
		// we have deletes - we cannot merge row groups
		return;
	}
	optimistic_writer.CheckFlushToDisk(*row_groups);
}

void LocalTableStorage::FlushToDisk() {
	optimistic_writer.FlushToDisk(*row_groups);
	optimistic_writer.FinalFlush();
}

bool LocalTableStorage::AppendToIndexes(DuckTransaction &transaction, RowGroupCollection &source,
                                        TableIndexList &index_list, const vector<LogicalType> &table_types,
                                        row_t &start_row) {
	// only need to scan for index append
	// figure out which columns we need to scan for the set of indexes
	auto columns = index_list.GetRequiredColumns();
	// create an empty mock chunk that contains all the correct types for the table
	DataChunk mock_chunk;
	mock_chunk.InitializeEmpty(table_types);
	bool success = true;
	source.Scan(transaction, columns, [&](DataChunk &chunk) -> bool {
		// construct the mock chunk by referencing the required columns
		for (idx_t i = 0; i < columns.size(); i++) {
			mock_chunk.data[columns[i]].Reference(chunk.data[i]);
		}
		mock_chunk.SetCardinality(chunk);
		// append this chunk to the indexes of the table
		if (!DataTable::AppendToIndexes(index_list, mock_chunk, start_row)) {
			success = false;
			return false;
		}
		start_row += chunk.size();
		return true;
	});
	return success;
}

void LocalTableStorage::AppendToIndexes(DuckTransaction &transaction, TableAppendState &append_state,
                                        idx_t append_count, bool append_to_table) {
	bool constraint_violated = false;
	if (append_to_table) {
		table->InitializeAppend(transaction, append_state, append_count);
	}
	if (append_to_table) {
		// appending: need to scan entire
		row_groups->Scan(transaction, [&](DataChunk &chunk) -> bool {
			// append this chunk to the indexes of the table
			if (!table->AppendToIndexes(chunk, append_state.current_row)) {
				constraint_violated = true;
				return false;
			}
			// append to base table
			table->Append(chunk, append_state);
			return true;
		});
	} else {
		constraint_violated = !AppendToIndexes(transaction, *row_groups, table->info->indexes, table->GetTypes(),
		                                       append_state.current_row);
	}
	if (constraint_violated) {
		PreservedError error;
		// need to revert the append
		row_t current_row = append_state.row_start;
		// remove the data from the indexes, if there are any indexes
		row_groups->Scan(transaction, [&](DataChunk &chunk) -> bool {
			// append this chunk to the indexes of the table
			try {
				table->RemoveFromIndexes(append_state, chunk, current_row);
			} catch (Exception &ex) {
				error = PreservedError(ex);
				return false;
			} catch (std::exception &ex) {
				error = PreservedError(ex);
				return false;
			}

			current_row += chunk.size();
			if (current_row >= append_state.current_row) {
				// finished deleting all rows from the index: abort now
				return false;
			}
			return true;
		});
		if (append_to_table) {
			table->RevertAppendInternal(append_state.row_start, append_count);
		}
		if (error) {
			error.Throw();
		}
		throw ConstraintException("PRIMARY KEY or UNIQUE constraint violated: duplicated key");
	}
}

OptimisticDataWriter *LocalTableStorage::CreateOptimisticWriter() {
	auto writer = make_unique<OptimisticDataWriter>(table);
	optimistic_writers.push_back(std::move(writer));
	return optimistic_writers.back().get();
}

void LocalTableStorage::Rollback() {
	optimistic_writer.Rollback();
	for (auto &writer : optimistic_writers) {
		writer->Rollback();
	}
	optimistic_writers.clear();
}

//===--------------------------------------------------------------------===//
// LocalTableManager
//===--------------------------------------------------------------------===//
LocalTableStorage *LocalTableManager::GetStorage(DataTable *table) {
	lock_guard<mutex> l(table_storage_lock);
	auto entry = table_storage.find(table);
	return entry == table_storage.end() ? nullptr : entry->second.get();
}

LocalTableStorage *LocalTableManager::GetOrCreateStorage(DataTable *table) {
	lock_guard<mutex> l(table_storage_lock);
	auto entry = table_storage.find(table);
	if (entry == table_storage.end()) {
		auto new_storage = make_shared<LocalTableStorage>(*table);
		auto storage = new_storage.get();
		table_storage.insert(make_pair(table, std::move(new_storage)));
		return storage;
	} else {
		return entry->second.get();
	}
}

bool LocalTableManager::IsEmpty() {
	lock_guard<mutex> l(table_storage_lock);
	return table_storage.empty();
}

shared_ptr<LocalTableStorage> LocalTableManager::MoveEntry(DataTable *table) {
	lock_guard<mutex> l(table_storage_lock);
	auto entry = table_storage.find(table);
	if (entry == table_storage.end()) {
		return nullptr;
	}
	auto storage_entry = std::move(entry->second);
	table_storage.erase(table);
	return storage_entry;
}

unordered_map<DataTable *, shared_ptr<LocalTableStorage>> LocalTableManager::MoveEntries() {
	lock_guard<mutex> l(table_storage_lock);
	return std::move(table_storage);
}

idx_t LocalTableManager::EstimatedSize() {
	lock_guard<mutex> l(table_storage_lock);
	idx_t estimated_size = 0;
	for (auto &storage : table_storage) {
		estimated_size += storage.second->EstimatedSize();
	}
	return estimated_size;
}

void LocalTableManager::InsertEntry(DataTable *table, shared_ptr<LocalTableStorage> entry) {
	lock_guard<mutex> l(table_storage_lock);
	D_ASSERT(table_storage.find(table) == table_storage.end());
	table_storage[table] = std::move(entry);
}

//===--------------------------------------------------------------------===//
// LocalStorage
//===--------------------------------------------------------------------===//
LocalStorage::LocalStorage(ClientContext &context, DuckTransaction &transaction)
    : context(context), transaction(transaction) {
}

LocalStorage &LocalStorage::Get(DuckTransaction &transaction) {
	return transaction.GetLocalStorage();
}

LocalStorage &LocalStorage::Get(ClientContext &context, AttachedDatabase &db) {
	return DuckTransaction::Get(context, db).GetLocalStorage();
}

LocalStorage &LocalStorage::Get(ClientContext &context, Catalog &catalog) {
	return LocalStorage::Get(context, catalog.GetAttached());
}

void LocalStorage::InitializeScan(DataTable *table, CollectionScanState &state, TableFilterSet *table_filters) {
	auto storage = table_manager.GetStorage(table);
	if (storage == nullptr) {
		return;
	}
	storage->InitializeScan(state, table_filters);
}

void LocalStorage::Scan(CollectionScanState &state, const vector<column_t> &column_ids, DataChunk &result) {
	state.Scan(transaction, result);
}

void LocalStorage::InitializeParallelScan(DataTable *table, ParallelCollectionScanState &state) {
	auto storage = table_manager.GetStorage(table);
	if (!storage) {
		state.max_row = 0;
		state.vector_index = 0;
		state.current_row_group = nullptr;
	} else {
		storage->row_groups->InitializeParallelScan(state);
	}
}

bool LocalStorage::NextParallelScan(ClientContext &context, DataTable *table, ParallelCollectionScanState &state,
                                    CollectionScanState &scan_state) {
	auto storage = table_manager.GetStorage(table);
	if (!storage) {
		return false;
	}
	return storage->row_groups->NextParallelScan(context, state, scan_state);
}

void LocalStorage::InitializeAppend(LocalAppendState &state, DataTable *table) {
	state.storage = table_manager.GetOrCreateStorage(table);
	state.storage->row_groups->InitializeAppend(TransactionData(transaction), state.append_state, 0);
}

void LocalStorage::Append(LocalAppendState &state, DataChunk &chunk) {
	// append to unique indices (if any)
	auto storage = state.storage;
	idx_t base_id = MAX_ROW_ID + storage->row_groups->GetTotalRows() + state.append_state.total_append_count;
	if (!DataTable::AppendToIndexes(storage->indexes, chunk, base_id)) {
		throw ConstraintException("PRIMARY KEY or UNIQUE constraint violated: duplicated key");
	}

	//! Append the chunk to the local storage
	auto new_row_group = storage->row_groups->Append(chunk, state.append_state);

	//! Check if we should pre-emptively flush blocks to disk
	if (new_row_group) {
		storage->CheckFlushToDisk();
	}
}

void LocalStorage::FinalizeAppend(LocalAppendState &state) {
	state.storage->row_groups->FinalizeAppend(state.append_state.transaction, state.append_state);
}

void LocalStorage::LocalMerge(DataTable *table, RowGroupCollection &collection) {
	auto storage = table_manager.GetOrCreateStorage(table);
	if (!storage->indexes.Empty()) {
		// append data to indexes if required
		row_t base_id = MAX_ROW_ID + storage->row_groups->GetTotalRows();
		bool success = storage->AppendToIndexes(transaction, collection, storage->indexes, table->GetTypes(), base_id);
		if (!success) {
			throw ConstraintException("PRIMARY KEY or UNIQUE constraint violated: duplicated key");
		}
	}
	storage->row_groups->MergeStorage(collection);
}

OptimisticDataWriter *LocalStorage::CreateOptimisticWriter(DataTable *table) {
	auto storage = table_manager.GetOrCreateStorage(table);
	return storage->CreateOptimisticWriter();
}

bool LocalStorage::ChangesMade() noexcept {
	return !table_manager.IsEmpty();
}

bool LocalStorage::Find(DataTable *table) {
	return table_manager.GetStorage(table) != nullptr;
}

idx_t LocalStorage::EstimatedSize() {
	return table_manager.EstimatedSize();
}

idx_t LocalStorage::Delete(DataTable *table, Vector &row_ids, idx_t count) {
	auto storage = table_manager.GetStorage(table);
	D_ASSERT(storage);

	// delete from unique indices (if any)
	if (!storage->indexes.Empty()) {
		storage->row_groups->RemoveFromIndexes(storage->indexes, row_ids, count);
	}

	auto ids = FlatVector::GetData<row_t>(row_ids);
	idx_t delete_count = storage->row_groups->Delete(TransactionData(0, 0), table, ids, count);
	storage->deleted_rows += delete_count;
	return delete_count;
}

void LocalStorage::Update(DataTable *table, Vector &row_ids, const vector<PhysicalIndex> &column_ids,
                          DataChunk &updates) {
	auto storage = table_manager.GetStorage(table);
	D_ASSERT(storage);

	auto ids = FlatVector::GetData<row_t>(row_ids);
	storage->row_groups->Update(TransactionData(0, 0), ids, column_ids, updates);
}

void LocalStorage::Flush(DataTable &table, LocalTableStorage &storage) {
	if (storage.row_groups->GetTotalRows() <= storage.deleted_rows) {
		return;
	}
	idx_t append_count = storage.row_groups->GetTotalRows() - storage.deleted_rows;

	TableAppendState append_state;
	table.AppendLock(append_state);
	if ((append_state.row_start == 0 || storage.row_groups->GetTotalRows() >= MERGE_THRESHOLD) &&
	    storage.deleted_rows == 0) {
		// table is currently empty OR we are bulk appending: move over the storage directly
		// first flush any out-standing storage nodes
		storage.FlushToDisk();
		// now append to the indexes (if there are any)
		// FIXME: we should be able to merge the transaction-local index directly into the main table index
		// as long we just rewrite some row-ids
		if (!table.info->indexes.Empty()) {
			storage.AppendToIndexes(transaction, append_state, append_count, false);
		}
		// finally move over the row groups
		table.MergeStorage(*storage.row_groups, storage.indexes);
	} else {
		// check if we have written data
		// if we have, we cannot merge to disk after all
		// so we need to revert the data we have already written
		storage.Rollback();
		// append to the indexes and append to the base table
		storage.AppendToIndexes(transaction, append_state, append_count, true);
	}
	transaction.PushAppend(&table, append_state.row_start, append_count);
}

void LocalStorage::Commit(LocalStorage::CommitState &commit_state, DuckTransaction &transaction) {
	// commit local storage
	// iterate over all entries in the table storage map and commit them
	// after this, the local storage is no longer required and can be cleared
	auto table_storage = table_manager.MoveEntries();
	for (auto &entry : table_storage) {
		auto table = entry.first;
		auto storage = entry.second.get();
		Flush(*table, *storage);

		entry.second.reset();
	}
}

void LocalStorage::Rollback() {
	// rollback local storage
	// after this, the local storage is no longer required and can be cleared
	auto table_storage = table_manager.MoveEntries();
	for (auto &entry : table_storage) {
		auto storage = entry.second.get();
		if (!storage) {
			continue;
		}
		storage->Rollback();

		entry.second.reset();
	}
}

idx_t LocalStorage::AddedRows(DataTable *table) {
	auto storage = table_manager.GetStorage(table);
	if (!storage) {
		return 0;
	}
	return storage->row_groups->GetTotalRows() - storage->deleted_rows;
}

void LocalStorage::MoveStorage(DataTable *old_dt, DataTable *new_dt) {
	// check if there are any pending appends for the old version of the table
	auto new_storage = table_manager.MoveEntry(old_dt);
	if (!new_storage) {
		return;
	}
	// take over the storage from the old entry
	new_storage->table = new_dt;
	table_manager.InsertEntry(new_dt, std::move(new_storage));
}

void LocalStorage::AddColumn(DataTable *old_dt, DataTable *new_dt, ColumnDefinition &new_column,
                             Expression *default_value) {
	// check if there are any pending appends for the old version of the table
	auto storage = table_manager.MoveEntry(old_dt);
	if (!storage) {
		return;
	}
	auto new_storage = make_unique<LocalTableStorage>(context, *new_dt, *storage, new_column, default_value);
	table_manager.InsertEntry(new_dt, std::move(new_storage));
}

void LocalStorage::DropColumn(DataTable *old_dt, DataTable *new_dt, idx_t removed_column) {
	// check if there are any pending appends for the old version of the table
	auto storage = table_manager.MoveEntry(old_dt);
	if (!storage) {
		return;
	}
	auto new_storage = make_unique<LocalTableStorage>(*new_dt, *storage, removed_column);
	table_manager.InsertEntry(new_dt, std::move(new_storage));
}

void LocalStorage::ChangeType(DataTable *old_dt, DataTable *new_dt, idx_t changed_idx, const LogicalType &target_type,
                              const vector<column_t> &bound_columns, Expression &cast_expr) {
	// check if there are any pending appends for the old version of the table
	auto storage = table_manager.MoveEntry(old_dt);
	if (!storage) {
		return;
	}
	auto new_storage =
	    make_unique<LocalTableStorage>(context, *new_dt, *storage, changed_idx, target_type, bound_columns, cast_expr);
	table_manager.InsertEntry(new_dt, std::move(new_storage));
}

void LocalStorage::FetchChunk(DataTable *table, Vector &row_ids, idx_t count, const vector<column_t> &col_ids,
                              DataChunk &chunk, ColumnFetchState &fetch_state) {
	auto storage = table_manager.GetStorage(table);
	if (!storage) {
		throw InternalException("LocalStorage::FetchChunk - local storage not found");
	}

	storage->row_groups->Fetch(transaction, chunk, col_ids, row_ids, count, fetch_state);
}

TableIndexList &LocalStorage::GetIndexes(DataTable *table) {
	auto storage = table_manager.GetStorage(table);
	if (!storage) {
		throw InternalException("LocalStorage::GetIndexes - local storage not found");
	}
	return storage->indexes;
}

void LocalStorage::VerifyNewConstraint(DataTable &parent, const BoundConstraint &constraint) {
	auto storage = table_manager.GetStorage(&parent);
	if (!storage) {
		return;
	}
	storage->row_groups->VerifyNewConstraint(parent, constraint);
}

} // namespace duckdb




namespace duckdb {

DataFileType MagicBytes::CheckMagicBytes(FileSystem *fs_p, const string &path) {
	LocalFileSystem lfs;
	FileSystem &fs = fs_p ? *fs_p : lfs;
	if (!fs.FileExists(path)) {
		return DataFileType::FILE_DOES_NOT_EXIST;
	}
	auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_READ);

	constexpr const idx_t MAGIC_BYTES_READ_SIZE = 16;
	char buffer[MAGIC_BYTES_READ_SIZE];

	handle->Read(buffer, MAGIC_BYTES_READ_SIZE);
	if (memcmp(buffer, "SQLite format 3\0\0\0", 16) == 0) {
		return DataFileType::SQLITE_FILE;
	}
	if (memcmp(buffer, "PAR1", 4) == 0) {
		return DataFileType::PARQUET_FILE;
	}
	if (memcmp(buffer + MainHeader::MAGIC_BYTE_OFFSET, MainHeader::MAGIC_BYTES, MainHeader::MAGIC_BYTE_SIZE) == 0) {
		return DataFileType::DUCKDB_FILE;
	}
	return DataFileType::FILE_DOES_NOT_EXIST;
}

} // namespace duckdb



#include <cstring>

namespace duckdb {

MetaBlockReader::MetaBlockReader(BlockManager &block_manager, block_id_t block_id, bool free_blocks_on_read)
    : block_manager(block_manager), offset(0), next_block(-1), free_blocks_on_read(free_blocks_on_read) {
	ReadNewBlock(block_id);
}

MetaBlockReader::~MetaBlockReader() {
}

void MetaBlockReader::ReadData(data_ptr_t buffer, idx_t read_size) {
	while (offset + read_size > handle.GetFileBuffer().size) {
		// cannot read entire entry from block
		// first read what we can from this block
		idx_t to_read = handle.GetFileBuffer().size - offset;
		if (to_read > 0) {
			memcpy(buffer, handle.Ptr() + offset, to_read);
			read_size -= to_read;
			buffer += to_read;
		}
		// then move to the next block
		if (next_block == INVALID_BLOCK) {
			throw IOException("Cannot read from INVALID_BLOCK.");
		}
		ReadNewBlock(next_block);
	}
	// we have enough left in this block to read from the buffer
	memcpy(buffer, handle.Ptr() + offset, read_size);
	offset += read_size;
}

void MetaBlockReader::ReadNewBlock(block_id_t id) {
	auto &buffer_manager = block_manager.buffer_manager;

	// Marking these blocks as modified will cause them to be moved to the free
	// list upon the next successful checkpoint. Marking them modified here
	// assumes MetaBlockReader is exclusively used for reading checkpoint data,
	// and thus any blocks we're reading will be obviated by the next checkpoint.
	if (free_blocks_on_read) {
		block_manager.MarkBlockAsModified(id);
	}
	block = block_manager.RegisterBlock(id, true);
	handle = buffer_manager.Pin(block);

	next_block = Load<block_id_t>(handle.Ptr());
	D_ASSERT(next_block >= -1);
	offset = sizeof(block_id_t);
}

} // namespace duckdb


#include <cstring>

namespace duckdb {

MetaBlockWriter::MetaBlockWriter(BlockManager &block_manager, block_id_t initial_block_id)
    : block_manager(block_manager) {
	if (initial_block_id == INVALID_BLOCK) {
		initial_block_id = MetaBlockWriter::GetNextBlockId();
	}
	block = block_manager.CreateBlock(initial_block_id, nullptr);
	Store<block_id_t>(-1, block->buffer);
	offset = sizeof(block_id_t);
}

MetaBlockWriter::~MetaBlockWriter() {
	// If there's an exception during checkpoint, this can get destroyed without
	// flushing the data...which is fine, because none of the unwritten data
	// will be referenced.
	//
	// Otherwise, we should have explicitly flushed (and thereby nulled the block).
	D_ASSERT(!block || Exception::UncaughtException());
}

block_id_t MetaBlockWriter::GetNextBlockId() {
	return block_manager.GetFreeBlockId();
}

BlockPointer MetaBlockWriter::GetBlockPointer() {
	BlockPointer pointer;
	pointer.block_id = block->id;
	pointer.offset = offset;
	return pointer;
}

void MetaBlockWriter::Flush() {
	AdvanceBlock();
	block = nullptr;
}

void MetaBlockWriter::AdvanceBlock() {
	written_blocks.insert(block->id);
	if (offset > sizeof(block_id_t)) {
		block_manager.Write(*block);
		offset = sizeof(block_id_t);
	}
}

void MetaBlockWriter::WriteData(const_data_ptr_t buffer, idx_t write_size) {
	while (offset + write_size > block->size) {
		// we need to make a new block
		// first copy what we can
		D_ASSERT(offset <= block->size);
		idx_t copy_amount = block->size - offset;
		if (copy_amount > 0) {
			memcpy(block->buffer + offset, buffer, copy_amount);
			buffer += copy_amount;
			offset += copy_amount;
			write_size -= copy_amount;
		}
		// now we need to get a new block id
		block_id_t new_block_id = GetNextBlockId();
		// write the block id of the new block to the start of the current block
		Store<block_id_t>(new_block_id, block->buffer);
		// first flush the old block
		AdvanceBlock();
		// now update the block id of the block
		block->id = new_block_id;
		Store<block_id_t>(-1, block->buffer);
	}
	memcpy(block->buffer + offset, buffer, write_size);
	offset += write_size;
}

} // namespace duckdb


namespace duckdb {

PartialBlockManager::PartialBlockManager(BlockManager &block_manager, uint32_t max_partial_block_size,
                                         uint32_t max_use_count)
    : block_manager(block_manager), max_partial_block_size(max_partial_block_size), max_use_count(max_use_count) {
}
PartialBlockManager::~PartialBlockManager() {
}
//===--------------------------------------------------------------------===//
// Partial Blocks
//===--------------------------------------------------------------------===//
PartialBlockAllocation PartialBlockManager::GetBlockAllocation(uint32_t segment_size) {
	PartialBlockAllocation allocation;
	allocation.block_manager = &block_manager;
	allocation.allocation_size = segment_size;

	// if the block is less than 80% full, we consider it a "partial block"
	// which means we will try to fit it with other blocks
	// check if there is a partial block available we can write to
	if (segment_size <= max_partial_block_size && GetPartialBlock(segment_size, allocation.partial_block)) {
		//! there is! increase the reference count of this block
		allocation.partial_block->state.block_use_count += 1;
		allocation.state = allocation.partial_block->state;
		block_manager.IncreaseBlockReferenceCount(allocation.state.block_id);
	} else {
		// full block: get a free block to write to
		AllocateBlock(allocation.state, segment_size);
	}
	return allocation;
}

void PartialBlockManager::AllocateBlock(PartialBlockState &state, uint32_t segment_size) {
	D_ASSERT(segment_size <= Storage::BLOCK_SIZE);
	state.block_id = block_manager.GetFreeBlockId();
	state.block_size = Storage::BLOCK_SIZE;
	state.offset_in_block = 0;
	state.block_use_count = 1;
}

bool PartialBlockManager::GetPartialBlock(idx_t segment_size, unique_ptr<PartialBlock> &partial_block) {
	auto entry = partially_filled_blocks.lower_bound(segment_size);
	if (entry == partially_filled_blocks.end()) {
		return false;
	}
	// found a partially filled block! fill in the info
	partial_block = std::move(entry->second);
	partially_filled_blocks.erase(entry);

	D_ASSERT(partial_block->state.offset_in_block > 0);
	D_ASSERT(ValueIsAligned(partial_block->state.offset_in_block));
	return true;
}

void PartialBlockManager::RegisterPartialBlock(PartialBlockAllocation &&allocation) {
	auto &state(allocation.partial_block->state);
	if (state.block_use_count < max_use_count) {
		auto new_size = AlignValue(allocation.allocation_size + state.offset_in_block);
		state.offset_in_block = new_size;
		auto new_space_left = state.block_size - new_size;
		// check if the block is STILL partially filled after adding the segment_size
		if (new_space_left >= Storage::BLOCK_SIZE - max_partial_block_size) {
			// the block is still partially filled: add it to the partially_filled_blocks list
			partially_filled_blocks.insert(make_pair(new_space_left, std::move(allocation.partial_block)));
		}
	}
	auto block_to_free = std::move(allocation.partial_block);
	if (!block_to_free && partially_filled_blocks.size() > MAX_BLOCK_MAP_SIZE) {
		// Free the page with the least space free.
		auto itr = partially_filled_blocks.begin();
		block_to_free = std::move(itr->second);
		partially_filled_blocks.erase(itr);
	}
	// Flush any block that we're not going to reuse.
	if (block_to_free) {
		block_to_free->Flush();
	}
}

void PartialBlockManager::FlushPartialBlocks() {
	for (auto &e : partially_filled_blocks) {
		e.second->Flush();
	}
	partially_filled_blocks.clear();
}

void PartialBlockManager::Clear() {
	for (auto &e : partially_filled_blocks) {
		e.second->Clear();
	}
	partially_filled_blocks.clear();
}

} // namespace duckdb













#include <algorithm>
#include <cstring>

namespace duckdb {

const char MainHeader::MAGIC_BYTES[] = "DUCK";

void MainHeader::Serialize(Serializer &ser) {
	ser.WriteData((data_ptr_t)MAGIC_BYTES, MAGIC_BYTE_SIZE);
	ser.Write<uint64_t>(version_number);
	FieldWriter writer(ser);
	for (idx_t i = 0; i < FLAG_COUNT; i++) {
		writer.WriteField<uint64_t>(flags[i]);
	}
	writer.Finalize();
}

void MainHeader::CheckMagicBytes(FileHandle &handle) {
	data_t magic_bytes[MAGIC_BYTE_SIZE];
	if (handle.GetFileSize() < MainHeader::MAGIC_BYTE_SIZE + MainHeader::MAGIC_BYTE_OFFSET) {
		throw IOException("The file \"%s\" exists, but it is not a valid DuckDB database file!", handle.path);
	}
	handle.Read(magic_bytes, MainHeader::MAGIC_BYTE_SIZE, MainHeader::MAGIC_BYTE_OFFSET);
	if (memcmp(magic_bytes, MainHeader::MAGIC_BYTES, MainHeader::MAGIC_BYTE_SIZE) != 0) {
		throw IOException("The file \"%s\" exists, but it is not a valid DuckDB database file!", handle.path);
	}
}

MainHeader MainHeader::Deserialize(Deserializer &source) {
	data_t magic_bytes[MAGIC_BYTE_SIZE];
	MainHeader header;
	source.ReadData(magic_bytes, MainHeader::MAGIC_BYTE_SIZE);
	if (memcmp(magic_bytes, MainHeader::MAGIC_BYTES, MainHeader::MAGIC_BYTE_SIZE) != 0) {
		throw IOException("The file is not a valid DuckDB database file!");
	}
	header.version_number = source.Read<uint64_t>();
	// check the version number
	if (header.version_number != VERSION_NUMBER) {
		auto version = GetDuckDBVersion(header.version_number);
		string version_text;
		if (version) {
			// known version
			version_text = "DuckDB version " + string(version);
		} else {
			version_text = string("an ") + (VERSION_NUMBER > header.version_number ? "older development" : "newer") +
			               string(" version of DuckDB");
		}
		throw IOException(
		    "Trying to read a database file with version number %lld, but we can only read version %lld.\n"
		    "The database file was created with %s.\n\n"
		    "The storage of DuckDB is not yet stable; newer versions of DuckDB cannot read old database files and "
		    "vice versa.\n"
		    "The storage will be stabilized when version 1.0 releases.\n\n"
		    "For now, we recommend that you load the database file in a supported version of DuckDB, and use the "
		    "EXPORT DATABASE command "
		    "followed by IMPORT DATABASE on the current version of DuckDB.\n\n"
		    "See the storage page for more information: https://duckdb.org/internals/storage",
		    header.version_number, VERSION_NUMBER, version_text);
	}
	// read the flags
	FieldReader reader(source);
	for (idx_t i = 0; i < FLAG_COUNT; i++) {
		header.flags[i] = reader.ReadRequired<uint64_t>();
	}
	reader.Finalize();
	return header;
}

void DatabaseHeader::Serialize(Serializer &ser) {
	ser.Write<uint64_t>(iteration);
	ser.Write<block_id_t>(meta_block);
	ser.Write<block_id_t>(free_list);
	ser.Write<uint64_t>(block_count);
}

DatabaseHeader DatabaseHeader::Deserialize(Deserializer &source) {
	DatabaseHeader header;
	header.iteration = source.Read<uint64_t>();
	header.meta_block = source.Read<block_id_t>();
	header.free_list = source.Read<block_id_t>();
	header.block_count = source.Read<uint64_t>();
	return header;
}

template <class T>
void SerializeHeaderStructure(T header, data_ptr_t ptr) {
	BufferedSerializer ser(ptr, Storage::FILE_HEADER_SIZE);
	header.Serialize(ser);
}

template <class T>
T DeserializeHeaderStructure(data_ptr_t ptr) {
	BufferedDeserializer source(ptr, Storage::FILE_HEADER_SIZE);
	return T::Deserialize(source);
}

SingleFileBlockManager::SingleFileBlockManager(AttachedDatabase &db, string path_p, bool read_only, bool use_direct_io)
    : BlockManager(BufferManager::GetBufferManager(db)), db(db), path(std::move(path_p)),
      header_buffer(Allocator::Get(db), FileBufferType::MANAGED_BUFFER,
                    Storage::FILE_HEADER_SIZE - Storage::BLOCK_HEADER_SIZE),
      iteration_count(0), read_only(read_only), use_direct_io(use_direct_io) {
}

void SingleFileBlockManager::GetFileFlags(uint8_t &flags, FileLockType &lock, bool create_new) {
	if (read_only) {
		D_ASSERT(!create_new);
		flags = FileFlags::FILE_FLAGS_READ;
		lock = FileLockType::READ_LOCK;
	} else {
		flags = FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_READ;
		lock = FileLockType::WRITE_LOCK;
		if (create_new) {
			flags |= FileFlags::FILE_FLAGS_FILE_CREATE;
		}
	}
	if (use_direct_io) {
		flags |= FileFlags::FILE_FLAGS_DIRECT_IO;
	}
}

void SingleFileBlockManager::CreateNewDatabase() {
	uint8_t flags;
	FileLockType lock;
	GetFileFlags(flags, lock, true);

	// open the RDBMS handle
	auto &fs = FileSystem::Get(db);
	handle = fs.OpenFile(path, flags, lock);

	// if we create a new file, we fill the metadata of the file
	// first fill in the new header
	header_buffer.Clear();

	MainHeader main_header;
	main_header.version_number = VERSION_NUMBER;
	memset(main_header.flags, 0, sizeof(uint64_t) * 4);

	SerializeHeaderStructure<MainHeader>(main_header, header_buffer.buffer);
	// now write the header to the file
	ChecksumAndWrite(header_buffer, 0);
	header_buffer.Clear();

	// write the database headers
	// initialize meta_block and free_list to INVALID_BLOCK because the database file does not contain any actual
	// content yet
	DatabaseHeader h1, h2;
	// header 1
	h1.iteration = 0;
	h1.meta_block = INVALID_BLOCK;
	h1.free_list = INVALID_BLOCK;
	h1.block_count = 0;
	SerializeHeaderStructure<DatabaseHeader>(h1, header_buffer.buffer);
	ChecksumAndWrite(header_buffer, Storage::FILE_HEADER_SIZE);
	// header 2
	h2.iteration = 0;
	h2.meta_block = INVALID_BLOCK;
	h2.free_list = INVALID_BLOCK;
	h2.block_count = 0;
	SerializeHeaderStructure<DatabaseHeader>(h2, header_buffer.buffer);
	ChecksumAndWrite(header_buffer, Storage::FILE_HEADER_SIZE * 2);
	// ensure that writing to disk is completed before returning
	handle->Sync();
	// we start with h2 as active_header, this way our initial write will be in h1
	iteration_count = 0;
	active_header = 1;
	max_block = 0;
}

void SingleFileBlockManager::LoadExistingDatabase() {
	uint8_t flags;
	FileLockType lock;
	GetFileFlags(flags, lock, false);

	// open the RDBMS handle
	auto &fs = FileSystem::Get(db);
	handle = fs.OpenFile(path, flags, lock);

	MainHeader::CheckMagicBytes(*handle);
	// otherwise, we check the metadata of the file
	ReadAndChecksum(header_buffer, 0);
	DeserializeHeaderStructure<MainHeader>(header_buffer.buffer);

	// read the database headers from disk
	DatabaseHeader h1, h2;
	ReadAndChecksum(header_buffer, Storage::FILE_HEADER_SIZE);
	h1 = DeserializeHeaderStructure<DatabaseHeader>(header_buffer.buffer);
	ReadAndChecksum(header_buffer, Storage::FILE_HEADER_SIZE * 2);
	h2 = DeserializeHeaderStructure<DatabaseHeader>(header_buffer.buffer);
	// check the header with the highest iteration count
	if (h1.iteration > h2.iteration) {
		// h1 is active header
		active_header = 0;
		Initialize(h1);
	} else {
		// h2 is active header
		active_header = 1;
		Initialize(h2);
	}
	LoadFreeList();
}

void SingleFileBlockManager::ReadAndChecksum(FileBuffer &block, uint64_t location) const {
	// read the buffer from disk
	block.Read(*handle, location);
	// compute the checksum
	auto stored_checksum = Load<uint64_t>(block.InternalBuffer());
	uint64_t computed_checksum = Checksum(block.buffer, block.size);
	// verify the checksum
	if (stored_checksum != computed_checksum) {
		throw IOException("Corrupt database file: computed checksum %llu does not match stored checksum %llu in block",
		                  computed_checksum, stored_checksum);
	}
}

void SingleFileBlockManager::ChecksumAndWrite(FileBuffer &block, uint64_t location) const {
	// compute the checksum and write it to the start of the buffer (if not temp buffer)
	uint64_t checksum = Checksum(block.buffer, block.size);
	Store<uint64_t>(checksum, block.InternalBuffer());
	// now write the buffer
	block.Write(*handle, location);
}

void SingleFileBlockManager::Initialize(DatabaseHeader &header) {
	free_list_id = header.free_list;
	meta_block = header.meta_block;
	iteration_count = header.iteration;
	max_block = header.block_count;
}

void SingleFileBlockManager::LoadFreeList() {
	if (read_only) {
		// no need to load free list for read only db
		return;
	}
	if (free_list_id == INVALID_BLOCK) {
		// no free list
		return;
	}
	MetaBlockReader reader(*this, free_list_id);
	auto free_list_count = reader.Read<uint64_t>();
	free_list.clear();
	for (idx_t i = 0; i < free_list_count; i++) {
		free_list.insert(reader.Read<block_id_t>());
	}
	auto multi_use_blocks_count = reader.Read<uint64_t>();
	multi_use_blocks.clear();
	for (idx_t i = 0; i < multi_use_blocks_count; i++) {
		auto block_id = reader.Read<block_id_t>();
		auto usage_count = reader.Read<uint32_t>();
		multi_use_blocks[block_id] = usage_count;
	}
}

bool SingleFileBlockManager::IsRootBlock(block_id_t root) {
	return root == meta_block;
}

block_id_t SingleFileBlockManager::GetFreeBlockId() {
	lock_guard<mutex> lock(block_lock);
	block_id_t block;
	if (!free_list.empty()) {
		// free list is non empty
		// take an entry from the free list
		block = *free_list.begin();
		// erase the entry from the free list again
		free_list.erase(free_list.begin());
	} else {
		block = max_block++;
	}
	return block;
}

void SingleFileBlockManager::MarkBlockAsFree(block_id_t block_id) {
	lock_guard<mutex> lock(block_lock);
	D_ASSERT(block_id >= 0);
	D_ASSERT(block_id < max_block);
	D_ASSERT(free_list.find(block_id) == free_list.end());
	multi_use_blocks.erase(block_id);
	free_list.insert(block_id);
}

void SingleFileBlockManager::MarkBlockAsModified(block_id_t block_id) {
	lock_guard<mutex> lock(block_lock);
	D_ASSERT(block_id >= 0);
	D_ASSERT(block_id < max_block);

	// check if the block is a multi-use block
	auto entry = multi_use_blocks.find(block_id);
	if (entry != multi_use_blocks.end()) {
		// it is! reduce the reference count of the block
		entry->second--;
		// check the reference count: is the block still a multi-use block?
		if (entry->second <= 1) {
			// no longer a multi-use block!
			multi_use_blocks.erase(entry);
		}
		return;
	}
	// Check for multi-free
	// TODO: Fix the bug that causes this assert to fire, then uncomment it.
	// D_ASSERT(modified_blocks.find(block_id) == modified_blocks.end());
	D_ASSERT(free_list.find(block_id) == free_list.end());
	modified_blocks.insert(block_id);
}

void SingleFileBlockManager::IncreaseBlockReferenceCount(block_id_t block_id) {
	lock_guard<mutex> lock(block_lock);
	D_ASSERT(block_id >= 0);
	D_ASSERT(block_id < max_block);
	D_ASSERT(free_list.find(block_id) == free_list.end());
	auto entry = multi_use_blocks.find(block_id);
	if (entry != multi_use_blocks.end()) {
		entry->second++;
	} else {
		multi_use_blocks[block_id] = 2;
	}
}

block_id_t SingleFileBlockManager::GetMetaBlock() {
	return meta_block;
}

idx_t SingleFileBlockManager::TotalBlocks() {
	lock_guard<mutex> lock(block_lock);
	return max_block;
}

idx_t SingleFileBlockManager::FreeBlocks() {
	lock_guard<mutex> lock(block_lock);
	return free_list.size();
}

unique_ptr<Block> SingleFileBlockManager::CreateBlock(block_id_t block_id, FileBuffer *source_buffer) {
	if (source_buffer) {
		D_ASSERT(source_buffer->AllocSize() == Storage::BLOCK_ALLOC_SIZE);
		return make_unique<Block>(*source_buffer, block_id);
	} else {
		return make_unique<Block>(Allocator::Get(db), block_id);
	}
}

void SingleFileBlockManager::Read(Block &block) {
	D_ASSERT(block.id >= 0);
	D_ASSERT(std::find(free_list.begin(), free_list.end(), block.id) == free_list.end());
	ReadAndChecksum(block, BLOCK_START + block.id * Storage::BLOCK_ALLOC_SIZE);
}

void SingleFileBlockManager::Write(FileBuffer &buffer, block_id_t block_id) {
	D_ASSERT(block_id >= 0);
	ChecksumAndWrite(buffer, BLOCK_START + block_id * Storage::BLOCK_ALLOC_SIZE);
}

vector<block_id_t> SingleFileBlockManager::GetFreeListBlocks() {
	vector<block_id_t> free_list_blocks;

	if (!free_list.empty() || !multi_use_blocks.empty() || !modified_blocks.empty()) {
		// there are blocks in the free list or multi_use_blocks
		// figure out how many blocks we need to write these to the file
		auto free_list_size = sizeof(uint64_t) + sizeof(block_id_t) * (free_list.size() + modified_blocks.size());
		auto multi_use_blocks_size =
		    sizeof(uint64_t) + (sizeof(block_id_t) + sizeof(uint32_t)) * multi_use_blocks.size();
		auto total_size = free_list_size + multi_use_blocks_size;
		// because of potential alignment issues and needing to store a next pointer in a block we subtract
		// a bit from the max block size
		auto space_in_block = Storage::BLOCK_SIZE - 4 * sizeof(block_id_t);
		auto total_blocks = (total_size + space_in_block - 1) / space_in_block;
		D_ASSERT(total_size > 0);
		D_ASSERT(total_blocks > 0);

		// reserve the blocks that we are going to write
		// since these blocks are no longer free we cannot just include them in the free list!
		for (idx_t i = 0; i < total_blocks; i++) {
			auto block_id = GetFreeBlockId();
			free_list_blocks.push_back(block_id);
		}
	}

	return free_list_blocks;
}

class FreeListBlockWriter : public MetaBlockWriter {
public:
	FreeListBlockWriter(BlockManager &block_manager, vector<block_id_t> &free_list_blocks_p)
	    : MetaBlockWriter(block_manager, free_list_blocks_p[0]), free_list_blocks(free_list_blocks_p), index(1) {
	}

	vector<block_id_t> &free_list_blocks;
	idx_t index;

protected:
	block_id_t GetNextBlockId() override {
		if (index >= free_list_blocks.size()) {
			throw InternalException(
			    "Free List Block Writer ran out of blocks, this means not enough blocks were allocated up front");
		}
		return free_list_blocks[index++];
	}
};

void SingleFileBlockManager::WriteHeader(DatabaseHeader header) {
	// set the iteration count
	header.iteration = ++iteration_count;

	vector<block_id_t> free_list_blocks = GetFreeListBlocks();

	// now handle the free list
	// add all modified blocks to the free list: they can now be written to again
	for (auto &block : modified_blocks) {
		free_list.insert(block);
	}
	modified_blocks.clear();

	if (!free_list_blocks.empty()) {
		// there are blocks to write, either in the free_list or in the modified_blocks
		// we write these blocks specifically to the free_list_blocks
		// a normal MetaBlockWriter will fetch blocks to use from the free_list
		// but since we are WRITING the free_list, this behavior is sub-optimal

		FreeListBlockWriter writer(*this, free_list_blocks);

		auto ptr = writer.GetBlockPointer();
		D_ASSERT(ptr.block_id == free_list_blocks[0]);
		header.free_list = ptr.block_id;
		for (auto &block_id : free_list_blocks) {
			modified_blocks.insert(block_id);
		}

		writer.Write<uint64_t>(free_list.size());
		for (auto &block_id : free_list) {
			writer.Write<block_id_t>(block_id);
		}
		writer.Write<uint64_t>(multi_use_blocks.size());
		for (auto &entry : multi_use_blocks) {
			writer.Write<block_id_t>(entry.first);
			writer.Write<uint32_t>(entry.second);
		}
		writer.Flush();
	} else {
		// no blocks in the free list
		header.free_list = INVALID_BLOCK;
	}
	header.block_count = max_block;

	auto &config = DBConfig::Get(db);
	if (config.options.checkpoint_abort == CheckpointAbort::DEBUG_ABORT_AFTER_FREE_LIST_WRITE) {
		throw FatalException("Checkpoint aborted after free list write because of PRAGMA checkpoint_abort flag");
	}

	if (!use_direct_io) {
		// if we are not using Direct IO we need to fsync BEFORE we write the header to ensure that all the previous
		// blocks are written as well
		handle->Sync();
	}
	// set the header inside the buffer
	header_buffer.Clear();
	Store<DatabaseHeader>(header, header_buffer.buffer);
	// now write the header to the file, active_header determines whether we write to h1 or h2
	// note that if active_header is h1 we write to h2, and vice versa
	ChecksumAndWrite(header_buffer, active_header == 1 ? Storage::FILE_HEADER_SIZE : Storage::FILE_HEADER_SIZE * 2);
	// switch active header to the other header
	active_header = 1 - active_header;
	//! Ensure the header write ends up on disk
	handle->Sync();
}

} // namespace duckdb











namespace duckdb {

BaseStatistics::BaseStatistics(LogicalType type, StatisticsType stats_type)
    : type(std::move(type)), stats_type(stats_type) {
}

BaseStatistics::~BaseStatistics() {
}

void BaseStatistics::InitializeBase() {
	validity_stats = make_unique<ValidityStatistics>(false);
	if (stats_type == GLOBAL_STATS) {
		distinct_stats = make_unique<DistinctStatistics>();
	}
}

bool BaseStatistics::CanHaveNull() const {
	if (!validity_stats) {
		// we don't know
		// solid maybe
		return true;
	}
	return ((ValidityStatistics &)*validity_stats).has_null;
}

bool BaseStatistics::CanHaveNoNull() const {
	if (!validity_stats) {
		// we don't know
		// solid maybe
		return true;
	}
	return ((ValidityStatistics &)*validity_stats).has_no_null;
}

void BaseStatistics::UpdateDistinctStatistics(Vector &v, idx_t count) {
	if (!distinct_stats) {
		return;
	}
	auto &d_stats = (DistinctStatistics &)*distinct_stats;
	d_stats.Update(v, count);
}

void MergeInternal(unique_ptr<BaseStatistics> &orig, const unique_ptr<BaseStatistics> &other) {
	if (other) {
		if (orig) {
			orig->Merge(*other);
		} else {
			orig = other->Copy();
		}
	}
}

void BaseStatistics::Merge(const BaseStatistics &other) {
	D_ASSERT(type == other.type);
	MergeInternal(validity_stats, other.validity_stats);
	if (stats_type == GLOBAL_STATS) {
		MergeInternal(distinct_stats, other.distinct_stats);
	}
}

idx_t BaseStatistics::GetDistinctCount() {
	if (distinct_stats) {
		auto &d_stats = (DistinctStatistics &)*distinct_stats;
		return d_stats.GetCount();
	}
	return 0;
}

unique_ptr<BaseStatistics> BaseStatistics::CreateEmpty(LogicalType type, StatisticsType stats_type) {
	unique_ptr<BaseStatistics> result;
	switch (type.InternalType()) {
	case PhysicalType::BIT:
		return make_unique<ValidityStatistics>(false, false);
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
	case PhysicalType::INT16:
	case PhysicalType::INT32:
	case PhysicalType::INT64:
	case PhysicalType::UINT8:
	case PhysicalType::UINT16:
	case PhysicalType::UINT32:
	case PhysicalType::UINT64:
	case PhysicalType::INT128:
	case PhysicalType::FLOAT:
	case PhysicalType::DOUBLE:
		result = make_unique<NumericStatistics>(std::move(type), stats_type);
		break;
	case PhysicalType::VARCHAR:
		result = make_unique<StringStatistics>(std::move(type), stats_type);
		break;
	case PhysicalType::STRUCT:
		result = make_unique<StructStatistics>(std::move(type));
		break;
	case PhysicalType::LIST:
		result = make_unique<ListStatistics>(std::move(type));
		break;
	case PhysicalType::INTERVAL:
	default:
		result = make_unique<BaseStatistics>(std::move(type), stats_type);
	}
	result->InitializeBase();
	return result;
}

unique_ptr<BaseStatistics> BaseStatistics::Copy() const {
	auto result = make_unique<BaseStatistics>(type, stats_type);
	result->CopyBase(*this);
	return result;
}

void BaseStatistics::CopyBase(const BaseStatistics &orig) {
	if (orig.validity_stats) {
		validity_stats = orig.validity_stats->Copy();
	}
	if (orig.distinct_stats) {
		distinct_stats = orig.distinct_stats->Copy();
	}
}

void BaseStatistics::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	ValidityStatistics(CanHaveNull(), CanHaveNoNull()).Serialize(writer);
	Serialize(writer);
	auto ptype = type.InternalType();
	if (ptype != PhysicalType::BIT) {
		writer.WriteField<StatisticsType>(stats_type);
		writer.WriteOptional<BaseStatistics>(distinct_stats);
	}
	writer.Finalize();
}

void BaseStatistics::Serialize(FieldWriter &writer) const {
}

unique_ptr<BaseStatistics> BaseStatistics::Deserialize(Deserializer &source, LogicalType type) {
	FieldReader reader(source);
	auto validity_stats = ValidityStatistics::Deserialize(reader);
	unique_ptr<BaseStatistics> result;
	auto ptype = type.InternalType();
	switch (ptype) {
	case PhysicalType::BIT:
		result = ValidityStatistics::Deserialize(reader);
		break;
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
	case PhysicalType::INT16:
	case PhysicalType::INT32:
	case PhysicalType::INT64:
	case PhysicalType::UINT8:
	case PhysicalType::UINT16:
	case PhysicalType::UINT32:
	case PhysicalType::UINT64:
	case PhysicalType::INT128:
	case PhysicalType::FLOAT:
	case PhysicalType::DOUBLE:
		result = NumericStatistics::Deserialize(reader, std::move(type));
		break;
	case PhysicalType::VARCHAR:
		result = StringStatistics::Deserialize(reader, std::move(type));
		break;
	case PhysicalType::STRUCT:
		result = StructStatistics::Deserialize(reader, std::move(type));
		break;
	case PhysicalType::LIST:
		result = ListStatistics::Deserialize(reader, std::move(type));
		break;
	case PhysicalType::INTERVAL:
		result = make_unique<BaseStatistics>(std::move(type), StatisticsType::LOCAL_STATS);
		break;
	default:
		throw InternalException("Unimplemented type for statistics deserialization");
	}

	if (ptype != PhysicalType::BIT) {
		result->validity_stats = std::move(validity_stats);
		result->stats_type = reader.ReadField<StatisticsType>(StatisticsType::LOCAL_STATS);
		result->distinct_stats = reader.ReadOptional<DistinctStatistics>(nullptr);
	}

	reader.Finalize();
	return result;
}

string BaseStatistics::ToString() const {
	return StringUtil::Format("%s%s", validity_stats ? validity_stats->ToString() : "",
	                          distinct_stats ? distinct_stats->ToString() : "");
}

void BaseStatistics::Verify(Vector &vector, const SelectionVector &sel, idx_t count) const {
	D_ASSERT(vector.GetType() == this->type);
	if (validity_stats) {
		validity_stats->Verify(vector, sel, count);
	}
}

void BaseStatistics::Verify(Vector &vector, idx_t count) const {
	auto sel = FlatVector::IncrementalSelectionVector();
	Verify(vector, *sel, count);
}

} // namespace duckdb


namespace duckdb {

ColumnStatistics::ColumnStatistics(unique_ptr<BaseStatistics> stats_p) : stats(std::move(stats_p)) {
}

shared_ptr<ColumnStatistics> ColumnStatistics::CreateEmptyStats(const LogicalType &type) {
	auto col_stats = BaseStatistics::CreateEmpty(type, StatisticsType::GLOBAL_STATS);
	return make_shared<ColumnStatistics>(std::move(col_stats));
}

} // namespace duckdb





#include <math.h>

namespace duckdb {

DistinctStatistics::DistinctStatistics()
    : BaseStatistics(LogicalType::INVALID, StatisticsType::LOCAL_STATS), log(make_unique<HyperLogLog>()),
      sample_count(0), total_count(0) {
}

DistinctStatistics::DistinctStatistics(unique_ptr<HyperLogLog> log, idx_t sample_count, idx_t total_count)
    : BaseStatistics(LogicalType::INVALID, StatisticsType::LOCAL_STATS), log(std::move(log)),
      sample_count(sample_count), total_count(total_count) {
}

unique_ptr<BaseStatistics> DistinctStatistics::Copy() const {
	return make_unique<DistinctStatistics>(log->Copy(), sample_count, total_count);
}

void DistinctStatistics::Merge(const BaseStatistics &other_p) {
	BaseStatistics::Merge(other_p);
	auto &other = (const DistinctStatistics &)other_p;
	log = log->Merge(*other.log);
	sample_count += other.sample_count;
	total_count += other.total_count;
}

void DistinctStatistics::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	Serialize(writer);
	writer.Finalize();
}

void DistinctStatistics::Serialize(FieldWriter &writer) const {
	writer.WriteField<idx_t>(sample_count);
	writer.WriteField<idx_t>(total_count);
	log->Serialize(writer);
}

unique_ptr<DistinctStatistics> DistinctStatistics::Deserialize(Deserializer &source) {
	FieldReader reader(source);
	auto result = Deserialize(reader);
	reader.Finalize();
	return result;
}

unique_ptr<DistinctStatistics> DistinctStatistics::Deserialize(FieldReader &reader) {
	auto sample_count = reader.ReadRequired<idx_t>();
	auto total_count = reader.ReadRequired<idx_t>();
	return make_unique<DistinctStatistics>(HyperLogLog::Deserialize(reader), sample_count, total_count);
}

void DistinctStatistics::Update(Vector &v, idx_t count, bool sample) {
	UnifiedVectorFormat vdata;
	v.ToUnifiedFormat(count, vdata);
	Update(vdata, v.GetType(), count, sample);
}

void DistinctStatistics::Update(UnifiedVectorFormat &vdata, const LogicalType &type, idx_t count, bool sample) {
	if (count == 0) {
		return;
	}

	total_count += count;
	if (sample) {
		count = MinValue<idx_t>(idx_t(SAMPLE_RATE * MaxValue<idx_t>(STANDARD_VECTOR_SIZE, count)), count);
	}
	sample_count += count;

	uint64_t indices[STANDARD_VECTOR_SIZE];
	uint8_t counts[STANDARD_VECTOR_SIZE];

	HyperLogLog::ProcessEntries(vdata, type, indices, counts, count);
	log->AddToLog(vdata, count, indices, counts);
}

string DistinctStatistics::ToString() const {
	return StringUtil::Format("[Approx Unique: %s]", to_string(GetCount()));
}

idx_t DistinctStatistics::GetCount() const {
	if (sample_count == 0 || total_count == 0) {
		return 0;
	}

	double u = MinValue<idx_t>(log->Count(), sample_count);
	double s = sample_count;
	double n = total_count;

	// Assume this proportion of the the sampled values occurred only once
	double u1 = pow(u / s, 2) * u;

	// Estimate total uniques using Good Turing Estimation
	idx_t estimate = u + u1 / s * (n - s);
	return MinValue<idx_t>(estimate, total_count);
}

} // namespace duckdb






namespace duckdb {

ListStatistics::ListStatistics(LogicalType type_p) : BaseStatistics(std::move(type_p), StatisticsType::LOCAL_STATS) {
	D_ASSERT(type.InternalType() == PhysicalType::LIST);
	InitializeBase();
	auto &child_type = ListType::GetChildType(type);
	child_stats = BaseStatistics::CreateEmpty(child_type, StatisticsType::LOCAL_STATS);
}

void ListStatistics::Merge(const BaseStatistics &other_p) {
	BaseStatistics::Merge(other_p);

	auto &other = (const ListStatistics &)other_p;
	if (child_stats && other.child_stats) {
		child_stats->Merge(*other.child_stats);
	} else {
		child_stats.reset();
	}
}

// LCOV_EXCL_START
FilterPropagateResult ListStatistics::CheckZonemap(ExpressionType comparison_type, const Value &constant) const {
	throw InternalException("List zonemaps are not supported yet");
}
// LCOV_EXCL_STOP

unique_ptr<BaseStatistics> ListStatistics::Copy() const {
	auto result = make_unique<ListStatistics>(type);
	result->CopyBase(*this);

	result->child_stats = child_stats ? child_stats->Copy() : nullptr;
	return std::move(result);
}

void ListStatistics::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*child_stats);
}

unique_ptr<BaseStatistics> ListStatistics::Deserialize(FieldReader &reader, LogicalType type) {
	D_ASSERT(type.InternalType() == PhysicalType::LIST);
	auto result = make_unique<ListStatistics>(std::move(type));
	auto &child_type = ListType::GetChildType(result->type);
	result->child_stats = reader.ReadRequiredSerializable<BaseStatistics>(child_type);
	return std::move(result);
}

string ListStatistics::ToString() const {
	return StringUtil::Format("[%s]%s", child_stats ? child_stats->ToString() : "No Stats", BaseStatistics::ToString());
}

void ListStatistics::Verify(Vector &vector, const SelectionVector &sel, idx_t count) const {
	BaseStatistics::Verify(vector, sel, count);

	if (child_stats) {
		auto &child_entry = ListVector::GetEntry(vector);
		UnifiedVectorFormat vdata;
		vector.ToUnifiedFormat(count, vdata);

		auto list_data = (list_entry_t *)vdata.data;
		idx_t total_list_count = 0;
		for (idx_t i = 0; i < count; i++) {
			auto idx = sel.get_index(i);
			auto index = vdata.sel->get_index(idx);
			auto list = list_data[index];
			if (vdata.validity.RowIsValid(index)) {
				for (idx_t list_idx = 0; list_idx < list.length; list_idx++) {
					total_list_count++;
				}
			}
		}
		SelectionVector list_sel(total_list_count);
		idx_t list_count = 0;
		for (idx_t i = 0; i < count; i++) {
			auto idx = sel.get_index(i);
			auto index = vdata.sel->get_index(idx);
			auto list = list_data[index];
			if (vdata.validity.RowIsValid(index)) {
				for (idx_t list_idx = 0; list_idx < list.length; list_idx++) {
					list_sel.set_index(list_count++, list.offset + list_idx);
				}
			}
		}

		child_stats->Verify(child_entry, list_sel, list_count);
	}
}

} // namespace duckdb






namespace duckdb {

template <>
void NumericStatistics::Update<interval_t>(SegmentStatistics &stats, interval_t new_value) {
}

template <>
void NumericStatistics::Update<list_entry_t>(SegmentStatistics &stats, list_entry_t new_value) {
}

NumericStatistics::NumericStatistics(LogicalType type_p, StatisticsType stats_type)
    : BaseStatistics(std::move(type_p), stats_type) {
	InitializeBase();
	min = Value::MaximumValue(type);
	max = Value::MinimumValue(type);
}

NumericStatistics::NumericStatistics(LogicalType type_p, Value min_p, Value max_p, StatisticsType stats_type)
    : BaseStatistics(std::move(type_p), stats_type), min(std::move(min_p)), max(std::move(max_p)) {
	InitializeBase();
}

void NumericStatistics::Merge(const BaseStatistics &other_p) {
	BaseStatistics::Merge(other_p);
	auto &other = (const NumericStatistics &)other_p;
	if (other.min.IsNull() || min.IsNull()) {
		min = Value(type);
	} else if (other.min < min) {
		min = other.min;
	}
	if (other.max.IsNull() || max.IsNull()) {
		max = Value(type);
	} else if (other.max > max) {
		max = other.max;
	}
}

FilterPropagateResult NumericStatistics::CheckZonemap(ExpressionType comparison_type, const Value &constant) const {
	if (constant.IsNull()) {
		return FilterPropagateResult::FILTER_ALWAYS_FALSE;
	}
	if (min.IsNull() || max.IsNull()) {
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	switch (comparison_type) {
	case ExpressionType::COMPARE_EQUAL:
		if (constant == min && constant == max) {
			return FilterPropagateResult::FILTER_ALWAYS_TRUE;
		} else if (constant >= min && constant <= max) {
			return FilterPropagateResult::NO_PRUNING_POSSIBLE;
		} else {
			return FilterPropagateResult::FILTER_ALWAYS_FALSE;
		}
	case ExpressionType::COMPARE_NOTEQUAL:
		if (constant < min || constant > max) {
			return FilterPropagateResult::FILTER_ALWAYS_TRUE;
		} else if (min == max && min == constant) {
			// corner case of a cluster with one numeric equal to the target constant
			return FilterPropagateResult::FILTER_ALWAYS_FALSE;
		}
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		// X >= C
		// this can be true only if max(X) >= C
		// if min(X) >= C, then this is always true
		if (min >= constant) {
			return FilterPropagateResult::FILTER_ALWAYS_TRUE;
		} else if (max >= constant) {
			return FilterPropagateResult::NO_PRUNING_POSSIBLE;
		} else {
			return FilterPropagateResult::FILTER_ALWAYS_FALSE;
		}
	case ExpressionType::COMPARE_GREATERTHAN:
		// X > C
		// this can be true only if max(X) > C
		// if min(X) > C, then this is always true
		if (min > constant) {
			return FilterPropagateResult::FILTER_ALWAYS_TRUE;
		} else if (max > constant) {
			return FilterPropagateResult::NO_PRUNING_POSSIBLE;
		} else {
			return FilterPropagateResult::FILTER_ALWAYS_FALSE;
		}
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		// X <= C
		// this can be true only if min(X) <= C
		// if max(X) <= C, then this is always true
		if (max <= constant) {
			return FilterPropagateResult::FILTER_ALWAYS_TRUE;
		} else if (min <= constant) {
			return FilterPropagateResult::NO_PRUNING_POSSIBLE;
		} else {
			return FilterPropagateResult::FILTER_ALWAYS_FALSE;
		}
	case ExpressionType::COMPARE_LESSTHAN:
		// X < C
		// this can be true only if min(X) < C
		// if max(X) < C, then this is always true
		if (max < constant) {
			return FilterPropagateResult::FILTER_ALWAYS_TRUE;
		} else if (min < constant) {
			return FilterPropagateResult::NO_PRUNING_POSSIBLE;
		} else {
			return FilterPropagateResult::FILTER_ALWAYS_FALSE;
		}
	default:
		throw InternalException("Expression type in zonemap check not implemented");
	}
}

unique_ptr<BaseStatistics> NumericStatistics::Copy() const {
	auto result = make_unique<NumericStatistics>(type, min, max, stats_type);
	result->CopyBase(*this);
	return std::move(result);
}

bool NumericStatistics::IsConstant() const {
	return max <= min;
}

void SerializeNumericStatsValue(const Value &val, FieldWriter &writer) {
	writer.WriteField<bool>(val.IsNull());
	if (val.IsNull()) {
		return;
	}
	switch (val.type().InternalType()) {
	case PhysicalType::BOOL:
		writer.WriteField<bool>(BooleanValue::Get(val));
		break;
	case PhysicalType::INT8:
		writer.WriteField<int8_t>(TinyIntValue::Get(val));
		break;
	case PhysicalType::INT16:
		writer.WriteField<int16_t>(SmallIntValue::Get(val));
		break;
	case PhysicalType::INT32:
		writer.WriteField<int32_t>(IntegerValue::Get(val));
		break;
	case PhysicalType::INT64:
		writer.WriteField<int64_t>(BigIntValue::Get(val));
		break;
	case PhysicalType::UINT8:
		writer.WriteField<int8_t>(UTinyIntValue::Get(val));
		break;
	case PhysicalType::UINT16:
		writer.WriteField<int16_t>(USmallIntValue::Get(val));
		break;
	case PhysicalType::UINT32:
		writer.WriteField<int32_t>(UIntegerValue::Get(val));
		break;
	case PhysicalType::UINT64:
		writer.WriteField<int64_t>(UBigIntValue::Get(val));
		break;
	case PhysicalType::INT128:
		writer.WriteField<hugeint_t>(HugeIntValue::Get(val));
		break;
	case PhysicalType::FLOAT:
		writer.WriteField<float>(FloatValue::Get(val));
		break;
	case PhysicalType::DOUBLE:
		writer.WriteField<double>(DoubleValue::Get(val));
		break;
	default:
		throw InternalException("Unsupported type for serializing numeric statistics");
	}
}

void NumericStatistics::Serialize(FieldWriter &writer) const {
	SerializeNumericStatsValue(min, writer);
	SerializeNumericStatsValue(max, writer);
}

Value DeserializeNumericStatsValue(const LogicalType &type, FieldReader &reader) {
	auto is_null = reader.ReadRequired<bool>();
	if (is_null) {
		return Value(type);
	}
	Value result;
	switch (type.InternalType()) {
	case PhysicalType::BOOL:
		result = Value::BOOLEAN(reader.ReadRequired<bool>());
		break;
	case PhysicalType::INT8:
		result = Value::TINYINT(reader.ReadRequired<int8_t>());
		break;
	case PhysicalType::INT16:
		result = Value::SMALLINT(reader.ReadRequired<int16_t>());
		break;
	case PhysicalType::INT32:
		result = Value::INTEGER(reader.ReadRequired<int32_t>());
		break;
	case PhysicalType::INT64:
		result = Value::BIGINT(reader.ReadRequired<int64_t>());
		break;
	case PhysicalType::UINT8:
		result = Value::UTINYINT(reader.ReadRequired<uint8_t>());
		break;
	case PhysicalType::UINT16:
		result = Value::USMALLINT(reader.ReadRequired<uint16_t>());
		break;
	case PhysicalType::UINT32:
		result = Value::UINTEGER(reader.ReadRequired<uint32_t>());
		break;
	case PhysicalType::UINT64:
		result = Value::UBIGINT(reader.ReadRequired<uint64_t>());
		break;
	case PhysicalType::INT128:
		result = Value::HUGEINT(reader.ReadRequired<hugeint_t>());
		break;
	case PhysicalType::FLOAT:
		result = Value::FLOAT(reader.ReadRequired<float>());
		break;
	case PhysicalType::DOUBLE:
		result = Value::DOUBLE(reader.ReadRequired<double>());
		break;
	default:
		throw InternalException("Unsupported type for deserializing numeric statistics");
	}
	result.Reinterpret(type);
	return result;
}

unique_ptr<BaseStatistics> NumericStatistics::Deserialize(FieldReader &reader, LogicalType type) {
	auto min = DeserializeNumericStatsValue(type, reader);
	auto max = DeserializeNumericStatsValue(type, reader);
	return make_unique_base<BaseStatistics, NumericStatistics>(std::move(type), std::move(min), std::move(max),
	                                                           StatisticsType::LOCAL_STATS);
}

string NumericStatistics::ToString() const {
	return StringUtil::Format("[Min: %s, Max: %s]%s", min.ToString(), max.ToString(), BaseStatistics::ToString());
}

template <class T>
void NumericStatistics::TemplatedVerify(Vector &vector, const SelectionVector &sel, idx_t count) const {
	UnifiedVectorFormat vdata;
	vector.ToUnifiedFormat(count, vdata);

	auto data = (T *)vdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto idx = sel.get_index(i);
		auto index = vdata.sel->get_index(idx);
		if (!vdata.validity.RowIsValid(index)) {
			continue;
		}
		if (!min.IsNull() && LessThan::Operation(data[index], min.GetValueUnsafe<T>())) { // LCOV_EXCL_START
			throw InternalException("Statistics mismatch: value is smaller than min.\nStatistics: %s\nVector: %s",
			                        ToString(), vector.ToString(count));
		} // LCOV_EXCL_STOP
		if (!max.IsNull() && GreaterThan::Operation(data[index], max.GetValueUnsafe<T>())) {
			throw InternalException("Statistics mismatch: value is bigger than max.\nStatistics: %s\nVector: %s",
			                        ToString(), vector.ToString(count));
		}
	}
}

void NumericStatistics::Verify(Vector &vector, const SelectionVector &sel, idx_t count) const {
	BaseStatistics::Verify(vector, sel, count);

	switch (type.InternalType()) {
	case PhysicalType::BOOL:
		break;
	case PhysicalType::INT8:
		TemplatedVerify<int8_t>(vector, sel, count);
		break;
	case PhysicalType::INT16:
		TemplatedVerify<int16_t>(vector, sel, count);
		break;
	case PhysicalType::INT32:
		TemplatedVerify<int32_t>(vector, sel, count);
		break;
	case PhysicalType::INT64:
		TemplatedVerify<int64_t>(vector, sel, count);
		break;
	case PhysicalType::UINT8:
		TemplatedVerify<uint8_t>(vector, sel, count);
		break;
	case PhysicalType::UINT16:
		TemplatedVerify<uint16_t>(vector, sel, count);
		break;
	case PhysicalType::UINT32:
		TemplatedVerify<uint32_t>(vector, sel, count);
		break;
	case PhysicalType::UINT64:
		TemplatedVerify<uint64_t>(vector, sel, count);
		break;
	case PhysicalType::INT128:
		TemplatedVerify<hugeint_t>(vector, sel, count);
		break;
	case PhysicalType::FLOAT:
		TemplatedVerify<float>(vector, sel, count);
		break;
	case PhysicalType::DOUBLE:
		TemplatedVerify<double>(vector, sel, count);
		break;
	default:
		throw InternalException("Unsupported type %s for numeric statistics verify", type.ToString());
	}
}

} // namespace duckdb




namespace duckdb {

SegmentStatistics::SegmentStatistics(LogicalType type) : type(std::move(type)) {
	Reset();
}

SegmentStatistics::SegmentStatistics(LogicalType type, unique_ptr<BaseStatistics> stats)
    : type(std::move(type)), statistics(std::move(stats)) {
	if (!statistics) {
		Reset();
	}
}

void SegmentStatistics::Reset() {
	statistics = BaseStatistics::CreateEmpty(type, StatisticsType::LOCAL_STATS);
}

} // namespace duckdb







namespace duckdb {

StringStatistics::StringStatistics(LogicalType type_p, StatisticsType stats_type)
    : BaseStatistics(std::move(type_p), stats_type) {
	InitializeBase();
	for (idx_t i = 0; i < MAX_STRING_MINMAX_SIZE; i++) {
		min[i] = 0xFF;
		max[i] = 0;
	}
	max_string_length = 0;
	has_unicode = false;
	has_overflow_strings = false;
}

unique_ptr<BaseStatistics> StringStatistics::Copy() const {
	auto result = make_unique<StringStatistics>(type, stats_type);
	result->CopyBase(*this);

	memcpy(result->min, min, MAX_STRING_MINMAX_SIZE);
	memcpy(result->max, max, MAX_STRING_MINMAX_SIZE);
	result->has_unicode = has_unicode;
	result->max_string_length = max_string_length;
	return std::move(result);
}

void StringStatistics::Serialize(FieldWriter &writer) const {
	writer.WriteBlob(min, MAX_STRING_MINMAX_SIZE);
	writer.WriteBlob(max, MAX_STRING_MINMAX_SIZE);
	writer.WriteField<bool>(has_unicode);
	writer.WriteField<uint32_t>(max_string_length);
	writer.WriteField<bool>(has_overflow_strings);
}

unique_ptr<BaseStatistics> StringStatistics::Deserialize(FieldReader &reader, LogicalType type) {
	auto stats = make_unique<StringStatistics>(std::move(type), StatisticsType::LOCAL_STATS);
	reader.ReadBlob(stats->min, MAX_STRING_MINMAX_SIZE);
	reader.ReadBlob(stats->max, MAX_STRING_MINMAX_SIZE);
	stats->has_unicode = reader.ReadRequired<bool>();
	stats->max_string_length = reader.ReadRequired<uint32_t>();
	stats->has_overflow_strings = reader.ReadRequired<bool>();
	return std::move(stats);
}

static int StringValueComparison(const_data_ptr_t data, idx_t len, const_data_ptr_t comparison) {
	D_ASSERT(len <= StringStatistics::MAX_STRING_MINMAX_SIZE);
	for (idx_t i = 0; i < len; i++) {
		if (data[i] < comparison[i]) {
			return -1;
		} else if (data[i] > comparison[i]) {
			return 1;
		}
	}
	return 0;
}

static void ConstructValue(const_data_ptr_t data, idx_t size, data_t target[]) {
	idx_t value_size =
	    size > StringStatistics::MAX_STRING_MINMAX_SIZE ? StringStatistics::MAX_STRING_MINMAX_SIZE : size;
	memcpy(target, data, value_size);
	for (idx_t i = value_size; i < StringStatistics::MAX_STRING_MINMAX_SIZE; i++) {
		target[i] = '\0';
	}
}

void StringStatistics::Update(const string_t &value) {
	auto data = (const_data_ptr_t)value.GetDataUnsafe();
	auto size = value.GetSize();

	//! we can only fit 8 bytes, so we might need to trim our string
	// construct the value
	data_t target[MAX_STRING_MINMAX_SIZE];
	ConstructValue(data, size, target);

	// update the min and max
	if (StringValueComparison(target, MAX_STRING_MINMAX_SIZE, min) < 0) {
		memcpy(min, target, MAX_STRING_MINMAX_SIZE);
	}
	if (StringValueComparison(target, MAX_STRING_MINMAX_SIZE, max) > 0) {
		memcpy(max, target, MAX_STRING_MINMAX_SIZE);
	}
	if (size > max_string_length) {
		max_string_length = size;
	}
	if (type.id() == LogicalTypeId::VARCHAR && !has_unicode) {
		auto unicode = Utf8Proc::Analyze((const char *)data, size);
		if (unicode == UnicodeType::UNICODE) {
			has_unicode = true;
		} else if (unicode == UnicodeType::INVALID) {
			throw InternalException(
			    ErrorManager::InvalidUnicodeError(string((char *)data, size), "segment statistics update"));
		}
	}
}

void StringStatistics::Merge(const BaseStatistics &other_p) {
	BaseStatistics::Merge(other_p);
	auto &other = (const StringStatistics &)other_p;
	if (StringValueComparison(other.min, MAX_STRING_MINMAX_SIZE, min) < 0) {
		memcpy(min, other.min, MAX_STRING_MINMAX_SIZE);
	}
	if (StringValueComparison(other.max, MAX_STRING_MINMAX_SIZE, max) > 0) {
		memcpy(max, other.max, MAX_STRING_MINMAX_SIZE);
	}
	has_unicode = has_unicode || other.has_unicode;
	max_string_length = MaxValue<uint32_t>(max_string_length, other.max_string_length);
	has_overflow_strings = has_overflow_strings || other.has_overflow_strings;
}

FilterPropagateResult StringStatistics::CheckZonemap(ExpressionType comparison_type, const string &constant) const {
	auto data = (const_data_ptr_t)constant.c_str();
	auto size = constant.size();

	idx_t value_size = size > MAX_STRING_MINMAX_SIZE ? MAX_STRING_MINMAX_SIZE : size;
	int min_comp = StringValueComparison(data, value_size, min);
	int max_comp = StringValueComparison(data, value_size, max);
	switch (comparison_type) {
	case ExpressionType::COMPARE_EQUAL:
		if (min_comp >= 0 && max_comp <= 0) {
			return FilterPropagateResult::NO_PRUNING_POSSIBLE;
		} else {
			return FilterPropagateResult::FILTER_ALWAYS_FALSE;
		}
	case ExpressionType::COMPARE_NOTEQUAL:
		if (min_comp < 0 || max_comp > 0) {
			return FilterPropagateResult::FILTER_ALWAYS_TRUE;
		}
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
	case ExpressionType::COMPARE_GREATERTHAN:
		if (max_comp <= 0) {
			return FilterPropagateResult::NO_PRUNING_POSSIBLE;
		} else {
			return FilterPropagateResult::FILTER_ALWAYS_FALSE;
		}
	case ExpressionType::COMPARE_LESSTHAN:
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		if (min_comp >= 0) {
			return FilterPropagateResult::NO_PRUNING_POSSIBLE;
		} else {
			return FilterPropagateResult::FILTER_ALWAYS_FALSE;
		}
	default:
		throw InternalException("Expression type not implemented for string statistics zone map");
	}
}

static idx_t GetValidMinMaxSubstring(const_data_ptr_t data) {
	for (idx_t i = 0; i < StringStatistics::MAX_STRING_MINMAX_SIZE; i++) {
		if (data[i] == '\0') {
			return i;
		}
		if ((data[i] & 0x80) != 0) {
			return i;
		}
	}
	return StringStatistics::MAX_STRING_MINMAX_SIZE;
}

string StringStatistics::ToString() const {
	idx_t min_len = GetValidMinMaxSubstring(min);
	idx_t max_len = GetValidMinMaxSubstring(max);
	return StringUtil::Format("[Min: %s, Max: %s, Has Unicode: %s, Max String Length: %lld]%s",
	                          string((const char *)min, min_len), string((const char *)max, max_len),
	                          has_unicode ? "true" : "false", max_string_length, BaseStatistics::ToString());
}

void StringStatistics::Verify(Vector &vector, const SelectionVector &sel, idx_t count) const {
	BaseStatistics::Verify(vector, sel, count);

	string_t min_string((const char *)min, MAX_STRING_MINMAX_SIZE);
	string_t max_string((const char *)max, MAX_STRING_MINMAX_SIZE);

	UnifiedVectorFormat vdata;
	vector.ToUnifiedFormat(count, vdata);
	auto data = (string_t *)vdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto idx = sel.get_index(i);
		auto index = vdata.sel->get_index(idx);
		if (!vdata.validity.RowIsValid(index)) {
			continue;
		}
		auto value = data[index];
		auto data = value.GetDataUnsafe();
		auto len = value.GetSize();
		// LCOV_EXCL_START
		if (len > max_string_length) {
			throw InternalException(
			    "Statistics mismatch: string value exceeds maximum string length.\nStatistics: %s\nVector: %s",
			    ToString(), vector.ToString(count));
		}
		if (type.id() == LogicalTypeId::VARCHAR && !has_unicode) {
			auto unicode = Utf8Proc::Analyze(data, len);
			if (unicode == UnicodeType::UNICODE) {
				throw InternalException("Statistics mismatch: string value contains unicode, but statistics says it "
				                        "shouldn't.\nStatistics: %s\nVector: %s",
				                        ToString(), vector.ToString(count));
			} else if (unicode == UnicodeType::INVALID) {
				throw InternalException("Invalid unicode detected in vector: %s", vector.ToString(count));
			}
		}
		if (StringValueComparison((const_data_ptr_t)data, MinValue<idx_t>(len, MAX_STRING_MINMAX_SIZE), min) < 0) {
			throw InternalException("Statistics mismatch: value is smaller than min.\nStatistics: %s\nVector: %s",
			                        ToString(), vector.ToString(count));
		}
		if (StringValueComparison((const_data_ptr_t)data, MinValue<idx_t>(len, MAX_STRING_MINMAX_SIZE), max) > 0) {
			throw InternalException("Statistics mismatch: value is bigger than max.\nStatistics: %s\nVector: %s",
			                        ToString(), vector.ToString(count));
		}
		// LCOV_EXCL_STOP
	}
}

} // namespace duckdb





namespace duckdb {

StructStatistics::StructStatistics(LogicalType type_p)
    : BaseStatistics(std::move(type_p), StatisticsType::LOCAL_STATS) {
	D_ASSERT(type.InternalType() == PhysicalType::STRUCT);
	InitializeBase();

	auto &child_types = StructType::GetChildTypes(type);
	child_stats.resize(child_types.size());
	for (idx_t i = 0; i < child_types.size(); i++) {
		child_stats[i] = BaseStatistics::CreateEmpty(child_types[i].second, StatisticsType::LOCAL_STATS);
	}
}

void StructStatistics::Merge(const BaseStatistics &other_p) {
	BaseStatistics::Merge(other_p);

	auto &other = (const StructStatistics &)other_p;
	D_ASSERT(other.child_stats.size() == child_stats.size());
	for (idx_t i = 0; i < child_stats.size(); i++) {
		if (child_stats[i] && other.child_stats[i]) {
			child_stats[i]->Merge(*other.child_stats[i]);
		} else {
			child_stats[i].reset();
		}
	}
}

// LCOV_EXCL_START
FilterPropagateResult StructStatistics::CheckZonemap(ExpressionType comparison_type, const Value &constant) const {
	throw InternalException("Struct zonemaps are not supported yet");
}
// LCOV_EXCL_STOP

unique_ptr<BaseStatistics> StructStatistics::Copy() const {
	auto result = make_unique<StructStatistics>(type);
	result->CopyBase(*this);

	for (idx_t i = 0; i < child_stats.size(); i++) {
		result->child_stats[i] = child_stats[i] ? child_stats[i]->Copy() : nullptr;
	}
	return std::move(result);
}

void StructStatistics::Serialize(FieldWriter &writer) const {
	writer.WriteField<uint32_t>(child_stats.size());
	auto &serializer = writer.GetSerializer();
	for (idx_t i = 0; i < child_stats.size(); i++) {
		serializer.Write<bool>(child_stats[i] ? true : false);
		if (child_stats[i]) {
			child_stats[i]->Serialize(serializer);
		}
	}
}

unique_ptr<BaseStatistics> StructStatistics::Deserialize(FieldReader &reader, LogicalType type) {
	D_ASSERT(type.InternalType() == PhysicalType::STRUCT);
	auto result = make_unique<StructStatistics>(std::move(type));
	auto &child_types = StructType::GetChildTypes(result->type);

	auto child_type_count = reader.ReadRequired<uint32_t>();
	if (child_types.size() != child_type_count) {
		throw InternalException("Struct stats deserialization failure: child count does not match type count!");
	}
	auto &source = reader.GetSource();
	for (idx_t i = 0; i < child_types.size(); i++) {
		auto has_child = source.Read<bool>();
		if (has_child) {
			result->child_stats[i] = BaseStatistics::Deserialize(source, child_types[i].second);
		} else {
			result->child_stats[i].reset();
		}
	}
	return std::move(result);
}

string StructStatistics::ToString() const {
	string result;
	result += " {";
	auto &child_types = StructType::GetChildTypes(type);
	for (idx_t i = 0; i < child_types.size(); i++) {
		if (i > 0) {
			result += ", ";
		}
		result += child_types[i].first + ": " + (child_stats[i] ? child_stats[i]->ToString() : "No Stats");
	}
	result += "}";
	result += BaseStatistics::ToString();
	return result;
}

void StructStatistics::Verify(Vector &vector, const SelectionVector &sel, idx_t count) const {
	BaseStatistics::Verify(vector, sel, count);

	auto &child_entries = StructVector::GetEntries(vector);
	for (idx_t i = 0; i < child_entries.size(); i++) {
		if (child_stats[i]) {
			child_stats[i]->Verify(*child_entries[i], sel, count);
		}
	}
}

} // namespace duckdb







namespace duckdb {

ValidityStatistics::ValidityStatistics(bool has_null, bool has_no_null)
    : BaseStatistics(LogicalType(LogicalTypeId::VALIDITY), StatisticsType::LOCAL_STATS), has_null(has_null),
      has_no_null(has_no_null) {
}

unique_ptr<BaseStatistics> ValidityStatistics::Combine(const unique_ptr<BaseStatistics> &lstats,
                                                       const unique_ptr<BaseStatistics> &rstats) {
	if (!lstats && !rstats) {
		return nullptr;
	} else if (!lstats) {
		return rstats->Copy();
	} else if (!rstats) {
		return lstats->Copy();
	} else {
		auto &l = (ValidityStatistics &)*lstats;
		auto &r = (ValidityStatistics &)*rstats;
		return make_unique<ValidityStatistics>(l.has_null || r.has_null, l.has_no_null || r.has_no_null);
	}
}

bool ValidityStatistics::IsConstant() const {
	if (!has_null) {
		return true;
	}
	if (!has_no_null) {
		return true;
	}
	return false;
}

void ValidityStatistics::Merge(const BaseStatistics &other_p) {
	auto &other = (ValidityStatistics &)other_p;
	has_null = has_null || other.has_null;
	has_no_null = has_no_null || other.has_no_null;
}

unique_ptr<BaseStatistics> ValidityStatistics::Copy() const {
	return make_unique<ValidityStatistics>(has_null, has_no_null);
}

void ValidityStatistics::Serialize(FieldWriter &writer) const {
	writer.WriteField<bool>(has_null);
	writer.WriteField<bool>(has_no_null);
}

unique_ptr<ValidityStatistics> ValidityStatistics::Deserialize(FieldReader &reader) {
	bool has_null = reader.ReadRequired<bool>();
	bool has_no_null = reader.ReadRequired<bool>();
	return make_unique<ValidityStatistics>(has_null, has_no_null);
}

void ValidityStatistics::Verify(Vector &vector, const SelectionVector &sel, idx_t count) const {
	if (has_null && has_no_null) {
		// nothing to verify
		return;
	}
	UnifiedVectorFormat vdata;
	vector.ToUnifiedFormat(count, vdata);
	for (idx_t i = 0; i < count; i++) {
		auto idx = sel.get_index(i);
		auto index = vdata.sel->get_index(idx);
		bool row_is_valid = vdata.validity.RowIsValid(index);
		if (row_is_valid && !has_no_null) {
			throw InternalException(
			    "Statistics mismatch: vector labeled as having only NULL values, but vector contains valid values: %s",
			    vector.ToString(count));
		}
		if (!row_is_valid && !has_null) {
			throw InternalException(
			    "Statistics mismatch: vector labeled as not having NULL values, but vector contains null values: %s",
			    vector.ToString(count));
		}
	}
}

string ValidityStatistics::ToString() const {
	auto has_n = has_null ? "true" : "false";
	auto has_n_n = has_no_null ? "true" : "false";
	return StringUtil::Format("[Has Null: %s, Has No Null: %s]", has_n, has_n_n);
}

} // namespace duckdb


namespace duckdb {

const uint64_t VERSION_NUMBER = 43;

struct StorageVersionInfo {
	const char *version_name;
	idx_t storage_version;
};

static StorageVersionInfo storage_version_info[] = {{"v0.6.0 or v0.6.1", 39},
                                                    {"v0.5.0 or v0.5.1", 38},
                                                    {"v0.3.3, v0.3.4 or v0.4.0", 33},
                                                    {"v0.3.2", 31},
                                                    {"v0.3.1", 27},
                                                    {"v0.3.0", 25},
                                                    {"v0.2.9", 21},
                                                    {"v0.2.8", 18},
                                                    {"v0.2.7", 17},
                                                    {"v0.2.6", 15},
                                                    {"v0.2.5", 13},
                                                    {"v0.2.4", 11},
                                                    {"v0.2.3", 6},
                                                    {"v0.2.2", 4},
                                                    {"v0.2.1 and prior", 1},
                                                    {nullptr, 0}};

const char *GetDuckDBVersion(idx_t version_number) {
	for (idx_t i = 0; storage_version_info[i].version_name; i++) {
		if (version_number == storage_version_info[i].storage_version) {
			return storage_version_info[i].version_name;
		}
	}
	return nullptr;
}

} // namespace duckdb




namespace duckdb {

StorageLockKey::StorageLockKey(StorageLock &lock, StorageLockType type) : lock(lock), type(type) {
}

StorageLockKey::~StorageLockKey() {
	if (type == StorageLockType::EXCLUSIVE) {
		lock.ReleaseExclusiveLock();
	} else {
		D_ASSERT(type == StorageLockType::SHARED);
		lock.ReleaseSharedLock();
	}
}

StorageLock::StorageLock() : read_count(0) {
}

unique_ptr<StorageLockKey> StorageLock::GetExclusiveLock() {
	exclusive_lock.lock();
	while (read_count != 0) {
	}
	return make_unique<StorageLockKey>(*this, StorageLockType::EXCLUSIVE);
}

unique_ptr<StorageLockKey> StorageLock::GetSharedLock() {
	exclusive_lock.lock();
	read_count++;
	exclusive_lock.unlock();
	return make_unique<StorageLockKey>(*this, StorageLockType::SHARED);
}

void StorageLock::ReleaseExclusiveLock() {
	exclusive_lock.unlock();
}

void StorageLock::ReleaseSharedLock() {
	read_count--;
}

} // namespace duckdb















namespace duckdb {

StorageManager::StorageManager(AttachedDatabase &db, string path_p, bool read_only)
    : db(db), path(std::move(path_p)), read_only(read_only) {
	if (path.empty()) {
		path = ":memory:";
	}
}

StorageManager::~StorageManager() {
}

StorageManager &StorageManager::Get(AttachedDatabase &db) {
	return db.GetStorageManager();
}
StorageManager &StorageManager::Get(Catalog &catalog) {
	return StorageManager::Get(catalog.GetAttached());
}

DatabaseInstance &StorageManager::GetDatabase() {
	return db.GetDatabase();
}

BufferManager &BufferManager::GetBufferManager(ClientContext &context) {
	return BufferManager::GetBufferManager(*context.db);
}

ObjectCache &ObjectCache::GetObjectCache(ClientContext &context) {
	return context.db->GetObjectCache();
}

bool ObjectCache::ObjectCacheEnabled(ClientContext &context) {
	return context.db->config.options.object_cache_enable;
}

bool StorageManager::InMemory() {
	D_ASSERT(!path.empty());
	return path == ":memory:";
}

void StorageManager::Initialize() {
	bool in_memory = InMemory();
	if (in_memory && read_only) {
		throw CatalogException("Cannot launch in-memory database in read-only mode!");
	}

	// create or load the database from disk, if not in-memory mode
	LoadDatabase();
}

///////////////////////////////////////////////////////////////////////////
class SingleFileTableIOManager : public TableIOManager {
public:
	explicit SingleFileTableIOManager(BlockManager &block_manager) : block_manager(block_manager) {
	}

	BlockManager &block_manager;

public:
	BlockManager &GetIndexBlockManager() override {
		return block_manager;
	}
	BlockManager &GetBlockManagerForRowData() override {
		return block_manager;
	}
};

SingleFileStorageManager::SingleFileStorageManager(AttachedDatabase &db, string path, bool read_only)
    : StorageManager(db, std::move(path), read_only) {
}

void SingleFileStorageManager::LoadDatabase() {
	if (InMemory()) {
		block_manager = make_unique<InMemoryBlockManager>(BufferManager::GetBufferManager(db));
		table_io_manager = make_unique<SingleFileTableIOManager>(*block_manager);
		return;
	}

	string wal_path = path + ".wal";
	auto &fs = FileSystem::Get(db);
	auto &config = DBConfig::Get(db);
	bool truncate_wal = false;
	// first check if the database exists
	if (!fs.FileExists(path)) {
		if (read_only) {
			throw CatalogException("Cannot open database \"%s\" in read-only mode: database does not exist", path);
		}
		// check if the WAL exists
		if (fs.FileExists(wal_path)) {
			// WAL file exists but database file does not
			// remove the WAL
			fs.RemoveFile(wal_path);
		}
		// initialize the block manager while creating a new db file
		auto sf_block_manager = make_unique<SingleFileBlockManager>(db, path, read_only, config.options.use_direct_io);
		sf_block_manager->CreateNewDatabase();
		block_manager = std::move(sf_block_manager);
		table_io_manager = make_unique<SingleFileTableIOManager>(*block_manager);
	} else {
		// initialize the block manager while loading the current db file
		auto sf_block_manager = make_unique<SingleFileBlockManager>(db, path, read_only, config.options.use_direct_io);
		sf_block_manager->LoadExistingDatabase();
		block_manager = std::move(sf_block_manager);
		table_io_manager = make_unique<SingleFileTableIOManager>(*block_manager);

		//! Load from storage
		auto checkpointer = SingleFileCheckpointReader(*this);
		checkpointer.LoadFromStorage();
		// finish load checkpoint, clear the cached handles of meta blocks
		block_manager->ClearMetaBlockHandles();
		// check if the WAL file exists
		if (fs.FileExists(wal_path)) {
			// replay the WAL
			truncate_wal = WriteAheadLog::Replay(db, wal_path);
		}
	}
	// initialize the WAL file
	if (!read_only) {
		wal = make_unique<WriteAheadLog>(db, wal_path);
		if (truncate_wal) {
			wal->Truncate(0);
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

class SingleFileStorageCommitState : public StorageCommitState {
	idx_t initial_wal_size = 0;
	idx_t initial_written = 0;
	WriteAheadLog *log;
	bool checkpoint;

public:
	SingleFileStorageCommitState(StorageManager &storage_manager, bool checkpoint);
	~SingleFileStorageCommitState() override;

	// Make the commit persistent
	void FlushCommit() override;
};

SingleFileStorageCommitState::SingleFileStorageCommitState(StorageManager &storage_manager, bool checkpoint)
    : checkpoint(checkpoint) {
	log = storage_manager.GetWriteAheadLog();
	if (log) {
		auto initial_size = log->GetWALSize();
		initial_written = log->GetTotalWritten();
		initial_wal_size = initial_size < 0 ? 0 : idx_t(initial_size);

		if (checkpoint) {
			// check if we are checkpointing after this commit
			// if we are checkpointing, we don't need to write anything to the WAL
			// this saves us a lot of unnecessary writes to disk in the case of large commits
			log->skip_writing = true;
		}
	} else {
		D_ASSERT(!checkpoint);
	}
}

// Make the commit persistent
void SingleFileStorageCommitState::FlushCommit() {
	if (log) {
		// flush the WAL if any changes were made
		if (log->GetTotalWritten() > initial_written) {
			(void)checkpoint;
			D_ASSERT(!checkpoint);
			D_ASSERT(!log->skip_writing);
			log->Flush();
		}
		log->skip_writing = false;
	}
	// Null so that the destructor will not truncate the log.
	log = nullptr;
}

SingleFileStorageCommitState::~SingleFileStorageCommitState() {
	// If log is non-null, then commit threw an exception before flushing.
	if (log) {
		log->skip_writing = false;
		if (log->GetTotalWritten() > initial_written) {
			// remove any entries written into the WAL by truncating it
			log->Truncate(initial_wal_size);
		}
	}
}

unique_ptr<StorageCommitState> SingleFileStorageManager::GenStorageCommitState(Transaction &transaction,
                                                                               bool checkpoint) {
	return make_unique<SingleFileStorageCommitState>(*this, checkpoint);
}

bool SingleFileStorageManager::IsCheckpointClean(block_id_t checkpoint_id) {
	return block_manager->IsRootBlock(checkpoint_id);
}

void SingleFileStorageManager::CreateCheckpoint(bool delete_wal, bool force_checkpoint) {
	if (InMemory() || read_only || !wal) {
		return;
	}
	auto &config = DBConfig::Get(db);
	if (wal->GetWALSize() > 0 || config.options.force_checkpoint || force_checkpoint) {
		// we only need to checkpoint if there is anything in the WAL
		SingleFileCheckpointWriter checkpointer(db, *block_manager);
		checkpointer.CreateCheckpoint();
	}
	if (delete_wal) {
		wal->Delete();
		wal.reset();
	}
}

DatabaseSize SingleFileStorageManager::GetDatabaseSize() {
	// All members default to zero
	DatabaseSize ds;
	if (!InMemory()) {
		ds.total_blocks = block_manager->TotalBlocks();
		ds.block_size = Storage::BLOCK_ALLOC_SIZE;
		ds.free_blocks = block_manager->FreeBlocks();
		ds.used_blocks = ds.total_blocks - ds.free_blocks;
		ds.bytes = (ds.total_blocks * ds.block_size);
		if (auto wal = GetWriteAheadLog()) {
			ds.wal_size = wal->GetWALSize();
		}
	}
	return ds;
}

bool SingleFileStorageManager::AutomaticCheckpoint(idx_t estimated_wal_bytes) {
	auto log = GetWriteAheadLog();
	if (!log) {
		return false;
	}

	auto &config = DBConfig::Get(db);
	auto initial_size = log->GetWALSize();
	idx_t expected_wal_size = initial_size + estimated_wal_bytes;
	return expected_wal_size > config.options.checkpoint_wal_size;
}

shared_ptr<TableIOManager> SingleFileStorageManager::GetTableIOManager(BoundCreateTableInfo *info /*info*/) {
	// This is an unmanaged reference. No ref/deref overhead. Lifetime of the
	// TableIoManager follows lifetime of the StorageManager (this).
	return shared_ptr<TableIOManager>(shared_ptr<char>(nullptr), table_io_manager.get());
}

} // namespace duckdb




namespace duckdb {

struct TransactionVersionOperator {
	static bool UseInsertedVersion(transaction_t start_time, transaction_t transaction_id, transaction_t id) {
		return id < start_time || id == transaction_id;
	}

	static bool UseDeletedVersion(transaction_t start_time, transaction_t transaction_id, transaction_t id) {
		return !UseInsertedVersion(start_time, transaction_id, id);
	}
};

struct CommittedVersionOperator {
	static bool UseInsertedVersion(transaction_t start_time, transaction_t transaction_id, transaction_t id) {
		return true;
	}

	static bool UseDeletedVersion(transaction_t min_start_time, transaction_t min_transaction_id, transaction_t id) {
		return (id >= min_start_time && id < TRANSACTION_ID_START) || (id >= min_transaction_id);
	}
};

static bool UseVersion(TransactionData transaction, transaction_t id) {
	return TransactionVersionOperator::UseInsertedVersion(transaction.start_time, transaction.transaction_id, id);
}

unique_ptr<ChunkInfo> ChunkInfo::Deserialize(Deserializer &source) {
	auto type = source.Read<ChunkInfoType>();
	switch (type) {
	case ChunkInfoType::EMPTY_INFO:
		return nullptr;
	case ChunkInfoType::CONSTANT_INFO:
		return ChunkConstantInfo::Deserialize(source);
	case ChunkInfoType::VECTOR_INFO:
		return ChunkVectorInfo::Deserialize(source);
	default:
		throw SerializationException("Could not deserialize Chunk Info Type: unrecognized type");
	}
}

//===--------------------------------------------------------------------===//
// Constant info
//===--------------------------------------------------------------------===//
ChunkConstantInfo::ChunkConstantInfo(idx_t start)
    : ChunkInfo(start, ChunkInfoType::CONSTANT_INFO), insert_id(0), delete_id(NOT_DELETED_ID) {
}

template <class OP>
idx_t ChunkConstantInfo::TemplatedGetSelVector(transaction_t start_time, transaction_t transaction_id,
                                               SelectionVector &sel_vector, idx_t max_count) {
	if (OP::UseInsertedVersion(start_time, transaction_id, insert_id) &&
	    OP::UseDeletedVersion(start_time, transaction_id, delete_id)) {
		return max_count;
	}
	return 0;
}

idx_t ChunkConstantInfo::GetSelVector(TransactionData transaction, SelectionVector &sel_vector, idx_t max_count) {
	return TemplatedGetSelVector<TransactionVersionOperator>(transaction.start_time, transaction.transaction_id,
	                                                         sel_vector, max_count);
}

idx_t ChunkConstantInfo::GetCommittedSelVector(transaction_t min_start_id, transaction_t min_transaction_id,
                                               SelectionVector &sel_vector, idx_t max_count) {
	return TemplatedGetSelVector<CommittedVersionOperator>(min_start_id, min_transaction_id, sel_vector, max_count);
}

bool ChunkConstantInfo::Fetch(TransactionData transaction, row_t row) {
	return UseVersion(transaction, insert_id) && !UseVersion(transaction, delete_id);
}

void ChunkConstantInfo::CommitAppend(transaction_t commit_id, idx_t start, idx_t end) {
	D_ASSERT(start == 0 && end == STANDARD_VECTOR_SIZE);
	insert_id = commit_id;
}

void ChunkConstantInfo::Serialize(Serializer &serializer) {
	// we only need to write this node if any tuple deletions have been committed
	bool is_deleted = insert_id >= TRANSACTION_ID_START || delete_id < TRANSACTION_ID_START;
	if (!is_deleted) {
		serializer.Write<ChunkInfoType>(ChunkInfoType::EMPTY_INFO);
		return;
	}
	serializer.Write<ChunkInfoType>(type);
	serializer.Write<idx_t>(start);
}

unique_ptr<ChunkInfo> ChunkConstantInfo::Deserialize(Deserializer &source) {
	auto start = source.Read<idx_t>();

	auto info = make_unique<ChunkConstantInfo>(start);
	info->insert_id = 0;
	info->delete_id = 0;
	return std::move(info);
}

//===--------------------------------------------------------------------===//
// Vector info
//===--------------------------------------------------------------------===//
ChunkVectorInfo::ChunkVectorInfo(idx_t start)
    : ChunkInfo(start, ChunkInfoType::VECTOR_INFO), insert_id(0), same_inserted_id(true), any_deleted(false) {
	for (idx_t i = 0; i < STANDARD_VECTOR_SIZE; i++) {
		inserted[i] = 0;
		deleted[i] = NOT_DELETED_ID;
	}
}

template <class OP>
idx_t ChunkVectorInfo::TemplatedGetSelVector(transaction_t start_time, transaction_t transaction_id,
                                             SelectionVector &sel_vector, idx_t max_count) {
	idx_t count = 0;
	if (same_inserted_id && !any_deleted) {
		// all tuples have the same inserted id: and no tuples were deleted
		if (OP::UseInsertedVersion(start_time, transaction_id, insert_id)) {
			return max_count;
		} else {
			return 0;
		}
	} else if (same_inserted_id) {
		if (!OP::UseInsertedVersion(start_time, transaction_id, insert_id)) {
			return 0;
		}
		// have to check deleted flag
		for (idx_t i = 0; i < max_count; i++) {
			if (OP::UseDeletedVersion(start_time, transaction_id, deleted[i])) {
				sel_vector.set_index(count++, i);
			}
		}
	} else if (!any_deleted) {
		// have to check inserted flag
		for (idx_t i = 0; i < max_count; i++) {
			if (OP::UseInsertedVersion(start_time, transaction_id, inserted[i])) {
				sel_vector.set_index(count++, i);
			}
		}
	} else {
		// have to check both flags
		for (idx_t i = 0; i < max_count; i++) {
			if (OP::UseInsertedVersion(start_time, transaction_id, inserted[i]) &&
			    OP::UseDeletedVersion(start_time, transaction_id, deleted[i])) {
				sel_vector.set_index(count++, i);
			}
		}
	}
	return count;
}

idx_t ChunkVectorInfo::GetSelVector(transaction_t start_time, transaction_t transaction_id, SelectionVector &sel_vector,
                                    idx_t max_count) {
	return TemplatedGetSelVector<TransactionVersionOperator>(start_time, transaction_id, sel_vector, max_count);
}

idx_t ChunkVectorInfo::GetCommittedSelVector(transaction_t min_start_id, transaction_t min_transaction_id,
                                             SelectionVector &sel_vector, idx_t max_count) {
	return TemplatedGetSelVector<CommittedVersionOperator>(min_start_id, min_transaction_id, sel_vector, max_count);
}

idx_t ChunkVectorInfo::GetSelVector(TransactionData transaction, SelectionVector &sel_vector, idx_t max_count) {
	return GetSelVector(transaction.start_time, transaction.transaction_id, sel_vector, max_count);
}

bool ChunkVectorInfo::Fetch(TransactionData transaction, row_t row) {
	return UseVersion(transaction, inserted[row]) && !UseVersion(transaction, deleted[row]);
}

idx_t ChunkVectorInfo::Delete(transaction_t transaction_id, row_t rows[], idx_t count) {
	any_deleted = true;

	idx_t deleted_tuples = 0;
	for (idx_t i = 0; i < count; i++) {
		if (deleted[rows[i]] == transaction_id) {
			continue;
		}
		// first check the chunk for conflicts
		if (deleted[rows[i]] != NOT_DELETED_ID) {
			// tuple was already deleted by another transaction
			throw TransactionException("Conflict on tuple deletion!");
		}
		// after verifying that there are no conflicts we mark the tuple as deleted
		deleted[rows[i]] = transaction_id;
		rows[deleted_tuples] = rows[i];
		deleted_tuples++;
	}
	return deleted_tuples;
}

void ChunkVectorInfo::CommitDelete(transaction_t commit_id, row_t rows[], idx_t count) {
	for (idx_t i = 0; i < count; i++) {
		deleted[rows[i]] = commit_id;
	}
}

void ChunkVectorInfo::Append(idx_t start, idx_t end, transaction_t commit_id) {
	if (start == 0) {
		insert_id = commit_id;
	} else if (insert_id != commit_id) {
		same_inserted_id = false;
		insert_id = NOT_DELETED_ID;
	}
	for (idx_t i = start; i < end; i++) {
		inserted[i] = commit_id;
	}
}

void ChunkVectorInfo::CommitAppend(transaction_t commit_id, idx_t start, idx_t end) {
	if (same_inserted_id) {
		insert_id = commit_id;
	}
	for (idx_t i = start; i < end; i++) {
		inserted[i] = commit_id;
	}
}

void ChunkVectorInfo::Serialize(Serializer &serializer) {
	SelectionVector sel(STANDARD_VECTOR_SIZE);
	transaction_t start_time = TRANSACTION_ID_START - 1;
	transaction_t transaction_id = DConstants::INVALID_INDEX;
	idx_t count = GetSelVector(start_time, transaction_id, sel, STANDARD_VECTOR_SIZE);
	if (count == STANDARD_VECTOR_SIZE) {
		// nothing is deleted: skip writing anything
		serializer.Write<ChunkInfoType>(ChunkInfoType::EMPTY_INFO);
		return;
	}
	if (count == 0) {
		// everything is deleted: write a constant vector
		serializer.Write<ChunkInfoType>(ChunkInfoType::CONSTANT_INFO);
		serializer.Write<idx_t>(start);
		return;
	}
	// write a boolean vector
	serializer.Write<ChunkInfoType>(ChunkInfoType::VECTOR_INFO);
	serializer.Write<idx_t>(start);
	bool deleted_tuples[STANDARD_VECTOR_SIZE];
	for (idx_t i = 0; i < STANDARD_VECTOR_SIZE; i++) {
		deleted_tuples[i] = true;
	}
	for (idx_t i = 0; i < count; i++) {
		deleted_tuples[sel.get_index(i)] = false;
	}
	serializer.WriteData((data_ptr_t)deleted_tuples, sizeof(bool) * STANDARD_VECTOR_SIZE);
}

unique_ptr<ChunkInfo> ChunkVectorInfo::Deserialize(Deserializer &source) {
	auto start = source.Read<idx_t>();

	auto result = make_unique<ChunkVectorInfo>(start);
	result->any_deleted = true;
	bool deleted_tuples[STANDARD_VECTOR_SIZE];
	source.ReadData((data_ptr_t)deleted_tuples, sizeof(bool) * STANDARD_VECTOR_SIZE);
	for (idx_t i = 0; i < STANDARD_VECTOR_SIZE; i++) {
		if (deleted_tuples[i]) {
			result->deleted[i] = 0;
		}
	}
	return std::move(result);
}

} // namespace duckdb














namespace duckdb {

ColumnCheckpointState::ColumnCheckpointState(RowGroup &row_group, ColumnData &column_data,
                                             PartialBlockManager &partial_block_manager)
    : row_group(row_group), column_data(column_data), partial_block_manager(partial_block_manager) {
}

ColumnCheckpointState::~ColumnCheckpointState() {
}

unique_ptr<BaseStatistics> ColumnCheckpointState::GetStatistics() {
	D_ASSERT(global_stats);
	return std::move(global_stats);
}

struct PartialBlockForCheckpoint : PartialBlock {
	struct PartialColumnSegment {
		ColumnData *data;
		ColumnSegment *segment;
		uint32_t offset_in_block;
	};

public:
	PartialBlockForCheckpoint(ColumnData *first_data, ColumnSegment *first_segment, BlockManager &block_manager,
	                          PartialBlockState state)
	    : PartialBlock(state), first_data(first_data), first_segment(first_segment), block_manager(block_manager) {
	}

	~PartialBlockForCheckpoint() override {
		D_ASSERT(IsFlushed() || Exception::UncaughtException());
	}

	// We will copy all subsequent segment data into the memory corresponding
	// to the first segment. Once the block is full (or checkpoint is complete)
	// we'll invoke Flush(), which will cause
	// the block to get written to storage (via BlockManger::ConvertToPersistent),
	// and all segments to have their references updated
	// (via ColumnSegment::ConvertToPersistent)
	ColumnData *first_data;
	ColumnSegment *first_segment;
	BlockManager &block_manager;
	vector<PartialColumnSegment> tail_segments;

public:
	bool IsFlushed() {
		// first_segment is zeroed on Flush
		return !first_segment;
	}

	void Flush() override {
		// At this point, we've already copied all data from tail_segments
		// into the page owned by first_segment. We flush all segment data to
		// disk with the following call.
		first_data->IncrementVersion();
		first_segment->ConvertToPersistent(&block_manager, state.block_id);
		// Now that the page is persistent, update tail_segments to point to the
		// newly persistent block.
		for (auto e : tail_segments) {
			e.data->IncrementVersion();
			e.segment->MarkAsPersistent(first_segment->block, e.offset_in_block);
		}
		first_segment = nullptr;
		tail_segments.clear();
	}

	void Clear() override {
		first_data = nullptr;
		first_segment = nullptr;
		tail_segments.clear();
	}

	void AddSegmentToTail(ColumnData *data, ColumnSegment *segment, uint32_t offset_in_block) {
		tail_segments.push_back({data, segment, offset_in_block});
	}
};

void ColumnCheckpointState::FlushSegment(unique_ptr<ColumnSegment> segment, idx_t segment_size) {
	D_ASSERT(segment_size <= Storage::BLOCK_SIZE);
	auto tuple_count = segment->count.load();
	if (tuple_count == 0) { // LCOV_EXCL_START
		return;
	} // LCOV_EXCL_STOP

	// merge the segment stats into the global stats
	global_stats->Merge(*segment->stats.statistics);

	// get the buffer of the segment and pin it
	auto &db = column_data.GetDatabase();
	auto &buffer_manager = BufferManager::GetBufferManager(db);
	block_id_t block_id = INVALID_BLOCK;
	uint32_t offset_in_block = 0;

	if (!segment->stats.statistics->IsConstant()) {
		// non-constant block
		PartialBlockAllocation allocation = partial_block_manager.GetBlockAllocation(segment_size);
		block_id = allocation.state.block_id;
		offset_in_block = allocation.state.offset_in_block;

		if (allocation.partial_block) {
			// Use an existing block.
			D_ASSERT(offset_in_block > 0);
			auto pstate = (PartialBlockForCheckpoint *)allocation.partial_block.get();
			// pin the source block
			auto old_handle = buffer_manager.Pin(segment->block);
			// pin the target block
			auto new_handle = buffer_manager.Pin(pstate->first_segment->block);
			// memcpy the contents of the old block to the new block
			memcpy(new_handle.Ptr() + offset_in_block, old_handle.Ptr(), segment_size);
			pstate->AddSegmentToTail(&column_data, segment.get(), offset_in_block);
		} else {
			// Create a new block for future reuse.
			if (segment->SegmentSize() != Storage::BLOCK_SIZE) {
				// the segment is smaller than the block size
				// allocate a new block and copy the data over
				D_ASSERT(segment->SegmentSize() < Storage::BLOCK_SIZE);
				segment->Resize(Storage::BLOCK_SIZE);
			}
			D_ASSERT(offset_in_block == 0);
			allocation.partial_block = make_unique<PartialBlockForCheckpoint>(
			    &column_data, segment.get(), *allocation.block_manager, allocation.state);
		}
		// Writer will decide whether to reuse this block.
		partial_block_manager.RegisterPartialBlock(std::move(allocation));
	} else {
		// constant block: no need to write anything to disk besides the stats
		// set up the compression function to constant
		auto &config = DBConfig::GetConfig(db);
		segment->function =
		    config.GetCompressionFunction(CompressionType::COMPRESSION_CONSTANT, segment->type.InternalType());
		segment->ConvertToPersistent(nullptr, INVALID_BLOCK);
	}

	// construct the data pointer
	DataPointer data_pointer;
	data_pointer.block_pointer.block_id = block_id;
	data_pointer.block_pointer.offset = offset_in_block;
	data_pointer.row_start = row_group.start;
	if (!data_pointers.empty()) {
		auto &last_pointer = data_pointers.back();
		data_pointer.row_start = last_pointer.row_start + last_pointer.tuple_count;
	}
	data_pointer.tuple_count = tuple_count;
	data_pointer.compression_type = segment->function->type;
	data_pointer.statistics = segment->stats.statistics->Copy();

	// append the segment to the new segment tree
	new_tree.AppendSegment(std::move(segment));
	data_pointers.push_back(std::move(data_pointer));
}

void ColumnCheckpointState::WriteDataPointers(RowGroupWriter &writer) {
	writer.WriteColumnDataPointers(*this);
}

void ColumnCheckpointState::GetBlockIds(unordered_set<block_id_t> &result) {
	for (auto &pointer : data_pointers) {
		if (pointer.block_pointer.block_id == INVALID_BLOCK) {
			continue;
		}
		result.insert(pointer.block_pointer.block_id);
	}
}

} // namespace duckdb



















namespace duckdb {

ColumnData::ColumnData(BlockManager &block_manager, DataTableInfo &info, idx_t column_index, idx_t start_row,
                       LogicalType type, ColumnData *parent)
    : block_manager(block_manager), info(info), column_index(column_index), start(start_row), type(std::move(type)),
      parent(parent), version(0) {
}

ColumnData::ColumnData(ColumnData &other, idx_t start, ColumnData *parent)
    : block_manager(other.block_manager), info(other.info), column_index(other.column_index), start(start),
      type(std::move(other.type)), parent(parent), updates(std::move(other.updates)),
      version(parent ? parent->version + 1 : 0) {
	idx_t offset = 0;
	for (auto segment = other.data.GetRootSegment(); segment; segment = segment->Next()) {
		auto &other = (ColumnSegment &)*segment;
		this->data.AppendSegment(ColumnSegment::CreateSegment(other, start + offset));
		offset += segment->count;
	}
}

ColumnData::~ColumnData() {
}

DatabaseInstance &ColumnData::GetDatabase() const {
	return info.db.GetDatabase();
}

DataTableInfo &ColumnData::GetTableInfo() const {
	return info;
}

const LogicalType &ColumnData::RootType() const {
	if (parent) {
		return parent->RootType();
	}
	return type;
}

void ColumnData::IncrementVersion() {
	version++;
}

idx_t ColumnData::GetMaxEntry() {
	auto l = data.Lock();
	auto first_segment = data.GetRootSegment(l);
	auto last_segment = data.GetLastSegment(l);
	if (!first_segment) {
		D_ASSERT(!last_segment);
		return 0;
	} else {
		D_ASSERT(last_segment->start >= first_segment->start);
		return last_segment->start + last_segment->count - first_segment->start;
	}
}

void ColumnData::InitializeScan(ColumnScanState &state) {
	state.current = (ColumnSegment *)data.GetRootSegment();
	state.row_index = state.current ? state.current->start : 0;
	state.internal_index = state.row_index;
	state.initialized = false;
	state.version = version;
	state.scan_state.reset();
}

void ColumnData::InitializeScanWithOffset(ColumnScanState &state, idx_t row_idx) {
	state.current = (ColumnSegment *)data.GetSegment(row_idx);
	state.row_index = row_idx;
	state.internal_index = state.current->start;
	state.initialized = false;
	state.version = version;
	state.scan_state.reset();
}

idx_t ColumnData::ScanVector(ColumnScanState &state, Vector &result, idx_t remaining) {
	state.previous_states.clear();
	if (state.version != version) {
		InitializeScanWithOffset(state, state.row_index);
		state.current->InitializeScan(state);
		state.initialized = true;
	} else if (!state.initialized) {
		D_ASSERT(state.current);
		state.current->InitializeScan(state);
		state.internal_index = state.current->start;
		state.initialized = true;
	}
	D_ASSERT(data.HasSegment(state.current));
	D_ASSERT(state.version == version);
	D_ASSERT(state.internal_index <= state.row_index);
	if (state.internal_index < state.row_index) {
		state.current->Skip(state);
	}
	D_ASSERT(state.current->type == type);
	idx_t initial_remaining = remaining;
	while (remaining > 0) {
		D_ASSERT(state.row_index >= state.current->start &&
		         state.row_index <= state.current->start + state.current->count);
		idx_t scan_count = MinValue<idx_t>(remaining, state.current->start + state.current->count - state.row_index);
		idx_t result_offset = initial_remaining - remaining;
		if (scan_count > 0) {
			state.current->Scan(state, scan_count, result, result_offset, scan_count == initial_remaining);

			state.row_index += scan_count;
			remaining -= scan_count;
		}

		if (remaining > 0) {
			if (!state.current->next) {
				break;
			}
			state.previous_states.emplace_back(std::move(state.scan_state));
			state.current = (ColumnSegment *)state.current->Next();
			state.current->InitializeScan(state);
			state.segment_checked = false;
			D_ASSERT(state.row_index >= state.current->start &&
			         state.row_index <= state.current->start + state.current->count);
		}
	}
	state.internal_index = state.row_index;
	return initial_remaining - remaining;
}

template <bool SCAN_COMMITTED, bool ALLOW_UPDATES>
idx_t ColumnData::ScanVector(TransactionData transaction, idx_t vector_index, ColumnScanState &state, Vector &result) {
	auto scan_count = ScanVector(state, result, STANDARD_VECTOR_SIZE);

	lock_guard<mutex> update_guard(update_lock);
	if (updates) {
		if (!ALLOW_UPDATES && updates->HasUncommittedUpdates(vector_index)) {
			throw TransactionException("Cannot create index with outstanding updates");
		}
		result.Flatten(scan_count);
		if (SCAN_COMMITTED) {
			updates->FetchCommitted(vector_index, result);
		} else {
			updates->FetchUpdates(transaction, vector_index, result);
		}
	}
	return scan_count;
}

template idx_t ColumnData::ScanVector<false, false>(TransactionData transaction, idx_t vector_index,
                                                    ColumnScanState &state, Vector &result);
template idx_t ColumnData::ScanVector<true, false>(TransactionData transaction, idx_t vector_index,
                                                   ColumnScanState &state, Vector &result);
template idx_t ColumnData::ScanVector<false, true>(TransactionData transaction, idx_t vector_index,
                                                   ColumnScanState &state, Vector &result);
template idx_t ColumnData::ScanVector<true, true>(TransactionData transaction, idx_t vector_index,
                                                  ColumnScanState &state, Vector &result);

idx_t ColumnData::Scan(TransactionData transaction, idx_t vector_index, ColumnScanState &state, Vector &result) {
	return ScanVector<false, true>(transaction, vector_index, state, result);
}

idx_t ColumnData::ScanCommitted(idx_t vector_index, ColumnScanState &state, Vector &result, bool allow_updates) {
	if (allow_updates) {
		return ScanVector<true, true>(TransactionData(0, 0), vector_index, state, result);
	} else {
		return ScanVector<true, false>(TransactionData(0, 0), vector_index, state, result);
	}
}

void ColumnData::ScanCommittedRange(idx_t row_group_start, idx_t offset_in_row_group, idx_t count, Vector &result) {
	ColumnScanState child_state;
	InitializeScanWithOffset(child_state, row_group_start + offset_in_row_group);
	auto scan_count = ScanVector(child_state, result, count);
	if (updates) {
		result.Flatten(scan_count);
		updates->FetchCommittedRange(offset_in_row_group, count, result);
	}
}

idx_t ColumnData::ScanCount(ColumnScanState &state, Vector &result, idx_t count) {
	if (count == 0) {
		return 0;
	}
	// ScanCount can only be used if there are no updates
	D_ASSERT(!updates);
	return ScanVector(state, result, count);
}

void ColumnData::Select(TransactionData transaction, idx_t vector_index, ColumnScanState &state, Vector &result,
                        SelectionVector &sel, idx_t &count, const TableFilter &filter) {
	idx_t scan_count = Scan(transaction, vector_index, state, result);
	result.Flatten(scan_count);
	ColumnSegment::FilterSelection(sel, result, filter, count, FlatVector::Validity(result));
}

void ColumnData::FilterScan(TransactionData transaction, idx_t vector_index, ColumnScanState &state, Vector &result,
                            SelectionVector &sel, idx_t count) {
	Scan(transaction, vector_index, state, result);
	result.Slice(sel, count);
}

void ColumnData::FilterScanCommitted(idx_t vector_index, ColumnScanState &state, Vector &result, SelectionVector &sel,
                                     idx_t count, bool allow_updates) {
	ScanCommitted(vector_index, state, result, allow_updates);
	result.Slice(sel, count);
}

void ColumnData::Skip(ColumnScanState &state, idx_t count) {
	state.Next(count);
}

void ColumnData::Append(BaseStatistics &stats, ColumnAppendState &state, Vector &vector, idx_t count) {
	UnifiedVectorFormat vdata;
	vector.ToUnifiedFormat(count, vdata);
	AppendData(stats, state, vdata, count);
}

void ColumnData::InitializeAppend(ColumnAppendState &state) {
	auto l = data.Lock();
	if (data.IsEmpty(l)) {
		// no segments yet, append an empty segment
		AppendTransientSegment(l, start);
	}
	auto segment = (ColumnSegment *)data.GetLastSegment(l);
	if (segment->segment_type == ColumnSegmentType::PERSISTENT) {
		// no transient segments yet
		auto total_rows = segment->start + segment->count;
		AppendTransientSegment(l, total_rows);
		state.current = (ColumnSegment *)data.GetLastSegment(l);
	} else {
		state.current = (ColumnSegment *)segment;
	}

	D_ASSERT(state.current->segment_type == ColumnSegmentType::TRANSIENT);
	state.current->InitializeAppend(state);
	D_ASSERT(state.current->function->append);
}

void ColumnData::AppendData(BaseStatistics &stats, ColumnAppendState &state, UnifiedVectorFormat &vdata, idx_t count) {
	idx_t offset = 0;
	while (true) {
		// append the data from the vector
		idx_t copied_elements = state.current->Append(state, vdata, offset, count);
		stats.Merge(*state.current->stats.statistics);
		if (copied_elements == count) {
			// finished copying everything
			break;
		}

		// we couldn't fit everything we wanted in the current column segment, create a new one
		{
			auto l = data.Lock();
			AppendTransientSegment(l, state.current->start + state.current->count);
			state.current = (ColumnSegment *)data.GetLastSegment(l);
			state.current->InitializeAppend(state);
		}
		offset += copied_elements;
		count -= copied_elements;
	}
}

void ColumnData::RevertAppend(row_t start_row) {
	auto l = data.Lock();
	// check if this row is in the segment tree at all
	auto last_segment = data.GetLastSegment(l);
	if (idx_t(start_row) >= last_segment->start + last_segment->count) {
		// the start row is equal to the final portion of the column data: nothing was ever appended here
		D_ASSERT(idx_t(start_row) == last_segment->start + last_segment->count);
		return;
	}
	// find the segment index that the current row belongs to
	idx_t segment_index = data.GetSegmentIndex(l, start_row);
	auto segment = data.GetSegmentByIndex(l, segment_index);
	auto &transient = (ColumnSegment &)*segment;
	D_ASSERT(transient.segment_type == ColumnSegmentType::TRANSIENT);

	// remove any segments AFTER this segment: they should be deleted entirely
	data.EraseSegments(l, segment_index);

	segment->next = nullptr;
	transient.RevertAppend(start_row);
}

idx_t ColumnData::Fetch(ColumnScanState &state, row_t row_id, Vector &result) {
	D_ASSERT(row_id >= 0);
	D_ASSERT(idx_t(row_id) >= start);
	// perform the fetch within the segment
	state.row_index = start + ((row_id - start) / STANDARD_VECTOR_SIZE * STANDARD_VECTOR_SIZE);
	state.current = (ColumnSegment *)data.GetSegment(state.row_index);
	state.internal_index = state.current->start;
	return ScanVector(state, result, STANDARD_VECTOR_SIZE);
}

void ColumnData::FetchRow(TransactionData transaction, ColumnFetchState &state, row_t row_id, Vector &result,
                          idx_t result_idx) {
	auto segment = (ColumnSegment *)data.GetSegment(row_id);

	// now perform the fetch within the segment
	segment->FetchRow(state, row_id, result, result_idx);
	// merge any updates made to this row
	lock_guard<mutex> update_guard(update_lock);
	if (updates) {
		updates->FetchRow(transaction, row_id, result, result_idx);
	}
}

void ColumnData::Update(TransactionData transaction, idx_t column_index, Vector &update_vector, row_t *row_ids,
                        idx_t update_count) {
	lock_guard<mutex> update_guard(update_lock);
	if (!updates) {
		updates = make_unique<UpdateSegment>(*this);
	}
	Vector base_vector(type);
	ColumnScanState state;
	auto fetch_count = Fetch(state, row_ids[0], base_vector);

	base_vector.Flatten(fetch_count);
	updates->Update(transaction, column_index, update_vector, row_ids, update_count, base_vector);
}

void ColumnData::UpdateColumn(TransactionData transaction, const vector<column_t> &column_path, Vector &update_vector,
                              row_t *row_ids, idx_t update_count, idx_t depth) {
	// this method should only be called at the end of the path in the base column case
	D_ASSERT(depth >= column_path.size());
	ColumnData::Update(transaction, column_path[0], update_vector, row_ids, update_count);
}

unique_ptr<BaseStatistics> ColumnData::GetUpdateStatistics() {
	lock_guard<mutex> update_guard(update_lock);
	return updates ? updates->GetStatistics() : nullptr;
}

void ColumnData::AppendTransientSegment(SegmentLock &l, idx_t start_row) {
	idx_t segment_size = Storage::BLOCK_SIZE;
	if (start_row == idx_t(MAX_ROW_ID)) {
#if STANDARD_VECTOR_SIZE < 1024
		segment_size = 1024 * GetTypeIdSize(type.InternalType());
#else
		segment_size = STANDARD_VECTOR_SIZE * GetTypeIdSize(type.InternalType());
#endif
	}
	auto new_segment = ColumnSegment::CreateTransientSegment(GetDatabase(), type, start_row, segment_size);
	data.AppendSegment(l, std::move(new_segment));
}

void ColumnData::CommitDropColumn() {
	auto segment = (ColumnSegment *)data.GetRootSegment();
	while (segment) {
		if (segment->segment_type == ColumnSegmentType::PERSISTENT) {
			auto block_id = segment->GetBlockId();
			if (block_id != INVALID_BLOCK) {
				block_manager.MarkBlockAsModified(block_id);
			}
		}
		segment = (ColumnSegment *)segment->Next();
	}
}

unique_ptr<ColumnCheckpointState> ColumnData::CreateCheckpointState(RowGroup &row_group,
                                                                    PartialBlockManager &partial_block_manager) {
	return make_unique<ColumnCheckpointState>(row_group, *this, partial_block_manager);
}

void ColumnData::CheckpointScan(ColumnSegment *segment, ColumnScanState &state, idx_t row_group_start, idx_t count,
                                Vector &scan_vector) {
	segment->Scan(state, count, scan_vector, 0, true);
	if (updates) {
		scan_vector.Flatten(count);
		updates->FetchCommittedRange(state.row_index - row_group_start, count, scan_vector);
	}
}

unique_ptr<ColumnCheckpointState> ColumnData::Checkpoint(RowGroup &row_group,
                                                         PartialBlockManager &partial_block_manager,
                                                         ColumnCheckpointInfo &checkpoint_info) {
	// scan the segments of the column data
	// set up the checkpoint state
	auto checkpoint_state = CreateCheckpointState(row_group, partial_block_manager);
	checkpoint_state->global_stats = BaseStatistics::CreateEmpty(type, StatisticsType::LOCAL_STATS);

	auto l = data.Lock();
	auto nodes = data.MoveSegments(l);
	if (nodes.empty()) {
		// empty table: flush the empty list
		return checkpoint_state;
	}
	lock_guard<mutex> update_guard(update_lock);

	ColumnDataCheckpointer checkpointer(*this, row_group, *checkpoint_state, checkpoint_info);
	checkpointer.Checkpoint(std::move(nodes));

	// replace the old tree with the new one
	data.Replace(l, checkpoint_state->new_tree);
	version++;

	return checkpoint_state;
}

void ColumnData::DeserializeColumn(Deserializer &source) {
	// load the data pointers for the column
	idx_t data_pointer_count = source.Read<idx_t>();
	for (idx_t data_ptr = 0; data_ptr < data_pointer_count; data_ptr++) {
		// read the data pointer
		DataPointer data_pointer;
		data_pointer.row_start = source.Read<idx_t>();
		data_pointer.tuple_count = source.Read<idx_t>();
		data_pointer.block_pointer.block_id = source.Read<block_id_t>();
		data_pointer.block_pointer.offset = source.Read<uint32_t>();
		data_pointer.compression_type = source.Read<CompressionType>();
		data_pointer.statistics = BaseStatistics::Deserialize(source, type);

		// create a persistent segment
		auto segment = ColumnSegment::CreatePersistentSegment(
		    GetDatabase(), block_manager, data_pointer.block_pointer.block_id, data_pointer.block_pointer.offset, type,
		    data_pointer.row_start, data_pointer.tuple_count, data_pointer.compression_type,
		    std::move(data_pointer.statistics));
		data.AppendSegment(std::move(segment));
	}
}

shared_ptr<ColumnData> ColumnData::Deserialize(BlockManager &block_manager, DataTableInfo &info, idx_t column_index,
                                               idx_t start_row, Deserializer &source, const LogicalType &type,
                                               ColumnData *parent) {
	auto entry = ColumnData::CreateColumn(block_manager, info, column_index, start_row, type, parent);
	entry->DeserializeColumn(source);
	return entry;
}

void ColumnData::GetStorageInfo(idx_t row_group_index, vector<idx_t> col_path, TableStorageInfo &result) {
	D_ASSERT(!col_path.empty());

	// convert the column path to a string
	string col_path_str = "[";
	for (idx_t i = 0; i < col_path.size(); i++) {
		if (i > 0) {
			col_path_str += ", ";
		}
		col_path_str += to_string(col_path[i]);
	}
	col_path_str += "]";

	// iterate over the segments
	idx_t segment_idx = 0;
	auto segment = (ColumnSegment *)data.GetRootSegment();
	while (segment) {
		ColumnSegmentInfo column_info;
		column_info.row_group_index = row_group_index;
		;
		column_info.column_id = col_path[0];
		column_info.column_path = col_path_str;
		column_info.segment_idx = segment_idx;
		column_info.segment_type = type.ToString();
		column_info.segment_start = segment->start;
		column_info.segment_count = segment->count;
		column_info.compression_type = CompressionTypeToString(segment->function->type);
		column_info.segment_stats =
		    segment->stats.statistics ? segment->stats.statistics->ToString() : string("No Stats");
		column_info.has_updates = updates ? true : false;
		// persistent
		// block_id
		// block_offset
		if (segment->segment_type == ColumnSegmentType::PERSISTENT) {
			column_info.persistent = true;
			column_info.block_id = segment->GetBlockId();
			column_info.block_offset = segment->GetBlockOffset();
		} else {
			column_info.persistent = false;
		}
		result.column_segments.push_back(std::move(column_info));

		segment_idx++;
		segment = (ColumnSegment *)segment->Next();
	}
}

void ColumnData::Verify(RowGroup &parent) {
#ifdef DEBUG
	D_ASSERT(this->start == parent.start);
	data.Verify();
	auto root = data.GetRootSegment();
	if (root) {
		D_ASSERT(root != nullptr);
		D_ASSERT(root->start == this->start);
		idx_t prev_end = root->start;
		while (root) {
			D_ASSERT(prev_end == root->start);
			prev_end = root->start + root->count;
			if (!root->next) {
				D_ASSERT(prev_end == parent.start + parent.count);
			}
			root = root->Next();
		}
	}
#endif
}

template <class RET, class OP>
static RET CreateColumnInternal(BlockManager &block_manager, DataTableInfo &info, idx_t column_index, idx_t start_row,
                                const LogicalType &type, ColumnData *parent) {
	if (type.InternalType() == PhysicalType::STRUCT) {
		return OP::template Create<StructColumnData>(block_manager, info, column_index, start_row, type, parent);
	} else if (type.InternalType() == PhysicalType::LIST) {
		return OP::template Create<ListColumnData>(block_manager, info, column_index, start_row, type, parent);
	} else if (type.id() == LogicalTypeId::VALIDITY) {
		return OP::template Create<ValidityColumnData>(block_manager, info, column_index, start_row, parent);
	}
	return OP::template Create<StandardColumnData>(block_manager, info, column_index, start_row, type, parent);
}

template <class RET, class OP>
static RET CreateColumnInternal(ColumnData &other, idx_t start_row, ColumnData *parent) {
	if (other.type.InternalType() == PhysicalType::STRUCT) {
		return OP::template Create<StructColumnData>(other, start_row, parent);
	} else if (other.type.InternalType() == PhysicalType::LIST) {
		return OP::template Create<ListColumnData>(other, start_row, parent);
	} else if (other.type.id() == LogicalTypeId::VALIDITY) {
		return OP::template Create<ValidityColumnData>(other, start_row, parent);
	}
	return OP::template Create<StandardColumnData>(other, start_row, parent);
}

shared_ptr<ColumnData> ColumnData::CreateColumn(BlockManager &block_manager, DataTableInfo &info, idx_t column_index,
                                                idx_t start_row, const LogicalType &type, ColumnData *parent) {
	return CreateColumnInternal<shared_ptr<ColumnData>, SharedConstructor>(block_manager, info, column_index, start_row,
	                                                                       type, parent);
}

shared_ptr<ColumnData> ColumnData::CreateColumn(ColumnData &other, idx_t start_row, ColumnData *parent) {
	return CreateColumnInternal<shared_ptr<ColumnData>, SharedConstructor>(other, start_row, parent);
}

unique_ptr<ColumnData> ColumnData::CreateColumnUnique(BlockManager &block_manager, DataTableInfo &info,
                                                      idx_t column_index, idx_t start_row, const LogicalType &type,
                                                      ColumnData *parent) {
	return CreateColumnInternal<unique_ptr<ColumnData>, UniqueConstructor>(block_manager, info, column_index, start_row,
	                                                                       type, parent);
}

unique_ptr<ColumnData> ColumnData::CreateColumnUnique(ColumnData &other, idx_t start_row, ColumnData *parent) {
	return CreateColumnInternal<unique_ptr<ColumnData>, UniqueConstructor>(other, start_row, parent);
}

} // namespace duckdb





namespace duckdb {

ColumnDataCheckpointer::ColumnDataCheckpointer(ColumnData &col_data_p, RowGroup &row_group_p,
                                               ColumnCheckpointState &state_p, ColumnCheckpointInfo &checkpoint_info_p)
    : col_data(col_data_p), row_group(row_group_p), state(state_p),
      is_validity(GetType().id() == LogicalTypeId::VALIDITY),
      intermediate(is_validity ? LogicalType::BOOLEAN : GetType(), true, is_validity),
      checkpoint_info(checkpoint_info_p) {
	auto &config = DBConfig::GetConfig(GetDatabase());
	compression_functions = config.GetCompressionFunctions(GetType().InternalType());
}

DatabaseInstance &ColumnDataCheckpointer::GetDatabase() {
	return col_data.GetDatabase();
}

const LogicalType &ColumnDataCheckpointer::GetType() const {
	return col_data.type;
}

ColumnData &ColumnDataCheckpointer::GetColumnData() {
	return col_data;
}

RowGroup &ColumnDataCheckpointer::GetRowGroup() {
	return row_group;
}

ColumnCheckpointState &ColumnDataCheckpointer::GetCheckpointState() {
	return state;
}

void ColumnDataCheckpointer::ScanSegments(const std::function<void(Vector &, idx_t)> &callback) {
	Vector scan_vector(intermediate.GetType(), nullptr);
	for (idx_t segment_idx = 0; segment_idx < nodes.size(); segment_idx++) {
		auto segment = (ColumnSegment *)nodes[segment_idx].node.get();
		ColumnScanState scan_state;
		scan_state.current = segment;
		segment->InitializeScan(scan_state);

		for (idx_t base_row_index = 0; base_row_index < segment->count; base_row_index += STANDARD_VECTOR_SIZE) {
			scan_vector.Reference(intermediate);

			idx_t count = MinValue<idx_t>(segment->count - base_row_index, STANDARD_VECTOR_SIZE);
			scan_state.row_index = segment->start + base_row_index;

			col_data.CheckpointScan(segment, scan_state, row_group.start, count, scan_vector);

			callback(scan_vector, count);
		}
	}
}

CompressionType ForceCompression(vector<CompressionFunction *> &compression_functions,
                                 CompressionType compression_type) {
	// On of the force_compression flags has been set
	// check if this compression method is available
	bool found = false;
	for (idx_t i = 0; i < compression_functions.size(); i++) {
		if (compression_functions[i]->type == compression_type) {
			found = true;
			break;
		}
	}
	if (found) {
		// the force_compression method is available
		// clear all other compression methods
		// except the uncompressed method, so we can fall back on that
		for (idx_t i = 0; i < compression_functions.size(); i++) {
			if (compression_functions[i]->type == CompressionType::COMPRESSION_UNCOMPRESSED) {
				continue;
			}
			if (compression_functions[i]->type != compression_type) {
				compression_functions[i] = nullptr;
			}
		}
	}
	return found ? compression_type : CompressionType::COMPRESSION_AUTO;
}

unique_ptr<AnalyzeState> ColumnDataCheckpointer::DetectBestCompressionMethod(idx_t &compression_idx) {
	D_ASSERT(!compression_functions.empty());
	auto &config = DBConfig::GetConfig(GetDatabase());
	CompressionType forced_method = CompressionType::COMPRESSION_AUTO;

	auto compression_type = checkpoint_info.compression_type;
	if (compression_type != CompressionType::COMPRESSION_AUTO) {
		forced_method = ForceCompression(compression_functions, compression_type);
	}
	if (compression_type == CompressionType::COMPRESSION_AUTO &&
	    config.options.force_compression != CompressionType::COMPRESSION_AUTO) {
		forced_method = ForceCompression(compression_functions, config.options.force_compression);
	}
	// set up the analyze states for each compression method
	vector<unique_ptr<AnalyzeState>> analyze_states;
	analyze_states.reserve(compression_functions.size());
	for (idx_t i = 0; i < compression_functions.size(); i++) {
		if (!compression_functions[i]) {
			analyze_states.push_back(nullptr);
			continue;
		}
		analyze_states.push_back(compression_functions[i]->init_analyze(col_data, col_data.type.InternalType()));
	}

	// scan over all the segments and run the analyze step
	ScanSegments([&](Vector &scan_vector, idx_t count) {
		for (idx_t i = 0; i < compression_functions.size(); i++) {
			if (!compression_functions[i]) {
				continue;
			}
			auto success = compression_functions[i]->analyze(*analyze_states[i], scan_vector, count);
			if (!success) {
				// could not use this compression function on this data set
				// erase it
				compression_functions[i] = nullptr;
				analyze_states[i].reset();
			}
		}
	});

	// now that we have passed over all the data, we need to figure out the best method
	// we do this using the final_analyze method
	unique_ptr<AnalyzeState> state;
	compression_idx = DConstants::INVALID_INDEX;
	idx_t best_score = NumericLimits<idx_t>::Maximum();
	for (idx_t i = 0; i < compression_functions.size(); i++) {
		if (!compression_functions[i]) {
			continue;
		}
		//! Check if the method type is the forced method (if forced is used)
		bool forced_method_found = compression_functions[i]->type == forced_method;
		auto score = compression_functions[i]->final_analyze(*analyze_states[i]);

		//! The finalize method can return this value from final_analyze to indicate it should not be used.
		if (score == DConstants::INVALID_INDEX) {
			continue;
		}

		if (score < best_score || forced_method_found) {
			compression_idx = i;
			best_score = score;
			state = std::move(analyze_states[i]);
		}
		//! If we have found the forced method, we're done
		if (forced_method_found) {
			break;
		}
	}
	return state;
}

void ColumnDataCheckpointer::WriteToDisk() {
	// there were changes or transient segments
	// we need to rewrite the column segments to disk

	// first we check the current segments
	// if there are any persistent segments, we will mark their old block ids as modified
	// since the segments will be rewritten their old on disk data is no longer required
	auto &block_manager = col_data.block_manager;
	for (idx_t segment_idx = 0; segment_idx < nodes.size(); segment_idx++) {
		auto segment = (ColumnSegment *)nodes[segment_idx].node.get();
		if (segment->segment_type == ColumnSegmentType::PERSISTENT) {
			// persistent segment has updates: mark it as modified and rewrite the block with the merged updates
			auto block_id = segment->GetBlockId();
			if (block_id != INVALID_BLOCK) {
				block_manager.MarkBlockAsModified(block_id);
			}
		}
	}

	// now we need to write our segment
	// we will first run an analyze step that determines which compression function to use
	idx_t compression_idx;
	auto analyze_state = DetectBestCompressionMethod(compression_idx);

	if (!analyze_state) {
		throw FatalException("No suitable compression/storage method found to store column");
	}

	// now that we have analyzed the compression functions we can start writing to disk
	auto best_function = compression_functions[compression_idx];
	auto compress_state = best_function->init_compression(*this, std::move(analyze_state));
	ScanSegments(
	    [&](Vector &scan_vector, idx_t count) { best_function->compress(*compress_state, scan_vector, count); });
	best_function->compress_finalize(*compress_state);

	nodes.clear();
}

bool ColumnDataCheckpointer::HasChanges() {
	for (idx_t segment_idx = 0; segment_idx < nodes.size(); segment_idx++) {
		auto segment = (ColumnSegment *)nodes[segment_idx].node.get();
		if (segment->segment_type == ColumnSegmentType::TRANSIENT) {
			// transient segment: always need to write to disk
			return true;
		} else {
			// persistent segment; check if there were any updates or deletions in this segment
			idx_t start_row_idx = segment->start - row_group.start;
			idx_t end_row_idx = start_row_idx + segment->count;
			if (col_data.updates && col_data.updates->HasUpdates(start_row_idx, end_row_idx)) {
				return true;
			}
		}
	}
	return false;
}

void ColumnDataCheckpointer::WritePersistentSegments() {
	// all segments are persistent and there are no updates
	// we only need to write the metadata
	for (idx_t segment_idx = 0; segment_idx < nodes.size(); segment_idx++) {
		auto segment = (ColumnSegment *)nodes[segment_idx].node.get();
		D_ASSERT(segment->segment_type == ColumnSegmentType::PERSISTENT);

		// set up the data pointer directly using the data from the persistent segment
		DataPointer pointer;
		pointer.block_pointer.block_id = segment->GetBlockId();
		pointer.block_pointer.offset = segment->GetBlockOffset();
		pointer.row_start = segment->start;
		pointer.tuple_count = segment->count;
		pointer.compression_type = segment->function->type;
		pointer.statistics = segment->stats.statistics->Copy();

		// merge the persistent stats into the global column stats
		state.global_stats->Merge(*segment->stats.statistics);

		// directly append the current segment to the new tree
		state.new_tree.AppendSegment(std::move(nodes[segment_idx].node));

		state.data_pointers.push_back(std::move(pointer));
	}
}

void ColumnDataCheckpointer::Checkpoint(vector<SegmentNode> nodes) {
	D_ASSERT(!nodes.empty());
	this->nodes = std::move(nodes);
	// first check if any of the segments have changes
	if (!HasChanges()) {
		// no changes: only need to write the metadata for this column
		WritePersistentSegments();
	} else {
		// there are changes: rewrite the set of columns
		WriteToDisk();
	}
}

} // namespace duckdb













#include <cstring>

namespace duckdb {

unique_ptr<ColumnSegment> ColumnSegment::CreatePersistentSegment(DatabaseInstance &db, BlockManager &block_manager,
                                                                 block_id_t block_id, idx_t offset,
                                                                 const LogicalType &type, idx_t start, idx_t count,
                                                                 CompressionType compression_type,
                                                                 unique_ptr<BaseStatistics> statistics) {
	auto &config = DBConfig::GetConfig(db);
	CompressionFunction *function;
	shared_ptr<BlockHandle> block;
	if (block_id == INVALID_BLOCK) {
		// constant segment, no need to allocate an actual block
		function = config.GetCompressionFunction(CompressionType::COMPRESSION_CONSTANT, type.InternalType());
	} else {
		function = config.GetCompressionFunction(compression_type, type.InternalType());
		block = block_manager.RegisterBlock(block_id);
	}
	auto segment_size = Storage::BLOCK_SIZE;
	return make_unique<ColumnSegment>(db, std::move(block), type, ColumnSegmentType::PERSISTENT, start, count, function,
	                                  std::move(statistics), block_id, offset, segment_size);
}

unique_ptr<ColumnSegment> ColumnSegment::CreateTransientSegment(DatabaseInstance &db, const LogicalType &type,
                                                                idx_t start, idx_t segment_size) {
	auto &config = DBConfig::GetConfig(db);
	auto function = config.GetCompressionFunction(CompressionType::COMPRESSION_UNCOMPRESSED, type.InternalType());
	auto &buffer_manager = BufferManager::GetBufferManager(db);
	shared_ptr<BlockHandle> block;
	// transient: allocate a buffer for the uncompressed segment
	if (segment_size < Storage::BLOCK_SIZE) {
		block = buffer_manager.RegisterSmallMemory(segment_size);
	} else {
		buffer_manager.Allocate(segment_size, false, &block);
	}
	return make_unique<ColumnSegment>(db, std::move(block), type, ColumnSegmentType::TRANSIENT, start, 0, function,
	                                  nullptr, INVALID_BLOCK, 0, segment_size);
}

unique_ptr<ColumnSegment> ColumnSegment::CreateSegment(ColumnSegment &other, idx_t start) {
	return make_unique<ColumnSegment>(other, start);
}

ColumnSegment::ColumnSegment(DatabaseInstance &db, shared_ptr<BlockHandle> block, LogicalType type_p,
                             ColumnSegmentType segment_type, idx_t start, idx_t count, CompressionFunction *function_p,
                             unique_ptr<BaseStatistics> statistics, block_id_t block_id_p, idx_t offset_p,
                             idx_t segment_size_p)
    : SegmentBase(start, count), db(db), type(std::move(type_p)), type_size(GetTypeIdSize(type.InternalType())),
      segment_type(segment_type), function(function_p), stats(type, std::move(statistics)), block(std::move(block)),
      block_id(block_id_p), offset(offset_p), segment_size(segment_size_p) {
	D_ASSERT(function);
	if (function->init_segment) {
		segment_state = function->init_segment(*this, block_id);
	}
}

ColumnSegment::ColumnSegment(ColumnSegment &other, idx_t start)
    : SegmentBase(start, other.count), db(other.db), type(std::move(other.type)), type_size(other.type_size),
      segment_type(other.segment_type), function(other.function), stats(std::move(other.stats)),
      block(std::move(other.block)), block_id(other.block_id), offset(other.offset), segment_size(other.segment_size),
      segment_state(std::move(other.segment_state)) {
}

ColumnSegment::~ColumnSegment() {
}

//===--------------------------------------------------------------------===//
// Scan
//===--------------------------------------------------------------------===//
void ColumnSegment::InitializeScan(ColumnScanState &state) {
	state.scan_state = function->init_scan(*this);
}

void ColumnSegment::Scan(ColumnScanState &state, idx_t scan_count, Vector &result, idx_t result_offset,
                         bool entire_vector) {
	if (entire_vector) {
		D_ASSERT(result_offset == 0);
		Scan(state, scan_count, result);
	} else {
		D_ASSERT(result.GetVectorType() == VectorType::FLAT_VECTOR);
		ScanPartial(state, scan_count, result, result_offset);
		D_ASSERT(result.GetVectorType() == VectorType::FLAT_VECTOR);
	}
}

void ColumnSegment::Skip(ColumnScanState &state) {
	function->skip(*this, state, state.row_index - state.internal_index);
	state.internal_index = state.row_index;
}

void ColumnSegment::Scan(ColumnScanState &state, idx_t scan_count, Vector &result) {
	function->scan_vector(*this, state, scan_count, result);
}

void ColumnSegment::ScanPartial(ColumnScanState &state, idx_t scan_count, Vector &result, idx_t result_offset) {
	function->scan_partial(*this, state, scan_count, result, result_offset);
}

//===--------------------------------------------------------------------===//
// Fetch
//===--------------------------------------------------------------------===//
void ColumnSegment::FetchRow(ColumnFetchState &state, row_t row_id, Vector &result, idx_t result_idx) {
	function->fetch_row(*this, state, row_id - this->start, result, result_idx);
}

//===--------------------------------------------------------------------===//
// Append
//===--------------------------------------------------------------------===//
idx_t ColumnSegment::SegmentSize() const {
	return segment_size;
}

void ColumnSegment::Resize(idx_t new_size) {
	D_ASSERT(new_size > this->segment_size);
	D_ASSERT(offset == 0);
	auto &buffer_manager = BufferManager::GetBufferManager(db);
	auto old_handle = buffer_manager.Pin(block);
	shared_ptr<BlockHandle> new_block;
	auto new_handle = buffer_manager.Allocate(Storage::BLOCK_SIZE, false, &new_block);
	memcpy(new_handle.Ptr(), old_handle.Ptr(), segment_size);
	this->block_id = new_block->BlockId();
	this->block = std::move(new_block);
	this->segment_size = new_size;
}

void ColumnSegment::InitializeAppend(ColumnAppendState &state) {
	D_ASSERT(segment_type == ColumnSegmentType::TRANSIENT);
	if (!function->init_append) {
		throw InternalException("Attempting to init append to a segment without init_append method");
	}
	state.append_state = function->init_append(*this);
}

idx_t ColumnSegment::Append(ColumnAppendState &state, UnifiedVectorFormat &append_data, idx_t offset, idx_t count) {
	D_ASSERT(segment_type == ColumnSegmentType::TRANSIENT);
	if (!function->append) {
		throw InternalException("Attempting to append to a segment without append method");
	}
	return function->append(*state.append_state, *this, stats, append_data, offset, count);
}

idx_t ColumnSegment::FinalizeAppend(ColumnAppendState &state) {
	D_ASSERT(segment_type == ColumnSegmentType::TRANSIENT);
	if (!function->finalize_append) {
		throw InternalException("Attempting to call FinalizeAppend on a segment without a finalize_append method");
	}
	auto result_count = function->finalize_append(*this, stats);
	state.append_state.reset();
	return result_count;
}

void ColumnSegment::RevertAppend(idx_t start_row) {
	D_ASSERT(segment_type == ColumnSegmentType::TRANSIENT);
	if (function->revert_append) {
		function->revert_append(*this, start_row);
	}
	this->count = start_row - this->start;
}

//===--------------------------------------------------------------------===//
// Convert To Persistent
//===--------------------------------------------------------------------===//
void ColumnSegment::ConvertToPersistent(BlockManager *block_manager, block_id_t block_id_p) {
	D_ASSERT(segment_type == ColumnSegmentType::TRANSIENT);
	segment_type = ColumnSegmentType::PERSISTENT;

	block_id = block_id_p;
	offset = 0;

	D_ASSERT(stats.statistics);
	if (block_id == INVALID_BLOCK) {
		// constant block: reset the block buffer
		D_ASSERT(stats.statistics->IsConstant());
		block.reset();
	} else {
		D_ASSERT(!stats.statistics->IsConstant());
		// non-constant block: write the block to disk
		// the data for the block already exists in-memory of our block
		// instead of copying the data we alter some metadata so the buffer points to an on-disk block
		block = block_manager->ConvertToPersistent(block_id, std::move(block));
	}

	segment_state.reset();
	if (function->init_segment) {
		segment_state = function->init_segment(*this, block_id);
	}
}

void ColumnSegment::MarkAsPersistent(shared_ptr<BlockHandle> block_p, uint32_t offset_p) {
	D_ASSERT(segment_type == ColumnSegmentType::TRANSIENT);
	segment_type = ColumnSegmentType::PERSISTENT;

	block_id = block_p->BlockId();
	offset = offset_p;
	block = std::move(block_p);

	segment_state.reset();
	if (function->init_segment) {
		segment_state = function->init_segment(*this, block_id);
	}
}

//===--------------------------------------------------------------------===//
// Filter Selection
//===--------------------------------------------------------------------===//
template <class T, class OP, bool HAS_NULL>
static idx_t TemplatedFilterSelection(T *vec, T *predicate, SelectionVector &sel, idx_t approved_tuple_count,
                                      ValidityMask &mask, SelectionVector &result_sel) {
	idx_t result_count = 0;
	for (idx_t i = 0; i < approved_tuple_count; i++) {
		auto idx = sel.get_index(i);
		if ((!HAS_NULL || mask.RowIsValid(idx)) && OP::Operation(vec[idx], *predicate)) {
			result_sel.set_index(result_count++, idx);
		}
	}
	return result_count;
}

template <class T>
static void FilterSelectionSwitch(T *vec, T *predicate, SelectionVector &sel, idx_t &approved_tuple_count,
                                  ExpressionType comparison_type, ValidityMask &mask) {
	SelectionVector new_sel(approved_tuple_count);
	// the inplace loops take the result as the last parameter
	switch (comparison_type) {
	case ExpressionType::COMPARE_EQUAL: {
		if (mask.AllValid()) {
			approved_tuple_count =
			    TemplatedFilterSelection<T, Equals, false>(vec, predicate, sel, approved_tuple_count, mask, new_sel);
		} else {
			approved_tuple_count =
			    TemplatedFilterSelection<T, Equals, true>(vec, predicate, sel, approved_tuple_count, mask, new_sel);
		}
		break;
	}
	case ExpressionType::COMPARE_NOTEQUAL: {
		if (mask.AllValid()) {
			approved_tuple_count =
			    TemplatedFilterSelection<T, NotEquals, false>(vec, predicate, sel, approved_tuple_count, mask, new_sel);
		} else {
			approved_tuple_count =
			    TemplatedFilterSelection<T, NotEquals, true>(vec, predicate, sel, approved_tuple_count, mask, new_sel);
		}
		break;
	}
	case ExpressionType::COMPARE_LESSTHAN: {
		if (mask.AllValid()) {
			approved_tuple_count =
			    TemplatedFilterSelection<T, LessThan, false>(vec, predicate, sel, approved_tuple_count, mask, new_sel);
		} else {
			approved_tuple_count =
			    TemplatedFilterSelection<T, LessThan, true>(vec, predicate, sel, approved_tuple_count, mask, new_sel);
		}
		break;
	}
	case ExpressionType::COMPARE_GREATERTHAN: {
		if (mask.AllValid()) {
			approved_tuple_count = TemplatedFilterSelection<T, GreaterThan, false>(vec, predicate, sel,
			                                                                       approved_tuple_count, mask, new_sel);
		} else {
			approved_tuple_count = TemplatedFilterSelection<T, GreaterThan, true>(vec, predicate, sel,
			                                                                      approved_tuple_count, mask, new_sel);
		}
		break;
	}
	case ExpressionType::COMPARE_LESSTHANOREQUALTO: {
		if (mask.AllValid()) {
			approved_tuple_count = TemplatedFilterSelection<T, LessThanEquals, false>(
			    vec, predicate, sel, approved_tuple_count, mask, new_sel);
		} else {
			approved_tuple_count = TemplatedFilterSelection<T, LessThanEquals, true>(
			    vec, predicate, sel, approved_tuple_count, mask, new_sel);
		}
		break;
	}
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO: {
		if (mask.AllValid()) {
			approved_tuple_count = TemplatedFilterSelection<T, GreaterThanEquals, false>(
			    vec, predicate, sel, approved_tuple_count, mask, new_sel);
		} else {
			approved_tuple_count = TemplatedFilterSelection<T, GreaterThanEquals, true>(
			    vec, predicate, sel, approved_tuple_count, mask, new_sel);
		}
		break;
	}
	default:
		throw NotImplementedException("Unknown comparison type for filter pushed down to table!");
	}
	sel.Initialize(new_sel);
}

template <bool IS_NULL>
static idx_t TemplatedNullSelection(SelectionVector &sel, idx_t &approved_tuple_count, ValidityMask &mask) {
	if (mask.AllValid()) {
		// no NULL values
		if (IS_NULL) {
			approved_tuple_count = 0;
			return 0;
		} else {
			return approved_tuple_count;
		}
	} else {
		SelectionVector result_sel(approved_tuple_count);
		idx_t result_count = 0;
		for (idx_t i = 0; i < approved_tuple_count; i++) {
			auto idx = sel.get_index(i);
			if (mask.RowIsValid(idx) != IS_NULL) {
				result_sel.set_index(result_count++, idx);
			}
		}
		sel.Initialize(result_sel);
		approved_tuple_count = result_count;
		return result_count;
	}
}

idx_t ColumnSegment::FilterSelection(SelectionVector &sel, Vector &result, const TableFilter &filter,
                                     idx_t &approved_tuple_count, ValidityMask &mask) {
	switch (filter.filter_type) {
	case TableFilterType::CONJUNCTION_OR: {
		// similar to the CONJUNCTION_AND, but we need to take care of the SelectionVectors (OR all of them)
		idx_t count_total = 0;
		SelectionVector result_sel(approved_tuple_count);
		auto &conjunction_or = (const ConjunctionOrFilter &)filter;
		for (auto &child_filter : conjunction_or.child_filters) {
			SelectionVector temp_sel;
			temp_sel.Initialize(sel);
			idx_t temp_tuple_count = approved_tuple_count;
			idx_t temp_count = FilterSelection(temp_sel, result, *child_filter, temp_tuple_count, mask);
			// tuples passed, move them into the actual result vector
			for (idx_t i = 0; i < temp_count; i++) {
				auto new_idx = temp_sel.get_index(i);
				bool is_new_idx = true;
				for (idx_t res_idx = 0; res_idx < count_total; res_idx++) {
					if (result_sel.get_index(res_idx) == new_idx) {
						is_new_idx = false;
						break;
					}
				}
				if (is_new_idx) {
					result_sel.set_index(count_total++, new_idx);
				}
			}
		}
		sel.Initialize(result_sel);
		approved_tuple_count = count_total;
		return approved_tuple_count;
	}
	case TableFilterType::CONJUNCTION_AND: {
		auto &conjunction_and = (const ConjunctionAndFilter &)filter;
		for (auto &child_filter : conjunction_and.child_filters) {
			FilterSelection(sel, result, *child_filter, approved_tuple_count, mask);
		}
		return approved_tuple_count;
	}
	case TableFilterType::CONSTANT_COMPARISON: {
		auto &constant_filter = (const ConstantFilter &)filter;
		// the inplace loops take the result as the last parameter
		switch (result.GetType().InternalType()) {
		case PhysicalType::UINT8: {
			auto result_flat = FlatVector::GetData<uint8_t>(result);
			Vector predicate_vector(constant_filter.constant);
			auto predicate = FlatVector::GetData<uint8_t>(predicate_vector);
			FilterSelectionSwitch<uint8_t>(result_flat, predicate, sel, approved_tuple_count,
			                               constant_filter.comparison_type, mask);
			break;
		}
		case PhysicalType::UINT16: {
			auto result_flat = FlatVector::GetData<uint16_t>(result);
			Vector predicate_vector(constant_filter.constant);
			auto predicate = FlatVector::GetData<uint16_t>(predicate_vector);
			FilterSelectionSwitch<uint16_t>(result_flat, predicate, sel, approved_tuple_count,
			                                constant_filter.comparison_type, mask);
			break;
		}
		case PhysicalType::UINT32: {
			auto result_flat = FlatVector::GetData<uint32_t>(result);
			Vector predicate_vector(constant_filter.constant);
			auto predicate = FlatVector::GetData<uint32_t>(predicate_vector);
			FilterSelectionSwitch<uint32_t>(result_flat, predicate, sel, approved_tuple_count,
			                                constant_filter.comparison_type, mask);
			break;
		}
		case PhysicalType::UINT64: {
			auto result_flat = FlatVector::GetData<uint64_t>(result);
			Vector predicate_vector(constant_filter.constant);
			auto predicate = FlatVector::GetData<uint64_t>(predicate_vector);
			FilterSelectionSwitch<uint64_t>(result_flat, predicate, sel, approved_tuple_count,
			                                constant_filter.comparison_type, mask);
			break;
		}
		case PhysicalType::INT8: {
			auto result_flat = FlatVector::GetData<int8_t>(result);
			Vector predicate_vector(constant_filter.constant);
			auto predicate = FlatVector::GetData<int8_t>(predicate_vector);
			FilterSelectionSwitch<int8_t>(result_flat, predicate, sel, approved_tuple_count,
			                              constant_filter.comparison_type, mask);
			break;
		}
		case PhysicalType::INT16: {
			auto result_flat = FlatVector::GetData<int16_t>(result);
			Vector predicate_vector(constant_filter.constant);
			auto predicate = FlatVector::GetData<int16_t>(predicate_vector);
			FilterSelectionSwitch<int16_t>(result_flat, predicate, sel, approved_tuple_count,
			                               constant_filter.comparison_type, mask);
			break;
		}
		case PhysicalType::INT32: {
			auto result_flat = FlatVector::GetData<int32_t>(result);
			Vector predicate_vector(constant_filter.constant);
			auto predicate = FlatVector::GetData<int32_t>(predicate_vector);
			FilterSelectionSwitch<int32_t>(result_flat, predicate, sel, approved_tuple_count,
			                               constant_filter.comparison_type, mask);
			break;
		}
		case PhysicalType::INT64: {
			auto result_flat = FlatVector::GetData<int64_t>(result);
			Vector predicate_vector(constant_filter.constant);
			auto predicate = FlatVector::GetData<int64_t>(predicate_vector);
			FilterSelectionSwitch<int64_t>(result_flat, predicate, sel, approved_tuple_count,
			                               constant_filter.comparison_type, mask);
			break;
		}
		case PhysicalType::INT128: {
			auto result_flat = FlatVector::GetData<hugeint_t>(result);
			Vector predicate_vector(constant_filter.constant);
			auto predicate = FlatVector::GetData<hugeint_t>(predicate_vector);
			FilterSelectionSwitch<hugeint_t>(result_flat, predicate, sel, approved_tuple_count,
			                                 constant_filter.comparison_type, mask);
			break;
		}
		case PhysicalType::FLOAT: {
			auto result_flat = FlatVector::GetData<float>(result);
			Vector predicate_vector(constant_filter.constant);
			auto predicate = FlatVector::GetData<float>(predicate_vector);
			FilterSelectionSwitch<float>(result_flat, predicate, sel, approved_tuple_count,
			                             constant_filter.comparison_type, mask);
			break;
		}
		case PhysicalType::DOUBLE: {
			auto result_flat = FlatVector::GetData<double>(result);
			Vector predicate_vector(constant_filter.constant);
			auto predicate = FlatVector::GetData<double>(predicate_vector);
			FilterSelectionSwitch<double>(result_flat, predicate, sel, approved_tuple_count,
			                              constant_filter.comparison_type, mask);
			break;
		}
		case PhysicalType::VARCHAR: {
			auto result_flat = FlatVector::GetData<string_t>(result);
			Vector predicate_vector(constant_filter.constant);
			auto predicate = FlatVector::GetData<string_t>(predicate_vector);
			FilterSelectionSwitch<string_t>(result_flat, predicate, sel, approved_tuple_count,
			                                constant_filter.comparison_type, mask);
			break;
		}
		case PhysicalType::BOOL: {
			auto result_flat = FlatVector::GetData<bool>(result);
			Vector predicate_vector(constant_filter.constant);
			auto predicate = FlatVector::GetData<bool>(predicate_vector);
			FilterSelectionSwitch<bool>(result_flat, predicate, sel, approved_tuple_count,
			                            constant_filter.comparison_type, mask);
			break;
		}
		default:
			throw InvalidTypeException(result.GetType(), "Invalid type for filter pushed down to table comparison");
		}
		return approved_tuple_count;
	}
	case TableFilterType::IS_NULL:
		return TemplatedNullSelection<true>(sel, approved_tuple_count, mask);
	case TableFilterType::IS_NOT_NULL:
		return TemplatedNullSelection<false>(sel, approved_tuple_count, mask);
	default:
		throw InternalException("FIXME: unsupported type for filter selection");
	}
}

} // namespace duckdb




namespace duckdb {

ListColumnData::ListColumnData(BlockManager &block_manager, DataTableInfo &info, idx_t column_index, idx_t start_row,
                               LogicalType type_p, ColumnData *parent)
    : ColumnData(block_manager, info, column_index, start_row, std::move(type_p), parent),
      validity(block_manager, info, 0, start_row, this) {
	D_ASSERT(type.InternalType() == PhysicalType::LIST);
	auto &child_type = ListType::GetChildType(type);
	// the child column, with column index 1 (0 is the validity mask)
	child_column = ColumnData::CreateColumnUnique(block_manager, info, 1, start_row, child_type, this);
}

ListColumnData::ListColumnData(ColumnData &original, idx_t start_row, ColumnData *parent)
    : ColumnData(original, start_row, parent), validity(((ListColumnData &)original).validity, start_row, this) {
	auto &list_data = (ListColumnData &)original;
	child_column = ColumnData::CreateColumnUnique(*list_data.child_column, start_row, this);
}

bool ListColumnData::CheckZonemap(ColumnScanState &state, TableFilter &filter) {
	// table filters are not supported yet for list columns
	return false;
}

void ListColumnData::InitializeScan(ColumnScanState &state) {
	ColumnData::InitializeScan(state);

	// initialize the validity segment
	ColumnScanState validity_state;
	validity.InitializeScan(validity_state);
	state.child_states.push_back(std::move(validity_state));

	// initialize the child scan
	ColumnScanState child_state;
	child_column->InitializeScan(child_state);
	state.child_states.push_back(std::move(child_state));
}

list_entry_t ListColumnData::FetchListEntry(idx_t row_idx) {
	auto segment = (ColumnSegment *)data.GetSegment(row_idx);
	ColumnFetchState fetch_state;
	Vector result(type, 1);
	segment->FetchRow(fetch_state, row_idx, result, 0);

	// initialize the child scan with the required offset
	auto list_data = FlatVector::GetData<list_entry_t>(result);
	return list_data[0];
}

void ListColumnData::InitializeScanWithOffset(ColumnScanState &state, idx_t row_idx) {
	if (row_idx == 0) {
		InitializeScan(state);
		return;
	}
	ColumnData::InitializeScanWithOffset(state, row_idx);

	// initialize the validity segment
	ColumnScanState validity_state;
	validity.InitializeScanWithOffset(validity_state, row_idx);
	state.child_states.push_back(std::move(validity_state));

	// we need to read the list at position row_idx to get the correct row offset of the child
	auto list_entry = FetchListEntry(row_idx);
	auto child_offset = list_entry.offset;

	D_ASSERT(child_offset <= child_column->GetMaxEntry());
	ColumnScanState child_state;
	if (child_offset < child_column->GetMaxEntry()) {
		child_column->InitializeScanWithOffset(child_state, start + child_offset);
	}
	state.child_states.push_back(std::move(child_state));
}

idx_t ListColumnData::Scan(TransactionData transaction, idx_t vector_index, ColumnScanState &state, Vector &result) {
	return ScanCount(state, result, STANDARD_VECTOR_SIZE);
}

idx_t ListColumnData::ScanCommitted(idx_t vector_index, ColumnScanState &state, Vector &result, bool allow_updates) {
	return ScanCount(state, result, STANDARD_VECTOR_SIZE);
}

idx_t ListColumnData::ScanCount(ColumnScanState &state, Vector &result, idx_t count) {
	if (count == 0) {
		return 0;
	}
	// updates not supported for lists
	D_ASSERT(!updates);

	idx_t scan_count = ScanVector(state, result, count);
	D_ASSERT(scan_count > 0);
	validity.ScanCount(state.child_states[0], result, count);

	auto data = FlatVector::GetData<list_entry_t>(result);
	auto first_entry = data[0];
	auto last_entry = data[scan_count - 1];

#ifdef DEBUG
	for (idx_t i = 1; i < scan_count; i++) {
		D_ASSERT(data[i].offset == data[i - 1].offset + data[i - 1].length);
	}
#endif
	// shift all offsets so they are 0 at the first entry
	for (idx_t i = 0; i < scan_count; i++) {
		data[i].offset -= first_entry.offset;
	}

	D_ASSERT(last_entry.offset >= first_entry.offset);
	idx_t child_scan_count = last_entry.offset + last_entry.length - first_entry.offset;
	ListVector::Reserve(result, child_scan_count);

	if (child_scan_count > 0) {
		auto &child_entry = ListVector::GetEntry(result);
		D_ASSERT(child_entry.GetType().InternalType() == PhysicalType::STRUCT ||
		         state.child_states[1].row_index + child_scan_count <=
		             child_column->start + child_column->GetMaxEntry());
		child_column->ScanCount(state.child_states[1], child_entry, child_scan_count);
	}

	ListVector::SetListSize(result, child_scan_count);
	return scan_count;
}

void ListColumnData::Skip(ColumnScanState &state, idx_t count) {
	// skip inside the validity segment
	validity.Skip(state.child_states[0], count);

	// we need to read the list entries/offsets to figure out how much to skip
	// note that we only need to read the first and last entry
	// however, let's just read all "count" entries for now
	auto data = unique_ptr<list_entry_t[]>(new list_entry_t[count]);
	Vector result(type, (data_ptr_t)data.get());
	idx_t scan_count = ScanVector(state, result, count);
	if (scan_count == 0) {
		return;
	}

	auto &first_entry = data[0];
	auto &last_entry = data[scan_count - 1];
	idx_t child_scan_count = last_entry.offset + last_entry.length - first_entry.offset;
	if (child_scan_count == 0) {
		return;
	}

	// skip the child state forward by the child_scan_count
	child_column->Skip(state.child_states[1], child_scan_count);
}

void ListColumnData::InitializeAppend(ColumnAppendState &state) {
	// initialize the list offset append
	ColumnData::InitializeAppend(state);

	// initialize the validity append
	ColumnAppendState validity_append_state;
	validity.InitializeAppend(validity_append_state);
	state.child_appends.push_back(std::move(validity_append_state));

	// initialize the child column append
	ColumnAppendState child_append_state;
	child_column->InitializeAppend(child_append_state);
	state.child_appends.push_back(std::move(child_append_state));
}

void ListColumnData::Append(BaseStatistics &stats_p, ColumnAppendState &state, Vector &vector, idx_t count) {
	D_ASSERT(count > 0);
	auto &stats = (ListStatistics &)stats_p;

	UnifiedVectorFormat list_data;
	vector.ToUnifiedFormat(count, list_data);
	auto &list_validity = list_data.validity;

	// construct the list_entry_t entries to append to the column data
	auto input_offsets = (list_entry_t *)list_data.data;
	auto start_offset = child_column->GetMaxEntry();
	idx_t child_count = 0;

	ValidityMask append_mask(count);
	auto append_offsets = unique_ptr<list_entry_t[]>(new list_entry_t[count]);
	bool child_contiguous = false;
	for (idx_t i = 0; i < count; i++) {
		auto input_idx = list_data.sel->get_index(i);
		if (list_validity.RowIsValid(input_idx)) {
			auto &input_list = input_offsets[input_idx];
			if (input_list.offset != child_count) {
				child_contiguous = false;
			}
			append_offsets[i].offset = start_offset + child_count;
			append_offsets[i].length = input_list.length;
			child_count += input_list.length;
		} else {
			append_mask.SetInvalid(i);
			if (i > 0) {
				append_offsets[i].offset = append_offsets[i - 1].offset + append_offsets[i - 1].length;
			} else {
				append_offsets[i].offset = start_offset;
			}
			append_offsets[i].length = 0;
		}
	}
	auto &list_child = ListVector::GetEntry(vector);
	Vector child_vector(list_child);
	if (!child_contiguous) {
		// if the child of the list vector is a non-contiguous vector (i.e. list elements are repeating or have gaps)
		// we first push a selection vector and flatten the child vector to turn it into a contiguous vector
		SelectionVector child_sel(child_count);
		idx_t current_count = 0;
		for (idx_t i = 0; i < count; i++) {
			auto input_idx = list_data.sel->get_index(i);
			if (list_validity.RowIsValid(input_idx)) {
				auto &input_list = input_offsets[input_idx];
				for (idx_t list_idx = 0; list_idx < input_list.length; list_idx++) {
					child_sel.set_index(current_count++, input_list.offset + list_idx);
				}
			}
		}
		D_ASSERT(current_count == child_count);
		child_vector.Slice(list_child, child_sel, child_count);
	}
#ifdef DEBUG
	D_ASSERT(append_offsets[0].offset == start_offset);
	for (idx_t i = 1; i < count; i++) {
		D_ASSERT(append_offsets[i].offset == append_offsets[i - 1].offset + append_offsets[i - 1].length);
	}
	D_ASSERT(append_offsets[count - 1].offset + append_offsets[count - 1].length - append_offsets[0].offset ==
	         child_count);
#endif

	UnifiedVectorFormat vdata;
	vdata.validity = append_mask;
	vdata.sel = FlatVector::IncrementalSelectionVector();
	vdata.data = (data_ptr_t)append_offsets.get();

	// append the list offsets
	ColumnData::AppendData(stats, state, vdata, count);
	// append the validity data
	validity.AppendData(*stats.validity_stats, state.child_appends[0], vdata, count);
	// append the child vector
	if (child_count > 0) {
		child_column->Append(*stats.child_stats, state.child_appends[1], child_vector, child_count);
	}
}

void ListColumnData::RevertAppend(row_t start_row) {
	ColumnData::RevertAppend(start_row);
	validity.RevertAppend(start_row);
	auto column_count = GetMaxEntry();
	if (column_count > start) {
		// revert append in the child column
		auto list_entry = FetchListEntry(column_count - 1);
		child_column->RevertAppend(list_entry.offset + list_entry.length);
	}
}

idx_t ListColumnData::Fetch(ColumnScanState &state, row_t row_id, Vector &result) {
	throw NotImplementedException("List Fetch");
}

void ListColumnData::Update(TransactionData transaction, idx_t column_index, Vector &update_vector, row_t *row_ids,
                            idx_t update_count) {
	throw NotImplementedException("List Update is not supported.");
}

void ListColumnData::UpdateColumn(TransactionData transaction, const vector<column_t> &column_path,
                                  Vector &update_vector, row_t *row_ids, idx_t update_count, idx_t depth) {
	throw NotImplementedException("List Update Column is not supported");
}

unique_ptr<BaseStatistics> ListColumnData::GetUpdateStatistics() {
	return nullptr;
}

void ListColumnData::FetchRow(TransactionData transaction, ColumnFetchState &state, row_t row_id, Vector &result,
                              idx_t result_idx) {
	// insert any child states that are required
	// we need two (validity & list child)
	// note that we need a scan state for the child vector
	// this is because we will (potentially) fetch more than one tuple from the list child
	if (state.child_states.empty()) {
		auto child_state = make_unique<ColumnFetchState>();
		state.child_states.push_back(std::move(child_state));
	}
	// fetch the list_entry_t and the validity mask for that list
	auto segment = (ColumnSegment *)data.GetSegment(row_id);

	// now perform the fetch within the segment
	segment->FetchRow(state, row_id, result, result_idx);
	validity.FetchRow(transaction, *state.child_states[0], row_id, result, result_idx);

	auto &validity = FlatVector::Validity(result);
	auto list_data = FlatVector::GetData<list_entry_t>(result);
	auto &list_entry = list_data[result_idx];
	auto original_offset = list_entry.offset;
	// set the list entry offset to the size of the current list
	list_entry.offset = ListVector::GetListSize(result);
	if (!validity.RowIsValid(result_idx)) {
		// the list is NULL! no need to fetch the child
		D_ASSERT(list_entry.length == 0);
		return;
	}

	// now we need to read from the child all the elements between [offset...length]
	auto child_scan_count = list_entry.length;
	if (child_scan_count > 0) {
		auto child_state = make_unique<ColumnScanState>();
		auto &child_type = ListType::GetChildType(result.GetType());
		Vector child_scan(child_type, child_scan_count);
		// seek the scan towards the specified position and read [length] entries
		child_column->InitializeScanWithOffset(*child_state, start + original_offset);
		D_ASSERT(child_type.InternalType() == PhysicalType::STRUCT ||
		         child_state->row_index + child_scan_count - this->start <= child_column->GetMaxEntry());
		child_column->ScanCount(*child_state, child_scan, child_scan_count);

		ListVector::Append(result, child_scan, child_scan_count);
	}
}

void ListColumnData::CommitDropColumn() {
	validity.CommitDropColumn();
	child_column->CommitDropColumn();
}

struct ListColumnCheckpointState : public ColumnCheckpointState {
	ListColumnCheckpointState(RowGroup &row_group, ColumnData &column_data, PartialBlockManager &partial_block_manager)
	    : ColumnCheckpointState(row_group, column_data, partial_block_manager) {
		global_stats = make_unique<ListStatistics>(column_data.type);
	}

	unique_ptr<ColumnCheckpointState> validity_state;
	unique_ptr<ColumnCheckpointState> child_state;

public:
	unique_ptr<BaseStatistics> GetStatistics() override {
		auto stats = global_stats->Copy();
		auto &list_stats = (ListStatistics &)*stats;
		stats->validity_stats = validity_state->GetStatistics();
		list_stats.child_stats = child_state->GetStatistics();
		return stats;
	}

	void WriteDataPointers(RowGroupWriter &writer) override {
		ColumnCheckpointState::WriteDataPointers(writer);
		validity_state->WriteDataPointers(writer);
		child_state->WriteDataPointers(writer);
	}
	void GetBlockIds(unordered_set<block_id_t> &result) override {
		ColumnCheckpointState::GetBlockIds(result);
		validity_state->GetBlockIds(result);
		child_state->GetBlockIds(result);
	}
};

unique_ptr<ColumnCheckpointState> ListColumnData::CreateCheckpointState(RowGroup &row_group,
                                                                        PartialBlockManager &partial_block_manager) {
	return make_unique<ListColumnCheckpointState>(row_group, *this, partial_block_manager);
}

unique_ptr<ColumnCheckpointState> ListColumnData::Checkpoint(RowGroup &row_group,
                                                             PartialBlockManager &partial_block_manager,
                                                             ColumnCheckpointInfo &checkpoint_info) {
	auto validity_state = validity.Checkpoint(row_group, partial_block_manager, checkpoint_info);
	auto base_state = ColumnData::Checkpoint(row_group, partial_block_manager, checkpoint_info);
	auto child_state = child_column->Checkpoint(row_group, partial_block_manager, checkpoint_info);

	auto &checkpoint_state = (ListColumnCheckpointState &)*base_state;
	checkpoint_state.validity_state = std::move(validity_state);
	checkpoint_state.child_state = std::move(child_state);
	return base_state;
}

void ListColumnData::DeserializeColumn(Deserializer &source) {
	ColumnData::DeserializeColumn(source);
	validity.DeserializeColumn(source);
	child_column->DeserializeColumn(source);
}

void ListColumnData::GetStorageInfo(idx_t row_group_index, vector<idx_t> col_path, TableStorageInfo &result) {
	col_path.push_back(0);
	validity.GetStorageInfo(row_group_index, col_path, result);
	col_path.back() = 1;
	child_column->GetStorageInfo(row_group_index, col_path, result);
}

} // namespace duckdb



namespace duckdb {

PersistentTableData::PersistentTableData(idx_t column_count) {
}

PersistentTableData::~PersistentTableData() {
}

} // namespace duckdb


















namespace duckdb {

constexpr const idx_t RowGroup::ROW_GROUP_VECTOR_COUNT;
constexpr const idx_t RowGroup::ROW_GROUP_SIZE;

RowGroup::RowGroup(AttachedDatabase &db, BlockManager &block_manager, DataTableInfo &table_info, idx_t start,
                   idx_t count)
    : SegmentBase(start, count), db(db), block_manager(block_manager), table_info(table_info) {

	Verify();
}

RowGroup::RowGroup(AttachedDatabase &db, BlockManager &block_manager, DataTableInfo &table_info,
                   const vector<LogicalType> &types, RowGroupPointer &&pointer)
    : SegmentBase(pointer.row_start, pointer.tuple_count), db(db), block_manager(block_manager),
      table_info(table_info) {
	// deserialize the columns
	if (pointer.data_pointers.size() != types.size()) {
		throw IOException("Row group column count is unaligned with table column count. Corrupt file?");
	}
	for (idx_t i = 0; i < pointer.data_pointers.size(); i++) {
		auto &block_pointer = pointer.data_pointers[i];
		MetaBlockReader column_data_reader(block_manager, block_pointer.block_id);
		column_data_reader.offset = block_pointer.offset;
		this->columns.push_back(
		    ColumnData::Deserialize(block_manager, table_info, i, start, column_data_reader, types[i], nullptr));
	}

	// set up the statistics
	for (auto &stats : pointer.statistics) {
		auto stats_type = stats->type;
		this->stats.push_back(make_shared<SegmentStatistics>(stats_type, std::move(stats)));
	}
	this->version_info = std::move(pointer.versions);

	Verify();
}

RowGroup::RowGroup(RowGroup &row_group, idx_t start)
    : SegmentBase(start, row_group.count), db(row_group.db), block_manager(row_group.block_manager),
      table_info(row_group.table_info), version_info(std::move(row_group.version_info)),
      stats(std::move(row_group.stats)) {
	for (auto &column : row_group.columns) {
		this->columns.push_back(ColumnData::CreateColumn(*column, start));
	}
	if (version_info) {
		version_info->SetStart(start);
	}
	Verify();
}

void VersionNode::SetStart(idx_t start) {
	idx_t current_start = start;
	for (idx_t i = 0; i < RowGroup::ROW_GROUP_VECTOR_COUNT; i++) {
		if (info[i]) {
			info[i]->start = current_start;
		}
		current_start += STANDARD_VECTOR_SIZE;
	}
}

RowGroup::~RowGroup() {
}

DatabaseInstance &RowGroup::GetDatabase() {
	return db.GetDatabase();
}

void RowGroup::InitializeEmpty(const vector<LogicalType> &types) {
	// set up the segment trees for the column segments
	for (idx_t i = 0; i < types.size(); i++) {
		auto column_data = ColumnData::CreateColumn(block_manager, GetTableInfo(), i, start, types[i]);
		stats.push_back(make_shared<SegmentStatistics>(types[i]));
		columns.push_back(std::move(column_data));
	}
}

bool RowGroup::InitializeScanWithOffset(RowGroupScanState &state, idx_t vector_offset) {
	auto &column_ids = state.GetColumnIds();
	auto filters = state.GetFilters();
	auto parent_max_row = state.GetParentMaxRow();
	if (filters) {
		if (!CheckZonemap(*filters, column_ids)) {
			return false;
		}
	}

	state.row_group = this;
	state.vector_index = vector_offset;
	state.max_row = this->start > parent_max_row ? 0 : MinValue<idx_t>(this->count, parent_max_row - this->start);
	state.column_scans = unique_ptr<ColumnScanState[]>(new ColumnScanState[column_ids.size()]);
	for (idx_t i = 0; i < column_ids.size(); i++) {
		auto column = column_ids[i];
		if (column != COLUMN_IDENTIFIER_ROW_ID) {
			columns[column]->InitializeScanWithOffset(state.column_scans[i],
			                                          start + vector_offset * STANDARD_VECTOR_SIZE);
		} else {
			state.column_scans[i].current = nullptr;
		}
	}
	return true;
}

bool RowGroup::InitializeScan(RowGroupScanState &state) {
	auto &column_ids = state.GetColumnIds();
	auto filters = state.GetFilters();
	auto parent_max_row = state.GetParentMaxRow();
	if (filters) {
		if (!CheckZonemap(*filters, column_ids)) {
			return false;
		}
	}
	state.row_group = this;
	state.vector_index = 0;
	state.max_row = this->start > parent_max_row ? 0 : MinValue<idx_t>(this->count, parent_max_row - this->start);
	state.column_scans = unique_ptr<ColumnScanState[]>(new ColumnScanState[column_ids.size()]);
	for (idx_t i = 0; i < column_ids.size(); i++) {
		auto column = column_ids[i];
		if (column != COLUMN_IDENTIFIER_ROW_ID) {
			columns[column]->InitializeScan(state.column_scans[i]);
		} else {
			state.column_scans[i].current = nullptr;
		}
	}
	return true;
}

unique_ptr<RowGroup> RowGroup::AlterType(const LogicalType &target_type, idx_t changed_idx,
                                         ExpressionExecutor &executor, RowGroupScanState &scan_state,
                                         DataChunk &scan_chunk) {
	Verify();

	// construct a new column data for this type
	auto column_data = ColumnData::CreateColumn(block_manager, GetTableInfo(), changed_idx, start, target_type);

	ColumnAppendState append_state;
	column_data->InitializeAppend(append_state);

	// scan the original table, and fill the new column with the transformed value
	InitializeScan(scan_state);

	Vector append_vector(target_type);
	auto altered_col_stats = make_shared<SegmentStatistics>(target_type);
	while (true) {
		// scan the table
		scan_chunk.Reset();
		ScanCommitted(scan_state, scan_chunk, TableScanType::TABLE_SCAN_COMMITTED_ROWS);
		if (scan_chunk.size() == 0) {
			break;
		}
		// execute the expression
		executor.ExecuteExpression(scan_chunk, append_vector);
		column_data->Append(*altered_col_stats->statistics, append_state, append_vector, scan_chunk.size());
	}

	// set up the row_group based on this row_group
	auto row_group = make_unique<RowGroup>(db, block_manager, table_info, this->start, this->count);
	row_group->version_info = version_info;
	for (idx_t i = 0; i < columns.size(); i++) {
		if (i == changed_idx) {
			// this is the altered column: use the new column
			row_group->columns.push_back(std::move(column_data));
			row_group->stats.push_back(std::move(altered_col_stats));
		} else {
			// this column was not altered: use the data directly
			row_group->columns.push_back(columns[i]);
			row_group->stats.push_back(stats[i]);
		}
	}
	row_group->Verify();
	return row_group;
}

unique_ptr<RowGroup> RowGroup::AddColumn(ColumnDefinition &new_column, ExpressionExecutor &executor,
                                         Expression *default_value, Vector &result) {
	Verify();

	// construct a new column data for the new column
	auto added_column =
	    ColumnData::CreateColumn(block_manager, GetTableInfo(), columns.size(), start, new_column.Type());
	auto added_col_stats = make_shared<SegmentStatistics>(
	    new_column.Type(), BaseStatistics::CreateEmpty(new_column.Type(), StatisticsType::LOCAL_STATS));

	idx_t rows_to_write = this->count;
	if (rows_to_write > 0) {
		DataChunk dummy_chunk;

		ColumnAppendState state;
		added_column->InitializeAppend(state);
		for (idx_t i = 0; i < rows_to_write; i += STANDARD_VECTOR_SIZE) {
			idx_t rows_in_this_vector = MinValue<idx_t>(rows_to_write - i, STANDARD_VECTOR_SIZE);
			if (default_value) {
				dummy_chunk.SetCardinality(rows_in_this_vector);
				executor.ExecuteExpression(dummy_chunk, result);
			}
			added_column->Append(*added_col_stats->statistics, state, result, rows_in_this_vector);
		}
	}

	// set up the row_group based on this row_group
	auto row_group = make_unique<RowGroup>(db, block_manager, table_info, this->start, this->count);
	row_group->version_info = version_info;
	row_group->columns = columns;
	row_group->stats = stats;
	// now add the new column
	row_group->columns.push_back(std::move(added_column));
	row_group->stats.push_back(std::move(added_col_stats));

	row_group->Verify();
	return row_group;
}

unique_ptr<RowGroup> RowGroup::RemoveColumn(idx_t removed_column) {
	Verify();

	D_ASSERT(removed_column < columns.size());

	auto row_group = make_unique<RowGroup>(db, block_manager, table_info, this->start, this->count);
	row_group->version_info = version_info;
	row_group->columns = columns;
	row_group->stats = stats;
	// now remove the column
	row_group->columns.erase(row_group->columns.begin() + removed_column);
	row_group->stats.erase(row_group->stats.begin() + removed_column);

	row_group->Verify();
	return row_group;
}

void RowGroup::CommitDrop() {
	for (idx_t column_idx = 0; column_idx < columns.size(); column_idx++) {
		CommitDropColumn(column_idx);
	}
}

void RowGroup::CommitDropColumn(idx_t column_idx) {
	D_ASSERT(column_idx < columns.size());
	columns[column_idx]->CommitDropColumn();
}

void RowGroup::NextVector(RowGroupScanState &state) {
	state.vector_index++;
	auto &column_ids = state.GetColumnIds();
	for (idx_t i = 0; i < column_ids.size(); i++) {
		auto column = column_ids[i];
		if (column == COLUMN_IDENTIFIER_ROW_ID) {
			continue;
		}
		D_ASSERT(column < columns.size());
		columns[column]->Skip(state.column_scans[i]);
	}
}

bool RowGroup::CheckZonemap(TableFilterSet &filters, const vector<column_t> &column_ids) {
	for (auto &entry : filters.filters) {
		auto column_index = entry.first;
		auto &filter = entry.second;
		auto base_column_index = column_ids[column_index];

		auto propagate_result = filter->CheckStatistics(*stats[base_column_index]->statistics);
		if (propagate_result == FilterPropagateResult::FILTER_ALWAYS_FALSE ||
		    propagate_result == FilterPropagateResult::FILTER_FALSE_OR_NULL) {
			return false;
		}
	}
	return true;
}

bool RowGroup::CheckZonemapSegments(RowGroupScanState &state) {
	auto &column_ids = state.GetColumnIds();
	auto filters = state.GetFilters();
	if (!filters) {
		return true;
	}
	for (auto &entry : filters->filters) {
		D_ASSERT(entry.first < column_ids.size());
		auto column_idx = entry.first;
		auto base_column_idx = column_ids[column_idx];
		bool read_segment = columns[base_column_idx]->CheckZonemap(state.column_scans[column_idx], *entry.second);
		if (!read_segment) {
			idx_t target_row =
			    state.column_scans[column_idx].current->start + state.column_scans[column_idx].current->count;
			D_ASSERT(target_row >= this->start);
			D_ASSERT(target_row <= this->start + this->count);
			idx_t target_vector_index = (target_row - this->start) / STANDARD_VECTOR_SIZE;
			if (state.vector_index == target_vector_index) {
				// we can't skip any full vectors because this segment contains less than a full vector
				// for now we just bail-out
				// FIXME: we could check if we can ALSO skip the next segments, in which case skipping a full vector
				// might be possible
				// we don't care that much though, since a single segment that fits less than a full vector is
				// exceedingly rare
				return true;
			}
			while (state.vector_index < target_vector_index) {
				NextVector(state);
			}
			return false;
		}
	}

	return true;
}

template <TableScanType TYPE>
void RowGroup::TemplatedScan(TransactionData transaction, RowGroupScanState &state, DataChunk &result) {
	const bool ALLOW_UPDATES = TYPE != TableScanType::TABLE_SCAN_COMMITTED_ROWS_DISALLOW_UPDATES &&
	                           TYPE != TableScanType::TABLE_SCAN_COMMITTED_ROWS_OMIT_PERMANENTLY_DELETED;
	auto table_filters = state.GetFilters();
	auto &column_ids = state.GetColumnIds();
	auto adaptive_filter = state.GetAdaptiveFilter();
	while (true) {
		if (state.vector_index * STANDARD_VECTOR_SIZE >= state.max_row) {
			// exceeded the amount of rows to scan
			return;
		}
		idx_t current_row = state.vector_index * STANDARD_VECTOR_SIZE;
		auto max_count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, state.max_row - current_row);

		//! first check the zonemap if we have to scan this partition
		if (!CheckZonemapSegments(state)) {
			continue;
		}
		// second, scan the version chunk manager to figure out which tuples to load for this transaction
		idx_t count;
		SelectionVector valid_sel(STANDARD_VECTOR_SIZE);
		if (TYPE == TableScanType::TABLE_SCAN_REGULAR) {
			count = state.row_group->GetSelVector(transaction, state.vector_index, valid_sel, max_count);
			if (count == 0) {
				// nothing to scan for this vector, skip the entire vector
				NextVector(state);
				continue;
			}
		} else if (TYPE == TableScanType::TABLE_SCAN_COMMITTED_ROWS_OMIT_PERMANENTLY_DELETED) {
			count = state.row_group->GetCommittedSelVector(transaction.start_time, transaction.transaction_id,
			                                               state.vector_index, valid_sel, max_count);
			if (count == 0) {
				// nothing to scan for this vector, skip the entire vector
				NextVector(state);
				continue;
			}
		} else {
			count = max_count;
		}
		if (count == max_count && !table_filters) {
			// scan all vectors completely: full scan without deletions or table filters
			for (idx_t i = 0; i < column_ids.size(); i++) {
				auto column = column_ids[i];
				if (column == COLUMN_IDENTIFIER_ROW_ID) {
					// scan row id
					D_ASSERT(result.data[i].GetType().InternalType() == ROW_TYPE);
					result.data[i].Sequence(this->start + current_row, 1, count);
				} else {
					if (TYPE != TableScanType::TABLE_SCAN_REGULAR) {
						columns[column]->ScanCommitted(state.vector_index, state.column_scans[i], result.data[i],
						                               ALLOW_UPDATES);
					} else {
						columns[column]->Scan(transaction, state.vector_index, state.column_scans[i], result.data[i]);
					}
				}
			}
		} else {
			// partial scan: we have deletions or table filters
			idx_t approved_tuple_count = count;
			SelectionVector sel;
			if (count != max_count) {
				sel.Initialize(valid_sel);
			} else {
				sel.Initialize(nullptr);
			}
			//! first, we scan the columns with filters, fetch their data and generate a selection vector.
			//! get runtime statistics
			auto start_time = high_resolution_clock::now();
			if (table_filters) {
				D_ASSERT(adaptive_filter);
				D_ASSERT(ALLOW_UPDATES);
				for (idx_t i = 0; i < table_filters->filters.size(); i++) {
					auto tf_idx = adaptive_filter->permutation[i];
					auto col_idx = column_ids[tf_idx];
					columns[col_idx]->Select(transaction, state.vector_index, state.column_scans[tf_idx],
					                         result.data[tf_idx], sel, approved_tuple_count,
					                         *table_filters->filters[tf_idx]);
				}
				for (auto &table_filter : table_filters->filters) {
					result.data[table_filter.first].Slice(sel, approved_tuple_count);
				}
			}
			if (approved_tuple_count == 0) {
				// all rows were filtered out by the table filters
				// skip this vector in all the scans that were not scanned yet
				D_ASSERT(table_filters);
				result.Reset();
				for (idx_t i = 0; i < column_ids.size(); i++) {
					auto col_idx = column_ids[i];
					if (col_idx == COLUMN_IDENTIFIER_ROW_ID) {
						continue;
					}
					if (table_filters->filters.find(i) == table_filters->filters.end()) {
						columns[col_idx]->Skip(state.column_scans[i]);
					}
				}
				state.vector_index++;
				continue;
			}
			//! Now we use the selection vector to fetch data for the other columns.
			for (idx_t i = 0; i < column_ids.size(); i++) {
				if (!table_filters || table_filters->filters.find(i) == table_filters->filters.end()) {
					auto column = column_ids[i];
					if (column == COLUMN_IDENTIFIER_ROW_ID) {
						D_ASSERT(result.data[i].GetType().InternalType() == PhysicalType::INT64);
						result.data[i].SetVectorType(VectorType::FLAT_VECTOR);
						auto result_data = (int64_t *)FlatVector::GetData(result.data[i]);
						for (size_t sel_idx = 0; sel_idx < approved_tuple_count; sel_idx++) {
							result_data[sel_idx] = this->start + current_row + sel.get_index(sel_idx);
						}
					} else {
						if (TYPE == TableScanType::TABLE_SCAN_REGULAR) {
							columns[column]->FilterScan(transaction, state.vector_index, state.column_scans[i],
							                            result.data[i], sel, approved_tuple_count);
						} else {
							columns[column]->FilterScanCommitted(state.vector_index, state.column_scans[i],
							                                     result.data[i], sel, approved_tuple_count,
							                                     ALLOW_UPDATES);
						}
					}
				}
			}
			auto end_time = high_resolution_clock::now();
			if (adaptive_filter && table_filters->filters.size() > 1) {
				adaptive_filter->AdaptRuntimeStatistics(duration_cast<duration<double>>(end_time - start_time).count());
			}
			D_ASSERT(approved_tuple_count > 0);
			count = approved_tuple_count;
		}
		result.SetCardinality(count);
		state.vector_index++;
		break;
	}
}

void RowGroup::Scan(TransactionData transaction, RowGroupScanState &state, DataChunk &result) {
	TemplatedScan<TableScanType::TABLE_SCAN_REGULAR>(transaction, state, result);
}

void RowGroup::ScanCommitted(RowGroupScanState &state, DataChunk &result, TableScanType type) {
	auto &transaction_manager = DuckTransactionManager::Get(db);

	auto lowest_active_start = transaction_manager.LowestActiveStart();
	auto lowest_active_id = transaction_manager.LowestActiveId();
	TransactionData data(lowest_active_id, lowest_active_start);
	switch (type) {
	case TableScanType::TABLE_SCAN_COMMITTED_ROWS:
		TemplatedScan<TableScanType::TABLE_SCAN_COMMITTED_ROWS>(data, state, result);
		break;
	case TableScanType::TABLE_SCAN_COMMITTED_ROWS_DISALLOW_UPDATES:
		TemplatedScan<TableScanType::TABLE_SCAN_COMMITTED_ROWS_DISALLOW_UPDATES>(data, state, result);
		break;
	case TableScanType::TABLE_SCAN_COMMITTED_ROWS_OMIT_PERMANENTLY_DELETED:
		TemplatedScan<TableScanType::TABLE_SCAN_COMMITTED_ROWS_OMIT_PERMANENTLY_DELETED>(data, state, result);
		break;
	default:
		throw InternalException("Unrecognized table scan type");
	}
}

ChunkInfo *RowGroup::GetChunkInfo(idx_t vector_idx) {
	if (!version_info) {
		return nullptr;
	}
	return version_info->info[vector_idx].get();
}

idx_t RowGroup::GetSelVector(TransactionData transaction, idx_t vector_idx, SelectionVector &sel_vector,
                             idx_t max_count) {
	lock_guard<mutex> lock(row_group_lock);

	auto info = GetChunkInfo(vector_idx);
	if (!info) {
		return max_count;
	}
	return info->GetSelVector(transaction, sel_vector, max_count);
}

idx_t RowGroup::GetCommittedSelVector(transaction_t start_time, transaction_t transaction_id, idx_t vector_idx,
                                      SelectionVector &sel_vector, idx_t max_count) {
	lock_guard<mutex> lock(row_group_lock);

	auto info = GetChunkInfo(vector_idx);
	if (!info) {
		return max_count;
	}
	return info->GetCommittedSelVector(start_time, transaction_id, sel_vector, max_count);
}

bool RowGroup::Fetch(TransactionData transaction, idx_t row) {
	D_ASSERT(row < this->count);
	lock_guard<mutex> lock(row_group_lock);

	idx_t vector_index = row / STANDARD_VECTOR_SIZE;
	auto info = GetChunkInfo(vector_index);
	if (!info) {
		return true;
	}
	return info->Fetch(transaction, row - vector_index * STANDARD_VECTOR_SIZE);
}

void RowGroup::FetchRow(TransactionData transaction, ColumnFetchState &state, const vector<column_t> &column_ids,
                        row_t row_id, DataChunk &result, idx_t result_idx) {
	for (idx_t col_idx = 0; col_idx < column_ids.size(); col_idx++) {
		auto column = column_ids[col_idx];
		if (column == COLUMN_IDENTIFIER_ROW_ID) {
			// row id column: fill in the row ids
			D_ASSERT(result.data[col_idx].GetType().InternalType() == PhysicalType::INT64);
			result.data[col_idx].SetVectorType(VectorType::FLAT_VECTOR);
			auto data = FlatVector::GetData<row_t>(result.data[col_idx]);
			data[result_idx] = row_id;
		} else {
			// regular column: fetch data from the base column
			columns[column]->FetchRow(transaction, state, row_id, result.data[col_idx], result_idx);
		}
	}
}

void RowGroup::AppendVersionInfo(TransactionData transaction, idx_t count) {
	idx_t row_group_start = this->count.load();
	idx_t row_group_end = row_group_start + count;
	if (row_group_end > RowGroup::ROW_GROUP_SIZE) {
		row_group_end = RowGroup::ROW_GROUP_SIZE;
	}
	lock_guard<mutex> lock(row_group_lock);

	// create the version_info if it doesn't exist yet
	if (!version_info) {
		version_info = make_unique<VersionNode>();
	}
	idx_t start_vector_idx = row_group_start / STANDARD_VECTOR_SIZE;
	idx_t end_vector_idx = (row_group_end - 1) / STANDARD_VECTOR_SIZE;
	for (idx_t vector_idx = start_vector_idx; vector_idx <= end_vector_idx; vector_idx++) {
		idx_t start = vector_idx == start_vector_idx ? row_group_start - start_vector_idx * STANDARD_VECTOR_SIZE : 0;
		idx_t end =
		    vector_idx == end_vector_idx ? row_group_end - end_vector_idx * STANDARD_VECTOR_SIZE : STANDARD_VECTOR_SIZE;
		if (start == 0 && end == STANDARD_VECTOR_SIZE) {
			// entire vector is encapsulated by append: append a single constant
			auto constant_info = make_unique<ChunkConstantInfo>(this->start + vector_idx * STANDARD_VECTOR_SIZE);
			constant_info->insert_id = transaction.transaction_id;
			constant_info->delete_id = NOT_DELETED_ID;
			version_info->info[vector_idx] = std::move(constant_info);
		} else {
			// part of a vector is encapsulated: append to that part
			ChunkVectorInfo *info;
			if (!version_info->info[vector_idx]) {
				// first time appending to this vector: create new info
				auto insert_info = make_unique<ChunkVectorInfo>(this->start + vector_idx * STANDARD_VECTOR_SIZE);
				info = insert_info.get();
				version_info->info[vector_idx] = std::move(insert_info);
			} else {
				D_ASSERT(version_info->info[vector_idx]->type == ChunkInfoType::VECTOR_INFO);
				// use existing vector
				info = (ChunkVectorInfo *)version_info->info[vector_idx].get();
			}
			info->Append(start, end, transaction.transaction_id);
		}
	}
	this->count = row_group_end;
}

void RowGroup::CommitAppend(transaction_t commit_id, idx_t row_group_start, idx_t count) {
	D_ASSERT(version_info.get());
	idx_t row_group_end = row_group_start + count;
	lock_guard<mutex> lock(row_group_lock);

	idx_t start_vector_idx = row_group_start / STANDARD_VECTOR_SIZE;
	idx_t end_vector_idx = (row_group_end - 1) / STANDARD_VECTOR_SIZE;
	for (idx_t vector_idx = start_vector_idx; vector_idx <= end_vector_idx; vector_idx++) {
		idx_t start = vector_idx == start_vector_idx ? row_group_start - start_vector_idx * STANDARD_VECTOR_SIZE : 0;
		idx_t end =
		    vector_idx == end_vector_idx ? row_group_end - end_vector_idx * STANDARD_VECTOR_SIZE : STANDARD_VECTOR_SIZE;

		auto info = version_info->info[vector_idx].get();
		info->CommitAppend(commit_id, start, end);
	}
}

void RowGroup::RevertAppend(idx_t row_group_start) {
	if (!version_info) {
		return;
	}
	idx_t start_row = row_group_start - this->start;
	idx_t start_vector_idx = (start_row + (STANDARD_VECTOR_SIZE - 1)) / STANDARD_VECTOR_SIZE;
	for (idx_t vector_idx = start_vector_idx; vector_idx < RowGroup::ROW_GROUP_VECTOR_COUNT; vector_idx++) {
		version_info->info[vector_idx].reset();
	}
	for (auto &column : columns) {
		column->RevertAppend(row_group_start);
	}
	this->count = MinValue<idx_t>(row_group_start - this->start, this->count);
	Verify();
}

void RowGroup::InitializeAppend(RowGroupAppendState &append_state) {
	append_state.row_group = this;
	append_state.offset_in_row_group = this->count;
	// for each column, initialize the append state
	append_state.states = unique_ptr<ColumnAppendState[]>(new ColumnAppendState[columns.size()]);
	for (idx_t i = 0; i < columns.size(); i++) {
		columns[i]->InitializeAppend(append_state.states[i]);
	}
}

void RowGroup::Append(RowGroupAppendState &state, DataChunk &chunk, idx_t append_count) {
	// append to the current row_group
	for (idx_t i = 0; i < columns.size(); i++) {
		columns[i]->Append(*stats[i]->statistics, state.states[i], chunk.data[i], append_count);
	}
	state.offset_in_row_group += append_count;
}

void RowGroup::Update(TransactionData transaction, DataChunk &update_chunk, row_t *ids, idx_t offset, idx_t count,
                      const vector<PhysicalIndex> &column_ids) {
#ifdef DEBUG
	for (size_t i = offset; i < offset + count; i++) {
		D_ASSERT(ids[i] >= row_t(this->start) && ids[i] < row_t(this->start + this->count));
	}
#endif
	for (idx_t i = 0; i < column_ids.size(); i++) {
		auto column = column_ids[i];
		D_ASSERT(column.index != COLUMN_IDENTIFIER_ROW_ID);
		D_ASSERT(columns[column.index]->type.id() == update_chunk.data[i].GetType().id());
		if (offset > 0) {
			Vector sliced_vector(update_chunk.data[i], offset, offset + count);
			sliced_vector.Flatten(count);
			columns[column.index]->Update(transaction, column.index, sliced_vector, ids + offset, count);
		} else {
			columns[column.index]->Update(transaction, column.index, update_chunk.data[i], ids, count);
		}
		MergeStatistics(column.index, *columns[column.index]->GetUpdateStatistics());
	}
}

void RowGroup::UpdateColumn(TransactionData transaction, DataChunk &updates, Vector &row_ids,
                            const vector<column_t> &column_path) {
	D_ASSERT(updates.ColumnCount() == 1);
	auto ids = FlatVector::GetData<row_t>(row_ids);

	auto primary_column_idx = column_path[0];
	D_ASSERT(primary_column_idx != COLUMN_IDENTIFIER_ROW_ID);
	D_ASSERT(primary_column_idx < columns.size());
	columns[primary_column_idx]->UpdateColumn(transaction, column_path, updates.data[0], ids, updates.size(), 1);
	MergeStatistics(primary_column_idx, *columns[primary_column_idx]->GetUpdateStatistics());
}

unique_ptr<BaseStatistics> RowGroup::GetStatistics(idx_t column_idx) {
	D_ASSERT(column_idx < stats.size());

	lock_guard<mutex> slock(stats_lock);
	return stats[column_idx]->statistics->Copy();
}

void RowGroup::MergeStatistics(idx_t column_idx, const BaseStatistics &other) {
	D_ASSERT(column_idx < stats.size());

	lock_guard<mutex> slock(stats_lock);
	stats[column_idx]->statistics->Merge(other);
}

void RowGroup::MergeIntoStatistics(idx_t column_idx, BaseStatistics &other) {
	D_ASSERT(column_idx < stats.size());

	lock_guard<mutex> slock(stats_lock);
	other.Merge(*stats[column_idx]->statistics);
}

RowGroupWriteData RowGroup::WriteToDisk(PartialBlockManager &manager,
                                        const vector<CompressionType> &compression_types) {
	RowGroupWriteData result;
	result.states.reserve(columns.size());
	result.statistics.reserve(columns.size());

	// Checkpoint the individual columns of the row group
	// Here we're iterating over columns. Each column can have multiple segments.
	// (Some columns will be wider than others, and require different numbers
	// of blocks to encode.) Segments cannot span blocks.
	//
	// Some of these columns are composite (list, struct). The data is written
	// first sequentially, and the pointers are written later, so that the
	// pointers all end up densely packed, and thus more cache-friendly.
	for (idx_t column_idx = 0; column_idx < columns.size(); column_idx++) {
		auto &column = columns[column_idx];
		ColumnCheckpointInfo checkpoint_info {compression_types[column_idx]};
		auto checkpoint_state = column->Checkpoint(*this, manager, checkpoint_info);
		D_ASSERT(checkpoint_state);

		auto stats = checkpoint_state->GetStatistics();
		D_ASSERT(stats);

		result.statistics.push_back(std::move(stats));
		result.states.push_back(std::move(checkpoint_state));
	}
	D_ASSERT(result.states.size() == result.statistics.size());
	return result;
}

RowGroupPointer RowGroup::Checkpoint(RowGroupWriter &writer, vector<unique_ptr<BaseStatistics>> &global_stats) {
	RowGroupPointer row_group_pointer;

	vector<CompressionType> compression_types;
	compression_types.reserve(columns.size());
	for (idx_t column_idx = 0; column_idx < columns.size(); column_idx++) {
		compression_types.push_back(writer.GetColumnCompressionType(column_idx));
	}
	auto result = WriteToDisk(writer.GetPartialBlockManager(), compression_types);
	for (idx_t column_idx = 0; column_idx < columns.size(); column_idx++) {
		global_stats[column_idx]->Merge(*result.statistics[column_idx]);
	}
	row_group_pointer.statistics = std::move(result.statistics);

	// construct the row group pointer and write the column meta data to disk
	D_ASSERT(result.states.size() == columns.size());
	row_group_pointer.row_start = start;
	row_group_pointer.tuple_count = count;
	for (auto &state : result.states) {
		// get the current position of the table data writer
		auto &data_writer = writer.GetPayloadWriter();
		auto pointer = data_writer.GetBlockPointer();

		// store the stats and the data pointers in the row group pointers
		row_group_pointer.data_pointers.push_back(pointer);

		// Write pointers to the column segments.
		//
		// Just as above, the state can refer to many other states, so this
		// can cascade recursively into more pointer writes.
		state->WriteDataPointers(writer);
	}
	row_group_pointer.versions = version_info;
	Verify();
	return row_group_pointer;
}

void RowGroup::CheckpointDeletes(VersionNode *versions, Serializer &serializer) {
	if (!versions) {
		// no version information: write nothing
		serializer.Write<idx_t>(0);
		return;
	}
	// first count how many ChunkInfo's we need to deserialize
	idx_t chunk_info_count = 0;
	for (idx_t vector_idx = 0; vector_idx < RowGroup::ROW_GROUP_VECTOR_COUNT; vector_idx++) {
		auto chunk_info = versions->info[vector_idx].get();
		if (!chunk_info) {
			continue;
		}
		chunk_info_count++;
	}
	// now serialize the actual version information
	serializer.Write<idx_t>(chunk_info_count);
	for (idx_t vector_idx = 0; vector_idx < RowGroup::ROW_GROUP_VECTOR_COUNT; vector_idx++) {
		auto chunk_info = versions->info[vector_idx].get();
		if (!chunk_info) {
			continue;
		}
		serializer.Write<idx_t>(vector_idx);
		chunk_info->Serialize(serializer);
	}
}

shared_ptr<VersionNode> RowGroup::DeserializeDeletes(Deserializer &source) {
	auto chunk_count = source.Read<idx_t>();
	if (chunk_count == 0) {
		// no deletes
		return nullptr;
	}
	auto version_info = make_shared<VersionNode>();
	for (idx_t i = 0; i < chunk_count; i++) {
		idx_t vector_index = source.Read<idx_t>();
		if (vector_index >= RowGroup::ROW_GROUP_VECTOR_COUNT) {
			throw Exception("In DeserializeDeletes, vector_index is out of range for the row group. Corrupted file?");
		}
		version_info->info[vector_index] = ChunkInfo::Deserialize(source);
	}
	return version_info;
}

void RowGroup::Serialize(RowGroupPointer &pointer, Serializer &main_serializer) {
	FieldWriter writer(main_serializer);
	writer.WriteField<uint64_t>(pointer.row_start);
	writer.WriteField<uint64_t>(pointer.tuple_count);
	auto &serializer = writer.GetSerializer();
	for (auto &stats : pointer.statistics) {
		stats->Serialize(serializer);
	}
	for (auto &data_pointer : pointer.data_pointers) {
		serializer.Write<block_id_t>(data_pointer.block_id);
		serializer.Write<uint64_t>(data_pointer.offset);
	}
	CheckpointDeletes(pointer.versions.get(), serializer);
	writer.Finalize();
}

RowGroupPointer RowGroup::Deserialize(Deserializer &main_source, const ColumnList &columns) {
	RowGroupPointer result;

	FieldReader reader(main_source);
	result.row_start = reader.ReadRequired<uint64_t>();
	result.tuple_count = reader.ReadRequired<uint64_t>();

	auto physical_columns = columns.PhysicalColumnCount();
	result.data_pointers.reserve(physical_columns);
	result.statistics.reserve(physical_columns);

	auto &source = reader.GetSource();
	for (auto &col : columns.Physical()) {
		auto stats = BaseStatistics::Deserialize(source, col.Type());
		result.statistics.push_back(std::move(stats));
	}
	for (idx_t i = 0; i < columns.PhysicalColumnCount(); i++) {
		BlockPointer pointer;
		pointer.block_id = source.Read<block_id_t>();
		pointer.offset = source.Read<uint64_t>();
		result.data_pointers.push_back(pointer);
	}
	result.versions = DeserializeDeletes(source);

	reader.Finalize();
	return result;
}

//===--------------------------------------------------------------------===//
// GetStorageInfo
//===--------------------------------------------------------------------===//
void RowGroup::GetStorageInfo(idx_t row_group_index, TableStorageInfo &result) {
	for (idx_t col_idx = 0; col_idx < columns.size(); col_idx++) {
		columns[col_idx]->GetStorageInfo(row_group_index, {col_idx}, result);
	}
}

//===--------------------------------------------------------------------===//
// Version Delete Information
//===--------------------------------------------------------------------===//
class VersionDeleteState {
public:
	VersionDeleteState(RowGroup &info, TransactionData transaction, DataTable *table, idx_t base_row)
	    : info(info), transaction(transaction), table(table), current_info(nullptr),
	      current_chunk(DConstants::INVALID_INDEX), count(0), base_row(base_row), delete_count(0) {
	}

	RowGroup &info;
	TransactionData transaction;
	DataTable *table;
	ChunkVectorInfo *current_info;
	idx_t current_chunk;
	row_t rows[STANDARD_VECTOR_SIZE];
	idx_t count;
	idx_t base_row;
	idx_t chunk_row;
	idx_t delete_count;

public:
	void Delete(row_t row_id);
	void Flush();
};

idx_t RowGroup::Delete(TransactionData transaction, DataTable *table, row_t *ids, idx_t count) {
	lock_guard<mutex> lock(row_group_lock);
	VersionDeleteState del_state(*this, transaction, table, this->start);

	// obtain a write lock
	for (idx_t i = 0; i < count; i++) {
		D_ASSERT(ids[i] >= 0);
		D_ASSERT(idx_t(ids[i]) >= this->start && idx_t(ids[i]) < this->start + this->count);
		del_state.Delete(ids[i] - this->start);
	}
	del_state.Flush();
	return del_state.delete_count;
}

void RowGroup::Verify() {
#ifdef DEBUG
	for (auto &column : columns) {
		column->Verify(*this);
	}
#endif
}

void VersionDeleteState::Delete(row_t row_id) {
	D_ASSERT(row_id >= 0);
	idx_t vector_idx = row_id / STANDARD_VECTOR_SIZE;
	idx_t idx_in_vector = row_id - vector_idx * STANDARD_VECTOR_SIZE;
	if (current_chunk != vector_idx) {
		Flush();

		if (!info.version_info) {
			info.version_info = make_unique<VersionNode>();
		}

		if (!info.version_info->info[vector_idx]) {
			// no info yet: create it
			info.version_info->info[vector_idx] =
			    make_unique<ChunkVectorInfo>(info.start + vector_idx * STANDARD_VECTOR_SIZE);
		} else if (info.version_info->info[vector_idx]->type == ChunkInfoType::CONSTANT_INFO) {
			auto &constant = (ChunkConstantInfo &)*info.version_info->info[vector_idx];
			// info exists but it's a constant info: convert to a vector info
			auto new_info = make_unique<ChunkVectorInfo>(info.start + vector_idx * STANDARD_VECTOR_SIZE);
			new_info->insert_id = constant.insert_id.load();
			for (idx_t i = 0; i < STANDARD_VECTOR_SIZE; i++) {
				new_info->inserted[i] = constant.insert_id.load();
			}
			info.version_info->info[vector_idx] = std::move(new_info);
		}
		D_ASSERT(info.version_info->info[vector_idx]->type == ChunkInfoType::VECTOR_INFO);
		current_info = (ChunkVectorInfo *)info.version_info->info[vector_idx].get();
		current_chunk = vector_idx;
		chunk_row = vector_idx * STANDARD_VECTOR_SIZE;
	}
	rows[count++] = idx_in_vector;
}

void VersionDeleteState::Flush() {
	if (count == 0) {
		return;
	}
	// it is possible for delete statements to delete the same tuple multiple times when combined with a USING clause
	// in the current_info->Delete, we check which tuples are actually deleted (excluding duplicate deletions)
	// this is returned in the actual_delete_count
	auto actual_delete_count = current_info->Delete(transaction.transaction_id, rows, count);
	delete_count += actual_delete_count;
	if (transaction.transaction && actual_delete_count > 0) {
		// now push the delete into the undo buffer, but only if any deletes were actually performed
		transaction.transaction->PushDelete(table, current_info, rows, actual_delete_count, base_row + chunk_row);
	}
	count = 0;
}

} // namespace duckdb









namespace duckdb {

RowGroupCollection::RowGroupCollection(shared_ptr<DataTableInfo> info_p, BlockManager &block_manager,
                                       vector<LogicalType> types_p, idx_t row_start_p, idx_t total_rows_p)
    : block_manager(block_manager), total_rows(total_rows_p), info(std::move(info_p)), types(std::move(types_p)),
      row_start(row_start_p) {
	row_groups = make_shared<SegmentTree>();
}

idx_t RowGroupCollection::GetTotalRows() const {
	return total_rows.load();
}

const vector<LogicalType> &RowGroupCollection::GetTypes() const {
	return types;
}

Allocator &RowGroupCollection::GetAllocator() const {
	return Allocator::Get(info->db);
}

//===--------------------------------------------------------------------===//
// Initialize
//===--------------------------------------------------------------------===//
void RowGroupCollection::Initialize(PersistentTableData &data) {
	D_ASSERT(this->row_start == 0);
	auto l = row_groups->Lock();
	for (auto &row_group_pointer : data.row_groups) {
		auto new_row_group = make_unique<RowGroup>(info->db, block_manager, *info, types, std::move(row_group_pointer));
		auto row_group_count = new_row_group->start + new_row_group->count;
		if (row_group_count > this->total_rows) {
			this->total_rows = row_group_count;
		}
		row_groups->AppendSegment(l, std::move(new_row_group));
	}
	stats.Initialize(types, data);
}

void RowGroupCollection::InitializeEmpty() {
	stats.InitializeEmpty(types);
}

void RowGroupCollection::AppendRowGroup(SegmentLock &l, idx_t start_row) {
	D_ASSERT(start_row >= row_start);
	auto new_row_group = make_unique<RowGroup>(info->db, block_manager, *info, start_row, 0);
	new_row_group->InitializeEmpty(types);
	row_groups->AppendSegment(l, std::move(new_row_group));
}

RowGroup *RowGroupCollection::GetRowGroup(int64_t index) {
	return (RowGroup *)row_groups->GetSegmentByIndex(index);
}

void RowGroupCollection::Verify() {
#ifdef DEBUG
	idx_t current_total_rows = 0;
	row_groups->Verify();
	for (auto segment = row_groups->GetRootSegment(); segment; segment = segment->Next()) {
		auto &row_group = (RowGroup &)*segment;
		row_group.Verify();
		D_ASSERT(row_group.start == this->row_start + current_total_rows);
		current_total_rows += row_group.count;
	}
	D_ASSERT(current_total_rows == total_rows.load());
#endif
}

//===--------------------------------------------------------------------===//
// Scan
//===--------------------------------------------------------------------===//
void RowGroupCollection::InitializeScan(CollectionScanState &state, const vector<column_t> &column_ids,
                                        TableFilterSet *table_filters) {
	auto row_group = (RowGroup *)row_groups->GetRootSegment();
	D_ASSERT(row_group);
	state.max_row = row_start + total_rows;
	while (row_group && !row_group->InitializeScan(state.row_group_state)) {
		row_group = (RowGroup *)row_group->Next();
	}
}

void RowGroupCollection::InitializeCreateIndexScan(CreateIndexScanState &state) {
	state.segment_lock = row_groups->Lock();
}

void RowGroupCollection::InitializeScanWithOffset(CollectionScanState &state, const vector<column_t> &column_ids,
                                                  idx_t start_row, idx_t end_row) {
	auto row_group = (RowGroup *)row_groups->GetSegment(start_row);
	D_ASSERT(row_group);
	state.max_row = end_row;
	idx_t start_vector = (start_row - row_group->start) / STANDARD_VECTOR_SIZE;
	if (!row_group->InitializeScanWithOffset(state.row_group_state, start_vector)) {
		throw InternalException("Failed to initialize row group scan with offset");
	}
}

bool RowGroupCollection::InitializeScanInRowGroup(CollectionScanState &state, RowGroup *row_group, idx_t vector_index,
                                                  idx_t max_row) {
	state.max_row = max_row;
	return row_group->InitializeScanWithOffset(state.row_group_state, vector_index);
}

void RowGroupCollection::InitializeParallelScan(ParallelCollectionScanState &state) {
	state.current_row_group = (RowGroup *)row_groups->GetRootSegment();
	state.vector_index = 0;
	state.max_row = row_start + total_rows;
	state.batch_index = 0;
}

bool RowGroupCollection::NextParallelScan(ClientContext &context, ParallelCollectionScanState &state,
                                          CollectionScanState &scan_state) {
	while (state.current_row_group && state.current_row_group->count > 0) {
		idx_t vector_index;
		idx_t max_row;
		if (ClientConfig::GetConfig(context).verify_parallelism) {
			vector_index = state.vector_index;
			max_row = state.current_row_group->start +
			          MinValue<idx_t>(state.current_row_group->count,
			                          STANDARD_VECTOR_SIZE * state.vector_index + STANDARD_VECTOR_SIZE);
			D_ASSERT(vector_index * STANDARD_VECTOR_SIZE < state.current_row_group->count);
		} else {
			vector_index = 0;
			max_row = state.current_row_group->start + state.current_row_group->count;
		}
		max_row = MinValue<idx_t>(max_row, state.max_row);
		bool need_to_scan = InitializeScanInRowGroup(scan_state, state.current_row_group, vector_index, max_row);
		if (ClientConfig::GetConfig(context).verify_parallelism) {
			state.vector_index++;
			if (state.vector_index * STANDARD_VECTOR_SIZE >= state.current_row_group->count) {
				state.current_row_group = (RowGroup *)state.current_row_group->Next();
				state.vector_index = 0;
			}
		} else {
			state.current_row_group = (RowGroup *)state.current_row_group->Next();
		}
		scan_state.batch_index = ++state.batch_index;
		if (!need_to_scan) {
			// filters allow us to skip this row group: move to the next row group
			continue;
		}
		return true;
	}
	return false;
}

bool RowGroupCollection::Scan(DuckTransaction &transaction, const vector<column_t> &column_ids,
                              const std::function<bool(DataChunk &chunk)> &fun) {
	vector<LogicalType> scan_types;
	for (idx_t i = 0; i < column_ids.size(); i++) {
		scan_types.push_back(types[column_ids[i]]);
	}
	DataChunk chunk;
	chunk.Initialize(GetAllocator(), scan_types);

	// initialize the scan
	TableScanState state;
	state.Initialize(column_ids, nullptr);
	InitializeScan(state.local_state, column_ids, nullptr);

	while (true) {
		chunk.Reset();
		state.local_state.Scan(transaction, chunk);
		if (chunk.size() == 0) {
			return true;
		}
		if (!fun(chunk)) {
			return false;
		}
	}
}

bool RowGroupCollection::Scan(DuckTransaction &transaction, const std::function<bool(DataChunk &chunk)> &fun) {
	vector<column_t> column_ids;
	column_ids.reserve(types.size());
	for (idx_t i = 0; i < types.size(); i++) {
		column_ids.push_back(i);
	}
	return Scan(transaction, column_ids, fun);
}

//===--------------------------------------------------------------------===//
// Fetch
//===--------------------------------------------------------------------===//
void RowGroupCollection::Fetch(TransactionData transaction, DataChunk &result, const vector<column_t> &column_ids,
                               const Vector &row_identifiers, idx_t fetch_count, ColumnFetchState &state) {
	// figure out which row_group to fetch from
	auto row_ids = FlatVector::GetData<row_t>(row_identifiers);
	idx_t count = 0;
	for (idx_t i = 0; i < fetch_count; i++) {
		auto row_id = row_ids[i];
		RowGroup *row_group;
		{
			idx_t segment_index;
			auto l = row_groups->Lock();
			if (!row_groups->TryGetSegmentIndex(l, row_id, segment_index)) {
				// in parallel append scenarios it is possible for the row_id
				continue;
			}
			row_group = (RowGroup *)row_groups->GetSegmentByIndex(l, segment_index);
		}
		if (!row_group->Fetch(transaction, row_id - row_group->start)) {
			continue;
		}
		row_group->FetchRow(transaction, state, column_ids, row_id, result, count);
		count++;
	}
	result.SetCardinality(count);
}

//===--------------------------------------------------------------------===//
// Append
//===--------------------------------------------------------------------===//
TableAppendState::TableAppendState()
    : row_group_append_state(*this), total_append_count(0), start_row_group(nullptr), transaction(0, 0), remaining(0) {
}

TableAppendState::~TableAppendState() {
	D_ASSERT(Exception::UncaughtException() || remaining == 0);
}

bool RowGroupCollection::IsEmpty() const {
	auto l = row_groups->Lock();
	return IsEmpty(l);
}

bool RowGroupCollection::IsEmpty(SegmentLock &l) const {
	return row_groups->IsEmpty(l);
}

void RowGroupCollection::InitializeAppend(TransactionData transaction, TableAppendState &state, idx_t append_count) {
	state.row_start = total_rows;
	state.current_row = state.row_start;
	state.total_append_count = 0;

	// start writing to the row_groups
	auto l = row_groups->Lock();
	if (IsEmpty(l)) {
		// empty row group collection: empty first row group
		AppendRowGroup(l, row_start);
	}
	state.start_row_group = (RowGroup *)row_groups->GetLastSegment(l);
	D_ASSERT(this->row_start + total_rows == state.start_row_group->start + state.start_row_group->count);
	state.start_row_group->InitializeAppend(state.row_group_append_state);
	state.remaining = append_count;
	state.transaction = transaction;
	if (state.remaining > 0) {
		state.start_row_group->AppendVersionInfo(transaction, state.remaining);
		total_rows += state.remaining;
	}
}

void RowGroupCollection::InitializeAppend(TableAppendState &state) {
	TransactionData tdata(0, 0);
	InitializeAppend(tdata, state, 0);
}

bool RowGroupCollection::Append(DataChunk &chunk, TableAppendState &state) {
	D_ASSERT(chunk.ColumnCount() == types.size());
	chunk.Verify();

	bool new_row_group = false;
	idx_t append_count = chunk.size();
	idx_t remaining = chunk.size();
	state.total_append_count += append_count;
	while (true) {
		auto current_row_group = state.row_group_append_state.row_group;
		// check how much we can fit into the current row_group
		idx_t append_count =
		    MinValue<idx_t>(remaining, RowGroup::ROW_GROUP_SIZE - state.row_group_append_state.offset_in_row_group);
		if (append_count > 0) {
			current_row_group->Append(state.row_group_append_state, chunk, append_count);
			// merge the stats
			auto stats_lock = stats.GetLock();
			for (idx_t i = 0; i < types.size(); i++) {
				current_row_group->MergeIntoStatistics(i, *stats.GetStats(i).stats);
			}
		}
		remaining -= append_count;
		if (state.remaining > 0) {
			state.remaining -= append_count;
		}
		if (remaining > 0) {
			// we expect max 1 iteration of this loop (i.e. a single chunk should never overflow more than one
			// row_group)
			D_ASSERT(chunk.size() == remaining + append_count);
			// slice the input chunk
			if (remaining < chunk.size()) {
				SelectionVector sel(remaining);
				for (idx_t i = 0; i < remaining; i++) {
					sel.set_index(i, append_count + i);
				}
				chunk.Slice(sel, remaining);
			}
			// append a new row_group
			new_row_group = true;
			auto next_start = current_row_group->start + state.row_group_append_state.offset_in_row_group;

			auto l = row_groups->Lock();
			AppendRowGroup(l, next_start);
			// set up the append state for this row_group
			auto last_row_group = (RowGroup *)row_groups->GetLastSegment(l);
			last_row_group->InitializeAppend(state.row_group_append_state);
			if (state.remaining > 0) {
				last_row_group->AppendVersionInfo(state.transaction, state.remaining);
			}
			continue;
		} else {
			break;
		}
	}
	state.current_row += append_count;
	auto stats_lock = stats.GetLock();
	for (idx_t col_idx = 0; col_idx < types.size(); col_idx++) {
		auto type = types[col_idx].InternalType();
		if (type == PhysicalType::LIST || type == PhysicalType::STRUCT) {
			continue;
		}
		stats.GetStats(col_idx).stats->UpdateDistinctStatistics(chunk.data[col_idx], chunk.size());
	}
	return new_row_group;
}

void RowGroupCollection::FinalizeAppend(TransactionData transaction, TableAppendState &state) {
	auto remaining = state.total_append_count;
	auto row_group = state.start_row_group;
	while (remaining > 0) {
		auto append_count = MinValue<idx_t>(remaining, RowGroup::ROW_GROUP_SIZE - row_group->count);
		row_group->AppendVersionInfo(transaction, append_count);
		remaining -= append_count;
		row_group = (RowGroup *)row_group->Next();
	}
	total_rows += state.total_append_count;

	state.total_append_count = 0;
	state.start_row_group = nullptr;

	Verify();
}

void RowGroupCollection::CommitAppend(transaction_t commit_id, idx_t row_start, idx_t count) {
	auto row_group = (RowGroup *)row_groups->GetSegment(row_start);
	D_ASSERT(row_group);
	idx_t current_row = row_start;
	idx_t remaining = count;
	while (true) {
		idx_t start_in_row_group = current_row - row_group->start;
		idx_t append_count = MinValue<idx_t>(row_group->count - start_in_row_group, remaining);

		row_group->CommitAppend(commit_id, start_in_row_group, append_count);

		current_row += append_count;
		remaining -= append_count;
		if (remaining == 0) {
			break;
		}
		row_group = (RowGroup *)row_group->Next();
	}
}

void RowGroupCollection::RevertAppendInternal(idx_t start_row, idx_t count) {
	if (total_rows != start_row + count) {
		throw InternalException("Interleaved appends: this should no longer happen");
	}
	total_rows = start_row;

	auto l = row_groups->Lock();
	// find the segment index that the current row belongs to
	idx_t segment_index = row_groups->GetSegmentIndex(l, start_row);
	auto segment = row_groups->GetSegmentByIndex(l, segment_index);
	auto &info = (RowGroup &)*segment;

	// remove any segments AFTER this segment: they should be deleted entirely
	row_groups->EraseSegments(l, segment_index);

	info.next = nullptr;
	info.RevertAppend(start_row);
}

void RowGroupCollection::MergeStorage(RowGroupCollection &data) {
	D_ASSERT(data.types == types);
	auto index = row_start + total_rows.load();
	for (auto segment = data.row_groups->GetRootSegment(); segment; segment = segment->Next()) {
		auto &row_group = (RowGroup &)*segment;
		auto new_group = make_unique<RowGroup>(row_group, index);
		index += new_group->count;
		row_groups->AppendSegment(std::move(new_group));
	}
	stats.MergeStats(data.stats);
	total_rows += data.total_rows.load();
}

//===--------------------------------------------------------------------===//
// Delete
//===--------------------------------------------------------------------===//
idx_t RowGroupCollection::Delete(TransactionData transaction, DataTable *table, row_t *ids, idx_t count) {
	idx_t delete_count = 0;
	// delete is in the row groups
	// we need to figure out for each id to which row group it belongs
	// usually all (or many) ids belong to the same row group
	// we iterate over the ids and check for every id if it belongs to the same row group as their predecessor
	idx_t pos = 0;
	do {
		idx_t start = pos;
		auto row_group = (RowGroup *)row_groups->GetSegment(ids[start]);
		for (pos++; pos < count; pos++) {
			D_ASSERT(ids[pos] >= 0);
			// check if this id still belongs to this row group
			if (idx_t(ids[pos]) < row_group->start) {
				// id is before row_group start -> it does not
				break;
			}
			if (idx_t(ids[pos]) >= row_group->start + row_group->count) {
				// id is after row group end -> it does not
				break;
			}
		}
		delete_count += row_group->Delete(transaction, table, ids + start, pos - start);
	} while (pos < count);
	return delete_count;
}

//===--------------------------------------------------------------------===//
// Update
//===--------------------------------------------------------------------===//
void RowGroupCollection::Update(TransactionData transaction, row_t *ids, const vector<PhysicalIndex> &column_ids,
                                DataChunk &updates) {
	idx_t pos = 0;
	do {
		idx_t start = pos;
		auto row_group = (RowGroup *)row_groups->GetSegment(ids[pos]);
		row_t base_id =
		    row_group->start + ((ids[pos] - row_group->start) / STANDARD_VECTOR_SIZE * STANDARD_VECTOR_SIZE);
		row_t max_id = MinValue<row_t>(base_id + STANDARD_VECTOR_SIZE, row_group->start + row_group->count);
		for (pos++; pos < updates.size(); pos++) {
			D_ASSERT(ids[pos] >= 0);
			// check if this id still belongs to this vector in this row group
			if (ids[pos] < base_id) {
				// id is before vector start -> it does not
				break;
			}
			if (ids[pos] >= max_id) {
				// id is after the maximum id in this vector -> it does not
				break;
			}
		}
		row_group->Update(transaction, updates, ids, start, pos - start, column_ids);

		auto l = stats.GetLock();
		for (idx_t i = 0; i < column_ids.size(); i++) {
			auto column_id = column_ids[i];
			stats.MergeStats(*l, column_id.index, *row_group->GetStatistics(column_id.index));
		}
	} while (pos < updates.size());
}

void RowGroupCollection::RemoveFromIndexes(TableIndexList &indexes, Vector &row_identifiers, idx_t count) {
	auto row_ids = FlatVector::GetData<row_t>(row_identifiers);

	// figure out which row_group to fetch from
	auto row_group = (RowGroup *)row_groups->GetSegment(row_ids[0]);
	auto row_group_vector_idx = (row_ids[0] - row_group->start) / STANDARD_VECTOR_SIZE;
	auto base_row_id = row_group_vector_idx * STANDARD_VECTOR_SIZE + row_group->start;

	// create a selection vector from the row_ids
	SelectionVector sel(STANDARD_VECTOR_SIZE);
	for (idx_t i = 0; i < count; i++) {
		auto row_in_vector = row_ids[i] - base_row_id;
		D_ASSERT(row_in_vector < STANDARD_VECTOR_SIZE);
		sel.set_index(i, row_in_vector);
	}

	// now fetch the columns from that row_group
	TableScanState state;
	state.table_state.max_row = row_start + total_rows;

	// FIXME: we do not need to fetch all columns, only the columns required by the indices!
	vector<column_t> column_ids;
	column_ids.reserve(types.size());
	for (idx_t i = 0; i < types.size(); i++) {
		column_ids.push_back(i);
	}
	state.Initialize(std::move(column_ids));

	DataChunk result;
	result.Initialize(GetAllocator(), types);

	row_group->InitializeScanWithOffset(state.table_state.row_group_state, row_group_vector_idx);
	row_group->ScanCommitted(state.table_state.row_group_state, result, TableScanType::TABLE_SCAN_COMMITTED_ROWS);
	result.Slice(sel, count);

	indexes.Scan([&](Index &index) {
		index.Delete(result, row_identifiers);
		return false;
	});
}

void RowGroupCollection::UpdateColumn(TransactionData transaction, Vector &row_ids, const vector<column_t> &column_path,
                                      DataChunk &updates) {
	auto first_id = FlatVector::GetValue<row_t>(row_ids, 0);
	if (first_id >= MAX_ROW_ID) {
		throw NotImplementedException("Cannot update a column-path on transaction local data");
	}
	// find the row_group this id belongs to
	auto primary_column_idx = column_path[0];
	auto row_group = (RowGroup *)row_groups->GetSegment(first_id);
	row_group->UpdateColumn(transaction, updates, row_ids, column_path);

	row_group->MergeIntoStatistics(primary_column_idx, *stats.GetStats(primary_column_idx).stats);
}

//===--------------------------------------------------------------------===//
// Checkpoint
//===--------------------------------------------------------------------===//
void RowGroupCollection::Checkpoint(TableDataWriter &writer, vector<unique_ptr<BaseStatistics>> &global_stats) {
	for (auto row_group = (RowGroup *)row_groups->GetRootSegment(); row_group;
	     row_group = (RowGroup *)row_group->Next()) {
		auto rowg_writer = writer.GetRowGroupWriter(*row_group);
		auto pointer = row_group->Checkpoint(*rowg_writer, global_stats);
		writer.AddRowGroup(std::move(pointer), std::move(rowg_writer));
	}
}

//===--------------------------------------------------------------------===//
// CommitDrop
//===--------------------------------------------------------------------===//
void RowGroupCollection::CommitDropColumn(idx_t index) {
	auto segment = (RowGroup *)row_groups->GetRootSegment();
	while (segment) {
		segment->CommitDropColumn(index);
		segment = (RowGroup *)segment->Next();
	}
}

void RowGroupCollection::CommitDropTable() {
	auto segment = (RowGroup *)row_groups->GetRootSegment();
	while (segment) {
		segment->CommitDrop();
		segment = (RowGroup *)segment->Next();
	}
}

//===--------------------------------------------------------------------===//
// GetStorageInfo
//===--------------------------------------------------------------------===//
void RowGroupCollection::GetStorageInfo(TableStorageInfo &result) {
	auto row_group = (RowGroup *)row_groups->GetRootSegment();
	idx_t row_group_index = 0;
	while (row_group) {
		row_group->GetStorageInfo(row_group_index, result);
		row_group_index++;

		row_group = (RowGroup *)row_group->Next();
	}
}

//===--------------------------------------------------------------------===//
// Alter
//===--------------------------------------------------------------------===//
shared_ptr<RowGroupCollection> RowGroupCollection::AddColumn(ClientContext &context, ColumnDefinition &new_column,
                                                             Expression *default_value) {
	idx_t new_column_idx = types.size();
	auto new_types = types;
	new_types.push_back(new_column.GetType());
	auto result =
	    make_shared<RowGroupCollection>(info, block_manager, std::move(new_types), row_start, total_rows.load());

	ExpressionExecutor executor(context);
	DataChunk dummy_chunk;
	Vector default_vector(new_column.GetType());
	if (!default_value) {
		FlatVector::Validity(default_vector).SetAllInvalid(STANDARD_VECTOR_SIZE);
	} else {
		executor.AddExpression(*default_value);
	}

	result->stats.InitializeAddColumn(stats, new_column.GetType());
	auto &new_column_stats = result->stats.GetStats(new_column_idx);

	// fill the column with its DEFAULT value, or NULL if none is specified
	auto new_stats = make_unique<SegmentStatistics>(new_column.GetType());
	auto current_row_group = (RowGroup *)row_groups->GetRootSegment();
	while (current_row_group) {
		auto new_row_group = current_row_group->AddColumn(new_column, executor, default_value, default_vector);
		// merge in the statistics
		new_row_group->MergeIntoStatistics(new_column_idx, *new_column_stats.stats);

		result->row_groups->AppendSegment(std::move(new_row_group));
		current_row_group = (RowGroup *)current_row_group->Next();
	}
	return result;
}

shared_ptr<RowGroupCollection> RowGroupCollection::RemoveColumn(idx_t col_idx) {
	D_ASSERT(col_idx < types.size());
	auto new_types = types;
	new_types.erase(new_types.begin() + col_idx);

	auto result =
	    make_shared<RowGroupCollection>(info, block_manager, std::move(new_types), row_start, total_rows.load());
	result->stats.InitializeRemoveColumn(stats, col_idx);

	auto current_row_group = (RowGroup *)row_groups->GetRootSegment();
	while (current_row_group) {
		auto new_row_group = current_row_group->RemoveColumn(col_idx);
		result->row_groups->AppendSegment(std::move(new_row_group));
		current_row_group = (RowGroup *)current_row_group->Next();
	}
	return result;
}

shared_ptr<RowGroupCollection> RowGroupCollection::AlterType(ClientContext &context, idx_t changed_idx,
                                                             const LogicalType &target_type,
                                                             vector<column_t> bound_columns, Expression &cast_expr) {
	D_ASSERT(changed_idx < types.size());
	auto new_types = types;
	new_types[changed_idx] = target_type;

	auto result =
	    make_shared<RowGroupCollection>(info, block_manager, std::move(new_types), row_start, total_rows.load());
	result->stats.InitializeAlterType(stats, changed_idx, target_type);

	vector<LogicalType> scan_types;
	for (idx_t i = 0; i < bound_columns.size(); i++) {
		if (bound_columns[i] == COLUMN_IDENTIFIER_ROW_ID) {
			scan_types.emplace_back(LogicalType::ROW_TYPE);
		} else {
			scan_types.push_back(types[bound_columns[i]]);
		}
	}
	DataChunk scan_chunk;
	scan_chunk.Initialize(GetAllocator(), scan_types);

	ExpressionExecutor executor(context);
	executor.AddExpression(cast_expr);

	TableScanState scan_state;
	scan_state.Initialize(bound_columns);
	scan_state.table_state.max_row = row_start + total_rows;

	// now alter the type of the column within all of the row_groups individually
	auto current_row_group = (RowGroup *)row_groups->GetRootSegment();
	auto &changed_stats = result->stats.GetStats(changed_idx);
	while (current_row_group) {
		auto new_row_group = current_row_group->AlterType(target_type, changed_idx, executor,
		                                                  scan_state.table_state.row_group_state, scan_chunk);
		new_row_group->MergeIntoStatistics(changed_idx, *changed_stats.stats);
		result->row_groups->AppendSegment(std::move(new_row_group));
		current_row_group = (RowGroup *)current_row_group->Next();
	}

	return result;
}

void RowGroupCollection::VerifyNewConstraint(DataTable &parent, const BoundConstraint &constraint) {
	if (total_rows == 0) {
		return;
	}
	// scan the original table, check if there's any null value
	auto &not_null_constraint = (BoundNotNullConstraint &)constraint;
	vector<LogicalType> scan_types;
	auto physical_index = not_null_constraint.index.index;
	D_ASSERT(physical_index < types.size());
	scan_types.push_back(types[physical_index]);
	DataChunk scan_chunk;
	scan_chunk.Initialize(GetAllocator(), scan_types);

	CreateIndexScanState state;
	vector<column_t> cids;
	cids.push_back(physical_index);
	// Use ScanCommitted to scan the latest committed data
	state.Initialize(cids, nullptr);
	InitializeScan(state.table_state, cids, nullptr);
	InitializeCreateIndexScan(state);
	while (true) {
		scan_chunk.Reset();
		state.table_state.ScanCommitted(scan_chunk, TableScanType::TABLE_SCAN_COMMITTED_ROWS_OMIT_PERMANENTLY_DELETED);
		if (scan_chunk.size() == 0) {
			break;
		}
		// Check constraint
		if (VectorOperations::HasNull(scan_chunk.data[0], scan_chunk.size())) {
			throw ConstraintException("NOT NULL constraint failed: %s.%s", info->table,
			                          parent.column_definitions[physical_index].GetName());
		}
	}
}

//===--------------------------------------------------------------------===//
// Statistics
//===--------------------------------------------------------------------===//
unique_ptr<BaseStatistics> RowGroupCollection::CopyStats(column_t column_id) {
	return stats.CopyStats(column_id);
}

void RowGroupCollection::SetStatistics(column_t column_id, const std::function<void(BaseStatistics &)> &set_fun) {
	D_ASSERT(column_id != COLUMN_IDENTIFIER_ROW_ID);
	auto stats_guard = stats.GetLock();
	set_fun(*stats.GetStats(column_id).stats);
}

} // namespace duckdb





namespace duckdb {

void TableScanState::Initialize(vector<column_t> column_ids, TableFilterSet *table_filters) {
	this->column_ids = std::move(column_ids);
	this->table_filters = table_filters;
	if (table_filters) {
		D_ASSERT(table_filters->filters.size() > 0);
		this->adaptive_filter = make_unique<AdaptiveFilter>(table_filters);
	}
}

const vector<column_t> &TableScanState::GetColumnIds() {
	D_ASSERT(!column_ids.empty());
	return column_ids;
}

TableFilterSet *TableScanState::GetFilters() {
	D_ASSERT(!table_filters || adaptive_filter.get());
	return table_filters;
}

AdaptiveFilter *TableScanState::GetAdaptiveFilter() {
	return adaptive_filter.get();
}

void ColumnScanState::NextInternal(idx_t count) {
	if (!current) {
		//! There is no column segment
		return;
	}
	row_index += count;
	while (row_index >= current->start + current->count) {
		current = (ColumnSegment *)current->Next();
		initialized = false;
		segment_checked = false;
		if (!current) {
			break;
		}
	}
	D_ASSERT(!current || (row_index >= current->start && row_index < current->start + current->count));
}

void ColumnScanState::Next(idx_t count) {
	NextInternal(count);
	for (auto &child_state : child_states) {
		child_state.Next(count);
	}
}

void ColumnScanState::NextVector() {
	Next(STANDARD_VECTOR_SIZE);
}

const vector<column_t> &RowGroupScanState::GetColumnIds() {
	return parent.GetColumnIds();
}

TableFilterSet *RowGroupScanState::GetFilters() {
	return parent.GetFilters();
}

AdaptiveFilter *RowGroupScanState::GetAdaptiveFilter() {
	return parent.GetAdaptiveFilter();
}

idx_t RowGroupScanState::GetParentMaxRow() {
	return parent.max_row;
}

const vector<column_t> &CollectionScanState::GetColumnIds() {
	return parent.GetColumnIds();
}

TableFilterSet *CollectionScanState::GetFilters() {
	return parent.GetFilters();
}

AdaptiveFilter *CollectionScanState::GetAdaptiveFilter() {
	return parent.GetAdaptiveFilter();
}

bool CollectionScanState::Scan(DuckTransaction &transaction, DataChunk &result) {
	auto current_row_group = row_group_state.row_group;
	while (current_row_group) {
		current_row_group->Scan(transaction, row_group_state, result);
		if (result.size() > 0) {
			return true;
		} else {
			do {
				current_row_group = row_group_state.row_group = (RowGroup *)current_row_group->Next();
				if (current_row_group) {
					bool scan_row_group = current_row_group->InitializeScan(row_group_state);
					if (scan_row_group) {
						// scan this row group
						break;
					}
				}
			} while (current_row_group);
		}
	}
	return false;
}

bool CollectionScanState::ScanCommitted(DataChunk &result, TableScanType type) {
	auto current_row_group = row_group_state.row_group;
	while (current_row_group) {
		current_row_group->ScanCommitted(row_group_state, result, type);
		if (result.size() > 0) {
			return true;
		} else {
			current_row_group = row_group_state.row_group = (RowGroup *)current_row_group->Next();
			if (current_row_group) {
				current_row_group->InitializeScan(row_group_state);
			}
		}
	}
	return false;
}

} // namespace duckdb




namespace duckdb {

SegmentLock SegmentTree::Lock() {
	return SegmentLock(node_lock);
}

bool SegmentTree::IsEmpty(SegmentLock &) {
	return nodes.empty();
}

SegmentBase *SegmentTree::GetRootSegment(SegmentLock &l) {
	return nodes.empty() ? nullptr : nodes[0].node.get();
}

vector<SegmentNode> SegmentTree::MoveSegments(SegmentLock &) {
	return std::move(nodes);
}

SegmentBase *SegmentTree::GetRootSegment() {
	auto l = Lock();
	return GetRootSegment(l);
}

SegmentBase *SegmentTree::GetSegmentByIndex(SegmentLock &, int64_t index) {
	if (index < 0) {
		index = nodes.size() + index;
		if (index < 0) {
			return nullptr;
		}
		return nodes[index].node.get();
	} else {
		if (idx_t(index) >= nodes.size()) {
			return nullptr;
		}
		return nodes[index].node.get();
	}
}
SegmentBase *SegmentTree::GetSegmentByIndex(int64_t index) {
	auto l = Lock();
	return GetSegmentByIndex(l, index);
}

SegmentBase *SegmentTree::GetLastSegment(SegmentLock &l) {
	if (nodes.empty()) {
		return nullptr;
	}
	return nodes.back().node.get();
}

SegmentBase *SegmentTree::GetLastSegment() {
	auto l = Lock();
	return GetLastSegment(l);
}

SegmentBase *SegmentTree::GetSegment(SegmentLock &l, idx_t row_number) {
	return nodes[GetSegmentIndex(l, row_number)].node.get();
}

SegmentBase *SegmentTree::GetSegment(idx_t row_number) {
	auto l = Lock();
	return GetSegment(l, row_number);
}

bool SegmentTree::TryGetSegmentIndex(SegmentLock &, idx_t row_number, idx_t &result) {
	if (nodes.empty()) {
		return false;
	}
	D_ASSERT(!nodes.empty());
	D_ASSERT(row_number >= nodes[0].row_start);
	D_ASSERT(row_number < nodes.back().row_start + nodes.back().node->count);
	idx_t lower = 0;
	idx_t upper = nodes.size() - 1;
	// binary search to find the node
	while (lower <= upper) {
		idx_t index = (lower + upper) / 2;
		D_ASSERT(index < nodes.size());
		auto &entry = nodes[index];
		D_ASSERT(entry.row_start == entry.node->start);
		if (row_number < entry.row_start) {
			upper = index - 1;
		} else if (row_number >= entry.row_start + entry.node->count) {
			lower = index + 1;
		} else {
			result = index;
			return true;
		}
	}
	return false;
}

idx_t SegmentTree::GetSegmentIndex(SegmentLock &l, idx_t row_number) {
	idx_t segment_index;
	if (TryGetSegmentIndex(l, row_number, segment_index)) {
		return segment_index;
	}
	string error;
	error = StringUtil::Format("Attempting to find row number \"%lld\" in %lld nodes\n", row_number, nodes.size());
	for (idx_t i = 0; i < nodes.size(); i++) {
		error +=
		    StringUtil::Format("Node %lld: Start %lld, Count %lld", i, nodes[i].row_start, nodes[i].node->count.load());
	}
	throw InternalException("Could not find node in column segment tree!\n%s%s", error, Exception::GetStackTrace());
}

idx_t SegmentTree::GetSegmentIndex(idx_t row_number) {
	auto l = Lock();
	return GetSegmentIndex(l, row_number);
}

bool SegmentTree::HasSegment(SegmentLock &, SegmentBase *segment) {
	for (auto &node : nodes) {
		if (node.node.get() == segment) {
			return true;
		}
	}
	return false;
}

bool SegmentTree::HasSegment(SegmentBase *segment) {
	auto l = Lock();
	return HasSegment(l, segment);
}

void SegmentTree::AppendSegment(SegmentLock &, unique_ptr<SegmentBase> segment) {
	D_ASSERT(segment);
	// add the node to the list of nodes
	if (!nodes.empty()) {
		nodes.back().node->next = segment.get();
	}
	SegmentNode node;
	node.row_start = segment->start;
	node.node = std::move(segment);
	nodes.push_back(std::move(node));
}

void SegmentTree::AppendSegment(unique_ptr<SegmentBase> segment) {
	auto l = Lock();
	AppendSegment(l, std::move(segment));
}

void SegmentTree::EraseSegments(SegmentLock &, idx_t segment_start) {
	if (segment_start >= nodes.size() - 1) {
		return;
	}
	nodes.erase(nodes.begin() + segment_start + 1, nodes.end());
}

void SegmentTree::Replace(SegmentLock &, SegmentTree &other) {
	nodes = std::move(other.nodes);
}

void SegmentTree::Replace(SegmentTree &other) {
	auto l = Lock();
	Replace(l, other);
}

void SegmentTree::Verify(SegmentLock &) {
#ifdef DEBUG
	idx_t base_start = nodes.empty() ? 0 : nodes[0].node->start;
	for (idx_t i = 0; i < nodes.size(); i++) {
		D_ASSERT(nodes[i].row_start == nodes[i].node->start);
		D_ASSERT(nodes[i].node->start == base_start);
		base_start += nodes[i].node->count;
	}
#endif
}

void SegmentTree::Verify() {
#ifdef DEBUG
	auto l = Lock();
	Verify(l);
#endif
}

} // namespace duckdb








namespace duckdb {

StandardColumnData::StandardColumnData(BlockManager &block_manager, DataTableInfo &info, idx_t column_index,
                                       idx_t start_row, LogicalType type, ColumnData *parent)
    : ColumnData(block_manager, info, column_index, start_row, std::move(type), parent),
      validity(block_manager, info, 0, start_row, this) {
}

StandardColumnData::StandardColumnData(ColumnData &original, idx_t start_row, ColumnData *parent)
    : ColumnData(original, start_row, parent), validity(((StandardColumnData &)original).validity, start_row, this) {
}

bool StandardColumnData::CheckZonemap(ColumnScanState &state, TableFilter &filter) {
	if (!state.segment_checked) {
		if (!state.current) {
			return true;
		}
		state.segment_checked = true;
		auto prune_result = filter.CheckStatistics(*state.current->stats.statistics);
		if (prune_result != FilterPropagateResult::FILTER_ALWAYS_FALSE) {
			return true;
		}
		if (updates) {
			auto update_stats = updates->GetStatistics();
			prune_result = filter.CheckStatistics(*update_stats);
			return prune_result != FilterPropagateResult::FILTER_ALWAYS_FALSE;
		} else {
			return false;
		}
	} else {
		return true;
	}
}

void StandardColumnData::InitializeScan(ColumnScanState &state) {
	ColumnData::InitializeScan(state);

	// initialize the validity segment
	ColumnScanState child_state;
	validity.InitializeScan(child_state);
	state.child_states.push_back(std::move(child_state));
}

void StandardColumnData::InitializeScanWithOffset(ColumnScanState &state, idx_t row_idx) {
	ColumnData::InitializeScanWithOffset(state, row_idx);

	// initialize the validity segment
	ColumnScanState child_state;
	validity.InitializeScanWithOffset(child_state, row_idx);
	state.child_states.push_back(std::move(child_state));
}

idx_t StandardColumnData::Scan(TransactionData transaction, idx_t vector_index, ColumnScanState &state,
                               Vector &result) {
	D_ASSERT(state.row_index == state.child_states[0].row_index);
	auto scan_count = ColumnData::Scan(transaction, vector_index, state, result);
	validity.Scan(transaction, vector_index, state.child_states[0], result);
	return scan_count;
}

idx_t StandardColumnData::ScanCommitted(idx_t vector_index, ColumnScanState &state, Vector &result,
                                        bool allow_updates) {
	D_ASSERT(state.row_index == state.child_states[0].row_index);
	auto scan_count = ColumnData::ScanCommitted(vector_index, state, result, allow_updates);
	validity.ScanCommitted(vector_index, state.child_states[0], result, allow_updates);
	return scan_count;
}

idx_t StandardColumnData::ScanCount(ColumnScanState &state, Vector &result, idx_t count) {
	auto scan_count = ColumnData::ScanCount(state, result, count);
	validity.ScanCount(state.child_states[0], result, count);
	return scan_count;
}

void StandardColumnData::InitializeAppend(ColumnAppendState &state) {
	ColumnData::InitializeAppend(state);

	ColumnAppendState child_append;
	validity.InitializeAppend(child_append);
	state.child_appends.push_back(std::move(child_append));
}

void StandardColumnData::AppendData(BaseStatistics &stats, ColumnAppendState &state, UnifiedVectorFormat &vdata,
                                    idx_t count) {
	ColumnData::AppendData(stats, state, vdata, count);

	validity.AppendData(*stats.validity_stats, state.child_appends[0], vdata, count);
}

void StandardColumnData::RevertAppend(row_t start_row) {
	ColumnData::RevertAppend(start_row);

	validity.RevertAppend(start_row);
}

idx_t StandardColumnData::Fetch(ColumnScanState &state, row_t row_id, Vector &result) {
	// fetch validity mask
	if (state.child_states.empty()) {
		ColumnScanState child_state;
		state.child_states.push_back(std::move(child_state));
	}
	auto scan_count = ColumnData::Fetch(state, row_id, result);
	validity.Fetch(state.child_states[0], row_id, result);
	return scan_count;
}

void StandardColumnData::Update(TransactionData transaction, idx_t column_index, Vector &update_vector, row_t *row_ids,
                                idx_t update_count) {
	ColumnData::Update(transaction, column_index, update_vector, row_ids, update_count);
	validity.Update(transaction, column_index, update_vector, row_ids, update_count);
}

void StandardColumnData::UpdateColumn(TransactionData transaction, const vector<column_t> &column_path,
                                      Vector &update_vector, row_t *row_ids, idx_t update_count, idx_t depth) {
	if (depth >= column_path.size()) {
		// update this column
		ColumnData::Update(transaction, column_path[0], update_vector, row_ids, update_count);
	} else {
		// update the child column (i.e. the validity column)
		validity.UpdateColumn(transaction, column_path, update_vector, row_ids, update_count, depth + 1);
	}
}

unique_ptr<BaseStatistics> StandardColumnData::GetUpdateStatistics() {
	auto stats = updates ? updates->GetStatistics() : nullptr;
	auto validity_stats = validity.GetUpdateStatistics();
	if (!stats && !validity_stats) {
		return nullptr;
	}
	if (!stats) {
		stats = BaseStatistics::CreateEmpty(type, StatisticsType::GLOBAL_STATS);
	}
	stats->validity_stats = std::move(validity_stats);
	return stats;
}

void StandardColumnData::FetchRow(TransactionData transaction, ColumnFetchState &state, row_t row_id, Vector &result,
                                  idx_t result_idx) {
	// find the segment the row belongs to
	if (state.child_states.empty()) {
		auto child_state = make_unique<ColumnFetchState>();
		state.child_states.push_back(std::move(child_state));
	}
	validity.FetchRow(transaction, *state.child_states[0], row_id, result, result_idx);
	ColumnData::FetchRow(transaction, state, row_id, result, result_idx);
}

void StandardColumnData::CommitDropColumn() {
	ColumnData::CommitDropColumn();
	validity.CommitDropColumn();
}

struct StandardColumnCheckpointState : public ColumnCheckpointState {
	StandardColumnCheckpointState(RowGroup &row_group, ColumnData &column_data,
	                              PartialBlockManager &partial_block_manager)
	    : ColumnCheckpointState(row_group, column_data, partial_block_manager) {
	}

	unique_ptr<ColumnCheckpointState> validity_state;

public:
	unique_ptr<BaseStatistics> GetStatistics() override {
		D_ASSERT(global_stats);
		global_stats->validity_stats = validity_state->GetStatistics();
		return std::move(global_stats);
	}

	void WriteDataPointers(RowGroupWriter &writer) override {
		ColumnCheckpointState::WriteDataPointers(writer);
		validity_state->WriteDataPointers(writer);
	}

	void GetBlockIds(unordered_set<block_id_t> &result) override {
		ColumnCheckpointState::GetBlockIds(result);
		validity_state->GetBlockIds(result);
	}
};

unique_ptr<ColumnCheckpointState>
StandardColumnData::CreateCheckpointState(RowGroup &row_group, PartialBlockManager &partial_block_manager) {
	return make_unique<StandardColumnCheckpointState>(row_group, *this, partial_block_manager);
}

unique_ptr<ColumnCheckpointState> StandardColumnData::Checkpoint(RowGroup &row_group,
                                                                 PartialBlockManager &partial_block_manager,
                                                                 ColumnCheckpointInfo &checkpoint_info) {
	auto validity_state = validity.Checkpoint(row_group, partial_block_manager, checkpoint_info);
	auto base_state = ColumnData::Checkpoint(row_group, partial_block_manager, checkpoint_info);
	auto &checkpoint_state = (StandardColumnCheckpointState &)*base_state;
	checkpoint_state.validity_state = std::move(validity_state);
	return base_state;
}

void StandardColumnData::CheckpointScan(ColumnSegment *segment, ColumnScanState &state, idx_t row_group_start,
                                        idx_t count, Vector &scan_vector) {
	ColumnData::CheckpointScan(segment, state, row_group_start, count, scan_vector);

	idx_t offset_in_row_group = state.row_index - row_group_start;
	validity.ScanCommittedRange(row_group_start, offset_in_row_group, count, scan_vector);
}

void StandardColumnData::DeserializeColumn(Deserializer &source) {
	ColumnData::DeserializeColumn(source);
	validity.DeserializeColumn(source);
}

void StandardColumnData::GetStorageInfo(idx_t row_group_index, vector<idx_t> col_path, TableStorageInfo &result) {
	ColumnData::GetStorageInfo(row_group_index, col_path, result);
	col_path.push_back(0);
	validity.GetStorageInfo(row_group_index, std::move(col_path), result);
}

void StandardColumnData::Verify(RowGroup &parent) {
#ifdef DEBUG
	ColumnData::Verify(parent);
	validity.Verify(parent);
#endif
}

} // namespace duckdb




namespace duckdb {

StructColumnData::StructColumnData(BlockManager &block_manager, DataTableInfo &info, idx_t column_index,
                                   idx_t start_row, LogicalType type_p, ColumnData *parent)
    : ColumnData(block_manager, info, column_index, start_row, std::move(type_p), parent),
      validity(block_manager, info, 0, start_row, this) {
	D_ASSERT(type.InternalType() == PhysicalType::STRUCT);
	auto &child_types = StructType::GetChildTypes(type);
	D_ASSERT(child_types.size() > 0);
	// the sub column index, starting at 1 (0 is the validity mask)
	idx_t sub_column_index = 1;
	for (auto &child_type : child_types) {
		sub_columns.push_back(
		    ColumnData::CreateColumnUnique(block_manager, info, sub_column_index, start_row, child_type.second, this));
		sub_column_index++;
	}
}

StructColumnData::StructColumnData(ColumnData &original, idx_t start_row, ColumnData *parent)
    : ColumnData(original, start_row, parent), validity(((StructColumnData &)original).validity, start_row, this) {
	auto &struct_data = (StructColumnData &)original;
	for (auto &child_col : struct_data.sub_columns) {
		sub_columns.push_back(ColumnData::CreateColumnUnique(*child_col, start_row, this));
	}
}

bool StructColumnData::CheckZonemap(ColumnScanState &state, TableFilter &filter) {
	// table filters are not supported yet for struct columns
	return false;
}

idx_t StructColumnData::GetMaxEntry() {
	return sub_columns[0]->GetMaxEntry();
}

void StructColumnData::InitializeScan(ColumnScanState &state) {
	D_ASSERT(state.child_states.empty());

	state.row_index = 0;
	state.current = nullptr;

	// initialize the validity segment
	ColumnScanState validity_state;
	validity.InitializeScan(validity_state);
	state.child_states.push_back(std::move(validity_state));

	// initialize the sub-columns
	for (auto &sub_column : sub_columns) {
		ColumnScanState child_state;
		sub_column->InitializeScan(child_state);
		state.child_states.push_back(std::move(child_state));
	}
}

void StructColumnData::InitializeScanWithOffset(ColumnScanState &state, idx_t row_idx) {
	D_ASSERT(state.child_states.empty());

	state.row_index = row_idx;
	state.current = nullptr;

	// initialize the validity segment
	ColumnScanState validity_state;
	validity.InitializeScanWithOffset(validity_state, row_idx);
	state.child_states.push_back(std::move(validity_state));

	// initialize the sub-columns
	for (auto &sub_column : sub_columns) {
		ColumnScanState child_state;
		sub_column->InitializeScanWithOffset(child_state, row_idx);
		state.child_states.push_back(std::move(child_state));
	}
}

idx_t StructColumnData::Scan(TransactionData transaction, idx_t vector_index, ColumnScanState &state, Vector &result) {
	auto scan_count = validity.Scan(transaction, vector_index, state.child_states[0], result);
	auto &child_entries = StructVector::GetEntries(result);
	for (idx_t i = 0; i < sub_columns.size(); i++) {
		sub_columns[i]->Scan(transaction, vector_index, state.child_states[i + 1], *child_entries[i]);
	}
	return scan_count;
}

idx_t StructColumnData::ScanCommitted(idx_t vector_index, ColumnScanState &state, Vector &result, bool allow_updates) {
	auto scan_count = validity.ScanCommitted(vector_index, state.child_states[0], result, allow_updates);
	auto &child_entries = StructVector::GetEntries(result);
	for (idx_t i = 0; i < sub_columns.size(); i++) {
		sub_columns[i]->ScanCommitted(vector_index, state.child_states[i + 1], *child_entries[i], allow_updates);
	}
	return scan_count;
}

idx_t StructColumnData::ScanCount(ColumnScanState &state, Vector &result, idx_t count) {
	auto scan_count = validity.ScanCount(state.child_states[0], result, count);
	auto &child_entries = StructVector::GetEntries(result);
	for (idx_t i = 0; i < sub_columns.size(); i++) {
		sub_columns[i]->ScanCount(state.child_states[i + 1], *child_entries[i], count);
	}
	return scan_count;
}

void StructColumnData::Skip(ColumnScanState &state, idx_t count) {
	validity.Skip(state.child_states[0], count);

	// skip inside the sub-columns
	for (idx_t child_idx = 0; child_idx < sub_columns.size(); child_idx++) {
		sub_columns[child_idx]->Skip(state.child_states[child_idx + 1], count);
	}
}

void StructColumnData::InitializeAppend(ColumnAppendState &state) {
	ColumnAppendState validity_append;
	validity.InitializeAppend(validity_append);
	state.child_appends.push_back(std::move(validity_append));

	for (auto &sub_column : sub_columns) {
		ColumnAppendState child_append;
		sub_column->InitializeAppend(child_append);
		state.child_appends.push_back(std::move(child_append));
	}
}

void StructColumnData::Append(BaseStatistics &stats, ColumnAppendState &state, Vector &vector, idx_t count) {
	vector.Flatten(count);

	// append the null values
	validity.Append(*stats.validity_stats, state.child_appends[0], vector, count);

	auto &struct_stats = (StructStatistics &)stats;
	auto &child_entries = StructVector::GetEntries(vector);
	for (idx_t i = 0; i < child_entries.size(); i++) {
		sub_columns[i]->Append(*struct_stats.child_stats[i], state.child_appends[i + 1], *child_entries[i], count);
	}
}

void StructColumnData::RevertAppend(row_t start_row) {
	validity.RevertAppend(start_row);
	for (auto &sub_column : sub_columns) {
		sub_column->RevertAppend(start_row);
	}
}

idx_t StructColumnData::Fetch(ColumnScanState &state, row_t row_id, Vector &result) {
	// fetch validity mask
	auto &child_entries = StructVector::GetEntries(result);
	// insert any child states that are required
	for (idx_t i = state.child_states.size(); i < child_entries.size() + 1; i++) {
		ColumnScanState child_state;
		state.child_states.push_back(std::move(child_state));
	}
	// fetch the validity state
	idx_t scan_count = validity.Fetch(state.child_states[0], row_id, result);
	// fetch the sub-column states
	for (idx_t i = 0; i < child_entries.size(); i++) {
		sub_columns[i]->Fetch(state.child_states[i + 1], row_id, *child_entries[i]);
	}
	return scan_count;
}

void StructColumnData::Update(TransactionData transaction, idx_t column_index, Vector &update_vector, row_t *row_ids,
                              idx_t update_count) {
	validity.Update(transaction, column_index, update_vector, row_ids, update_count);
	auto &child_entries = StructVector::GetEntries(update_vector);
	for (idx_t i = 0; i < child_entries.size(); i++) {
		sub_columns[i]->Update(transaction, column_index, *child_entries[i], row_ids, update_count);
	}
}

void StructColumnData::UpdateColumn(TransactionData transaction, const vector<column_t> &column_path,
                                    Vector &update_vector, row_t *row_ids, idx_t update_count, idx_t depth) {
	// we can never DIRECTLY update a struct column
	if (depth >= column_path.size()) {
		throw InternalException("Attempting to directly update a struct column - this should not be possible");
	}
	auto update_column = column_path[depth];
	if (update_column == 0) {
		// update the validity column
		validity.UpdateColumn(transaction, column_path, update_vector, row_ids, update_count, depth + 1);
	} else {
		if (update_column > sub_columns.size()) {
			throw InternalException("Update column_path out of range");
		}
		sub_columns[update_column - 1]->UpdateColumn(transaction, column_path, update_vector, row_ids, update_count,
		                                             depth + 1);
	}
}

unique_ptr<BaseStatistics> StructColumnData::GetUpdateStatistics() {
	// check if any child column has updates
	auto stats = BaseStatistics::CreateEmpty(type, StatisticsType::GLOBAL_STATS);
	auto &struct_stats = (StructStatistics &)*stats;
	stats->validity_stats = validity.GetUpdateStatistics();
	for (idx_t i = 0; i < sub_columns.size(); i++) {
		auto child_stats = sub_columns[i]->GetUpdateStatistics();
		if (child_stats) {
			struct_stats.child_stats[i] = std::move(child_stats);
		}
	}
	return stats;
}

void StructColumnData::FetchRow(TransactionData transaction, ColumnFetchState &state, row_t row_id, Vector &result,
                                idx_t result_idx) {
	// fetch validity mask
	auto &child_entries = StructVector::GetEntries(result);
	// insert any child states that are required
	for (idx_t i = state.child_states.size(); i < child_entries.size() + 1; i++) {
		auto child_state = make_unique<ColumnFetchState>();
		state.child_states.push_back(std::move(child_state));
	}
	// fetch the validity state
	validity.FetchRow(transaction, *state.child_states[0], row_id, result, result_idx);
	// fetch the sub-column states
	for (idx_t i = 0; i < child_entries.size(); i++) {
		sub_columns[i]->FetchRow(transaction, *state.child_states[i + 1], row_id, *child_entries[i], result_idx);
	}
}

void StructColumnData::CommitDropColumn() {
	validity.CommitDropColumn();
	for (auto &sub_column : sub_columns) {
		sub_column->CommitDropColumn();
	}
}

struct StructColumnCheckpointState : public ColumnCheckpointState {
	StructColumnCheckpointState(RowGroup &row_group, ColumnData &column_data,
	                            PartialBlockManager &partial_block_manager)
	    : ColumnCheckpointState(row_group, column_data, partial_block_manager) {
		global_stats = make_unique<StructStatistics>(column_data.type);
	}

	unique_ptr<ColumnCheckpointState> validity_state;
	vector<unique_ptr<ColumnCheckpointState>> child_states;

public:
	unique_ptr<BaseStatistics> GetStatistics() override {
		auto stats = make_unique<StructStatistics>(column_data.type);
		D_ASSERT(stats->child_stats.size() == child_states.size());
		stats->validity_stats = validity_state->GetStatistics();
		for (idx_t i = 0; i < child_states.size(); i++) {
			stats->child_stats[i] = child_states[i]->GetStatistics();
			D_ASSERT(stats->child_stats[i]);
		}
		return std::move(stats);
	}

	void WriteDataPointers(RowGroupWriter &writer) override {
		validity_state->WriteDataPointers(writer);
		for (auto &state : child_states) {
			state->WriteDataPointers(writer);
		}
	}
	void GetBlockIds(unordered_set<block_id_t> &result) override {
		validity_state->GetBlockIds(result);
		for (auto &state : child_states) {
			state->GetBlockIds(result);
		}
	}
};

unique_ptr<ColumnCheckpointState> StructColumnData::CreateCheckpointState(RowGroup &row_group,
                                                                          PartialBlockManager &partial_block_manager) {
	return make_unique<StructColumnCheckpointState>(row_group, *this, partial_block_manager);
}

unique_ptr<ColumnCheckpointState> StructColumnData::Checkpoint(RowGroup &row_group,
                                                               PartialBlockManager &partial_block_manager,
                                                               ColumnCheckpointInfo &checkpoint_info) {
	auto checkpoint_state = make_unique<StructColumnCheckpointState>(row_group, *this, partial_block_manager);
	checkpoint_state->validity_state = validity.Checkpoint(row_group, partial_block_manager, checkpoint_info);
	for (auto &sub_column : sub_columns) {
		checkpoint_state->child_states.push_back(
		    sub_column->Checkpoint(row_group, partial_block_manager, checkpoint_info));
	}
	return std::move(checkpoint_state);
}

void StructColumnData::DeserializeColumn(Deserializer &source) {
	validity.DeserializeColumn(source);
	for (auto &sub_column : sub_columns) {
		sub_column->DeserializeColumn(source);
	}
}

void StructColumnData::GetStorageInfo(idx_t row_group_index, vector<idx_t> col_path, TableStorageInfo &result) {
	col_path.push_back(0);
	validity.GetStorageInfo(row_group_index, col_path, result);
	for (idx_t i = 0; i < sub_columns.size(); i++) {
		col_path.back() = i + 1;
		sub_columns[i]->GetStorageInfo(row_group_index, col_path, result);
	}
}

void StructColumnData::Verify(RowGroup &parent) {
#ifdef DEBUG
	ColumnData::Verify(parent);
	validity.Verify(parent);
	for (auto &sub_column : sub_columns) {
		sub_column->Verify(parent);
	}
#endif
}

} // namespace duckdb



namespace duckdb {

void TableStatistics::Initialize(const vector<LogicalType> &types, PersistentTableData &data) {
	D_ASSERT(Empty());

	column_stats.reserve(data.column_stats.size());
	for (auto &stats : data.column_stats) {
		column_stats.push_back(make_shared<ColumnStatistics>(std::move(stats)));
	}
	if (column_stats.size() != types.size()) { // LCOV_EXCL_START
		throw IOException("Table statistics column count is not aligned with table column count. Corrupt file?");
	} // LCOV_EXCL_STOP
}

void TableStatistics::InitializeEmpty(const vector<LogicalType> &types) {
	D_ASSERT(Empty());

	for (auto &type : types) {
		column_stats.push_back(ColumnStatistics::CreateEmptyStats(type));
	}
}

void TableStatistics::InitializeAddColumn(TableStatistics &parent, const LogicalType &new_column_type) {
	D_ASSERT(Empty());

	lock_guard<mutex> stats_lock(parent.stats_lock);
	for (idx_t i = 0; i < parent.column_stats.size(); i++) {
		column_stats.push_back(parent.column_stats[i]);
	}
	column_stats.push_back(ColumnStatistics::CreateEmptyStats(new_column_type));
}

void TableStatistics::InitializeRemoveColumn(TableStatistics &parent, idx_t removed_column) {
	D_ASSERT(Empty());

	lock_guard<mutex> stats_lock(parent.stats_lock);
	for (idx_t i = 0; i < parent.column_stats.size(); i++) {
		if (i != removed_column) {
			column_stats.push_back(parent.column_stats[i]);
		}
	}
}

void TableStatistics::InitializeAlterType(TableStatistics &parent, idx_t changed_idx, const LogicalType &new_type) {
	D_ASSERT(Empty());

	lock_guard<mutex> stats_lock(parent.stats_lock);
	for (idx_t i = 0; i < parent.column_stats.size(); i++) {
		if (i == changed_idx) {
			column_stats.push_back(ColumnStatistics::CreateEmptyStats(new_type));
		} else {
			column_stats.push_back(parent.column_stats[i]);
		}
	}
}

void TableStatistics::InitializeAddConstraint(TableStatistics &parent) {
	D_ASSERT(Empty());

	lock_guard<mutex> stats_lock(parent.stats_lock);
	for (idx_t i = 0; i < parent.column_stats.size(); i++) {
		column_stats.push_back(parent.column_stats[i]);
	}
}

void TableStatistics::MergeStats(TableStatistics &other) {
	auto l = GetLock();
	D_ASSERT(column_stats.size() == other.column_stats.size());
	for (idx_t i = 0; i < column_stats.size(); i++) {
		column_stats[i]->stats->Merge(*other.column_stats[i]->stats);
	}
}

void TableStatistics::MergeStats(idx_t i, BaseStatistics &stats) {
	auto l = GetLock();
	MergeStats(*l, i, stats);
}

void TableStatistics::MergeStats(TableStatisticsLock &lock, idx_t i, BaseStatistics &stats) {
	column_stats[i]->stats->Merge(stats);
}

ColumnStatistics &TableStatistics::GetStats(idx_t i) {
	return *column_stats[i];
}

unique_ptr<BaseStatistics> TableStatistics::CopyStats(idx_t i) {
	lock_guard<mutex> l(stats_lock);
	return column_stats[i]->stats->Copy();
}

unique_ptr<TableStatisticsLock> TableStatistics::GetLock() {
	return make_unique<TableStatisticsLock>(stats_lock);
}

bool TableStatistics::Empty() {
	return column_stats.empty();
}

} // namespace duckdb










namespace duckdb {

static UpdateSegment::initialize_update_function_t GetInitializeUpdateFunction(PhysicalType type);
static UpdateSegment::fetch_update_function_t GetFetchUpdateFunction(PhysicalType type);
static UpdateSegment::fetch_committed_function_t GetFetchCommittedFunction(PhysicalType type);
static UpdateSegment::fetch_committed_range_function_t GetFetchCommittedRangeFunction(PhysicalType type);

static UpdateSegment::merge_update_function_t GetMergeUpdateFunction(PhysicalType type);
static UpdateSegment::rollback_update_function_t GetRollbackUpdateFunction(PhysicalType type);
static UpdateSegment::statistics_update_function_t GetStatisticsUpdateFunction(PhysicalType type);
static UpdateSegment::fetch_row_function_t GetFetchRowFunction(PhysicalType type);

UpdateSegment::UpdateSegment(ColumnData &column_data)
    : column_data(column_data), stats(column_data.type), heap(BufferAllocator::Get(column_data.GetDatabase())) {
	auto physical_type = column_data.type.InternalType();

	this->type_size = GetTypeIdSize(physical_type);

	this->initialize_update_function = GetInitializeUpdateFunction(physical_type);
	this->fetch_update_function = GetFetchUpdateFunction(physical_type);
	this->fetch_committed_function = GetFetchCommittedFunction(physical_type);
	this->fetch_committed_range = GetFetchCommittedRangeFunction(physical_type);
	this->fetch_row_function = GetFetchRowFunction(physical_type);
	this->merge_update_function = GetMergeUpdateFunction(physical_type);
	this->rollback_update_function = GetRollbackUpdateFunction(physical_type);
	this->statistics_update_function = GetStatisticsUpdateFunction(physical_type);
}

UpdateSegment::~UpdateSegment() {
}

void UpdateSegment::ClearUpdates() {
	stats.Reset();
	root.reset();
	heap.Destroy();
}

//===--------------------------------------------------------------------===//
// Update Info Helpers
//===--------------------------------------------------------------------===//
Value UpdateInfo::GetValue(idx_t index) {
	auto &type = segment->column_data.type;

	switch (type.id()) {
	case LogicalTypeId::VALIDITY:
		return Value::BOOLEAN(((bool *)tuple_data)[index]);
	case LogicalTypeId::INTEGER:
		return Value::INTEGER(((int32_t *)tuple_data)[index]);
	default:
		throw NotImplementedException("Unimplemented type for UpdateInfo::GetValue");
	}
}

void UpdateInfo::Print() {
	Printer::Print(ToString());
}

string UpdateInfo::ToString() {
	auto &type = segment->column_data.type;
	string result = "Update Info [" + type.ToString() + ", Count: " + to_string(N) +
	                ", Transaction Id: " + to_string(version_number) + "]\n";
	for (idx_t i = 0; i < N; i++) {
		result += to_string(tuples[i]) + ": " + GetValue(i).ToString() + "\n";
	}
	if (next) {
		result += "\nChild Segment: " + next->ToString();
	}
	return result;
}

void UpdateInfo::Verify() {
#ifdef DEBUG
	for (idx_t i = 1; i < N; i++) {
		D_ASSERT(tuples[i] > tuples[i - 1] && tuples[i] < STANDARD_VECTOR_SIZE);
	}
#endif
}

//===--------------------------------------------------------------------===//
// Update Fetch
//===--------------------------------------------------------------------===//
static void MergeValidityInfo(UpdateInfo *current, ValidityMask &result_mask) {
	auto info_data = (bool *)current->tuple_data;
	for (idx_t i = 0; i < current->N; i++) {
		result_mask.Set(current->tuples[i], info_data[i]);
	}
}

static void UpdateMergeValidity(transaction_t start_time, transaction_t transaction_id, UpdateInfo *info,
                                Vector &result) {
	auto &result_mask = FlatVector::Validity(result);
	UpdateInfo::UpdatesForTransaction(info, start_time, transaction_id,
	                                  [&](UpdateInfo *current) { MergeValidityInfo(current, result_mask); });
}

template <class T>
static void MergeUpdateInfo(UpdateInfo *current, T *result_data) {
	auto info_data = (T *)current->tuple_data;
	if (current->N == STANDARD_VECTOR_SIZE) {
		// special case: update touches ALL tuples of this vector
		// in this case we can just memcpy the data
		// since the layout of the update info is guaranteed to be [0, 1, 2, 3, ...]
		memcpy(result_data, info_data, sizeof(T) * current->N);
	} else {
		for (idx_t i = 0; i < current->N; i++) {
			result_data[current->tuples[i]] = info_data[i];
		}
	}
}

template <class T>
static void UpdateMergeFetch(transaction_t start_time, transaction_t transaction_id, UpdateInfo *info, Vector &result) {
	auto result_data = FlatVector::GetData<T>(result);
	UpdateInfo::UpdatesForTransaction(info, start_time, transaction_id,
	                                  [&](UpdateInfo *current) { MergeUpdateInfo<T>(current, result_data); });
}

static UpdateSegment::fetch_update_function_t GetFetchUpdateFunction(PhysicalType type) {
	switch (type) {
	case PhysicalType::BIT:
		return UpdateMergeValidity;
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		return UpdateMergeFetch<int8_t>;
	case PhysicalType::INT16:
		return UpdateMergeFetch<int16_t>;
	case PhysicalType::INT32:
		return UpdateMergeFetch<int32_t>;
	case PhysicalType::INT64:
		return UpdateMergeFetch<int64_t>;
	case PhysicalType::UINT8:
		return UpdateMergeFetch<uint8_t>;
	case PhysicalType::UINT16:
		return UpdateMergeFetch<uint16_t>;
	case PhysicalType::UINT32:
		return UpdateMergeFetch<uint32_t>;
	case PhysicalType::UINT64:
		return UpdateMergeFetch<uint64_t>;
	case PhysicalType::INT128:
		return UpdateMergeFetch<hugeint_t>;
	case PhysicalType::FLOAT:
		return UpdateMergeFetch<float>;
	case PhysicalType::DOUBLE:
		return UpdateMergeFetch<double>;
	case PhysicalType::INTERVAL:
		return UpdateMergeFetch<interval_t>;
	case PhysicalType::VARCHAR:
		return UpdateMergeFetch<string_t>;
	default:
		throw NotImplementedException("Unimplemented type for update segment");
	}
}

void UpdateSegment::FetchUpdates(TransactionData transaction, idx_t vector_index, Vector &result) {
	auto lock_handle = lock.GetSharedLock();
	if (!root) {
		return;
	}
	if (!root->info[vector_index]) {
		return;
	}
	// FIXME: normalify if this is not the case... need to pass in count?
	D_ASSERT(result.GetVectorType() == VectorType::FLAT_VECTOR);

	fetch_update_function(transaction.start_time, transaction.transaction_id, root->info[vector_index]->info.get(),
	                      result);
}

//===--------------------------------------------------------------------===//
// Fetch Committed
//===--------------------------------------------------------------------===//
static void FetchCommittedValidity(UpdateInfo *info, Vector &result) {
	auto &result_mask = FlatVector::Validity(result);
	MergeValidityInfo(info, result_mask);
}

template <class T>
static void TemplatedFetchCommitted(UpdateInfo *info, Vector &result) {
	auto result_data = FlatVector::GetData<T>(result);
	MergeUpdateInfo<T>(info, result_data);
}

static UpdateSegment::fetch_committed_function_t GetFetchCommittedFunction(PhysicalType type) {
	switch (type) {
	case PhysicalType::BIT:
		return FetchCommittedValidity;
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		return TemplatedFetchCommitted<int8_t>;
	case PhysicalType::INT16:
		return TemplatedFetchCommitted<int16_t>;
	case PhysicalType::INT32:
		return TemplatedFetchCommitted<int32_t>;
	case PhysicalType::INT64:
		return TemplatedFetchCommitted<int64_t>;
	case PhysicalType::UINT8:
		return TemplatedFetchCommitted<uint8_t>;
	case PhysicalType::UINT16:
		return TemplatedFetchCommitted<uint16_t>;
	case PhysicalType::UINT32:
		return TemplatedFetchCommitted<uint32_t>;
	case PhysicalType::UINT64:
		return TemplatedFetchCommitted<uint64_t>;
	case PhysicalType::INT128:
		return TemplatedFetchCommitted<hugeint_t>;
	case PhysicalType::FLOAT:
		return TemplatedFetchCommitted<float>;
	case PhysicalType::DOUBLE:
		return TemplatedFetchCommitted<double>;
	case PhysicalType::INTERVAL:
		return TemplatedFetchCommitted<interval_t>;
	case PhysicalType::VARCHAR:
		return TemplatedFetchCommitted<string_t>;
	default:
		throw NotImplementedException("Unimplemented type for update segment");
	}
}

void UpdateSegment::FetchCommitted(idx_t vector_index, Vector &result) {
	auto lock_handle = lock.GetSharedLock();

	if (!root) {
		return;
	}
	if (!root->info[vector_index]) {
		return;
	}
	// FIXME: normalify if this is not the case... need to pass in count?
	D_ASSERT(result.GetVectorType() == VectorType::FLAT_VECTOR);

	fetch_committed_function(root->info[vector_index]->info.get(), result);
}

//===--------------------------------------------------------------------===//
// Fetch Range
//===--------------------------------------------------------------------===//
static void MergeUpdateInfoRangeValidity(UpdateInfo *current, idx_t start, idx_t end, idx_t result_offset,
                                         ValidityMask &result_mask) {
	auto info_data = (bool *)current->tuple_data;
	for (idx_t i = 0; i < current->N; i++) {
		auto tuple_idx = current->tuples[i];
		if (tuple_idx < start) {
			continue;
		} else if (tuple_idx >= end) {
			break;
		}
		auto result_idx = result_offset + tuple_idx - start;
		result_mask.Set(result_idx, info_data[i]);
	}
}

static void FetchCommittedRangeValidity(UpdateInfo *info, idx_t start, idx_t end, idx_t result_offset, Vector &result) {
	auto &result_mask = FlatVector::Validity(result);
	MergeUpdateInfoRangeValidity(info, start, end, result_offset, result_mask);
}

template <class T>
static void MergeUpdateInfoRange(UpdateInfo *current, idx_t start, idx_t end, idx_t result_offset, T *result_data) {
	auto info_data = (T *)current->tuple_data;
	for (idx_t i = 0; i < current->N; i++) {
		auto tuple_idx = current->tuples[i];
		if (tuple_idx < start) {
			continue;
		} else if (tuple_idx >= end) {
			break;
		}
		auto result_idx = result_offset + tuple_idx - start;
		result_data[result_idx] = info_data[i];
	}
}

template <class T>
static void TemplatedFetchCommittedRange(UpdateInfo *info, idx_t start, idx_t end, idx_t result_offset,
                                         Vector &result) {
	auto result_data = FlatVector::GetData<T>(result);
	MergeUpdateInfoRange<T>(info, start, end, result_offset, result_data);
}

static UpdateSegment::fetch_committed_range_function_t GetFetchCommittedRangeFunction(PhysicalType type) {
	switch (type) {
	case PhysicalType::BIT:
		return FetchCommittedRangeValidity;
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		return TemplatedFetchCommittedRange<int8_t>;
	case PhysicalType::INT16:
		return TemplatedFetchCommittedRange<int16_t>;
	case PhysicalType::INT32:
		return TemplatedFetchCommittedRange<int32_t>;
	case PhysicalType::INT64:
		return TemplatedFetchCommittedRange<int64_t>;
	case PhysicalType::UINT8:
		return TemplatedFetchCommittedRange<uint8_t>;
	case PhysicalType::UINT16:
		return TemplatedFetchCommittedRange<uint16_t>;
	case PhysicalType::UINT32:
		return TemplatedFetchCommittedRange<uint32_t>;
	case PhysicalType::UINT64:
		return TemplatedFetchCommittedRange<uint64_t>;
	case PhysicalType::INT128:
		return TemplatedFetchCommittedRange<hugeint_t>;
	case PhysicalType::FLOAT:
		return TemplatedFetchCommittedRange<float>;
	case PhysicalType::DOUBLE:
		return TemplatedFetchCommittedRange<double>;
	case PhysicalType::INTERVAL:
		return TemplatedFetchCommittedRange<interval_t>;
	case PhysicalType::VARCHAR:
		return TemplatedFetchCommittedRange<string_t>;
	default:
		throw NotImplementedException("Unimplemented type for update segment");
	}
}

void UpdateSegment::FetchCommittedRange(idx_t start_row, idx_t count, Vector &result) {
	D_ASSERT(count > 0);
	if (!root) {
		return;
	}
	D_ASSERT(result.GetVectorType() == VectorType::FLAT_VECTOR);

	idx_t end_row = start_row + count;
	idx_t start_vector = start_row / STANDARD_VECTOR_SIZE;
	idx_t end_vector = (end_row - 1) / STANDARD_VECTOR_SIZE;
	D_ASSERT(start_vector <= end_vector);
	D_ASSERT(end_vector < RowGroup::ROW_GROUP_VECTOR_COUNT);

	for (idx_t vector_idx = start_vector; vector_idx <= end_vector; vector_idx++) {
		if (!root->info[vector_idx]) {
			continue;
		}
		idx_t start_in_vector = vector_idx == start_vector ? start_row - start_vector * STANDARD_VECTOR_SIZE : 0;
		idx_t end_in_vector =
		    vector_idx == end_vector ? end_row - end_vector * STANDARD_VECTOR_SIZE : STANDARD_VECTOR_SIZE;
		D_ASSERT(start_in_vector < end_in_vector);
		D_ASSERT(end_in_vector > 0 && end_in_vector <= STANDARD_VECTOR_SIZE);
		idx_t result_offset = ((vector_idx * STANDARD_VECTOR_SIZE) + start_in_vector) - start_row;
		fetch_committed_range(root->info[vector_idx]->info.get(), start_in_vector, end_in_vector, result_offset,
		                      result);
	}
}

//===--------------------------------------------------------------------===//
// Fetch Row
//===--------------------------------------------------------------------===//
static void FetchRowValidity(transaction_t start_time, transaction_t transaction_id, UpdateInfo *info, idx_t row_idx,
                             Vector &result, idx_t result_idx) {
	auto &result_mask = FlatVector::Validity(result);
	UpdateInfo::UpdatesForTransaction(info, start_time, transaction_id, [&](UpdateInfo *current) {
		auto info_data = (bool *)current->tuple_data;
		// FIXME: we could do a binary search in here
		for (idx_t i = 0; i < current->N; i++) {
			if (current->tuples[i] == row_idx) {
				result_mask.Set(result_idx, info_data[i]);
				break;
			} else if (current->tuples[i] > row_idx) {
				break;
			}
		}
	});
}

template <class T>
static void TemplatedFetchRow(transaction_t start_time, transaction_t transaction_id, UpdateInfo *info, idx_t row_idx,
                              Vector &result, idx_t result_idx) {
	auto result_data = FlatVector::GetData<T>(result);
	UpdateInfo::UpdatesForTransaction(info, start_time, transaction_id, [&](UpdateInfo *current) {
		auto info_data = (T *)current->tuple_data;
		// FIXME: we could do a binary search in here
		for (idx_t i = 0; i < current->N; i++) {
			if (current->tuples[i] == row_idx) {
				result_data[result_idx] = info_data[i];
				break;
			} else if (current->tuples[i] > row_idx) {
				break;
			}
		}
	});
}

static UpdateSegment::fetch_row_function_t GetFetchRowFunction(PhysicalType type) {
	switch (type) {
	case PhysicalType::BIT:
		return FetchRowValidity;
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		return TemplatedFetchRow<int8_t>;
	case PhysicalType::INT16:
		return TemplatedFetchRow<int16_t>;
	case PhysicalType::INT32:
		return TemplatedFetchRow<int32_t>;
	case PhysicalType::INT64:
		return TemplatedFetchRow<int64_t>;
	case PhysicalType::UINT8:
		return TemplatedFetchRow<uint8_t>;
	case PhysicalType::UINT16:
		return TemplatedFetchRow<uint16_t>;
	case PhysicalType::UINT32:
		return TemplatedFetchRow<uint32_t>;
	case PhysicalType::UINT64:
		return TemplatedFetchRow<uint64_t>;
	case PhysicalType::INT128:
		return TemplatedFetchRow<hugeint_t>;
	case PhysicalType::FLOAT:
		return TemplatedFetchRow<float>;
	case PhysicalType::DOUBLE:
		return TemplatedFetchRow<double>;
	case PhysicalType::INTERVAL:
		return TemplatedFetchRow<interval_t>;
	case PhysicalType::VARCHAR:
		return TemplatedFetchRow<string_t>;
	default:
		throw NotImplementedException("Unimplemented type for update segment fetch row");
	}
}

void UpdateSegment::FetchRow(TransactionData transaction, idx_t row_id, Vector &result, idx_t result_idx) {
	if (!root) {
		return;
	}
	idx_t vector_index = (row_id - column_data.start) / STANDARD_VECTOR_SIZE;
	if (!root->info[vector_index]) {
		return;
	}
	idx_t row_in_vector = row_id - vector_index * STANDARD_VECTOR_SIZE;
	fetch_row_function(transaction.start_time, transaction.transaction_id, root->info[vector_index]->info.get(),
	                   row_in_vector, result, result_idx);
}

//===--------------------------------------------------------------------===//
// Rollback update
//===--------------------------------------------------------------------===//
template <class T>
static void RollbackUpdate(UpdateInfo *base_info, UpdateInfo *rollback_info) {
	auto base_data = (T *)base_info->tuple_data;
	auto rollback_data = (T *)rollback_info->tuple_data;
	idx_t base_offset = 0;
	for (idx_t i = 0; i < rollback_info->N; i++) {
		auto id = rollback_info->tuples[i];
		while (base_info->tuples[base_offset] < id) {
			base_offset++;
			D_ASSERT(base_offset < base_info->N);
		}
		base_data[base_offset] = rollback_data[i];
	}
}

static UpdateSegment::rollback_update_function_t GetRollbackUpdateFunction(PhysicalType type) {
	switch (type) {
	case PhysicalType::BIT:
		return RollbackUpdate<bool>;
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		return RollbackUpdate<int8_t>;
	case PhysicalType::INT16:
		return RollbackUpdate<int16_t>;
	case PhysicalType::INT32:
		return RollbackUpdate<int32_t>;
	case PhysicalType::INT64:
		return RollbackUpdate<int64_t>;
	case PhysicalType::UINT8:
		return RollbackUpdate<uint8_t>;
	case PhysicalType::UINT16:
		return RollbackUpdate<uint16_t>;
	case PhysicalType::UINT32:
		return RollbackUpdate<uint32_t>;
	case PhysicalType::UINT64:
		return RollbackUpdate<uint64_t>;
	case PhysicalType::INT128:
		return RollbackUpdate<hugeint_t>;
	case PhysicalType::FLOAT:
		return RollbackUpdate<float>;
	case PhysicalType::DOUBLE:
		return RollbackUpdate<double>;
	case PhysicalType::INTERVAL:
		return RollbackUpdate<interval_t>;
	case PhysicalType::VARCHAR:
		return RollbackUpdate<string_t>;
	default:
		throw NotImplementedException("Unimplemented type for uncompressed segment");
	}
}

void UpdateSegment::RollbackUpdate(UpdateInfo *info) {
	// obtain an exclusive lock
	auto lock_handle = lock.GetExclusiveLock();

	// move the data from the UpdateInfo back into the base info
	D_ASSERT(root->info[info->vector_index]);
	rollback_update_function(root->info[info->vector_index]->info.get(), info);

	// clean up the update chain
	CleanupUpdateInternal(*lock_handle, info);
}

//===--------------------------------------------------------------------===//
// Cleanup Update
//===--------------------------------------------------------------------===//
void UpdateSegment::CleanupUpdateInternal(const StorageLockKey &lock, UpdateInfo *info) {
	D_ASSERT(info->prev);
	auto prev = info->prev;
	prev->next = info->next;
	if (prev->next) {
		prev->next->prev = prev;
	}
}

void UpdateSegment::CleanupUpdate(UpdateInfo *info) {
	// obtain an exclusive lock
	auto lock_handle = lock.GetExclusiveLock();
	CleanupUpdateInternal(*lock_handle, info);
}

//===--------------------------------------------------------------------===//
// Check for conflicts in update
//===--------------------------------------------------------------------===//
static void CheckForConflicts(UpdateInfo *info, TransactionData transaction, row_t *ids, const SelectionVector &sel,
                              idx_t count, row_t offset, UpdateInfo *&node) {
	if (!info) {
		return;
	}
	if (info->version_number == transaction.transaction_id) {
		// this UpdateInfo belongs to the current transaction, set it in the node
		node = info;
	} else if (info->version_number > transaction.start_time) {
		// potential conflict, check that tuple ids do not conflict
		// as both ids and info->tuples are sorted, this is similar to a merge join
		idx_t i = 0, j = 0;
		while (true) {
			auto id = ids[sel.get_index(i)] - offset;
			if (id == info->tuples[j]) {
				throw TransactionException("Conflict on update!");
			} else if (id < info->tuples[j]) {
				// id < the current tuple in info, move to next id
				i++;
				if (i == count) {
					break;
				}
			} else {
				// id > the current tuple, move to next tuple in info
				j++;
				if (j == info->N) {
					break;
				}
			}
		}
	}
	CheckForConflicts(info->next, transaction, ids, sel, count, offset, node);
}

//===--------------------------------------------------------------------===//
// Initialize update info
//===--------------------------------------------------------------------===//
void UpdateSegment::InitializeUpdateInfo(UpdateInfo &info, row_t *ids, const SelectionVector &sel, idx_t count,
                                         idx_t vector_index, idx_t vector_offset) {
	info.segment = this;
	info.vector_index = vector_index;
	info.prev = nullptr;
	info.next = nullptr;

	// set up the tuple ids
	info.N = count;
	for (idx_t i = 0; i < count; i++) {
		auto idx = sel.get_index(i);
		auto id = ids[idx];
		D_ASSERT(idx_t(id) >= vector_offset && idx_t(id) < vector_offset + STANDARD_VECTOR_SIZE);
		info.tuples[i] = id - vector_offset;
	};
}

static void InitializeUpdateValidity(UpdateInfo *base_info, Vector &base_data, UpdateInfo *update_info, Vector &update,
                                     const SelectionVector &sel) {
	auto &update_mask = FlatVector::Validity(update);
	auto tuple_data = (bool *)update_info->tuple_data;

	if (!update_mask.AllValid()) {
		for (idx_t i = 0; i < update_info->N; i++) {
			auto idx = sel.get_index(i);
			tuple_data[i] = update_mask.RowIsValidUnsafe(idx);
		}
	} else {
		for (idx_t i = 0; i < update_info->N; i++) {
			tuple_data[i] = true;
		}
	}

	auto &base_mask = FlatVector::Validity(base_data);
	auto base_tuple_data = (bool *)base_info->tuple_data;
	if (!base_mask.AllValid()) {
		for (idx_t i = 0; i < base_info->N; i++) {
			base_tuple_data[i] = base_mask.RowIsValidUnsafe(base_info->tuples[i]);
		}
	} else {
		for (idx_t i = 0; i < base_info->N; i++) {
			base_tuple_data[i] = true;
		}
	}
}

struct UpdateSelectElement {
	template <class T>
	static T Operation(UpdateSegment *segment, T element) {
		return element;
	}
};

template <>
string_t UpdateSelectElement::Operation(UpdateSegment *segment, string_t element) {
	return element.IsInlined() ? element : segment->GetStringHeap().AddBlob(element);
}

template <class T>
static void InitializeUpdateData(UpdateInfo *base_info, Vector &base_data, UpdateInfo *update_info, Vector &update,
                                 const SelectionVector &sel) {
	auto update_data = FlatVector::GetData<T>(update);
	auto tuple_data = (T *)update_info->tuple_data;

	for (idx_t i = 0; i < update_info->N; i++) {
		auto idx = sel.get_index(i);
		tuple_data[i] = update_data[idx];
	}

	auto base_array_data = FlatVector::GetData<T>(base_data);
	auto &base_validity = FlatVector::Validity(base_data);
	auto base_tuple_data = (T *)base_info->tuple_data;
	for (idx_t i = 0; i < base_info->N; i++) {
		auto base_idx = base_info->tuples[i];
		if (!base_validity.RowIsValid(base_idx)) {
			continue;
		}
		base_tuple_data[i] = UpdateSelectElement::Operation<T>(base_info->segment, base_array_data[base_idx]);
	}
}

static UpdateSegment::initialize_update_function_t GetInitializeUpdateFunction(PhysicalType type) {
	switch (type) {
	case PhysicalType::BIT:
		return InitializeUpdateValidity;
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		return InitializeUpdateData<int8_t>;
	case PhysicalType::INT16:
		return InitializeUpdateData<int16_t>;
	case PhysicalType::INT32:
		return InitializeUpdateData<int32_t>;
	case PhysicalType::INT64:
		return InitializeUpdateData<int64_t>;
	case PhysicalType::UINT8:
		return InitializeUpdateData<uint8_t>;
	case PhysicalType::UINT16:
		return InitializeUpdateData<uint16_t>;
	case PhysicalType::UINT32:
		return InitializeUpdateData<uint32_t>;
	case PhysicalType::UINT64:
		return InitializeUpdateData<uint64_t>;
	case PhysicalType::INT128:
		return InitializeUpdateData<hugeint_t>;
	case PhysicalType::FLOAT:
		return InitializeUpdateData<float>;
	case PhysicalType::DOUBLE:
		return InitializeUpdateData<double>;
	case PhysicalType::INTERVAL:
		return InitializeUpdateData<interval_t>;
	case PhysicalType::VARCHAR:
		return InitializeUpdateData<string_t>;
	default:
		throw NotImplementedException("Unimplemented type for update segment");
	}
}

//===--------------------------------------------------------------------===//
// Merge update info
//===--------------------------------------------------------------------===//
template <class F1, class F2, class F3>
static idx_t MergeLoop(row_t a[], sel_t b[], idx_t acount, idx_t bcount, idx_t aoffset, F1 merge, F2 pick_a, F3 pick_b,
                       const SelectionVector &asel) {
	idx_t aidx = 0, bidx = 0;
	idx_t count = 0;
	while (aidx < acount && bidx < bcount) {
		auto a_index = asel.get_index(aidx);
		auto a_id = a[a_index] - aoffset;
		auto b_id = b[bidx];
		if (a_id == b_id) {
			merge(a_id, a_index, bidx, count);
			aidx++;
			bidx++;
			count++;
		} else if (a_id < b_id) {
			pick_a(a_id, a_index, count);
			aidx++;
			count++;
		} else {
			pick_b(b_id, bidx, count);
			bidx++;
			count++;
		}
	}
	for (; aidx < acount; aidx++) {
		auto a_index = asel.get_index(aidx);
		pick_a(a[a_index] - aoffset, a_index, count);
		count++;
	}
	for (; bidx < bcount; bidx++) {
		pick_b(b[bidx], bidx, count);
		count++;
	}
	return count;
}

struct ExtractStandardEntry {
	template <class T, class V>
	static T Extract(V *data, idx_t entry) {
		return data[entry];
	}
};

struct ExtractValidityEntry {
	template <class T, class V>
	static T Extract(V *data, idx_t entry) {
		return data->RowIsValid(entry);
	}
};

template <class T, class V, class OP = ExtractStandardEntry>
static void MergeUpdateLoopInternal(UpdateInfo *base_info, V *base_table_data, UpdateInfo *update_info,
                                    V *update_vector_data, row_t *ids, idx_t count, const SelectionVector &sel) {
	auto base_id = base_info->segment->column_data.start + base_info->vector_index * STANDARD_VECTOR_SIZE;
#ifdef DEBUG
	// all of these should be sorted, otherwise the below algorithm does not work
	for (idx_t i = 1; i < count; i++) {
		auto prev_idx = sel.get_index(i - 1);
		auto idx = sel.get_index(i);
		D_ASSERT(ids[idx] > ids[prev_idx] && ids[idx] >= row_t(base_id) &&
		         ids[idx] < row_t(base_id + STANDARD_VECTOR_SIZE));
	}
#endif

	// we have a new batch of updates (update, ids, count)
	// we already have existing updates (base_info)
	// and potentially, this transaction already has updates present (update_info)
	// we need to merge these all together so that the latest updates get merged into base_info
	// and the "old" values (fetched from EITHER base_info OR from base_data) get placed into update_info
	auto base_info_data = (T *)base_info->tuple_data;
	auto update_info_data = (T *)update_info->tuple_data;

	// we first do the merging of the old values
	// what we are trying to do here is update the "update_info" of this transaction with all the old data we require
	// this means we need to merge (1) any previously updated values (stored in update_info->tuples)
	// together with (2)
	// to simplify this, we create new arrays here
	// we memcpy these over afterwards
	T result_values[STANDARD_VECTOR_SIZE];
	sel_t result_ids[STANDARD_VECTOR_SIZE];

	idx_t base_info_offset = 0;
	idx_t update_info_offset = 0;
	idx_t result_offset = 0;
	for (idx_t i = 0; i < count; i++) {
		auto idx = sel.get_index(i);
		// we have to merge the info for "ids[i]"
		auto update_id = ids[idx] - base_id;

		while (update_info_offset < update_info->N && update_info->tuples[update_info_offset] < update_id) {
			// old id comes before the current id: write it
			result_values[result_offset] = update_info_data[update_info_offset];
			result_ids[result_offset++] = update_info->tuples[update_info_offset];
			update_info_offset++;
		}
		// write the new id
		if (update_info_offset < update_info->N && update_info->tuples[update_info_offset] == update_id) {
			// we have an id that is equivalent in the current update info: write the update info
			result_values[result_offset] = update_info_data[update_info_offset];
			result_ids[result_offset++] = update_info->tuples[update_info_offset];
			update_info_offset++;
			continue;
		}

		/// now check if we have the current update_id in the base_info, or if we should fetch it from the base data
		while (base_info_offset < base_info->N && base_info->tuples[base_info_offset] < update_id) {
			base_info_offset++;
		}
		if (base_info_offset < base_info->N && base_info->tuples[base_info_offset] == update_id) {
			// it is! we have to move the tuple from base_info->ids[base_info_offset] to update_info
			result_values[result_offset] = base_info_data[base_info_offset];
		} else {
			// it is not! we have to move base_table_data[update_id] to update_info
			result_values[result_offset] = UpdateSelectElement::Operation<T>(
			    base_info->segment, OP::template Extract<T, V>(base_table_data, update_id));
		}
		result_ids[result_offset++] = update_id;
	}
	// write any remaining entries from the old updates
	while (update_info_offset < update_info->N) {
		result_values[result_offset] = update_info_data[update_info_offset];
		result_ids[result_offset++] = update_info->tuples[update_info_offset];
		update_info_offset++;
	}
	// now copy them back
	update_info->N = result_offset;
	memcpy(update_info_data, result_values, result_offset * sizeof(T));
	memcpy(update_info->tuples, result_ids, result_offset * sizeof(sel_t));

	// now we merge the new values into the base_info
	result_offset = 0;
	auto pick_new = [&](idx_t id, idx_t aidx, idx_t count) {
		result_values[result_offset] = OP::template Extract<T, V>(update_vector_data, aidx);
		result_ids[result_offset] = id;
		result_offset++;
	};
	auto pick_old = [&](idx_t id, idx_t bidx, idx_t count) {
		result_values[result_offset] = base_info_data[bidx];
		result_ids[result_offset] = id;
		result_offset++;
	};
	// now we perform a merge of the new ids with the old ids
	auto merge = [&](idx_t id, idx_t aidx, idx_t bidx, idx_t count) {
		pick_new(id, aidx, count);
	};
	MergeLoop(ids, base_info->tuples, count, base_info->N, base_id, merge, pick_new, pick_old, sel);

	base_info->N = result_offset;
	memcpy(base_info_data, result_values, result_offset * sizeof(T));
	memcpy(base_info->tuples, result_ids, result_offset * sizeof(sel_t));
}

static void MergeValidityLoop(UpdateInfo *base_info, Vector &base_data, UpdateInfo *update_info, Vector &update,
                              row_t *ids, idx_t count, const SelectionVector &sel) {
	auto &base_validity = FlatVector::Validity(base_data);
	auto &update_validity = FlatVector::Validity(update);
	MergeUpdateLoopInternal<bool, ValidityMask, ExtractValidityEntry>(base_info, &base_validity, update_info,
	                                                                  &update_validity, ids, count, sel);
}

template <class T>
static void MergeUpdateLoop(UpdateInfo *base_info, Vector &base_data, UpdateInfo *update_info, Vector &update,
                            row_t *ids, idx_t count, const SelectionVector &sel) {
	auto base_table_data = FlatVector::GetData<T>(base_data);
	auto update_vector_data = FlatVector::GetData<T>(update);
	MergeUpdateLoopInternal<T, T>(base_info, base_table_data, update_info, update_vector_data, ids, count, sel);
}

static UpdateSegment::merge_update_function_t GetMergeUpdateFunction(PhysicalType type) {
	switch (type) {
	case PhysicalType::BIT:
		return MergeValidityLoop;
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		return MergeUpdateLoop<int8_t>;
	case PhysicalType::INT16:
		return MergeUpdateLoop<int16_t>;
	case PhysicalType::INT32:
		return MergeUpdateLoop<int32_t>;
	case PhysicalType::INT64:
		return MergeUpdateLoop<int64_t>;
	case PhysicalType::UINT8:
		return MergeUpdateLoop<uint8_t>;
	case PhysicalType::UINT16:
		return MergeUpdateLoop<uint16_t>;
	case PhysicalType::UINT32:
		return MergeUpdateLoop<uint32_t>;
	case PhysicalType::UINT64:
		return MergeUpdateLoop<uint64_t>;
	case PhysicalType::INT128:
		return MergeUpdateLoop<hugeint_t>;
	case PhysicalType::FLOAT:
		return MergeUpdateLoop<float>;
	case PhysicalType::DOUBLE:
		return MergeUpdateLoop<double>;
	case PhysicalType::INTERVAL:
		return MergeUpdateLoop<interval_t>;
	case PhysicalType::VARCHAR:
		return MergeUpdateLoop<string_t>;
	default:
		throw NotImplementedException("Unimplemented type for uncompressed segment");
	}
}

//===--------------------------------------------------------------------===//
// Update statistics
//===--------------------------------------------------------------------===//
unique_ptr<BaseStatistics> UpdateSegment::GetStatistics() {
	lock_guard<mutex> stats_guard(stats_lock);
	return stats.statistics->Copy();
}

idx_t UpdateValidityStatistics(UpdateSegment *segment, SegmentStatistics &stats, Vector &update, idx_t count,
                               SelectionVector &sel) {
	auto &mask = FlatVector::Validity(update);
	auto &validity = (ValidityStatistics &)*stats.statistics;
	if (!mask.AllValid() && !validity.has_null) {
		for (idx_t i = 0; i < count; i++) {
			if (!mask.RowIsValid(i)) {
				validity.has_null = true;
				break;
			}
		}
	}
	sel.Initialize(nullptr);
	return count;
}

template <class T>
idx_t TemplatedUpdateNumericStatistics(UpdateSegment *segment, SegmentStatistics &stats, Vector &update, idx_t count,
                                       SelectionVector &sel) {
	auto update_data = FlatVector::GetData<T>(update);
	auto &mask = FlatVector::Validity(update);

	if (mask.AllValid()) {
		for (idx_t i = 0; i < count; i++) {
			NumericStatistics::Update<T>(stats, update_data[i]);
		}
		sel.Initialize(nullptr);
		return count;
	} else {
		idx_t not_null_count = 0;
		sel.Initialize(STANDARD_VECTOR_SIZE);
		for (idx_t i = 0; i < count; i++) {
			if (mask.RowIsValid(i)) {
				sel.set_index(not_null_count++, i);
				NumericStatistics::Update<T>(stats, update_data[i]);
			}
		}
		return not_null_count;
	}
}

idx_t UpdateStringStatistics(UpdateSegment *segment, SegmentStatistics &stats, Vector &update, idx_t count,
                             SelectionVector &sel) {
	auto update_data = FlatVector::GetData<string_t>(update);
	auto &mask = FlatVector::Validity(update);
	if (mask.AllValid()) {
		for (idx_t i = 0; i < count; i++) {
			((StringStatistics &)*stats.statistics).Update(update_data[i]);
			if (!update_data[i].IsInlined()) {
				update_data[i] = segment->GetStringHeap().AddBlob(update_data[i]);
			}
		}
		sel.Initialize(nullptr);
		return count;
	} else {
		idx_t not_null_count = 0;
		sel.Initialize(STANDARD_VECTOR_SIZE);
		for (idx_t i = 0; i < count; i++) {
			if (mask.RowIsValid(i)) {
				sel.set_index(not_null_count++, i);
				((StringStatistics &)*stats.statistics).Update(update_data[i]);
				if (!update_data[i].IsInlined()) {
					update_data[i] = segment->GetStringHeap().AddBlob(update_data[i]);
				}
			}
		}
		return not_null_count;
	}
}

UpdateSegment::statistics_update_function_t GetStatisticsUpdateFunction(PhysicalType type) {
	switch (type) {
	case PhysicalType::BIT:
		return UpdateValidityStatistics;
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		return TemplatedUpdateNumericStatistics<int8_t>;
	case PhysicalType::INT16:
		return TemplatedUpdateNumericStatistics<int16_t>;
	case PhysicalType::INT32:
		return TemplatedUpdateNumericStatistics<int32_t>;
	case PhysicalType::INT64:
		return TemplatedUpdateNumericStatistics<int64_t>;
	case PhysicalType::UINT8:
		return TemplatedUpdateNumericStatistics<uint8_t>;
	case PhysicalType::UINT16:
		return TemplatedUpdateNumericStatistics<uint16_t>;
	case PhysicalType::UINT32:
		return TemplatedUpdateNumericStatistics<uint32_t>;
	case PhysicalType::UINT64:
		return TemplatedUpdateNumericStatistics<uint64_t>;
	case PhysicalType::INT128:
		return TemplatedUpdateNumericStatistics<hugeint_t>;
	case PhysicalType::FLOAT:
		return TemplatedUpdateNumericStatistics<float>;
	case PhysicalType::DOUBLE:
		return TemplatedUpdateNumericStatistics<double>;
	case PhysicalType::INTERVAL:
		return TemplatedUpdateNumericStatistics<interval_t>;
	case PhysicalType::VARCHAR:
		return UpdateStringStatistics;
	default:
		throw NotImplementedException("Unimplemented type for uncompressed segment");
	}
}

//===--------------------------------------------------------------------===//
// Update
//===--------------------------------------------------------------------===//
static idx_t SortSelectionVector(SelectionVector &sel, idx_t count, row_t *ids) {
	D_ASSERT(count > 0);

	bool is_sorted = true;
	for (idx_t i = 1; i < count; i++) {
		auto prev_idx = sel.get_index(i - 1);
		auto idx = sel.get_index(i);
		if (ids[idx] <= ids[prev_idx]) {
			is_sorted = false;
			break;
		}
	}
	if (is_sorted) {
		// already sorted: bailout
		return count;
	}
	// not sorted: need to sort the selection vector
	SelectionVector sorted_sel(count);
	for (idx_t i = 0; i < count; i++) {
		sorted_sel.set_index(i, sel.get_index(i));
	}
	std::sort(sorted_sel.data(), sorted_sel.data() + count, [&](sel_t l, sel_t r) { return ids[l] < ids[r]; });
	// eliminate any duplicates
	idx_t pos = 1;
	for (idx_t i = 1; i < count; i++) {
		auto prev_idx = sorted_sel.get_index(i - 1);
		auto idx = sorted_sel.get_index(i);
		D_ASSERT(ids[idx] >= ids[prev_idx]);
		if (ids[prev_idx] != ids[idx]) {
			sorted_sel.set_index(pos++, idx);
		}
	}
#ifdef DEBUG
	for (idx_t i = 1; i < pos; i++) {
		auto prev_idx = sorted_sel.get_index(i - 1);
		auto idx = sorted_sel.get_index(i);
		D_ASSERT(ids[idx] > ids[prev_idx]);
	}
#endif

	sel.Initialize(sorted_sel);
	D_ASSERT(pos > 0);
	return pos;
}

UpdateInfo *CreateEmptyUpdateInfo(TransactionData transaction, idx_t type_size, idx_t count, unique_ptr<char[]> &data) {
	data = unique_ptr<char[]>(new char[sizeof(UpdateInfo) + (sizeof(sel_t) + type_size) * STANDARD_VECTOR_SIZE]);
	auto update_info = (UpdateInfo *)data.get();
	update_info->max = STANDARD_VECTOR_SIZE;
	update_info->tuples = (sel_t *)(((data_ptr_t)update_info) + sizeof(UpdateInfo));
	update_info->tuple_data = ((data_ptr_t)update_info) + sizeof(UpdateInfo) + sizeof(sel_t) * update_info->max;
	update_info->version_number = transaction.transaction_id;
	return update_info;
}

void UpdateSegment::Update(TransactionData transaction, idx_t column_index, Vector &update, row_t *ids, idx_t count,
                           Vector &base_data) {
	// obtain an exclusive lock
	auto write_lock = lock.GetExclusiveLock();

	update.Flatten(count);

	// update statistics
	SelectionVector sel;
	{
		lock_guard<mutex> stats_guard(stats_lock);
		count = statistics_update_function(this, stats, update, count, sel);
	}
	if (count == 0) {
		return;
	}

	// subsequent algorithms used by the update require row ids to be (1) sorted, and (2) unique
	// this is usually the case for "standard" queries (e.g. UPDATE tbl SET x=bla WHERE cond)
	// however, for more exotic queries involving e.g. cross products/joins this might not be the case
	// hence we explicitly check here if the ids are sorted and, if not, sort + duplicate eliminate them
	count = SortSelectionVector(sel, count, ids);
	D_ASSERT(count > 0);

	// create the versions for this segment, if there are none yet
	if (!root) {
		root = make_unique<UpdateNode>();
	}

	// get the vector index based on the first id
	// we assert that all updates must be part of the same vector
	auto first_id = ids[sel.get_index(0)];
	idx_t vector_index = (first_id - column_data.start) / STANDARD_VECTOR_SIZE;
	idx_t vector_offset = column_data.start + vector_index * STANDARD_VECTOR_SIZE;

	D_ASSERT(idx_t(first_id) >= column_data.start);
	D_ASSERT(vector_index < RowGroup::ROW_GROUP_VECTOR_COUNT);

	// first check the version chain
	UpdateInfo *node = nullptr;

	if (root->info[vector_index]) {
		// there is already a version here, check if there are any conflicts and search for the node that belongs to
		// this transaction in the version chain
		auto base_info = root->info[vector_index]->info.get();
		CheckForConflicts(base_info->next, transaction, ids, sel, count, vector_offset, node);

		// there are no conflicts
		// first, check if this thread has already done any updates
		auto node = base_info->next;
		while (node) {
			if (node->version_number == transaction.transaction_id) {
				// it has! use this node
				break;
			}
			node = node->next;
		}
		unique_ptr<char[]> update_info_data;
		if (!node) {
			// no updates made yet by this transaction: initially the update info to empty
			if (transaction.transaction) {
				auto &dtransaction = (DuckTransaction &)*transaction.transaction;
				node = dtransaction.CreateUpdateInfo(type_size, count);
			} else {
				node = CreateEmptyUpdateInfo(transaction, type_size, count, update_info_data);
			}
			node->segment = this;
			node->vector_index = vector_index;
			node->N = 0;
			node->column_index = column_index;

			// insert the new node into the chain
			node->next = base_info->next;
			if (node->next) {
				node->next->prev = node;
			}
			node->prev = base_info;
			base_info->next = transaction.transaction ? node : nullptr;
		}
		base_info->Verify();
		node->Verify();

		// now we are going to perform the merge
		merge_update_function(base_info, base_data, node, update, ids, count, sel);

		base_info->Verify();
		node->Verify();
	} else {
		// there is no version info yet: create the top level update info and fill it with the updates
		auto result = make_unique<UpdateNodeData>();

		result->info = make_unique<UpdateInfo>();
		result->tuples = unique_ptr<sel_t[]>(new sel_t[STANDARD_VECTOR_SIZE]);
		result->tuple_data = unique_ptr<data_t[]>(new data_t[STANDARD_VECTOR_SIZE * type_size]);
		result->info->tuples = result->tuples.get();
		result->info->tuple_data = result->tuple_data.get();
		result->info->version_number = TRANSACTION_ID_START - 1;
		result->info->column_index = column_index;
		InitializeUpdateInfo(*result->info, ids, sel, count, vector_index, vector_offset);

		// now create the transaction level update info in the undo log
		unique_ptr<char[]> update_info_data;
		UpdateInfo *transaction_node;
		if (transaction.transaction) {
			transaction_node = transaction.transaction->CreateUpdateInfo(type_size, count);
		} else {
			transaction_node = CreateEmptyUpdateInfo(transaction, type_size, count, update_info_data);
		}

		InitializeUpdateInfo(*transaction_node, ids, sel, count, vector_index, vector_offset);

		// we write the updates in the update node data, and write the updates in the info
		initialize_update_function(transaction_node, base_data, result->info.get(), update, sel);

		result->info->next = transaction.transaction ? transaction_node : nullptr;
		result->info->prev = nullptr;
		transaction_node->next = nullptr;
		transaction_node->prev = result->info.get();
		transaction_node->column_index = column_index;

		transaction_node->Verify();
		result->info->Verify();

		root->info[vector_index] = std::move(result);
	}
}

bool UpdateSegment::HasUpdates() const {
	return root.get() != nullptr;
}

bool UpdateSegment::HasUpdates(idx_t vector_index) const {
	if (!HasUpdates()) {
		return false;
	}
	return root->info[vector_index].get();
}

bool UpdateSegment::HasUncommittedUpdates(idx_t vector_index) {
	if (!HasUpdates(vector_index)) {
		return false;
	}
	auto read_lock = lock.GetSharedLock();
	auto entry = root->info[vector_index].get();
	if (entry->info->next) {
		return true;
	}
	return false;
}

bool UpdateSegment::HasUpdates(idx_t start_row_index, idx_t end_row_index) {
	if (!HasUpdates()) {
		return false;
	}
	auto read_lock = lock.GetSharedLock();
	idx_t base_vector_index = start_row_index / STANDARD_VECTOR_SIZE;
	idx_t end_vector_index = end_row_index / STANDARD_VECTOR_SIZE;
	for (idx_t i = base_vector_index; i <= end_vector_index; i++) {
		if (root->info[i]) {
			return true;
		}
	}
	return false;
}

} // namespace duckdb




namespace duckdb {

ValidityColumnData::ValidityColumnData(BlockManager &block_manager, DataTableInfo &info, idx_t column_index,
                                       idx_t start_row, ColumnData *parent)
    : ColumnData(block_manager, info, column_index, start_row, LogicalType(LogicalTypeId::VALIDITY), parent) {
}

ValidityColumnData::ValidityColumnData(ColumnData &original, idx_t start_row, ColumnData *parent)
    : ColumnData(original, start_row, parent) {
}

bool ValidityColumnData::CheckZonemap(ColumnScanState &state, TableFilter &filter) {
	return true;
}

} // namespace duckdb





namespace duckdb {
void TableIndexList::AddIndex(unique_ptr<Index> index) {
	D_ASSERT(index);
	lock_guard<mutex> lock(indexes_lock);
	indexes.push_back(std::move(index));
}
void TableIndexList::RemoveIndex(Index *index) {
	D_ASSERT(index);
	lock_guard<mutex> lock(indexes_lock);

	for (idx_t index_idx = 0; index_idx < indexes.size(); index_idx++) {
		auto &index_entry = indexes[index_idx];
		if (index_entry.get() == index) {
			indexes.erase(indexes.begin() + index_idx);
			break;
		}
	}
}

bool TableIndexList::Empty() {
	lock_guard<mutex> lock(indexes_lock);
	return indexes.empty();
}

idx_t TableIndexList::Count() {
	lock_guard<mutex> lock(indexes_lock);
	return indexes.size();
}

void TableIndexList::Move(TableIndexList &other) {
	D_ASSERT(indexes.empty());
	indexes = std::move(other.indexes);
}

Index *TableIndexList::FindForeignKeyIndex(const vector<PhysicalIndex> &fk_keys, ForeignKeyType fk_type) {
	Index *result = nullptr;
	Scan([&](Index &index) {
		if (DataTable::IsForeignKeyIndex(fk_keys, index, fk_type)) {
			result = &index;
		}
		return false;
	});
	return result;
}

void TableIndexList::VerifyForeignKey(const vector<PhysicalIndex> &fk_keys, DataChunk &chunk,
                                      ConflictManager &conflict_manager) {
	auto fk_type = conflict_manager.LookupType() == VerifyExistenceType::APPEND_FK
	                   ? ForeignKeyType::FK_TYPE_PRIMARY_KEY_TABLE
	                   : ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE;

	// check whether or not the chunk can be inserted or deleted into the referenced table' storage
	auto index = FindForeignKeyIndex(fk_keys, fk_type);
	if (!index) {
		throw InternalException("Internal Foreign Key error: could not find index to verify...");
	}
	conflict_manager.SetIndexCount(1);

	index->LookupValues(chunk, conflict_manager);
}

vector<column_t> TableIndexList::GetRequiredColumns() {
	lock_guard<mutex> lock(indexes_lock);
	set<column_t> unique_indexes;
	for (auto &index : indexes) {
		for (auto col_index : index->column_ids) {
			unique_indexes.insert(col_index);
		}
	}
	vector<column_t> result;
	result.reserve(unique_indexes.size());
	for (auto column_index : unique_indexes) {
		result.emplace_back(column_index);
	}
	return result;
}

vector<BlockPointer> TableIndexList::SerializeIndexes(duckdb::MetaBlockWriter &writer) {
	vector<BlockPointer> blocks_info;
	for (auto &index : indexes) {
		blocks_info.emplace_back(index->Serialize(writer));
	}
	return blocks_info;
}

} // namespace duckdb





















namespace duckdb {

bool WriteAheadLog::Replay(AttachedDatabase &database, string &path) {
	auto initial_reader = make_unique<BufferedFileReader>(FileSystem::Get(database), path.c_str());
	if (initial_reader->Finished()) {
		// WAL is empty
		return false;
	}
	Connection con(database.GetDatabase());
	con.BeginTransaction();

	// first deserialize the WAL to look for a checkpoint flag
	// if there is a checkpoint flag, we might have already flushed the contents of the WAL to disk
	ReplayState checkpoint_state(database, *con.context, *initial_reader);
	checkpoint_state.deserialize_only = true;
	try {
		while (true) {
			// read the current entry
			WALType entry_type = initial_reader->Read<WALType>();
			if (entry_type == WALType::WAL_FLUSH) {
				// check if the file is exhausted
				if (initial_reader->Finished()) {
					// we finished reading the file: break
					break;
				}
			} else {
				// replay the entry
				checkpoint_state.ReplayEntry(entry_type);
			}
		}
	} catch (std::exception &ex) { // LCOV_EXCL_START
		Printer::Print(StringUtil::Format("Exception in WAL playback during initial read: %s\n", ex.what()));
		return false;
	} catch (...) {
		Printer::Print("Unknown Exception in WAL playback during initial read");
		return false;
	} // LCOV_EXCL_STOP
	initial_reader.reset();
	if (checkpoint_state.checkpoint_id != INVALID_BLOCK) {
		// there is a checkpoint flag: check if we need to deserialize the WAL
		auto &manager = database.GetStorageManager();
		if (manager.IsCheckpointClean(checkpoint_state.checkpoint_id)) {
			// the contents of the WAL have already been checkpointed
			// we can safely truncate the WAL and ignore its contents
			return true;
		}
	}

	// we need to recover from the WAL: actually set up the replay state
	BufferedFileReader reader(FileSystem::Get(database), path.c_str());
	ReplayState state(database, *con.context, reader);

	// replay the WAL
	// note that everything is wrapped inside a try/catch block here
	// there can be errors in WAL replay because of a corrupt WAL file
	// in this case we should throw a warning but startup anyway
	try {
		while (true) {
			// read the current entry
			WALType entry_type = reader.Read<WALType>();
			if (entry_type == WALType::WAL_FLUSH) {
				// flush: commit the current transaction
				con.Commit();
				// check if the file is exhausted
				if (reader.Finished()) {
					// we finished reading the file: break
					break;
				}
				// otherwise we keep on reading
				con.BeginTransaction();
			} else {
				// replay the entry
				state.ReplayEntry(entry_type);
			}
		}
	} catch (std::exception &ex) { // LCOV_EXCL_START
		// FIXME: this should report a proper warning in the connection
		Printer::Print(StringUtil::Format("Exception in WAL playback: %s\n", ex.what()));
		// exception thrown in WAL replay: rollback
		con.Rollback();
	} catch (...) {
		Printer::Print("Unknown Exception in WAL playback: %s\n");
		// exception thrown in WAL replay: rollback
		con.Rollback();
	} // LCOV_EXCL_STOP
	return false;
}

//===--------------------------------------------------------------------===//
// Replay Entries
//===--------------------------------------------------------------------===//
void ReplayState::ReplayEntry(WALType entry_type) {
	switch (entry_type) {
	case WALType::CREATE_TABLE:
		ReplayCreateTable();
		break;
	case WALType::DROP_TABLE:
		ReplayDropTable();
		break;
	case WALType::ALTER_INFO:
		ReplayAlter();
		break;
	case WALType::CREATE_VIEW:
		ReplayCreateView();
		break;
	case WALType::DROP_VIEW:
		ReplayDropView();
		break;
	case WALType::CREATE_SCHEMA:
		ReplayCreateSchema();
		break;
	case WALType::DROP_SCHEMA:
		ReplayDropSchema();
		break;
	case WALType::CREATE_SEQUENCE:
		ReplayCreateSequence();
		break;
	case WALType::DROP_SEQUENCE:
		ReplayDropSequence();
		break;
	case WALType::SEQUENCE_VALUE:
		ReplaySequenceValue();
		break;
	case WALType::CREATE_MACRO:
		ReplayCreateMacro();
		break;
	case WALType::DROP_MACRO:
		ReplayDropMacro();
		break;
	case WALType::CREATE_TABLE_MACRO:
		ReplayCreateTableMacro();
		break;
	case WALType::DROP_TABLE_MACRO:
		ReplayDropTableMacro();
		break;
	case WALType::USE_TABLE:
		ReplayUseTable();
		break;
	case WALType::INSERT_TUPLE:
		ReplayInsert();
		break;
	case WALType::DELETE_TUPLE:
		ReplayDelete();
		break;
	case WALType::UPDATE_TUPLE:
		ReplayUpdate();
		break;
	case WALType::CHECKPOINT:
		ReplayCheckpoint();
		break;
	case WALType::CREATE_TYPE:
		ReplayCreateType();
		break;
	case WALType::DROP_TYPE:
		ReplayDropType();
		break;
	default:
		throw InternalException("Invalid WAL entry type!");
	}
}

//===--------------------------------------------------------------------===//
// Replay Table
//===--------------------------------------------------------------------===//
void ReplayState::ReplayCreateTable() {
	auto info = TableCatalogEntry::Deserialize(source, context);
	if (deserialize_only) {
		return;
	}

	// bind the constraints to the table again
	auto binder = Binder::CreateBinder(context);
	auto bound_info = binder->BindCreateTableInfo(std::move(info));

	catalog.CreateTable(context, bound_info.get());
}

void ReplayState::ReplayDropTable() {
	DropInfo info;

	info.type = CatalogType::TABLE_ENTRY;
	info.schema = source.Read<string>();
	info.name = source.Read<string>();
	if (deserialize_only) {
		return;
	}

	catalog.DropEntry(context, &info);
}

void ReplayState::ReplayAlter() {
	auto info = AlterInfo::Deserialize(source);
	if (deserialize_only) {
		return;
	}
	catalog.Alter(context, info.get());
}

//===--------------------------------------------------------------------===//
// Replay View
//===--------------------------------------------------------------------===//
void ReplayState::ReplayCreateView() {
	auto entry = ViewCatalogEntry::Deserialize(source, context);
	if (deserialize_only) {
		return;
	}

	catalog.CreateView(context, entry.get());
}

void ReplayState::ReplayDropView() {
	DropInfo info;
	info.type = CatalogType::VIEW_ENTRY;
	info.schema = source.Read<string>();
	info.name = source.Read<string>();
	if (deserialize_only) {
		return;
	}
	catalog.DropEntry(context, &info);
}

//===--------------------------------------------------------------------===//
// Replay Schema
//===--------------------------------------------------------------------===//
void ReplayState::ReplayCreateSchema() {
	CreateSchemaInfo info;
	info.schema = source.Read<string>();
	if (deserialize_only) {
		return;
	}

	catalog.CreateSchema(context, &info);
}

void ReplayState::ReplayDropSchema() {
	DropInfo info;

	info.type = CatalogType::SCHEMA_ENTRY;
	info.name = source.Read<string>();
	if (deserialize_only) {
		return;
	}

	catalog.DropEntry(context, &info);
}

//===--------------------------------------------------------------------===//
// Replay Custom Type
//===--------------------------------------------------------------------===//
void ReplayState::ReplayCreateType() {
	auto info = TypeCatalogEntry::Deserialize(source);
	if (deserialize_only) {
		return;
	}

	catalog.CreateType(context, info.get());
}

void ReplayState::ReplayDropType() {
	DropInfo info;

	info.type = CatalogType::TYPE_ENTRY;
	info.schema = source.Read<string>();
	info.name = source.Read<string>();
	if (deserialize_only) {
		return;
	}

	catalog.DropEntry(context, &info);
}

//===--------------------------------------------------------------------===//
// Replay Sequence
//===--------------------------------------------------------------------===//
void ReplayState::ReplayCreateSequence() {
	auto entry = SequenceCatalogEntry::Deserialize(source);
	if (deserialize_only) {
		return;
	}

	catalog.CreateSequence(context, entry.get());
}

void ReplayState::ReplayDropSequence() {
	DropInfo info;
	info.type = CatalogType::SEQUENCE_ENTRY;
	info.schema = source.Read<string>();
	info.name = source.Read<string>();
	if (deserialize_only) {
		return;
	}

	catalog.DropEntry(context, &info);
}

void ReplayState::ReplaySequenceValue() {
	auto schema = source.Read<string>();
	auto name = source.Read<string>();
	auto usage_count = source.Read<uint64_t>();
	auto counter = source.Read<int64_t>();
	if (deserialize_only) {
		return;
	}

	// fetch the sequence from the catalog
	auto seq = catalog.GetEntry<SequenceCatalogEntry>(context, schema, name);
	if (usage_count > seq->usage_count) {
		seq->usage_count = usage_count;
		seq->counter = counter;
	}
}

//===--------------------------------------------------------------------===//
// Replay Macro
//===--------------------------------------------------------------------===//
void ReplayState::ReplayCreateMacro() {
	auto entry = ScalarMacroCatalogEntry::Deserialize(source, context);
	if (deserialize_only) {
		return;
	}

	catalog.CreateFunction(context, entry.get());
}

void ReplayState::ReplayDropMacro() {
	DropInfo info;
	info.type = CatalogType::MACRO_ENTRY;
	info.schema = source.Read<string>();
	info.name = source.Read<string>();
	if (deserialize_only) {
		return;
	}

	catalog.DropEntry(context, &info);
}

//===--------------------------------------------------------------------===//
// Replay Table Macro
//===--------------------------------------------------------------------===//
void ReplayState::ReplayCreateTableMacro() {
	auto entry = TableMacroCatalogEntry::Deserialize(source, context);
	if (deserialize_only) {
		return;
	}

	catalog.CreateFunction(context, entry.get());
}

void ReplayState::ReplayDropTableMacro() {
	DropInfo info;
	info.type = CatalogType::TABLE_MACRO_ENTRY;
	info.schema = source.Read<string>();
	info.name = source.Read<string>();
	if (deserialize_only) {
		return;
	}

	catalog.DropEntry(context, &info);
}

//===--------------------------------------------------------------------===//
// Replay Data
//===--------------------------------------------------------------------===//
void ReplayState::ReplayUseTable() {
	auto schema_name = source.Read<string>();
	auto table_name = source.Read<string>();
	if (deserialize_only) {
		return;
	}
	current_table = catalog.GetEntry<TableCatalogEntry>(context, schema_name, table_name);
}

void ReplayState::ReplayInsert() {
	DataChunk chunk;
	chunk.Deserialize(source);
	if (deserialize_only) {
		return;
	}
	if (!current_table) {
		throw Exception("Corrupt WAL: insert without table");
	}

	// append to the current table
	current_table->GetStorage().LocalAppend(*current_table, context, chunk);
}

void ReplayState::ReplayDelete() {
	DataChunk chunk;
	chunk.Deserialize(source);
	if (deserialize_only) {
		return;
	}
	if (!current_table) {
		throw InternalException("Corrupt WAL: delete without table");
	}

	D_ASSERT(chunk.ColumnCount() == 1 && chunk.data[0].GetType() == LogicalType::ROW_TYPE);
	row_t row_ids[1];
	Vector row_identifiers(LogicalType::ROW_TYPE, (data_ptr_t)row_ids);

	auto source_ids = FlatVector::GetData<row_t>(chunk.data[0]);
	// delete the tuples from the current table
	for (idx_t i = 0; i < chunk.size(); i++) {
		row_ids[0] = source_ids[i];
		current_table->GetStorage().Delete(*current_table, context, row_identifiers, 1);
	}
}

void ReplayState::ReplayUpdate() {
	vector<column_t> column_path;
	auto column_index_count = source.Read<idx_t>();
	column_path.reserve(column_index_count);
	for (idx_t i = 0; i < column_index_count; i++) {
		column_path.push_back(source.Read<column_t>());
	}
	DataChunk chunk;
	chunk.Deserialize(source);
	if (deserialize_only) {
		return;
	}
	if (!current_table) {
		throw InternalException("Corrupt WAL: update without table");
	}

	if (column_path[0] >= current_table->GetColumns().PhysicalColumnCount()) {
		throw InternalException("Corrupt WAL: column index for update out of bounds");
	}

	// remove the row id vector from the chunk
	auto row_ids = std::move(chunk.data.back());
	chunk.data.pop_back();

	// now perform the update
	current_table->GetStorage().UpdateColumn(*current_table, context, row_ids, column_path, chunk);
}

void ReplayState::ReplayCheckpoint() {
	checkpoint_id = source.Read<block_id_t>();
}

} // namespace duckdb









#include <cstring>

namespace duckdb {

WriteAheadLog::WriteAheadLog(AttachedDatabase &database, const string &path) : skip_writing(false), database(database) {
	wal_path = path;
	writer = make_unique<BufferedFileWriter>(FileSystem::Get(database), path.c_str(),
	                                         FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_FILE_CREATE |
	                                             FileFlags::FILE_FLAGS_APPEND);
}

WriteAheadLog::~WriteAheadLog() {
}

int64_t WriteAheadLog::GetWALSize() {
	D_ASSERT(writer);
	return writer->GetFileSize();
}

idx_t WriteAheadLog::GetTotalWritten() {
	D_ASSERT(writer);
	return writer->GetTotalWritten();
}

void WriteAheadLog::Truncate(int64_t size) {
	writer->Truncate(size);
}

void WriteAheadLog::Delete() {
	if (!writer) {
		return;
	}
	writer.reset();

	auto &fs = FileSystem::Get(database);
	fs.RemoveFile(wal_path);
}

//===--------------------------------------------------------------------===//
// Write Entries
//===--------------------------------------------------------------------===//
void WriteAheadLog::WriteCheckpoint(block_id_t meta_block) {
	writer->Write<WALType>(WALType::CHECKPOINT);
	writer->Write<block_id_t>(meta_block);
}

//===--------------------------------------------------------------------===//
// CREATE TABLE
//===--------------------------------------------------------------------===//
void WriteAheadLog::WriteCreateTable(TableCatalogEntry *entry) {
	if (skip_writing) {
		return;
	}
	writer->Write<WALType>(WALType::CREATE_TABLE);
	entry->Serialize(*writer);
}

//===--------------------------------------------------------------------===//
// DROP TABLE
//===--------------------------------------------------------------------===//
void WriteAheadLog::WriteDropTable(TableCatalogEntry *entry) {
	if (skip_writing) {
		return;
	}
	writer->Write<WALType>(WALType::DROP_TABLE);
	writer->WriteString(entry->schema->name);
	writer->WriteString(entry->name);
}

//===--------------------------------------------------------------------===//
// CREATE SCHEMA
//===--------------------------------------------------------------------===//
void WriteAheadLog::WriteCreateSchema(SchemaCatalogEntry *entry) {
	if (skip_writing) {
		return;
	}
	writer->Write<WALType>(WALType::CREATE_SCHEMA);
	writer->WriteString(entry->name);
}

//===--------------------------------------------------------------------===//
// SEQUENCES
//===--------------------------------------------------------------------===//
void WriteAheadLog::WriteCreateSequence(SequenceCatalogEntry *entry) {
	if (skip_writing) {
		return;
	}
	writer->Write<WALType>(WALType::CREATE_SEQUENCE);
	entry->Serialize(*writer);
}

void WriteAheadLog::WriteDropSequence(SequenceCatalogEntry *entry) {
	if (skip_writing) {
		return;
	}
	writer->Write<WALType>(WALType::DROP_SEQUENCE);
	writer->WriteString(entry->schema->name);
	writer->WriteString(entry->name);
}

void WriteAheadLog::WriteSequenceValue(SequenceCatalogEntry *entry, SequenceValue val) {
	if (skip_writing) {
		return;
	}
	writer->Write<WALType>(WALType::SEQUENCE_VALUE);
	writer->WriteString(entry->schema->name);
	writer->WriteString(entry->name);
	writer->Write<uint64_t>(val.usage_count);
	writer->Write<int64_t>(val.counter);
}

//===--------------------------------------------------------------------===//
// MACRO'S
//===--------------------------------------------------------------------===//
void WriteAheadLog::WriteCreateMacro(ScalarMacroCatalogEntry *entry) {
	if (skip_writing) {
		return;
	}
	writer->Write<WALType>(WALType::CREATE_MACRO);
	entry->Serialize(*writer);
}

void WriteAheadLog::WriteDropMacro(ScalarMacroCatalogEntry *entry) {
	if (skip_writing) {
		return;
	}
	writer->Write<WALType>(WALType::DROP_MACRO);
	writer->WriteString(entry->schema->name);
	writer->WriteString(entry->name);
}

void WriteAheadLog::WriteCreateTableMacro(TableMacroCatalogEntry *entry) {
	if (skip_writing) {
		return;
	}
	writer->Write<WALType>(WALType::CREATE_TABLE_MACRO);
	entry->Serialize(*writer);
}

void WriteAheadLog::WriteDropTableMacro(TableMacroCatalogEntry *entry) {
	if (skip_writing) {
		return;
	}
	writer->Write<WALType>(WALType::DROP_TABLE_MACRO);
	writer->WriteString(entry->schema->name);
	writer->WriteString(entry->name);
}

//===--------------------------------------------------------------------===//
// Custom Types
//===--------------------------------------------------------------------===//
void WriteAheadLog::WriteCreateType(TypeCatalogEntry *entry) {
	if (skip_writing) {
		return;
	}
	writer->Write<WALType>(WALType::CREATE_TYPE);
	entry->Serialize(*writer);
}

void WriteAheadLog::WriteDropType(TypeCatalogEntry *entry) {
	if (skip_writing) {
		return;
	}
	writer->Write<WALType>(WALType::DROP_TYPE);
	writer->WriteString(entry->schema->name);
	writer->WriteString(entry->name);
}

//===--------------------------------------------------------------------===//
// VIEWS
//===--------------------------------------------------------------------===//
void WriteAheadLog::WriteCreateView(ViewCatalogEntry *entry) {
	if (skip_writing) {
		return;
	}
	writer->Write<WALType>(WALType::CREATE_VIEW);
	entry->Serialize(*writer);
}

void WriteAheadLog::WriteDropView(ViewCatalogEntry *entry) {
	if (skip_writing) {
		return;
	}
	writer->Write<WALType>(WALType::DROP_VIEW);
	writer->WriteString(entry->schema->name);
	writer->WriteString(entry->name);
}

//===--------------------------------------------------------------------===//
// DROP SCHEMA
//===--------------------------------------------------------------------===//
void WriteAheadLog::WriteDropSchema(SchemaCatalogEntry *entry) {
	if (skip_writing) {
		return;
	}
	writer->Write<WALType>(WALType::DROP_SCHEMA);
	writer->WriteString(entry->name);
}

//===--------------------------------------------------------------------===//
// DATA
//===--------------------------------------------------------------------===//
void WriteAheadLog::WriteSetTable(string &schema, string &table) {
	if (skip_writing) {
		return;
	}
	writer->Write<WALType>(WALType::USE_TABLE);
	writer->WriteString(schema);
	writer->WriteString(table);
}

void WriteAheadLog::WriteInsert(DataChunk &chunk) {
	if (skip_writing) {
		return;
	}
	D_ASSERT(chunk.size() > 0);
	chunk.Verify();

	writer->Write<WALType>(WALType::INSERT_TUPLE);
	chunk.Serialize(*writer);
}

void WriteAheadLog::WriteDelete(DataChunk &chunk) {
	if (skip_writing) {
		return;
	}
	D_ASSERT(chunk.size() > 0);
	D_ASSERT(chunk.ColumnCount() == 1 && chunk.data[0].GetType() == LogicalType::ROW_TYPE);
	chunk.Verify();

	writer->Write<WALType>(WALType::DELETE_TUPLE);
	chunk.Serialize(*writer);
}

void WriteAheadLog::WriteUpdate(DataChunk &chunk, const vector<column_t> &column_indexes) {
	if (skip_writing) {
		return;
	}
	D_ASSERT(chunk.size() > 0);
	D_ASSERT(chunk.ColumnCount() == 2);
	D_ASSERT(chunk.data[1].GetType().id() == LogicalType::ROW_TYPE);
	chunk.Verify();

	writer->Write<WALType>(WALType::UPDATE_TUPLE);
	writer->Write<idx_t>(column_indexes.size());
	for (auto &col_idx : column_indexes) {
		writer->Write<column_t>(col_idx);
	}
	chunk.Serialize(*writer);
}

//===--------------------------------------------------------------------===//
// Write ALTER Statement
//===--------------------------------------------------------------------===//
void WriteAheadLog::WriteAlter(AlterInfo &info) {
	if (skip_writing) {
		return;
	}
	writer->Write<WALType>(WALType::ALTER_INFO);
	info.Serialize(*writer);
}

//===--------------------------------------------------------------------===//
// FLUSH
//===--------------------------------------------------------------------===//
void WriteAheadLog::Flush() {
	if (skip_writing) {
		return;
	}
	// write an empty entry
	writer->Write<WALType>(WALType::WAL_FLUSH);
	// flushes all changes made to the WAL to disk
	writer->Sync();
}

} // namespace duckdb











namespace duckdb {

CleanupState::CleanupState() : current_table(nullptr), count(0) {
}

CleanupState::~CleanupState() {
	Flush();
}

void CleanupState::CleanupEntry(UndoFlags type, data_ptr_t data) {
	switch (type) {
	case UndoFlags::CATALOG_ENTRY: {
		auto catalog_entry = Load<CatalogEntry *>(data);
		D_ASSERT(catalog_entry);
		D_ASSERT(catalog_entry->set);
		catalog_entry->set->CleanupEntry(catalog_entry);
		break;
	}
	case UndoFlags::DELETE_TUPLE: {
		auto info = (DeleteInfo *)data;
		CleanupDelete(info);
		break;
	}
	case UndoFlags::UPDATE_TUPLE: {
		auto info = (UpdateInfo *)data;
		CleanupUpdate(info);
		break;
	}
	default:
		break;
	}
}

void CleanupState::CleanupUpdate(UpdateInfo *info) {
	// remove the update info from the update chain
	// first obtain an exclusive lock on the segment
	info->segment->CleanupUpdate(info);
}

void CleanupState::CleanupDelete(DeleteInfo *info) {
	auto version_table = info->table;
	D_ASSERT(version_table->info->cardinality >= info->count);
	version_table->info->cardinality -= info->count;
	if (version_table->info->indexes.Empty()) {
		// this table has no indexes: no cleanup to be done
		return;
	}
	if (current_table != version_table) {
		// table for this entry differs from previous table: flush and switch to the new table
		Flush();
		current_table = version_table;
	}
	count = 0;
	for (idx_t i = 0; i < info->count; i++) {
		row_numbers[count++] = info->vinfo->start + info->rows[i];
	}
	Flush();
}

void CleanupState::Flush() {
	if (count == 0) {
		return;
	}

	// set up the row identifiers vector
	Vector row_identifiers(LogicalType::ROW_TYPE, (data_ptr_t)row_numbers);

	// delete the tuples from all the indexes
	try {
		current_table->RemoveFromIndexes(row_identifiers, count);
	} catch (...) {
	}

	count = 0;
}

} // namespace duckdb



















namespace duckdb {

CommitState::CommitState(ClientContext &context, transaction_t commit_id, WriteAheadLog *log)
    : log(log), commit_id(commit_id), current_table_info(nullptr), context(context) {
}

void CommitState::SwitchTable(DataTableInfo *table_info, UndoFlags new_op) {
	if (current_table_info != table_info) {
		// write the current table to the log
		log->WriteSetTable(table_info->schema, table_info->table);
		current_table_info = table_info;
	}
}

void CommitState::WriteCatalogEntry(CatalogEntry *entry, data_ptr_t dataptr) {
	if (entry->temporary || entry->parent->temporary) {
		return;
	}
	D_ASSERT(log);
	// look at the type of the parent entry
	auto parent = entry->parent;
	switch (parent->type) {
	case CatalogType::TABLE_ENTRY:
		if (entry->type == CatalogType::TABLE_ENTRY) {
			auto table_entry = (DuckTableEntry *)entry;
			D_ASSERT(table_entry->IsDuckTable());
			// ALTER TABLE statement, read the extra data after the entry
			auto extra_data_size = Load<idx_t>(dataptr);
			auto extra_data = (data_ptr_t)(dataptr + sizeof(idx_t));
			// deserialize it
			BufferedDeserializer source(extra_data, extra_data_size);
			auto info = AlterInfo::Deserialize(source);
			// write the alter table in the log
			table_entry->CommitAlter(*info);
			log->WriteAlter(*info);
		} else {
			// CREATE TABLE statement
			log->WriteCreateTable((TableCatalogEntry *)parent);
		}
		break;
	case CatalogType::SCHEMA_ENTRY:
		if (entry->type == CatalogType::SCHEMA_ENTRY) {
			// ALTER TABLE statement, skip it
			return;
		}
		log->WriteCreateSchema((SchemaCatalogEntry *)parent);
		break;
	case CatalogType::VIEW_ENTRY:
		if (entry->type == CatalogType::VIEW_ENTRY) {
			// ALTER TABLE statement, read the extra data after the entry
			auto extra_data_size = Load<idx_t>(dataptr);
			auto extra_data = (data_ptr_t)(dataptr + sizeof(idx_t));
			// deserialize it
			BufferedDeserializer source(extra_data, extra_data_size);
			auto info = AlterInfo::Deserialize(source);
			// write the alter table in the log
			log->WriteAlter(*info);
		} else {
			log->WriteCreateView((ViewCatalogEntry *)parent);
		}
		break;
	case CatalogType::SEQUENCE_ENTRY:
		log->WriteCreateSequence((SequenceCatalogEntry *)parent);
		break;
	case CatalogType::MACRO_ENTRY:
		log->WriteCreateMacro((ScalarMacroCatalogEntry *)parent);
		break;
	case CatalogType::TABLE_MACRO_ENTRY:
		log->WriteCreateTableMacro((TableMacroCatalogEntry *)parent);
		break;

	case CatalogType::TYPE_ENTRY:
		log->WriteCreateType((TypeCatalogEntry *)parent);
		break;
	case CatalogType::DELETED_ENTRY:
		switch (entry->type) {
		case CatalogType::TABLE_ENTRY: {
			auto table_entry = (DuckTableEntry *)entry;
			D_ASSERT(table_entry->IsDuckTable());
			table_entry->CommitDrop();
			log->WriteDropTable(table_entry);
			break;
		}
		case CatalogType::SCHEMA_ENTRY:
			log->WriteDropSchema((SchemaCatalogEntry *)entry);
			break;
		case CatalogType::VIEW_ENTRY:
			log->WriteDropView((ViewCatalogEntry *)entry);
			break;
		case CatalogType::SEQUENCE_ENTRY:
			log->WriteDropSequence((SequenceCatalogEntry *)entry);
			break;
		case CatalogType::MACRO_ENTRY:
			log->WriteDropMacro((ScalarMacroCatalogEntry *)entry);
			break;
		case CatalogType::TABLE_MACRO_ENTRY:
			log->WriteDropTableMacro((TableMacroCatalogEntry *)entry);
			break;
		case CatalogType::TYPE_ENTRY:
			log->WriteDropType((TypeCatalogEntry *)entry);
			break;
		case CatalogType::INDEX_ENTRY:
		case CatalogType::PREPARED_STATEMENT:
		case CatalogType::SCALAR_FUNCTION_ENTRY:
			// do nothing, indexes/prepared statements/functions aren't persisted to disk
			break;
		default:
			throw InternalException("Don't know how to drop this type!");
		}
		break;
	case CatalogType::INDEX_ENTRY:
	case CatalogType::PREPARED_STATEMENT:
	case CatalogType::AGGREGATE_FUNCTION_ENTRY:
	case CatalogType::SCALAR_FUNCTION_ENTRY:
	case CatalogType::TABLE_FUNCTION_ENTRY:
	case CatalogType::COPY_FUNCTION_ENTRY:
	case CatalogType::PRAGMA_FUNCTION_ENTRY:
	case CatalogType::COLLATION_ENTRY:
		// do nothing, these entries are not persisted to disk
		break;
	default:
		throw InternalException("UndoBuffer - don't know how to write this entry to the WAL");
	}
}

void CommitState::WriteDelete(DeleteInfo *info) {
	D_ASSERT(log);
	// switch to the current table, if necessary
	SwitchTable(info->table->info.get(), UndoFlags::DELETE_TUPLE);

	if (!delete_chunk) {
		delete_chunk = make_unique<DataChunk>();
		vector<LogicalType> delete_types = {LogicalType::ROW_TYPE};
		delete_chunk->Initialize(Allocator::DefaultAllocator(), delete_types);
	}
	auto rows = FlatVector::GetData<row_t>(delete_chunk->data[0]);
	for (idx_t i = 0; i < info->count; i++) {
		rows[i] = info->base_row + info->rows[i];
	}
	delete_chunk->SetCardinality(info->count);
	log->WriteDelete(*delete_chunk);
}

void CommitState::WriteUpdate(UpdateInfo *info) {
	D_ASSERT(log);
	// switch to the current table, if necessary
	auto &column_data = info->segment->column_data;
	auto &table_info = column_data.GetTableInfo();

	SwitchTable(&table_info, UndoFlags::UPDATE_TUPLE);

	// initialize the update chunk
	vector<LogicalType> update_types;
	if (column_data.type.id() == LogicalTypeId::VALIDITY) {
		update_types.emplace_back(LogicalType::BOOLEAN);
	} else {
		update_types.push_back(column_data.type);
	}
	update_types.emplace_back(LogicalType::ROW_TYPE);

	update_chunk = make_unique<DataChunk>();
	update_chunk->Initialize(Allocator::DefaultAllocator(), update_types);

	// fetch the updated values from the base segment
	info->segment->FetchCommitted(info->vector_index, update_chunk->data[0]);

	// write the row ids into the chunk
	auto row_ids = FlatVector::GetData<row_t>(update_chunk->data[1]);
	idx_t start = column_data.start + info->vector_index * STANDARD_VECTOR_SIZE;
	for (idx_t i = 0; i < info->N; i++) {
		row_ids[info->tuples[i]] = start + info->tuples[i];
	}
	if (column_data.type.id() == LogicalTypeId::VALIDITY) {
		// zero-initialize the booleans
		// FIXME: this is only required because of NullValue<T> in Vector::Serialize...
		auto booleans = FlatVector::GetData<bool>(update_chunk->data[0]);
		for (idx_t i = 0; i < info->N; i++) {
			auto idx = info->tuples[i];
			booleans[idx] = false;
		}
	}
	SelectionVector sel(info->tuples);
	update_chunk->Slice(sel, info->N);

	// construct the column index path
	vector<column_t> column_indexes;
	auto column_data_ptr = &column_data;
	while (column_data_ptr->parent) {
		column_indexes.push_back(column_data_ptr->column_index);
		column_data_ptr = column_data_ptr->parent;
	}
	column_indexes.push_back(info->column_index);
	std::reverse(column_indexes.begin(), column_indexes.end());

	log->WriteUpdate(*update_chunk, column_indexes);
}

template <bool HAS_LOG>
void CommitState::CommitEntry(UndoFlags type, data_ptr_t data) {
	switch (type) {
	case UndoFlags::CATALOG_ENTRY: {
		// set the commit timestamp of the catalog entry to the given id
		auto catalog_entry = Load<CatalogEntry *>(data);
		D_ASSERT(catalog_entry->parent);

		auto &catalog = catalog_entry->catalog;
		D_ASSERT(catalog);
		D_ASSERT(catalog->IsDuckCatalog());

		// Grab a write lock on the catalog
		auto &duck_catalog = (DuckCatalog &)*catalog;
		lock_guard<mutex> write_lock(duck_catalog.GetWriteLock());
		catalog_entry->set->UpdateTimestamp(catalog_entry->parent, commit_id);
		if (catalog_entry->name != catalog_entry->parent->name) {
			catalog_entry->set->UpdateTimestamp(catalog_entry, commit_id);
		}
		if (HAS_LOG) {
			// push the catalog update to the WAL
			WriteCatalogEntry(catalog_entry, data + sizeof(CatalogEntry *));
		}
		break;
	}
	case UndoFlags::INSERT_TUPLE: {
		// append:
		auto info = (AppendInfo *)data;
		if (HAS_LOG && !info->table->info->IsTemporary()) {
			info->table->WriteToLog(*log, info->start_row, info->count);
		}
		// mark the tuples as committed
		info->table->CommitAppend(commit_id, info->start_row, info->count);
		break;
	}
	case UndoFlags::DELETE_TUPLE: {
		// deletion:
		auto info = (DeleteInfo *)data;
		if (HAS_LOG && !info->table->info->IsTemporary()) {
			WriteDelete(info);
		}
		// mark the tuples as committed
		info->vinfo->CommitDelete(commit_id, info->rows, info->count);
		break;
	}
	case UndoFlags::UPDATE_TUPLE: {
		// update:
		auto info = (UpdateInfo *)data;
		if (HAS_LOG && !info->segment->column_data.GetTableInfo().IsTemporary()) {
			WriteUpdate(info);
		}
		info->version_number = commit_id;
		break;
	}
	default:
		throw InternalException("UndoBuffer - don't know how to commit this type!");
	}
}

void CommitState::RevertCommit(UndoFlags type, data_ptr_t data) {
	transaction_t transaction_id = commit_id;
	switch (type) {
	case UndoFlags::CATALOG_ENTRY: {
		// set the commit timestamp of the catalog entry to the given id
		auto catalog_entry = Load<CatalogEntry *>(data);
		D_ASSERT(catalog_entry->parent);
		catalog_entry->set->UpdateTimestamp(catalog_entry->parent, transaction_id);
		if (catalog_entry->name != catalog_entry->parent->name) {
			catalog_entry->set->UpdateTimestamp(catalog_entry, transaction_id);
		}
		break;
	}
	case UndoFlags::INSERT_TUPLE: {
		auto info = (AppendInfo *)data;
		// revert this append
		info->table->RevertAppend(info->start_row, info->count);
		break;
	}
	case UndoFlags::DELETE_TUPLE: {
		// deletion:
		auto info = (DeleteInfo *)data;
		info->table->info->cardinality += info->count;
		// revert the commit by writing the (uncommitted) transaction_id back into the version info
		info->vinfo->CommitDelete(transaction_id, info->rows, info->count);
		break;
	}
	case UndoFlags::UPDATE_TUPLE: {
		// update:
		auto info = (UpdateInfo *)data;
		info->version_number = transaction_id;
		break;
	}
	default:
		throw InternalException("UndoBuffer - don't know how to revert commit of this type!");
	}
}

template void CommitState::CommitEntry<true>(UndoFlags type, data_ptr_t data);
template void CommitState::CommitEntry<false>(UndoFlags type, data_ptr_t data);

} // namespace duckdb



















namespace duckdb {

TransactionData::TransactionData(DuckTransaction &transaction_p) // NOLINT
    : transaction(&transaction_p), transaction_id(transaction_p.transaction_id), start_time(transaction_p.start_time) {
}
TransactionData::TransactionData(transaction_t transaction_id_p, transaction_t start_time_p)
    : transaction(nullptr), transaction_id(transaction_id_p), start_time(start_time_p) {
}

DuckTransaction::DuckTransaction(TransactionManager &manager, ClientContext &context_p, transaction_t start_time,
                                 transaction_t transaction_id)
    : Transaction(manager, context_p), start_time(start_time), transaction_id(transaction_id), commit_id(0),
      highest_active_query(0), undo_buffer(context_p), storage(make_unique<LocalStorage>(context_p, *this)) {
}

DuckTransaction::~DuckTransaction() {
}

DuckTransaction &DuckTransaction::Get(ClientContext &context, AttachedDatabase &db) {
	return DuckTransaction::Get(context, db.GetCatalog());
}

DuckTransaction &DuckTransaction::Get(ClientContext &context, Catalog &catalog) {
	auto &transaction = Transaction::Get(context, catalog);
	if (!transaction.IsDuckTransaction()) {
		throw InternalException("DuckTransaction::Get called on non-DuckDB transaction");
	}
	return (DuckTransaction &)transaction;
}

LocalStorage &DuckTransaction::GetLocalStorage() {
	return *storage;
}

void DuckTransaction::PushCatalogEntry(CatalogEntry *entry, data_ptr_t extra_data, idx_t extra_data_size) {
	idx_t alloc_size = sizeof(CatalogEntry *);
	if (extra_data_size > 0) {
		alloc_size += extra_data_size + sizeof(idx_t);
	}
	auto baseptr = undo_buffer.CreateEntry(UndoFlags::CATALOG_ENTRY, alloc_size);
	// store the pointer to the catalog entry
	Store<CatalogEntry *>(entry, baseptr);
	if (extra_data_size > 0) {
		// copy the extra data behind the catalog entry pointer (if any)
		baseptr += sizeof(CatalogEntry *);
		// first store the extra data size
		Store<idx_t>(extra_data_size, baseptr);
		baseptr += sizeof(idx_t);
		// then copy over the actual data
		memcpy(baseptr, extra_data, extra_data_size);
	}
}

void DuckTransaction::PushDelete(DataTable *table, ChunkVectorInfo *vinfo, row_t rows[], idx_t count, idx_t base_row) {
	auto delete_info =
	    (DeleteInfo *)undo_buffer.CreateEntry(UndoFlags::DELETE_TUPLE, sizeof(DeleteInfo) + sizeof(row_t) * count);
	delete_info->vinfo = vinfo;
	delete_info->table = table;
	delete_info->count = count;
	delete_info->base_row = base_row;
	memcpy(delete_info->rows, rows, sizeof(row_t) * count);
}

void DuckTransaction::PushAppend(DataTable *table, idx_t start_row, idx_t row_count) {
	auto append_info = (AppendInfo *)undo_buffer.CreateEntry(UndoFlags::INSERT_TUPLE, sizeof(AppendInfo));
	append_info->table = table;
	append_info->start_row = start_row;
	append_info->count = row_count;
}

UpdateInfo *DuckTransaction::CreateUpdateInfo(idx_t type_size, idx_t entries) {
	auto update_info = (UpdateInfo *)undo_buffer.CreateEntry(
	    UndoFlags::UPDATE_TUPLE, sizeof(UpdateInfo) + (sizeof(sel_t) + type_size) * STANDARD_VECTOR_SIZE);
	update_info->max = STANDARD_VECTOR_SIZE;
	update_info->tuples = (sel_t *)(((data_ptr_t)update_info) + sizeof(UpdateInfo));
	update_info->tuple_data = ((data_ptr_t)update_info) + sizeof(UpdateInfo) + sizeof(sel_t) * update_info->max;
	update_info->version_number = transaction_id;
	return update_info;
}

bool DuckTransaction::ChangesMade() {
	return undo_buffer.ChangesMade() || storage->ChangesMade();
}

bool DuckTransaction::AutomaticCheckpoint(AttachedDatabase &db) {
	auto &storage_manager = db.GetStorageManager();
	return storage_manager.AutomaticCheckpoint(storage->EstimatedSize() + undo_buffer.EstimatedSize());
}

string DuckTransaction::Commit(AttachedDatabase &db, transaction_t commit_id, bool checkpoint) noexcept {
	// "checkpoint" parameter indicates if the caller will checkpoint. If checkpoint ==
	//    true: Then this function will NOT write to the WAL or flush/persist.
	//          This method only makes commit in memory, expecting caller to checkpoint/flush.
	//    false: Then this function WILL write to the WAL and Flush/Persist it.
	this->commit_id = commit_id;

	UndoBuffer::IteratorState iterator_state;
	LocalStorage::CommitState commit_state;
	unique_ptr<StorageCommitState> storage_commit_state;
	WriteAheadLog *log;
	if (!db.IsSystem()) {
		auto &storage_manager = db.GetStorageManager();
		log = storage_manager.GetWriteAheadLog();
		storage_commit_state = storage_manager.GenStorageCommitState(*this, checkpoint);
	} else {
		log = nullptr;
	}
	try {
		storage->Commit(commit_state, *this);
		undo_buffer.Commit(iterator_state, log, commit_id);
		if (log) {
			// commit any sequences that were used to the WAL
			for (auto &entry : sequence_usage) {
				log->WriteSequenceValue(entry.first, entry.second);
			}
		}
		if (storage_commit_state) {
			storage_commit_state->FlushCommit();
		}
		return string();
	} catch (std::exception &ex) {
		undo_buffer.RevertCommit(iterator_state, transaction_id);
		return ex.what();
	}
}

void DuckTransaction::Rollback() noexcept {
	storage->Rollback();
	undo_buffer.Rollback();
}

void DuckTransaction::Cleanup() {
	undo_buffer.Cleanup();
}

} // namespace duckdb















namespace duckdb {

struct CheckpointLock {
	explicit CheckpointLock(DuckTransactionManager &manager) : manager(manager), is_locked(false) {
	}
	~CheckpointLock() {
		Unlock();
	}

	DuckTransactionManager &manager;
	bool is_locked;

	void Lock() {
		D_ASSERT(!manager.thread_is_checkpointing);
		manager.thread_is_checkpointing = true;
		is_locked = true;
	}
	void Unlock() {
		if (!is_locked) {
			return;
		}
		D_ASSERT(manager.thread_is_checkpointing);
		manager.thread_is_checkpointing = false;
		is_locked = false;
	}
};

DuckTransactionManager::DuckTransactionManager(AttachedDatabase &db)
    : TransactionManager(db), thread_is_checkpointing(false) {
	// start timestamp starts at two
	current_start_timestamp = 2;
	// transaction ID starts very high:
	// it should be much higher than the current start timestamp
	// if transaction_id < start_timestamp for any set of active transactions
	// uncommited data could be read by
	current_transaction_id = TRANSACTION_ID_START;
	lowest_active_id = TRANSACTION_ID_START;
	lowest_active_start = MAX_TRANSACTION_ID;
}

DuckTransactionManager::~DuckTransactionManager() {
}

DuckTransactionManager &DuckTransactionManager::Get(AttachedDatabase &db) {
	auto &transaction_manager = TransactionManager::Get(db);
	if (!transaction_manager.IsDuckTransactionManager()) {
		throw InternalException("Calling DuckTransactionManager::Get on non-DuckDB transaction manager");
	}
	return (DuckTransactionManager &)transaction_manager;
}

Transaction *DuckTransactionManager::StartTransaction(ClientContext &context) {
	// obtain the transaction lock during this function
	lock_guard<mutex> lock(transaction_lock);
	if (current_start_timestamp >= TRANSACTION_ID_START) { // LCOV_EXCL_START
		throw InternalException("Cannot start more transactions, ran out of "
		                        "transaction identifiers!");
	} // LCOV_EXCL_STOP

	// obtain the start time and transaction ID of this transaction
	transaction_t start_time = current_start_timestamp++;
	transaction_t transaction_id = current_transaction_id++;
	if (active_transactions.empty()) {
		lowest_active_start = start_time;
		lowest_active_id = transaction_id;
	}

	// create the actual transaction
	auto transaction = make_unique<DuckTransaction>(*this, context, start_time, transaction_id);
	auto transaction_ptr = transaction.get();

	// store it in the set of active transactions
	active_transactions.push_back(std::move(transaction));
	return transaction_ptr;
}

struct ClientLockWrapper {
	ClientLockWrapper(mutex &client_lock, shared_ptr<ClientContext> connection)
	    : connection(std::move(connection)), connection_lock(make_unique<lock_guard<mutex>>(client_lock)) {
	}

	shared_ptr<ClientContext> connection;
	unique_ptr<lock_guard<mutex>> connection_lock;
};

void DuckTransactionManager::LockClients(vector<ClientLockWrapper> &client_locks, ClientContext &context) {
	auto &connection_manager = ConnectionManager::Get(context);
	client_locks.emplace_back(connection_manager.connections_lock, nullptr);
	auto connection_list = connection_manager.GetConnectionList();
	for (auto &con : connection_list) {
		if (con.get() == &context) {
			continue;
		}
		auto &context_lock = con->context_lock;
		client_locks.emplace_back(context_lock, std::move(con));
	}
}

void DuckTransactionManager::Checkpoint(ClientContext &context, bool force) {
	auto &storage_manager = db.GetStorageManager();
	if (storage_manager.InMemory()) {
		return;
	}

	// first check if no other thread is checkpointing right now
	auto lock = unique_lock<mutex>(transaction_lock);
	if (thread_is_checkpointing) {
		throw TransactionException("Cannot CHECKPOINT: another thread is checkpointing right now");
	}
	CheckpointLock checkpoint_lock(*this);
	checkpoint_lock.Lock();
	lock.unlock();

	// lock all the clients AND the connection manager now
	// this ensures no new queries can be started, and no new connections to the database can be made
	// to avoid deadlock we release the transaction lock while locking the clients
	vector<ClientLockWrapper> client_locks;
	LockClients(client_locks, context);

	auto current = &DuckTransaction::Get(context, db);
	lock.lock();
	if (current->ChangesMade()) {
		throw TransactionException("Cannot CHECKPOINT: the current transaction has transaction local changes");
	}
	if (!force) {
		if (!CanCheckpoint(current)) {
			throw TransactionException("Cannot CHECKPOINT: there are other transactions. Use FORCE CHECKPOINT to abort "
			                           "the other transactions and force a checkpoint");
		}
	} else {
		if (!CanCheckpoint(current)) {
			for (size_t i = 0; i < active_transactions.size(); i++) {
				auto &transaction = active_transactions[i];
				// rollback the transaction
				transaction->Rollback();
				auto transaction_context = transaction->context.lock();

				// remove the transaction id from the list of active transactions
				// potentially resulting in garbage collection
				RemoveTransaction(transaction.get());
				if (transaction_context) {
					transaction_context->transaction.ClearTransaction();
				}
				i--;
			}
			D_ASSERT(CanCheckpoint(nullptr));
		}
	}
	storage_manager.CreateCheckpoint();
}

bool DuckTransactionManager::CanCheckpoint(DuckTransaction *current) {
	if (db.IsSystem()) {
		return false;
	}
	auto &storage_manager = db.GetStorageManager();
	if (storage_manager.InMemory()) {
		return false;
	}
	if (!recently_committed_transactions.empty() || !old_transactions.empty()) {
		return false;
	}
	for (auto &transaction : active_transactions) {
		if (transaction.get() != current) {
			return false;
		}
	}
	return true;
}

string DuckTransactionManager::CommitTransaction(ClientContext &context, Transaction *transaction_p) {
	auto transaction = (DuckTransaction *)transaction_p;
	vector<ClientLockWrapper> client_locks;
	auto lock = make_unique<lock_guard<mutex>>(transaction_lock);
	CheckpointLock checkpoint_lock(*this);
	// check if we can checkpoint
	bool checkpoint = thread_is_checkpointing ? false : CanCheckpoint(transaction);
	if (checkpoint) {
		if (transaction->AutomaticCheckpoint(db)) {
			checkpoint_lock.Lock();
			// we might be able to checkpoint: lock all clients
			// to avoid deadlock we release the transaction lock while locking the clients
			lock.reset();

			LockClients(client_locks, context);

			lock = make_unique<lock_guard<mutex>>(transaction_lock);
			checkpoint = CanCheckpoint(transaction);
			if (!checkpoint) {
				checkpoint_lock.Unlock();
				client_locks.clear();
			}
		} else {
			checkpoint = false;
		}
	}
	// obtain a commit id for the transaction
	transaction_t commit_id = current_start_timestamp++;
	// commit the UndoBuffer of the transaction
	string error = transaction->Commit(db, commit_id, checkpoint);
	if (!error.empty()) {
		// commit unsuccessful: rollback the transaction instead
		checkpoint = false;
		transaction->commit_id = 0;
		transaction->Rollback();
	}
	if (!checkpoint) {
		// we won't checkpoint after all: unlock the clients again
		checkpoint_lock.Unlock();
		client_locks.clear();
	}

	// commit successful: remove the transaction id from the list of active transactions
	// potentially resulting in garbage collection
	RemoveTransaction(transaction);
	// now perform a checkpoint if (1) we are able to checkpoint, and (2) the WAL has reached sufficient size to
	// checkpoint
	if (checkpoint) {
		// checkpoint the database to disk
		auto &storage_manager = db.GetStorageManager();
		storage_manager.CreateCheckpoint(false, true);
	}
	return error;
}

void DuckTransactionManager::RollbackTransaction(Transaction *transaction_p) {
	auto transaction = (DuckTransaction *)transaction_p;
	// obtain the transaction lock during this function
	lock_guard<mutex> lock(transaction_lock);

	// rollback the transaction
	transaction->Rollback();

	// remove the transaction id from the list of active transactions
	// potentially resulting in garbage collection
	RemoveTransaction(transaction);
}

void DuckTransactionManager::RemoveTransaction(DuckTransaction *transaction) noexcept {
	// remove the transaction from the list of active transactions
	idx_t t_index = active_transactions.size();
	// check for the lowest and highest start time in the list of transactions
	transaction_t lowest_start_time = TRANSACTION_ID_START;
	transaction_t lowest_transaction_id = MAX_TRANSACTION_ID;
	transaction_t lowest_active_query = MAXIMUM_QUERY_ID;
	for (idx_t i = 0; i < active_transactions.size(); i++) {
		if (active_transactions[i].get() == transaction) {
			t_index = i;
		} else {
			transaction_t active_query = active_transactions[i]->active_query;
			lowest_start_time = MinValue(lowest_start_time, active_transactions[i]->start_time);
			lowest_active_query = MinValue(lowest_active_query, active_query);
			lowest_transaction_id = MinValue(lowest_transaction_id, active_transactions[i]->transaction_id);
		}
	}
	lowest_active_start = lowest_start_time;
	lowest_active_id = lowest_transaction_id;

	transaction_t lowest_stored_query = lowest_start_time;
	D_ASSERT(t_index != active_transactions.size());
	auto current_transaction = std::move(active_transactions[t_index]);
	auto current_query = DatabaseManager::Get(db).ActiveQueryNumber();
	if (transaction->commit_id != 0) {
		// the transaction was committed, add it to the list of recently
		// committed transactions
		recently_committed_transactions.push_back(std::move(current_transaction));
	} else {
		// the transaction was aborted, but we might still need its information
		// add it to the set of transactions awaiting GC
		current_transaction->highest_active_query = current_query;
		old_transactions.push_back(std::move(current_transaction));
	}
	// remove the transaction from the set of currently active transactions
	active_transactions.erase(active_transactions.begin() + t_index);
	// traverse the recently_committed transactions to see if we can remove any
	idx_t i = 0;
	for (; i < recently_committed_transactions.size(); i++) {
		D_ASSERT(recently_committed_transactions[i]);
		lowest_stored_query = MinValue(recently_committed_transactions[i]->start_time, lowest_stored_query);
		if (recently_committed_transactions[i]->commit_id < lowest_start_time) {
			// changes made BEFORE this transaction are no longer relevant
			// we can cleanup the undo buffer

			// HOWEVER: any currently running QUERY can still be using
			// the version information after the cleanup!

			// if we remove the UndoBuffer immediately, we have a race
			// condition

			// we can only safely do the actual memory cleanup when all the
			// currently active queries have finished running! (actually,
			// when all the currently active scans have finished running...)
			recently_committed_transactions[i]->Cleanup();
			// store the current highest active query
			recently_committed_transactions[i]->highest_active_query = current_query;
			// move it to the list of transactions awaiting GC
			old_transactions.push_back(std::move(recently_committed_transactions[i]));
		} else {
			// recently_committed_transactions is ordered on commit_id
			// implicitly thus if the current one is bigger than
			// lowest_start_time any subsequent ones are also bigger
			break;
		}
	}
	if (i > 0) {
		// we garbage collected transactions: remove them from the list
		recently_committed_transactions.erase(recently_committed_transactions.begin(),
		                                      recently_committed_transactions.begin() + i);
	}
	// check if we can free the memory of any old transactions
	i = active_transactions.empty() ? old_transactions.size() : 0;
	for (; i < old_transactions.size(); i++) {
		D_ASSERT(old_transactions[i]);
		D_ASSERT(old_transactions[i]->highest_active_query > 0);
		if (old_transactions[i]->highest_active_query >= lowest_active_query) {
			// there is still a query running that could be using
			// this transactions' data
			break;
		}
	}
	if (i > 0) {
		// we garbage collected transactions: remove them from the list
		old_transactions.erase(old_transactions.begin(), old_transactions.begin() + i);
	}
}

} // namespace duckdb





namespace duckdb {

MetaTransaction::MetaTransaction(ClientContext &context_p, timestamp_t start_timestamp_p, idx_t catalog_version_p)
    : context(context_p), start_timestamp(start_timestamp_p), catalog_version(catalog_version_p), read_only(true),
      active_query(MAXIMUM_QUERY_ID), modified_database(nullptr) {
}

MetaTransaction &MetaTransaction::Get(ClientContext &context) {
	return context.transaction.ActiveTransaction();
}

ValidChecker &ValidChecker::Get(MetaTransaction &transaction) {
	return transaction.transaction_validity;
}

Transaction &Transaction::Get(ClientContext &context, AttachedDatabase &db) {
	auto &meta_transaction = MetaTransaction::Get(context);
	return meta_transaction.GetTransaction(&db);
}

Transaction &MetaTransaction::GetTransaction(AttachedDatabase *db) {
	auto entry = transactions.find(db);
	if (entry == transactions.end()) {
		auto new_transaction = db->GetTransactionManager().StartTransaction(context);
		if (!new_transaction) {
			throw InternalException("StartTransaction did not return a valid transaction");
		}
		new_transaction->active_query = active_query;
		all_transactions.push_back(db);
		transactions[db] = new_transaction;
		return *new_transaction;
	} else {
		D_ASSERT(entry->second->active_query == active_query);
		return *entry->second;
	}
}

Transaction &Transaction::Get(ClientContext &context, Catalog &catalog) {
	return Transaction::Get(context, catalog.GetAttached());
}

string MetaTransaction::Commit() {
	string error;
	// commit transactions in reverse order
	for (idx_t i = all_transactions.size(); i > 0; i--) {
		auto db = all_transactions[i - 1];
		auto entry = transactions.find(db);
		if (entry == transactions.end()) {
			throw InternalException("Could not find transaction corresponding to database in MetaTransaction");
		}
		auto &transaction_manager = db->GetTransactionManager();
		auto transaction = entry->second;
		if (error.empty()) {
			// commit
			error = transaction_manager.CommitTransaction(context, transaction);
		} else {
			// we have encountered an error previously - roll back subsequent entries
			transaction_manager.RollbackTransaction(transaction);
		}
	}
	return error;
}

void MetaTransaction::Rollback() {
	// rollback transactions in reverse order
	for (idx_t i = all_transactions.size(); i > 0; i--) {
		auto db = all_transactions[i - 1];
		auto &transaction_manager = db->GetTransactionManager();
		auto entry = transactions.find(db);
		D_ASSERT(entry != transactions.end());
		auto transaction = entry->second;
		transaction_manager.RollbackTransaction(transaction);
	}
}

idx_t MetaTransaction::GetActiveQuery() {
	return active_query;
}

void MetaTransaction::SetActiveQuery(transaction_t query_number) {
	active_query = query_number;
	for (auto &entry : transactions) {
		entry.second->active_query = query_number;
	}
}

void MetaTransaction::ModifyDatabase(AttachedDatabase *db) {
	if (db->IsSystem() || db->IsTemporary()) {
		// we can always modify the system and temp databases
		return;
	}
	if (!modified_database) {
		modified_database = db;
		return;
	}
	if (db != modified_database) {
		throw TransactionException(
		    "Attempting to write to database \"%s\" in a transaction that has already modified database \"%s\" - a "
		    "single transaction can only write to a single attached database.",
		    db->GetName(), modified_database->GetName());
	}
}

} // namespace duckdb












namespace duckdb {

void RollbackState::RollbackEntry(UndoFlags type, data_ptr_t data) {
	switch (type) {
	case UndoFlags::CATALOG_ENTRY: {
		// undo this catalog entry
		auto catalog_entry = Load<CatalogEntry *>(data);
		D_ASSERT(catalog_entry->set);
		catalog_entry->set->Undo(catalog_entry);
		break;
	}
	case UndoFlags::INSERT_TUPLE: {
		auto info = (AppendInfo *)data;
		// revert the append in the base table
		info->table->RevertAppend(info->start_row, info->count);
		break;
	}
	case UndoFlags::DELETE_TUPLE: {
		auto info = (DeleteInfo *)data;
		// reset the deleted flag on rollback
		info->vinfo->CommitDelete(NOT_DELETED_ID, info->rows, info->count);
		break;
	}
	case UndoFlags::UPDATE_TUPLE: {
		auto info = (UpdateInfo *)data;
		info->segment->RollbackUpdate(info);
		break;
	}
	default: // LCOV_EXCL_START
		D_ASSERT(type == UndoFlags::EMPTY_ENTRY);
		break;
	} // LCOV_EXCL_STOP
}

} // namespace duckdb





namespace duckdb {

Transaction::Transaction(TransactionManager &manager_p, ClientContext &context_p)
    : manager(manager_p), context(context_p.shared_from_this()), active_query(MAXIMUM_QUERY_ID) {
}

Transaction::~Transaction() {
}

bool Transaction::IsReadOnly() {
	auto ctxt = context.lock();
	if (!ctxt) {
		throw InternalException("Transaction::IsReadOnly() called after client context has been destroyed");
	}
	auto &db = manager.GetDB();
	return MetaTransaction::Get(*ctxt).ModifiedDatabase() != &db;
}

} // namespace duckdb








namespace duckdb {

TransactionContext::TransactionContext(ClientContext &context)
    : context(context), auto_commit(true), current_transaction(nullptr) {
}

TransactionContext::~TransactionContext() {
	if (current_transaction) {
		try {
			Rollback();
		} catch (...) {
		}
	}
}

void TransactionContext::BeginTransaction() {
	if (current_transaction) {
		throw TransactionException("cannot start a transaction within a transaction");
	}
	auto start_timestamp = Timestamp::GetCurrentTimestamp();
	auto catalog_version = Catalog::GetSystemCatalog(context).GetCatalogVersion();
	current_transaction = make_unique<MetaTransaction>(context, start_timestamp, catalog_version);

	auto &config = DBConfig::GetConfig(context);
	if (config.options.immediate_transaction_mode) {
		// if immediate transaction mode is enabled then start all transactions immediately
		auto databases = DatabaseManager::Get(context).GetDatabases(context);
		for (auto db : databases) {
			current_transaction->GetTransaction(db);
		}
	}
}

void TransactionContext::Commit() {
	if (!current_transaction) {
		throw TransactionException("failed to commit: no transaction active");
	}
	auto transaction = std::move(current_transaction);
	ClearTransaction();
	string error = transaction->Commit();
	if (!error.empty()) {
		throw TransactionException("Failed to commit: %s", error);
	}
}

void TransactionContext::SetAutoCommit(bool value) {
	auto_commit = value;
	if (!auto_commit && !current_transaction) {
		BeginTransaction();
	}
}

void TransactionContext::Rollback() {
	if (!current_transaction) {
		throw TransactionException("failed to rollback: no transaction active");
	}
	auto transaction = std::move(current_transaction);
	ClearTransaction();
	transaction->Rollback();
}

void TransactionContext::ClearTransaction() {
	SetAutoCommit(true);
	current_transaction = nullptr;
}

idx_t TransactionContext::GetActiveQuery() {
	if (!current_transaction) {
		throw InternalException("GetActiveQuery called without active transaction");
	}
	return current_transaction->GetActiveQuery();
}

void TransactionContext::ResetActiveQuery() {
	if (current_transaction) {
		SetActiveQuery(MAXIMUM_QUERY_ID);
	}
}

void TransactionContext::SetActiveQuery(transaction_t query_number) {
	if (!current_transaction) {
		throw InternalException("SetActiveQuery called without active transaction");
	}
	current_transaction->SetActiveQuery(query_number);
}

} // namespace duckdb


namespace duckdb {

TransactionManager::TransactionManager(AttachedDatabase &db) : db(db) {
}

TransactionManager::~TransactionManager() {
}

} // namespace duckdb













#include <unordered_map>

namespace duckdb {
constexpr uint32_t UNDO_ENTRY_HEADER_SIZE = sizeof(UndoFlags) + sizeof(uint32_t);

UndoBuffer::UndoBuffer(ClientContext &context_p) : context(context_p), allocator(BufferAllocator::Get(context_p)) {
}

data_ptr_t UndoBuffer::CreateEntry(UndoFlags type, idx_t len) {
	D_ASSERT(len <= NumericLimits<uint32_t>::Maximum());
	len = AlignValue(len);
	idx_t needed_space = len + UNDO_ENTRY_HEADER_SIZE;
	auto data = allocator.Allocate(needed_space);
	Store<UndoFlags>(type, data);
	data += sizeof(UndoFlags);
	Store<uint32_t>(len, data);
	data += sizeof(uint32_t);
	return data;
}

template <class T>
void UndoBuffer::IterateEntries(UndoBuffer::IteratorState &state, T &&callback) {
	// iterate in insertion order: start with the tail
	state.current = allocator.GetTail();
	while (state.current) {
		state.start = state.current->data.get();
		state.end = state.start + state.current->current_position;
		while (state.start < state.end) {
			UndoFlags type = Load<UndoFlags>(state.start);
			state.start += sizeof(UndoFlags);

			uint32_t len = Load<uint32_t>(state.start);
			state.start += sizeof(uint32_t);
			callback(type, state.start);
			state.start += len;
		}
		state.current = state.current->prev;
	}
}

template <class T>
void UndoBuffer::IterateEntries(UndoBuffer::IteratorState &state, UndoBuffer::IteratorState &end_state, T &&callback) {
	// iterate in insertion order: start with the tail
	state.current = allocator.GetTail();
	while (state.current) {
		state.start = state.current->data.get();
		state.end =
		    state.current == end_state.current ? end_state.start : state.start + state.current->current_position;
		while (state.start < state.end) {
			auto type = Load<UndoFlags>(state.start);
			state.start += sizeof(UndoFlags);
			auto len = Load<uint32_t>(state.start);
			state.start += sizeof(uint32_t);
			callback(type, state.start);
			state.start += len;
		}
		if (state.current == end_state.current) {
			// finished executing until the current end state
			return;
		}
		state.current = state.current->prev;
	}
}

template <class T>
void UndoBuffer::ReverseIterateEntries(T &&callback) {
	// iterate in reverse insertion order: start with the head
	auto current = allocator.GetHead();
	while (current) {
		data_ptr_t start = current->data.get();
		data_ptr_t end = start + current->current_position;
		// create a vector with all nodes in this chunk
		vector<pair<UndoFlags, data_ptr_t>> nodes;
		while (start < end) {
			auto type = Load<UndoFlags>(start);
			start += sizeof(UndoFlags);
			auto len = Load<uint32_t>(start);
			start += sizeof(uint32_t);
			nodes.emplace_back(type, start);
			start += len;
		}
		// iterate over it in reverse order
		for (idx_t i = nodes.size(); i > 0; i--) {
			callback(nodes[i - 1].first, nodes[i - 1].second);
		}
		current = current->next.get();
	}
}

bool UndoBuffer::ChangesMade() {
	return !allocator.IsEmpty();
}

idx_t UndoBuffer::EstimatedSize() {
	idx_t estimated_size = 0;
	auto node = allocator.GetHead();
	while (node) {
		estimated_size += node->current_position;
		node = node->next.get();
	}
	return estimated_size;
}

void UndoBuffer::Cleanup() {
	// garbage collect everything in the Undo Chunk
	// this should only happen if
	//  (1) the transaction this UndoBuffer belongs to has successfully
	//  committed
	//      (on Rollback the Rollback() function should be called, that clears
	//      the chunks)
	//  (2) there is no active transaction with start_id < commit_id of this
	//  transaction
	CleanupState state;
	UndoBuffer::IteratorState iterator_state;
	IterateEntries(iterator_state, [&](UndoFlags type, data_ptr_t data) { state.CleanupEntry(type, data); });
}

void UndoBuffer::Commit(UndoBuffer::IteratorState &iterator_state, WriteAheadLog *log, transaction_t commit_id) {
	CommitState state(context, commit_id, log);
	if (log) {
		// commit WITH write ahead log
		IterateEntries(iterator_state, [&](UndoFlags type, data_ptr_t data) { state.CommitEntry<true>(type, data); });
	} else {
		// commit WITHOUT write ahead log
		IterateEntries(iterator_state, [&](UndoFlags type, data_ptr_t data) { state.CommitEntry<false>(type, data); });
	}
}

void UndoBuffer::RevertCommit(UndoBuffer::IteratorState &end_state, transaction_t transaction_id) {
	CommitState state(context, transaction_id, nullptr);
	UndoBuffer::IteratorState start_state;
	IterateEntries(start_state, end_state, [&](UndoFlags type, data_ptr_t data) { state.RevertCommit(type, data); });
}

void UndoBuffer::Rollback() noexcept {
	// rollback needs to be performed in reverse
	RollbackState state;
	ReverseIterateEntries([&](UndoFlags type, data_ptr_t data) { state.RollbackEntry(type, data); });
}
} // namespace duckdb


namespace duckdb {

CopiedStatementVerifier::CopiedStatementVerifier(unique_ptr<SQLStatement> statement_p)
    : StatementVerifier(VerificationType::COPIED, "Copied", std::move(statement_p)) {
}

unique_ptr<StatementVerifier> CopiedStatementVerifier::Create(const SQLStatement &statement) {
	return make_unique<CopiedStatementVerifier>(statement.Copy());
}

} // namespace duckdb




namespace duckdb {

DeserializedStatementVerifier::DeserializedStatementVerifier(unique_ptr<SQLStatement> statement_p)
    : StatementVerifier(VerificationType::DESERIALIZED, "Deserialized", std::move(statement_p)) {
}

unique_ptr<StatementVerifier> DeserializedStatementVerifier::Create(const SQLStatement &statement) {
	auto &select_stmt = (SelectStatement &)statement;
	BufferedSerializer serializer;
	select_stmt.Serialize(serializer);
	BufferedDeserializer source(serializer);
	return make_unique<DeserializedStatementVerifier>(SelectStatement::Deserialize(source));
}

} // namespace duckdb


namespace duckdb {

ExternalStatementVerifier::ExternalStatementVerifier(unique_ptr<SQLStatement> statement_p)
    : StatementVerifier(VerificationType::EXTERNAL, "External", std::move(statement_p)) {
}

unique_ptr<StatementVerifier> ExternalStatementVerifier::Create(const SQLStatement &statement) {
	return make_unique<ExternalStatementVerifier>(statement.Copy());
}

} // namespace duckdb




namespace duckdb {

ParsedStatementVerifier::ParsedStatementVerifier(unique_ptr<SQLStatement> statement_p)
    : StatementVerifier(VerificationType::PARSED, "Parsed", std::move(statement_p)) {
}

unique_ptr<StatementVerifier> ParsedStatementVerifier::Create(const SQLStatement &statement) {
	auto query_str = statement.ToString();
	Parser parser;
	try {
		parser.ParseQuery(query_str);
	} catch (std::exception &ex) {
		throw InternalException("Parsed statement verification failed. Query:\n%s\n\nError: %s", query_str, ex.what());
	}
	D_ASSERT(parser.statements.size() == 1);
	D_ASSERT(parser.statements[0]->type == StatementType::SELECT_STATEMENT);
	return make_unique<ParsedStatementVerifier>(std::move(parser.statements[0]));
}

} // namespace duckdb









namespace duckdb {

PreparedStatementVerifier::PreparedStatementVerifier(unique_ptr<SQLStatement> statement_p)
    : StatementVerifier(VerificationType::PREPARED, "Prepared", std::move(statement_p)) {
}

unique_ptr<StatementVerifier> PreparedStatementVerifier::Create(const SQLStatement &statement) {
	return make_unique<PreparedStatementVerifier>(statement.Copy());
}

void PreparedStatementVerifier::Extract() {
	auto &select = *statement;
	// replace all the constants from the select statement and replace them with parameter expressions
	ParsedExpressionIterator::EnumerateQueryNodeChildren(
	    *select.node, [&](unique_ptr<ParsedExpression> &child) { ConvertConstants(child); });
	statement->n_param = values.size();
	// create the PREPARE and EXECUTE statements
	string name = "__duckdb_verification_prepared_statement";
	auto prepare = make_unique<PrepareStatement>();
	prepare->name = name;
	prepare->statement = std::move(statement);

	auto execute = make_unique<ExecuteStatement>();
	execute->name = name;
	execute->values = std::move(values);

	auto dealloc = make_unique<DropStatement>();
	dealloc->info->type = CatalogType::PREPARED_STATEMENT;
	dealloc->info->name = string(name);

	prepare_statement = std::move(prepare);
	execute_statement = std::move(execute);
	dealloc_statement = std::move(dealloc);
}

void PreparedStatementVerifier::ConvertConstants(unique_ptr<ParsedExpression> &child) {
	if (child->type == ExpressionType::VALUE_CONSTANT) {
		// constant: extract the constant value
		auto alias = child->alias;
		child->alias = string();
		// check if the value already exists
		idx_t index = values.size();
		for (idx_t v_idx = 0; v_idx < values.size(); v_idx++) {
			if (values[v_idx]->Equals(child.get())) {
				// duplicate value! refer to the original value
				index = v_idx;
				break;
			}
		}
		if (index == values.size()) {
			values.push_back(std::move(child));
		}
		// replace it with an expression
		auto parameter = make_unique<ParameterExpression>();
		parameter->parameter_nr = index + 1;
		parameter->alias = alias;
		child = std::move(parameter);
		return;
	}
	ParsedExpressionIterator::EnumerateChildren(*child,
	                                            [&](unique_ptr<ParsedExpression> &child) { ConvertConstants(child); });
}

bool PreparedStatementVerifier::Run(
    ClientContext &context, const string &query,
    const std::function<unique_ptr<QueryResult>(const string &, unique_ptr<SQLStatement>)> &run) {
	bool failed = false;
	// verify that we can extract all constants from the query and run the query as a prepared statement
	// create the PREPARE and EXECUTE statements
	Extract();
	// execute the prepared statements
	try {
		auto prepare_result = run(string(), std::move(prepare_statement));
		if (prepare_result->HasError()) {
			prepare_result->ThrowError("Failed prepare during verify: ");
		}
		auto execute_result = run(string(), std::move(execute_statement));
		if (execute_result->HasError()) {
			execute_result->ThrowError("Failed execute during verify: ");
		}
		materialized_result = unique_ptr_cast<QueryResult, MaterializedQueryResult>(std::move(execute_result));
	} catch (const Exception &ex) {
		if (ex.type != ExceptionType::PARAMETER_NOT_ALLOWED) {
			materialized_result = make_unique<MaterializedQueryResult>(PreservedError(ex));
		}
		failed = true;
	} catch (std::exception &ex) {
		materialized_result = make_unique<MaterializedQueryResult>(PreservedError(ex));
		failed = true;
	}
	run(string(), std::move(dealloc_statement));
	context.interrupted = false;

	return failed;
}

} // namespace duckdb












namespace duckdb {

StatementVerifier::StatementVerifier(VerificationType type, string name, unique_ptr<SQLStatement> statement_p)
    : type(type), name(std::move(name)),
      statement(unique_ptr_cast<SQLStatement, SelectStatement>(std::move(statement_p))),
      select_list(statement->node->GetSelectList()) {
}

StatementVerifier::StatementVerifier(unique_ptr<SQLStatement> statement_p)
    : StatementVerifier(VerificationType::ORIGINAL, "Original", std::move(statement_p)) {
}

StatementVerifier::~StatementVerifier() noexcept {
}

unique_ptr<StatementVerifier> StatementVerifier::Create(VerificationType type, const SQLStatement &statement_p) {
	switch (type) {
	case VerificationType::COPIED:
		return CopiedStatementVerifier::Create(statement_p);
	case VerificationType::DESERIALIZED:
		return DeserializedStatementVerifier::Create(statement_p);
	case VerificationType::PARSED:
		return ParsedStatementVerifier::Create(statement_p);
	case VerificationType::UNOPTIMIZED:
		return UnoptimizedStatementVerifier::Create(statement_p);
	case VerificationType::PREPARED:
		return PreparedStatementVerifier::Create(statement_p);
	case VerificationType::EXTERNAL:
		return ExternalStatementVerifier::Create(statement_p);
	case VerificationType::INVALID:
	default:
		throw InternalException("Invalid statement verification type!");
	}
}

void StatementVerifier::CheckExpressions(const StatementVerifier &other) const {
	// Only the original statement should check other statements
	D_ASSERT(type == VerificationType::ORIGINAL);

	// Check equality
	if (other.RequireEquality()) {
		D_ASSERT(statement->Equals(other.statement.get()));
	}

#ifdef DEBUG
	// Now perform checking on the expressions
	D_ASSERT(select_list.size() == other.select_list.size());
	const auto expr_count = select_list.size();
	if (other.RequireEquality()) {
		for (idx_t i = 0; i < expr_count; i++) {
			D_ASSERT(!select_list[i]->Equals(nullptr));
			// Run the ToString, to verify that it doesn't crash
			select_list[i]->ToString();

			if (select_list[i]->HasSubquery()) {
				continue;
			}

			// Check that the expressions are equivalent
			D_ASSERT(select_list[i]->Equals(other.select_list[i].get()));
			// Check that the hashes are equivalent too
			D_ASSERT(select_list[i]->Hash() == other.select_list[i]->Hash());

			other.select_list[i]->Verify();
		}
	}
#endif
}

void StatementVerifier::CheckExpressions() const {
#ifdef DEBUG
	D_ASSERT(type == VerificationType::ORIGINAL);
	// Perform additional checking within the expressions
	const auto expr_count = select_list.size();
	for (idx_t outer_idx = 0; outer_idx < expr_count; outer_idx++) {
		auto hash = select_list[outer_idx]->Hash();
		for (idx_t inner_idx = 0; inner_idx < expr_count; inner_idx++) {
			auto hash2 = select_list[inner_idx]->Hash();
			if (hash != hash2) {
				// if the hashes are not equivalent, the expressions should not be equivalent
				D_ASSERT(!select_list[outer_idx]->Equals(select_list[inner_idx].get()));
			}
		}
	}
#endif
}

bool StatementVerifier::Run(
    ClientContext &context, const string &query,
    const std::function<unique_ptr<QueryResult>(const string &, unique_ptr<SQLStatement>)> &run) {
	bool failed = false;

	context.interrupted = false;
	context.config.enable_optimizer = !DisableOptimizer();
	context.config.force_external = ForceExternal();
	try {
		auto result = run(query, std::move(statement));
		if (result->HasError()) {
			failed = true;
		}
		materialized_result = unique_ptr_cast<QueryResult, MaterializedQueryResult>(std::move(result));
	} catch (const Exception &ex) {
		failed = true;
		materialized_result = make_unique<MaterializedQueryResult>(PreservedError(ex));
	} catch (std::exception &ex) {
		failed = true;
		materialized_result = make_unique<MaterializedQueryResult>(PreservedError(ex));
	}
	context.interrupted = false;

	return failed;
}

string StatementVerifier::CompareResults(const StatementVerifier &other) {
	D_ASSERT(type == VerificationType::ORIGINAL);
	string error;
	if (materialized_result->HasError() != other.materialized_result->HasError()) { // LCOV_EXCL_START
		string result = other.name + " statement differs from original result!\n";
		result += "Original Result:\n" + materialized_result->ToString();
		result += other.name + ":\n" + other.materialized_result->ToString();
		return result;
	} // LCOV_EXCL_STOP
	if (materialized_result->HasError()) {
		return "";
	}
	if (!ColumnDataCollection::ResultEquals(materialized_result->Collection(), other.materialized_result->Collection(),
	                                        error)) { // LCOV_EXCL_START
		string result = other.name + " statement differs from original result!\n";
		result += "Original Result:\n" + materialized_result->ToString();
		result += other.name + ":\n" + other.materialized_result->ToString();
		result += "\n\n---------------------------------\n" + error;
		return result;
	} // LCOV_EXCL_STOP

	return "";
}

} // namespace duckdb


namespace duckdb {

UnoptimizedStatementVerifier::UnoptimizedStatementVerifier(unique_ptr<SQLStatement> statement_p)
    : StatementVerifier(VerificationType::UNOPTIMIZED, "Unoptimized", std::move(statement_p)) {
}

unique_ptr<StatementVerifier> UnoptimizedStatementVerifier::Create(const SQLStatement &statement_p) {
	return make_unique<UnoptimizedStatementVerifier>(statement_p.Copy());
}

} // namespace duckdb
