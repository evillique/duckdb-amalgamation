// See https://raw.githubusercontent.com/duckdb/duckdb/master/LICENSE for licensing information

#include "duckdb.hpp"
#include "duckdb-internal.hpp"
#ifndef DUCKDB_AMALGAMATION
#error header mismatch
#endif


#include <cstring>

namespace duckdb {

BufferedSerializer::BufferedSerializer(idx_t maximum_size)
    : BufferedSerializer(unique_ptr<data_t[]>(new data_t[maximum_size]), maximum_size) {
}

BufferedSerializer::BufferedSerializer(unique_ptr<data_t[]> data, idx_t size) : maximum_size(size), data(data.get()) {
	blob.size = 0;
	blob.data = std::move(data);
}

BufferedSerializer::BufferedSerializer(data_ptr_t data, idx_t size) : maximum_size(size), data(data) {
	blob.size = 0;
}

void BufferedSerializer::WriteData(const_data_ptr_t buffer, idx_t write_size) {
	if (blob.size + write_size >= maximum_size) {
		do {
			maximum_size *= 2;
		} while (blob.size + write_size > maximum_size);
		auto new_data = new data_t[maximum_size];
		memcpy(new_data, data, blob.size);
		data = new_data;
		blob.data = unique_ptr<data_t[]>(new_data);
	}

	memcpy(data + blob.size, buffer, write_size);
	blob.size += write_size;
}

} // namespace duckdb


namespace duckdb {

template <>
string Deserializer::Read() {
	uint32_t size = Read<uint32_t>();
	if (size == 0) {
		return string();
	}
	auto buffer = unique_ptr<data_t[]>(new data_t[size]);
	ReadData(buffer.get(), size);
	return string((char *)buffer.get(), size);
}

void Deserializer::ReadStringVector(vector<string> &list) {
	uint32_t sz = Read<uint32_t>();
	list.resize(sz);
	for (idx_t i = 0; i < sz; i++) {
		list[i] = Read<string>();
	}
}

} // namespace duckdb





namespace duckdb {

bool Comparators::TieIsBreakable(const idx_t &tie_col, const data_ptr_t &row_ptr, const SortLayout &sort_layout) {
	const auto &col_idx = sort_layout.sorting_to_blob_col.at(tie_col);
	// Check if the blob is NULL
	ValidityBytes row_mask(row_ptr);
	idx_t entry_idx;
	idx_t idx_in_entry;
	ValidityBytes::GetEntryIndex(col_idx, entry_idx, idx_in_entry);
	if (!row_mask.RowIsValid(row_mask.GetValidityEntry(entry_idx), idx_in_entry)) {
		// Can't break a NULL tie
		return false;
	}
	auto &row_layout = sort_layout.blob_layout;
	if (row_layout.GetTypes()[col_idx].InternalType() != PhysicalType::VARCHAR) {
		// Nested type, must be broken
		return true;
	}
	const auto &tie_col_offset = row_layout.GetOffsets()[col_idx];
	auto tie_string = Load<string_t>(row_ptr + tie_col_offset);
	if (tie_string.GetSize() < sort_layout.prefix_lengths[tie_col]) {
		// No need to break the tie - we already compared the full string
		return false;
	}
	return true;
}

int Comparators::CompareTuple(const SBScanState &left, const SBScanState &right, const data_ptr_t &l_ptr,
                              const data_ptr_t &r_ptr, const SortLayout &sort_layout, const bool &external_sort) {
	// Compare the sorting columns one by one
	int comp_res = 0;
	data_ptr_t l_ptr_offset = l_ptr;
	data_ptr_t r_ptr_offset = r_ptr;
	for (idx_t col_idx = 0; col_idx < sort_layout.column_count; col_idx++) {
		comp_res = FastMemcmp(l_ptr_offset, r_ptr_offset, sort_layout.column_sizes[col_idx]);
		if (comp_res == 0 && !sort_layout.constant_size[col_idx]) {
			comp_res = BreakBlobTie(col_idx, left, right, sort_layout, external_sort);
		}
		if (comp_res != 0) {
			break;
		}
		l_ptr_offset += sort_layout.column_sizes[col_idx];
		r_ptr_offset += sort_layout.column_sizes[col_idx];
	}
	return comp_res;
}

int Comparators::CompareVal(const data_ptr_t l_ptr, const data_ptr_t r_ptr, const LogicalType &type) {
	switch (type.InternalType()) {
	case PhysicalType::VARCHAR:
		return TemplatedCompareVal<string_t>(l_ptr, r_ptr);
	case PhysicalType::LIST:
	case PhysicalType::STRUCT: {
		auto l_nested_ptr = Load<data_ptr_t>(l_ptr);
		auto r_nested_ptr = Load<data_ptr_t>(r_ptr);
		return CompareValAndAdvance(l_nested_ptr, r_nested_ptr, type, true);
	}
	default:
		throw NotImplementedException("Unimplemented CompareVal for type %s", type.ToString());
	}
}

int Comparators::BreakBlobTie(const idx_t &tie_col, const SBScanState &left, const SBScanState &right,
                              const SortLayout &sort_layout, const bool &external) {
	data_ptr_t l_data_ptr = left.DataPtr(*left.sb->blob_sorting_data);
	data_ptr_t r_data_ptr = right.DataPtr(*right.sb->blob_sorting_data);
	if (!TieIsBreakable(tie_col, l_data_ptr, sort_layout)) {
		// Quick check to see if ties can be broken
		return 0;
	}
	// Align the pointers
	const idx_t &col_idx = sort_layout.sorting_to_blob_col.at(tie_col);
	const auto &tie_col_offset = sort_layout.blob_layout.GetOffsets()[col_idx];
	l_data_ptr += tie_col_offset;
	r_data_ptr += tie_col_offset;
	// Do the comparison
	const int order = sort_layout.order_types[tie_col] == OrderType::DESCENDING ? -1 : 1;
	const auto &type = sort_layout.blob_layout.GetTypes()[col_idx];
	int result;
	if (external) {
		// Store heap pointers
		data_ptr_t l_heap_ptr = left.HeapPtr(*left.sb->blob_sorting_data);
		data_ptr_t r_heap_ptr = right.HeapPtr(*right.sb->blob_sorting_data);
		// Unswizzle offset to pointer
		UnswizzleSingleValue(l_data_ptr, l_heap_ptr, type);
		UnswizzleSingleValue(r_data_ptr, r_heap_ptr, type);
		// Compare
		result = CompareVal(l_data_ptr, r_data_ptr, type);
		// Swizzle the pointers back to offsets
		SwizzleSingleValue(l_data_ptr, l_heap_ptr, type);
		SwizzleSingleValue(r_data_ptr, r_heap_ptr, type);
	} else {
		result = CompareVal(l_data_ptr, r_data_ptr, type);
	}
	return order * result;
}

template <class T>
int Comparators::TemplatedCompareVal(const data_ptr_t &left_ptr, const data_ptr_t &right_ptr) {
	const auto left_val = Load<T>(left_ptr);
	const auto right_val = Load<T>(right_ptr);
	if (Equals::Operation<T>(left_val, right_val)) {
		return 0;
	} else if (LessThan::Operation<T>(left_val, right_val)) {
		return -1;
	} else {
		return 1;
	}
}

int Comparators::CompareValAndAdvance(data_ptr_t &l_ptr, data_ptr_t &r_ptr, const LogicalType &type, bool valid) {
	switch (type.InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		return TemplatedCompareAndAdvance<int8_t>(l_ptr, r_ptr);
	case PhysicalType::INT16:
		return TemplatedCompareAndAdvance<int16_t>(l_ptr, r_ptr);
	case PhysicalType::INT32:
		return TemplatedCompareAndAdvance<int32_t>(l_ptr, r_ptr);
	case PhysicalType::INT64:
		return TemplatedCompareAndAdvance<int64_t>(l_ptr, r_ptr);
	case PhysicalType::UINT8:
		return TemplatedCompareAndAdvance<uint8_t>(l_ptr, r_ptr);
	case PhysicalType::UINT16:
		return TemplatedCompareAndAdvance<uint16_t>(l_ptr, r_ptr);
	case PhysicalType::UINT32:
		return TemplatedCompareAndAdvance<uint32_t>(l_ptr, r_ptr);
	case PhysicalType::UINT64:
		return TemplatedCompareAndAdvance<uint64_t>(l_ptr, r_ptr);
	case PhysicalType::INT128:
		return TemplatedCompareAndAdvance<hugeint_t>(l_ptr, r_ptr);
	case PhysicalType::FLOAT:
		return TemplatedCompareAndAdvance<float>(l_ptr, r_ptr);
	case PhysicalType::DOUBLE:
		return TemplatedCompareAndAdvance<double>(l_ptr, r_ptr);
	case PhysicalType::INTERVAL:
		return TemplatedCompareAndAdvance<interval_t>(l_ptr, r_ptr);
	case PhysicalType::VARCHAR:
		return CompareStringAndAdvance(l_ptr, r_ptr, valid);
	case PhysicalType::LIST:
		return CompareListAndAdvance(l_ptr, r_ptr, ListType::GetChildType(type), valid);
	case PhysicalType::STRUCT:
		return CompareStructAndAdvance(l_ptr, r_ptr, StructType::GetChildTypes(type), valid);
	default:
		throw NotImplementedException("Unimplemented CompareValAndAdvance for type %s", type.ToString());
	}
}

template <class T>
int Comparators::TemplatedCompareAndAdvance(data_ptr_t &left_ptr, data_ptr_t &right_ptr) {
	auto result = TemplatedCompareVal<T>(left_ptr, right_ptr);
	left_ptr += sizeof(T);
	right_ptr += sizeof(T);
	return result;
}

int Comparators::CompareStringAndAdvance(data_ptr_t &left_ptr, data_ptr_t &right_ptr, bool valid) {
	if (!valid) {
		return 0;
	}
	// Construct the string_t
	uint32_t left_string_size = Load<uint32_t>(left_ptr);
	uint32_t right_string_size = Load<uint32_t>(right_ptr);
	left_ptr += sizeof(uint32_t);
	right_ptr += sizeof(uint32_t);
	string_t left_val((const char *)left_ptr, left_string_size);
	string_t right_val((const char *)right_ptr, right_string_size);
	left_ptr += left_string_size;
	right_ptr += right_string_size;
	// Compare
	return TemplatedCompareVal<string_t>((data_ptr_t)&left_val, (data_ptr_t)&right_val);
}

int Comparators::CompareStructAndAdvance(data_ptr_t &left_ptr, data_ptr_t &right_ptr,
                                         const child_list_t<LogicalType> &types, bool valid) {
	idx_t count = types.size();
	// Load validity masks
	ValidityBytes left_validity(left_ptr);
	ValidityBytes right_validity(right_ptr);
	left_ptr += (count + 7) / 8;
	right_ptr += (count + 7) / 8;
	// Initialize variables
	bool left_valid;
	bool right_valid;
	idx_t entry_idx;
	idx_t idx_in_entry;
	// Compare
	int comp_res = 0;
	for (idx_t i = 0; i < count; i++) {
		ValidityBytes::GetEntryIndex(i, entry_idx, idx_in_entry);
		left_valid = left_validity.RowIsValid(left_validity.GetValidityEntry(entry_idx), idx_in_entry);
		right_valid = right_validity.RowIsValid(right_validity.GetValidityEntry(entry_idx), idx_in_entry);
		auto &type = types[i].second;
		if ((left_valid == right_valid) || TypeIsConstantSize(type.InternalType())) {
			comp_res = CompareValAndAdvance(left_ptr, right_ptr, types[i].second, left_valid && valid);
		}
		if (!left_valid && !right_valid) {
			comp_res = 0;
		} else if (!left_valid) {
			comp_res = 1;
		} else if (!right_valid) {
			comp_res = -1;
		}
		if (comp_res != 0) {
			break;
		}
	}
	return comp_res;
}

int Comparators::CompareListAndAdvance(data_ptr_t &left_ptr, data_ptr_t &right_ptr, const LogicalType &type,
                                       bool valid) {
	if (!valid) {
		return 0;
	}
	// Load list lengths
	auto left_len = Load<idx_t>(left_ptr);
	auto right_len = Load<idx_t>(right_ptr);
	left_ptr += sizeof(idx_t);
	right_ptr += sizeof(idx_t);
	// Load list validity masks
	ValidityBytes left_validity(left_ptr);
	ValidityBytes right_validity(right_ptr);
	left_ptr += (left_len + 7) / 8;
	right_ptr += (right_len + 7) / 8;
	// Compare
	int comp_res = 0;
	idx_t count = MinValue(left_len, right_len);
	if (TypeIsConstantSize(type.InternalType())) {
		// Templated code for fixed-size types
		switch (type.InternalType()) {
		case PhysicalType::BOOL:
		case PhysicalType::INT8:
			comp_res = TemplatedCompareListLoop<int8_t>(left_ptr, right_ptr, left_validity, right_validity, count);
			break;
		case PhysicalType::INT16:
			comp_res = TemplatedCompareListLoop<int16_t>(left_ptr, right_ptr, left_validity, right_validity, count);
			break;
		case PhysicalType::INT32:
			comp_res = TemplatedCompareListLoop<int32_t>(left_ptr, right_ptr, left_validity, right_validity, count);
			break;
		case PhysicalType::INT64:
			comp_res = TemplatedCompareListLoop<int64_t>(left_ptr, right_ptr, left_validity, right_validity, count);
			break;
		case PhysicalType::UINT8:
			comp_res = TemplatedCompareListLoop<uint8_t>(left_ptr, right_ptr, left_validity, right_validity, count);
			break;
		case PhysicalType::UINT16:
			comp_res = TemplatedCompareListLoop<uint16_t>(left_ptr, right_ptr, left_validity, right_validity, count);
			break;
		case PhysicalType::UINT32:
			comp_res = TemplatedCompareListLoop<uint32_t>(left_ptr, right_ptr, left_validity, right_validity, count);
			break;
		case PhysicalType::UINT64:
			comp_res = TemplatedCompareListLoop<uint64_t>(left_ptr, right_ptr, left_validity, right_validity, count);
			break;
		case PhysicalType::INT128:
			comp_res = TemplatedCompareListLoop<hugeint_t>(left_ptr, right_ptr, left_validity, right_validity, count);
			break;
		case PhysicalType::FLOAT:
			comp_res = TemplatedCompareListLoop<float>(left_ptr, right_ptr, left_validity, right_validity, count);
			break;
		case PhysicalType::DOUBLE:
			comp_res = TemplatedCompareListLoop<double>(left_ptr, right_ptr, left_validity, right_validity, count);
			break;
		case PhysicalType::INTERVAL:
			comp_res = TemplatedCompareListLoop<interval_t>(left_ptr, right_ptr, left_validity, right_validity, count);
			break;
		default:
			throw NotImplementedException("CompareListAndAdvance for fixed-size type %s", type.ToString());
		}
	} else {
		// Variable-sized list entries
		bool left_valid;
		bool right_valid;
		idx_t entry_idx;
		idx_t idx_in_entry;
		// Size (in bytes) of all variable-sizes entries is stored before the entries begin,
		// to make deserialization easier. We need to skip over them
		left_ptr += left_len * sizeof(idx_t);
		right_ptr += right_len * sizeof(idx_t);
		for (idx_t i = 0; i < count; i++) {
			ValidityBytes::GetEntryIndex(i, entry_idx, idx_in_entry);
			left_valid = left_validity.RowIsValid(left_validity.GetValidityEntry(entry_idx), idx_in_entry);
			right_valid = right_validity.RowIsValid(right_validity.GetValidityEntry(entry_idx), idx_in_entry);
			if (left_valid && right_valid) {
				switch (type.InternalType()) {
				case PhysicalType::LIST:
					comp_res = CompareListAndAdvance(left_ptr, right_ptr, ListType::GetChildType(type), left_valid);
					break;
				case PhysicalType::VARCHAR:
					comp_res = CompareStringAndAdvance(left_ptr, right_ptr, left_valid);
					break;
				case PhysicalType::STRUCT:
					comp_res =
					    CompareStructAndAdvance(left_ptr, right_ptr, StructType::GetChildTypes(type), left_valid);
					break;
				default:
					throw NotImplementedException("CompareListAndAdvance for variable-size type %s", type.ToString());
				}
			} else if (!left_valid && !right_valid) {
				comp_res = 0;
			} else if (left_valid) {
				comp_res = -1;
			} else {
				comp_res = 1;
			}
			if (comp_res != 0) {
				break;
			}
		}
	}
	// All values that we looped over were equal
	if (comp_res == 0 && left_len != right_len) {
		// Smaller lists first
		if (left_len < right_len) {
			comp_res = -1;
		} else {
			comp_res = 1;
		}
	}
	return comp_res;
}

template <class T>
int Comparators::TemplatedCompareListLoop(data_ptr_t &left_ptr, data_ptr_t &right_ptr,
                                          const ValidityBytes &left_validity, const ValidityBytes &right_validity,
                                          const idx_t &count) {
	int comp_res = 0;
	bool left_valid;
	bool right_valid;
	idx_t entry_idx;
	idx_t idx_in_entry;
	for (idx_t i = 0; i < count; i++) {
		ValidityBytes::GetEntryIndex(i, entry_idx, idx_in_entry);
		left_valid = left_validity.RowIsValid(left_validity.GetValidityEntry(entry_idx), idx_in_entry);
		right_valid = right_validity.RowIsValid(right_validity.GetValidityEntry(entry_idx), idx_in_entry);
		comp_res = TemplatedCompareAndAdvance<T>(left_ptr, right_ptr);
		if (!left_valid && !right_valid) {
			comp_res = 0;
		} else if (!left_valid) {
			comp_res = 1;
		} else if (!right_valid) {
			comp_res = -1;
		}
		if (comp_res != 0) {
			break;
		}
	}
	return comp_res;
}

void Comparators::UnswizzleSingleValue(data_ptr_t data_ptr, const data_ptr_t &heap_ptr, const LogicalType &type) {
	if (type.InternalType() == PhysicalType::VARCHAR) {
		data_ptr += string_t::HEADER_SIZE;
	}
	Store<data_ptr_t>(heap_ptr + Load<idx_t>(data_ptr), data_ptr);
}

void Comparators::SwizzleSingleValue(data_ptr_t data_ptr, const data_ptr_t &heap_ptr, const LogicalType &type) {
	if (type.InternalType() == PhysicalType::VARCHAR) {
		data_ptr += string_t::HEADER_SIZE;
	}
	Store<idx_t>(Load<data_ptr_t>(data_ptr) - heap_ptr, data_ptr);
}

} // namespace duckdb




namespace duckdb {

MergeSorter::MergeSorter(GlobalSortState &state, BufferManager &buffer_manager)
    : state(state), buffer_manager(buffer_manager), sort_layout(state.sort_layout) {
}

void MergeSorter::PerformInMergeRound() {
	while (true) {
		{
			lock_guard<mutex> pair_guard(state.lock);
			if (state.pair_idx == state.num_pairs) {
				break;
			}
			GetNextPartition();
		}
		MergePartition();
	}
}

void MergeSorter::MergePartition() {
	auto &left_block = *left->sb;
	auto &right_block = *right->sb;
#ifdef DEBUG
	D_ASSERT(left_block.radix_sorting_data.size() == left_block.payload_data->data_blocks.size());
	D_ASSERT(right_block.radix_sorting_data.size() == right_block.payload_data->data_blocks.size());
	if (!state.payload_layout.AllConstant() && state.external) {
		D_ASSERT(left_block.payload_data->data_blocks.size() == left_block.payload_data->heap_blocks.size());
		D_ASSERT(right_block.payload_data->data_blocks.size() == right_block.payload_data->heap_blocks.size());
	}
	if (!sort_layout.all_constant) {
		D_ASSERT(left_block.radix_sorting_data.size() == left_block.blob_sorting_data->data_blocks.size());
		D_ASSERT(right_block.radix_sorting_data.size() == right_block.blob_sorting_data->data_blocks.size());
		if (state.external) {
			D_ASSERT(left_block.blob_sorting_data->data_blocks.size() ==
			         left_block.blob_sorting_data->heap_blocks.size());
			D_ASSERT(right_block.blob_sorting_data->data_blocks.size() ==
			         right_block.blob_sorting_data->heap_blocks.size());
		}
	}
#endif
	// Set up the write block
	// Each merge task produces a SortedBlock with exactly state.block_capacity rows or less
	result->InitializeWrite();
	// Initialize arrays to store merge data
	bool left_smaller[STANDARD_VECTOR_SIZE];
	idx_t next_entry_sizes[STANDARD_VECTOR_SIZE];
	// Merge loop
#ifdef DEBUG
	auto l_count = left->Remaining();
	auto r_count = right->Remaining();
#endif
	while (true) {
		auto l_remaining = left->Remaining();
		auto r_remaining = right->Remaining();
		if (l_remaining + r_remaining == 0) {
			// Done
			break;
		}
		const idx_t next = MinValue(l_remaining + r_remaining, (idx_t)STANDARD_VECTOR_SIZE);
		if (l_remaining != 0 && r_remaining != 0) {
			// Compute the merge (not needed if one side is exhausted)
			ComputeMerge(next, left_smaller);
		}
		// Actually merge the data (radix, blob, and payload)
		MergeRadix(next, left_smaller);
		if (!sort_layout.all_constant) {
			MergeData(*result->blob_sorting_data, *left_block.blob_sorting_data, *right_block.blob_sorting_data, next,
			          left_smaller, next_entry_sizes, true);
			D_ASSERT(result->radix_sorting_data.size() == result->blob_sorting_data->data_blocks.size());
		}
		MergeData(*result->payload_data, *left_block.payload_data, *right_block.payload_data, next, left_smaller,
		          next_entry_sizes, false);
		D_ASSERT(result->radix_sorting_data.size() == result->payload_data->data_blocks.size());
	}
#ifdef DEBUG
	D_ASSERT(result->Count() == l_count + r_count);
#endif
}

void MergeSorter::GetNextPartition() {
	// Create result block
	state.sorted_blocks_temp[state.pair_idx].push_back(make_unique<SortedBlock>(buffer_manager, state));
	result = state.sorted_blocks_temp[state.pair_idx].back().get();
	// Determine which blocks must be merged
	auto &left_block = *state.sorted_blocks[state.pair_idx * 2];
	auto &right_block = *state.sorted_blocks[state.pair_idx * 2 + 1];
	const idx_t l_count = left_block.Count();
	const idx_t r_count = right_block.Count();
	// Initialize left and right reader
	left = make_unique<SBScanState>(buffer_manager, state);
	right = make_unique<SBScanState>(buffer_manager, state);
	// Compute the work that this thread must do using Merge Path
	idx_t l_end;
	idx_t r_end;
	if (state.l_start + state.r_start + state.block_capacity < l_count + r_count) {
		left->sb = state.sorted_blocks[state.pair_idx * 2].get();
		right->sb = state.sorted_blocks[state.pair_idx * 2 + 1].get();
		const idx_t intersection = state.l_start + state.r_start + state.block_capacity;
		GetIntersection(intersection, l_end, r_end);
		D_ASSERT(l_end <= l_count);
		D_ASSERT(r_end <= r_count);
		D_ASSERT(intersection == l_end + r_end);
	} else {
		l_end = l_count;
		r_end = r_count;
	}
	// Create slices of the data that this thread must merge
	left->SetIndices(0, 0);
	right->SetIndices(0, 0);
	left_input = left_block.CreateSlice(state.l_start, l_end, left->entry_idx);
	right_input = right_block.CreateSlice(state.r_start, r_end, right->entry_idx);
	left->sb = left_input.get();
	right->sb = right_input.get();
	state.l_start = l_end;
	state.r_start = r_end;
	D_ASSERT(left->Remaining() + right->Remaining() == state.block_capacity || (l_end == l_count && r_end == r_count));
	// Update global state
	if (state.l_start == l_count && state.r_start == r_count) {
		// Delete references to previous pair
		state.sorted_blocks[state.pair_idx * 2] = nullptr;
		state.sorted_blocks[state.pair_idx * 2 + 1] = nullptr;
		// Advance pair
		state.pair_idx++;
		state.l_start = 0;
		state.r_start = 0;
	}
}

int MergeSorter::CompareUsingGlobalIndex(SBScanState &l, SBScanState &r, const idx_t l_idx, const idx_t r_idx) {
	D_ASSERT(l_idx < l.sb->Count());
	D_ASSERT(r_idx < r.sb->Count());

	// Easy comparison using the previous result (intersections must increase monotonically)
	if (l_idx < state.l_start) {
		return -1;
	}
	if (r_idx < state.r_start) {
		return 1;
	}

	l.sb->GlobalToLocalIndex(l_idx, l.block_idx, l.entry_idx);
	r.sb->GlobalToLocalIndex(r_idx, r.block_idx, r.entry_idx);

	l.PinRadix(l.block_idx);
	r.PinRadix(r.block_idx);
	data_ptr_t l_ptr = l.radix_handle.Ptr() + l.entry_idx * sort_layout.entry_size;
	data_ptr_t r_ptr = r.radix_handle.Ptr() + r.entry_idx * sort_layout.entry_size;

	int comp_res;
	if (sort_layout.all_constant) {
		comp_res = FastMemcmp(l_ptr, r_ptr, sort_layout.comparison_size);
	} else {
		l.PinData(*l.sb->blob_sorting_data);
		r.PinData(*r.sb->blob_sorting_data);
		comp_res = Comparators::CompareTuple(l, r, l_ptr, r_ptr, sort_layout, state.external);
	}
	return comp_res;
}

void MergeSorter::GetIntersection(const idx_t diagonal, idx_t &l_idx, idx_t &r_idx) {
	const idx_t l_count = left->sb->Count();
	const idx_t r_count = right->sb->Count();
	// Cover some edge cases
	// Code coverage off because these edge cases cannot happen unless other code changes
	// Edge cases have been tested extensively while developing Merge Path in a script
	// LCOV_EXCL_START
	if (diagonal >= l_count + r_count) {
		l_idx = l_count;
		r_idx = r_count;
		return;
	} else if (diagonal == 0) {
		l_idx = 0;
		r_idx = 0;
		return;
	} else if (l_count == 0) {
		l_idx = 0;
		r_idx = diagonal;
		return;
	} else if (r_count == 0) {
		r_idx = 0;
		l_idx = diagonal;
		return;
	}
	// LCOV_EXCL_STOP
	// Determine offsets for the binary search
	const idx_t l_offset = MinValue(l_count, diagonal);
	const idx_t r_offset = diagonal > l_count ? diagonal - l_count : 0;
	D_ASSERT(l_offset + r_offset == diagonal);
	const idx_t search_space = diagonal > MaxValue(l_count, r_count) ? l_count + r_count - diagonal
	                                                                 : MinValue(diagonal, MinValue(l_count, r_count));
	// Double binary search
	idx_t li = 0;
	idx_t ri = search_space - 1;
	idx_t middle;
	int comp_res;
	while (li <= ri) {
		middle = (li + ri) / 2;
		l_idx = l_offset - middle;
		r_idx = r_offset + middle;
		if (l_idx == l_count || r_idx == 0) {
			comp_res = CompareUsingGlobalIndex(*left, *right, l_idx - 1, r_idx);
			if (comp_res > 0) {
				l_idx--;
				r_idx++;
			} else {
				return;
			}
			if (l_idx == 0 || r_idx == r_count) {
				// This case is incredibly difficult to cover as it is dependent on parallelism randomness
				// But it has been tested extensively during development in a script
				// LCOV_EXCL_START
				return;
				// LCOV_EXCL_STOP
			} else {
				break;
			}
		}
		comp_res = CompareUsingGlobalIndex(*left, *right, l_idx, r_idx);
		if (comp_res > 0) {
			li = middle + 1;
		} else {
			ri = middle - 1;
		}
	}
	int l_r_min1 = CompareUsingGlobalIndex(*left, *right, l_idx, r_idx - 1);
	int l_min1_r = CompareUsingGlobalIndex(*left, *right, l_idx - 1, r_idx);
	if (l_r_min1 > 0 && l_min1_r < 0) {
		return;
	} else if (l_r_min1 > 0) {
		l_idx--;
		r_idx++;
	} else if (l_min1_r < 0) {
		l_idx++;
		r_idx--;
	}
}

void MergeSorter::ComputeMerge(const idx_t &count, bool left_smaller[]) {
	auto &l = *left;
	auto &r = *right;
	auto &l_sorted_block = *l.sb;
	auto &r_sorted_block = *r.sb;
	// Save indices to restore afterwards
	idx_t l_block_idx_before = l.block_idx;
	idx_t l_entry_idx_before = l.entry_idx;
	idx_t r_block_idx_before = r.block_idx;
	idx_t r_entry_idx_before = r.entry_idx;
	// Data pointers for both sides
	data_ptr_t l_radix_ptr;
	data_ptr_t r_radix_ptr;
	// Compute the merge of the next 'count' tuples
	idx_t compared = 0;
	while (compared < count) {
		// Move to the next block (if needed)
		if (l.block_idx < l_sorted_block.radix_sorting_data.size() &&
		    l.entry_idx == l_sorted_block.radix_sorting_data[l.block_idx]->count) {
			l.block_idx++;
			l.entry_idx = 0;
		}
		if (r.block_idx < r_sorted_block.radix_sorting_data.size() &&
		    r.entry_idx == r_sorted_block.radix_sorting_data[r.block_idx]->count) {
			r.block_idx++;
			r.entry_idx = 0;
		}
		const bool l_done = l.block_idx == l_sorted_block.radix_sorting_data.size();
		const bool r_done = r.block_idx == r_sorted_block.radix_sorting_data.size();
		if (l_done || r_done) {
			// One of the sides is exhausted, no need to compare
			break;
		}
		// Pin the radix sorting data
		if (!l_done) {
			left->PinRadix(l.block_idx);
			l_radix_ptr = left->RadixPtr();
		}
		if (!r_done) {
			right->PinRadix(r.block_idx);
			r_radix_ptr = right->RadixPtr();
		}
		const idx_t &l_count = !l_done ? l_sorted_block.radix_sorting_data[l.block_idx]->count : 0;
		const idx_t &r_count = !r_done ? r_sorted_block.radix_sorting_data[r.block_idx]->count : 0;
		// Compute the merge
		if (sort_layout.all_constant) {
			// All sorting columns are constant size
			for (; compared < count && l.entry_idx < l_count && r.entry_idx < r_count; compared++) {
				left_smaller[compared] = FastMemcmp(l_radix_ptr, r_radix_ptr, sort_layout.comparison_size) < 0;
				const bool &l_smaller = left_smaller[compared];
				const bool r_smaller = !l_smaller;
				// Use comparison bool (0 or 1) to increment entries and pointers
				l.entry_idx += l_smaller;
				r.entry_idx += r_smaller;
				l_radix_ptr += l_smaller * sort_layout.entry_size;
				r_radix_ptr += r_smaller * sort_layout.entry_size;
			}
		} else {
			// Pin the blob data
			if (!l_done) {
				left->PinData(*l_sorted_block.blob_sorting_data);
			}
			if (!r_done) {
				right->PinData(*r_sorted_block.blob_sorting_data);
			}
			// Merge with variable size sorting columns
			for (; compared < count && l.entry_idx < l_count && r.entry_idx < r_count; compared++) {
				left_smaller[compared] =
				    Comparators::CompareTuple(*left, *right, l_radix_ptr, r_radix_ptr, sort_layout, state.external) < 0;
				const bool &l_smaller = left_smaller[compared];
				const bool r_smaller = !l_smaller;
				// Use comparison bool (0 or 1) to increment entries and pointers
				l.entry_idx += l_smaller;
				r.entry_idx += r_smaller;
				l_radix_ptr += l_smaller * sort_layout.entry_size;
				r_radix_ptr += r_smaller * sort_layout.entry_size;
			}
		}
	}
	// Reset block indices
	left->SetIndices(l_block_idx_before, l_entry_idx_before);
	right->SetIndices(r_block_idx_before, r_entry_idx_before);
}

void MergeSorter::MergeRadix(const idx_t &count, const bool left_smaller[]) {
	auto &l = *left;
	auto &r = *right;
	// Save indices to restore afterwards
	idx_t l_block_idx_before = l.block_idx;
	idx_t l_entry_idx_before = l.entry_idx;
	idx_t r_block_idx_before = r.block_idx;
	idx_t r_entry_idx_before = r.entry_idx;

	auto &l_blocks = l.sb->radix_sorting_data;
	auto &r_blocks = r.sb->radix_sorting_data;
	RowDataBlock *l_block = nullptr;
	RowDataBlock *r_block = nullptr;

	data_ptr_t l_ptr;
	data_ptr_t r_ptr;

	RowDataBlock *result_block = result->radix_sorting_data.back().get();
	auto result_handle = buffer_manager.Pin(result_block->block);
	data_ptr_t result_ptr = result_handle.Ptr() + result_block->count * sort_layout.entry_size;

	idx_t copied = 0;
	while (copied < count) {
		// Move to the next block (if needed)
		if (l.block_idx < l_blocks.size() && l.entry_idx == l_blocks[l.block_idx]->count) {
			// Delete reference to previous block
			l_blocks[l.block_idx]->block = nullptr;
			// Advance block
			l.block_idx++;
			l.entry_idx = 0;
		}
		if (r.block_idx < r_blocks.size() && r.entry_idx == r_blocks[r.block_idx]->count) {
			// Delete reference to previous block
			r_blocks[r.block_idx]->block = nullptr;
			// Advance block
			r.block_idx++;
			r.entry_idx = 0;
		}
		const bool l_done = l.block_idx == l_blocks.size();
		const bool r_done = r.block_idx == r_blocks.size();
		// Pin the radix sortable blocks
		idx_t l_count;
		if (!l_done) {
			l_block = l_blocks[l.block_idx].get();
			left->PinRadix(l.block_idx);
			l_ptr = l.RadixPtr();
			l_count = l_block->count;
		} else {
			l_count = 0;
		}
		idx_t r_count;
		if (!r_done) {
			r_block = r_blocks[r.block_idx].get();
			r.PinRadix(r.block_idx);
			r_ptr = r.RadixPtr();
			r_count = r_block->count;
		} else {
			r_count = 0;
		}
		// Copy using computed merge
		if (!l_done && !r_done) {
			// Both sides have data - merge
			MergeRows(l_ptr, l.entry_idx, l_count, r_ptr, r.entry_idx, r_count, *result_block, result_ptr,
			          sort_layout.entry_size, left_smaller, copied, count);
		} else if (r_done) {
			// Right side is exhausted
			FlushRows(l_ptr, l.entry_idx, l_count, *result_block, result_ptr, sort_layout.entry_size, copied, count);
		} else {
			// Left side is exhausted
			FlushRows(r_ptr, r.entry_idx, r_count, *result_block, result_ptr, sort_layout.entry_size, copied, count);
		}
	}
	// Reset block indices
	left->SetIndices(l_block_idx_before, l_entry_idx_before);
	right->SetIndices(r_block_idx_before, r_entry_idx_before);
}

void MergeSorter::MergeData(SortedData &result_data, SortedData &l_data, SortedData &r_data, const idx_t &count,
                            const bool left_smaller[], idx_t next_entry_sizes[], bool reset_indices) {
	auto &l = *left;
	auto &r = *right;
	// Save indices to restore afterwards
	idx_t l_block_idx_before = l.block_idx;
	idx_t l_entry_idx_before = l.entry_idx;
	idx_t r_block_idx_before = r.block_idx;
	idx_t r_entry_idx_before = r.entry_idx;

	const auto &layout = result_data.layout;
	const idx_t row_width = layout.GetRowWidth();
	const idx_t heap_pointer_offset = layout.GetHeapOffset();

	// Left and right row data to merge
	data_ptr_t l_ptr;
	data_ptr_t r_ptr;
	// Accompanying left and right heap data (if needed)
	data_ptr_t l_heap_ptr;
	data_ptr_t r_heap_ptr;

	// Result rows to write to
	RowDataBlock *result_data_block = result_data.data_blocks.back().get();
	auto result_data_handle = buffer_manager.Pin(result_data_block->block);
	data_ptr_t result_data_ptr = result_data_handle.Ptr() + result_data_block->count * row_width;
	// Result heap to write to (if needed)
	RowDataBlock *result_heap_block = nullptr;
	BufferHandle result_heap_handle;
	data_ptr_t result_heap_ptr;
	if (!layout.AllConstant() && state.external) {
		result_heap_block = result_data.heap_blocks.back().get();
		result_heap_handle = buffer_manager.Pin(result_heap_block->block);
		result_heap_ptr = result_heap_handle.Ptr() + result_heap_block->byte_offset;
	}

	idx_t copied = 0;
	while (copied < count) {
		// Move to new data blocks (if needed)
		if (l.block_idx < l_data.data_blocks.size() && l.entry_idx == l_data.data_blocks[l.block_idx]->count) {
			// Delete reference to previous block
			l_data.data_blocks[l.block_idx]->block = nullptr;
			if (!layout.AllConstant() && state.external) {
				l_data.heap_blocks[l.block_idx]->block = nullptr;
			}
			// Advance block
			l.block_idx++;
			l.entry_idx = 0;
		}
		if (r.block_idx < r_data.data_blocks.size() && r.entry_idx == r_data.data_blocks[r.block_idx]->count) {
			// Delete reference to previous block
			r_data.data_blocks[r.block_idx]->block = nullptr;
			if (!layout.AllConstant() && state.external) {
				r_data.heap_blocks[r.block_idx]->block = nullptr;
			}
			// Advance block
			r.block_idx++;
			r.entry_idx = 0;
		}
		const bool l_done = l.block_idx == l_data.data_blocks.size();
		const bool r_done = r.block_idx == r_data.data_blocks.size();
		// Pin the row data blocks
		if (!l_done) {
			l.PinData(l_data);
			l_ptr = l.DataPtr(l_data);
		}
		if (!r_done) {
			r.PinData(r_data);
			r_ptr = r.DataPtr(r_data);
		}
		const idx_t &l_count = !l_done ? l_data.data_blocks[l.block_idx]->count : 0;
		const idx_t &r_count = !r_done ? r_data.data_blocks[r.block_idx]->count : 0;
		// Perform the merge
		if (layout.AllConstant() || !state.external) {
			// If all constant size, or if we are doing an in-memory sort, we do not need to touch the heap
			if (!l_done && !r_done) {
				// Both sides have data - merge
				MergeRows(l_ptr, l.entry_idx, l_count, r_ptr, r.entry_idx, r_count, *result_data_block, result_data_ptr,
				          row_width, left_smaller, copied, count);
			} else if (r_done) {
				// Right side is exhausted
				FlushRows(l_ptr, l.entry_idx, l_count, *result_data_block, result_data_ptr, row_width, copied, count);
			} else {
				// Left side is exhausted
				FlushRows(r_ptr, r.entry_idx, r_count, *result_data_block, result_data_ptr, row_width, copied, count);
			}
		} else {
			// External sorting with variable size data. Pin the heap blocks too
			if (!l_done) {
				l_heap_ptr = l.BaseHeapPtr(l_data) + Load<idx_t>(l_ptr + heap_pointer_offset);
				D_ASSERT(l_heap_ptr - l.BaseHeapPtr(l_data) >= 0);
				D_ASSERT((idx_t)(l_heap_ptr - l.BaseHeapPtr(l_data)) < l_data.heap_blocks[l.block_idx]->byte_offset);
			}
			if (!r_done) {
				r_heap_ptr = r.BaseHeapPtr(r_data) + Load<idx_t>(r_ptr + heap_pointer_offset);
				D_ASSERT(r_heap_ptr - r.BaseHeapPtr(r_data) >= 0);
				D_ASSERT((idx_t)(r_heap_ptr - r.BaseHeapPtr(r_data)) < r_data.heap_blocks[r.block_idx]->byte_offset);
			}
			// Both the row and heap data need to be dealt with
			if (!l_done && !r_done) {
				// Both sides have data - merge
				idx_t l_idx_copy = l.entry_idx;
				idx_t r_idx_copy = r.entry_idx;
				data_ptr_t result_data_ptr_copy = result_data_ptr;
				idx_t copied_copy = copied;
				// Merge row data
				MergeRows(l_ptr, l_idx_copy, l_count, r_ptr, r_idx_copy, r_count, *result_data_block,
				          result_data_ptr_copy, row_width, left_smaller, copied_copy, count);
				const idx_t merged = copied_copy - copied;
				// Compute the entry sizes and number of heap bytes that will be copied
				idx_t copy_bytes = 0;
				data_ptr_t l_heap_ptr_copy = l_heap_ptr;
				data_ptr_t r_heap_ptr_copy = r_heap_ptr;
				for (idx_t i = 0; i < merged; i++) {
					// Store base heap offset in the row data
					Store<idx_t>(result_heap_block->byte_offset + copy_bytes, result_data_ptr + heap_pointer_offset);
					result_data_ptr += row_width;
					// Compute entry size and add to total
					const bool &l_smaller = left_smaller[copied + i];
					const bool r_smaller = !l_smaller;
					auto &entry_size = next_entry_sizes[copied + i];
					entry_size =
					    l_smaller * Load<uint32_t>(l_heap_ptr_copy) + r_smaller * Load<uint32_t>(r_heap_ptr_copy);
					D_ASSERT(entry_size >= sizeof(uint32_t));
					D_ASSERT(l_heap_ptr_copy - l.BaseHeapPtr(l_data) + l_smaller * entry_size <=
					         l_data.heap_blocks[l.block_idx]->byte_offset);
					D_ASSERT(r_heap_ptr_copy - r.BaseHeapPtr(r_data) + r_smaller * entry_size <=
					         r_data.heap_blocks[r.block_idx]->byte_offset);
					l_heap_ptr_copy += l_smaller * entry_size;
					r_heap_ptr_copy += r_smaller * entry_size;
					copy_bytes += entry_size;
				}
				// Reallocate result heap block size (if needed)
				if (result_heap_block->byte_offset + copy_bytes > result_heap_block->capacity) {
					idx_t new_capacity = result_heap_block->byte_offset + copy_bytes;
					buffer_manager.ReAllocate(result_heap_block->block, new_capacity);
					result_heap_block->capacity = new_capacity;
					result_heap_ptr = result_heap_handle.Ptr() + result_heap_block->byte_offset;
				}
				D_ASSERT(result_heap_block->byte_offset + copy_bytes <= result_heap_block->capacity);
				// Now copy the heap data
				for (idx_t i = 0; i < merged; i++) {
					const bool &l_smaller = left_smaller[copied + i];
					const bool r_smaller = !l_smaller;
					const auto &entry_size = next_entry_sizes[copied + i];
					memcpy(result_heap_ptr, (data_ptr_t)(l_smaller * (idx_t)l_heap_ptr + r_smaller * (idx_t)r_heap_ptr),
					       entry_size);
					D_ASSERT(Load<uint32_t>(result_heap_ptr) == entry_size);
					result_heap_ptr += entry_size;
					l_heap_ptr += l_smaller * entry_size;
					r_heap_ptr += r_smaller * entry_size;
					l.entry_idx += l_smaller;
					r.entry_idx += r_smaller;
				}
				// Update result indices and pointers
				result_heap_block->count += merged;
				result_heap_block->byte_offset += copy_bytes;
				copied += merged;
			} else if (r_done) {
				// Right side is exhausted - flush left
				FlushBlobs(layout, l_count, l_ptr, l.entry_idx, l_heap_ptr, *result_data_block, result_data_ptr,
				           *result_heap_block, result_heap_handle, result_heap_ptr, copied, count);
			} else {
				// Left side is exhausted - flush right
				FlushBlobs(layout, r_count, r_ptr, r.entry_idx, r_heap_ptr, *result_data_block, result_data_ptr,
				           *result_heap_block, result_heap_handle, result_heap_ptr, copied, count);
			}
			D_ASSERT(result_data_block->count == result_heap_block->count);
		}
	}
	if (reset_indices) {
		left->SetIndices(l_block_idx_before, l_entry_idx_before);
		right->SetIndices(r_block_idx_before, r_entry_idx_before);
	}
}

void MergeSorter::MergeRows(data_ptr_t &l_ptr, idx_t &l_entry_idx, const idx_t &l_count, data_ptr_t &r_ptr,
                            idx_t &r_entry_idx, const idx_t &r_count, RowDataBlock &target_block,
                            data_ptr_t &target_ptr, const idx_t &entry_size, const bool left_smaller[], idx_t &copied,
                            const idx_t &count) {
	const idx_t next = MinValue(count - copied, target_block.capacity - target_block.count);
	idx_t i;
	for (i = 0; i < next && l_entry_idx < l_count && r_entry_idx < r_count; i++) {
		const bool &l_smaller = left_smaller[copied + i];
		const bool r_smaller = !l_smaller;
		// Use comparison bool (0 or 1) to copy an entry from either side
		FastMemcpy(target_ptr, (data_ptr_t)(l_smaller * (idx_t)l_ptr + r_smaller * (idx_t)r_ptr), entry_size);
		target_ptr += entry_size;
		// Use the comparison bool to increment entries and pointers
		l_entry_idx += l_smaller;
		r_entry_idx += r_smaller;
		l_ptr += l_smaller * entry_size;
		r_ptr += r_smaller * entry_size;
	}
	// Update counts
	target_block.count += i;
	copied += i;
}

void MergeSorter::FlushRows(data_ptr_t &source_ptr, idx_t &source_entry_idx, const idx_t &source_count,
                            RowDataBlock &target_block, data_ptr_t &target_ptr, const idx_t &entry_size, idx_t &copied,
                            const idx_t &count) {
	// Compute how many entries we can fit
	idx_t next = MinValue(count - copied, target_block.capacity - target_block.count);
	next = MinValue(next, source_count - source_entry_idx);
	// Copy them all in a single memcpy
	const idx_t copy_bytes = next * entry_size;
	memcpy(target_ptr, source_ptr, copy_bytes);
	target_ptr += copy_bytes;
	source_ptr += copy_bytes;
	// Update counts
	source_entry_idx += next;
	target_block.count += next;
	copied += next;
}

void MergeSorter::FlushBlobs(const RowLayout &layout, const idx_t &source_count, data_ptr_t &source_data_ptr,
                             idx_t &source_entry_idx, data_ptr_t &source_heap_ptr, RowDataBlock &target_data_block,
                             data_ptr_t &target_data_ptr, RowDataBlock &target_heap_block,
                             BufferHandle &target_heap_handle, data_ptr_t &target_heap_ptr, idx_t &copied,
                             const idx_t &count) {
	const idx_t row_width = layout.GetRowWidth();
	const idx_t heap_pointer_offset = layout.GetHeapOffset();
	idx_t source_entry_idx_copy = source_entry_idx;
	data_ptr_t target_data_ptr_copy = target_data_ptr;
	idx_t copied_copy = copied;
	// Flush row data
	FlushRows(source_data_ptr, source_entry_idx_copy, source_count, target_data_block, target_data_ptr_copy, row_width,
	          copied_copy, count);
	const idx_t flushed = copied_copy - copied;
	// Compute the entry sizes and number of heap bytes that will be copied
	idx_t copy_bytes = 0;
	data_ptr_t source_heap_ptr_copy = source_heap_ptr;
	for (idx_t i = 0; i < flushed; i++) {
		// Store base heap offset in the row data
		Store<idx_t>(target_heap_block.byte_offset + copy_bytes, target_data_ptr + heap_pointer_offset);
		target_data_ptr += row_width;
		// Compute entry size and add to total
		auto entry_size = Load<uint32_t>(source_heap_ptr_copy);
		D_ASSERT(entry_size >= sizeof(uint32_t));
		source_heap_ptr_copy += entry_size;
		copy_bytes += entry_size;
	}
	// Reallocate result heap block size (if needed)
	if (target_heap_block.byte_offset + copy_bytes > target_heap_block.capacity) {
		idx_t new_capacity = target_heap_block.byte_offset + copy_bytes;
		buffer_manager.ReAllocate(target_heap_block.block, new_capacity);
		target_heap_block.capacity = new_capacity;
		target_heap_ptr = target_heap_handle.Ptr() + target_heap_block.byte_offset;
	}
	D_ASSERT(target_heap_block.byte_offset + copy_bytes <= target_heap_block.capacity);
	// Copy the heap data in one go
	memcpy(target_heap_ptr, source_heap_ptr, copy_bytes);
	target_heap_ptr += copy_bytes;
	source_heap_ptr += copy_bytes;
	source_entry_idx += flushed;
	copied += flushed;
	// Update result indices and pointers
	target_heap_block.count += flushed;
	target_heap_block.byte_offset += copy_bytes;
	D_ASSERT(target_heap_block.byte_offset <= target_heap_block.capacity);
}

} // namespace duckdb





namespace duckdb {

//! Calls std::sort on strings that are tied by their prefix after the radix sort
static void SortTiedBlobs(BufferManager &buffer_manager, const data_ptr_t dataptr, const idx_t &start, const idx_t &end,
                          const idx_t &tie_col, bool *ties, const data_ptr_t blob_ptr, const SortLayout &sort_layout) {
	const auto row_width = sort_layout.blob_layout.GetRowWidth();
	// Locate the first blob row in question
	data_ptr_t row_ptr = dataptr + start * sort_layout.entry_size;
	data_ptr_t blob_row_ptr = blob_ptr + Load<uint32_t>(row_ptr + sort_layout.comparison_size) * row_width;
	if (!Comparators::TieIsBreakable(tie_col, blob_row_ptr, sort_layout)) {
		// Quick check to see if ties can be broken
		return;
	}
	// Fill pointer array for sorting
	auto ptr_block = unique_ptr<data_ptr_t[]>(new data_ptr_t[end - start]);
	auto entry_ptrs = (data_ptr_t *)ptr_block.get();
	for (idx_t i = start; i < end; i++) {
		entry_ptrs[i - start] = row_ptr;
		row_ptr += sort_layout.entry_size;
	}
	// Slow pointer-based sorting
	const int order = sort_layout.order_types[tie_col] == OrderType::DESCENDING ? -1 : 1;
	const idx_t &col_idx = sort_layout.sorting_to_blob_col.at(tie_col);
	const auto &tie_col_offset = sort_layout.blob_layout.GetOffsets()[col_idx];
	auto logical_type = sort_layout.blob_layout.GetTypes()[col_idx];
	std::sort(entry_ptrs, entry_ptrs + end - start,
	          [&blob_ptr, &order, &sort_layout, &tie_col_offset, &row_width, &logical_type](const data_ptr_t l,
	                                                                                        const data_ptr_t r) {
		          idx_t left_idx = Load<uint32_t>(l + sort_layout.comparison_size);
		          idx_t right_idx = Load<uint32_t>(r + sort_layout.comparison_size);
		          data_ptr_t left_ptr = blob_ptr + left_idx * row_width + tie_col_offset;
		          data_ptr_t right_ptr = blob_ptr + right_idx * row_width + tie_col_offset;
		          return order * Comparators::CompareVal(left_ptr, right_ptr, logical_type) < 0;
	          });
	// Re-order
	auto temp_block = buffer_manager.GetBufferAllocator().Allocate((end - start) * sort_layout.entry_size);
	data_ptr_t temp_ptr = temp_block.get();
	for (idx_t i = 0; i < end - start; i++) {
		FastMemcpy(temp_ptr, entry_ptrs[i], sort_layout.entry_size);
		temp_ptr += sort_layout.entry_size;
	}
	memcpy(dataptr + start * sort_layout.entry_size, temp_block.get(), (end - start) * sort_layout.entry_size);
	// Determine if there are still ties (if this is not the last column)
	if (tie_col < sort_layout.column_count - 1) {
		data_ptr_t idx_ptr = dataptr + start * sort_layout.entry_size + sort_layout.comparison_size;
		// Load current entry
		data_ptr_t current_ptr = blob_ptr + Load<uint32_t>(idx_ptr) * row_width + tie_col_offset;
		for (idx_t i = 0; i < end - start - 1; i++) {
			// Load next entry and compare
			idx_ptr += sort_layout.entry_size;
			data_ptr_t next_ptr = blob_ptr + Load<uint32_t>(idx_ptr) * row_width + tie_col_offset;
			ties[start + i] = Comparators::CompareVal(current_ptr, next_ptr, logical_type) == 0;
			current_ptr = next_ptr;
		}
	}
}

//! Identifies sequences of rows that are tied by the prefix of a blob column, and sorts them
static void SortTiedBlobs(BufferManager &buffer_manager, SortedBlock &sb, bool *ties, data_ptr_t dataptr,
                          const idx_t &count, const idx_t &tie_col, const SortLayout &sort_layout) {
	D_ASSERT(!ties[count - 1]);
	auto &blob_block = *sb.blob_sorting_data->data_blocks.back();
	auto blob_handle = buffer_manager.Pin(blob_block.block);
	const data_ptr_t blob_ptr = blob_handle.Ptr();

	for (idx_t i = 0; i < count; i++) {
		if (!ties[i]) {
			continue;
		}
		idx_t j;
		for (j = i; j < count; j++) {
			if (!ties[j]) {
				break;
			}
		}
		SortTiedBlobs(buffer_manager, dataptr, i, j + 1, tie_col, ties, blob_ptr, sort_layout);
		i = j;
	}
}

//! Returns whether there are any 'true' values in the ties[] array
static bool AnyTies(bool ties[], const idx_t &count) {
	D_ASSERT(!ties[count - 1]);
	bool any_ties = false;
	for (idx_t i = 0; i < count - 1; i++) {
		any_ties = any_ties || ties[i];
	}
	return any_ties;
}

//! Compares subsequent rows to check for ties
static void ComputeTies(data_ptr_t dataptr, const idx_t &count, const idx_t &col_offset, const idx_t &tie_size,
                        bool ties[], const SortLayout &sort_layout) {
	D_ASSERT(!ties[count - 1]);
	D_ASSERT(col_offset + tie_size <= sort_layout.comparison_size);
	// Align dataptr
	dataptr += col_offset;
	for (idx_t i = 0; i < count - 1; i++) {
		ties[i] = ties[i] && FastMemcmp(dataptr, dataptr + sort_layout.entry_size, tie_size) == 0;
		dataptr += sort_layout.entry_size;
	}
}

//! Textbook LSD radix sort
void RadixSortLSD(BufferManager &buffer_manager, const data_ptr_t &dataptr, const idx_t &count, const idx_t &col_offset,
                  const idx_t &row_width, const idx_t &sorting_size) {
	auto temp_block = buffer_manager.GetBufferAllocator().Allocate(count * row_width);
	bool swap = false;

	idx_t counts[SortConstants::VALUES_PER_RADIX];
	for (idx_t r = 1; r <= sorting_size; r++) {
		// Init counts to 0
		memset(counts, 0, sizeof(counts));
		// Const some values for convenience
		const data_ptr_t source_ptr = swap ? temp_block.get() : dataptr;
		const data_ptr_t target_ptr = swap ? dataptr : temp_block.get();
		const idx_t offset = col_offset + sorting_size - r;
		// Collect counts
		data_ptr_t offset_ptr = source_ptr + offset;
		for (idx_t i = 0; i < count; i++) {
			counts[*offset_ptr]++;
			offset_ptr += row_width;
		}
		// Compute offsets from counts
		idx_t max_count = counts[0];
		for (idx_t val = 1; val < SortConstants::VALUES_PER_RADIX; val++) {
			max_count = MaxValue<idx_t>(max_count, counts[val]);
			counts[val] = counts[val] + counts[val - 1];
		}
		if (max_count == count) {
			continue;
		}
		// Re-order the data in temporary array
		data_ptr_t row_ptr = source_ptr + (count - 1) * row_width;
		for (idx_t i = 0; i < count; i++) {
			idx_t &radix_offset = --counts[*(row_ptr + offset)];
			FastMemcpy(target_ptr + radix_offset * row_width, row_ptr, row_width);
			row_ptr -= row_width;
		}
		swap = !swap;
	}
	// Move data back to original buffer (if it was swapped)
	if (swap) {
		memcpy(dataptr, temp_block.get(), count * row_width);
	}
}

//! Insertion sort, used when count of values is low
inline void InsertionSort(const data_ptr_t orig_ptr, const data_ptr_t temp_ptr, const idx_t &count,
                          const idx_t &col_offset, const idx_t &row_width, const idx_t &total_comp_width,
                          const idx_t &offset, bool swap) {
	const data_ptr_t source_ptr = swap ? temp_ptr : orig_ptr;
	const data_ptr_t target_ptr = swap ? orig_ptr : temp_ptr;
	if (count > 1) {
		const idx_t total_offset = col_offset + offset;
		auto temp_val = unique_ptr<data_t[]>(new data_t[row_width]);
		const data_ptr_t val = temp_val.get();
		const auto comp_width = total_comp_width - offset;
		for (idx_t i = 1; i < count; i++) {
			FastMemcpy(val, source_ptr + i * row_width, row_width);
			idx_t j = i;
			while (j > 0 &&
			       FastMemcmp(source_ptr + (j - 1) * row_width + total_offset, val + total_offset, comp_width) > 0) {
				FastMemcpy(source_ptr + j * row_width, source_ptr + (j - 1) * row_width, row_width);
				j--;
			}
			FastMemcpy(source_ptr + j * row_width, val, row_width);
		}
	}
	if (swap) {
		memcpy(target_ptr, source_ptr, count * row_width);
	}
}

//! MSD radix sort that switches to insertion sort with low bucket sizes
void RadixSortMSD(const data_ptr_t orig_ptr, const data_ptr_t temp_ptr, const idx_t &count, const idx_t &col_offset,
                  const idx_t &row_width, const idx_t &comp_width, const idx_t &offset, idx_t locations[], bool swap) {
	const data_ptr_t source_ptr = swap ? temp_ptr : orig_ptr;
	const data_ptr_t target_ptr = swap ? orig_ptr : temp_ptr;
	// Init counts to 0
	memset(locations, 0, SortConstants::MSD_RADIX_LOCATIONS * sizeof(idx_t));
	idx_t *counts = locations + 1;
	// Collect counts
	const idx_t total_offset = col_offset + offset;
	data_ptr_t offset_ptr = source_ptr + total_offset;
	for (idx_t i = 0; i < count; i++) {
		counts[*offset_ptr]++;
		offset_ptr += row_width;
	}
	// Compute locations from counts
	idx_t max_count = 0;
	for (idx_t radix = 0; radix < SortConstants::VALUES_PER_RADIX; radix++) {
		max_count = MaxValue<idx_t>(max_count, counts[radix]);
		counts[radix] += locations[radix];
	}
	if (max_count != count) {
		// Re-order the data in temporary array
		data_ptr_t row_ptr = source_ptr;
		for (idx_t i = 0; i < count; i++) {
			const idx_t &radix_offset = locations[*(row_ptr + total_offset)]++;
			FastMemcpy(target_ptr + radix_offset * row_width, row_ptr, row_width);
			row_ptr += row_width;
		}
		swap = !swap;
	}
	// Check if done
	if (offset == comp_width - 1) {
		if (swap) {
			memcpy(orig_ptr, temp_ptr, count * row_width);
		}
		return;
	}
	if (max_count == count) {
		RadixSortMSD(orig_ptr, temp_ptr, count, col_offset, row_width, comp_width, offset + 1,
		             locations + SortConstants::MSD_RADIX_LOCATIONS, swap);
		return;
	}
	// Recurse
	idx_t radix_count = locations[0];
	for (idx_t radix = 0; radix < SortConstants::VALUES_PER_RADIX; radix++) {
		const idx_t loc = (locations[radix] - radix_count) * row_width;
		if (radix_count > SortConstants::INSERTION_SORT_THRESHOLD) {
			RadixSortMSD(orig_ptr + loc, temp_ptr + loc, radix_count, col_offset, row_width, comp_width, offset + 1,
			             locations + SortConstants::MSD_RADIX_LOCATIONS, swap);
		} else if (radix_count != 0) {
			InsertionSort(orig_ptr + loc, temp_ptr + loc, radix_count, col_offset, row_width, comp_width, offset + 1,
			              swap);
		}
		radix_count = locations[radix + 1] - locations[radix];
	}
}

//! Calls different sort functions, depending on the count and sorting sizes
void RadixSort(BufferManager &buffer_manager, const data_ptr_t &dataptr, const idx_t &count, const idx_t &col_offset,
               const idx_t &sorting_size, const SortLayout &sort_layout, bool contains_string) {
	if (contains_string) {
		auto begin = duckdb_pdqsort::PDQIterator(dataptr, sort_layout.entry_size);
		auto end = begin + count;
		duckdb_pdqsort::PDQConstants constants(sort_layout.entry_size, col_offset, sorting_size, *end);
		duckdb_pdqsort::pdqsort_branchless(begin, begin + count, constants);
	} else if (count <= SortConstants::INSERTION_SORT_THRESHOLD) {
		InsertionSort(dataptr, nullptr, count, 0, sort_layout.entry_size, sort_layout.comparison_size, 0, false);
	} else if (sorting_size <= SortConstants::MSD_RADIX_SORT_SIZE_THRESHOLD) {
		RadixSortLSD(buffer_manager, dataptr, count, col_offset, sort_layout.entry_size, sorting_size);
	} else {
		auto temp_block = buffer_manager.Allocate(MaxValue(count * sort_layout.entry_size, (idx_t)Storage::BLOCK_SIZE));
		auto preallocated_array = unique_ptr<idx_t[]>(new idx_t[sorting_size * SortConstants::MSD_RADIX_LOCATIONS]);
		RadixSortMSD(dataptr, temp_block.Ptr(), count, col_offset, sort_layout.entry_size, sorting_size, 0,
		             preallocated_array.get(), false);
	}
}

//! Identifies sequences of rows that are tied, and calls radix sort on these
static void SubSortTiedTuples(BufferManager &buffer_manager, const data_ptr_t dataptr, const idx_t &count,
                              const idx_t &col_offset, const idx_t &sorting_size, bool ties[],
                              const SortLayout &sort_layout, bool contains_string) {
	D_ASSERT(!ties[count - 1]);
	for (idx_t i = 0; i < count; i++) {
		if (!ties[i]) {
			continue;
		}
		idx_t j;
		for (j = i + 1; j < count; j++) {
			if (!ties[j]) {
				break;
			}
		}
		RadixSort(buffer_manager, dataptr + i * sort_layout.entry_size, j - i + 1, col_offset, sorting_size,
		          sort_layout, contains_string);
		i = j;
	}
}

void LocalSortState::SortInMemory() {
	auto &sb = *sorted_blocks.back();
	auto &block = *sb.radix_sorting_data.back();
	const auto &count = block.count;
	auto handle = buffer_manager->Pin(block.block);
	const auto dataptr = handle.Ptr();
	// Assign an index to each row
	data_ptr_t idx_dataptr = dataptr + sort_layout->comparison_size;
	for (uint32_t i = 0; i < count; i++) {
		Store<uint32_t>(i, idx_dataptr);
		idx_dataptr += sort_layout->entry_size;
	}
	// Radix sort and break ties until no more ties, or until all columns are sorted
	idx_t sorting_size = 0;
	idx_t col_offset = 0;
	unique_ptr<bool[]> ties_ptr;
	bool *ties = nullptr;
	bool contains_string = false;
	for (idx_t i = 0; i < sort_layout->column_count; i++) {
		sorting_size += sort_layout->column_sizes[i];
		contains_string = contains_string || sort_layout->logical_types[i].InternalType() == PhysicalType::VARCHAR;
		if (sort_layout->constant_size[i] && i < sort_layout->column_count - 1) {
			// Add columns to the sorting size until we reach a variable size column, or the last column
			continue;
		}

		if (!ties) {
			// This is the first sort
			RadixSort(*buffer_manager, dataptr, count, col_offset, sorting_size, *sort_layout, contains_string);
			ties_ptr = unique_ptr<bool[]>(new bool[count]);
			ties = ties_ptr.get();
			std::fill_n(ties, count - 1, true);
			ties[count - 1] = false;
		} else {
			// For subsequent sorts, we only have to subsort the tied tuples
			SubSortTiedTuples(*buffer_manager, dataptr, count, col_offset, sorting_size, ties, *sort_layout,
			                  contains_string);
		}

		contains_string = false;

		if (sort_layout->constant_size[i] && i == sort_layout->column_count - 1) {
			// All columns are sorted, no ties to break because last column is constant size
			break;
		}

		ComputeTies(dataptr, count, col_offset, sorting_size, ties, *sort_layout);
		if (!AnyTies(ties, count)) {
			// No ties, stop sorting
			break;
		}

		if (!sort_layout->constant_size[i]) {
			SortTiedBlobs(*buffer_manager, sb, ties, dataptr, count, i, *sort_layout);
			if (!AnyTies(ties, count)) {
				// No more ties after tie-breaking, stop
				break;
			}
		}

		col_offset += sorting_size;
		sorting_size = 0;
	}
}

} // namespace duckdb







#include <algorithm>
#include <numeric>

namespace duckdb {

idx_t GetNestedSortingColSize(idx_t &col_size, const LogicalType &type) {
	auto physical_type = type.InternalType();
	if (TypeIsConstantSize(physical_type)) {
		col_size += GetTypeIdSize(physical_type);
		return 0;
	} else {
		switch (physical_type) {
		case PhysicalType::VARCHAR: {
			// Nested strings are between 4 and 11 chars long for alignment
			auto size_before_str = col_size;
			col_size += 11;
			col_size -= (col_size - 12) % 8;
			return col_size - size_before_str;
		}
		case PhysicalType::LIST:
			// Lists get 2 bytes (null and empty list)
			col_size += 2;
			return GetNestedSortingColSize(col_size, ListType::GetChildType(type));
		case PhysicalType::STRUCT:
			// Structs get 1 bytes (null)
			col_size++;
			return GetNestedSortingColSize(col_size, StructType::GetChildType(type, 0));
		default:
			throw NotImplementedException("Unable to order column with type %s", type.ToString());
		}
	}
}

SortLayout::SortLayout(const vector<BoundOrderByNode> &orders)
    : column_count(orders.size()), all_constant(true), comparison_size(0), entry_size(0) {
	vector<LogicalType> blob_layout_types;
	for (idx_t i = 0; i < column_count; i++) {
		const auto &order = orders[i];

		order_types.push_back(order.type);
		order_by_null_types.push_back(order.null_order);
		auto &expr = *order.expression;
		logical_types.push_back(expr.return_type);

		auto physical_type = expr.return_type.InternalType();
		constant_size.push_back(TypeIsConstantSize(physical_type));

		if (order.stats) {
			stats.push_back(order.stats.get());
			has_null.push_back(stats.back()->CanHaveNull());
		} else {
			stats.push_back(nullptr);
			has_null.push_back(true);
		}

		idx_t col_size = has_null.back() ? 1 : 0;
		prefix_lengths.push_back(0);
		if (!TypeIsConstantSize(physical_type) && physical_type != PhysicalType::VARCHAR) {
			prefix_lengths.back() = GetNestedSortingColSize(col_size, expr.return_type);
		} else if (physical_type == PhysicalType::VARCHAR) {
			idx_t size_before = col_size;
			if (stats.back()) {
				auto &str_stats = (StringStatistics &)*stats.back();
				col_size += str_stats.max_string_length;
				if (col_size > 12) {
					col_size = 12;
				} else {
					constant_size.back() = true;
				}
			} else {
				col_size = 12;
			}
			prefix_lengths.back() = col_size - size_before;
		} else {
			col_size += GetTypeIdSize(physical_type);
		}

		comparison_size += col_size;
		column_sizes.push_back(col_size);
	}
	entry_size = comparison_size + sizeof(uint32_t);

	// 8-byte alignment
	if (entry_size % 8 != 0) {
		// First assign more bytes to strings instead of aligning
		idx_t bytes_to_fill = 8 - (entry_size % 8);
		for (idx_t col_idx = 0; col_idx < column_count; col_idx++) {
			if (bytes_to_fill == 0) {
				break;
			}
			if (logical_types[col_idx].InternalType() == PhysicalType::VARCHAR && stats[col_idx]) {
				auto &str_stats = (StringStatistics &)*stats[col_idx];
				idx_t diff = str_stats.max_string_length - prefix_lengths[col_idx];
				if (diff > 0) {
					// Increase all sizes accordingly
					idx_t increase = MinValue(bytes_to_fill, diff);
					column_sizes[col_idx] += increase;
					prefix_lengths[col_idx] += increase;
					constant_size[col_idx] = increase == diff;
					comparison_size += increase;
					entry_size += increase;
					bytes_to_fill -= increase;
				}
			}
		}
		entry_size = AlignValue(entry_size);
	}

	for (idx_t col_idx = 0; col_idx < column_count; col_idx++) {
		all_constant = all_constant && constant_size[col_idx];
		if (!constant_size[col_idx]) {
			sorting_to_blob_col[col_idx] = blob_layout_types.size();
			blob_layout_types.push_back(logical_types[col_idx]);
		}
	}

	blob_layout.Initialize(blob_layout_types);
}

SortLayout SortLayout::GetPrefixComparisonLayout(idx_t num_prefix_cols) const {
	SortLayout result;
	result.column_count = num_prefix_cols;
	result.all_constant = true;
	result.comparison_size = 0;
	for (idx_t col_idx = 0; col_idx < num_prefix_cols; col_idx++) {
		result.order_types.push_back(order_types[col_idx]);
		result.order_by_null_types.push_back(order_by_null_types[col_idx]);
		result.logical_types.push_back(logical_types[col_idx]);

		result.all_constant = result.all_constant && constant_size[col_idx];
		result.constant_size.push_back(constant_size[col_idx]);

		result.comparison_size += column_sizes[col_idx];
		result.column_sizes.push_back(column_sizes[col_idx]);

		result.prefix_lengths.push_back(prefix_lengths[col_idx]);
		result.stats.push_back(stats[col_idx]);
		result.has_null.push_back(has_null[col_idx]);
	}
	result.entry_size = entry_size;
	result.blob_layout = blob_layout;
	result.sorting_to_blob_col = sorting_to_blob_col;
	return result;
}

LocalSortState::LocalSortState() : initialized(false) {
	if (!Radix::IsLittleEndian()) {
		throw NotImplementedException("Sorting is not supported on big endian architectures");
	}
}

void LocalSortState::Initialize(GlobalSortState &global_sort_state, BufferManager &buffer_manager_p) {
	sort_layout = &global_sort_state.sort_layout;
	payload_layout = &global_sort_state.payload_layout;
	buffer_manager = &buffer_manager_p;
	// Radix sorting data
	radix_sorting_data = make_unique<RowDataCollection>(
	    *buffer_manager, RowDataCollection::EntriesPerBlock(sort_layout->entry_size), sort_layout->entry_size);
	// Blob sorting data
	if (!sort_layout->all_constant) {
		auto blob_row_width = sort_layout->blob_layout.GetRowWidth();
		blob_sorting_data = make_unique<RowDataCollection>(
		    *buffer_manager, RowDataCollection::EntriesPerBlock(blob_row_width), blob_row_width);
		blob_sorting_heap = make_unique<RowDataCollection>(*buffer_manager, (idx_t)Storage::BLOCK_SIZE, 1, true);
	}
	// Payload data
	auto payload_row_width = payload_layout->GetRowWidth();
	payload_data = make_unique<RowDataCollection>(
	    *buffer_manager, RowDataCollection::EntriesPerBlock(payload_row_width), payload_row_width);
	payload_heap = make_unique<RowDataCollection>(*buffer_manager, (idx_t)Storage::BLOCK_SIZE, 1, true);
	// Init done
	initialized = true;
}

void LocalSortState::SinkChunk(DataChunk &sort, DataChunk &payload) {
	D_ASSERT(sort.size() == payload.size());
	// Build and serialize sorting data to radix sortable rows
	auto data_pointers = FlatVector::GetData<data_ptr_t>(addresses);
	auto handles = radix_sorting_data->Build(sort.size(), data_pointers, nullptr);
	for (idx_t sort_col = 0; sort_col < sort.ColumnCount(); sort_col++) {
		bool has_null = sort_layout->has_null[sort_col];
		bool nulls_first = sort_layout->order_by_null_types[sort_col] == OrderByNullType::NULLS_FIRST;
		bool desc = sort_layout->order_types[sort_col] == OrderType::DESCENDING;
		RowOperations::RadixScatter(sort.data[sort_col], sort.size(), sel_ptr, sort.size(), data_pointers, desc,
		                            has_null, nulls_first, sort_layout->prefix_lengths[sort_col],
		                            sort_layout->column_sizes[sort_col]);
	}

	// Also fully serialize blob sorting columns (to be able to break ties
	if (!sort_layout->all_constant) {
		DataChunk blob_chunk;
		blob_chunk.SetCardinality(sort.size());
		for (idx_t sort_col = 0; sort_col < sort.ColumnCount(); sort_col++) {
			if (!sort_layout->constant_size[sort_col]) {
				blob_chunk.data.emplace_back(sort.data[sort_col]);
			}
		}
		handles = blob_sorting_data->Build(blob_chunk.size(), data_pointers, nullptr);
		auto blob_data = blob_chunk.ToUnifiedFormat();
		RowOperations::Scatter(blob_chunk, blob_data.get(), sort_layout->blob_layout, addresses, *blob_sorting_heap,
		                       sel_ptr, blob_chunk.size());
		D_ASSERT(blob_sorting_heap->keep_pinned);
	}

	// Finally, serialize payload data
	handles = payload_data->Build(payload.size(), data_pointers, nullptr);
	auto input_data = payload.ToUnifiedFormat();
	RowOperations::Scatter(payload, input_data.get(), *payload_layout, addresses, *payload_heap, sel_ptr,
	                       payload.size());
	D_ASSERT(payload_heap->keep_pinned);
}

idx_t LocalSortState::SizeInBytes() const {
	idx_t size_in_bytes = radix_sorting_data->SizeInBytes() + payload_data->SizeInBytes();
	if (!sort_layout->all_constant) {
		size_in_bytes += blob_sorting_data->SizeInBytes() + blob_sorting_heap->SizeInBytes();
	}
	if (!payload_layout->AllConstant()) {
		size_in_bytes += payload_heap->SizeInBytes();
	}
	return size_in_bytes;
}

void LocalSortState::Sort(GlobalSortState &global_sort_state, bool reorder_heap) {
	D_ASSERT(radix_sorting_data->count == payload_data->count);
	if (radix_sorting_data->count == 0) {
		return;
	}
	// Move all data to a single SortedBlock
	sorted_blocks.emplace_back(make_unique<SortedBlock>(*buffer_manager, global_sort_state));
	auto &sb = *sorted_blocks.back();
	// Fixed-size sorting data
	auto sorting_block = ConcatenateBlocks(*radix_sorting_data);
	sb.radix_sorting_data.push_back(std::move(sorting_block));
	// Variable-size sorting data
	if (!sort_layout->all_constant) {
		auto &blob_data = *blob_sorting_data;
		auto new_block = ConcatenateBlocks(blob_data);
		sb.blob_sorting_data->data_blocks.push_back(std::move(new_block));
	}
	// Payload data
	auto payload_block = ConcatenateBlocks(*payload_data);
	sb.payload_data->data_blocks.push_back(std::move(payload_block));
	// Now perform the actual sort
	SortInMemory();
	// Re-order before the merge sort
	ReOrder(global_sort_state, reorder_heap);
}

unique_ptr<RowDataBlock> LocalSortState::ConcatenateBlocks(RowDataCollection &row_data) {
	//	Don't copy and delete if there is only one block.
	if (row_data.blocks.size() == 1) {
		auto new_block = std::move(row_data.blocks[0]);
		row_data.blocks.clear();
		row_data.count = 0;
		return new_block;
	}
	// Create block with the correct capacity
	auto buffer_manager = &row_data.buffer_manager;
	const idx_t &entry_size = row_data.entry_size;
	idx_t capacity = MaxValue(((idx_t)Storage::BLOCK_SIZE + entry_size - 1) / entry_size, row_data.count);
	auto new_block = make_unique<RowDataBlock>(*buffer_manager, capacity, entry_size);
	new_block->count = row_data.count;
	auto new_block_handle = buffer_manager->Pin(new_block->block);
	data_ptr_t new_block_ptr = new_block_handle.Ptr();
	// Copy the data of the blocks into a single block
	for (idx_t i = 0; i < row_data.blocks.size(); i++) {
		auto &block = row_data.blocks[i];
		auto block_handle = buffer_manager->Pin(block->block);
		memcpy(new_block_ptr, block_handle.Ptr(), block->count * entry_size);
		new_block_ptr += block->count * entry_size;
		block.reset();
	}
	row_data.blocks.clear();
	row_data.count = 0;
	return new_block;
}

void LocalSortState::ReOrder(SortedData &sd, data_ptr_t sorting_ptr, RowDataCollection &heap, GlobalSortState &gstate,
                             bool reorder_heap) {
	sd.swizzled = reorder_heap;
	auto &unordered_data_block = sd.data_blocks.back();
	const idx_t count = unordered_data_block->count;
	auto unordered_data_handle = buffer_manager->Pin(unordered_data_block->block);
	const data_ptr_t unordered_data_ptr = unordered_data_handle.Ptr();
	// Create new block that will hold re-ordered row data
	auto ordered_data_block =
	    make_unique<RowDataBlock>(*buffer_manager, unordered_data_block->capacity, unordered_data_block->entry_size);
	ordered_data_block->count = count;
	auto ordered_data_handle = buffer_manager->Pin(ordered_data_block->block);
	data_ptr_t ordered_data_ptr = ordered_data_handle.Ptr();
	// Re-order fixed-size row layout
	const idx_t row_width = sd.layout.GetRowWidth();
	const idx_t sorting_entry_size = gstate.sort_layout.entry_size;
	for (idx_t i = 0; i < count; i++) {
		auto index = Load<uint32_t>(sorting_ptr);
		FastMemcpy(ordered_data_ptr, unordered_data_ptr + index * row_width, row_width);
		ordered_data_ptr += row_width;
		sorting_ptr += sorting_entry_size;
	}
	ordered_data_block->block->SetSwizzling(sd.swizzled ? "LocalSortState::ReOrder.ordered_data" : nullptr);
	// Replace the unordered data block with the re-ordered data block
	sd.data_blocks.clear();
	sd.data_blocks.push_back(std::move(ordered_data_block));
	// Deal with the heap (if necessary)
	if (!sd.layout.AllConstant() && reorder_heap) {
		// Swizzle the column pointers to offsets
		RowOperations::SwizzleColumns(sd.layout, ordered_data_handle.Ptr(), count);
		sd.data_blocks.back()->block->SetSwizzling(nullptr);
		// Create a single heap block to store the ordered heap
		idx_t total_byte_offset =
		    std::accumulate(heap.blocks.begin(), heap.blocks.end(), 0,
		                    [](idx_t a, const unique_ptr<RowDataBlock> &b) { return a + b->byte_offset; });
		idx_t heap_block_size = MaxValue(total_byte_offset, (idx_t)Storage::BLOCK_SIZE);
		auto ordered_heap_block = make_unique<RowDataBlock>(*buffer_manager, heap_block_size, 1);
		ordered_heap_block->count = count;
		ordered_heap_block->byte_offset = total_byte_offset;
		auto ordered_heap_handle = buffer_manager->Pin(ordered_heap_block->block);
		data_ptr_t ordered_heap_ptr = ordered_heap_handle.Ptr();
		// Fill the heap in order
		ordered_data_ptr = ordered_data_handle.Ptr();
		const idx_t heap_pointer_offset = sd.layout.GetHeapOffset();
		for (idx_t i = 0; i < count; i++) {
			auto heap_row_ptr = Load<data_ptr_t>(ordered_data_ptr + heap_pointer_offset);
			auto heap_row_size = Load<uint32_t>(heap_row_ptr);
			memcpy(ordered_heap_ptr, heap_row_ptr, heap_row_size);
			ordered_heap_ptr += heap_row_size;
			ordered_data_ptr += row_width;
		}
		// Swizzle the base pointer to the offset of each row in the heap
		RowOperations::SwizzleHeapPointer(sd.layout, ordered_data_handle.Ptr(), ordered_heap_handle.Ptr(), count);
		// Move the re-ordered heap to the SortedData, and clear the local heap
		sd.heap_blocks.push_back(std::move(ordered_heap_block));
		heap.pinned_blocks.clear();
		heap.blocks.clear();
		heap.count = 0;
	}
}

void LocalSortState::ReOrder(GlobalSortState &gstate, bool reorder_heap) {
	auto &sb = *sorted_blocks.back();
	auto sorting_handle = buffer_manager->Pin(sb.radix_sorting_data.back()->block);
	const data_ptr_t sorting_ptr = sorting_handle.Ptr() + gstate.sort_layout.comparison_size;
	// Re-order variable size sorting columns
	if (!gstate.sort_layout.all_constant) {
		ReOrder(*sb.blob_sorting_data, sorting_ptr, *blob_sorting_heap, gstate, reorder_heap);
	}
	// And the payload
	ReOrder(*sb.payload_data, sorting_ptr, *payload_heap, gstate, reorder_heap);
}

GlobalSortState::GlobalSortState(BufferManager &buffer_manager, const vector<BoundOrderByNode> &orders,
                                 RowLayout &payload_layout)
    : buffer_manager(buffer_manager), sort_layout(SortLayout(orders)), payload_layout(payload_layout),
      block_capacity(0), external(false) {
}

void GlobalSortState::AddLocalState(LocalSortState &local_sort_state) {
	if (!local_sort_state.radix_sorting_data) {
		return;
	}

	// Sort accumulated data
	// we only re-order the heap when the data is expected to not fit in memory
	// re-ordering the heap avoids random access when reading/merging but incurs a significant cost of shuffling data
	// when data fits in memory, doing random access on reads is cheaper than re-shuffling
	local_sort_state.Sort(*this, external || !local_sort_state.sorted_blocks.empty());

	// Append local state sorted data to this global state
	lock_guard<mutex> append_guard(lock);
	for (auto &sb : local_sort_state.sorted_blocks) {
		sorted_blocks.push_back(std::move(sb));
	}
	auto &payload_heap = local_sort_state.payload_heap;
	for (idx_t i = 0; i < payload_heap->blocks.size(); i++) {
		heap_blocks.push_back(std::move(payload_heap->blocks[i]));
		pinned_blocks.push_back(std::move(payload_heap->pinned_blocks[i]));
	}
	if (!sort_layout.all_constant) {
		auto &blob_heap = local_sort_state.blob_sorting_heap;
		for (idx_t i = 0; i < blob_heap->blocks.size(); i++) {
			heap_blocks.push_back(std::move(blob_heap->blocks[i]));
			pinned_blocks.push_back(std::move(blob_heap->pinned_blocks[i]));
		}
	}
}

void GlobalSortState::PrepareMergePhase() {
	// Determine if we need to use do an external sort
	idx_t total_heap_size =
	    std::accumulate(sorted_blocks.begin(), sorted_blocks.end(), (idx_t)0,
	                    [](idx_t a, const unique_ptr<SortedBlock> &b) { return a + b->HeapSize(); });
	if (external || (pinned_blocks.empty() && total_heap_size > 0.25 * buffer_manager.GetMaxMemory())) {
		external = true;
	}
	// Use the data that we have to determine which partition size to use during the merge
	if (external && total_heap_size > 0) {
		// If we have variable size data we need to be conservative, as there might be skew
		idx_t max_block_size = 0;
		for (auto &sb : sorted_blocks) {
			idx_t size_in_bytes = sb->SizeInBytes();
			if (size_in_bytes > max_block_size) {
				max_block_size = size_in_bytes;
				block_capacity = sb->Count();
			}
		}
	} else {
		for (auto &sb : sorted_blocks) {
			block_capacity = MaxValue(block_capacity, sb->Count());
		}
	}
	// Unswizzle and pin heap blocks if we can fit everything in memory
	if (!external) {
		for (auto &sb : sorted_blocks) {
			sb->blob_sorting_data->Unswizzle();
			sb->payload_data->Unswizzle();
		}
	}
}

void GlobalSortState::InitializeMergeRound() {
	D_ASSERT(sorted_blocks_temp.empty());
	// If we reverse this list, the blocks that were merged last will be merged first in the next round
	// These are still in memory, therefore this reduces the amount of read/write to disk!
	std::reverse(sorted_blocks.begin(), sorted_blocks.end());
	// Uneven number of blocks - keep one on the side
	if (sorted_blocks.size() % 2 == 1) {
		odd_one_out = std::move(sorted_blocks.back());
		sorted_blocks.pop_back();
	}
	// Init merge path path indices
	pair_idx = 0;
	num_pairs = sorted_blocks.size() / 2;
	l_start = 0;
	r_start = 0;
	// Allocate room for merge results
	for (idx_t p_idx = 0; p_idx < num_pairs; p_idx++) {
		sorted_blocks_temp.emplace_back();
	}
}

void GlobalSortState::CompleteMergeRound(bool keep_radix_data) {
	sorted_blocks.clear();
	for (auto &sorted_block_vector : sorted_blocks_temp) {
		sorted_blocks.push_back(make_unique<SortedBlock>(buffer_manager, *this));
		sorted_blocks.back()->AppendSortedBlocks(sorted_block_vector);
	}
	sorted_blocks_temp.clear();
	if (odd_one_out) {
		sorted_blocks.push_back(std::move(odd_one_out));
		odd_one_out = nullptr;
	}
	// Only one block left: Done!
	if (sorted_blocks.size() == 1 && !keep_radix_data) {
		sorted_blocks[0]->radix_sorting_data.clear();
		sorted_blocks[0]->blob_sorting_data = nullptr;
	}
}
void GlobalSortState::Print() {
	PayloadScanner scanner(*this, false);
	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), scanner.GetPayloadTypes());
	for (;;) {
		scanner.Scan(chunk);
		const auto count = chunk.size();
		if (!count) {
			break;
		}
		chunk.Print();
	}
}

} // namespace duckdb







#include <numeric>

namespace duckdb {

SortedData::SortedData(SortedDataType type, const RowLayout &layout, BufferManager &buffer_manager,
                       GlobalSortState &state)
    : type(type), layout(layout), swizzled(state.external), buffer_manager(buffer_manager), state(state) {
}

idx_t SortedData::Count() {
	idx_t count = std::accumulate(data_blocks.begin(), data_blocks.end(), (idx_t)0,
	                              [](idx_t a, const unique_ptr<RowDataBlock> &b) { return a + b->count; });
	if (!layout.AllConstant() && state.external) {
		D_ASSERT(count == std::accumulate(heap_blocks.begin(), heap_blocks.end(), (idx_t)0,
		                                  [](idx_t a, const unique_ptr<RowDataBlock> &b) { return a + b->count; }));
	}
	return count;
}

void SortedData::CreateBlock() {
	auto capacity =
	    MaxValue(((idx_t)Storage::BLOCK_SIZE + layout.GetRowWidth() - 1) / layout.GetRowWidth(), state.block_capacity);
	data_blocks.push_back(make_unique<RowDataBlock>(buffer_manager, capacity, layout.GetRowWidth()));
	if (!layout.AllConstant() && state.external) {
		heap_blocks.push_back(make_unique<RowDataBlock>(buffer_manager, (idx_t)Storage::BLOCK_SIZE, 1));
		D_ASSERT(data_blocks.size() == heap_blocks.size());
	}
}

unique_ptr<SortedData> SortedData::CreateSlice(idx_t start_block_index, idx_t end_block_index, idx_t end_entry_index) {
	// Add the corresponding blocks to the result
	auto result = make_unique<SortedData>(type, layout, buffer_manager, state);
	for (idx_t i = start_block_index; i <= end_block_index; i++) {
		result->data_blocks.push_back(data_blocks[i]->Copy());
		if (!layout.AllConstant() && state.external) {
			result->heap_blocks.push_back(heap_blocks[i]->Copy());
		}
	}
	// All of the blocks that come before block with idx = start_block_idx can be reset (other references exist)
	for (idx_t i = 0; i < start_block_index; i++) {
		data_blocks[i]->block = nullptr;
		if (!layout.AllConstant() && state.external) {
			heap_blocks[i]->block = nullptr;
		}
	}
	// Use start and end entry indices to set the boundaries
	D_ASSERT(end_entry_index <= result->data_blocks.back()->count);
	result->data_blocks.back()->count = end_entry_index;
	if (!layout.AllConstant() && state.external) {
		result->heap_blocks.back()->count = end_entry_index;
	}
	return result;
}

void SortedData::Unswizzle() {
	if (layout.AllConstant() || !swizzled) {
		return;
	}
	for (idx_t i = 0; i < data_blocks.size(); i++) {
		auto &data_block = data_blocks[i];
		auto &heap_block = heap_blocks[i];
		D_ASSERT(data_block->block->IsSwizzled());
		auto data_handle_p = buffer_manager.Pin(data_block->block);
		auto heap_handle_p = buffer_manager.Pin(heap_block->block);
		RowOperations::UnswizzlePointers(layout, data_handle_p.Ptr(), heap_handle_p.Ptr(), data_block->count);
		data_block->block->SetSwizzling("SortedData::Unswizzle");
		state.heap_blocks.push_back(std::move(heap_block));
		state.pinned_blocks.push_back(std::move(heap_handle_p));
	}
	swizzled = false;
	heap_blocks.clear();
}

SortedBlock::SortedBlock(BufferManager &buffer_manager, GlobalSortState &state)
    : buffer_manager(buffer_manager), state(state), sort_layout(state.sort_layout),
      payload_layout(state.payload_layout) {
	blob_sorting_data = make_unique<SortedData>(SortedDataType::BLOB, sort_layout.blob_layout, buffer_manager, state);
	payload_data = make_unique<SortedData>(SortedDataType::PAYLOAD, payload_layout, buffer_manager, state);
}

idx_t SortedBlock::Count() const {
	idx_t count = std::accumulate(radix_sorting_data.begin(), radix_sorting_data.end(), 0,
	                              [](idx_t a, const unique_ptr<RowDataBlock> &b) { return a + b->count; });
	if (!sort_layout.all_constant) {
		D_ASSERT(count == blob_sorting_data->Count());
	}
	D_ASSERT(count == payload_data->Count());
	return count;
}

void SortedBlock::InitializeWrite() {
	CreateBlock();
	if (!sort_layout.all_constant) {
		blob_sorting_data->CreateBlock();
	}
	payload_data->CreateBlock();
}

void SortedBlock::CreateBlock() {
	auto capacity = MaxValue(((idx_t)Storage::BLOCK_SIZE + sort_layout.entry_size - 1) / sort_layout.entry_size,
	                         state.block_capacity);
	radix_sorting_data.push_back(make_unique<RowDataBlock>(buffer_manager, capacity, sort_layout.entry_size));
}

void SortedBlock::AppendSortedBlocks(vector<unique_ptr<SortedBlock>> &sorted_blocks) {
	D_ASSERT(Count() == 0);
	for (auto &sb : sorted_blocks) {
		for (auto &radix_block : sb->radix_sorting_data) {
			radix_sorting_data.push_back(std::move(radix_block));
		}
		if (!sort_layout.all_constant) {
			for (auto &blob_block : sb->blob_sorting_data->data_blocks) {
				blob_sorting_data->data_blocks.push_back(std::move(blob_block));
			}
			for (auto &heap_block : sb->blob_sorting_data->heap_blocks) {
				blob_sorting_data->heap_blocks.push_back(std::move(heap_block));
			}
		}
		for (auto &payload_data_block : sb->payload_data->data_blocks) {
			payload_data->data_blocks.push_back(std::move(payload_data_block));
		}
		if (!payload_data->layout.AllConstant()) {
			for (auto &payload_heap_block : sb->payload_data->heap_blocks) {
				payload_data->heap_blocks.push_back(std::move(payload_heap_block));
			}
		}
	}
}

void SortedBlock::GlobalToLocalIndex(const idx_t &global_idx, idx_t &local_block_index, idx_t &local_entry_index) {
	if (global_idx == Count()) {
		local_block_index = radix_sorting_data.size() - 1;
		local_entry_index = radix_sorting_data.back()->count;
		return;
	}
	D_ASSERT(global_idx < Count());
	local_entry_index = global_idx;
	for (local_block_index = 0; local_block_index < radix_sorting_data.size(); local_block_index++) {
		const idx_t &block_count = radix_sorting_data[local_block_index]->count;
		if (local_entry_index >= block_count) {
			local_entry_index -= block_count;
		} else {
			break;
		}
	}
	D_ASSERT(local_entry_index < radix_sorting_data[local_block_index]->count);
}

unique_ptr<SortedBlock> SortedBlock::CreateSlice(const idx_t start, const idx_t end, idx_t &entry_idx) {
	// Identify blocks/entry indices of this slice
	idx_t start_block_index;
	idx_t start_entry_index;
	GlobalToLocalIndex(start, start_block_index, start_entry_index);
	idx_t end_block_index;
	idx_t end_entry_index;
	GlobalToLocalIndex(end, end_block_index, end_entry_index);
	// Add the corresponding blocks to the result
	auto result = make_unique<SortedBlock>(buffer_manager, state);
	for (idx_t i = start_block_index; i <= end_block_index; i++) {
		result->radix_sorting_data.push_back(radix_sorting_data[i]->Copy());
	}
	// Reset all blocks that come before block with idx = start_block_idx (slice holds new reference)
	for (idx_t i = 0; i < start_block_index; i++) {
		radix_sorting_data[i]->block = nullptr;
	}
	// Use start and end entry indices to set the boundaries
	entry_idx = start_entry_index;
	D_ASSERT(end_entry_index <= result->radix_sorting_data.back()->count);
	result->radix_sorting_data.back()->count = end_entry_index;
	// Same for the var size sorting data
	if (!sort_layout.all_constant) {
		result->blob_sorting_data = blob_sorting_data->CreateSlice(start_block_index, end_block_index, end_entry_index);
	}
	// And the payload data
	result->payload_data = payload_data->CreateSlice(start_block_index, end_block_index, end_entry_index);
	return result;
}

idx_t SortedBlock::HeapSize() const {
	idx_t result = 0;
	if (!sort_layout.all_constant) {
		for (auto &block : blob_sorting_data->heap_blocks) {
			result += block->capacity;
		}
	}
	if (!payload_layout.AllConstant()) {
		for (auto &block : payload_data->heap_blocks) {
			result += block->capacity;
		}
	}
	return result;
}

idx_t SortedBlock::SizeInBytes() const {
	idx_t bytes = 0;
	for (idx_t i = 0; i < radix_sorting_data.size(); i++) {
		bytes += radix_sorting_data[i]->capacity * sort_layout.entry_size;
		if (!sort_layout.all_constant) {
			bytes += blob_sorting_data->data_blocks[i]->capacity * sort_layout.blob_layout.GetRowWidth();
			bytes += blob_sorting_data->heap_blocks[i]->capacity;
		}
		bytes += payload_data->data_blocks[i]->capacity * payload_layout.GetRowWidth();
		if (!payload_layout.AllConstant()) {
			bytes += payload_data->heap_blocks[i]->capacity;
		}
	}
	return bytes;
}

SBScanState::SBScanState(BufferManager &buffer_manager, GlobalSortState &state)
    : buffer_manager(buffer_manager), sort_layout(state.sort_layout), state(state), block_idx(0), entry_idx(0) {
}

void SBScanState::PinRadix(idx_t block_idx_to) {
	auto &radix_sorting_data = sb->radix_sorting_data;
	D_ASSERT(block_idx_to < radix_sorting_data.size());
	auto &block = radix_sorting_data[block_idx_to];
	if (!radix_handle.IsValid() || radix_handle.GetBlockHandle() != block->block) {
		radix_handle = buffer_manager.Pin(block->block);
	}
}

void SBScanState::PinData(SortedData &sd) {
	D_ASSERT(block_idx < sd.data_blocks.size());
	auto &data_handle = sd.type == SortedDataType::BLOB ? blob_sorting_data_handle : payload_data_handle;
	auto &heap_handle = sd.type == SortedDataType::BLOB ? blob_sorting_heap_handle : payload_heap_handle;

	auto &data_block = sd.data_blocks[block_idx];
	if (!data_handle.IsValid() || data_handle.GetBlockHandle() != data_block->block) {
		data_handle = buffer_manager.Pin(data_block->block);
	}
	if (sd.layout.AllConstant() || !state.external) {
		return;
	}
	auto &heap_block = sd.heap_blocks[block_idx];
	if (!heap_handle.IsValid() || heap_handle.GetBlockHandle() != heap_block->block) {
		heap_handle = buffer_manager.Pin(heap_block->block);
	}
}

data_ptr_t SBScanState::RadixPtr() const {
	return radix_handle.Ptr() + entry_idx * sort_layout.entry_size;
}

data_ptr_t SBScanState::DataPtr(SortedData &sd) const {
	auto &data_handle = sd.type == SortedDataType::BLOB ? blob_sorting_data_handle : payload_data_handle;
	D_ASSERT(sd.data_blocks[block_idx]->block->Readers() != 0 &&
	         data_handle.GetBlockHandle() == sd.data_blocks[block_idx]->block);
	return data_handle.Ptr() + entry_idx * sd.layout.GetRowWidth();
}

data_ptr_t SBScanState::HeapPtr(SortedData &sd) const {
	return BaseHeapPtr(sd) + Load<idx_t>(DataPtr(sd) + sd.layout.GetHeapOffset());
}

data_ptr_t SBScanState::BaseHeapPtr(SortedData &sd) const {
	auto &heap_handle = sd.type == SortedDataType::BLOB ? blob_sorting_heap_handle : payload_heap_handle;
	D_ASSERT(!sd.layout.AllConstant() && state.external);
	D_ASSERT(sd.heap_blocks[block_idx]->block->Readers() != 0 &&
	         heap_handle.GetBlockHandle() == sd.heap_blocks[block_idx]->block);
	return heap_handle.Ptr();
}

idx_t SBScanState::Remaining() const {
	const auto &blocks = sb->radix_sorting_data;
	idx_t remaining = 0;
	if (block_idx < blocks.size()) {
		remaining += blocks[block_idx]->count - entry_idx;
		for (idx_t i = block_idx + 1; i < blocks.size(); i++) {
			remaining += blocks[i]->count;
		}
	}
	return remaining;
}

void SBScanState::SetIndices(idx_t block_idx_to, idx_t entry_idx_to) {
	block_idx = block_idx_to;
	entry_idx = entry_idx_to;
}

PayloadScanner::PayloadScanner(SortedData &sorted_data, GlobalSortState &global_sort_state, bool flush_p) {
	auto count = sorted_data.Count();
	auto &layout = sorted_data.layout;

	// Create collections to put the data into so we can use RowDataCollectionScanner
	rows = make_unique<RowDataCollection>(global_sort_state.buffer_manager, (idx_t)Storage::BLOCK_SIZE, 1);
	rows->count = count;

	heap = make_unique<RowDataCollection>(global_sort_state.buffer_manager, (idx_t)Storage::BLOCK_SIZE, 1);
	if (!sorted_data.layout.AllConstant()) {
		heap->count = count;
	}

	if (flush_p) {
		// If we are flushing, we can just move the data
		rows->blocks = std::move(sorted_data.data_blocks);
		if (!layout.AllConstant()) {
			heap->blocks = std::move(sorted_data.heap_blocks);
		}
	} else {
		// Not flushing, create references to the blocks
		for (auto &block : sorted_data.data_blocks) {
			rows->blocks.emplace_back(block->Copy());
		}
		if (!layout.AllConstant()) {
			for (auto &block : sorted_data.heap_blocks) {
				heap->blocks.emplace_back(block->Copy());
			}
		}
	}

	scanner = make_unique<RowDataCollectionScanner>(*rows, *heap, layout, global_sort_state.external, flush_p);
}

PayloadScanner::PayloadScanner(GlobalSortState &global_sort_state, bool flush_p)
    : PayloadScanner(*global_sort_state.sorted_blocks[0]->payload_data, global_sort_state, flush_p) {
}

PayloadScanner::PayloadScanner(GlobalSortState &global_sort_state, idx_t block_idx, bool flush_p) {
	auto &sorted_data = *global_sort_state.sorted_blocks[0]->payload_data;
	auto count = sorted_data.data_blocks[block_idx]->count;
	auto &layout = sorted_data.layout;

	// Create collections to put the data into so we can use RowDataCollectionScanner
	rows = make_unique<RowDataCollection>(global_sort_state.buffer_manager, (idx_t)Storage::BLOCK_SIZE, 1);
	if (flush_p) {
		rows->blocks.emplace_back(std::move(sorted_data.data_blocks[block_idx]));
	} else {
		rows->blocks.emplace_back(sorted_data.data_blocks[block_idx]->Copy());
	}
	rows->count = count;

	heap = make_unique<RowDataCollection>(global_sort_state.buffer_manager, (idx_t)Storage::BLOCK_SIZE, 1);
	if (!sorted_data.layout.AllConstant() && sorted_data.swizzled) {
		if (flush_p) {
			heap->blocks.emplace_back(std::move(sorted_data.heap_blocks[block_idx]));
		} else {
			heap->blocks.emplace_back(sorted_data.heap_blocks[block_idx]->Copy());
		}
		heap->count = count;
	}

	scanner = make_unique<RowDataCollectionScanner>(*rows, *heap, layout, global_sort_state.external, flush_p);
}

void PayloadScanner::Scan(DataChunk &chunk) {
	scanner->Scan(chunk);
}

int SBIterator::ComparisonValue(ExpressionType comparison) {
	switch (comparison) {
	case ExpressionType::COMPARE_LESSTHAN:
	case ExpressionType::COMPARE_GREATERTHAN:
		return -1;
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return 0;
	default:
		throw InternalException("Unimplemented comparison type for IEJoin!");
	}
}

SBIterator::SBIterator(GlobalSortState &gss, ExpressionType comparison, idx_t entry_idx_p)
    : sort_layout(gss.sort_layout), block_count(gss.sorted_blocks[0]->radix_sorting_data.size()),
      block_capacity(gss.block_capacity), cmp_size(sort_layout.comparison_size), entry_size(sort_layout.entry_size),
      all_constant(sort_layout.all_constant), external(gss.external), cmp(ComparisonValue(comparison)),
      scan(gss.buffer_manager, gss), block_ptr(nullptr), entry_ptr(nullptr) {

	scan.sb = gss.sorted_blocks[0].get();
	scan.block_idx = block_count;
	SetIndex(entry_idx_p);
}

} // namespace duckdb






#include <algorithm>
#include <cctype>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdarg.h>
#include <string.h>

namespace duckdb {

bool StringUtil::Contains(const string &haystack, const string &needle) {
	return (haystack.find(needle) != string::npos);
}

void StringUtil::LTrim(string &str) {
	auto it = str.begin();
	while (it != str.end() && CharacterIsSpace(*it)) {
		it++;
	}
	str.erase(str.begin(), it);
}

// Remove trailing ' ', '\f', '\n', '\r', '\t', '\v'
void StringUtil::RTrim(string &str) {
	str.erase(find_if(str.rbegin(), str.rend(), [](int ch) { return ch > 0 && !CharacterIsSpace(ch); }).base(),
	          str.end());
}

void StringUtil::RTrim(string &str, const string &chars_to_trim) {
	str.erase(find_if(str.rbegin(), str.rend(),
	                  [&chars_to_trim](int ch) { return ch > 0 && chars_to_trim.find(ch) == string::npos; })
	              .base(),
	          str.end());
}

void StringUtil::Trim(string &str) {
	StringUtil::LTrim(str);
	StringUtil::RTrim(str);
}

bool StringUtil::StartsWith(string str, string prefix) {
	if (prefix.size() > str.size()) {
		return false;
	}
	return equal(prefix.begin(), prefix.end(), str.begin());
}

bool StringUtil::EndsWith(const string &str, const string &suffix) {
	if (suffix.size() > str.size()) {
		return false;
	}
	return equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

string StringUtil::Repeat(const string &str, idx_t n) {
	std::ostringstream os;
	for (idx_t i = 0; i < n; i++) {
		os << str;
	}
	return (os.str());
}

vector<string> StringUtil::Split(const string &str, char delimiter) {
	std::stringstream ss(str);
	vector<string> lines;
	string temp;
	while (getline(ss, temp, delimiter)) {
		lines.push_back(temp);
	}
	return (lines);
}

namespace string_util_internal {

inline void SkipSpaces(const string &str, idx_t &index) {
	while (index < str.size() && std::isspace(str[index])) {
		index++;
	}
}

inline void ConsumeLetter(const string &str, idx_t &index, char expected) {
	if (index >= str.size() || str[index] != expected) {
		throw ParserException("Invalid quoted list: %s", str);
	}

	index++;
}

template <typename F>
inline void TakeWhile(const string &str, idx_t &index, const F &cond, string &taker) {
	while (index < str.size() && cond(str[index])) {
		taker.push_back(str[index]);
		index++;
	}
}

inline string TakePossiblyQuotedItem(const string &str, idx_t &index, char delimiter, char quote) {
	string entry;

	if (str[index] == quote) {
		index++;
		TakeWhile(
		    str, index, [quote](char c) { return c != quote; }, entry);
		ConsumeLetter(str, index, quote);
	} else {
		TakeWhile(
		    str, index, [delimiter, quote](char c) { return c != delimiter && c != quote && !std::isspace(c); }, entry);
	}

	return entry;
}

} // namespace string_util_internal

vector<string> StringUtil::SplitWithQuote(const string &str, char delimiter, char quote) {
	vector<string> entries;
	idx_t i = 0;

	string_util_internal::SkipSpaces(str, i);
	while (i < str.size()) {
		if (!entries.empty()) {
			string_util_internal::ConsumeLetter(str, i, delimiter);
		}

		entries.emplace_back(string_util_internal::TakePossiblyQuotedItem(str, i, delimiter, quote));
		string_util_internal::SkipSpaces(str, i);
	}

	return entries;
}

string StringUtil::Join(const vector<string> &input, const string &separator) {
	return StringUtil::Join(input, input.size(), separator, [](const string &s) { return s; });
}

string StringUtil::BytesToHumanReadableString(idx_t bytes) {
	string db_size;
	auto kilobytes = bytes / 1000;
	auto megabytes = kilobytes / 1000;
	kilobytes -= megabytes * 1000;
	auto gigabytes = megabytes / 1000;
	megabytes -= gigabytes * 1000;
	auto terabytes = gigabytes / 1000;
	gigabytes -= terabytes * 1000;
	if (terabytes > 0) {
		return to_string(terabytes) + "." + to_string(gigabytes / 100) + "TB";
	} else if (gigabytes > 0) {
		return to_string(gigabytes) + "." + to_string(megabytes / 100) + "GB";
	} else if (megabytes > 0) {
		return to_string(megabytes) + "." + to_string(kilobytes / 100) + "MB";
	} else if (kilobytes > 0) {
		return to_string(kilobytes) + "KB";
	} else {
		return to_string(bytes) + " bytes";
	}
}

string StringUtil::Upper(const string &str) {
	string copy(str);
	transform(copy.begin(), copy.end(), copy.begin(), [](unsigned char c) { return std::toupper(c); });
	return (copy);
}

string StringUtil::Lower(const string &str) {
	string copy(str);
	transform(copy.begin(), copy.end(), copy.begin(), [](unsigned char c) { return std::tolower(c); });
	return (copy);
}

bool StringUtil::CIEquals(const string &l1, const string &l2) {
	return StringUtil::Lower(l1) == StringUtil::Lower(l2);
}

vector<string> StringUtil::Split(const string &input, const string &split) {
	vector<string> splits;

	idx_t last = 0;
	idx_t input_len = input.size();
	idx_t split_len = split.size();
	while (last <= input_len) {
		idx_t next = input.find(split, last);
		if (next == string::npos) {
			next = input_len;
		}

		// Push the substring [last, next) on to splits
		string substr = input.substr(last, next - last);
		if (substr.empty() == false) {
			splits.push_back(substr);
		}
		last = next + split_len;
	}
	return splits;
}

string StringUtil::Replace(string source, const string &from, const string &to) {
	if (from.empty()) {
		throw InternalException("Invalid argument to StringUtil::Replace - empty FROM");
	}
	idx_t start_pos = 0;
	while ((start_pos = source.find(from, start_pos)) != string::npos) {
		source.replace(start_pos, from.length(), to);
		start_pos += to.length(); // In case 'to' contains 'from', like
		                          // replacing 'x' with 'yx'
	}
	return source;
}

vector<string> StringUtil::TopNStrings(vector<pair<string, idx_t>> scores, idx_t n, idx_t threshold) {
	if (scores.empty()) {
		return vector<string>();
	}
	sort(scores.begin(), scores.end(), [](const pair<string, idx_t> &a, const pair<string, idx_t> &b) -> bool {
		return a.second < b.second || (a.second == b.second && a.first.size() < b.first.size());
	});
	vector<string> result;
	result.push_back(scores[0].first);
	for (idx_t i = 1; i < MinValue<idx_t>(scores.size(), n); i++) {
		if (scores[i].second > threshold) {
			break;
		}
		result.push_back(scores[i].first);
	}
	return result;
}

struct LevenshteinArray {
	LevenshteinArray(idx_t len1, idx_t len2) : len1(len1) {
		dist = unique_ptr<idx_t[]>(new idx_t[len1 * len2]);
	}

	idx_t &Score(idx_t i, idx_t j) {
		return dist[GetIndex(i, j)];
	}

private:
	idx_t len1;
	unique_ptr<idx_t[]> dist;

	idx_t GetIndex(idx_t i, idx_t j) {
		return j * len1 + i;
	}
};

// adapted from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#C++
idx_t StringUtil::LevenshteinDistance(const string &s1_p, const string &s2_p) {
	auto s1 = StringUtil::Lower(s1_p);
	auto s2 = StringUtil::Lower(s2_p);
	idx_t len1 = s1.size();
	idx_t len2 = s2.size();
	if (len1 == 0) {
		return len2;
	}
	if (len2 == 0) {
		return len1;
	}
	LevenshteinArray array(len1 + 1, len2 + 1);
	array.Score(0, 0) = 0;
	for (idx_t i = 0; i <= len1; i++) {
		array.Score(i, 0) = i;
	}
	for (idx_t j = 0; j <= len2; j++) {
		array.Score(0, j) = j;
	}
	for (idx_t i = 1; i <= len1; i++) {
		for (idx_t j = 1; j <= len2; j++) {
			// d[i][j] = std::min({ d[i - 1][j] + 1,
			//                      d[i][j - 1] + 1,
			//                      d[i - 1][j - 1] + (s1[i - 1] == s2[j - 1] ? 0 : 1) });
			int equal = s1[i - 1] == s2[j - 1] ? 0 : 1;
			idx_t adjacent_score1 = array.Score(i - 1, j) + 1;
			idx_t adjacent_score2 = array.Score(i, j - 1) + 1;
			idx_t adjacent_score3 = array.Score(i - 1, j - 1) + equal;

			idx_t t = MinValue<idx_t>(adjacent_score1, adjacent_score2);
			array.Score(i, j) = MinValue<idx_t>(t, adjacent_score3);
		}
	}
	return array.Score(len1, len2);
}

vector<string> StringUtil::TopNLevenshtein(const vector<string> &strings, const string &target, idx_t n,
                                           idx_t threshold) {
	vector<pair<string, idx_t>> scores;
	scores.reserve(strings.size());
	for (auto &str : strings) {
		if (target.size() < str.size()) {
			scores.emplace_back(str, LevenshteinDistance(str.substr(0, target.size()), target));
		} else {
			scores.emplace_back(str, LevenshteinDistance(str, target));
		}
	}
	return TopNStrings(scores, n, threshold);
}

string StringUtil::CandidatesMessage(const vector<string> &candidates, const string &candidate) {
	string result_str;
	if (!candidates.empty()) {
		result_str = "\n" + candidate + ": ";
		for (idx_t i = 0; i < candidates.size(); i++) {
			if (i > 0) {
				result_str += ", ";
			}
			result_str += "\"" + candidates[i] + "\"";
		}
	}
	return result_str;
}

string StringUtil::CandidatesErrorMessage(const vector<string> &strings, const string &target,
                                          const string &message_prefix, idx_t n) {
	auto closest_strings = StringUtil::TopNLevenshtein(strings, target, n);
	return StringUtil::CandidatesMessage(closest_strings, message_prefix);
}

} // namespace duckdb













#include <sstream>

namespace duckdb {

RenderTree::RenderTree(idx_t width_p, idx_t height_p) : width(width_p), height(height_p) {
	nodes = unique_ptr<unique_ptr<RenderTreeNode>[]>(new unique_ptr<RenderTreeNode>[(width + 1) * (height + 1)]);
}

RenderTreeNode *RenderTree::GetNode(idx_t x, idx_t y) {
	if (x >= width || y >= height) {
		return nullptr;
	}
	return nodes[GetPosition(x, y)].get();
}

bool RenderTree::HasNode(idx_t x, idx_t y) {
	if (x >= width || y >= height) {
		return false;
	}
	return nodes[GetPosition(x, y)].get() != nullptr;
}

idx_t RenderTree::GetPosition(idx_t x, idx_t y) {
	return y * width + x;
}

void RenderTree::SetNode(idx_t x, idx_t y, unique_ptr<RenderTreeNode> node) {
	nodes[GetPosition(x, y)] = std::move(node);
}

void TreeRenderer::RenderTopLayer(RenderTree &root, std::ostream &ss, idx_t y) {
	for (idx_t x = 0; x < root.width; x++) {
		if (x * config.NODE_RENDER_WIDTH >= config.MAXIMUM_RENDER_WIDTH) {
			break;
		}
		if (root.HasNode(x, y)) {
			ss << config.LTCORNER;
			ss << StringUtil::Repeat(config.HORIZONTAL, config.NODE_RENDER_WIDTH / 2 - 1);
			if (y == 0) {
				// top level node: no node above this one
				ss << config.HORIZONTAL;
			} else {
				// render connection to node above this one
				ss << config.DMIDDLE;
			}
			ss << StringUtil::Repeat(config.HORIZONTAL, config.NODE_RENDER_WIDTH / 2 - 1);
			ss << config.RTCORNER;
		} else {
			ss << StringUtil::Repeat(" ", config.NODE_RENDER_WIDTH);
		}
	}
	ss << std::endl;
}

void TreeRenderer::RenderBottomLayer(RenderTree &root, std::ostream &ss, idx_t y) {
	for (idx_t x = 0; x <= root.width; x++) {
		if (x * config.NODE_RENDER_WIDTH >= config.MAXIMUM_RENDER_WIDTH) {
			break;
		}
		if (root.HasNode(x, y)) {
			ss << config.LDCORNER;
			ss << StringUtil::Repeat(config.HORIZONTAL, config.NODE_RENDER_WIDTH / 2 - 1);
			if (root.HasNode(x, y + 1)) {
				// node below this one: connect to that one
				ss << config.TMIDDLE;
			} else {
				// no node below this one: end the box
				ss << config.HORIZONTAL;
			}
			ss << StringUtil::Repeat(config.HORIZONTAL, config.NODE_RENDER_WIDTH / 2 - 1);
			ss << config.RDCORNER;
		} else if (root.HasNode(x, y + 1)) {
			ss << StringUtil::Repeat(" ", config.NODE_RENDER_WIDTH / 2);
			ss << config.VERTICAL;
			ss << StringUtil::Repeat(" ", config.NODE_RENDER_WIDTH / 2);
		} else {
			ss << StringUtil::Repeat(" ", config.NODE_RENDER_WIDTH);
		}
	}
	ss << std::endl;
}

string AdjustTextForRendering(string source, idx_t max_render_width) {
	idx_t cpos = 0;
	idx_t render_width = 0;
	vector<pair<idx_t, idx_t>> render_widths;
	while (cpos < source.size()) {
		idx_t char_render_width = Utf8Proc::RenderWidth(source.c_str(), source.size(), cpos);
		cpos = Utf8Proc::NextGraphemeCluster(source.c_str(), source.size(), cpos);
		render_width += char_render_width;
		render_widths.emplace_back(cpos, render_width);
		if (render_width > max_render_width) {
			break;
		}
	}
	if (render_width > max_render_width) {
		// need to find a position to truncate
		for (idx_t pos = render_widths.size(); pos > 0; pos--) {
			if (render_widths[pos - 1].second < max_render_width - 4) {
				return source.substr(0, render_widths[pos - 1].first) + "..." +
				       string(max_render_width - render_widths[pos - 1].second - 3, ' ');
			}
		}
		source = "...";
	}
	// need to pad with spaces
	idx_t total_spaces = max_render_width - render_width;
	idx_t half_spaces = total_spaces / 2;
	idx_t extra_left_space = total_spaces % 2 == 0 ? 0 : 1;
	return string(half_spaces + extra_left_space, ' ') + source + string(half_spaces, ' ');
}

static bool NodeHasMultipleChildren(RenderTree &root, idx_t x, idx_t y) {
	for (; x < root.width && !root.HasNode(x + 1, y); x++) {
		if (root.HasNode(x + 1, y + 1)) {
			return true;
		}
	}
	return false;
}

void TreeRenderer::RenderBoxContent(RenderTree &root, std::ostream &ss, idx_t y) {
	// we first need to figure out how high our boxes are going to be
	vector<vector<string>> extra_info;
	idx_t extra_height = 0;
	extra_info.resize(root.width);
	for (idx_t x = 0; x < root.width; x++) {
		auto node = root.GetNode(x, y);
		if (node) {
			SplitUpExtraInfo(node->extra_text, extra_info[x]);
			if (extra_info[x].size() > extra_height) {
				extra_height = extra_info[x].size();
			}
		}
	}
	extra_height = MinValue<idx_t>(extra_height, config.MAX_EXTRA_LINES);
	idx_t halfway_point = (extra_height + 1) / 2;
	// now we render the actual node
	for (idx_t render_y = 0; render_y <= extra_height; render_y++) {
		for (idx_t x = 0; x < root.width; x++) {
			if (x * config.NODE_RENDER_WIDTH >= config.MAXIMUM_RENDER_WIDTH) {
				break;
			}
			auto node = root.GetNode(x, y);
			if (!node) {
				if (render_y == halfway_point) {
					bool has_child_to_the_right = NodeHasMultipleChildren(root, x, y);
					if (root.HasNode(x, y + 1)) {
						// node right below this one
						ss << StringUtil::Repeat(config.HORIZONTAL, config.NODE_RENDER_WIDTH / 2);
						ss << config.RTCORNER;
						if (has_child_to_the_right) {
							// but we have another child to the right! keep rendering the line
							ss << StringUtil::Repeat(config.HORIZONTAL, config.NODE_RENDER_WIDTH / 2);
						} else {
							// only a child below this one: fill the rest with spaces
							ss << StringUtil::Repeat(" ", config.NODE_RENDER_WIDTH / 2);
						}
					} else if (has_child_to_the_right) {
						// child to the right, but no child right below this one: render a full line
						ss << StringUtil::Repeat(config.HORIZONTAL, config.NODE_RENDER_WIDTH);
					} else {
						// empty spot: render spaces
						ss << StringUtil::Repeat(" ", config.NODE_RENDER_WIDTH);
					}
				} else if (render_y >= halfway_point) {
					if (root.HasNode(x, y + 1)) {
						// we have a node below this empty spot: render a vertical line
						ss << StringUtil::Repeat(" ", config.NODE_RENDER_WIDTH / 2);
						ss << config.VERTICAL;
						ss << StringUtil::Repeat(" ", config.NODE_RENDER_WIDTH / 2);
					} else {
						// empty spot: render spaces
						ss << StringUtil::Repeat(" ", config.NODE_RENDER_WIDTH);
					}
				} else {
					// empty spot: render spaces
					ss << StringUtil::Repeat(" ", config.NODE_RENDER_WIDTH);
				}
			} else {
				ss << config.VERTICAL;
				// figure out what to render
				string render_text;
				if (render_y == 0) {
					render_text = node->name;
				} else {
					if (render_y <= extra_info[x].size()) {
						render_text = extra_info[x][render_y - 1];
					}
				}
				render_text = AdjustTextForRendering(render_text, config.NODE_RENDER_WIDTH - 2);
				ss << render_text;

				if (render_y == halfway_point && NodeHasMultipleChildren(root, x, y)) {
					ss << config.LMIDDLE;
				} else {
					ss << config.VERTICAL;
				}
			}
		}
		ss << std::endl;
	}
}

string TreeRenderer::ToString(const LogicalOperator &op) {
	std::stringstream ss;
	Render(op, ss);
	return ss.str();
}

string TreeRenderer::ToString(const PhysicalOperator &op) {
	std::stringstream ss;
	Render(op, ss);
	return ss.str();
}

string TreeRenderer::ToString(const QueryProfiler::TreeNode &op) {
	std::stringstream ss;
	Render(op, ss);
	return ss.str();
}

string TreeRenderer::ToString(const Pipeline &op) {
	std::stringstream ss;
	Render(op, ss);
	return ss.str();
}

void TreeRenderer::Render(const LogicalOperator &op, std::ostream &ss) {
	auto tree = CreateTree(op);
	ToStream(*tree, ss);
}

void TreeRenderer::Render(const PhysicalOperator &op, std::ostream &ss) {
	auto tree = CreateTree(op);
	ToStream(*tree, ss);
}

void TreeRenderer::Render(const QueryProfiler::TreeNode &op, std::ostream &ss) {
	auto tree = CreateTree(op);
	ToStream(*tree, ss);
}

void TreeRenderer::Render(const Pipeline &op, std::ostream &ss) {
	auto tree = CreateTree(op);
	ToStream(*tree, ss);
}

void TreeRenderer::ToStream(RenderTree &root, std::ostream &ss) {
	while (root.width * config.NODE_RENDER_WIDTH > config.MAXIMUM_RENDER_WIDTH) {
		if (config.NODE_RENDER_WIDTH - 2 < config.MINIMUM_RENDER_WIDTH) {
			break;
		}
		config.NODE_RENDER_WIDTH -= 2;
	}

	for (idx_t y = 0; y < root.height; y++) {
		// start by rendering the top layer
		RenderTopLayer(root, ss, y);
		// now we render the content of the boxes
		RenderBoxContent(root, ss, y);
		// render the bottom layer of each of the boxes
		RenderBottomLayer(root, ss, y);
	}
}

bool TreeRenderer::CanSplitOnThisChar(char l) {
	return (l < '0' || (l > '9' && l < 'A') || (l > 'Z' && l < 'a')) && l != '_';
}

bool TreeRenderer::IsPadding(char l) {
	return l == ' ' || l == '\t' || l == '\n' || l == '\r';
}

string TreeRenderer::RemovePadding(string l) {
	idx_t start = 0, end = l.size();
	while (start < l.size() && IsPadding(l[start])) {
		start++;
	}
	while (end > 0 && IsPadding(l[end - 1])) {
		end--;
	}
	return l.substr(start, end - start);
}

void TreeRenderer::SplitStringBuffer(const string &source, vector<string> &result) {
	D_ASSERT(Utf8Proc::IsValid(source.c_str(), source.size()));
	idx_t max_line_render_size = config.NODE_RENDER_WIDTH - 2;
	// utf8 in prompt, get render width
	idx_t cpos = 0;
	idx_t start_pos = 0;
	idx_t render_width = 0;
	idx_t last_possible_split = 0;
	while (cpos < source.size()) {
		// check if we can split on this character
		if (CanSplitOnThisChar(source[cpos])) {
			last_possible_split = cpos;
		}
		size_t char_render_width = Utf8Proc::RenderWidth(source.c_str(), source.size(), cpos);
		idx_t next_cpos = Utf8Proc::NextGraphemeCluster(source.c_str(), source.size(), cpos);
		if (render_width + char_render_width > max_line_render_size) {
			if (last_possible_split <= start_pos + 8) {
				last_possible_split = cpos;
			}
			result.push_back(source.substr(start_pos, last_possible_split - start_pos));
			start_pos = last_possible_split;
			cpos = last_possible_split;
			render_width = 0;
		}
		cpos = next_cpos;
		render_width += char_render_width;
	}
	if (source.size() > start_pos) {
		result.push_back(source.substr(start_pos, source.size() - start_pos));
	}
}

void TreeRenderer::SplitUpExtraInfo(const string &extra_info, vector<string> &result) {
	if (extra_info.empty()) {
		return;
	}
	if (!Utf8Proc::IsValid(extra_info.c_str(), extra_info.size())) {
		return;
	}
	auto splits = StringUtil::Split(extra_info, "\n");
	if (!splits.empty() && splits[0] != "[INFOSEPARATOR]") {
		result.push_back(ExtraInfoSeparator());
	}
	for (auto &split : splits) {
		if (split == "[INFOSEPARATOR]") {
			result.push_back(ExtraInfoSeparator());
			continue;
		}
		string str = RemovePadding(split);
		if (str.empty()) {
			continue;
		}
		SplitStringBuffer(str, result);
	}
}

string TreeRenderer::ExtraInfoSeparator() {
	return StringUtil::Repeat(string(config.HORIZONTAL) + " ", (config.NODE_RENDER_WIDTH - 7) / 2);
}

unique_ptr<RenderTreeNode> TreeRenderer::CreateRenderNode(string name, string extra_info) {
	auto result = make_unique<RenderTreeNode>();
	result->name = std::move(name);
	result->extra_text = std::move(extra_info);
	return result;
}

class TreeChildrenIterator {
public:
	template <class T>
	static bool HasChildren(const T &op) {
		return !op.children.empty();
	}
	template <class T>
	static void Iterate(const T &op, const std::function<void(const T &child)> &callback) {
		for (auto &child : op.children) {
			callback(*child);
		}
	}
};

template <>
bool TreeChildrenIterator::HasChildren(const PhysicalOperator &op) {
	switch (op.type) {
	case PhysicalOperatorType::DELIM_JOIN:
	case PhysicalOperatorType::POSITIONAL_SCAN:
		return true;
	default:
		return !op.children.empty();
	}
}
template <>
void TreeChildrenIterator::Iterate(const PhysicalOperator &op,
                                   const std::function<void(const PhysicalOperator &child)> &callback) {
	for (auto &child : op.children) {
		callback(*child);
	}
	if (op.type == PhysicalOperatorType::DELIM_JOIN) {
		auto &delim = (PhysicalDelimJoin &)op;
		callback(*delim.join);
	} else if ((op.type == PhysicalOperatorType::POSITIONAL_SCAN)) {
		auto &pscan = (PhysicalPositionalScan &)op;
		for (auto &table : pscan.child_tables) {
			callback(*table);
		}
	}
}

struct PipelineRenderNode {
	explicit PipelineRenderNode(PhysicalOperator &op) : op(op) {
	}

	PhysicalOperator &op;
	unique_ptr<PipelineRenderNode> child;
};

template <>
bool TreeChildrenIterator::HasChildren(const PipelineRenderNode &op) {
	return op.child.get();
}

template <>
void TreeChildrenIterator::Iterate(const PipelineRenderNode &op,
                                   const std::function<void(const PipelineRenderNode &child)> &callback) {
	if (op.child) {
		callback(*op.child);
	}
}

template <class T>
static void GetTreeWidthHeight(const T &op, idx_t &width, idx_t &height) {
	if (!TreeChildrenIterator::HasChildren(op)) {
		width = 1;
		height = 1;
		return;
	}
	width = 0;
	height = 0;

	TreeChildrenIterator::Iterate<T>(op, [&](const T &child) {
		idx_t child_width, child_height;
		GetTreeWidthHeight<T>(child, child_width, child_height);
		width += child_width;
		height = MaxValue<idx_t>(height, child_height);
	});
	height++;
}

template <class T>
idx_t TreeRenderer::CreateRenderTreeRecursive(RenderTree &result, const T &op, idx_t x, idx_t y) {
	auto node = TreeRenderer::CreateNode(op);
	result.SetNode(x, y, std::move(node));

	if (!TreeChildrenIterator::HasChildren(op)) {
		return 1;
	}
	idx_t width = 0;
	// render the children of this node
	TreeChildrenIterator::Iterate<T>(
	    op, [&](const T &child) { width += CreateRenderTreeRecursive<T>(result, child, x + width, y + 1); });
	return width;
}

template <class T>
unique_ptr<RenderTree> TreeRenderer::CreateRenderTree(const T &op) {
	idx_t width, height;
	GetTreeWidthHeight<T>(op, width, height);

	auto result = make_unique<RenderTree>(width, height);

	// now fill in the tree
	CreateRenderTreeRecursive<T>(*result, op, 0, 0);
	return result;
}

unique_ptr<RenderTreeNode> TreeRenderer::CreateNode(const LogicalOperator &op) {
	return CreateRenderNode(op.GetName(), op.ParamsToString());
}

unique_ptr<RenderTreeNode> TreeRenderer::CreateNode(const PhysicalOperator &op) {
	return CreateRenderNode(op.GetName(), op.ParamsToString());
}

unique_ptr<RenderTreeNode> TreeRenderer::CreateNode(const PipelineRenderNode &op) {
	return CreateNode(op.op);
}

string TreeRenderer::ExtractExpressionsRecursive(ExpressionInfo &state) {
	string result = "\n[INFOSEPARATOR]";
	result += "\n" + state.function_name;
	result += "\n" + StringUtil::Format("%.9f", double(state.function_time));
	if (state.children.empty()) {
		return result;
	}
	// render the children of this node
	for (auto &child : state.children) {
		result += ExtractExpressionsRecursive(*child);
	}
	return result;
}

unique_ptr<RenderTreeNode> TreeRenderer::CreateNode(const QueryProfiler::TreeNode &op) {
	auto result = TreeRenderer::CreateRenderNode(op.name, op.extra_info);
	result->extra_text += "\n[INFOSEPARATOR]";
	result->extra_text += "\n" + to_string(op.info.elements);
	string timing = StringUtil::Format("%.2f", op.info.time);
	result->extra_text += "\n(" + timing + "s)";
	if (config.detailed) {
		for (auto &info : op.info.executors_info) {
			if (!info) {
				continue;
			}
			for (auto &executor_info : info->roots) {
				string sample_count = to_string(executor_info->sample_count);
				result->extra_text += "\n[INFOSEPARATOR]";
				result->extra_text += "\nsample_count: " + sample_count;
				string sample_tuples_count = to_string(executor_info->sample_tuples_count);
				result->extra_text += "\n[INFOSEPARATOR]";
				result->extra_text += "\nsample_tuples_count: " + sample_tuples_count;
				string total_count = to_string(executor_info->total_count);
				result->extra_text += "\n[INFOSEPARATOR]";
				result->extra_text += "\ntotal_count: " + total_count;
				for (auto &state : executor_info->root->children) {
					result->extra_text += ExtractExpressionsRecursive(*state);
				}
			}
		}
	}
	return result;
}

unique_ptr<RenderTree> TreeRenderer::CreateTree(const LogicalOperator &op) {
	return CreateRenderTree<LogicalOperator>(op);
}

unique_ptr<RenderTree> TreeRenderer::CreateTree(const PhysicalOperator &op) {
	return CreateRenderTree<PhysicalOperator>(op);
}

unique_ptr<RenderTree> TreeRenderer::CreateTree(const QueryProfiler::TreeNode &op) {
	return CreateRenderTree<QueryProfiler::TreeNode>(op);
}

unique_ptr<RenderTree> TreeRenderer::CreateTree(const Pipeline &op) {
	auto operators = op.GetOperators();
	D_ASSERT(!operators.empty());
	unique_ptr<PipelineRenderNode> node;
	for (auto &op : operators) {
		auto new_node = make_unique<PipelineRenderNode>(*op);
		new_node->child = std::move(node);
		node = std::move(new_node);
	}
	return CreateRenderTree<PipelineRenderNode>(*node);
}

} // namespace duckdb




namespace duckdb {

BatchedDataCollection::BatchedDataCollection(vector<LogicalType> types_p) : types(std::move(types_p)) {
}

void BatchedDataCollection::Append(DataChunk &input, idx_t batch_index) {
	D_ASSERT(batch_index != DConstants::INVALID_INDEX);
	ColumnDataCollection *collection;
	if (last_collection.collection && last_collection.batch_index == batch_index) {
		// we are inserting into the same collection as before: use it directly
		collection = last_collection.collection;
	} else {
		// new collection: check if there is already an entry
		D_ASSERT(data.find(batch_index) == data.end());
		unique_ptr<ColumnDataCollection> new_collection;
		if (last_collection.collection) {
			new_collection = make_unique<ColumnDataCollection>(*last_collection.collection);
		} else {
			new_collection = make_unique<ColumnDataCollection>(Allocator::DefaultAllocator(), types);
		}
		last_collection.collection = new_collection.get();
		last_collection.batch_index = batch_index;
		new_collection->InitializeAppend(last_collection.append_state);
		collection = new_collection.get();
		data.insert(make_pair(batch_index, std::move(new_collection)));
	}
	collection->Append(last_collection.append_state, input);
}

void BatchedDataCollection::Merge(BatchedDataCollection &other) {
	for (auto &entry : other.data) {
		if (data.find(entry.first) != data.end()) {
			throw InternalException(
			    "BatchedDataCollection::Merge error - batch index %d is present in both collections. This occurs when "
			    "batch indexes are not uniquely distributed over threads",
			    entry.first);
		}
		data[entry.first] = std::move(entry.second);
	}
	other.data.clear();
}

void BatchedDataCollection::InitializeScan(BatchedChunkScanState &state) {
	state.iterator = data.begin();
	if (state.iterator == data.end()) {
		return;
	}
	state.iterator->second->InitializeScan(state.scan_state);
}

void BatchedDataCollection::Scan(BatchedChunkScanState &state, DataChunk &output) {
	while (state.iterator != data.end()) {
		// check if there is a chunk remaining in this collection
		auto collection = state.iterator->second.get();
		collection->Scan(state.scan_state, output);
		if (output.size() > 0) {
			return;
		}
		// there isn't! move to the next collection
		state.iterator++;
		if (state.iterator == data.end()) {
			return;
		}
		state.iterator->second->InitializeScan(state.scan_state);
	}
}

unique_ptr<ColumnDataCollection> BatchedDataCollection::FetchCollection() {
	unique_ptr<ColumnDataCollection> result;
	for (auto &entry : data) {
		if (!result) {
			result = std::move(entry.second);
		} else {
			result->Combine(*entry.second);
		}
	}
	data.clear();
	if (!result) {
		// empty result
		return make_unique<ColumnDataCollection>(Allocator::DefaultAllocator(), types);
	}
	return result;
}

string BatchedDataCollection::ToString() const {
	string result;
	result += "Batched Data Collection\n";
	for (auto &entry : data) {
		result += "Batch Index - " + to_string(entry.first) + "\n";
		result += entry.second->ToString() + "\n\n";
	}
	return result;
}

void BatchedDataCollection::Print() const {
	Printer::Print(ToString());
}

} // namespace duckdb




namespace duckdb {

void Bit::SetEmptyBitString(string_t &target, string_t &input) {
	char *res_buf = target.GetDataWriteable();
	const char *buf = input.GetDataUnsafe();
	memset(res_buf, 0, input.GetSize());
	res_buf[0] = buf[0];
}

idx_t Bit::BitLength(string_t bits) {
	return ((bits.GetSize() - 1) * 8) - GetPadding(bits);
}

idx_t Bit::OctetLength(string_t bits) {
	return bits.GetSize() - 1;
}

idx_t Bit::BitCount(string_t bits) {
	idx_t count = 0;
	const char *buf = bits.GetDataUnsafe();
	for (idx_t byte_idx = 1; byte_idx < OctetLength(bits) + 1; byte_idx++) {
		for (idx_t bit_idx = 0; bit_idx < 8; bit_idx++) {
			count += (buf[byte_idx] & (1 << bit_idx)) ? 1 : 0;
		}
	}
	return count;
}

idx_t Bit::BitPosition(string_t substring, string_t bits) {
	const char *buf = bits.GetDataUnsafe();
	auto len = bits.GetSize();
	auto substr_len = BitLength(substring);
	idx_t substr_idx = 0;

	for (idx_t bit_idx = GetPadding(bits); bit_idx < 8; bit_idx++) {
		idx_t bit = buf[1] & (1 << (7 - bit_idx)) ? 1 : 0;
		if (bit == GetBit(substring, substr_idx)) {
			substr_idx++;
			if (substr_idx == substr_len) {
				return (bit_idx - GetPadding(bits)) - substr_len + 2;
			}
		} else {
			substr_idx = 0;
		}
	}

	for (idx_t byte_idx = 2; byte_idx < len; byte_idx++) {
		for (idx_t bit_idx = 0; bit_idx < 8; bit_idx++) {
			idx_t bit = buf[byte_idx] & (1 << (7 - bit_idx)) ? 1 : 0;
			if (bit == GetBit(substring, substr_idx)) {
				substr_idx++;
				if (substr_idx == substr_len) {
					return (((byte_idx - 1) * 8) + bit_idx - GetPadding(bits)) - substr_len + 2;
				}
			} else {
				substr_idx = 0;
			}
		}
	}
	return 0;
}

void Bit::ToString(string_t bits, char *output) {
	auto data = (const_data_ptr_t)bits.GetDataUnsafe();
	auto len = bits.GetSize();

	idx_t padding = GetPadding(bits);
	idx_t output_idx = 0;
	for (idx_t bit_idx = padding; bit_idx < 8; bit_idx++) {
		output[output_idx++] = data[1] & (1 << (7 - bit_idx)) ? '1' : '0';
	}
	for (idx_t byte_idx = 2; byte_idx < len; byte_idx++) {
		for (idx_t bit_idx = 0; bit_idx < 8; bit_idx++) {
			output[output_idx++] = data[byte_idx] & (1 << (7 - bit_idx)) ? '1' : '0';
		}
	}
}

string Bit::ToString(string_t str) {
	auto len = BitLength(str);
	auto buffer = std::unique_ptr<char[]>(new char[len]);
	ToString(str, buffer.get());
	return string(buffer.get(), len);
}

bool Bit::TryGetBitStringSize(string_t str, idx_t &str_len, string *error_message) {
	auto data = (const_data_ptr_t)str.GetDataUnsafe();
	auto len = str.GetSize();
	str_len = 0;
	for (idx_t i = 0; i < len; i++) {
		if (data[i] == '0' || data[i] == '1') {
			str_len++;
		} else {
			string error = StringUtil::Format("Invalid character encountered in string -> bit conversion: '%s'",
			                                  string((char *)data + i, 1));
			HandleCastError::AssignError(error, error_message);
			return false;
		}
	}
	str_len = str_len % 8 ? (str_len / 8) + 1 : str_len / 8;
	str_len++; // additional first byte to store info on zero padding
	return true;
}

idx_t Bit::GetBitSize(string_t str) {
	string error_message;
	idx_t str_len;
	if (!Bit::TryGetBitStringSize(str, str_len, &error_message)) {
		throw ConversionException(error_message);
	}
	return str_len;
}

void Bit::ToBit(string_t str, data_ptr_t output) {
	auto data = (const_data_ptr_t)str.GetDataUnsafe();
	auto len = str.GetSize();

	char byte = 0;
	idx_t padded_byte = len % 8;
	for (idx_t i = 0; i < padded_byte; i++) {
		byte <<= 1;
		if (data[i] == '1') {
			byte |= 1;
		}
	}
	if (padded_byte != 0) {
		*(output++) = (8 - padded_byte); // the first byte contains the number of padded zeroes
	}
	*(output++) = byte;

	for (idx_t byte_idx = padded_byte; byte_idx < len; byte_idx += 8) {
		byte = 0;
		for (idx_t bit_idx = 0; bit_idx < 8; bit_idx++) {
			byte <<= 1;
			if (data[byte_idx + bit_idx] == '1') {
				byte |= 1;
			}
		}
		*(output++) = byte;
	}
}

string Bit::ToBit(string_t str) {
	auto bit_len = GetBitSize(str);
	auto buffer = std::unique_ptr<char[]>(new char[bit_len]);
	Bit::ToBit(str, (data_ptr_t)buffer.get());
	return string(buffer.get(), bit_len);
}

idx_t Bit::GetBit(string_t bit_string, idx_t n) {
	const char *buf = bit_string.GetDataUnsafe();
	n += GetPadding(bit_string);

	char byte = buf[(n / 8) + 1] >> (7 - (n % 8));
	return (byte & 1 ? 1 : 0);
}

void Bit::SetBit(const string_t &bit_string, idx_t n, idx_t new_value, string_t &result) {
	char *result_buf = result.GetDataWriteable();
	const char *buf = bit_string.GetDataUnsafe();
	n += GetPadding(bit_string);

	memcpy(result_buf, buf, bit_string.GetSize());
	char shift_byte = 1 << (7 - (n % 8));
	if (new_value == 0) {
		shift_byte = ~shift_byte;
		result_buf[(n / 8) + 1] = buf[(n / 8) + 1] & shift_byte;
	} else {
		result_buf[(n / 8) + 1] = buf[(n / 8) + 1] | shift_byte;
	}
}

void Bit::SetBit(string_t &bit_string, idx_t n, idx_t new_value) {
	char *buf = bit_string.GetDataWriteable();
	n += GetPadding(bit_string);

	char shift_byte = 1 << (7 - (n % 8));
	if (new_value == 0) {
		shift_byte = ~shift_byte;
		buf[(n / 8) + 1] &= shift_byte;
	} else {
		buf[(n / 8) + 1] |= shift_byte;
	}
}

inline idx_t Bit::GetPadding(const string_t &bit_string) {
	auto data = (const_data_ptr_t)bit_string.GetDataUnsafe();
	return data[0];
}

// **** BITWISE OPERATORS ****
void Bit::RightShift(const string_t &bit_string, const idx_t &shift, string_t &result) {
	char *res_buf = result.GetDataWriteable();
	const char *buf = bit_string.GetDataUnsafe();
	res_buf[0] = buf[0];
	for (idx_t i = 0; i < Bit::BitLength(result); i++) {
		if (i < shift) {
			Bit::SetBit(result, i, 0);
		} else {
			idx_t bit = Bit::GetBit(bit_string, i - shift);
			Bit::SetBit(result, i, bit);
		}
	}
}

void Bit::LeftShift(const string_t &bit_string, const idx_t &shift, string_t &result) {
	char *res_buf = result.GetDataWriteable();
	const char *buf = bit_string.GetDataUnsafe();
	res_buf[0] = buf[0];
	for (idx_t i = 0; i < Bit::BitLength(bit_string); i++) {
		if (i < (Bit::BitLength(bit_string) - shift)) {
			idx_t bit = Bit::GetBit(bit_string, shift + i);
			Bit::SetBit(result, i, bit);
		} else {
			Bit::SetBit(result, i, 0);
		}
	}
}

void Bit::BitwiseAnd(const string_t &rhs, const string_t &lhs, string_t &result) {
	if (Bit::BitLength(lhs) != Bit::BitLength(rhs)) {
		throw InvalidInputException("Cannot AND bit strings of different sizes");
	}

	char *buf = result.GetDataWriteable();
	const char *r_buf = rhs.GetDataUnsafe();
	const char *l_buf = lhs.GetDataUnsafe();

	buf[0] = l_buf[0];
	for (idx_t i = 1; i < lhs.GetSize(); i++) {
		buf[i] = l_buf[i] & r_buf[i];
	}
}

void Bit::BitwiseOr(const string_t &rhs, const string_t &lhs, string_t &result) {
	if (Bit::BitLength(lhs) != Bit::BitLength(rhs)) {
		throw InvalidInputException("Cannot OR bit strings of different sizes");
	}

	char *buf = result.GetDataWriteable();
	const char *r_buf = rhs.GetDataUnsafe();
	const char *l_buf = lhs.GetDataUnsafe();

	buf[0] = l_buf[0];
	for (idx_t i = 1; i < lhs.GetSize(); i++) {
		buf[i] = l_buf[i] | r_buf[i];
	}
}

void Bit::BitwiseXor(const string_t &rhs, const string_t &lhs, string_t &result) {
	if (Bit::BitLength(lhs) != Bit::BitLength(rhs)) {
		throw InvalidInputException("Cannot XOR bit strings of different sizes");
	}

	char *buf = result.GetDataWriteable();
	const char *r_buf = rhs.GetDataUnsafe();
	const char *l_buf = lhs.GetDataUnsafe();

	buf[0] = l_buf[0];
	for (idx_t i = 1; i < lhs.GetSize(); i++) {
		buf[i] = l_buf[i] ^ r_buf[i];
	}
}

void Bit::BitwiseNot(const string_t &input, string_t &result) {
	char *result_buf = result.GetDataWriteable();
	const char *buf = input.GetDataUnsafe();

	result_buf[0] = buf[0];
	for (idx_t i = 1; i < input.GetSize(); i++) {
		result_buf[i] = ~buf[i];
	}
}
} // namespace duckdb







namespace duckdb {

constexpr const char *Blob::HEX_TABLE;
const int Blob::HEX_MAP[256] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
    -1, -1, -1, -1, -1, -1, -1, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

bool IsRegularCharacter(data_t c) {
	return c >= 32 && c <= 127 && c != '\\' && c != '\'' && c != '"';
}

idx_t Blob::GetStringSize(string_t blob) {
	auto data = (const_data_ptr_t)blob.GetDataUnsafe();
	auto len = blob.GetSize();
	idx_t str_len = 0;
	for (idx_t i = 0; i < len; i++) {
		if (IsRegularCharacter(data[i])) {
			// ascii characters are rendered as-is
			str_len++;
		} else {
			// non-ascii characters are rendered as hexadecimal (e.g. \x00)
			str_len += 4;
		}
	}
	return str_len;
}

void Blob::ToString(string_t blob, char *output) {
	auto data = (const_data_ptr_t)blob.GetDataUnsafe();
	auto len = blob.GetSize();
	idx_t str_idx = 0;
	for (idx_t i = 0; i < len; i++) {
		if (IsRegularCharacter(data[i])) {
			// ascii characters are rendered as-is
			output[str_idx++] = data[i];
		} else {
			auto byte_a = data[i] >> 4;
			auto byte_b = data[i] & 0x0F;
			D_ASSERT(byte_a >= 0 && byte_a < 16);
			D_ASSERT(byte_b >= 0 && byte_b < 16);
			// non-ascii characters are rendered as hexadecimal (e.g. \x00)
			output[str_idx++] = '\\';
			output[str_idx++] = 'x';
			output[str_idx++] = Blob::HEX_TABLE[byte_a];
			output[str_idx++] = Blob::HEX_TABLE[byte_b];
		}
	}
	D_ASSERT(str_idx == GetStringSize(blob));
}

string Blob::ToString(string_t blob) {
	auto str_len = GetStringSize(blob);
	auto buffer = std::unique_ptr<char[]>(new char[str_len]);
	Blob::ToString(blob, buffer.get());
	return string(buffer.get(), str_len);
}

bool Blob::TryGetBlobSize(string_t str, idx_t &str_len, string *error_message) {
	auto data = (const_data_ptr_t)str.GetDataUnsafe();
	auto len = str.GetSize();
	str_len = 0;
	for (idx_t i = 0; i < len; i++) {
		if (data[i] == '\\') {
			if (i + 3 >= len) {
				string error = "Invalid hex escape code encountered in string -> blob conversion: "
				               "unterminated escape code at end of blob";
				HandleCastError::AssignError(error, error_message);
				return false;
			}
			if (data[i + 1] != 'x' || Blob::HEX_MAP[data[i + 2]] < 0 || Blob::HEX_MAP[data[i + 3]] < 0) {
				string error =
				    StringUtil::Format("Invalid hex escape code encountered in string -> blob conversion: %s",
				                       string((char *)data + i, 4));
				HandleCastError::AssignError(error, error_message);
				return false;
			}
			str_len++;
			i += 3;
		} else if (data[i] <= 127) {
			str_len++;
		} else {
			string error = "Invalid byte encountered in STRING -> BLOB conversion. All non-ascii characters "
			               "must be escaped with hex codes (e.g. \\xAA)";
			HandleCastError::AssignError(error, error_message);
			return false;
		}
	}
	return true;
}

idx_t Blob::GetBlobSize(string_t str) {
	string error_message;
	idx_t str_len;
	if (!Blob::TryGetBlobSize(str, str_len, &error_message)) {
		throw ConversionException(error_message);
	}
	return str_len;
}

void Blob::ToBlob(string_t str, data_ptr_t output) {
	auto data = (const_data_ptr_t)str.GetDataUnsafe();
	auto len = str.GetSize();
	idx_t blob_idx = 0;
	for (idx_t i = 0; i < len; i++) {
		if (data[i] == '\\') {
			int byte_a = Blob::HEX_MAP[data[i + 2]];
			int byte_b = Blob::HEX_MAP[data[i + 3]];
			D_ASSERT(i + 3 < len);
			D_ASSERT(byte_a >= 0 && byte_b >= 0);
			D_ASSERT(data[i + 1] == 'x');
			output[blob_idx++] = (byte_a << 4) + byte_b;
			i += 3;
		} else if (data[i] <= 127) {
			output[blob_idx++] = data_t(data[i]);
		} else {
			throw ConversionException("Invalid byte encountered in STRING -> BLOB conversion. All non-ascii characters "
			                          "must be escaped with hex codes (e.g. \\xAA)");
		}
	}
	D_ASSERT(blob_idx == GetBlobSize(str));
}

string Blob::ToBlob(string_t str) {
	auto blob_len = GetBlobSize(str);
	auto buffer = std::unique_ptr<char[]>(new char[blob_len]);
	Blob::ToBlob(str, (data_ptr_t)buffer.get());
	return string(buffer.get(), blob_len);
}

// base64 functions are adapted from https://gist.github.com/tomykaira/f0fd86b6c73063283afe550bc5d77594
idx_t Blob::ToBase64Size(string_t blob) {
	// every 4 characters in base64 encode 3 bytes, plus (potential) padding at the end
	auto input_size = blob.GetSize();
	return ((input_size + 2) / 3) * 4;
}

void Blob::ToBase64(string_t blob, char *output) {
	auto input_data = (const_data_ptr_t)blob.GetDataUnsafe();
	auto input_size = blob.GetSize();
	idx_t out_idx = 0;
	idx_t i;
	// convert the bulk of the string to base64
	// this happens in steps of 3 bytes -> 4 output bytes
	for (i = 0; i + 2 < input_size; i += 3) {
		output[out_idx++] = Blob::BASE64_MAP[(input_data[i] >> 2) & 0x3F];
		output[out_idx++] = Blob::BASE64_MAP[((input_data[i] & 0x3) << 4) | ((input_data[i + 1] & 0xF0) >> 4)];
		output[out_idx++] = Blob::BASE64_MAP[((input_data[i + 1] & 0xF) << 2) | ((input_data[i + 2] & 0xC0) >> 6)];
		output[out_idx++] = Blob::BASE64_MAP[input_data[i + 2] & 0x3F];
	}

	if (i < input_size) {
		// there are one or two bytes left over: we have to insert padding
		// first write the first 6 bits of the first byte
		output[out_idx++] = Blob::BASE64_MAP[(input_data[i] >> 2) & 0x3F];
		// now check the character count
		if (i == input_size - 1) {
			// single byte left over: convert the remainder of that byte and insert padding
			output[out_idx++] = Blob::BASE64_MAP[((input_data[i] & 0x3) << 4)];
			output[out_idx++] = Blob::BASE64_PADDING;
		} else {
			// two bytes left over: convert the second byte as well
			output[out_idx++] = Blob::BASE64_MAP[((input_data[i] & 0x3) << 4) | ((input_data[i + 1] & 0xF0) >> 4)];
			output[out_idx++] = Blob::BASE64_MAP[((input_data[i + 1] & 0xF) << 2)];
		}
		output[out_idx++] = Blob::BASE64_PADDING;
	}
}

static constexpr int BASE64_DECODING_TABLE[256] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, -1, -1, 63, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    -1, -1, -1, -1, -1, -1, -1, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, -1, -1, -1, -1, -1, -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    45, 46, 47, 48, 49, 50, 51, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

idx_t Blob::FromBase64Size(string_t str) {
	auto input_data = str.GetDataUnsafe();
	auto input_size = str.GetSize();
	if (input_size % 4 != 0) {
		// valid base64 needs to always be cleanly divisible by 4
		throw ConversionException("Could not decode string \"%s\" as base64: length must be a multiple of 4",
		                          str.GetString());
	}
	if (input_size < 4) {
		// empty string
		return 0;
	}
	auto base_size = input_size / 4 * 3;
	// check for padding to figure out the length
	if (input_data[input_size - 2] == Blob::BASE64_PADDING) {
		// two bytes of padding
		return base_size - 2;
	}
	if (input_data[input_size - 1] == Blob::BASE64_PADDING) {
		// one byte of padding
		return base_size - 1;
	}
	// no padding
	return base_size;
}

template <bool ALLOW_PADDING>
uint32_t DecodeBase64Bytes(const string_t &str, const_data_ptr_t input_data, idx_t base_idx) {
	int decoded_bytes[4];
	for (idx_t decode_idx = 0; decode_idx < 4; decode_idx++) {
		if (ALLOW_PADDING && decode_idx >= 2 && input_data[base_idx + decode_idx] == Blob::BASE64_PADDING) {
			// the last two bytes of a base64 string can have padding: in this case we set the byte to 0
			decoded_bytes[decode_idx] = 0;
		} else {
			decoded_bytes[decode_idx] = BASE64_DECODING_TABLE[input_data[base_idx + decode_idx]];
		}
		if (decoded_bytes[decode_idx] < 0) {
			throw ConversionException(
			    "Could not decode string \"%s\" as base64: invalid byte value '%d' at position %d", str.GetString(),
			    input_data[base_idx + decode_idx], base_idx + decode_idx);
		}
	}
	return (decoded_bytes[0] << 3 * 6) + (decoded_bytes[1] << 2 * 6) + (decoded_bytes[2] << 1 * 6) +
	       (decoded_bytes[3] << 0 * 6);
}

void Blob::FromBase64(string_t str, data_ptr_t output, idx_t output_size) {
	D_ASSERT(output_size == FromBase64Size(str));
	auto input_data = (const_data_ptr_t)str.GetDataUnsafe();
	auto input_size = str.GetSize();
	if (input_size == 0) {
		return;
	}
	idx_t out_idx = 0;
	idx_t i = 0;
	for (i = 0; i + 4 < input_size; i += 4) {
		auto combined = DecodeBase64Bytes<false>(str, input_data, i);
		output[out_idx++] = (combined >> 2 * 8) & 0xFF;
		output[out_idx++] = (combined >> 1 * 8) & 0xFF;
		output[out_idx++] = (combined >> 0 * 8) & 0xFF;
	}
	// decode the final four bytes: padding is allowed here
	auto combined = DecodeBase64Bytes<true>(str, input_data, i);
	output[out_idx++] = (combined >> 2 * 8) & 0xFF;
	if (out_idx < output_size) {
		output[out_idx++] = (combined >> 1 * 8) & 0xFF;
	}
	if (out_idx < output_size) {
		output[out_idx++] = (combined >> 0 * 8) & 0xFF;
	}
}

} // namespace duckdb



namespace duckdb {

const int64_t NumericHelper::POWERS_OF_TEN[] {1,
                                              10,
                                              100,
                                              1000,
                                              10000,
                                              100000,
                                              1000000,
                                              10000000,
                                              100000000,
                                              1000000000,
                                              10000000000,
                                              100000000000,
                                              1000000000000,
                                              10000000000000,
                                              100000000000000,
                                              1000000000000000,
                                              10000000000000000,
                                              100000000000000000,
                                              1000000000000000000};

const double NumericHelper::DOUBLE_POWERS_OF_TEN[] {1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,
                                                    1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
                                                    1e20, 1e21, 1e22, 1e23, 1e24, 1e25, 1e26, 1e27, 1e28, 1e29,
                                                    1e30, 1e31, 1e32, 1e33, 1e34, 1e35, 1e36, 1e37, 1e38, 1e39};

template <>
int NumericHelper::UnsignedLength(uint8_t value) {
	int length = 1;
	length += value >= 10;
	length += value >= 100;
	return length;
}

template <>
int NumericHelper::UnsignedLength(uint16_t value) {
	int length = 1;
	length += value >= 10;
	length += value >= 100;
	length += value >= 1000;
	length += value >= 10000;
	return length;
}

template <>
int NumericHelper::UnsignedLength(uint32_t value) {
	if (value >= 10000) {
		int length = 5;
		length += value >= 100000;
		length += value >= 1000000;
		length += value >= 10000000;
		length += value >= 100000000;
		length += value >= 1000000000;
		return length;
	} else {
		int length = 1;
		length += value >= 10;
		length += value >= 100;
		length += value >= 1000;
		return length;
	}
}

template <>
int NumericHelper::UnsignedLength(uint64_t value) {
	if (value >= 10000000000ULL) {
		if (value >= 1000000000000000ULL) {
			int length = 16;
			length += value >= 10000000000000000ULL;
			length += value >= 100000000000000000ULL;
			length += value >= 1000000000000000000ULL;
			length += value >= 10000000000000000000ULL;
			return length;
		} else {
			int length = 11;
			length += value >= 100000000000ULL;
			length += value >= 1000000000000ULL;
			length += value >= 10000000000000ULL;
			length += value >= 100000000000000ULL;
			return length;
		}
	} else {
		if (value >= 100000ULL) {
			int length = 6;
			length += value >= 1000000ULL;
			length += value >= 10000000ULL;
			length += value >= 100000000ULL;
			length += value >= 1000000000ULL;
			return length;
		} else {
			int length = 1;
			length += value >= 10ULL;
			length += value >= 100ULL;
			length += value >= 1000ULL;
			length += value >= 10000ULL;
			return length;
		}
	}
}

template <>
std::string NumericHelper::ToString(hugeint_t value) {
	return Hugeint::ToString(value);
}

} // namespace duckdb










#include <algorithm>
#include <cstring>

namespace duckdb {

ChunkCollection::ChunkCollection(Allocator &allocator) : allocator(allocator), count(0) {
}

ChunkCollection::ChunkCollection(ClientContext &context) : ChunkCollection(Allocator::Get(context)) {
}

void ChunkCollection::Verify() {
#ifdef DEBUG
	for (auto &chunk : chunks) {
		chunk->Verify();
	}
#endif
}

void ChunkCollection::Append(ChunkCollection &other) {
	for (auto &chunk : other.chunks) {
		Append(*chunk);
	}
}

void ChunkCollection::Merge(ChunkCollection &other) {
	if (other.count == 0) {
		return;
	}
	if (count == 0) {
		chunks = std::move(other.chunks);
		types = std::move(other.types);
		count = other.count;
		return;
	}
	unique_ptr<DataChunk> old_back;
	if (!chunks.empty() && chunks.back()->size() != STANDARD_VECTOR_SIZE) {
		old_back = std::move(chunks.back());
		chunks.pop_back();
		count -= old_back->size();
	}
	for (auto &chunk : other.chunks) {
		chunks.push_back(std::move(chunk));
	}
	count += other.count;
	if (old_back) {
		Append(*old_back);
	}
	Verify();
}

void ChunkCollection::Append(DataChunk &new_chunk) {
	if (new_chunk.size() == 0) {
		return;
	}
	new_chunk.Verify();

	// we have to ensure that every chunk in the ChunkCollection is completely
	// filled, otherwise our O(1) lookup in GetValue and SetValue does not work
	// first fill the latest chunk, if it exists
	count += new_chunk.size();

	idx_t remaining_data = new_chunk.size();
	idx_t offset = 0;
	if (chunks.empty()) {
		// first chunk
		types = new_chunk.GetTypes();
	} else {
		// the types of the new chunk should match the types of the previous one
		D_ASSERT(types.size() == new_chunk.ColumnCount());
		auto new_types = new_chunk.GetTypes();
		for (idx_t i = 0; i < types.size(); i++) {
			if (new_types[i] != types[i]) {
				throw TypeMismatchException(new_types[i], types[i], "Type mismatch when combining rows");
			}
			if (types[i].InternalType() == PhysicalType::LIST) {
				// need to check all the chunks because they can have only-null list entries
				for (auto &chunk : chunks) {
					auto &chunk_vec = chunk->data[i];
					auto &new_vec = new_chunk.data[i];
					auto &chunk_type = chunk_vec.GetType();
					auto &new_type = new_vec.GetType();
					if (chunk_type != new_type) {
						throw TypeMismatchException(chunk_type, new_type, "Type mismatch when combining lists");
					}
				}
			}
			// TODO check structs, too
		}

		// first append data to the current chunk
		DataChunk &last_chunk = *chunks.back();
		idx_t added_data = MinValue<idx_t>(remaining_data, STANDARD_VECTOR_SIZE - last_chunk.size());
		if (added_data > 0) {
			// copy <added_data> elements to the last chunk
			new_chunk.Flatten();
			// have to be careful here: setting the cardinality without calling normalify can cause incorrect partial
			// decompression
			idx_t old_count = new_chunk.size();
			new_chunk.SetCardinality(added_data);

			last_chunk.Append(new_chunk);
			remaining_data -= added_data;
			// reset the chunk to the old data
			new_chunk.SetCardinality(old_count);
			offset = added_data;
		}
	}

	if (remaining_data > 0) {
		// create a new chunk and fill it with the remainder
		auto chunk = make_unique<DataChunk>();
		chunk->Initialize(allocator, types);
		new_chunk.Copy(*chunk, offset);
		chunks.push_back(std::move(chunk));
	}
}

void ChunkCollection::Append(unique_ptr<DataChunk> new_chunk) {
	if (types.empty()) {
		types = new_chunk->GetTypes();
	}
	D_ASSERT(types == new_chunk->GetTypes());
	count += new_chunk->size();
	chunks.push_back(std::move(new_chunk));
}

void ChunkCollection::Fuse(ChunkCollection &other) {
	if (count == 0) {
		chunks.reserve(other.ChunkCount());
		for (idx_t chunk_idx = 0; chunk_idx < other.ChunkCount(); ++chunk_idx) {
			auto lhs = make_unique<DataChunk>();
			auto &rhs = other.GetChunk(chunk_idx);
			lhs->data.reserve(rhs.data.size());
			for (auto &v : rhs.data) {
				lhs->data.emplace_back(Vector(v));
			}
			lhs->SetCardinality(rhs.size());
			chunks.push_back(std::move(lhs));
		}
		count = other.Count();
	} else {
		D_ASSERT(this->ChunkCount() == other.ChunkCount());
		for (idx_t chunk_idx = 0; chunk_idx < ChunkCount(); ++chunk_idx) {
			auto &lhs = this->GetChunk(chunk_idx);
			auto &rhs = other.GetChunk(chunk_idx);
			D_ASSERT(lhs.size() == rhs.size());
			for (auto &v : rhs.data) {
				lhs.data.emplace_back(Vector(v));
			}
		}
	}
	types.insert(types.end(), other.types.begin(), other.types.end());
}

Value ChunkCollection::GetValue(idx_t column, idx_t index) {
	return chunks[LocateChunk(index)]->GetValue(column, index % STANDARD_VECTOR_SIZE);
}

void ChunkCollection::SetValue(idx_t column, idx_t index, const Value &value) {
	chunks[LocateChunk(index)]->SetValue(column, index % STANDARD_VECTOR_SIZE, value);
}

void ChunkCollection::CopyCell(idx_t column, idx_t index, Vector &target, idx_t target_offset) {
	auto &chunk = GetChunkForRow(index);
	auto &source = chunk.data[column];
	const auto source_offset = index % STANDARD_VECTOR_SIZE;
	VectorOperations::Copy(source, target, source_offset + 1, source_offset, target_offset);
}

string ChunkCollection::ToString() const {
	return chunks.empty() ? "ChunkCollection [ 0 ]"
	                      : "ChunkCollection [ " + std::to_string(count) + " ]: \n" + chunks[0]->ToString();
}

void ChunkCollection::Print() const {
	Printer::Print(ToString());
}

} // namespace duckdb





namespace duckdb {

ColumnDataAllocator::ColumnDataAllocator(Allocator &allocator) : type(ColumnDataAllocatorType::IN_MEMORY_ALLOCATOR) {
	alloc.allocator = &allocator;
}

ColumnDataAllocator::ColumnDataAllocator(BufferManager &buffer_manager)
    : type(ColumnDataAllocatorType::BUFFER_MANAGER_ALLOCATOR) {
	alloc.buffer_manager = &buffer_manager;
}

ColumnDataAllocator::ColumnDataAllocator(ClientContext &context, ColumnDataAllocatorType allocator_type)
    : type(allocator_type) {
	switch (type) {
	case ColumnDataAllocatorType::BUFFER_MANAGER_ALLOCATOR:
		alloc.buffer_manager = &BufferManager::GetBufferManager(context);
		break;
	case ColumnDataAllocatorType::IN_MEMORY_ALLOCATOR:
		alloc.allocator = &Allocator::Get(context);
		break;
	default:
		throw InternalException("Unrecognized column data allocator type");
	}
}

ColumnDataAllocator::ColumnDataAllocator(ColumnDataAllocator &other) {
	type = other.GetType();
	switch (type) {
	case ColumnDataAllocatorType::BUFFER_MANAGER_ALLOCATOR:
		alloc.allocator = other.alloc.allocator;
		break;
	case ColumnDataAllocatorType::IN_MEMORY_ALLOCATOR:
		alloc.buffer_manager = other.alloc.buffer_manager;
		break;
	default:
		throw InternalException("Unrecognized column data allocator type");
	}
}

BufferHandle ColumnDataAllocator::Pin(uint32_t block_id) {
	D_ASSERT(type == ColumnDataAllocatorType::BUFFER_MANAGER_ALLOCATOR);
	shared_ptr<BlockHandle> handle;
	if (shared) {
		// we only need to grab the lock when accessing the vector, because vector access is not thread-safe:
		// the vector can be resized by another thread while we try to access it
		lock_guard<mutex> guard(lock);
		handle = blocks[block_id].handle;
	} else {
		handle = blocks[block_id].handle;
	}
	return alloc.buffer_manager->Pin(handle);
}

BufferHandle ColumnDataAllocator::AllocateBlock(idx_t size) {
	D_ASSERT(type == ColumnDataAllocatorType::BUFFER_MANAGER_ALLOCATOR);
	auto block_size = MaxValue<idx_t>(size, Storage::BLOCK_SIZE);
	BlockMetaData data;
	data.size = 0;
	data.capacity = block_size;
	auto pin = alloc.buffer_manager->Allocate(block_size, false, &data.handle);
	blocks.push_back(std::move(data));
	return pin;
}

void ColumnDataAllocator::AllocateEmptyBlock(idx_t size) {
	auto allocation_amount = MaxValue<idx_t>(NextPowerOfTwo(size), 4096);
	if (!blocks.empty()) {
		auto last_capacity = blocks.back().capacity;
		auto next_capacity = MinValue<idx_t>(last_capacity * 2, last_capacity + Storage::BLOCK_SIZE);
		allocation_amount = MaxValue<idx_t>(next_capacity, allocation_amount);
	}
	D_ASSERT(type == ColumnDataAllocatorType::IN_MEMORY_ALLOCATOR);
	BlockMetaData data;
	data.size = 0;
	data.capacity = allocation_amount;
	data.handle = nullptr;
	blocks.push_back(std::move(data));
}

void ColumnDataAllocator::AssignPointer(uint32_t &block_id, uint32_t &offset, data_ptr_t pointer) {
	auto pointer_value = uintptr_t(pointer);
	if (sizeof(uintptr_t) == sizeof(uint32_t)) {
		block_id = uint32_t(pointer_value);
	} else if (sizeof(uintptr_t) == sizeof(uint64_t)) {
		block_id = uint32_t(pointer_value & 0xFFFFFFFF);
		offset = uint32_t(pointer_value >> 32);
	} else {
		throw InternalException("ColumnDataCollection: Architecture not supported!?");
	}
}

void ColumnDataAllocator::AllocateBuffer(idx_t size, uint32_t &block_id, uint32_t &offset,
                                         ChunkManagementState *chunk_state) {
	D_ASSERT(allocated_data.empty());
	if (blocks.empty() || blocks.back().Capacity() < size) {
		auto pinned_block = AllocateBlock(size);
		if (chunk_state) {
			D_ASSERT(!blocks.empty());
			auto new_block_id = blocks.size() - 1;
			chunk_state->handles[new_block_id] = std::move(pinned_block);
		}
	}
	auto &block = blocks.back();
	D_ASSERT(size <= block.capacity - block.size);
	block_id = blocks.size() - 1;
	if (chunk_state && chunk_state->handles.find(block_id) == chunk_state->handles.end()) {
		// not guaranteed to be pinned already by this thread (if shared allocator)
		chunk_state->handles[block_id] = alloc.buffer_manager->Pin(blocks[block_id].handle);
	}
	offset = block.size;
	block.size += size;
}

void ColumnDataAllocator::AllocateMemory(idx_t size, uint32_t &block_id, uint32_t &offset,
                                         ChunkManagementState *chunk_state) {
	D_ASSERT(blocks.size() == allocated_data.size());
	if (blocks.empty() || blocks.back().Capacity() < size) {
		AllocateEmptyBlock(size);
		auto &last_block = blocks.back();
		auto allocated = alloc.allocator->Allocate(last_block.capacity);
		allocated_data.push_back(std::move(allocated));
	}
	auto &block = blocks.back();
	D_ASSERT(size <= block.capacity - block.size);
	AssignPointer(block_id, offset, allocated_data.back().get() + block.size);
	block.size += size;
}

void ColumnDataAllocator::AllocateData(idx_t size, uint32_t &block_id, uint32_t &offset,
                                       ChunkManagementState *chunk_state) {
	switch (type) {
	case ColumnDataAllocatorType::BUFFER_MANAGER_ALLOCATOR:
		if (shared) {
			lock_guard<mutex> guard(lock);
			AllocateBuffer(size, block_id, offset, chunk_state);
		} else {
			AllocateBuffer(size, block_id, offset, chunk_state);
		}
		break;
	case ColumnDataAllocatorType::IN_MEMORY_ALLOCATOR:
		D_ASSERT(!shared);
		AllocateMemory(size, block_id, offset, chunk_state);
		break;
	default:
		throw InternalException("Unrecognized allocator type");
	}
}

void ColumnDataAllocator::Initialize(ColumnDataAllocator &other) {
	D_ASSERT(other.HasBlocks());
	blocks.push_back(other.blocks.back());
}

data_ptr_t ColumnDataAllocator::GetDataPointer(ChunkManagementState &state, uint32_t block_id, uint32_t offset) {
	if (type == ColumnDataAllocatorType::IN_MEMORY_ALLOCATOR) {
		// in-memory allocator: construct pointer from block_id and offset
		if (sizeof(uintptr_t) == sizeof(uint32_t)) {
			uintptr_t pointer_value = uintptr_t(block_id);
			return (data_ptr_t)pointer_value;
		} else if (sizeof(uintptr_t) == sizeof(uint64_t)) {
			uintptr_t pointer_value = (uintptr_t(offset) << 32) | uintptr_t(block_id);
			return (data_ptr_t)pointer_value;
		} else {
			throw InternalException("ColumnDataCollection: Architecture not supported!?");
		}
	}
	D_ASSERT(state.handles.find(block_id) != state.handles.end());
	return state.handles[block_id].Ptr() + offset;
}

void ColumnDataAllocator::UnswizzlePointers(ChunkManagementState &state, Vector &result, uint16_t v_offset,
                                            uint16_t count, uint32_t block_id, uint32_t offset) {
	D_ASSERT(result.GetType().InternalType() == PhysicalType::VARCHAR);
	lock_guard<mutex> guard(lock);

	auto &validity = FlatVector::Validity(result);
	auto strings = FlatVector::GetData<string_t>(result);

	// find first non-inlined string
	uint32_t i = v_offset;
	const uint32_t end = v_offset + count;
	for (; i < end; i++) {
		if (!validity.RowIsValid(i)) {
			continue;
		}
		if (!strings[i].IsInlined()) {
			break;
		}
	}
	// at least one string must be non-inlined, otherwise this function should not be called
	D_ASSERT(i < end);

	auto base_ptr = (char *)GetDataPointer(state, block_id, offset);
	if (strings[i].GetDataUnsafe() == base_ptr) {
		// pointers are still valid
		return;
	}

	// pointer mismatch! pointers are invalid, set them correctly
	for (; i < end; i++) {
		if (!validity.RowIsValid(i)) {
			continue;
		}
		if (strings[i].IsInlined()) {
			continue;
		}
		strings[i].SetPointer(base_ptr);
		base_ptr += strings[i].GetSize();
	}
}

void ColumnDataAllocator::DeleteBlock(uint32_t block_id) {
	blocks[block_id].handle->SetCanDestroy(true);
}

Allocator &ColumnDataAllocator::GetAllocator() {
	return type == ColumnDataAllocatorType::IN_MEMORY_ALLOCATOR ? *alloc.allocator
	                                                            : alloc.buffer_manager->GetBufferAllocator();
}

void ColumnDataAllocator::InitializeChunkState(ChunkManagementState &state, ChunkMetaData &chunk) {
	if (type != ColumnDataAllocatorType::BUFFER_MANAGER_ALLOCATOR) {
		// nothing to pin
		return;
	}
	// release any handles that are no longer required
	bool found_handle;
	do {
		found_handle = false;
		for (auto it = state.handles.begin(); it != state.handles.end(); it++) {
			if (chunk.block_ids.find(it->first) != chunk.block_ids.end()) {
				// still required: do not release
				continue;
			}
			state.handles.erase(it);
			found_handle = true;
			break;
		}
	} while (found_handle);

	// grab any handles that are now required
	for (auto &block_id : chunk.block_ids) {
		if (state.handles.find(block_id) != state.handles.end()) {
			// already pinned: don't need to do anything
			continue;
		}
		state.handles[block_id] = Pin(block_id);
	}
}

uint32_t BlockMetaData::Capacity() {
	D_ASSERT(size <= capacity);
	return capacity - size;
}

} // namespace duckdb








namespace duckdb {

struct ColumnDataMetaData;

typedef void (*column_data_copy_function_t)(ColumnDataMetaData &meta_data, const UnifiedVectorFormat &source_data,
                                            Vector &source, idx_t offset, idx_t copy_count);

struct ColumnDataCopyFunction {
	column_data_copy_function_t function;
	vector<ColumnDataCopyFunction> child_functions;
};

struct ColumnDataMetaData {
	ColumnDataMetaData(ColumnDataCopyFunction &copy_function, ColumnDataCollectionSegment &segment,
	                   ColumnDataAppendState &state, ChunkMetaData &chunk_data, VectorDataIndex vector_data_index)
	    : copy_function(copy_function), segment(segment), state(state), chunk_data(chunk_data),
	      vector_data_index(vector_data_index) {
	}
	ColumnDataMetaData(ColumnDataCopyFunction &copy_function, ColumnDataMetaData &parent,
	                   VectorDataIndex vector_data_index)
	    : copy_function(copy_function), segment(parent.segment), state(parent.state), chunk_data(parent.chunk_data),
	      vector_data_index(vector_data_index) {
	}

	ColumnDataCopyFunction &copy_function;
	ColumnDataCollectionSegment &segment;
	ColumnDataAppendState &state;
	ChunkMetaData &chunk_data;
	VectorDataIndex vector_data_index;
	idx_t child_list_size = DConstants::INVALID_INDEX;

	VectorMetaData &GetVectorMetaData() {
		return segment.GetVectorData(vector_data_index);
	}
};

//! Explicitly initialized without types
ColumnDataCollection::ColumnDataCollection(Allocator &allocator_p) {
	types.clear();
	count = 0;
	this->finished_append = false;
	allocator = make_shared<ColumnDataAllocator>(allocator_p);
}

ColumnDataCollection::ColumnDataCollection(Allocator &allocator_p, vector<LogicalType> types_p) {
	Initialize(std::move(types_p));
	allocator = make_shared<ColumnDataAllocator>(allocator_p);
}

ColumnDataCollection::ColumnDataCollection(BufferManager &buffer_manager, vector<LogicalType> types_p) {
	Initialize(std::move(types_p));
	allocator = make_shared<ColumnDataAllocator>(buffer_manager);
}

ColumnDataCollection::ColumnDataCollection(shared_ptr<ColumnDataAllocator> allocator_p, vector<LogicalType> types_p) {
	Initialize(std::move(types_p));
	this->allocator = std::move(allocator_p);
}

ColumnDataCollection::ColumnDataCollection(ClientContext &context, vector<LogicalType> types_p,
                                           ColumnDataAllocatorType type)
    : ColumnDataCollection(make_shared<ColumnDataAllocator>(context, type), std::move(types_p)) {
	D_ASSERT(!types.empty());
}

ColumnDataCollection::ColumnDataCollection(ColumnDataCollection &other)
    : ColumnDataCollection(other.allocator, other.types) {
	other.finished_append = true;
	D_ASSERT(!types.empty());
}

ColumnDataCollection::~ColumnDataCollection() {
}

void ColumnDataCollection::Initialize(vector<LogicalType> types_p) {
	this->types = std::move(types_p);
	this->count = 0;
	this->finished_append = false;
	D_ASSERT(!types.empty());
	copy_functions.reserve(types.size());
	for (auto &type : types) {
		copy_functions.push_back(GetCopyFunction(type));
	}
}

void ColumnDataCollection::CreateSegment() {
	segments.emplace_back(make_unique<ColumnDataCollectionSegment>(allocator, types));
}

Allocator &ColumnDataCollection::GetAllocator() const {
	return allocator->GetAllocator();
}

//===--------------------------------------------------------------------===//
// ColumnDataRow
//===--------------------------------------------------------------------===//
ColumnDataRow::ColumnDataRow(DataChunk &chunk_p, idx_t row_index, idx_t base_index)
    : chunk(chunk_p), row_index(row_index), base_index(base_index) {
}

Value ColumnDataRow::GetValue(idx_t column_index) const {
	D_ASSERT(column_index < chunk.ColumnCount());
	D_ASSERT(row_index < chunk.size());
	return chunk.data[column_index].GetValue(row_index);
}

idx_t ColumnDataRow::RowIndex() const {
	return base_index + row_index;
}

//===--------------------------------------------------------------------===//
// ColumnDataRowCollection
//===--------------------------------------------------------------------===//
ColumnDataRowCollection::ColumnDataRowCollection(const ColumnDataCollection &collection) {
	if (collection.Count() == 0) {
		return;
	}
	// read all the chunks
	ColumnDataScanState temp_scan_state;
	collection.InitializeScan(temp_scan_state, ColumnDataScanProperties::DISALLOW_ZERO_COPY);
	while (true) {
		auto chunk = make_unique<DataChunk>();
		collection.InitializeScanChunk(*chunk);
		if (!collection.Scan(temp_scan_state, *chunk)) {
			break;
		}
		chunks.push_back(std::move(chunk));
	}
	// now create all of the column data rows
	rows.reserve(collection.Count());
	idx_t base_row = 0;
	for (auto &chunk : chunks) {
		for (idx_t row_idx = 0; row_idx < chunk->size(); row_idx++) {
			rows.emplace_back(*chunk, row_idx, base_row);
		}
		base_row += chunk->size();
	}
}

ColumnDataRow &ColumnDataRowCollection::operator[](idx_t i) {
	return rows[i];
}

const ColumnDataRow &ColumnDataRowCollection::operator[](idx_t i) const {
	return rows[i];
}

Value ColumnDataRowCollection::GetValue(idx_t column, idx_t index) const {
	return rows[index].GetValue(column);
}

//===--------------------------------------------------------------------===//
// ColumnDataChunkIterator
//===--------------------------------------------------------------------===//
ColumnDataChunkIterationHelper ColumnDataCollection::Chunks() const {
	vector<column_t> column_ids;
	for (idx_t i = 0; i < ColumnCount(); i++) {
		column_ids.push_back(i);
	}
	return Chunks(column_ids);
}

ColumnDataChunkIterationHelper ColumnDataCollection::Chunks(vector<column_t> column_ids) const {
	return ColumnDataChunkIterationHelper(*this, std::move(column_ids));
}

ColumnDataChunkIterationHelper::ColumnDataChunkIterationHelper(const ColumnDataCollection &collection_p,
                                                               vector<column_t> column_ids_p)
    : collection(collection_p), column_ids(std::move(column_ids_p)) {
}

ColumnDataChunkIterationHelper::ColumnDataChunkIterator::ColumnDataChunkIterator(
    const ColumnDataCollection *collection_p, vector<column_t> column_ids_p)
    : collection(collection_p), scan_chunk(make_shared<DataChunk>()), row_index(0) {
	if (!collection) {
		return;
	}
	collection->InitializeScan(scan_state, std::move(column_ids_p));
	collection->InitializeScanChunk(scan_state, *scan_chunk);
	collection->Scan(scan_state, *scan_chunk);
}

void ColumnDataChunkIterationHelper::ColumnDataChunkIterator::Next() {
	if (!collection) {
		return;
	}
	if (!collection->Scan(scan_state, *scan_chunk)) {
		collection = nullptr;
		row_index = 0;
	} else {
		row_index += scan_chunk->size();
	}
}

ColumnDataChunkIterationHelper::ColumnDataChunkIterator &
ColumnDataChunkIterationHelper::ColumnDataChunkIterator::operator++() {
	Next();
	return *this;
}

bool ColumnDataChunkIterationHelper::ColumnDataChunkIterator::operator!=(const ColumnDataChunkIterator &other) const {
	return collection != other.collection || row_index != other.row_index;
}

DataChunk &ColumnDataChunkIterationHelper::ColumnDataChunkIterator::operator*() const {
	return *scan_chunk;
}

//===--------------------------------------------------------------------===//
// ColumnDataRowIterator
//===--------------------------------------------------------------------===//
ColumnDataRowIterationHelper ColumnDataCollection::Rows() const {
	return ColumnDataRowIterationHelper(*this);
}

ColumnDataRowIterationHelper::ColumnDataRowIterationHelper(const ColumnDataCollection &collection_p)
    : collection(collection_p) {
}

ColumnDataRowIterationHelper::ColumnDataRowIterator::ColumnDataRowIterator(const ColumnDataCollection *collection_p)
    : collection(collection_p), scan_chunk(make_shared<DataChunk>()), current_row(*scan_chunk, 0, 0) {
	if (!collection) {
		return;
	}
	collection->InitializeScan(scan_state);
	collection->InitializeScanChunk(*scan_chunk);
	collection->Scan(scan_state, *scan_chunk);
}

void ColumnDataRowIterationHelper::ColumnDataRowIterator::Next() {
	if (!collection) {
		return;
	}
	current_row.row_index++;
	if (current_row.row_index >= scan_chunk->size()) {
		current_row.base_index += scan_chunk->size();
		current_row.row_index = 0;
		if (!collection->Scan(scan_state, *scan_chunk)) {
			// exhausted collection: move iterator to nop state
			current_row.base_index = 0;
			collection = nullptr;
		}
	}
}

ColumnDataRowIterationHelper::ColumnDataRowIterator ColumnDataRowIterationHelper::begin() { // NOLINT
	return ColumnDataRowIterationHelper::ColumnDataRowIterator(collection.Count() == 0 ? nullptr : &collection);
}
ColumnDataRowIterationHelper::ColumnDataRowIterator ColumnDataRowIterationHelper::end() { // NOLINT
	return ColumnDataRowIterationHelper::ColumnDataRowIterator(nullptr);
}

ColumnDataRowIterationHelper::ColumnDataRowIterator &ColumnDataRowIterationHelper::ColumnDataRowIterator::operator++() {
	Next();
	return *this;
}

bool ColumnDataRowIterationHelper::ColumnDataRowIterator::operator!=(const ColumnDataRowIterator &other) const {
	return collection != other.collection || current_row.row_index != other.current_row.row_index ||
	       current_row.base_index != other.current_row.base_index;
}

const ColumnDataRow &ColumnDataRowIterationHelper::ColumnDataRowIterator::operator*() const {
	return current_row;
}

//===--------------------------------------------------------------------===//
// Append
//===--------------------------------------------------------------------===//
void ColumnDataCollection::InitializeAppend(ColumnDataAppendState &state) {
	D_ASSERT(!finished_append);
	state.vector_data.resize(types.size());
	if (segments.empty()) {
		CreateSegment();
	}
	auto &segment = *segments.back();
	if (segment.chunk_data.empty()) {
		segment.AllocateNewChunk();
	}
	segment.InitializeChunkState(segment.chunk_data.size() - 1, state.current_chunk_state);
}

void ColumnDataCopyValidity(const UnifiedVectorFormat &source_data, validity_t *target, idx_t source_offset,
                            idx_t target_offset, idx_t copy_count) {
	ValidityMask validity(target);
	if (target_offset == 0) {
		// first time appending to this vector
		// all data here is still uninitialized
		// initialize the validity mask to set all to valid
		validity.SetAllValid(STANDARD_VECTOR_SIZE);
	}
	// FIXME: we can do something more optimized here using bitshifts & bitwise ors
	if (!source_data.validity.AllValid()) {
		for (idx_t i = 0; i < copy_count; i++) {
			auto idx = source_data.sel->get_index(source_offset + i);
			if (!source_data.validity.RowIsValid(idx)) {
				validity.SetInvalid(target_offset + i);
			}
		}
	}
}

template <class T>
struct BaseValueCopy {
	static idx_t TypeSize() {
		return sizeof(T);
	}

	template <class OP>
	static void Assign(ColumnDataMetaData &meta_data, data_ptr_t target, data_ptr_t source, idx_t target_idx,
	                   idx_t source_idx) {
		auto result_data = (T *)target;
		auto source_data = (T *)source;
		result_data[target_idx] = OP::Operation(meta_data, source_data[source_idx]);
	}
};

template <class T>
struct StandardValueCopy : public BaseValueCopy<T> {
	static T Operation(ColumnDataMetaData &, T input) {
		return input;
	}
};

struct StringValueCopy : public BaseValueCopy<string_t> {
	static string_t Operation(ColumnDataMetaData &meta_data, string_t input) {
		return input.IsInlined() ? input : meta_data.segment.heap.AddBlob(input);
	}
};

struct ConstListValueCopy : public BaseValueCopy<list_entry_t> {
	using TYPE = list_entry_t;

	static TYPE Operation(ColumnDataMetaData &meta_data, TYPE input) {
		input.offset = meta_data.child_list_size;
		return input;
	}
};

struct ListValueCopy : public BaseValueCopy<list_entry_t> {
	using TYPE = list_entry_t;

	static TYPE Operation(ColumnDataMetaData &meta_data, TYPE input) {
		input.offset = meta_data.child_list_size;
		meta_data.child_list_size += input.length;
		return input;
	}
};

struct StructValueCopy {
	static idx_t TypeSize() {
		return 0;
	}

	template <class OP>
	static void Assign(ColumnDataMetaData &meta_data, data_ptr_t target, data_ptr_t source, idx_t target_idx,
	                   idx_t source_idx) {
	}
};

template <class OP>
static void TemplatedColumnDataCopy(ColumnDataMetaData &meta_data, const UnifiedVectorFormat &source_data,
                                    Vector &source, idx_t offset, idx_t count) {
	auto &segment = meta_data.segment;
	auto &append_state = meta_data.state;

	auto current_index = meta_data.vector_data_index;
	idx_t remaining = count;
	while (remaining > 0) {
		auto &current_segment = segment.GetVectorData(current_index);
		idx_t append_count = MinValue<idx_t>(STANDARD_VECTOR_SIZE - current_segment.count, remaining);

		auto base_ptr = segment.allocator->GetDataPointer(append_state.current_chunk_state, current_segment.block_id,
		                                                  current_segment.offset);
		auto validity_data = ColumnDataCollectionSegment::GetValidityPointer(base_ptr, OP::TypeSize());

		ValidityMask result_validity(validity_data);
		if (current_segment.count == 0) {
			// first time appending to this vector
			// all data here is still uninitialized
			// initialize the validity mask to set all to valid
			result_validity.SetAllValid(STANDARD_VECTOR_SIZE);
		}
		for (idx_t i = 0; i < append_count; i++) {
			auto source_idx = source_data.sel->get_index(offset + i);
			if (source_data.validity.RowIsValid(source_idx)) {
				OP::template Assign<OP>(meta_data, base_ptr, source_data.data, current_segment.count + i, source_idx);
			} else {
				result_validity.SetInvalid(current_segment.count + i);
			}
		}
		current_segment.count += append_count;
		offset += append_count;
		remaining -= append_count;
		if (remaining > 0) {
			// need to append more, check if we need to allocate a new vector or not
			if (!current_segment.next_data.IsValid()) {
				segment.AllocateVector(source.GetType(), meta_data.chunk_data, append_state, current_index);
			}
			D_ASSERT(segment.GetVectorData(current_index).next_data.IsValid());
			current_index = segment.GetVectorData(current_index).next_data;
		}
	}
}

template <class T>
static void ColumnDataCopy(ColumnDataMetaData &meta_data, const UnifiedVectorFormat &source_data, Vector &source,
                           idx_t offset, idx_t copy_count) {
	TemplatedColumnDataCopy<StandardValueCopy<T>>(meta_data, source_data, source, offset, copy_count);
}

template <>
void ColumnDataCopy<string_t>(ColumnDataMetaData &meta_data, const UnifiedVectorFormat &source_data, Vector &source,
                              idx_t offset, idx_t copy_count) {

	const auto &allocator_type = meta_data.segment.allocator->GetType();
	if (allocator_type == ColumnDataAllocatorType::IN_MEMORY_ALLOCATOR) {
		// strings cannot be spilled to disk - use StringHeap
		TemplatedColumnDataCopy<StringValueCopy>(meta_data, source_data, source, offset, copy_count);
		return;
	}
	D_ASSERT(allocator_type == ColumnDataAllocatorType::BUFFER_MANAGER_ALLOCATOR);

	auto &segment = meta_data.segment;
	auto &append_state = meta_data.state;

	VectorDataIndex child_index;
	if (meta_data.GetVectorMetaData().child_index.IsValid()) {
		// find the last child index
		child_index = segment.GetChildIndex(meta_data.GetVectorMetaData().child_index);
		auto next_child_index = segment.GetVectorData(child_index).next_data;
		while (next_child_index.IsValid()) {
			child_index = next_child_index;
			next_child_index = segment.GetVectorData(child_index).next_data;
		}
	}

	auto current_index = meta_data.vector_data_index;
	idx_t remaining = copy_count;
	while (remaining > 0) {
		// how many values fit in the current string vector
		idx_t vector_remaining =
		    MinValue<idx_t>(STANDARD_VECTOR_SIZE - segment.GetVectorData(current_index).count, remaining);

		// 'append_count' is less if we cannot fit that amount of non-inlined strings on one buffer-managed block
		idx_t append_count;
		idx_t heap_size = 0;
		const auto source_entries = (string_t *)source_data.data;
		for (append_count = 0; append_count < vector_remaining; append_count++) {
			auto source_idx = source_data.sel->get_index(offset + append_count);
			if (!source_data.validity.RowIsValid(source_idx)) {
				continue;
			}
			const auto &entry = source_entries[source_idx];
			if (entry.IsInlined()) {
				continue;
			}
			if (heap_size + entry.GetSize() > Storage::BLOCK_SIZE) {
				break;
			}
			heap_size += entry.GetSize();
		}

		if (vector_remaining != 0 && append_count == 0) {
			// single string is longer than Storage::BLOCK_SIZE
			// we allocate one block at a time for long strings
			auto source_idx = source_data.sel->get_index(offset + append_count);
			D_ASSERT(source_data.validity.RowIsValid(source_idx));
			D_ASSERT(!source_entries[source_idx].IsInlined());
			D_ASSERT(source_entries[source_idx].GetSize() > Storage::BLOCK_SIZE);
			heap_size += source_entries[source_idx].GetSize();
			append_count++;
		}

		// allocate string heap for the next 'append_count' strings
		data_ptr_t heap_ptr = nullptr;
		if (heap_size != 0) {
			child_index = segment.AllocateStringHeap(heap_size, meta_data.chunk_data, append_state, child_index);
			if (!meta_data.GetVectorMetaData().child_index.IsValid()) {
				meta_data.GetVectorMetaData().child_index = meta_data.segment.AddChildIndex(child_index);
			}
			auto &child_segment = segment.GetVectorData(child_index);
			heap_ptr = segment.allocator->GetDataPointer(append_state.current_chunk_state, child_segment.block_id,
			                                             child_segment.offset);
		}

		auto &current_segment = segment.GetVectorData(current_index);
		auto base_ptr = segment.allocator->GetDataPointer(append_state.current_chunk_state, current_segment.block_id,
		                                                  current_segment.offset);
		auto validity_data = ColumnDataCollectionSegment::GetValidityPointer(base_ptr, sizeof(string_t));
		ValidityMask target_validity(validity_data);
		if (current_segment.count == 0) {
			// first time appending to this vector
			// all data here is still uninitialized
			// initialize the validity mask to set all to valid
			target_validity.SetAllValid(STANDARD_VECTOR_SIZE);
		}

		auto target_entries = (string_t *)base_ptr;
		for (idx_t i = 0; i < append_count; i++) {
			auto source_idx = source_data.sel->get_index(offset + i);
			auto target_idx = current_segment.count + i;
			if (!source_data.validity.RowIsValid(source_idx)) {
				target_validity.SetInvalid(target_idx);
				continue;
			}
			const auto &source_entry = source_entries[source_idx];
			auto &target_entry = target_entries[target_idx];
			if (source_entry.IsInlined()) {
				target_entry = source_entry;
			} else {
				D_ASSERT(heap_ptr != nullptr);
				memcpy(heap_ptr, source_entry.GetDataUnsafe(), source_entry.GetSize());
				target_entry = string_t((const char *)heap_ptr, source_entry.GetSize());
				heap_ptr += source_entry.GetSize();
			}
		}

		if (heap_size != 0) {
			current_segment.swizzle_data.emplace_back(child_index, current_segment.count, append_count);
		}

		current_segment.count += append_count;
		offset += append_count;
		remaining -= append_count;

		if (vector_remaining - append_count == 0) {
			// need to append more, check if we need to allocate a new vector or not
			if (!current_segment.next_data.IsValid()) {
				segment.AllocateVector(source.GetType(), meta_data.chunk_data, append_state, current_index);
			}
			D_ASSERT(segment.GetVectorData(current_index).next_data.IsValid());
			current_index = segment.GetVectorData(current_index).next_data;
		}
	}
}

template <>
void ColumnDataCopy<list_entry_t>(ColumnDataMetaData &meta_data, const UnifiedVectorFormat &source_data, Vector &source,
                                  idx_t offset, idx_t copy_count) {

	auto &segment = meta_data.segment;

	auto &child_vector = ListVector::GetEntry(source);
	auto &child_type = child_vector.GetType();

	if (!meta_data.GetVectorMetaData().child_index.IsValid()) {
		auto child_index = segment.AllocateVector(child_type, meta_data.chunk_data, meta_data.state);
		meta_data.GetVectorMetaData().child_index = meta_data.segment.AddChildIndex(child_index);
	}

	auto &child_function = meta_data.copy_function.child_functions[0];
	auto child_index = segment.GetChildIndex(meta_data.GetVectorMetaData().child_index);

	// figure out the current list size by traversing the set of child entries
	idx_t current_list_size = 0;
	auto current_child_index = child_index;
	while (current_child_index.IsValid()) {
		auto &child_vdata = segment.GetVectorData(current_child_index);
		current_list_size += child_vdata.count;
		current_child_index = child_vdata.next_data;
	}

	// set the child vector
	UnifiedVectorFormat child_vector_data;
	ColumnDataMetaData child_meta_data(child_function, meta_data, child_index);
	auto info = ListVector::GetConsecutiveChildListInfo(source, offset, copy_count);

	if (info.needs_slicing) {
		SelectionVector sel(info.child_list_info.length);
		ListVector::GetConsecutiveChildSelVector(source, sel, offset, copy_count);

		auto sliced_child_vector = Vector(child_vector, sel, info.child_list_info.length);
		sliced_child_vector.Flatten(info.child_list_info.length);
		info.child_list_info.offset = 0;

		sliced_child_vector.ToUnifiedFormat(info.child_list_info.length, child_vector_data);
		child_function.function(child_meta_data, child_vector_data, sliced_child_vector, info.child_list_info.offset,
		                        info.child_list_info.length);

	} else {
		child_vector.ToUnifiedFormat(info.child_list_info.length, child_vector_data);
		child_function.function(child_meta_data, child_vector_data, child_vector, info.child_list_info.offset,
		                        info.child_list_info.length);
	}

	// now copy the list entries
	meta_data.child_list_size = current_list_size;
	if (info.is_constant) {
		TemplatedColumnDataCopy<ConstListValueCopy>(meta_data, source_data, source, offset, copy_count);
	} else {
		TemplatedColumnDataCopy<ListValueCopy>(meta_data, source_data, source, offset, copy_count);
	}
}

void ColumnDataCopyStruct(ColumnDataMetaData &meta_data, const UnifiedVectorFormat &source_data, Vector &source,
                          idx_t offset, idx_t copy_count) {
	auto &segment = meta_data.segment;

	// copy the NULL values for the main struct vector
	TemplatedColumnDataCopy<StructValueCopy>(meta_data, source_data, source, offset, copy_count);

	auto &child_types = StructType::GetChildTypes(source.GetType());
	// now copy all the child vectors
	D_ASSERT(meta_data.GetVectorMetaData().child_index.IsValid());
	auto &child_vectors = StructVector::GetEntries(source);
	for (idx_t child_idx = 0; child_idx < child_types.size(); child_idx++) {
		auto &child_function = meta_data.copy_function.child_functions[child_idx];
		auto child_index = segment.GetChildIndex(meta_data.GetVectorMetaData().child_index, child_idx);
		ColumnDataMetaData child_meta_data(child_function, meta_data, child_index);

		UnifiedVectorFormat child_data;
		child_vectors[child_idx]->ToUnifiedFormat(copy_count, child_data);

		child_function.function(child_meta_data, child_data, *child_vectors[child_idx], offset, copy_count);
	}
}

ColumnDataCopyFunction ColumnDataCollection::GetCopyFunction(const LogicalType &type) {
	ColumnDataCopyFunction result;
	column_data_copy_function_t function;
	switch (type.InternalType()) {
	case PhysicalType::BOOL:
		function = ColumnDataCopy<bool>;
		break;
	case PhysicalType::INT8:
		function = ColumnDataCopy<int8_t>;
		break;
	case PhysicalType::INT16:
		function = ColumnDataCopy<int16_t>;
		break;
	case PhysicalType::INT32:
		function = ColumnDataCopy<int32_t>;
		break;
	case PhysicalType::INT64:
		function = ColumnDataCopy<int64_t>;
		break;
	case PhysicalType::INT128:
		function = ColumnDataCopy<hugeint_t>;
		break;
	case PhysicalType::UINT8:
		function = ColumnDataCopy<uint8_t>;
		break;
	case PhysicalType::UINT16:
		function = ColumnDataCopy<uint16_t>;
		break;
	case PhysicalType::UINT32:
		function = ColumnDataCopy<uint32_t>;
		break;
	case PhysicalType::UINT64:
		function = ColumnDataCopy<uint64_t>;
		break;
	case PhysicalType::FLOAT:
		function = ColumnDataCopy<float>;
		break;
	case PhysicalType::DOUBLE:
		function = ColumnDataCopy<double>;
		break;
	case PhysicalType::INTERVAL:
		function = ColumnDataCopy<interval_t>;
		break;
	case PhysicalType::VARCHAR:
		function = ColumnDataCopy<string_t>;
		break;
	case PhysicalType::STRUCT: {
		function = ColumnDataCopyStruct;
		auto &child_types = StructType::GetChildTypes(type);
		for (auto &kv : child_types) {
			result.child_functions.push_back(GetCopyFunction(kv.second));
		}
		break;
	}
	case PhysicalType::LIST: {
		function = ColumnDataCopy<list_entry_t>;
		auto child_function = GetCopyFunction(ListType::GetChildType(type));
		result.child_functions.push_back(child_function);
		break;
	}
	default:
		throw InternalException("Unsupported type for ColumnDataCollection::GetCopyFunction");
	}
	result.function = function;
	return result;
}

static bool IsComplexType(const LogicalType &type) {
	switch (type.InternalType()) {
	case PhysicalType::STRUCT:
	case PhysicalType::LIST:
		return true;
	default:
		return false;
	};
}

void ColumnDataCollection::Append(ColumnDataAppendState &state, DataChunk &input) {
	D_ASSERT(!finished_append);
	D_ASSERT(types == input.GetTypes());

	auto &segment = *segments.back();
	for (idx_t vector_idx = 0; vector_idx < types.size(); vector_idx++) {
		if (IsComplexType(input.data[vector_idx].GetType())) {
			input.data[vector_idx].Flatten(input.size());
		}
		input.data[vector_idx].ToUnifiedFormat(input.size(), state.vector_data[vector_idx]);
	}

	idx_t remaining = input.size();
	while (remaining > 0) {
		auto &chunk_data = segment.chunk_data.back();
		idx_t append_amount = MinValue<idx_t>(remaining, STANDARD_VECTOR_SIZE - chunk_data.count);
		if (append_amount > 0) {
			idx_t offset = input.size() - remaining;
			for (idx_t vector_idx = 0; vector_idx < types.size(); vector_idx++) {
				ColumnDataMetaData meta_data(copy_functions[vector_idx], segment, state, chunk_data,
				                             chunk_data.vector_data[vector_idx]);
				copy_functions[vector_idx].function(meta_data, state.vector_data[vector_idx], input.data[vector_idx],
				                                    offset, append_amount);
			}
			chunk_data.count += append_amount;
		}
		remaining -= append_amount;
		if (remaining > 0) {
			// more to do
			// allocate a new chunk
			segment.AllocateNewChunk();
			segment.InitializeChunkState(segment.chunk_data.size() - 1, state.current_chunk_state);
		}
	}
	segment.count += input.size();
	count += input.size();
}

void ColumnDataCollection::Append(DataChunk &input) {
	ColumnDataAppendState state;
	InitializeAppend(state);
	Append(state, input);
}

//===--------------------------------------------------------------------===//
// Scan
//===--------------------------------------------------------------------===//
void ColumnDataCollection::InitializeScan(ColumnDataScanState &state, ColumnDataScanProperties properties) const {
	vector<column_t> column_ids;
	column_ids.reserve(types.size());
	for (idx_t i = 0; i < types.size(); i++) {
		column_ids.push_back(i);
	}
	InitializeScan(state, std::move(column_ids), properties);
}

void ColumnDataCollection::InitializeScan(ColumnDataScanState &state, vector<column_t> column_ids,
                                          ColumnDataScanProperties properties) const {
	state.chunk_index = 0;
	state.segment_index = 0;
	state.current_row_index = 0;
	state.next_row_index = 0;
	state.current_chunk_state.handles.clear();
	state.properties = properties;
	state.column_ids = std::move(column_ids);
}

void ColumnDataCollection::InitializeScan(ColumnDataParallelScanState &state,
                                          ColumnDataScanProperties properties) const {
	InitializeScan(state.scan_state, properties);
}

void ColumnDataCollection::InitializeScan(ColumnDataParallelScanState &state, vector<column_t> column_ids,
                                          ColumnDataScanProperties properties) const {
	InitializeScan(state.scan_state, std::move(column_ids), properties);
}

bool ColumnDataCollection::Scan(ColumnDataParallelScanState &state, ColumnDataLocalScanState &lstate,
                                DataChunk &result) const {
	result.Reset();

	idx_t chunk_index;
	idx_t segment_index;
	idx_t row_index;
	{
		lock_guard<mutex> l(state.lock);
		if (!NextScanIndex(state.scan_state, chunk_index, segment_index, row_index)) {
			return false;
		}
	}
	ScanAtIndex(state, lstate, result, chunk_index, segment_index, row_index);
	return true;
}

void ColumnDataCollection::InitializeScanChunk(DataChunk &chunk) const {
	chunk.Initialize(allocator->GetAllocator(), types);
}

void ColumnDataCollection::InitializeScanChunk(ColumnDataScanState &state, DataChunk &chunk) const {
	D_ASSERT(!state.column_ids.empty());
	vector<LogicalType> chunk_types;
	chunk_types.reserve(state.column_ids.size());
	for (idx_t i = 0; i < state.column_ids.size(); i++) {
		auto column_idx = state.column_ids[i];
		D_ASSERT(column_idx < types.size());
		chunk_types.push_back(types[column_idx]);
	}
	chunk.Initialize(allocator->GetAllocator(), chunk_types);
}

bool ColumnDataCollection::NextScanIndex(ColumnDataScanState &state, idx_t &chunk_index, idx_t &segment_index,
                                         idx_t &row_index) const {
	row_index = state.current_row_index = state.next_row_index;
	// check if we still have collections to scan
	if (state.segment_index >= segments.size()) {
		// no more data left in the scan
		return false;
	}
	// check within the current collection if we still have chunks to scan
	while (state.chunk_index >= segments[state.segment_index]->chunk_data.size()) {
		// exhausted all chunks for this internal data structure: move to the next one
		state.chunk_index = 0;
		state.segment_index++;
		state.current_chunk_state.handles.clear();
		if (state.segment_index >= segments.size()) {
			return false;
		}
	}
	state.next_row_index += segments[state.segment_index]->chunk_data[state.chunk_index].count;
	segment_index = state.segment_index;
	chunk_index = state.chunk_index++;
	return true;
}

void ColumnDataCollection::ScanAtIndex(ColumnDataParallelScanState &state, ColumnDataLocalScanState &lstate,
                                       DataChunk &result, idx_t chunk_index, idx_t segment_index,
                                       idx_t row_index) const {
	if (segment_index != lstate.current_segment_index) {
		lstate.current_chunk_state.handles.clear();
		lstate.current_segment_index = segment_index;
	}
	auto &segment = *segments[segment_index];
	lstate.current_chunk_state.properties = state.scan_state.properties;
	segment.ReadChunk(chunk_index, lstate.current_chunk_state, result, state.scan_state.column_ids);
	lstate.current_row_index = row_index;
	result.Verify();
}

bool ColumnDataCollection::Scan(ColumnDataScanState &state, DataChunk &result) const {
	result.Reset();

	idx_t chunk_index;
	idx_t segment_index;
	idx_t row_index;
	if (!NextScanIndex(state, chunk_index, segment_index, row_index)) {
		return false;
	}

	// found a chunk to scan -> scan it
	auto &segment = *segments[segment_index];
	state.current_chunk_state.properties = state.properties;
	segment.ReadChunk(chunk_index, state.current_chunk_state, result, state.column_ids);
	result.Verify();
	return true;
}

ColumnDataRowCollection ColumnDataCollection::GetRows() const {
	return ColumnDataRowCollection(*this);
}

//===--------------------------------------------------------------------===//
// Combine
//===--------------------------------------------------------------------===//
void ColumnDataCollection::Combine(ColumnDataCollection &other) {
	if (other.count == 0) {
		return;
	}
	if (types != other.types) {
		throw InternalException("Attempting to combine ColumnDataCollections with mismatching types");
	}
	this->count += other.count;
	this->segments.reserve(segments.size() + other.segments.size());
	for (auto &other_seg : other.segments) {
		segments.push_back(std::move(other_seg));
	}
	Verify();
}

//===--------------------------------------------------------------------===//
// Fetch
//===--------------------------------------------------------------------===//
idx_t ColumnDataCollection::ChunkCount() const {
	idx_t chunk_count = 0;
	for (auto &segment : segments) {
		chunk_count += segment->ChunkCount();
	}
	return chunk_count;
}

void ColumnDataCollection::FetchChunk(idx_t chunk_idx, DataChunk &result) const {
	D_ASSERT(chunk_idx < ChunkCount());
	for (auto &segment : segments) {
		if (chunk_idx >= segment->ChunkCount()) {
			chunk_idx -= segment->ChunkCount();
		} else {
			segment->FetchChunk(chunk_idx, result);
			return;
		}
	}
	throw InternalException("Failed to find chunk in ColumnDataCollection");
}

//===--------------------------------------------------------------------===//
// Helpers
//===--------------------------------------------------------------------===//
void ColumnDataCollection::Verify() {
#ifdef DEBUG
	// verify counts
	idx_t total_segment_count = 0;
	for (auto &segment : segments) {
		segment->Verify();
		total_segment_count += segment->count;
	}
	D_ASSERT(total_segment_count == this->count);
#endif
}

string ColumnDataCollection::ToString() const {
	return "Column Data Collection";
}

void ColumnDataCollection::Print() const {
	Printer::Print(ToString());
}

void ColumnDataCollection::Reset() {
	count = 0;
	segments.clear();

	// Refreshes the ColumnDataAllocator to prevent holding on to allocated data unnecessarily
	allocator = make_shared<ColumnDataAllocator>(*allocator);
}

bool ColumnDataCollection::ResultEquals(const ColumnDataCollection &left, const ColumnDataCollection &right,
                                        string &error_message) {
	if (left.ColumnCount() != right.ColumnCount()) {
		error_message = "Column count mismatch";
		return false;
	}
	if (left.Count() != right.Count()) {
		error_message = "Row count mismatch";
		return false;
	}
	auto left_rows = left.GetRows();
	auto right_rows = right.GetRows();
	for (idx_t r = 0; r < left.Count(); r++) {
		for (idx_t c = 0; c < left.ColumnCount(); c++) {
			auto lvalue = left_rows.GetValue(c, r);
			auto rvalue = left_rows.GetValue(c, r);
			if (!Value::DefaultValuesAreEqual(lvalue, rvalue)) {
				error_message =
				    StringUtil::Format("%s <> %s (row: %lld, col: %lld)\n", lvalue.ToString(), rvalue.ToString(), r, c);
				return false;
			}
		}
	}
	return true;
}

const vector<unique_ptr<ColumnDataCollectionSegment>> &ColumnDataCollection::GetSegments() const {
	return segments;
}

} // namespace duckdb


namespace duckdb {

ColumnDataCollectionSegment::ColumnDataCollectionSegment(shared_ptr<ColumnDataAllocator> allocator_p,
                                                         vector<LogicalType> types_p)
    : allocator(std::move(allocator_p)), types(std::move(types_p)), count(0), heap(allocator->GetAllocator()) {
}

idx_t ColumnDataCollectionSegment::GetDataSize(idx_t type_size) {
	return AlignValue(type_size * STANDARD_VECTOR_SIZE);
}

validity_t *ColumnDataCollectionSegment::GetValidityPointer(data_ptr_t base_ptr, idx_t type_size) {
	return (validity_t *)(base_ptr + GetDataSize(type_size));
}

VectorDataIndex ColumnDataCollectionSegment::AllocateVectorInternal(const LogicalType &type, ChunkMetaData &chunk_meta,
                                                                    ChunkManagementState *chunk_state) {
	VectorMetaData meta_data;
	meta_data.count = 0;

	auto internal_type = type.InternalType();
	auto type_size = internal_type == PhysicalType::STRUCT ? 0 : GetTypeIdSize(internal_type);
	allocator->AllocateData(GetDataSize(type_size) + ValidityMask::STANDARD_MASK_SIZE, meta_data.block_id,
	                        meta_data.offset, chunk_state);
	if (allocator->GetType() == ColumnDataAllocatorType::BUFFER_MANAGER_ALLOCATOR) {
		chunk_meta.block_ids.insert(meta_data.block_id);
	}

	auto index = vector_data.size();
	vector_data.push_back(meta_data);
	return VectorDataIndex(index);
}

VectorDataIndex ColumnDataCollectionSegment::AllocateVector(const LogicalType &type, ChunkMetaData &chunk_meta,
                                                            ChunkManagementState *chunk_state,
                                                            VectorDataIndex prev_index) {
	auto index = AllocateVectorInternal(type, chunk_meta, chunk_state);
	if (prev_index.IsValid()) {
		GetVectorData(prev_index).next_data = index;
	}
	if (type.InternalType() == PhysicalType::STRUCT) {
		// initialize the struct children
		auto &child_types = StructType::GetChildTypes(type);
		auto base_child_index = ReserveChildren(child_types.size());
		for (idx_t child_idx = 0; child_idx < child_types.size(); child_idx++) {
			VectorDataIndex prev_child_index;
			if (prev_index.IsValid()) {
				prev_child_index = GetChildIndex(GetVectorData(prev_index).child_index, child_idx);
			}
			auto child_index = AllocateVector(child_types[child_idx].second, chunk_meta, chunk_state, prev_child_index);
			SetChildIndex(base_child_index, child_idx, child_index);
		}
		GetVectorData(index).child_index = base_child_index;
	}
	return index;
}

VectorDataIndex ColumnDataCollectionSegment::AllocateVector(const LogicalType &type, ChunkMetaData &chunk_meta,
                                                            ColumnDataAppendState &append_state,
                                                            VectorDataIndex prev_index) {
	return AllocateVector(type, chunk_meta, &append_state.current_chunk_state, prev_index);
}

VectorDataIndex ColumnDataCollectionSegment::AllocateStringHeap(idx_t size, ChunkMetaData &chunk_meta,
                                                                ColumnDataAppendState &append_state,
                                                                VectorDataIndex prev_index) {
	D_ASSERT(allocator->GetType() == ColumnDataAllocatorType::BUFFER_MANAGER_ALLOCATOR);
	D_ASSERT(size != 0);

	VectorMetaData meta_data;
	meta_data.count = 0;

	allocator->AllocateData(AlignValue(size), meta_data.block_id, meta_data.offset, &append_state.current_chunk_state);
	chunk_meta.block_ids.insert(meta_data.block_id);

	VectorDataIndex index(vector_data.size());
	vector_data.push_back(meta_data);

	if (prev_index.IsValid()) {
		GetVectorData(prev_index).next_data = index;
	}

	return index;
}

void ColumnDataCollectionSegment::AllocateNewChunk() {
	ChunkMetaData meta_data;
	meta_data.count = 0;
	meta_data.vector_data.reserve(types.size());
	for (idx_t i = 0; i < types.size(); i++) {
		auto vector_idx = AllocateVector(types[i], meta_data);
		meta_data.vector_data.push_back(vector_idx);
	}
	chunk_data.push_back(std::move(meta_data));
}

void ColumnDataCollectionSegment::InitializeChunkState(idx_t chunk_index, ChunkManagementState &state) {
	auto &chunk = chunk_data[chunk_index];
	allocator->InitializeChunkState(state, chunk);
}

VectorDataIndex ColumnDataCollectionSegment::GetChildIndex(VectorChildIndex index, idx_t child_entry) {
	D_ASSERT(index.IsValid());
	D_ASSERT(index.index + child_entry < child_indices.size());
	return VectorDataIndex(child_indices[index.index + child_entry]);
}

VectorChildIndex ColumnDataCollectionSegment::AddChildIndex(VectorDataIndex index) {
	auto result = child_indices.size();
	child_indices.push_back(index);
	return VectorChildIndex(result);
}

VectorChildIndex ColumnDataCollectionSegment::ReserveChildren(idx_t child_count) {
	auto result = child_indices.size();
	for (idx_t i = 0; i < child_count; i++) {
		child_indices.emplace_back();
	}
	return VectorChildIndex(result);
}

void ColumnDataCollectionSegment::SetChildIndex(VectorChildIndex base_idx, idx_t child_number, VectorDataIndex index) {
	D_ASSERT(base_idx.IsValid());
	D_ASSERT(index.IsValid());
	D_ASSERT(base_idx.index + child_number < child_indices.size());
	child_indices[base_idx.index + child_number] = index;
}

idx_t ColumnDataCollectionSegment::ReadVectorInternal(ChunkManagementState &state, VectorDataIndex vector_index,
                                                      Vector &result) {
	auto &vector_type = result.GetType();
	auto internal_type = vector_type.InternalType();
	auto type_size = GetTypeIdSize(internal_type);
	auto &vdata = GetVectorData(vector_index);

	auto base_ptr = allocator->GetDataPointer(state, vdata.block_id, vdata.offset);
	auto validity_data = GetValidityPointer(base_ptr, type_size);
	if (!vdata.next_data.IsValid() && state.properties != ColumnDataScanProperties::DISALLOW_ZERO_COPY) {
		// no next data, we can do a zero-copy read of this vector
		FlatVector::SetData(result, base_ptr);
		FlatVector::Validity(result).Initialize(validity_data);
		return vdata.count;
	}

	// the data for this vector is spread over multiple vector data entries
	// we need to copy over the data for each of the vectors
	// first figure out how many rows we need to copy by looping over all of the child vector indexes
	idx_t vector_count = 0;
	auto next_index = vector_index;
	while (next_index.IsValid()) {
		auto &current_vdata = GetVectorData(next_index);
		vector_count += current_vdata.count;
		next_index = current_vdata.next_data;
	}
	// resize the result vector
	result.Resize(0, vector_count);
	next_index = vector_index;
	// now perform the copy of each of the vectors
	auto target_data = FlatVector::GetData(result);
	auto &target_validity = FlatVector::Validity(result);
	idx_t current_offset = 0;
	while (next_index.IsValid()) {
		auto &current_vdata = GetVectorData(next_index);
		base_ptr = allocator->GetDataPointer(state, current_vdata.block_id, current_vdata.offset);
		validity_data = GetValidityPointer(base_ptr, type_size);
		if (type_size > 0) {
			memcpy(target_data + current_offset * type_size, base_ptr, current_vdata.count * type_size);
		}
		// FIXME: use bitwise operations here
		ValidityMask current_validity(validity_data);
		for (idx_t k = 0; k < current_vdata.count; k++) {
			target_validity.Set(current_offset + k, current_validity.RowIsValid(k));
		}
		current_offset += current_vdata.count;
		next_index = current_vdata.next_data;
	}
	return vector_count;
}

idx_t ColumnDataCollectionSegment::ReadVector(ChunkManagementState &state, VectorDataIndex vector_index,
                                              Vector &result) {
	auto &vector_type = result.GetType();
	auto internal_type = vector_type.InternalType();
	auto &vdata = GetVectorData(vector_index);
	if (vdata.count == 0) {
		return 0;
	}
	auto vcount = ReadVectorInternal(state, vector_index, result);
	if (internal_type == PhysicalType::LIST) {
		// list: copy child
		auto &child_vector = ListVector::GetEntry(result);
		auto child_count = ReadVector(state, GetChildIndex(vdata.child_index), child_vector);
		ListVector::SetListSize(result, child_count);
	} else if (internal_type == PhysicalType::STRUCT) {
		auto &child_vectors = StructVector::GetEntries(result);
		for (idx_t child_idx = 0; child_idx < child_vectors.size(); child_idx++) {
			auto child_count =
			    ReadVector(state, GetChildIndex(vdata.child_index, child_idx), *child_vectors[child_idx]);
			if (child_count != vcount) {
				throw InternalException("Column Data Collection: mismatch in struct child sizes");
			}
		}
	} else if (internal_type == PhysicalType::VARCHAR &&
	           allocator->GetType() == ColumnDataAllocatorType::BUFFER_MANAGER_ALLOCATOR) {
		for (auto &swizzle_segment : vdata.swizzle_data) {
			auto &string_heap_segment = GetVectorData(swizzle_segment.child_index);
			allocator->UnswizzlePointers(state, result, swizzle_segment.offset, swizzle_segment.count,
			                             string_heap_segment.block_id, string_heap_segment.offset);
		}
	}
	return vcount;
}

void ColumnDataCollectionSegment::ReadChunk(idx_t chunk_index, ChunkManagementState &state, DataChunk &chunk,
                                            const vector<column_t> &column_ids) {
	D_ASSERT(chunk.ColumnCount() == column_ids.size());
	D_ASSERT(state.properties != ColumnDataScanProperties::INVALID);
	InitializeChunkState(chunk_index, state);
	auto &chunk_meta = chunk_data[chunk_index];
	for (idx_t i = 0; i < column_ids.size(); i++) {
		auto vector_idx = column_ids[i];
		D_ASSERT(vector_idx < chunk_meta.vector_data.size());
		ReadVector(state, chunk_meta.vector_data[vector_idx], chunk.data[i]);
	}
	chunk.SetCardinality(chunk_meta.count);
}

idx_t ColumnDataCollectionSegment::ChunkCount() const {
	return chunk_data.size();
}

void ColumnDataCollectionSegment::FetchChunk(idx_t chunk_idx, DataChunk &result) {
	vector<column_t> column_ids;
	column_ids.reserve(types.size());
	for (idx_t i = 0; i < types.size(); i++) {
		column_ids.push_back(i);
	}
	FetchChunk(chunk_idx, result, column_ids);
}

void ColumnDataCollectionSegment::FetchChunk(idx_t chunk_idx, DataChunk &result, const vector<column_t> &column_ids) {
	D_ASSERT(chunk_idx < chunk_data.size());
	ChunkManagementState state;
	state.properties = ColumnDataScanProperties::DISALLOW_ZERO_COPY;
	ReadChunk(chunk_idx, state, result, column_ids);
}

void ColumnDataCollectionSegment::Verify() {
#ifdef DEBUG
	idx_t total_count = 0;
	for (idx_t i = 0; i < chunk_data.size(); i++) {
		total_count += chunk_data[i].count;
	}
	D_ASSERT(total_count == this->count);
#endif
}

} // namespace duckdb


#include <algorithm>

namespace duckdb {

using ChunkReference = ColumnDataConsumer::ChunkReference;

ChunkReference::ChunkReference(ColumnDataCollectionSegment *segment_p, uint32_t chunk_index_p)
    : segment(segment_p), chunk_index_in_segment(chunk_index_p) {
}

uint32_t ChunkReference::GetMinimumBlockID() const {
	const auto &block_ids = segment->chunk_data[chunk_index_in_segment].block_ids;
	return *std::min_element(block_ids.begin(), block_ids.end());
}

ColumnDataConsumer::ColumnDataConsumer(ColumnDataCollection &collection_p, vector<column_t> column_ids)
    : collection(collection_p), column_ids(std::move(column_ids)) {
}

void ColumnDataConsumer::InitializeScan() {
	chunk_count = collection.ChunkCount();
	current_chunk_index = 0;
	chunk_delete_index = DConstants::INVALID_INDEX;

	// Initialize chunk references and sort them, so we can scan them in a sane order, regardless of how it was created
	chunk_references.reserve(chunk_count);
	for (auto &segment : collection.GetSegments()) {
		for (idx_t chunk_index = 0; chunk_index < segment->chunk_data.size(); chunk_index++) {
			chunk_references.emplace_back(segment.get(), chunk_index);
		}
	}
	std::sort(chunk_references.begin(), chunk_references.end());
}

bool ColumnDataConsumer::AssignChunk(ColumnDataConsumerScanState &state) {
	lock_guard<mutex> guard(lock);
	if (current_chunk_index == chunk_count) {
		// All chunks have been assigned
		state.current_chunk_state.handles.clear();
		state.chunk_index = DConstants::INVALID_INDEX;
		return false;
	}
	// Assign chunk index
	state.chunk_index = current_chunk_index++;
	D_ASSERT(chunks_in_progress.find(state.chunk_index) == chunks_in_progress.end());
	chunks_in_progress.insert(state.chunk_index);
	return true;
}

void ColumnDataConsumer::ScanChunk(ColumnDataConsumerScanState &state, DataChunk &chunk) const {
	D_ASSERT(state.chunk_index < chunk_count);
	auto &chunk_ref = chunk_references[state.chunk_index];
	if (state.allocator != chunk_ref.segment->allocator.get()) {
		// Previously scanned a chunk from a different allocator, reset the handles
		state.allocator = chunk_ref.segment->allocator.get();
		state.current_chunk_state.handles.clear();
	}
	chunk_ref.segment->ReadChunk(chunk_ref.chunk_index_in_segment, state.current_chunk_state, chunk, column_ids);
}

void ColumnDataConsumer::FinishChunk(ColumnDataConsumerScanState &state) {
	D_ASSERT(state.chunk_index < chunk_count);
	idx_t delete_index_start;
	idx_t delete_index_end;
	{
		lock_guard<mutex> guard(lock);
		D_ASSERT(chunks_in_progress.find(state.chunk_index) != chunks_in_progress.end());
		delete_index_start = chunk_delete_index;
		delete_index_end = *std::min_element(chunks_in_progress.begin(), chunks_in_progress.end());
		chunks_in_progress.erase(state.chunk_index);
		chunk_delete_index = delete_index_end;
	}
	ConsumeChunks(delete_index_start, delete_index_end);
}
void ColumnDataConsumer::ConsumeChunks(idx_t delete_index_start, idx_t delete_index_end) {
	for (idx_t chunk_index = delete_index_start; chunk_index < delete_index_end; chunk_index++) {
		if (chunk_index == 0) {
			continue;
		}
		auto &prev_chunk_ref = chunk_references[chunk_index - 1];
		auto &curr_chunk_ref = chunk_references[chunk_index];
		auto prev_allocator = prev_chunk_ref.segment->allocator.get();
		auto curr_allocator = curr_chunk_ref.segment->allocator.get();
		auto prev_min_block_id = prev_chunk_ref.GetMinimumBlockID();
		auto curr_min_block_id = curr_chunk_ref.GetMinimumBlockID();
		if (prev_allocator != curr_allocator) {
			// Moved to the next allocator, delete all remaining blocks in the previous one
			for (uint32_t block_id = prev_min_block_id; block_id < prev_allocator->BlockCount(); block_id++) {
				prev_allocator->DeleteBlock(block_id);
			}
			continue;
		}
		// Same allocator, see if we can delete blocks
		for (uint32_t block_id = prev_min_block_id; block_id < curr_min_block_id; block_id++) {
			prev_allocator->DeleteBlock(block_id);
		}
	}
}

} // namespace duckdb



namespace duckdb {

bool ConflictInfo::ConflictTargetMatches(Index &index) const {
	if (only_check_unique && !index.IsUnique()) {
		// We only support checking ON CONFLICT for Unique/Primary key constraints
		return false;
	}
	if (column_ids.empty()) {
		return true;
	}
	// Check whether the column ids match
	return column_ids == index.column_id_set;
}

} // namespace duckdb





namespace duckdb {

ConflictManager::ConflictManager(VerifyExistenceType lookup_type, idx_t input_size, ConflictInfo *conflict_info)
    : lookup_type(lookup_type), input_size(input_size), conflict_info(conflict_info), conflicts(input_size, false),
      mode(ConflictManagerMode::THROW) {
}

ManagedSelection &ConflictManager::InternalSelection() {
	if (!conflicts.Initialized()) {
		conflicts.Initialize(input_size);
	}
	return conflicts;
}

const unordered_set<idx_t> &ConflictManager::InternalConflictSet() const {
	D_ASSERT(conflict_set);
	return *conflict_set;
}

Vector &ConflictManager::InternalRowIds() {
	if (!row_ids) {
		row_ids = make_unique<Vector>(LogicalType::ROW_TYPE, input_size);
	}
	return *row_ids;
}

Vector &ConflictManager::InternalIntermediate() {
	if (!intermediate_vector) {
		intermediate_vector = make_unique<Vector>(LogicalType::BOOLEAN, true, true, input_size);
	}
	return *intermediate_vector;
}

const ConflictInfo &ConflictManager::GetConflictInfo() const {
	D_ASSERT(conflict_info);
	return *conflict_info;
}

void ConflictManager::FinishLookup() {
	if (mode == ConflictManagerMode::THROW) {
		return;
	}
	if (!SingleIndexTarget()) {
		return;
	}
	if (conflicts.Count() != 0) {
		// We have recorded conflicts from the one index we're interested in
		// We set this so we don't duplicate the conflicts when there are duplicate indexes
		// that also match our conflict target
		single_index_finished = true;
	}
}

void ConflictManager::SetMode(ConflictManagerMode mode) {
	// Only allow SCAN when we have conflict info
	D_ASSERT(mode != ConflictManagerMode::SCAN || conflict_info != nullptr);
	this->mode = mode;
}

void ConflictManager::AddToConflictSet(idx_t chunk_index) {
	if (!conflict_set) {
		conflict_set = make_unique<unordered_set<idx_t>>();
	}
	auto &set = *conflict_set;
	set.insert(chunk_index);
}

void ConflictManager::AddConflictInternal(idx_t chunk_index, row_t row_id) {
	D_ASSERT(mode == ConflictManagerMode::SCAN);

	// Only when we should not throw on conflict should we get here
	D_ASSERT(!ShouldThrow(chunk_index));
	AddToConflictSet(chunk_index);
	if (SingleIndexTarget()) {
		// If we have identical indexes, only the conflicts of the first index should be recorded
		// as the other index(es) would produce the exact same conflicts anyways
		if (single_index_finished) {
			return;
		}

		// We can be more efficient because we don't need to merge conflicts of multiple indexes
		auto &selection = InternalSelection();
		auto &row_ids = InternalRowIds();
		auto data = FlatVector::GetData<row_t>(row_ids);
		data[selection.Count()] = row_id;
		selection.Append(chunk_index);
	} else {
		auto &intermediate = InternalIntermediate();
		auto data = FlatVector::GetData<bool>(intermediate);
		// Mark this index in the chunk as producing a conflict
		data[chunk_index] = true;
		if (row_id_map.empty()) {
			row_id_map.resize(input_size);
		}
		row_id_map[chunk_index] = row_id;
	}
}

bool ConflictManager::IsConflict(LookupResultType type) {
	switch (type) {
	case LookupResultType::LOOKUP_NULL: {
		if (ShouldIgnoreNulls()) {
			return false;
		}
		// If nulls are not ignored, treat this as a hit instead
		return IsConflict(LookupResultType::LOOKUP_HIT);
	}
	case LookupResultType::LOOKUP_HIT: {
		return true;
	}
	case LookupResultType::LOOKUP_MISS: {
		// FIXME: If we record a miss as a conflict when the verify type is APPEND_FK, then we can simplify the checks
		// in VerifyForeignKeyConstraint This also means we should not record a hit as a conflict when the verify type
		// is APPEND_FK
		return false;
	}
	default: {
		throw NotImplementedException("Type not implemented for LookupResultType");
	}
	}
}

bool ConflictManager::AddHit(idx_t chunk_index, row_t row_id) {
	D_ASSERT(chunk_index < input_size);
	// First check if this causes a conflict
	if (!IsConflict(LookupResultType::LOOKUP_HIT)) {
		return false;
	}

	// Then check if we should throw on a conflict
	if (ShouldThrow(chunk_index)) {
		return true;
	}
	if (mode == ConflictManagerMode::THROW) {
		// When our mode is THROW, and the chunk index is part of the previously scanned conflicts
		// then we ignore the conflict instead
		D_ASSERT(!ShouldThrow(chunk_index));
		return false;
	}
	D_ASSERT(conflict_info);
	// Because we don't throw, we need to register the conflict
	AddConflictInternal(chunk_index, row_id);
	return false;
}

bool ConflictManager::AddMiss(idx_t chunk_index) {
	D_ASSERT(chunk_index < input_size);
	return IsConflict(LookupResultType::LOOKUP_MISS);
}

bool ConflictManager::AddNull(idx_t chunk_index) {
	D_ASSERT(chunk_index < input_size);
	if (!IsConflict(LookupResultType::LOOKUP_NULL)) {
		return false;
	}
	return AddHit(chunk_index, DConstants::INVALID_INDEX);
}

bool ConflictManager::SingleIndexTarget() const {
	D_ASSERT(conflict_info);
	// We are only interested in a specific index
	return !conflict_info->column_ids.empty();
}

bool ConflictManager::ShouldThrow(idx_t chunk_index) const {
	if (mode == ConflictManagerMode::SCAN) {
		return false;
	}
	D_ASSERT(mode == ConflictManagerMode::THROW);
	if (conflict_set == nullptr) {
		// No conflicts were scanned, so this conflict is not in the set
		return true;
	}
	auto &set = InternalConflictSet();
	if (set.count(chunk_index)) {
		return false;
	}
	// None of the scanned conflicts arose from this insert tuple
	return true;
}

bool ConflictManager::ShouldIgnoreNulls() const {
	switch (lookup_type) {
	case VerifyExistenceType::APPEND:
		return true;
	case VerifyExistenceType::APPEND_FK:
		return false;
	case VerifyExistenceType::DELETE_FK:
		return true;
	default:
		throw InternalException("Type not implemented for VerifyExistenceType");
	}
}

Vector &ConflictManager::RowIds() {
	D_ASSERT(finalized);
	return *row_ids;
}

const ManagedSelection &ConflictManager::Conflicts() const {
	D_ASSERT(finalized);
	return conflicts;
}

idx_t ConflictManager::ConflictCount() const {
	return conflicts.Count();
}

void ConflictManager::Finalize() {
	D_ASSERT(!finalized);
	if (SingleIndexTarget()) {
		// Selection vector has been directly populated already, no need to finalize
		finalized = true;
		return;
	}
	finalized = true;
	if (!intermediate_vector) {
		// No conflicts were found, we're done
		return;
	}
	auto &intermediate = InternalIntermediate();
	auto data = FlatVector::GetData<bool>(intermediate);
	auto &selection = InternalSelection();
	// Create the selection vector from the encountered conflicts
	for (idx_t i = 0; i < input_size; i++) {
		if (data[i]) {
			selection.Append(i);
		}
	}
	// Now create the row_ids Vector, aligned with the selection vector
	auto &row_ids = InternalRowIds();
	auto row_id_data = FlatVector::GetData<row_t>(row_ids);

	for (idx_t i = 0; i < selection.Count(); i++) {
		D_ASSERT(!row_id_map.empty());
		auto index = selection[i];
		D_ASSERT(index < row_id_map.size());
		auto row_id = row_id_map[index];
		row_id_data[i] = row_id;
	}
	intermediate_vector.reset();
}

VerifyExistenceType ConflictManager::LookupType() const {
	return this->lookup_type;
}

void ConflictManager::SetIndexCount(idx_t count) {
	index_count = count;
}

} // namespace duckdb






















namespace duckdb {

DataChunk::DataChunk() : count(0), capacity(STANDARD_VECTOR_SIZE) {
}

DataChunk::~DataChunk() {
}

void DataChunk::InitializeEmpty(const vector<LogicalType> &types) {
	InitializeEmpty(types.begin(), types.end());
}

void DataChunk::Initialize(Allocator &allocator, const vector<LogicalType> &types, idx_t capacity_p) {
	Initialize(allocator, types.begin(), types.end(), capacity_p);
}

void DataChunk::Initialize(ClientContext &context, const vector<LogicalType> &types, idx_t capacity_p) {
	Initialize(Allocator::Get(context), types, capacity_p);
}

void DataChunk::Initialize(Allocator &allocator, vector<LogicalType>::const_iterator begin,
                           vector<LogicalType>::const_iterator end, idx_t capacity_p) {
	D_ASSERT(data.empty());                   // can only be initialized once
	D_ASSERT(std::distance(begin, end) != 0); // empty chunk not allowed
	capacity = capacity_p;
	for (; begin != end; begin++) {
		VectorCache cache(allocator, *begin, capacity);
		data.emplace_back(cache);
		vector_caches.push_back(std::move(cache));
	}
}

void DataChunk::Initialize(ClientContext &context, vector<LogicalType>::const_iterator begin,
                           vector<LogicalType>::const_iterator end, idx_t capacity_p) {
	Initialize(Allocator::Get(context), begin, end, capacity_p);
}

void DataChunk::InitializeEmpty(vector<LogicalType>::const_iterator begin, vector<LogicalType>::const_iterator end) {
	capacity = STANDARD_VECTOR_SIZE;
	D_ASSERT(data.empty());                   // can only be initialized once
	D_ASSERT(std::distance(begin, end) != 0); // empty chunk not allowed
	for (; begin != end; begin++) {
		data.emplace_back(Vector(*begin, nullptr));
	}
}

void DataChunk::Reset() {
	if (data.empty()) {
		return;
	}
	if (vector_caches.size() != data.size()) {
		throw InternalException("VectorCache and column count mismatch in DataChunk::Reset");
	}
	for (idx_t i = 0; i < ColumnCount(); i++) {
		data[i].ResetFromCache(vector_caches[i]);
	}
	capacity = STANDARD_VECTOR_SIZE;
	SetCardinality(0);
}

void DataChunk::Destroy() {
	data.clear();
	vector_caches.clear();
	capacity = 0;
	SetCardinality(0);
}

Value DataChunk::GetValue(idx_t col_idx, idx_t index) const {
	D_ASSERT(index < size());
	return data[col_idx].GetValue(index);
}

void DataChunk::SetValue(idx_t col_idx, idx_t index, const Value &val) {
	data[col_idx].SetValue(index, val);
}

bool DataChunk::AllConstant() const {
	for (auto &v : data) {
		if (v.GetVectorType() != VectorType::CONSTANT_VECTOR) {
			return false;
		}
	}
	return true;
}

void DataChunk::Reference(DataChunk &chunk) {
	D_ASSERT(chunk.ColumnCount() <= ColumnCount());
	SetCapacity(chunk);
	SetCardinality(chunk);
	for (idx_t i = 0; i < chunk.ColumnCount(); i++) {
		data[i].Reference(chunk.data[i]);
	}
}

void DataChunk::Move(DataChunk &chunk) {
	SetCardinality(chunk);
	SetCapacity(chunk);
	data = std::move(chunk.data);
	vector_caches = std::move(chunk.vector_caches);

	chunk.Destroy();
}

void DataChunk::Copy(DataChunk &other, idx_t offset) const {
	D_ASSERT(ColumnCount() == other.ColumnCount());
	D_ASSERT(other.size() == 0);

	for (idx_t i = 0; i < ColumnCount(); i++) {
		D_ASSERT(other.data[i].GetVectorType() == VectorType::FLAT_VECTOR);
		VectorOperations::Copy(data[i], other.data[i], size(), offset, 0);
	}
	other.SetCardinality(size() - offset);
}

void DataChunk::Copy(DataChunk &other, const SelectionVector &sel, const idx_t source_count, const idx_t offset) const {
	D_ASSERT(ColumnCount() == other.ColumnCount());
	D_ASSERT(other.size() == 0);
	D_ASSERT((offset + source_count) <= size());

	for (idx_t i = 0; i < ColumnCount(); i++) {
		D_ASSERT(other.data[i].GetVectorType() == VectorType::FLAT_VECTOR);
		VectorOperations::Copy(data[i], other.data[i], sel, source_count, offset, 0);
	}
	other.SetCardinality(source_count - offset);
}

void DataChunk::Split(DataChunk &other, idx_t split_idx) {
	D_ASSERT(other.size() == 0);
	D_ASSERT(other.data.empty());
	D_ASSERT(split_idx < data.size());
	const idx_t num_cols = data.size();
	for (idx_t col_idx = split_idx; col_idx < num_cols; col_idx++) {
		other.data.push_back(std::move(data[col_idx]));
		other.vector_caches.push_back(std::move(vector_caches[col_idx]));
	}
	for (idx_t col_idx = split_idx; col_idx < num_cols; col_idx++) {
		data.pop_back();
		vector_caches.pop_back();
	}
	other.SetCapacity(*this);
	other.SetCardinality(*this);
}

void DataChunk::Fuse(DataChunk &other) {
	D_ASSERT(other.size() == size());
	const idx_t num_cols = other.data.size();
	for (idx_t col_idx = 0; col_idx < num_cols; ++col_idx) {
		data.emplace_back(std::move(other.data[col_idx]));
		vector_caches.emplace_back(std::move(other.vector_caches[col_idx]));
	}
	other.Destroy();
}

void DataChunk::ReferenceColumns(DataChunk &other, const vector<column_t> &column_ids) {
	D_ASSERT(ColumnCount() == column_ids.size());
	Reset();
	for (idx_t col_idx = 0; col_idx < ColumnCount(); col_idx++) {
		auto &other_col = other.data[column_ids[col_idx]];
		auto &this_col = data[col_idx];
		D_ASSERT(other_col.GetType() == this_col.GetType());
		this_col.Reference(other_col);
	}
	SetCardinality(other.size());
}

void DataChunk::Append(const DataChunk &other, bool resize, SelectionVector *sel, idx_t sel_count) {
	idx_t new_size = sel ? size() + sel_count : size() + other.size();
	if (other.size() == 0) {
		return;
	}
	if (ColumnCount() != other.ColumnCount()) {
		throw InternalException("Column counts of appending chunk doesn't match!");
	}
	if (new_size > capacity) {
		if (resize) {
			auto new_capacity = NextPowerOfTwo(new_size);
			for (idx_t i = 0; i < ColumnCount(); i++) {
				data[i].Resize(size(), new_capacity);
			}
			capacity = new_capacity;
		} else {
			throw InternalException("Can't append chunk to other chunk without resizing");
		}
	}
	for (idx_t i = 0; i < ColumnCount(); i++) {
		D_ASSERT(data[i].GetVectorType() == VectorType::FLAT_VECTOR);
		if (sel) {
			VectorOperations::Copy(other.data[i], data[i], *sel, sel_count, 0, size());
		} else {
			VectorOperations::Copy(other.data[i], data[i], other.size(), 0, size());
		}
	}
	SetCardinality(new_size);
}

void DataChunk::Flatten() {
	for (idx_t i = 0; i < ColumnCount(); i++) {
		data[i].Flatten(size());
	}
}

vector<LogicalType> DataChunk::GetTypes() {
	vector<LogicalType> types;
	for (idx_t i = 0; i < ColumnCount(); i++) {
		types.push_back(data[i].GetType());
	}
	return types;
}

string DataChunk::ToString() const {
	string retval = "Chunk - [" + to_string(ColumnCount()) + " Columns]\n";
	for (idx_t i = 0; i < ColumnCount(); i++) {
		retval += "- " + data[i].ToString(size()) + "\n";
	}
	return retval;
}

void DataChunk::Serialize(Serializer &serializer) {
	// write the count
	serializer.Write<sel_t>(size());
	serializer.Write<idx_t>(ColumnCount());
	for (idx_t col_idx = 0; col_idx < ColumnCount(); col_idx++) {
		// write the types
		data[col_idx].GetType().Serialize(serializer);
	}
	// write the data
	for (idx_t col_idx = 0; col_idx < ColumnCount(); col_idx++) {
		data[col_idx].Serialize(size(), serializer);
	}
}

void DataChunk::Deserialize(Deserializer &source) {
	auto rows = source.Read<sel_t>();
	idx_t column_count = source.Read<idx_t>();

	vector<LogicalType> types;
	for (idx_t i = 0; i < column_count; i++) {
		types.push_back(LogicalType::Deserialize(source));
	}
	Initialize(Allocator::DefaultAllocator(), types);
	// now load the column data
	SetCardinality(rows);
	for (idx_t i = 0; i < column_count; i++) {
		data[i].Deserialize(rows, source);
	}
	Verify();
}

void DataChunk::Slice(const SelectionVector &sel_vector, idx_t count_p) {
	this->count = count_p;
	SelCache merge_cache;
	for (idx_t c = 0; c < ColumnCount(); c++) {
		data[c].Slice(sel_vector, count_p, merge_cache);
	}
}

void DataChunk::Slice(DataChunk &other, const SelectionVector &sel, idx_t count_p, idx_t col_offset) {
	D_ASSERT(other.ColumnCount() <= col_offset + ColumnCount());
	this->count = count_p;
	SelCache merge_cache;
	for (idx_t c = 0; c < other.ColumnCount(); c++) {
		if (other.data[c].GetVectorType() == VectorType::DICTIONARY_VECTOR) {
			// already a dictionary! merge the dictionaries
			data[col_offset + c].Reference(other.data[c]);
			data[col_offset + c].Slice(sel, count_p, merge_cache);
		} else {
			data[col_offset + c].Slice(other.data[c], sel, count_p);
		}
	}
}

unique_ptr<UnifiedVectorFormat[]> DataChunk::ToUnifiedFormat() {
	auto orrified_data = unique_ptr<UnifiedVectorFormat[]>(new UnifiedVectorFormat[ColumnCount()]);
	for (idx_t col_idx = 0; col_idx < ColumnCount(); col_idx++) {
		data[col_idx].ToUnifiedFormat(size(), orrified_data[col_idx]);
	}
	return orrified_data;
}

void DataChunk::Hash(Vector &result) {
	D_ASSERT(result.GetType().id() == LogicalType::HASH);
	VectorOperations::Hash(data[0], result, size());
	for (idx_t i = 1; i < ColumnCount(); i++) {
		VectorOperations::CombineHash(result, data[i], size());
	}
}

void DataChunk::Hash(vector<idx_t> &column_ids, Vector &result) {
	D_ASSERT(result.GetType().id() == LogicalType::HASH);
	D_ASSERT(column_ids.size() > 0);

	VectorOperations::Hash(data[column_ids[0]], result, size());
	for (idx_t i = 1; i < column_ids.size(); i++) {
		VectorOperations::CombineHash(result, data[column_ids[i]], size());
	}
}

void DataChunk::Verify() {
#ifdef DEBUG
	D_ASSERT(size() <= capacity);
	// verify that all vectors in this chunk have the chunk selection vector
	for (idx_t i = 0; i < ColumnCount(); i++) {
		data[i].Verify(size());
	}
#endif
}

void DataChunk::Print() {
	Printer::Print(ToString());
}

} // namespace duckdb










#include <cstring>
#include <cctype>
#include <algorithm>

namespace duckdb {

static_assert(sizeof(date_t) == sizeof(int32_t), "date_t was padded");

const char *Date::PINF = "infinity";  // NOLINT
const char *Date::NINF = "-infinity"; // NOLINT
const char *Date::EPOCH = "epoch";    // NOLINT

const string_t Date::MONTH_NAMES_ABBREVIATED[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
const string_t Date::MONTH_NAMES[] = {"January", "February", "March",     "April",   "May",      "June",
                                      "July",    "August",   "September", "October", "November", "December"};
const string_t Date::DAY_NAMES[] = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};
const string_t Date::DAY_NAMES_ABBREVIATED[] = {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};

const int32_t Date::NORMAL_DAYS[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
const int32_t Date::CUMULATIVE_DAYS[] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365};
const int32_t Date::LEAP_DAYS[] = {0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
const int32_t Date::CUMULATIVE_LEAP_DAYS[] = {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366};
const int8_t Date::MONTH_PER_DAY_OF_YEAR[] = {
    1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
    1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
    3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
    4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
    5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
    6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
    7,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
    8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,
    9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12};
const int8_t Date::LEAP_MONTH_PER_DAY_OF_YEAR[] = {
    1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
    1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
    3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
    4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
    5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
    6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
    7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
    8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,
    9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12};
const int32_t Date::CUMULATIVE_YEAR_DAYS[] = {
    0,      365,    730,    1096,   1461,   1826,   2191,   2557,   2922,   3287,   3652,   4018,   4383,   4748,
    5113,   5479,   5844,   6209,   6574,   6940,   7305,   7670,   8035,   8401,   8766,   9131,   9496,   9862,
    10227,  10592,  10957,  11323,  11688,  12053,  12418,  12784,  13149,  13514,  13879,  14245,  14610,  14975,
    15340,  15706,  16071,  16436,  16801,  17167,  17532,  17897,  18262,  18628,  18993,  19358,  19723,  20089,
    20454,  20819,  21184,  21550,  21915,  22280,  22645,  23011,  23376,  23741,  24106,  24472,  24837,  25202,
    25567,  25933,  26298,  26663,  27028,  27394,  27759,  28124,  28489,  28855,  29220,  29585,  29950,  30316,
    30681,  31046,  31411,  31777,  32142,  32507,  32872,  33238,  33603,  33968,  34333,  34699,  35064,  35429,
    35794,  36160,  36525,  36890,  37255,  37621,  37986,  38351,  38716,  39082,  39447,  39812,  40177,  40543,
    40908,  41273,  41638,  42004,  42369,  42734,  43099,  43465,  43830,  44195,  44560,  44926,  45291,  45656,
    46021,  46387,  46752,  47117,  47482,  47847,  48212,  48577,  48942,  49308,  49673,  50038,  50403,  50769,
    51134,  51499,  51864,  52230,  52595,  52960,  53325,  53691,  54056,  54421,  54786,  55152,  55517,  55882,
    56247,  56613,  56978,  57343,  57708,  58074,  58439,  58804,  59169,  59535,  59900,  60265,  60630,  60996,
    61361,  61726,  62091,  62457,  62822,  63187,  63552,  63918,  64283,  64648,  65013,  65379,  65744,  66109,
    66474,  66840,  67205,  67570,  67935,  68301,  68666,  69031,  69396,  69762,  70127,  70492,  70857,  71223,
    71588,  71953,  72318,  72684,  73049,  73414,  73779,  74145,  74510,  74875,  75240,  75606,  75971,  76336,
    76701,  77067,  77432,  77797,  78162,  78528,  78893,  79258,  79623,  79989,  80354,  80719,  81084,  81450,
    81815,  82180,  82545,  82911,  83276,  83641,  84006,  84371,  84736,  85101,  85466,  85832,  86197,  86562,
    86927,  87293,  87658,  88023,  88388,  88754,  89119,  89484,  89849,  90215,  90580,  90945,  91310,  91676,
    92041,  92406,  92771,  93137,  93502,  93867,  94232,  94598,  94963,  95328,  95693,  96059,  96424,  96789,
    97154,  97520,  97885,  98250,  98615,  98981,  99346,  99711,  100076, 100442, 100807, 101172, 101537, 101903,
    102268, 102633, 102998, 103364, 103729, 104094, 104459, 104825, 105190, 105555, 105920, 106286, 106651, 107016,
    107381, 107747, 108112, 108477, 108842, 109208, 109573, 109938, 110303, 110669, 111034, 111399, 111764, 112130,
    112495, 112860, 113225, 113591, 113956, 114321, 114686, 115052, 115417, 115782, 116147, 116513, 116878, 117243,
    117608, 117974, 118339, 118704, 119069, 119435, 119800, 120165, 120530, 120895, 121260, 121625, 121990, 122356,
    122721, 123086, 123451, 123817, 124182, 124547, 124912, 125278, 125643, 126008, 126373, 126739, 127104, 127469,
    127834, 128200, 128565, 128930, 129295, 129661, 130026, 130391, 130756, 131122, 131487, 131852, 132217, 132583,
    132948, 133313, 133678, 134044, 134409, 134774, 135139, 135505, 135870, 136235, 136600, 136966, 137331, 137696,
    138061, 138427, 138792, 139157, 139522, 139888, 140253, 140618, 140983, 141349, 141714, 142079, 142444, 142810,
    143175, 143540, 143905, 144271, 144636, 145001, 145366, 145732, 146097};

void Date::ExtractYearOffset(int32_t &n, int32_t &year, int32_t &year_offset) {
	year = Date::EPOCH_YEAR;
	// first we normalize n to be in the year range [1970, 2370]
	// since leap years repeat every 400 years, we can safely normalize just by "shifting" the CumulativeYearDays array
	while (n < 0) {
		n += Date::DAYS_PER_YEAR_INTERVAL;
		year -= Date::YEAR_INTERVAL;
	}
	while (n >= Date::DAYS_PER_YEAR_INTERVAL) {
		n -= Date::DAYS_PER_YEAR_INTERVAL;
		year += Date::YEAR_INTERVAL;
	}
	// interpolation search
	// we can find an upper bound of the year by assuming each year has 365 days
	year_offset = n / 365;
	// because of leap years we might be off by a little bit: compensate by decrementing the year offset until we find
	// our year
	while (n < Date::CUMULATIVE_YEAR_DAYS[year_offset]) {
		year_offset--;
		D_ASSERT(year_offset >= 0);
	}
	year += year_offset;
	D_ASSERT(n >= Date::CUMULATIVE_YEAR_DAYS[year_offset]);
}

void Date::Convert(date_t d, int32_t &year, int32_t &month, int32_t &day) {
	auto n = d.days;
	int32_t year_offset;
	Date::ExtractYearOffset(n, year, year_offset);

	day = n - Date::CUMULATIVE_YEAR_DAYS[year_offset];
	D_ASSERT(day >= 0 && day <= 365);

	bool is_leap_year = (Date::CUMULATIVE_YEAR_DAYS[year_offset + 1] - Date::CUMULATIVE_YEAR_DAYS[year_offset]) == 366;
	if (is_leap_year) {
		month = Date::LEAP_MONTH_PER_DAY_OF_YEAR[day];
		day -= Date::CUMULATIVE_LEAP_DAYS[month - 1];
	} else {
		month = Date::MONTH_PER_DAY_OF_YEAR[day];
		day -= Date::CUMULATIVE_DAYS[month - 1];
	}
	day++;
	D_ASSERT(day > 0 && day <= (is_leap_year ? Date::LEAP_DAYS[month] : Date::NORMAL_DAYS[month]));
	D_ASSERT(month > 0 && month <= 12);
}

bool Date::TryFromDate(int32_t year, int32_t month, int32_t day, date_t &result) {
	int32_t n = 0;
	if (!Date::IsValid(year, month, day)) {
		return false;
	}
	n += Date::IsLeapYear(year) ? Date::CUMULATIVE_LEAP_DAYS[month - 1] : Date::CUMULATIVE_DAYS[month - 1];
	n += day - 1;
	if (year < 1970) {
		int32_t diff_from_base = 1970 - year;
		int32_t year_index = 400 - (diff_from_base % 400);
		int32_t fractions = diff_from_base / 400;
		n += Date::CUMULATIVE_YEAR_DAYS[year_index];
		n -= Date::DAYS_PER_YEAR_INTERVAL;
		n -= fractions * Date::DAYS_PER_YEAR_INTERVAL;
	} else if (year >= 2370) {
		int32_t diff_from_base = year - 2370;
		int32_t year_index = diff_from_base % 400;
		int32_t fractions = diff_from_base / 400;
		n += Date::CUMULATIVE_YEAR_DAYS[year_index];
		n += Date::DAYS_PER_YEAR_INTERVAL;
		n += fractions * Date::DAYS_PER_YEAR_INTERVAL;
	} else {
		n += Date::CUMULATIVE_YEAR_DAYS[year - 1970];
	}
#ifdef DEBUG
	int32_t y, m, d;
	Date::Convert(date_t(n), y, m, d);
	D_ASSERT(year == y);
	D_ASSERT(month == m);
	D_ASSERT(day == d);
#endif
	result = date_t(n);
	return true;
}

date_t Date::FromDate(int32_t year, int32_t month, int32_t day) {
	date_t result;
	if (!Date::TryFromDate(year, month, day, result)) {
		throw ConversionException("Date out of range: %d-%d-%d", year, month, day);
	}
	return result;
}

bool Date::ParseDoubleDigit(const char *buf, idx_t len, idx_t &pos, int32_t &result) {
	if (pos < len && StringUtil::CharacterIsDigit(buf[pos])) {
		result = buf[pos++] - '0';
		if (pos < len && StringUtil::CharacterIsDigit(buf[pos])) {
			result = (buf[pos++] - '0') + result * 10;
		}
		return true;
	}
	return false;
}

static bool TryConvertDateSpecial(const char *buf, idx_t len, idx_t &pos, const char *special) {
	auto p = pos;
	for (; p < len && *special; ++p) {
		const auto s = *special++;
		if (!s || StringUtil::CharacterToLower(buf[p]) != s) {
			return false;
		}
	}
	if (*special) {
		return false;
	}
	pos = p;
	return true;
}

bool Date::TryConvertDate(const char *buf, idx_t len, idx_t &pos, date_t &result, bool &special, bool strict) {
	special = false;
	pos = 0;
	if (len == 0) {
		return false;
	}

	int32_t day = 0;
	int32_t month = -1;
	int32_t year = 0;
	bool yearneg = false;
	int sep;

	// skip leading spaces
	while (pos < len && StringUtil::CharacterIsSpace(buf[pos])) {
		pos++;
	}

	if (pos >= len) {
		return false;
	}
	if (buf[pos] == '-') {
		yearneg = true;
		pos++;
		if (pos >= len) {
			return false;
		}
	}
	if (!StringUtil::CharacterIsDigit(buf[pos])) {
		// Check for special values
		if (TryConvertDateSpecial(buf, len, pos, PINF)) {
			result = yearneg ? date_t::ninfinity() : date_t::infinity();
		} else if (TryConvertDateSpecial(buf, len, pos, EPOCH)) {
			result = date_t::epoch();
		} else {
			return false;
		}
		// skip trailing spaces - parsing must be strict here
		while (pos < len && StringUtil::CharacterIsSpace(buf[pos])) {
			pos++;
		}
		special = true;
		return pos == len;
	}
	// first parse the year
	for (; pos < len && StringUtil::CharacterIsDigit(buf[pos]); pos++) {
		if (year >= 100000000) {
			return false;
		}
		year = (buf[pos] - '0') + year * 10;
	}
	if (yearneg) {
		year = -year;
	}

	if (pos >= len) {
		return false;
	}

	// fetch the separator
	sep = buf[pos++];
	if (sep != ' ' && sep != '-' && sep != '/' && sep != '\\') {
		// invalid separator
		return false;
	}

	// parse the month
	if (!Date::ParseDoubleDigit(buf, len, pos, month)) {
		return false;
	}

	if (pos >= len) {
		return false;
	}

	if (buf[pos++] != sep) {
		return false;
	}

	if (pos >= len) {
		return false;
	}

	// now parse the day
	if (!Date::ParseDoubleDigit(buf, len, pos, day)) {
		return false;
	}

	// check for an optional trailing " (BC)""
	if (len - pos >= 5 && StringUtil::CharacterIsSpace(buf[pos]) && buf[pos + 1] == '(' &&
	    StringUtil::CharacterToLower(buf[pos + 2]) == 'b' && StringUtil::CharacterToLower(buf[pos + 3]) == 'c' &&
	    buf[pos + 4] == ')') {
		if (yearneg || year == 0) {
			return false;
		}
		year = -year + 1;
		pos += 5;
	}

	// in strict mode, check remaining string for non-space characters
	if (strict) {
		// skip trailing spaces
		while (pos < len && StringUtil::CharacterIsSpace((unsigned char)buf[pos])) {
			pos++;
		}
		// check position. if end was not reached, non-space chars remaining
		if (pos < len) {
			return false;
		}
	} else {
		// in non-strict mode, check for any direct trailing digits
		if (pos < len && StringUtil::CharacterIsDigit((unsigned char)buf[pos])) {
			return false;
		}
	}

	return Date::TryFromDate(year, month, day, result);
}

string Date::ConversionError(const string &str) {
	return StringUtil::Format("date field value out of range: \"%s\", "
	                          "expected format is (YYYY-MM-DD)",
	                          str);
}

string Date::ConversionError(string_t str) {
	return ConversionError(str.GetString());
}

date_t Date::FromCString(const char *buf, idx_t len, bool strict) {
	date_t result;
	idx_t pos;
	bool special = false;
	if (!TryConvertDate(buf, len, pos, result, special, strict)) {
		throw ConversionException(ConversionError(string(buf, len)));
	}
	return result;
}

date_t Date::FromString(const string &str, bool strict) {
	return Date::FromCString(str.c_str(), str.size(), strict);
}

string Date::ToString(date_t date) {
	// PG displays temporal infinities in lowercase,
	// but numerics in Titlecase.
	if (date == date_t::infinity()) {
		return PINF;
	} else if (date == date_t::ninfinity()) {
		return NINF;
	}
	int32_t date_units[3];
	idx_t year_length;
	bool add_bc;
	Date::Convert(date, date_units[0], date_units[1], date_units[2]);

	auto length = DateToStringCast::Length(date_units, year_length, add_bc);
	auto buffer = unique_ptr<char[]>(new char[length]);
	DateToStringCast::Format(buffer.get(), date_units, year_length, add_bc);
	return string(buffer.get(), length);
}

string Date::Format(int32_t year, int32_t month, int32_t day) {
	return ToString(Date::FromDate(year, month, day));
}

bool Date::IsLeapYear(int32_t year) {
	return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
}

bool Date::IsValid(int32_t year, int32_t month, int32_t day) {
	if (month < 1 || month > 12) {
		return false;
	}
	if (day < 1) {
		return false;
	}
	if (year <= DATE_MIN_YEAR) {
		if (year < DATE_MIN_YEAR) {
			return false;
		} else if (year == DATE_MIN_YEAR) {
			if (month < DATE_MIN_MONTH || (month == DATE_MIN_MONTH && day < DATE_MIN_DAY)) {
				return false;
			}
		}
	}
	if (year >= DATE_MAX_YEAR) {
		if (year > DATE_MAX_YEAR) {
			return false;
		} else if (year == DATE_MAX_YEAR) {
			if (month > DATE_MAX_MONTH || (month == DATE_MAX_MONTH && day > DATE_MAX_DAY)) {
				return false;
			}
		}
	}
	return Date::IsLeapYear(year) ? day <= Date::LEAP_DAYS[month] : day <= Date::NORMAL_DAYS[month];
}

int32_t Date::MonthDays(int32_t year, int32_t month) {
	D_ASSERT(month >= 1 && month <= 12);
	return Date::IsLeapYear(year) ? Date::LEAP_DAYS[month] : Date::NORMAL_DAYS[month];
}

date_t Date::EpochDaysToDate(int32_t epoch) {
	return (date_t)epoch;
}

int32_t Date::EpochDays(date_t date) {
	return date.days;
}

date_t Date::EpochToDate(int64_t epoch) {
	return date_t(epoch / Interval::SECS_PER_DAY);
}

int64_t Date::Epoch(date_t date) {
	return ((int64_t)date.days) * Interval::SECS_PER_DAY;
}

int64_t Date::EpochNanoseconds(date_t date) {
	int64_t result;
	if (!TryMultiplyOperator::Operation<int64_t, int64_t, int64_t>(date.days, Interval::MICROS_PER_DAY * 1000,
	                                                               result)) {
		throw ConversionException("Could not convert DATE (%s) to nanoseconds", Date::ToString(date));
	}
	return result;
}

int64_t Date::EpochMicroseconds(date_t date) {
	int64_t result;
	if (!TryMultiplyOperator::Operation<int64_t, int64_t, int64_t>(date.days, Interval::MICROS_PER_DAY, result)) {
		throw ConversionException("Could not convert DATE (%s) to microseconds", Date::ToString(date));
	}
	return result;
}

int32_t Date::ExtractYear(date_t d, int32_t *last_year) {
	auto n = d.days;
	// cached look up: check if year of this date is the same as the last one we looked up
	// note that this only works for years in the range [1970, 2370]
	if (n >= Date::CUMULATIVE_YEAR_DAYS[*last_year] && n < Date::CUMULATIVE_YEAR_DAYS[*last_year + 1]) {
		return Date::EPOCH_YEAR + *last_year;
	}
	int32_t year;
	Date::ExtractYearOffset(n, year, *last_year);
	return year;
}

int32_t Date::ExtractYear(timestamp_t ts, int32_t *last_year) {
	return Date::ExtractYear(Timestamp::GetDate(ts), last_year);
}

int32_t Date::ExtractYear(date_t d) {
	int32_t year, year_offset;
	Date::ExtractYearOffset(d.days, year, year_offset);
	return year;
}

int32_t Date::ExtractMonth(date_t date) {
	int32_t out_year, out_month, out_day;
	Date::Convert(date, out_year, out_month, out_day);
	return out_month;
}

int32_t Date::ExtractDay(date_t date) {
	int32_t out_year, out_month, out_day;
	Date::Convert(date, out_year, out_month, out_day);
	return out_day;
}

int32_t Date::ExtractDayOfTheYear(date_t date) {
	int32_t year, year_offset;
	Date::ExtractYearOffset(date.days, year, year_offset);
	return date.days - Date::CUMULATIVE_YEAR_DAYS[year_offset] + 1;
}

int32_t Date::ExtractISODayOfTheWeek(date_t date) {
	// date of 0 is 1970-01-01, which was a Thursday (4)
	// -7 = 4
	// -6 = 5
	// -5 = 6
	// -4 = 7
	// -3 = 1
	// -2 = 2
	// -1 = 3
	// 0  = 4
	// 1  = 5
	// 2  = 6
	// 3  = 7
	// 4  = 1
	// 5  = 2
	// 6  = 3
	// 7  = 4
	if (date.days < 0) {
		// negative date: start off at 4 and cycle downwards
		return (7 - ((-int64_t(date.days) + 3) % 7));
	} else {
		// positive date: start off at 4 and cycle upwards
		return ((int64_t(date.days) + 3) % 7) + 1;
	}
}

template <typename T>
static T PythonDivMod(const T &x, const T &y, T &r) {
	// D_ASSERT(y > 0);
	T quo = x / y;
	r = x - quo * y;
	if (r < 0) {
		--quo;
		r += y;
	}
	// D_ASSERT(0 <= r && r < y);
	return quo;
}

static date_t GetISOWeekOne(int32_t year) {
	const auto first_day = Date::FromDate(year, 1, 1); /* ord of 1/1 */
	/* 0 if 1/1 is a Monday, 1 if a Tue, etc. */
	const auto first_weekday = Date::ExtractISODayOfTheWeek(first_day) - 1;
	/* ordinal of closest Monday at or before 1/1 */
	auto week1_monday = first_day - first_weekday;

	if (first_weekday > 3) { /* if 1/1 was Fri, Sat, Sun */
		week1_monday += 7;
	}

	return week1_monday;
}

static int32_t GetISOYearWeek(const date_t date, int32_t &year) {
	int32_t month, day;
	Date::Convert(date, year, month, day);
	auto week1_monday = GetISOWeekOne(year);
	auto week = PythonDivMod((date.days - week1_monday.days), 7, day);
	if (week < 0) {
		week1_monday = GetISOWeekOne(--year);
		week = PythonDivMod((date.days - week1_monday.days), 7, day);
	} else if (week >= 52 && date >= GetISOWeekOne(year + 1)) {
		++year;
		week = 0;
	}

	return week + 1;
}

void Date::ExtractISOYearWeek(date_t date, int32_t &year, int32_t &week) {
	week = GetISOYearWeek(date, year);
}

int32_t Date::ExtractISOWeekNumber(date_t date) {
	int32_t year, week;
	ExtractISOYearWeek(date, year, week);
	return week;
}

int32_t Date::ExtractISOYearNumber(date_t date) {
	int32_t year, week;
	ExtractISOYearWeek(date, year, week);
	return year;
}

int32_t Date::ExtractWeekNumberRegular(date_t date, bool monday_first) {
	int32_t year, month, day;
	Date::Convert(date, year, month, day);
	month -= 1;
	day -= 1;
	// get the day of the year
	auto day_of_the_year =
	    (Date::IsLeapYear(year) ? Date::CUMULATIVE_LEAP_DAYS[month] : Date::CUMULATIVE_DAYS[month]) + day;
	// now figure out the first monday or sunday of the year
	// what day is January 1st?
	auto day_of_jan_first = Date::ExtractISODayOfTheWeek(Date::FromDate(year, 1, 1));
	// monday = 1, sunday = 7
	int32_t first_week_start;
	if (monday_first) {
		// have to find next "1"
		if (day_of_jan_first == 1) {
			// jan 1 is monday: starts immediately
			first_week_start = 0;
		} else {
			// jan 1 is not monday: count days until next monday
			first_week_start = 8 - day_of_jan_first;
		}
	} else {
		first_week_start = 7 - day_of_jan_first;
	}
	if (day_of_the_year < first_week_start) {
		// day occurs before first week starts: week 0
		return 0;
	}
	return ((day_of_the_year - first_week_start) / 7) + 1;
}

// Returns the date of the monday of the current week.
date_t Date::GetMondayOfCurrentWeek(date_t date) {
	int32_t dotw = Date::ExtractISODayOfTheWeek(date);
	return date - (dotw - 1);
}

} // namespace duckdb



namespace duckdb {

template <class SIGNED, class UNSIGNED>
string TemplatedDecimalToString(SIGNED value, uint8_t width, uint8_t scale) {
	auto len = DecimalToString::DecimalLength<SIGNED, UNSIGNED>(value, width, scale);
	auto data = unique_ptr<char[]>(new char[len + 1]);
	DecimalToString::FormatDecimal<SIGNED, UNSIGNED>(value, width, scale, data.get(), len);
	return string(data.get(), len);
}

string Decimal::ToString(int16_t value, uint8_t width, uint8_t scale) {
	return TemplatedDecimalToString<int16_t, uint16_t>(value, width, scale);
}

string Decimal::ToString(int32_t value, uint8_t width, uint8_t scale) {
	return TemplatedDecimalToString<int32_t, uint32_t>(value, width, scale);
}

string Decimal::ToString(int64_t value, uint8_t width, uint8_t scale) {
	return TemplatedDecimalToString<int64_t, uint64_t>(value, width, scale);
}

string Decimal::ToString(hugeint_t value, uint8_t width, uint8_t scale) {
	auto len = HugeintToStringCast::DecimalLength(value, width, scale);
	auto data = unique_ptr<char[]>(new char[len + 1]);
	HugeintToStringCast::FormatDecimal(value, width, scale, data.get(), len);
	return string(data.get(), len);
}

} // namespace duckdb






#include <functional>
#include <cmath>

namespace duckdb {

template <>
hash_t Hash(uint64_t val) {
	return murmurhash64(val);
}

template <>
hash_t Hash(int64_t val) {
	return murmurhash64((uint64_t)val);
}

template <>
hash_t Hash(hugeint_t val) {
	return murmurhash64(val.lower) ^ murmurhash64(val.upper);
}

template <class T>
struct FloatingPointEqualityTransform {
	static void OP(T &val) {
		if (val == (T)0.0) {
			// Turn negative zero into positive zero
			val = (T)0.0;
		} else if (std::isnan(val)) {
			val = std::numeric_limits<T>::quiet_NaN();
		}
	}
};

template <>
hash_t Hash(float val) {
	static_assert(sizeof(float) == sizeof(uint32_t), "");
	FloatingPointEqualityTransform<float>::OP(val);
	uint32_t uval = Load<uint32_t>((const_data_ptr_t)&val);
	return murmurhash64(uval);
}

template <>
hash_t Hash(double val) {
	static_assert(sizeof(double) == sizeof(uint64_t), "");
	FloatingPointEqualityTransform<double>::OP(val);
	uint64_t uval = Load<uint64_t>((const_data_ptr_t)&val);
	return murmurhash64(uval);
}

template <>
hash_t Hash(interval_t val) {
	return Hash(val.days) ^ Hash(val.months) ^ Hash(val.micros);
}

template <>
hash_t Hash(const char *str) {
	return Hash(str, strlen(str));
}

template <>
hash_t Hash(string_t val) {
	return Hash(val.GetDataUnsafe(), val.GetSize());
}

template <>
hash_t Hash(char *val) {
	return Hash<const char *>(val);
}

// MIT License
// Copyright (c) 2018-2021 Martin Ankerl
// https://github.com/martinus/robin-hood-hashing/blob/3.11.5/LICENSE
hash_t HashBytes(void *ptr, size_t len) noexcept {
	static constexpr uint64_t M = UINT64_C(0xc6a4a7935bd1e995);
	static constexpr uint64_t SEED = UINT64_C(0xe17a1465);
	static constexpr unsigned int R = 47;

	auto const *const data64 = static_cast<uint64_t const *>(ptr);
	uint64_t h = SEED ^ (len * M);

	size_t const n_blocks = len / 8;
	for (size_t i = 0; i < n_blocks; ++i) {
		auto k = Load<uint64_t>(reinterpret_cast<const_data_ptr_t>(data64 + i));

		k *= M;
		k ^= k >> R;
		k *= M;

		h ^= k;
		h *= M;
	}

	auto const *const data8 = reinterpret_cast<uint8_t const *>(data64 + n_blocks);
	switch (len & 7U) {
	case 7:
		h ^= static_cast<uint64_t>(data8[6]) << 48U;
		DUCKDB_EXPLICIT_FALLTHROUGH;
	case 6:
		h ^= static_cast<uint64_t>(data8[5]) << 40U;
		DUCKDB_EXPLICIT_FALLTHROUGH;
	case 5:
		h ^= static_cast<uint64_t>(data8[4]) << 32U;
		DUCKDB_EXPLICIT_FALLTHROUGH;
	case 4:
		h ^= static_cast<uint64_t>(data8[3]) << 24U;
		DUCKDB_EXPLICIT_FALLTHROUGH;
	case 3:
		h ^= static_cast<uint64_t>(data8[2]) << 16U;
		DUCKDB_EXPLICIT_FALLTHROUGH;
	case 2:
		h ^= static_cast<uint64_t>(data8[1]) << 8U;
		DUCKDB_EXPLICIT_FALLTHROUGH;
	case 1:
		h ^= static_cast<uint64_t>(data8[0]);
		h *= M;
		DUCKDB_EXPLICIT_FALLTHROUGH;
	default:
		break;
	}
	h ^= h >> R;
	h *= M;
	h ^= h >> R;
	return static_cast<hash_t>(h);
}

hash_t Hash(const char *val, size_t size) {
	return HashBytes((void *)val, size);
}

hash_t Hash(uint8_t *val, size_t size) {
	return HashBytes((void *)val, size);
}

} // namespace duckdb








#include <cmath>
#include <limits>

namespace duckdb {

//===--------------------------------------------------------------------===//
// String Conversion
//===--------------------------------------------------------------------===//
const hugeint_t Hugeint::POWERS_OF_TEN[] {
    hugeint_t(1),
    hugeint_t(10),
    hugeint_t(100),
    hugeint_t(1000),
    hugeint_t(10000),
    hugeint_t(100000),
    hugeint_t(1000000),
    hugeint_t(10000000),
    hugeint_t(100000000),
    hugeint_t(1000000000),
    hugeint_t(10000000000),
    hugeint_t(100000000000),
    hugeint_t(1000000000000),
    hugeint_t(10000000000000),
    hugeint_t(100000000000000),
    hugeint_t(1000000000000000),
    hugeint_t(10000000000000000),
    hugeint_t(100000000000000000),
    hugeint_t(1000000000000000000),
    hugeint_t(1000000000000000000) * hugeint_t(10),
    hugeint_t(1000000000000000000) * hugeint_t(100),
    hugeint_t(1000000000000000000) * hugeint_t(1000),
    hugeint_t(1000000000000000000) * hugeint_t(10000),
    hugeint_t(1000000000000000000) * hugeint_t(100000),
    hugeint_t(1000000000000000000) * hugeint_t(1000000),
    hugeint_t(1000000000000000000) * hugeint_t(10000000),
    hugeint_t(1000000000000000000) * hugeint_t(100000000),
    hugeint_t(1000000000000000000) * hugeint_t(1000000000),
    hugeint_t(1000000000000000000) * hugeint_t(10000000000),
    hugeint_t(1000000000000000000) * hugeint_t(100000000000),
    hugeint_t(1000000000000000000) * hugeint_t(1000000000000),
    hugeint_t(1000000000000000000) * hugeint_t(10000000000000),
    hugeint_t(1000000000000000000) * hugeint_t(100000000000000),
    hugeint_t(1000000000000000000) * hugeint_t(1000000000000000),
    hugeint_t(1000000000000000000) * hugeint_t(10000000000000000),
    hugeint_t(1000000000000000000) * hugeint_t(100000000000000000),
    hugeint_t(1000000000000000000) * hugeint_t(1000000000000000000),
    hugeint_t(1000000000000000000) * hugeint_t(1000000000000000000) * hugeint_t(10),
    hugeint_t(1000000000000000000) * hugeint_t(1000000000000000000) * hugeint_t(100)};

static uint8_t PositiveHugeintHighestBit(hugeint_t bits) {
	uint8_t out = 0;
	if (bits.upper) {
		out = 64;
		uint64_t up = bits.upper;
		while (up) {
			up >>= 1;
			out++;
		}
	} else {
		uint64_t low = bits.lower;
		while (low) {
			low >>= 1;
			out++;
		}
	}
	return out;
}

static bool PositiveHugeintIsBitSet(hugeint_t lhs, uint8_t bit_position) {
	if (bit_position < 64) {
		return lhs.lower & (uint64_t(1) << uint64_t(bit_position));
	} else {
		return lhs.upper & (uint64_t(1) << uint64_t(bit_position - 64));
	}
}

hugeint_t PositiveHugeintLeftShift(hugeint_t lhs, uint32_t amount) {
	D_ASSERT(amount > 0 && amount < 64);
	hugeint_t result;
	result.lower = lhs.lower << amount;
	result.upper = (lhs.upper << amount) + (lhs.lower >> (64 - amount));
	return result;
}

hugeint_t Hugeint::DivModPositive(hugeint_t lhs, uint64_t rhs, uint64_t &remainder) {
	D_ASSERT(lhs.upper >= 0);
	// DivMod code adapted from:
	// https://github.com/calccrypto/uint128_t/blob/master/uint128_t.cpp

	// initialize the result and remainder to 0
	hugeint_t div_result;
	div_result.lower = 0;
	div_result.upper = 0;
	remainder = 0;

	uint8_t highest_bit_set = PositiveHugeintHighestBit(lhs);
	// now iterate over the amount of bits that are set in the LHS
	for (uint8_t x = highest_bit_set; x > 0; x--) {
		// left-shift the current result and remainder by 1
		div_result = PositiveHugeintLeftShift(div_result, 1);
		remainder <<= 1;
		// we get the value of the bit at position X, where position 0 is the least-significant bit
		if (PositiveHugeintIsBitSet(lhs, x - 1)) {
			// increment the remainder
			remainder++;
		}
		if (remainder >= rhs) {
			// the remainder has passed the division multiplier: add one to the divide result
			remainder -= rhs;
			div_result.lower++;
			if (div_result.lower == 0) {
				// overflow
				div_result.upper++;
			}
		}
	}
	return div_result;
}

string Hugeint::ToString(hugeint_t input) {
	uint64_t remainder;
	string result;
	bool negative = input.upper < 0;
	if (negative) {
		NegateInPlace(input);
	}
	while (true) {
		if (!input.lower && !input.upper) {
			break;
		}
		input = Hugeint::DivModPositive(input, 10, remainder);
		result = string(1, '0' + remainder) + result; // NOLINT
	}
	if (result.empty()) {
		// value is zero
		return "0";
	}
	return negative ? "-" + result : result;
}

//===--------------------------------------------------------------------===//
// Multiply
//===--------------------------------------------------------------------===//
bool Hugeint::TryMultiply(hugeint_t lhs, hugeint_t rhs, hugeint_t &result) {
	bool lhs_negative = lhs.upper < 0;
	bool rhs_negative = rhs.upper < 0;
	if (lhs_negative) {
		NegateInPlace(lhs);
	}
	if (rhs_negative) {
		NegateInPlace(rhs);
	}
#if ((__GNUC__ >= 5) || defined(__clang__)) && defined(__SIZEOF_INT128__)
	__uint128_t left = __uint128_t(lhs.lower) + (__uint128_t(lhs.upper) << 64);
	__uint128_t right = __uint128_t(rhs.lower) + (__uint128_t(rhs.upper) << 64);
	__uint128_t result_i128;
	if (__builtin_mul_overflow(left, right, &result_i128)) {
		return false;
	}
	uint64_t upper = uint64_t(result_i128 >> 64);
	if (upper & 0x8000000000000000) {
		return false;
	}
	result.upper = int64_t(upper);
	result.lower = uint64_t(result_i128 & 0xffffffffffffffff);
#else
	// Multiply code adapted from:
	// https://github.com/calccrypto/uint128_t/blob/master/uint128_t.cpp

	// split values into 4 32-bit parts
	uint64_t top[4] = {uint64_t(lhs.upper) >> 32, uint64_t(lhs.upper) & 0xffffffff, lhs.lower >> 32,
	                   lhs.lower & 0xffffffff};
	uint64_t bottom[4] = {uint64_t(rhs.upper) >> 32, uint64_t(rhs.upper) & 0xffffffff, rhs.lower >> 32,
	                      rhs.lower & 0xffffffff};
	uint64_t products[4][4];

	// multiply each component of the values
	for (auto x = 0; x < 4; x++) {
		for (auto y = 0; y < 4; y++) {
			products[x][y] = top[x] * bottom[y];
		}
	}

	// if any of these products are set to a non-zero value, there is always an overflow
	if (products[0][0] || products[0][1] || products[0][2] || products[1][0] || products[2][0] || products[1][1]) {
		return false;
	}
	// if the high bits of any of these are set, there is always an overflow
	if ((products[0][3] & 0xffffffff80000000) || (products[1][2] & 0xffffffff80000000) ||
	    (products[2][1] & 0xffffffff80000000) || (products[3][0] & 0xffffffff80000000)) {
		return false;
	}

	// otherwise we merge the result of the different products together in-order

	// first row
	uint64_t fourth32 = (products[3][3] & 0xffffffff);
	uint64_t third32 = (products[3][2] & 0xffffffff) + (products[3][3] >> 32);
	uint64_t second32 = (products[3][1] & 0xffffffff) + (products[3][2] >> 32);
	uint64_t first32 = (products[3][0] & 0xffffffff) + (products[3][1] >> 32);

	// second row
	third32 += (products[2][3] & 0xffffffff);
	second32 += (products[2][2] & 0xffffffff) + (products[2][3] >> 32);
	first32 += (products[2][1] & 0xffffffff) + (products[2][2] >> 32);

	// third row
	second32 += (products[1][3] & 0xffffffff);
	first32 += (products[1][2] & 0xffffffff) + (products[1][3] >> 32);

	// fourth row
	first32 += (products[0][3] & 0xffffffff);

	// move carry to next digit
	third32 += fourth32 >> 32;
	second32 += third32 >> 32;
	first32 += second32 >> 32;

	// check if the combination of the different products resulted in an overflow
	if (first32 & 0xffffff80000000) {
		return false;
	}

	// remove carry from current digit
	fourth32 &= 0xffffffff;
	third32 &= 0xffffffff;
	second32 &= 0xffffffff;
	first32 &= 0xffffffff;

	// combine components
	result.lower = (third32 << 32) | fourth32;
	result.upper = (first32 << 32) | second32;
#endif
	if (lhs_negative ^ rhs_negative) {
		NegateInPlace(result);
	}
	return true;
}

hugeint_t Hugeint::Multiply(hugeint_t lhs, hugeint_t rhs) {
	hugeint_t result;
	if (!TryMultiply(lhs, rhs, result)) {
		throw OutOfRangeException("Overflow in HUGEINT multiplication!");
	}
	return result;
}

//===--------------------------------------------------------------------===//
// Divide
//===--------------------------------------------------------------------===//
hugeint_t Hugeint::DivMod(hugeint_t lhs, hugeint_t rhs, hugeint_t &remainder) {
	// division by zero not allowed
	D_ASSERT(!(rhs.upper == 0 && rhs.lower == 0));

	bool lhs_negative = lhs.upper < 0;
	bool rhs_negative = rhs.upper < 0;
	if (lhs_negative) {
		Hugeint::NegateInPlace(lhs);
	}
	if (rhs_negative) {
		Hugeint::NegateInPlace(rhs);
	}
	// DivMod code adapted from:
	// https://github.com/calccrypto/uint128_t/blob/master/uint128_t.cpp

	// initialize the result and remainder to 0
	hugeint_t div_result;
	div_result.lower = 0;
	div_result.upper = 0;
	remainder.lower = 0;
	remainder.upper = 0;

	uint8_t highest_bit_set = PositiveHugeintHighestBit(lhs);
	// now iterate over the amount of bits that are set in the LHS
	for (uint8_t x = highest_bit_set; x > 0; x--) {
		// left-shift the current result and remainder by 1
		div_result = PositiveHugeintLeftShift(div_result, 1);
		remainder = PositiveHugeintLeftShift(remainder, 1);

		// we get the value of the bit at position X, where position 0 is the least-significant bit
		if (PositiveHugeintIsBitSet(lhs, x - 1)) {
			// increment the remainder
			Hugeint::AddInPlace(remainder, 1);
		}
		if (Hugeint::GreaterThanEquals(remainder, rhs)) {
			// the remainder has passed the division multiplier: add one to the divide result
			remainder = Hugeint::Subtract(remainder, rhs);
			Hugeint::AddInPlace(div_result, 1);
		}
	}
	if (lhs_negative ^ rhs_negative) {
		Hugeint::NegateInPlace(div_result);
	}
	if (lhs_negative) {
		Hugeint::NegateInPlace(remainder);
	}
	return div_result;
}

hugeint_t Hugeint::Divide(hugeint_t lhs, hugeint_t rhs) {
	hugeint_t remainder;
	return Hugeint::DivMod(lhs, rhs, remainder);
}

hugeint_t Hugeint::Modulo(hugeint_t lhs, hugeint_t rhs) {
	hugeint_t remainder;
	Hugeint::DivMod(lhs, rhs, remainder);
	return remainder;
}

//===--------------------------------------------------------------------===//
// Add/Subtract
//===--------------------------------------------------------------------===//
bool Hugeint::AddInPlace(hugeint_t &lhs, hugeint_t rhs) {
	int overflow = lhs.lower + rhs.lower < lhs.lower;
	if (rhs.upper >= 0) {
		// RHS is positive: check for overflow
		if (lhs.upper > (std::numeric_limits<int64_t>::max() - rhs.upper - overflow)) {
			return false;
		}
		lhs.upper = lhs.upper + overflow + rhs.upper;
	} else {
		// RHS is negative: check for underflow
		if (lhs.upper < std::numeric_limits<int64_t>::min() - rhs.upper - overflow) {
			return false;
		}
		lhs.upper = lhs.upper + (overflow + rhs.upper);
	}
	lhs.lower += rhs.lower;
	if (lhs.upper == std::numeric_limits<int64_t>::min() && lhs.lower == 0) {
		return false;
	}
	return true;
}

bool Hugeint::SubtractInPlace(hugeint_t &lhs, hugeint_t rhs) {
	// underflow
	int underflow = lhs.lower - rhs.lower > lhs.lower;
	if (rhs.upper >= 0) {
		// RHS is positive: check for underflow
		if (lhs.upper < (std::numeric_limits<int64_t>::min() + rhs.upper + underflow)) {
			return false;
		}
		lhs.upper = (lhs.upper - rhs.upper) - underflow;
	} else {
		// RHS is negative: check for overflow
		if (lhs.upper > std::numeric_limits<int64_t>::min() &&
		    lhs.upper - 1 >= (std::numeric_limits<int64_t>::max() + rhs.upper + underflow)) {
			return false;
		}
		lhs.upper = lhs.upper - (rhs.upper + underflow);
	}
	lhs.lower -= rhs.lower;
	if (lhs.upper == std::numeric_limits<int64_t>::min() && lhs.lower == 0) {
		return false;
	}
	return true;
}

hugeint_t Hugeint::Add(hugeint_t lhs, hugeint_t rhs) {
	if (!AddInPlace(lhs, rhs)) {
		throw OutOfRangeException("Overflow in HUGEINT addition");
	}
	return lhs;
}

hugeint_t Hugeint::Subtract(hugeint_t lhs, hugeint_t rhs) {
	if (!SubtractInPlace(lhs, rhs)) {
		throw OutOfRangeException("Underflow in HUGEINT addition");
	}
	return lhs;
}

//===--------------------------------------------------------------------===//
// Hugeint Cast/Conversion
//===--------------------------------------------------------------------===//
template <class DST, bool SIGNED = true>
bool HugeintTryCastInteger(hugeint_t input, DST &result) {
	switch (input.upper) {
	case 0:
		// positive number: check if the positive number is in range
		if (input.lower <= uint64_t(NumericLimits<DST>::Maximum())) {
			result = DST(input.lower);
			return true;
		}
		break;
	case -1:
		if (!SIGNED) {
			return false;
		}
		// negative number: check if the negative number is in range
		if (input.lower >= NumericLimits<uint64_t>::Maximum() - uint64_t(NumericLimits<DST>::Maximum())) {
			result = -DST(NumericLimits<uint64_t>::Maximum() - input.lower) - 1;
			return true;
		}
		break;
	default:
		break;
	}
	return false;
}

template <>
bool Hugeint::TryCast(hugeint_t input, int8_t &result) {
	return HugeintTryCastInteger<int8_t>(input, result);
}

template <>
bool Hugeint::TryCast(hugeint_t input, int16_t &result) {
	return HugeintTryCastInteger<int16_t>(input, result);
}

template <>
bool Hugeint::TryCast(hugeint_t input, int32_t &result) {
	return HugeintTryCastInteger<int32_t>(input, result);
}

template <>
bool Hugeint::TryCast(hugeint_t input, int64_t &result) {
	return HugeintTryCastInteger<int64_t>(input, result);
}

template <>
bool Hugeint::TryCast(hugeint_t input, uint8_t &result) {
	return HugeintTryCastInteger<uint8_t, false>(input, result);
}

template <>
bool Hugeint::TryCast(hugeint_t input, uint16_t &result) {
	return HugeintTryCastInteger<uint16_t, false>(input, result);
}

template <>
bool Hugeint::TryCast(hugeint_t input, uint32_t &result) {
	return HugeintTryCastInteger<uint32_t, false>(input, result);
}

template <>
bool Hugeint::TryCast(hugeint_t input, uint64_t &result) {
	return HugeintTryCastInteger<uint64_t, false>(input, result);
}

template <>
bool Hugeint::TryCast(hugeint_t input, hugeint_t &result) {
	result = input;
	return true;
}

template <>
bool Hugeint::TryCast(hugeint_t input, float &result) {
	double dbl_result;
	Hugeint::TryCast(input, dbl_result);
	result = (float)dbl_result;
	return true;
}

template <class REAL_T>
bool CastBigintToFloating(hugeint_t input, REAL_T &result) {
	switch (input.upper) {
	case -1:
		// special case for upper = -1 to avoid rounding issues in small negative numbers
		result = -REAL_T(NumericLimits<uint64_t>::Maximum() - input.lower) - 1;
		break;
	default:
		result = REAL_T(input.lower) + REAL_T(input.upper) * REAL_T(NumericLimits<uint64_t>::Maximum());
		break;
	}
	return true;
}

template <>
bool Hugeint::TryCast(hugeint_t input, double &result) {
	return CastBigintToFloating<double>(input, result);
}

template <>
bool Hugeint::TryCast(hugeint_t input, long double &result) {
	return CastBigintToFloating<long double>(input, result);
}

template <class DST>
hugeint_t HugeintConvertInteger(DST input) {
	hugeint_t result;
	result.lower = (uint64_t)input;
	result.upper = (input < 0) * -1;
	return result;
}

template <>
bool Hugeint::TryConvert(int8_t value, hugeint_t &result) {
	result = HugeintConvertInteger<int8_t>(value);
	return true;
}

template <>
bool Hugeint::TryConvert(const char *value, hugeint_t &result) {
	auto len = strlen(value);
	string_t string_val(value, len);
	return TryCast::Operation<string_t, hugeint_t>(string_val, result, true);
}

template <>
bool Hugeint::TryConvert(int16_t value, hugeint_t &result) {
	result = HugeintConvertInteger<int16_t>(value);
	return true;
}

template <>
bool Hugeint::TryConvert(int32_t value, hugeint_t &result) {
	result = HugeintConvertInteger<int32_t>(value);
	return true;
}

template <>
bool Hugeint::TryConvert(int64_t value, hugeint_t &result) {
	result = HugeintConvertInteger<int64_t>(value);
	return true;
}
template <>
bool Hugeint::TryConvert(uint8_t value, hugeint_t &result) {
	result = HugeintConvertInteger<uint8_t>(value);
	return true;
}
template <>
bool Hugeint::TryConvert(uint16_t value, hugeint_t &result) {
	result = HugeintConvertInteger<uint16_t>(value);
	return true;
}
template <>
bool Hugeint::TryConvert(uint32_t value, hugeint_t &result) {
	result = HugeintConvertInteger<uint32_t>(value);
	return true;
}
template <>
bool Hugeint::TryConvert(uint64_t value, hugeint_t &result) {
	result = HugeintConvertInteger<uint64_t>(value);
	return true;
}

template <>
bool Hugeint::TryConvert(hugeint_t value, hugeint_t &result) {
	result = value;
	return true;
}

template <>
bool Hugeint::TryConvert(float value, hugeint_t &result) {
	return Hugeint::TryConvert(double(value), result);
}

template <class REAL_T>
bool ConvertFloatingToBigint(REAL_T value, hugeint_t &result) {
	if (!Value::IsFinite<REAL_T>(value)) {
		return false;
	}
	if (value <= -170141183460469231731687303715884105728.0 || value >= 170141183460469231731687303715884105727.0) {
		return false;
	}
	bool negative = value < 0;
	if (negative) {
		value = -value;
	}
	result.lower = (uint64_t)fmod(value, REAL_T(NumericLimits<uint64_t>::Maximum()));
	result.upper = (uint64_t)(value / REAL_T(NumericLimits<uint64_t>::Maximum()));
	if (negative) {
		Hugeint::NegateInPlace(result);
	}
	return true;
}

template <>
bool Hugeint::TryConvert(double value, hugeint_t &result) {
	return ConvertFloatingToBigint<double>(value, result);
}

template <>
bool Hugeint::TryConvert(long double value, hugeint_t &result) {
	return ConvertFloatingToBigint<long double>(value, result);
}

//===--------------------------------------------------------------------===//
// hugeint_t operators
//===--------------------------------------------------------------------===//
hugeint_t::hugeint_t(int64_t value) {
	auto result = Hugeint::Convert(value);
	this->lower = result.lower;
	this->upper = result.upper;
}

bool hugeint_t::operator==(const hugeint_t &rhs) const {
	return Hugeint::Equals(*this, rhs);
}

bool hugeint_t::operator!=(const hugeint_t &rhs) const {
	return Hugeint::NotEquals(*this, rhs);
}

bool hugeint_t::operator<(const hugeint_t &rhs) const {
	return Hugeint::LessThan(*this, rhs);
}

bool hugeint_t::operator<=(const hugeint_t &rhs) const {
	return Hugeint::LessThanEquals(*this, rhs);
}

bool hugeint_t::operator>(const hugeint_t &rhs) const {
	return Hugeint::GreaterThan(*this, rhs);
}

bool hugeint_t::operator>=(const hugeint_t &rhs) const {
	return Hugeint::GreaterThanEquals(*this, rhs);
}

hugeint_t hugeint_t::operator+(const hugeint_t &rhs) const {
	return Hugeint::Add(*this, rhs);
}

hugeint_t hugeint_t::operator-(const hugeint_t &rhs) const {
	return Hugeint::Subtract(*this, rhs);
}

hugeint_t hugeint_t::operator*(const hugeint_t &rhs) const {
	return Hugeint::Multiply(*this, rhs);
}

hugeint_t hugeint_t::operator/(const hugeint_t &rhs) const {
	return Hugeint::Divide(*this, rhs);
}

hugeint_t hugeint_t::operator%(const hugeint_t &rhs) const {
	return Hugeint::Modulo(*this, rhs);
}

hugeint_t hugeint_t::operator-() const {
	return Hugeint::Negate(*this);
}

hugeint_t hugeint_t::operator>>(const hugeint_t &rhs) const {
	hugeint_t result;
	uint64_t shift = rhs.lower;
	if (rhs.upper != 0 || shift >= 128) {
		return hugeint_t(0);
	} else if (shift == 0) {
		return *this;
	} else if (shift == 64) {
		result.upper = (upper < 0) ? -1 : 0;
		result.lower = upper;
	} else if (shift < 64) {
		// perform lower shift in unsigned integer, and mask away the most significant bit
		result.lower = (uint64_t(upper) << (64 - shift)) | (lower >> shift);
		result.upper = upper >> shift;
	} else {
		D_ASSERT(shift < 128);
		result.lower = upper >> (shift - 64);
		result.upper = (upper < 0) ? -1 : 0;
	}
	return result;
}

hugeint_t hugeint_t::operator<<(const hugeint_t &rhs) const {
	if (upper < 0) {
		return hugeint_t(0);
	}
	hugeint_t result;
	uint64_t shift = rhs.lower;
	if (rhs.upper != 0 || shift >= 128) {
		return hugeint_t(0);
	} else if (shift == 64) {
		result.upper = lower;
		result.lower = 0;
	} else if (shift == 0) {
		return *this;
	} else if (shift < 64) {
		// perform upper shift in unsigned integer, and mask away the most significant bit
		uint64_t upper_shift = ((uint64_t(upper) << shift) + (lower >> (64 - shift))) & 0x7FFFFFFFFFFFFFFF;
		result.lower = lower << shift;
		result.upper = upper_shift;
	} else {
		D_ASSERT(shift < 128);
		result.lower = 0;
		result.upper = (lower << (shift - 64)) & 0x7FFFFFFFFFFFFFFF;
	}
	return result;
}

hugeint_t hugeint_t::operator&(const hugeint_t &rhs) const {
	hugeint_t result;
	result.lower = lower & rhs.lower;
	result.upper = upper & rhs.upper;
	return result;
}

hugeint_t hugeint_t::operator|(const hugeint_t &rhs) const {
	hugeint_t result;
	result.lower = lower | rhs.lower;
	result.upper = upper | rhs.upper;
	return result;
}

hugeint_t hugeint_t::operator^(const hugeint_t &rhs) const {
	hugeint_t result;
	result.lower = lower ^ rhs.lower;
	result.upper = upper ^ rhs.upper;
	return result;
}

hugeint_t hugeint_t::operator~() const {
	hugeint_t result;
	result.lower = ~lower;
	result.upper = ~upper;
	return result;
}

hugeint_t &hugeint_t::operator+=(const hugeint_t &rhs) {
	Hugeint::AddInPlace(*this, rhs);
	return *this;
}
hugeint_t &hugeint_t::operator-=(const hugeint_t &rhs) {
	Hugeint::SubtractInPlace(*this, rhs);
	return *this;
}
hugeint_t &hugeint_t::operator*=(const hugeint_t &rhs) {
	*this = Hugeint::Multiply(*this, rhs);
	return *this;
}
hugeint_t &hugeint_t::operator/=(const hugeint_t &rhs) {
	*this = Hugeint::Divide(*this, rhs);
	return *this;
}
hugeint_t &hugeint_t::operator%=(const hugeint_t &rhs) {
	*this = Hugeint::Modulo(*this, rhs);
	return *this;
}
hugeint_t &hugeint_t::operator>>=(const hugeint_t &rhs) {
	*this = *this >> rhs;
	return *this;
}
hugeint_t &hugeint_t::operator<<=(const hugeint_t &rhs) {
	*this = *this << rhs;
	return *this;
}
hugeint_t &hugeint_t::operator&=(const hugeint_t &rhs) {
	lower &= rhs.lower;
	upper &= rhs.upper;
	return *this;
}
hugeint_t &hugeint_t::operator|=(const hugeint_t &rhs) {
	lower |= rhs.lower;
	upper |= rhs.upper;
	return *this;
}
hugeint_t &hugeint_t::operator^=(const hugeint_t &rhs) {
	lower ^= rhs.lower;
	upper ^= rhs.upper;
	return *this;
}

string hugeint_t::ToString() const {
	return Hugeint::ToString(*this);
}

} // namespace duckdb






namespace duckdb {

HyperLogLog::HyperLogLog() : hll(nullptr) {
	hll = duckdb_hll::hll_create();
	// Insert into a dense hll can be vectorized, sparse cannot, so we immediately convert
	duckdb_hll::hllSparseToDense((duckdb_hll::robj *)hll);
}

HyperLogLog::HyperLogLog(void *hll) : hll(hll) {
}

HyperLogLog::~HyperLogLog() {
	duckdb_hll::hll_destroy((duckdb_hll::robj *)hll);
}

void HyperLogLog::Add(data_ptr_t element, idx_t size) {
	if (duckdb_hll::hll_add((duckdb_hll::robj *)hll, element, size) == HLL_C_ERR) {
		throw InternalException("Could not add to HLL?");
	}
}

idx_t HyperLogLog::Count() const {
	// exception from size_t ban
	size_t result;

	if (duckdb_hll::hll_count((duckdb_hll::robj *)hll, &result) != HLL_C_OK) {
		throw InternalException("Could not count HLL?");
	}
	return result;
}

unique_ptr<HyperLogLog> HyperLogLog::Merge(HyperLogLog &other) {
	duckdb_hll::robj *hlls[2];
	hlls[0] = (duckdb_hll::robj *)hll;
	hlls[1] = (duckdb_hll::robj *)other.hll;
	auto new_hll = duckdb_hll::hll_merge(hlls, 2);
	if (!new_hll) {
		throw InternalException("Could not merge HLLs");
	}
	return unique_ptr<HyperLogLog>(new HyperLogLog((void *)new_hll));
}

HyperLogLog *HyperLogLog::MergePointer(HyperLogLog &other) {
	duckdb_hll::robj *hlls[2];
	hlls[0] = (duckdb_hll::robj *)hll;
	hlls[1] = (duckdb_hll::robj *)other.hll;
	auto new_hll = duckdb_hll::hll_merge(hlls, 2);
	if (!new_hll) {
		throw Exception("Could not merge HLLs");
	}
	return new HyperLogLog((void *)new_hll);
}

unique_ptr<HyperLogLog> HyperLogLog::Merge(HyperLogLog logs[], idx_t count) {
	auto hlls_uptr = unique_ptr<duckdb_hll::robj *[]> {
		new duckdb_hll::robj *[count]
	};
	auto hlls = hlls_uptr.get();
	for (idx_t i = 0; i < count; i++) {
		hlls[i] = (duckdb_hll::robj *)logs[i].hll;
	}
	auto new_hll = duckdb_hll::hll_merge(hlls, count);
	if (!new_hll) {
		throw InternalException("Could not merge HLLs");
	}
	return unique_ptr<HyperLogLog>(new HyperLogLog((void *)new_hll));
}

idx_t HyperLogLog::GetSize() {
	return duckdb_hll::get_size();
}

data_ptr_t HyperLogLog::GetPtr() const {
	return (data_ptr_t)((duckdb_hll::robj *)hll)->ptr;
}

unique_ptr<HyperLogLog> HyperLogLog::Copy() {
	auto result = make_unique<HyperLogLog>();
	lock_guard<mutex> guard(lock);
	memcpy(result->GetPtr(), GetPtr(), GetSize());
	D_ASSERT(result->Count() == Count());
	return result;
}

void HyperLogLog::Serialize(FieldWriter &writer) const {
	writer.WriteField<HLLStorageType>(HLLStorageType::UNCOMPRESSED);
	writer.WriteBlob(GetPtr(), GetSize());
}

unique_ptr<HyperLogLog> HyperLogLog::Deserialize(FieldReader &reader) {
	auto result = make_unique<HyperLogLog>();
	auto storage_type = reader.ReadRequired<HLLStorageType>();
	switch (storage_type) {
	case HLLStorageType::UNCOMPRESSED:
		reader.ReadBlob(result->GetPtr(), GetSize());
		break;
	default:
		throw SerializationException("Unknown HyperLogLog storage type!");
	}
	return result;
}

//===--------------------------------------------------------------------===//
// Vectorized HLL implementation
//===--------------------------------------------------------------------===//
//! Taken from https://nullprogram.com/blog/2018/07/31/
template <class T>
inline uint64_t TemplatedHash(const T &elem) {
	uint64_t x = elem;
	x ^= x >> 30;
	x *= UINT64_C(0xbf58476d1ce4e5b9);
	x ^= x >> 27;
	x *= UINT64_C(0x94d049bb133111eb);
	x ^= x >> 31;
	return x;
}

template <>
inline uint64_t TemplatedHash(const hugeint_t &elem) {
	return TemplatedHash<uint64_t>(Load<uint64_t>((data_ptr_t)&elem.upper)) ^ TemplatedHash<uint64_t>(elem.lower);
}

template <idx_t rest>
inline void CreateIntegerRecursive(const data_ptr_t &data, uint64_t &x) {
	x ^= (uint64_t)data[rest - 1] << ((rest - 1) * 8);
	return CreateIntegerRecursive<rest - 1>(data, x);
}

template <>
inline void CreateIntegerRecursive<1>(const data_ptr_t &data, uint64_t &x) {
	x ^= (uint64_t)data[0];
}

inline uint64_t HashOtherSize(const data_ptr_t &data, const idx_t &len) {
	uint64_t x = 0;
	switch (len & 7) {
	case 7:
		CreateIntegerRecursive<7>(data, x);
		break;
	case 6:
		CreateIntegerRecursive<6>(data, x);
		break;
	case 5:
		CreateIntegerRecursive<5>(data, x);
		break;
	case 4:
		CreateIntegerRecursive<4>(data, x);
		break;
	case 3:
		CreateIntegerRecursive<3>(data, x);
		break;
	case 2:
		CreateIntegerRecursive<2>(data, x);
		break;
	case 1:
		CreateIntegerRecursive<1>(data, x);
		break;
	case 0:
		break;
	}
	return TemplatedHash<uint64_t>(x);
}

template <>
inline uint64_t TemplatedHash(const string_t &elem) {
	data_ptr_t data = (data_ptr_t)elem.GetDataUnsafe();
	const auto &len = elem.GetSize();
	uint64_t h = 0;
	for (idx_t i = 0; i + sizeof(uint64_t) <= len; i += sizeof(uint64_t)) {
		h ^= TemplatedHash<uint64_t>(Load<uint64_t>(data));
		data += sizeof(uint64_t);
	}
	switch (len & (sizeof(uint64_t) - 1)) {
	case 4:
		h ^= TemplatedHash<uint32_t>(Load<uint32_t>(data));
		break;
	case 2:
		h ^= TemplatedHash<uint16_t>(Load<uint16_t>(data));
		break;
	case 1:
		h ^= TemplatedHash<uint8_t>(Load<uint8_t>(data));
		break;
	default:
		h ^= HashOtherSize(data, len);
	}
	return h;
}

template <class T>
void TemplatedComputeHashes(UnifiedVectorFormat &vdata, const idx_t &count, uint64_t hashes[]) {
	T *data = (T *)vdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto idx = vdata.sel->get_index(i);
		if (vdata.validity.RowIsValid(idx)) {
			hashes[i] = TemplatedHash<T>(data[idx]);
		} else {
			hashes[i] = 0;
		}
	}
}

static void ComputeHashes(UnifiedVectorFormat &vdata, const LogicalType &type, uint64_t hashes[], idx_t count) {
	switch (type.InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
	case PhysicalType::UINT8:
		return TemplatedComputeHashes<uint8_t>(vdata, count, hashes);
	case PhysicalType::INT16:
	case PhysicalType::UINT16:
		return TemplatedComputeHashes<uint16_t>(vdata, count, hashes);
	case PhysicalType::INT32:
	case PhysicalType::UINT32:
	case PhysicalType::FLOAT:
		return TemplatedComputeHashes<uint32_t>(vdata, count, hashes);
	case PhysicalType::INT64:
	case PhysicalType::UINT64:
	case PhysicalType::DOUBLE:
		return TemplatedComputeHashes<uint64_t>(vdata, count, hashes);
	case PhysicalType::INT128:
	case PhysicalType::INTERVAL:
		static_assert(sizeof(hugeint_t) == sizeof(interval_t), "ComputeHashes assumes these are the same size!");
		return TemplatedComputeHashes<hugeint_t>(vdata, count, hashes);
	case PhysicalType::VARCHAR:
		return TemplatedComputeHashes<string_t>(vdata, count, hashes);
	default:
		throw InternalException("Unimplemented type for HyperLogLog::ComputeHashes");
	}
}

//! Taken from https://stackoverflow.com/a/72088344
static inline uint8_t CountTrailingZeros(uint64_t &x) {
	static constexpr const uint64_t DEBRUIJN = 0x03f79d71b4cb0a89;
	static constexpr const uint8_t LOOKUP[] = {0,  47, 1,  56, 48, 27, 2,  60, 57, 49, 41, 37, 28, 16, 3,  61,
	                                           54, 58, 35, 52, 50, 42, 21, 44, 38, 32, 29, 23, 17, 11, 4,  62,
	                                           46, 55, 26, 59, 40, 36, 15, 53, 34, 51, 20, 43, 31, 22, 10, 45,
	                                           25, 39, 14, 33, 19, 30, 9,  24, 13, 18, 8,  12, 7,  6,  5,  63};
	return LOOKUP[(DEBRUIJN * (x ^ (x - 1))) >> 58];
}

static inline void ComputeIndexAndCount(uint64_t &hash, uint8_t &prefix) {
	uint64_t index = hash & ((1 << 12) - 1); /* Register index. */
	hash >>= 12;                             /* Remove bits used to address the register. */
	hash |= ((uint64_t)1 << (64 - 12));      /* Make sure the count will be <= Q+1. */

	prefix = CountTrailingZeros(hash) + 1; /* Add 1 since we count the "00000...1" pattern. */
	hash = index;
}

void HyperLogLog::ProcessEntries(UnifiedVectorFormat &vdata, const LogicalType &type, uint64_t hashes[],
                                 uint8_t counts[], idx_t count) {
	ComputeHashes(vdata, type, hashes, count);
	for (idx_t i = 0; i < count; i++) {
		ComputeIndexAndCount(hashes[i], counts[i]);
	}
}

void HyperLogLog::AddToLogs(UnifiedVectorFormat &vdata, idx_t count, uint64_t indices[], uint8_t counts[],
                            HyperLogLog **logs[], const SelectionVector *log_sel) {
	AddToLogsInternal(vdata, count, indices, counts, (void ****)logs, log_sel);
}

void HyperLogLog::AddToLog(UnifiedVectorFormat &vdata, idx_t count, uint64_t indices[], uint8_t counts[]) {
	lock_guard<mutex> guard(lock);
	AddToSingleLogInternal(vdata, count, indices, counts, hll);
}

} // namespace duckdb













namespace duckdb {

bool Interval::FromString(const string &str, interval_t &result) {
	string error_message;
	return Interval::FromCString(str.c_str(), str.size(), result, &error_message, false);
}

template <class T>
void IntervalTryAddition(T &target, int64_t input, int64_t multiplier) {
	int64_t addition;
	if (!TryMultiplyOperator::Operation<int64_t, int64_t, int64_t>(input, multiplier, addition)) {
		throw OutOfRangeException("interval value is out of range");
	}
	T addition_base = Cast::Operation<int64_t, T>(addition);
	if (!TryAddOperator::Operation<T, T, T>(target, addition_base, target)) {
		throw OutOfRangeException("interval value is out of range");
	}
}

bool Interval::FromCString(const char *str, idx_t len, interval_t &result, string *error_message, bool strict) {
	idx_t pos = 0;
	idx_t start_pos;
	bool negative;
	bool found_any = false;
	int64_t number;
	DatePartSpecifier specifier;
	string specifier_str;

	result.days = 0;
	result.micros = 0;
	result.months = 0;

	if (len == 0) {
		return false;
	}

	switch (str[pos]) {
	case '@':
		pos++;
		goto standard_interval;
	case 'P':
	case 'p':
		pos++;
		goto posix_interval;
	default:
		goto standard_interval;
	}
standard_interval:
	// start parsing a standard interval (e.g. 2 years 3 months...)
	for (; pos < len; pos++) {
		char c = str[pos];
		if (c == ' ' || c == '\t' || c == '\n') {
			// skip spaces
			continue;
		} else if (c >= '0' && c <= '9') {
			// start parsing a positive number
			negative = false;
			goto interval_parse_number;
		} else if (c == '-') {
			// negative number
			negative = true;
			pos++;
			goto interval_parse_number;
		} else if (c == 'a' || c == 'A') {
			// parse the word "ago" as the final specifier
			goto interval_parse_ago;
		} else {
			// unrecognized character, expected a number or end of string
			return false;
		}
	}
	goto end_of_string;
interval_parse_number:
	start_pos = pos;
	for (; pos < len; pos++) {
		char c = str[pos];
		if (c >= '0' && c <= '9') {
			// the number continues
			continue;
		} else if (c == ':') {
			// colon: we are parsing a time
			goto interval_parse_time;
		} else {
			if (pos == start_pos) {
				return false;
			}
			// finished the number, parse it from the string
			string_t nr_string(str + start_pos, pos - start_pos);
			number = Cast::Operation<string_t, int64_t>(nr_string);
			if (negative) {
				number = -number;
			}
			goto interval_parse_identifier;
		}
	}
	goto end_of_string;
interval_parse_time : {
	// parse the remainder of the time as a Time type
	dtime_t time;
	idx_t pos;
	if (!Time::TryConvertTime(str + start_pos, len - start_pos, pos, time)) {
		return false;
	}
	result.micros += time.micros;
	found_any = true;
	goto end_of_string;
}
interval_parse_identifier:
	for (; pos < len; pos++) {
		char c = str[pos];
		if (c == ' ' || c == '\t' || c == '\n') {
			// skip spaces at the start
			continue;
		} else {
			break;
		}
	}
	// now parse the identifier
	start_pos = pos;
	for (; pos < len; pos++) {
		char c = str[pos];
		if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
			// keep parsing the string
			continue;
		} else {
			break;
		}
	}
	specifier_str = string(str + start_pos, pos - start_pos);
	if (!TryGetDatePartSpecifier(specifier_str, specifier)) {
		HandleCastError::AssignError(StringUtil::Format("extract specifier \"%s\" not recognized", specifier_str),
		                             error_message);
		return false;
	}
	// add the specifier to the interval
	switch (specifier) {
	case DatePartSpecifier::MILLENNIUM:
		IntervalTryAddition<int32_t>(result.months, number, MONTHS_PER_MILLENIUM);
		break;
	case DatePartSpecifier::CENTURY:
		IntervalTryAddition<int32_t>(result.months, number, MONTHS_PER_CENTURY);
		break;
	case DatePartSpecifier::DECADE:
		IntervalTryAddition<int32_t>(result.months, number, MONTHS_PER_DECADE);
		break;
	case DatePartSpecifier::YEAR:
		IntervalTryAddition<int32_t>(result.months, number, MONTHS_PER_YEAR);
		break;
	case DatePartSpecifier::QUARTER:
		IntervalTryAddition<int32_t>(result.months, number, MONTHS_PER_QUARTER);
		break;
	case DatePartSpecifier::MONTH:
		IntervalTryAddition<int32_t>(result.months, number, 1);
		break;
	case DatePartSpecifier::DAY:
		IntervalTryAddition<int32_t>(result.days, number, 1);
		break;
	case DatePartSpecifier::WEEK:
		IntervalTryAddition<int32_t>(result.days, number, DAYS_PER_WEEK);
		break;
	case DatePartSpecifier::MICROSECONDS:
		IntervalTryAddition<int64_t>(result.micros, number, 1);
		break;
	case DatePartSpecifier::MILLISECONDS:
		IntervalTryAddition<int64_t>(result.micros, number, MICROS_PER_MSEC);
		break;
	case DatePartSpecifier::SECOND:
		IntervalTryAddition<int64_t>(result.micros, number, MICROS_PER_SEC);
		break;
	case DatePartSpecifier::MINUTE:
		IntervalTryAddition<int64_t>(result.micros, number, MICROS_PER_MINUTE);
		break;
	case DatePartSpecifier::HOUR:
		IntervalTryAddition<int64_t>(result.micros, number, MICROS_PER_HOUR);
		break;
	default:
		HandleCastError::AssignError(
		    StringUtil::Format("extract specifier \"%s\" not supported for interval", specifier_str), error_message);
		return false;
	}
	found_any = true;
	goto standard_interval;
interval_parse_ago:
	D_ASSERT(str[pos] == 'a' || str[pos] == 'A');
	// parse the "ago" string at the end of the interval
	if (len - pos < 3) {
		return false;
	}
	pos++;
	if (!(str[pos] == 'g' || str[pos] == 'G')) {
		return false;
	}
	pos++;
	if (!(str[pos] == 'o' || str[pos] == 'O')) {
		return false;
	}
	pos++;
	// parse any trailing whitespace
	for (; pos < len; pos++) {
		char c = str[pos];
		if (c == ' ' || c == '\t' || c == '\n') {
			continue;
		} else {
			return false;
		}
	}
	// invert all the values
	result.months = -result.months;
	result.days = -result.days;
	result.micros = -result.micros;
	goto end_of_string;
end_of_string:
	if (!found_any) {
		// end of string and no identifiers were found: cannot convert empty interval
		return false;
	}
	return true;
posix_interval:
	return false;
}

string Interval::ToString(const interval_t &interval) {
	char buffer[70];
	idx_t length = IntervalToStringCast::Format(interval, buffer);
	return string(buffer, length);
}

int64_t Interval::GetMilli(const interval_t &val) {
	int64_t milli_month, milli_day, milli;
	if (!TryMultiplyOperator::Operation((int64_t)val.months, Interval::MICROS_PER_MONTH / 1000, milli_month)) {
		throw ConversionException("Could not convert Interval to Milliseconds");
	}
	if (!TryMultiplyOperator::Operation((int64_t)val.days, Interval::MICROS_PER_DAY / 1000, milli_day)) {
		throw ConversionException("Could not convert Interval to Milliseconds");
	}
	milli = val.micros / 1000;
	if (!TryAddOperator::Operation<int64_t, int64_t, int64_t>(milli, milli_month, milli)) {
		throw ConversionException("Could not convert Interval to Milliseconds");
	}
	if (!TryAddOperator::Operation<int64_t, int64_t, int64_t>(milli, milli_day, milli)) {
		throw ConversionException("Could not convert Interval to Milliseconds");
	}
	return milli;
}

int64_t Interval::GetMicro(const interval_t &val) {
	int64_t micro_month, micro_day, micro_total;
	micro_total = val.micros;
	if (!TryMultiplyOperator::Operation((int64_t)val.months, MICROS_PER_MONTH, micro_month)) {
		throw ConversionException("Could not convert Month to Microseconds");
	}
	if (!TryMultiplyOperator::Operation((int64_t)val.days, MICROS_PER_DAY, micro_day)) {
		throw ConversionException("Could not convert Day to Microseconds");
	}
	if (!TryAddOperator::Operation<int64_t, int64_t, int64_t>(micro_total, micro_month, micro_total)) {
		throw ConversionException("Could not convert Interval to Microseconds");
	}
	if (!TryAddOperator::Operation<int64_t, int64_t, int64_t>(micro_total, micro_day, micro_total)) {
		throw ConversionException("Could not convert Interval to Microseconds");
	}

	return micro_total;
}

int64_t Interval::GetNanoseconds(const interval_t &val) {
	int64_t nano;
	const auto micro_total = GetMicro(val);
	if (!TryMultiplyOperator::Operation(micro_total, NANOS_PER_MICRO, nano)) {
		throw ConversionException("Could not convert Interval to Nanoseconds");
	}

	return nano;
}

interval_t Interval::GetAge(timestamp_t timestamp_1, timestamp_t timestamp_2) {
	D_ASSERT(Timestamp::IsFinite(timestamp_1) && Timestamp::IsFinite(timestamp_2));
	date_t date1, date2;
	dtime_t time1, time2;

	Timestamp::Convert(timestamp_1, date1, time1);
	Timestamp::Convert(timestamp_2, date2, time2);

	// and from date extract the years, months and days
	int32_t year1, month1, day1;
	int32_t year2, month2, day2;
	Date::Convert(date1, year1, month1, day1);
	Date::Convert(date2, year2, month2, day2);
	// finally perform the differences
	auto year_diff = year1 - year2;
	auto month_diff = month1 - month2;
	auto day_diff = day1 - day2;

	// and from time extract hours, minutes, seconds and milliseconds
	int32_t hour1, min1, sec1, micros1;
	int32_t hour2, min2, sec2, micros2;
	Time::Convert(time1, hour1, min1, sec1, micros1);
	Time::Convert(time2, hour2, min2, sec2, micros2);
	// finally perform the differences
	auto hour_diff = hour1 - hour2;
	auto min_diff = min1 - min2;
	auto sec_diff = sec1 - sec2;
	auto micros_diff = micros1 - micros2;

	// flip sign if necessary
	bool sign_flipped = false;
	if (timestamp_1 < timestamp_2) {
		year_diff = -year_diff;
		month_diff = -month_diff;
		day_diff = -day_diff;
		hour_diff = -hour_diff;
		min_diff = -min_diff;
		sec_diff = -sec_diff;
		micros_diff = -micros_diff;
		sign_flipped = true;
	}
	// now propagate any negative field into the next higher field
	while (micros_diff < 0) {
		micros_diff += MICROS_PER_SEC;
		sec_diff--;
	}
	while (sec_diff < 0) {
		sec_diff += SECS_PER_MINUTE;
		min_diff--;
	}
	while (min_diff < 0) {
		min_diff += MINS_PER_HOUR;
		hour_diff--;
	}
	while (hour_diff < 0) {
		hour_diff += HOURS_PER_DAY;
		day_diff--;
	}
	while (day_diff < 0) {
		if (timestamp_1 < timestamp_2) {
			day_diff += Date::IsLeapYear(year1) ? Date::LEAP_DAYS[month1] : Date::NORMAL_DAYS[month1];
			month_diff--;
		} else {
			day_diff += Date::IsLeapYear(year2) ? Date::LEAP_DAYS[month2] : Date::NORMAL_DAYS[month2];
			month_diff--;
		}
	}
	while (month_diff < 0) {
		month_diff += MONTHS_PER_YEAR;
		year_diff--;
	}

	// recover sign if necessary
	if (sign_flipped) {
		year_diff = -year_diff;
		month_diff = -month_diff;
		day_diff = -day_diff;
		hour_diff = -hour_diff;
		min_diff = -min_diff;
		sec_diff = -sec_diff;
		micros_diff = -micros_diff;
	}
	interval_t interval;
	interval.months = year_diff * MONTHS_PER_YEAR + month_diff;
	interval.days = day_diff;
	interval.micros = Time::FromTime(hour_diff, min_diff, sec_diff, micros_diff).micros;

	return interval;
}

interval_t Interval::GetDifference(timestamp_t timestamp_1, timestamp_t timestamp_2) {
	if (!Timestamp::IsFinite(timestamp_1) || !Timestamp::IsFinite(timestamp_2)) {
		throw InvalidInputException("Cannot subtract infinite timestamps");
	}
	const auto us_1 = Timestamp::GetEpochMicroSeconds(timestamp_1);
	const auto us_2 = Timestamp::GetEpochMicroSeconds(timestamp_2);
	int64_t delta_us;
	if (!TrySubtractOperator::Operation(us_1, us_2, delta_us)) {
		throw ConversionException("Timestamp difference is out of bounds");
	}
	return FromMicro(delta_us);
}

interval_t Interval::FromMicro(int64_t delta_us) {
	interval_t result;
	result.months = 0;
	result.days = delta_us / Interval::MICROS_PER_DAY;
	result.micros = delta_us % Interval::MICROS_PER_DAY;

	return result;
}

static void NormalizeIntervalEntries(interval_t input, int64_t &months, int64_t &days, int64_t &micros) {
	int64_t extra_months_d = input.days / Interval::DAYS_PER_MONTH;
	int64_t extra_months_micros = input.micros / Interval::MICROS_PER_MONTH;
	input.days -= extra_months_d * Interval::DAYS_PER_MONTH;
	input.micros -= extra_months_micros * Interval::MICROS_PER_MONTH;

	int64_t extra_days_micros = input.micros / Interval::MICROS_PER_DAY;
	input.micros -= extra_days_micros * Interval::MICROS_PER_DAY;

	months = input.months + extra_months_d + extra_months_micros;
	days = input.days + extra_days_micros;
	micros = input.micros;
}

bool Interval::Equals(interval_t left, interval_t right) {
	return left.months == right.months && left.days == right.days && left.micros == right.micros;
}

bool Interval::GreaterThan(interval_t left, interval_t right) {
	int64_t lmonths, ldays, lmicros;
	int64_t rmonths, rdays, rmicros;
	NormalizeIntervalEntries(left, lmonths, ldays, lmicros);
	NormalizeIntervalEntries(right, rmonths, rdays, rmicros);

	if (lmonths > rmonths) {
		return true;
	} else if (lmonths < rmonths) {
		return false;
	}
	if (ldays > rdays) {
		return true;
	} else if (ldays < rdays) {
		return false;
	}
	return lmicros > rmicros;
}

bool Interval::GreaterThanEquals(interval_t left, interval_t right) {
	return GreaterThan(left, right) || Equals(left, right);
}

interval_t Interval::Invert(interval_t interval) {
	interval.days = -interval.days;
	interval.micros = -interval.micros;
	interval.months = -interval.months;
	return interval;
}

date_t Interval::Add(date_t left, interval_t right) {
	if (!Date::IsFinite(left)) {
		return left;
	}
	date_t result;
	if (right.months != 0) {
		int32_t year, month, day;
		Date::Convert(left, year, month, day);
		int32_t year_diff = right.months / Interval::MONTHS_PER_YEAR;
		year += year_diff;
		month += right.months - year_diff * Interval::MONTHS_PER_YEAR;
		if (month > Interval::MONTHS_PER_YEAR) {
			year++;
			month -= Interval::MONTHS_PER_YEAR;
		} else if (month <= 0) {
			year--;
			month += Interval::MONTHS_PER_YEAR;
		}
		day = MinValue<int32_t>(day, Date::MonthDays(year, month));
		result = Date::FromDate(year, month, day);
	} else {
		result = left;
	}
	if (right.days != 0) {
		if (!TryAddOperator::Operation(result.days, right.days, result.days)) {
			throw OutOfRangeException("Date out of range");
		}
	}
	if (right.micros != 0) {
		if (!TryAddOperator::Operation(result.days, int32_t(right.micros / Interval::MICROS_PER_DAY), result.days)) {
			throw OutOfRangeException("Date out of range");
		}
	}
	if (!Date::IsFinite(result)) {
		throw OutOfRangeException("Date out of range");
	}
	return result;
}

dtime_t Interval::Add(dtime_t left, interval_t right, date_t &date) {
	int64_t diff = right.micros - ((right.micros / Interval::MICROS_PER_DAY) * Interval::MICROS_PER_DAY);
	left += diff;
	if (left.micros >= Interval::MICROS_PER_DAY) {
		left.micros -= Interval::MICROS_PER_DAY;
		date.days++;
	} else if (left.micros < 0) {
		left.micros += Interval::MICROS_PER_DAY;
		date.days--;
	}
	return left;
}

timestamp_t Interval::Add(timestamp_t left, interval_t right) {
	if (!Timestamp::IsFinite(left)) {
		return left;
	}
	date_t date;
	dtime_t time;
	Timestamp::Convert(left, date, time);
	auto new_date = Interval::Add(date, right);
	auto new_time = Interval::Add(time, right, new_date);
	return Timestamp::FromDatetime(new_date, new_time);
}

} // namespace duckdb






namespace duckdb {

PartitionedColumnData::PartitionedColumnData(PartitionedColumnDataType type_p, ClientContext &context_p,
                                             vector<LogicalType> types_p)
    : type(type_p), context(context_p), types(std::move(types_p)), allocators(make_shared<PartitionAllocators>()) {
}

PartitionedColumnData::PartitionedColumnData(const PartitionedColumnData &other)
    : type(other.type), context(other.context), types(other.types), allocators(other.allocators) {
}

unique_ptr<PartitionedColumnData> PartitionedColumnData::CreateShared() {
	switch (type) {
	case PartitionedColumnDataType::RADIX:
		return make_unique<RadixPartitionedColumnData>((RadixPartitionedColumnData &)*this);
	case PartitionedColumnDataType::HIVE:
		return make_unique<HivePartitionedColumnData>((HivePartitionedColumnData &)*this);
	default:
		throw NotImplementedException("CreateShared for this type of PartitionedColumnData");
	}
}

PartitionedColumnData::~PartitionedColumnData() {
}

void PartitionedColumnData::InitializeAppendState(PartitionedColumnDataAppendState &state) const {
	state.partition_sel.Initialize();
	state.slice_chunk.Initialize(context, types);
	InitializeAppendStateInternal(state);
}

unique_ptr<DataChunk> PartitionedColumnData::CreatePartitionBuffer() const {
	auto result = make_unique<DataChunk>();
	result->Initialize(BufferManager::GetBufferManager(context).GetBufferAllocator(), types, BufferSize());
	return result;
}

void PartitionedColumnData::Append(PartitionedColumnDataAppendState &state, DataChunk &input) {
	// Compute partition indices and store them in state.partition_indices
	ComputePartitionIndices(state, input);

	// Compute the counts per partition
	const auto count = input.size();
	unordered_map<idx_t, list_entry_t> partition_entries;
	const auto partition_indices = FlatVector::GetData<idx_t>(state.partition_indices);
	switch (state.partition_indices.GetVectorType()) {
	case VectorType::FLAT_VECTOR:
		for (idx_t i = 0; i < count; i++) {
			const auto &partition_index = partition_indices[i];
			auto partition_entry = partition_entries.find(partition_index);
			if (partition_entry == partition_entries.end()) {
				partition_entries[partition_index] = list_entry_t(0, 1);
			} else {
				partition_entry->second.length++;
			}
		}
		break;
	case VectorType::CONSTANT_VECTOR:
		partition_entries[partition_indices[0]] = list_entry_t(0, count);
		break;
	default:
		throw InternalException("Unexpected VectorType in PartitionedColumnData::Append");
	}

	// Early out: check if everything belongs to a single partition
	if (partition_entries.size() == 1) {
		const auto &partition_index = partition_entries.begin()->first;
		auto &partition = *partitions[partition_index];
		auto &partition_append_state = state.partition_append_states[partition_index];
		partition.Append(*partition_append_state, input);
		return;
	}

	// Compute offsets from the counts
	idx_t offset = 0;
	for (auto &pc : partition_entries) {
		auto &partition_entry = pc.second;
		partition_entry.offset = offset;
		offset += partition_entry.length;
	}

	// Now initialize a single selection vector that acts as a selection vector for every partition
	auto &all_partitions_sel = state.partition_sel;
	for (idx_t i = 0; i < count; i++) {
		const auto &partition_index = partition_indices[i];
		auto &partition_offset = partition_entries[partition_index].offset;
		all_partitions_sel[partition_offset++] = i;
	}

	// Loop through the partitions to append the new data to the partition buffers, and flush the buffers if necessary
	SelectionVector partition_sel;
	for (auto &pc : partition_entries) {
		const auto &partition_index = pc.first;

		// Partition, buffer, and append state for this partition index
		auto &partition = *partitions[partition_index];
		auto &partition_buffer = *state.partition_buffers[partition_index];
		auto &partition_append_state = state.partition_append_states[partition_index];

		// Length and offset into the selection vector for this chunk, for this partition
		const auto &partition_entry = pc.second;
		const auto &partition_length = partition_entry.length;
		const auto partition_offset = partition_entry.offset - partition_length;

		// Create a selection vector for this partition using the offset into the single selection vector
		partition_sel.Initialize(all_partitions_sel.data() + partition_offset);

		if (partition_length >= HalfBufferSize()) {
			// Slice the input chunk using the selection vector
			state.slice_chunk.Reset();
			state.slice_chunk.Slice(input, partition_sel, partition_length);

			// Append it to the partition directly
			partition.Append(*partition_append_state, state.slice_chunk);
		} else {
			// Append the input chunk to the partition buffer using the selection vector
			partition_buffer.Append(input, false, &partition_sel, partition_length);

			if (partition_buffer.size() >= HalfBufferSize()) {
				// Next batch won't fit in the buffer, flush it to the partition
				partition.Append(*partition_append_state, partition_buffer);
				partition_buffer.Reset();
				partition_buffer.SetCapacity(BufferSize());
			}
		}
	}
}

void PartitionedColumnData::FlushAppendState(PartitionedColumnDataAppendState &state) {
	for (idx_t i = 0; i < state.partition_buffers.size(); i++) {
		auto &partition_buffer = *state.partition_buffers[i];
		if (partition_buffer.size() > 0) {
			partitions[i]->Append(partition_buffer);
		}
	}
}

void PartitionedColumnData::Combine(PartitionedColumnData &other) {
	// Now combine the state's partitions into this
	lock_guard<mutex> guard(lock);

	if (partitions.empty()) {
		// This is the first merge, we just copy them over
		partitions = std::move(other.partitions);
	} else {
		D_ASSERT(partitions.size() == other.partitions.size());
		// Combine the append state's partitions into this PartitionedColumnData
		for (idx_t i = 0; i < other.partitions.size(); i++) {
			partitions[i]->Combine(*other.partitions[i]);
		}
	}
}

vector<unique_ptr<ColumnDataCollection>> &PartitionedColumnData::GetPartitions() {
	return partitions;
}

void PartitionedColumnData::CreateAllocator() {
	allocators->allocators.emplace_back(make_shared<ColumnDataAllocator>(BufferManager::GetBufferManager(context)));
	allocators->allocators.back()->MakeShared();
}

} // namespace duckdb


namespace duckdb {

RowDataCollection::RowDataCollection(BufferManager &buffer_manager, idx_t block_capacity, idx_t entry_size,
                                     bool keep_pinned)
    : buffer_manager(buffer_manager), count(0), block_capacity(block_capacity), entry_size(entry_size),
      keep_pinned(keep_pinned) {
	D_ASSERT(block_capacity * entry_size + entry_size > Storage::BLOCK_SIZE);
}

idx_t RowDataCollection::AppendToBlock(RowDataBlock &block, BufferHandle &handle,
                                       vector<BlockAppendEntry> &append_entries, idx_t remaining, idx_t entry_sizes[]) {
	idx_t append_count = 0;
	data_ptr_t dataptr;
	if (entry_sizes) {
		D_ASSERT(entry_size == 1);
		// compute how many entries fit if entry size is variable
		dataptr = handle.Ptr() + block.byte_offset;
		for (idx_t i = 0; i < remaining; i++) {
			if (block.byte_offset + entry_sizes[i] > block.capacity) {
				if (block.count == 0 && append_count == 0 && entry_sizes[i] > block.capacity) {
					// special case: single entry is bigger than block capacity
					// resize current block to fit the entry, append it, and move to the next block
					block.capacity = entry_sizes[i];
					buffer_manager.ReAllocate(block.block, block.capacity);
					dataptr = handle.Ptr();
					append_count++;
					block.byte_offset += entry_sizes[i];
				}
				break;
			}
			append_count++;
			block.byte_offset += entry_sizes[i];
		}
	} else {
		append_count = MinValue<idx_t>(remaining, block.capacity - block.count);
		dataptr = handle.Ptr() + block.count * entry_size;
	}
	append_entries.emplace_back(dataptr, append_count);
	block.count += append_count;
	return append_count;
}

RowDataBlock &RowDataCollection::CreateBlock() {
	blocks.push_back(make_unique<RowDataBlock>(buffer_manager, block_capacity, entry_size));
	return *blocks.back();
}

vector<BufferHandle> RowDataCollection::Build(idx_t added_count, data_ptr_t key_locations[], idx_t entry_sizes[],
                                              const SelectionVector *sel) {
	vector<BufferHandle> handles;
	vector<BlockAppendEntry> append_entries;

	// first allocate space of where to serialize the keys and payload columns
	idx_t remaining = added_count;
	{
		// first append to the last block (if any)
		lock_guard<mutex> append_lock(rdc_lock);
		count += added_count;

		if (!blocks.empty()) {
			auto &last_block = *blocks.back();
			if (last_block.count < last_block.capacity) {
				// last block has space: pin the buffer of this block
				auto handle = buffer_manager.Pin(last_block.block);
				// now append to the block
				idx_t append_count = AppendToBlock(last_block, handle, append_entries, remaining, entry_sizes);
				remaining -= append_count;
				handles.push_back(std::move(handle));
			}
		}
		while (remaining > 0) {
			// now for the remaining data, allocate new buffers to store the data and append there
			auto &new_block = CreateBlock();
			auto handle = buffer_manager.Pin(new_block.block);

			// offset the entry sizes array if we have added entries already
			idx_t *offset_entry_sizes = entry_sizes ? entry_sizes + added_count - remaining : nullptr;

			idx_t append_count = AppendToBlock(new_block, handle, append_entries, remaining, offset_entry_sizes);
			D_ASSERT(new_block.count > 0);
			remaining -= append_count;

			if (keep_pinned) {
				pinned_blocks.push_back(std::move(handle));
			} else {
				handles.push_back(std::move(handle));
			}
		}
	}
	// now set up the key_locations based on the append entries
	idx_t append_idx = 0;
	for (auto &append_entry : append_entries) {
		idx_t next = append_idx + append_entry.count;
		if (entry_sizes) {
			for (; append_idx < next; append_idx++) {
				key_locations[append_idx] = append_entry.baseptr;
				append_entry.baseptr += entry_sizes[append_idx];
			}
		} else {
			for (; append_idx < next; append_idx++) {
				auto idx = sel->get_index(append_idx);
				key_locations[idx] = append_entry.baseptr;
				append_entry.baseptr += entry_size;
			}
		}
	}
	// return the unique pointers to the handles because they must stay pinned
	return handles;
}

void RowDataCollection::Merge(RowDataCollection &other) {
	if (other.count == 0) {
		return;
	}
	RowDataCollection temp(buffer_manager, Storage::BLOCK_SIZE, 1);
	{
		//	One lock at a time to avoid deadlocks
		lock_guard<mutex> read_lock(other.rdc_lock);
		temp.count = other.count;
		temp.block_capacity = other.block_capacity;
		temp.entry_size = other.entry_size;
		temp.blocks = std::move(other.blocks);
		temp.pinned_blocks = std::move(other.pinned_blocks);
	}
	other.Clear();

	lock_guard<mutex> write_lock(rdc_lock);
	count += temp.count;
	block_capacity = MaxValue(block_capacity, temp.block_capacity);
	entry_size = MaxValue(entry_size, temp.entry_size);
	for (auto &block : temp.blocks) {
		blocks.emplace_back(std::move(block));
	}
	for (auto &handle : temp.pinned_blocks) {
		pinned_blocks.emplace_back(std::move(handle));
	}
}

} // namespace duckdb






namespace duckdb {

void RowDataCollectionScanner::AlignHeapBlocks(RowDataCollection &swizzled_block_collection,
                                               RowDataCollection &swizzled_string_heap,
                                               RowDataCollection &block_collection, RowDataCollection &string_heap,
                                               const RowLayout &layout) {
	if (block_collection.count == 0) {
		return;
	}

	if (layout.AllConstant()) {
		// No heap blocks! Just merge fixed-size data
		swizzled_block_collection.Merge(block_collection);
		return;
	}

	// We create one heap block per data block and swizzle the pointers
	D_ASSERT(string_heap.keep_pinned == swizzled_string_heap.keep_pinned);
	auto &buffer_manager = block_collection.buffer_manager;
	auto &heap_blocks = string_heap.blocks;
	idx_t heap_block_idx = 0;
	idx_t heap_block_remaining = heap_blocks[heap_block_idx]->count;
	for (auto &data_block : block_collection.blocks) {
		if (heap_block_remaining == 0) {
			heap_block_remaining = heap_blocks[++heap_block_idx]->count;
		}

		// Pin the data block and swizzle the pointers within the rows
		auto data_handle = buffer_manager.Pin(data_block->block);
		auto data_ptr = data_handle.Ptr();
		if (!string_heap.keep_pinned) {
			D_ASSERT(!data_block->block->IsSwizzled());
			RowOperations::SwizzleColumns(layout, data_ptr, data_block->count);
			data_block->block->SetSwizzling(nullptr);
		}
		// At this point the data block is pinned and the heap pointer is valid
		// so we can copy heap data as needed

		// We want to copy as little of the heap data as possible, check how the data and heap blocks line up
		if (heap_block_remaining >= data_block->count) {
			// Easy: current heap block contains all strings for this data block, just copy (reference) the block
			swizzled_string_heap.blocks.emplace_back(heap_blocks[heap_block_idx]->Copy());
			swizzled_string_heap.blocks.back()->count = data_block->count;

			// Swizzle the heap pointer if we are not pinning the heap
			auto &heap_block = swizzled_string_heap.blocks.back()->block;
			auto heap_handle = buffer_manager.Pin(heap_block);
			if (!swizzled_string_heap.keep_pinned) {
				auto heap_ptr = Load<data_ptr_t>(data_ptr + layout.GetHeapOffset());
				auto heap_offset = heap_ptr - heap_handle.Ptr();
				RowOperations::SwizzleHeapPointer(layout, data_ptr, heap_ptr, data_block->count, heap_offset);
			} else {
				swizzled_string_heap.pinned_blocks.emplace_back(std::move(heap_handle));
			}

			// Update counter
			heap_block_remaining -= data_block->count;
		} else {
			// Strings for this data block are spread over the current heap block and the next (and possibly more)
			if (string_heap.keep_pinned) {
				// The heap is changing underneath the data block,
				// so swizzle the string pointers to make them portable.
				RowOperations::SwizzleColumns(layout, data_ptr, data_block->count);
			}
			idx_t data_block_remaining = data_block->count;
			vector<std::pair<data_ptr_t, idx_t>> ptrs_and_sizes;
			idx_t total_size = 0;
			const auto base_row_ptr = data_ptr;
			while (data_block_remaining > 0) {
				if (heap_block_remaining == 0) {
					heap_block_remaining = heap_blocks[++heap_block_idx]->count;
				}
				auto next = MinValue<idx_t>(data_block_remaining, heap_block_remaining);

				// Figure out where to start copying strings, and how many bytes we need to copy
				auto heap_start_ptr = Load<data_ptr_t>(data_ptr + layout.GetHeapOffset());
				auto heap_end_ptr =
				    Load<data_ptr_t>(data_ptr + layout.GetHeapOffset() + (next - 1) * layout.GetRowWidth());
				idx_t size = heap_end_ptr - heap_start_ptr + Load<uint32_t>(heap_end_ptr);
				ptrs_and_sizes.emplace_back(heap_start_ptr, size);
				D_ASSERT(size <= heap_blocks[heap_block_idx]->byte_offset);

				// Swizzle the heap pointer
				RowOperations::SwizzleHeapPointer(layout, data_ptr, heap_start_ptr, next, total_size);
				total_size += size;

				// Update where we are in the data and heap blocks
				data_ptr += next * layout.GetRowWidth();
				data_block_remaining -= next;
				heap_block_remaining -= next;
			}

			// Finally, we allocate a new heap block and copy data to it
			swizzled_string_heap.blocks.emplace_back(
			    make_unique<RowDataBlock>(buffer_manager, MaxValue<idx_t>(total_size, (idx_t)Storage::BLOCK_SIZE), 1));
			auto new_heap_handle = buffer_manager.Pin(swizzled_string_heap.blocks.back()->block);
			auto new_heap_ptr = new_heap_handle.Ptr();
			if (swizzled_string_heap.keep_pinned) {
				// Since the heap blocks are pinned, we can unswizzle the data again.
				swizzled_string_heap.pinned_blocks.emplace_back(std::move(new_heap_handle));
				RowOperations::UnswizzlePointers(layout, base_row_ptr, new_heap_ptr, data_block->count);
				RowOperations::UnswizzleHeapPointer(layout, base_row_ptr, new_heap_ptr, data_block->count);
			}
			for (auto &ptr_and_size : ptrs_and_sizes) {
				memcpy(new_heap_ptr, ptr_and_size.first, ptr_and_size.second);
				new_heap_ptr += ptr_and_size.second;
			}
		}
	}

	// We're done with variable-sized data, now just merge the fixed-size data
	swizzled_block_collection.Merge(block_collection);
	D_ASSERT(swizzled_block_collection.blocks.size() == swizzled_string_heap.blocks.size());

	// Update counts and cleanup
	swizzled_string_heap.count = string_heap.count;
	string_heap.Clear();
}

void RowDataCollectionScanner::ScanState::PinData() {
	auto &rows = scanner.rows;
	D_ASSERT(block_idx < rows.blocks.size());
	auto &data_block = rows.blocks[block_idx];
	if (!data_handle.IsValid() || data_handle.GetBlockHandle() != data_block->block) {
		data_handle = rows.buffer_manager.Pin(data_block->block);
	}
	if (scanner.layout.AllConstant() || !scanner.external) {
		return;
	}

	auto &heap = scanner.heap;
	D_ASSERT(block_idx < heap.blocks.size());
	auto &heap_block = heap.blocks[block_idx];
	if (!heap_handle.IsValid() || heap_handle.GetBlockHandle() != heap_block->block) {
		heap_handle = heap.buffer_manager.Pin(heap_block->block);
	}
}

RowDataCollectionScanner::RowDataCollectionScanner(RowDataCollection &rows_p, RowDataCollection &heap_p,
                                                   const RowLayout &layout_p, bool external_p, bool flush_p)
    : rows(rows_p), heap(heap_p), layout(layout_p), read_state(*this), total_count(rows.count), total_scanned(0),
      external(external_p), flush(flush_p), unswizzling(!layout.AllConstant() && external && !heap.keep_pinned) {

	if (unswizzling) {
		D_ASSERT(rows.blocks.size() == heap.blocks.size());
	}

	ValidateUnscannedBlock();
}

void RowDataCollectionScanner::SwizzleBlock(RowDataBlock &data_block, RowDataBlock &heap_block) {
	// Pin the data block and swizzle the pointers within the rows
	D_ASSERT(!data_block.block->IsSwizzled());
	auto data_handle = rows.buffer_manager.Pin(data_block.block);
	auto data_ptr = data_handle.Ptr();
	RowOperations::SwizzleColumns(layout, data_ptr, data_block.count);
	data_block.block->SetSwizzling(nullptr);

	// Swizzle the heap pointers
	auto heap_handle = heap.buffer_manager.Pin(heap_block.block);
	auto heap_ptr = Load<data_ptr_t>(data_ptr + layout.GetHeapOffset());
	auto heap_offset = heap_ptr - heap_handle.Ptr();
	RowOperations::SwizzleHeapPointer(layout, data_ptr, heap_ptr, data_block.count, heap_offset);
}

void RowDataCollectionScanner::ReSwizzle() {
	if (rows.count == 0) {
		return;
	}

	if (!unswizzling) {
		// No swizzled blocks!
		return;
	}

	D_ASSERT(rows.blocks.size() == heap.blocks.size());
	for (idx_t i = 0; i < rows.blocks.size(); ++i) {
		auto &data_block = rows.blocks[i];
		if (data_block->block && !data_block->block->IsSwizzled()) {
			SwizzleBlock(*data_block, *heap.blocks[i]);
		}
	}
}

void RowDataCollectionScanner::ValidateUnscannedBlock() const {
	if (unswizzling && read_state.block_idx < rows.blocks.size()) {
		D_ASSERT(rows.blocks[read_state.block_idx]->block->IsSwizzled());
	}
}

void RowDataCollectionScanner::Scan(DataChunk &chunk) {
	auto count = MinValue((idx_t)STANDARD_VECTOR_SIZE, total_count - total_scanned);
	if (count == 0) {
		chunk.SetCardinality(count);
		return;
	}

	const idx_t &row_width = layout.GetRowWidth();
	// Set up a batch of pointers to scan data from
	idx_t scanned = 0;
	auto data_pointers = FlatVector::GetData<data_ptr_t>(addresses);

	// We must pin ALL blocks we are going to gather from
	vector<BufferHandle> pinned_blocks;
	while (scanned < count) {
		read_state.PinData();
		auto &data_block = rows.blocks[read_state.block_idx];
		idx_t next = MinValue(data_block->count - read_state.entry_idx, count - scanned);
		const data_ptr_t data_ptr = read_state.data_handle.Ptr() + read_state.entry_idx * row_width;
		// Set up the next pointers
		data_ptr_t row_ptr = data_ptr;
		for (idx_t i = 0; i < next; i++) {
			data_pointers[scanned + i] = row_ptr;
			row_ptr += row_width;
		}
		// Unswizzle the offsets back to pointers (if needed)
		if (unswizzling) {
			RowOperations::UnswizzlePointers(layout, data_ptr, read_state.heap_handle.Ptr(), next);
			rows.blocks[read_state.block_idx]->block->SetSwizzling("RowDataCollectionScanner::Scan");
		}
		// Update state indices
		read_state.entry_idx += next;
		if (read_state.entry_idx == data_block->count) {
			// Pin completed blocks so we don't lose them
			pinned_blocks.emplace_back(rows.buffer_manager.Pin(data_block->block));
			if (unswizzling) {
				auto &heap_block = heap.blocks[read_state.block_idx];
				pinned_blocks.emplace_back(heap.buffer_manager.Pin(heap_block->block));
			}
			read_state.block_idx++;
			read_state.entry_idx = 0;
			ValidateUnscannedBlock();
		}
		scanned += next;
	}
	D_ASSERT(scanned == count);
	// Deserialize the payload data
	for (idx_t col_no = 0; col_no < layout.ColumnCount(); col_no++) {
		RowOperations::Gather(addresses, *FlatVector::IncrementalSelectionVector(), chunk.data[col_no],
		                      *FlatVector::IncrementalSelectionVector(), count, layout, col_no);
	}
	chunk.SetCardinality(count);
	chunk.Verify();
	total_scanned += scanned;

	//	Switch to a new set of pinned blocks
	read_state.pinned_blocks.swap(pinned_blocks);

	if (flush) {
		// Release blocks we have passed.
		for (idx_t i = 0; i < read_state.block_idx; ++i) {
			rows.blocks[i]->block = nullptr;
			if (unswizzling) {
				heap.blocks[i]->block = nullptr;
			}
		}
	} else if (unswizzling) {
		// Reswizzle blocks we have passed so they can be flushed safely.
		for (idx_t i = 0; i < read_state.block_idx; ++i) {
			auto &data_block = rows.blocks[i];
			if (data_block->block && !data_block->block->IsSwizzled()) {
				SwizzleBlock(*data_block, *heap.blocks[i]);
			}
		}
	}
}

void RowDataCollectionScanner::Reset(bool flush_p) {
	flush = flush_p;
	total_scanned = 0;

	read_state.block_idx = 0;
	read_state.entry_idx = 0;
}

} // namespace duckdb
//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/types/row_layout.cpp
//
//
//===----------------------------------------------------------------------===//





namespace duckdb {

RowLayout::RowLayout()
    : flag_width(0), data_width(0), aggr_width(0), row_width(0), all_constant(true), heap_pointer_offset(0) {
}

void RowLayout::Initialize(vector<LogicalType> types_p, Aggregates aggregates_p, bool align) {
	offsets.clear();
	types = std::move(types_p);

	// Null mask at the front - 1 bit per value.
	flag_width = ValidityBytes::ValidityMaskSize(types.size());
	row_width = flag_width;

	// Whether all columns are constant size.
	for (const auto &type : types) {
		all_constant = all_constant && TypeIsConstantSize(type.InternalType());
	}

	// This enables pointer swizzling for out-of-core computation.
	if (!all_constant) {
		// When unswizzled the pointer lives here.
		// When swizzled, the pointer is replaced by an offset.
		heap_pointer_offset = row_width;
		// The 8 byte pointer will be replaced with an 8 byte idx_t when swizzled.
		// However, this cannot be sizeof(data_ptr_t), since 32 bit builds use 4 byte pointers.
		row_width += sizeof(idx_t);
	}

	// Data columns. No alignment required.
	for (const auto &type : types) {
		offsets.push_back(row_width);
		const auto internal_type = type.InternalType();
		if (TypeIsConstantSize(internal_type) || internal_type == PhysicalType::VARCHAR) {
			row_width += GetTypeIdSize(type.InternalType());
		} else {
			// Variable size types use pointers to the actual data (can be swizzled).
			// Again, we would use sizeof(data_ptr_t), but this is not guaranteed to be equal to sizeof(idx_t).
			row_width += sizeof(idx_t);
		}
	}

	// Alignment padding for aggregates
#ifndef DUCKDB_ALLOW_UNDEFINED
	if (align) {
		row_width = AlignValue(row_width);
	}
#endif
	data_width = row_width - flag_width;

	// Aggregate fields.
	aggregates = std::move(aggregates_p);
	for (auto &aggregate : aggregates) {
		offsets.push_back(row_width);
		row_width += aggregate.payload_size;
#ifndef DUCKDB_ALLOW_UNDEFINED
		D_ASSERT(aggregate.payload_size == AlignValue(aggregate.payload_size));
#endif
	}
	aggr_width = row_width - data_width - flag_width;

	// Alignment padding for the next row
#ifndef DUCKDB_ALLOW_UNDEFINED
	if (align) {
		row_width = AlignValue(row_width);
	}
#endif
}

void RowLayout::Initialize(vector<LogicalType> types_p, bool align) {
	Initialize(std::move(types_p), Aggregates(), align);
}

void RowLayout::Initialize(Aggregates aggregates_p, bool align) {
	Initialize(vector<LogicalType>(), std::move(aggregates_p), align);
}

} // namespace duckdb




namespace duckdb {

SelectionData::SelectionData(idx_t count) {
	owned_data = unique_ptr<sel_t[]>(new sel_t[count]);
#ifdef DEBUG
	for (idx_t i = 0; i < count; i++) {
		owned_data[i] = std::numeric_limits<sel_t>::max();
	}
#endif
}

// LCOV_EXCL_START
string SelectionVector::ToString(idx_t count) const {
	string result = "Selection Vector (" + to_string(count) + ") [";
	for (idx_t i = 0; i < count; i++) {
		if (i != 0) {
			result += ", ";
		}
		result += to_string(get_index(i));
	}
	result += "]";
	return result;
}

void SelectionVector::Print(idx_t count) const {
	Printer::Print(ToString(count));
}
// LCOV_EXCL_STOP

buffer_ptr<SelectionData> SelectionVector::Slice(const SelectionVector &sel, idx_t count) const {
	auto data = make_buffer<SelectionData>(count);
	auto result_ptr = data->owned_data.get();
	// for every element, we perform result[i] = target[new[i]]
	for (idx_t i = 0; i < count; i++) {
		auto new_idx = sel.get_index(i);
		auto idx = this->get_index(new_idx);
		result_ptr[i] = idx;
	}
	return data;
}

} // namespace duckdb







#include <cstring>

namespace duckdb {

StringHeap::StringHeap(Allocator &allocator) : allocator(allocator) {
}

void StringHeap::Destroy() {
	allocator.Destroy();
}

void StringHeap::Move(StringHeap &other) {
	other.allocator.Move(allocator);
}

string_t StringHeap::AddString(const char *data, idx_t len) {
	D_ASSERT(Utf8Proc::Analyze(data, len) != UnicodeType::INVALID);
	return AddBlob(data, len);
}

string_t StringHeap::AddString(const char *data) {
	return AddString(data, strlen(data));
}

string_t StringHeap::AddString(const string &data) {
	return AddString(data.c_str(), data.size());
}

string_t StringHeap::AddString(const string_t &data) {
	return AddString(data.GetDataUnsafe(), data.GetSize());
}

string_t StringHeap::AddBlob(const char *data, idx_t len) {
	auto insert_string = EmptyString(len);
	auto insert_pos = insert_string.GetDataWriteable();
	memcpy(insert_pos, data, len);
	insert_string.Finalize();
	return insert_string;
}

string_t StringHeap::AddBlob(const string_t &data) {
	return AddBlob(data.GetDataUnsafe(), data.GetSize());
}

string_t StringHeap::EmptyString(idx_t len) {
	D_ASSERT(len >= string_t::INLINE_LENGTH);
	auto insert_pos = (const char *)allocator.Allocate(len);
	return string_t(insert_pos, len);
}

} // namespace duckdb






namespace duckdb {

void string_t::Verify() const {
	auto dataptr = GetDataUnsafe();
	(void)dataptr;
	D_ASSERT(dataptr);

#ifdef DEBUG
	auto utf_type = Utf8Proc::Analyze(dataptr, GetSize());
	D_ASSERT(utf_type != UnicodeType::INVALID);
#endif

	// verify that the prefix contains the first four characters of the string
	for (idx_t i = 0; i < MinValue<uint32_t>(PREFIX_LENGTH, GetSize()); i++) {
		D_ASSERT(GetPrefix()[i] == dataptr[i]);
	}
	// verify that for strings with length < INLINE_LENGTH, the rest of the string is zero
	for (idx_t i = GetSize(); i < INLINE_LENGTH; i++) {
		D_ASSERT(GetDataUnsafe()[i] == '\0');
	}
}

} // namespace duckdb









#include <cctype>
#include <cstring>
#include <sstream>

namespace duckdb {

static_assert(sizeof(dtime_t) == sizeof(int64_t), "dtime_t was padded");

// string format is hh:mm:ss.microsecondsZ
// microseconds and Z are optional
// ISO 8601

bool Time::TryConvertInternal(const char *buf, idx_t len, idx_t &pos, dtime_t &result, bool strict) {
	int32_t hour = -1, min = -1, sec = -1, micros = -1;
	pos = 0;

	if (len == 0) {
		return false;
	}

	int sep;

	// skip leading spaces
	while (pos < len && StringUtil::CharacterIsSpace(buf[pos])) {
		pos++;
	}

	if (pos >= len) {
		return false;
	}

	if (!StringUtil::CharacterIsDigit(buf[pos])) {
		return false;
	}

	if (!Date::ParseDoubleDigit(buf, len, pos, hour)) {
		return false;
	}
	if (hour < 0 || hour >= 24) {
		return false;
	}

	if (pos >= len) {
		return false;
	}

	// fetch the separator
	sep = buf[pos++];
	if (sep != ':') {
		// invalid separator
		return false;
	}

	if (!Date::ParseDoubleDigit(buf, len, pos, min)) {
		return false;
	}
	if (min < 0 || min >= 60) {
		return false;
	}

	if (pos >= len) {
		return false;
	}

	if (buf[pos++] != sep) {
		return false;
	}

	if (!Date::ParseDoubleDigit(buf, len, pos, sec)) {
		return false;
	}
	if (sec < 0 || sec >= 60) {
		return false;
	}

	micros = 0;
	if (pos < len && buf[pos] == '.') {
		pos++;
		// we expect some microseconds
		int32_t mult = 100000;
		for (; pos < len && StringUtil::CharacterIsDigit(buf[pos]); pos++, mult /= 10) {
			if (mult > 0) {
				micros += (buf[pos] - '0') * mult;
			}
		}
	}

	// in strict mode, check remaining string for non-space characters
	if (strict) {
		// skip trailing spaces
		while (pos < len && StringUtil::CharacterIsSpace(buf[pos])) {
			pos++;
		}
		// check position. if end was not reached, non-space chars remaining
		if (pos < len) {
			return false;
		}
	}

	result = Time::FromTime(hour, min, sec, micros);
	return true;
}

bool Time::TryConvertTime(const char *buf, idx_t len, idx_t &pos, dtime_t &result, bool strict) {
	if (!Time::TryConvertInternal(buf, len, pos, result, strict)) {
		if (!strict) {
			// last chance, check if we can parse as timestamp
			timestamp_t timestamp;
			if (Timestamp::TryConvertTimestamp(buf, len, timestamp)) {
				if (!Timestamp::IsFinite(timestamp)) {
					return false;
				}
				result = Timestamp::GetTime(timestamp);
				return true;
			}
		}
		return false;
	}
	return true;
}

string Time::ConversionError(const string &str) {
	return StringUtil::Format("time field value out of range: \"%s\", "
	                          "expected format is ([YYYY-MM-DD ]HH:MM:SS[.MS])",
	                          str);
}

string Time::ConversionError(string_t str) {
	return Time::ConversionError(str.GetString());
}

dtime_t Time::FromCString(const char *buf, idx_t len, bool strict) {
	dtime_t result;
	idx_t pos;
	if (!Time::TryConvertTime(buf, len, pos, result, strict)) {
		throw ConversionException(ConversionError(string(buf, len)));
	}
	return result;
}

dtime_t Time::FromString(const string &str, bool strict) {
	return Time::FromCString(str.c_str(), str.size(), strict);
}

string Time::ToString(dtime_t time) {
	int32_t time_units[4];
	Time::Convert(time, time_units[0], time_units[1], time_units[2], time_units[3]);

	char micro_buffer[6];
	auto length = TimeToStringCast::Length(time_units, micro_buffer);
	auto buffer = unique_ptr<char[]>(new char[length]);
	TimeToStringCast::Format(buffer.get(), length, time_units, micro_buffer);
	return string(buffer.get(), length);
}

string Time::ToUTCOffset(int hour_offset, int minute_offset) {
	dtime_t time((hour_offset * Interval::MINS_PER_HOUR + minute_offset) * Interval::MICROS_PER_MINUTE);

	char buffer[1 + 2 + 1 + 2];
	idx_t length = 0;
	buffer[length++] = (time.micros < 0 ? '-' : '+');
	time.micros = std::abs(time.micros);

	int32_t time_units[4];
	Time::Convert(time, time_units[0], time_units[1], time_units[2], time_units[3]);

	TimeToStringCast::FormatTwoDigits(buffer + length, time_units[0]);
	length += 2;
	if (time_units[1]) {
		buffer[length++] = ':';
		TimeToStringCast::FormatTwoDigits(buffer + length, time_units[1]);
		length += 2;
	}

	return string(buffer, length);
}

dtime_t Time::FromTime(int32_t hour, int32_t minute, int32_t second, int32_t microseconds) {
	int64_t result;
	result = hour;                                             // hours
	result = result * Interval::MINS_PER_HOUR + minute;        // hours -> minutes
	result = result * Interval::SECS_PER_MINUTE + second;      // minutes -> seconds
	result = result * Interval::MICROS_PER_SEC + microseconds; // seconds -> microseconds
	return dtime_t(result);
}

// LCOV_EXCL_START
#ifdef DEBUG
static bool AssertValidTime(int32_t hour, int32_t minute, int32_t second, int32_t microseconds) {
	if (hour < 0 || hour >= 24) {
		return false;
	}
	if (minute < 0 || minute >= 60) {
		return false;
	}
	if (second < 0 || second > 60) {
		return false;
	}
	if (microseconds < 0 || microseconds > 1000000) {
		return false;
	}
	return true;
}
#endif
// LCOV_EXCL_STOP

void Time::Convert(dtime_t dtime, int32_t &hour, int32_t &min, int32_t &sec, int32_t &micros) {
	int64_t time = dtime.micros;
	hour = int32_t(time / Interval::MICROS_PER_HOUR);
	time -= int64_t(hour) * Interval::MICROS_PER_HOUR;
	min = int32_t(time / Interval::MICROS_PER_MINUTE);
	time -= int64_t(min) * Interval::MICROS_PER_MINUTE;
	sec = int32_t(time / Interval::MICROS_PER_SEC);
	time -= int64_t(sec) * Interval::MICROS_PER_SEC;
	micros = int32_t(time);
#ifdef DEBUG
	D_ASSERT(AssertValidTime(hour, min, sec, micros));
#endif
}

} // namespace duckdb











#include <ctime>

namespace duckdb {

static_assert(sizeof(timestamp_t) == sizeof(int64_t), "timestamp_t was padded");

// timestamp/datetime uses 64 bits, high 32 bits for date and low 32 bits for time
// string format is YYYY-MM-DDThh:mm:ssZ
// T may be a space
// Z is optional
// ISO 8601

bool Timestamp::TryConvertTimestampTZ(const char *str, idx_t len, timestamp_t &result, bool &has_offset, string_t &tz) {
	idx_t pos;
	date_t date;
	dtime_t time;
	has_offset = false;
	if (!Date::TryConvertDate(str, len, pos, date, has_offset)) {
		return false;
	}
	if (pos == len) {
		// no time: only a date or special
		if (date == date_t::infinity()) {
			result = timestamp_t::infinity();
			return true;
		} else if (date == date_t::ninfinity()) {
			result = timestamp_t::ninfinity();
			return true;
		}
		return Timestamp::TryFromDatetime(date, dtime_t(0), result);
	}
	// try to parse a time field
	if (str[pos] == ' ' || str[pos] == 'T') {
		pos++;
	}
	idx_t time_pos = 0;
	if (!Time::TryConvertTime(str + pos, len - pos, time_pos, time)) {
		return false;
	}
	pos += time_pos;
	if (!Timestamp::TryFromDatetime(date, time, result)) {
		return false;
	}
	if (pos < len) {
		// skip a "Z" at the end (as per the ISO8601 specs)
		int hour_offset, minute_offset;
		if (str[pos] == 'Z') {
			pos++;
			has_offset = true;
		} else if (Timestamp::TryParseUTCOffset(str, pos, len, hour_offset, minute_offset)) {
			result -= hour_offset * Interval::MICROS_PER_HOUR + minute_offset * Interval::MICROS_PER_MINUTE;
			has_offset = true;
		} else {
			// Parse a time zone: / [A-Za-z0-9/_]+/
			if (str[pos++] != ' ') {
				return false;
			}
			auto tz_name = str + pos;
			for (; pos < len && CharacterIsTimeZone(str[pos]); ++pos) {
				continue;
			}
			auto tz_len = str + pos - tz_name;
			if (tz_len) {
				tz = string_t(tz_name, tz_len);
			}
			// Note that the caller must reinterpret the instant we return to the given time zone
		}

		// skip any spaces at the end
		while (pos < len && StringUtil::CharacterIsSpace(str[pos])) {
			pos++;
		}
		if (pos < len) {
			return false;
		}
	}
	return true;
}

bool Timestamp::TryConvertTimestamp(const char *str, idx_t len, timestamp_t &result) {
	string_t tz(nullptr, 0);
	bool has_offset = false;
	// We don't understand TZ without an extension, so fail if one was provided.
	return TryConvertTimestampTZ(str, len, result, has_offset, tz) && !tz.GetSize();
}

string Timestamp::ConversionError(const string &str) {
	return StringUtil::Format("timestamp field value out of range: \"%s\", "
	                          "expected format is (YYYY-MM-DD HH:MM:SS[.US][±HH:MM| ZONE])",
	                          str);
}

string Timestamp::ConversionError(string_t str) {
	return Timestamp::ConversionError(str.GetString());
}

timestamp_t Timestamp::FromCString(const char *str, idx_t len) {
	timestamp_t result;
	if (!Timestamp::TryConvertTimestamp(str, len, result)) {
		throw ConversionException(Timestamp::ConversionError(string(str, len)));
	}
	return result;
}

bool Timestamp::TryParseUTCOffset(const char *str, idx_t &pos, idx_t len, int &hour_offset, int &minute_offset) {
	minute_offset = 0;
	idx_t curpos = pos;
	// parse the next 3 characters
	if (curpos + 3 > len) {
		// no characters left to parse
		return false;
	}
	char sign_char = str[curpos];
	if (sign_char != '+' && sign_char != '-') {
		// expected either + or -
		return false;
	}
	curpos++;
	if (!StringUtil::CharacterIsDigit(str[curpos]) || !StringUtil::CharacterIsDigit(str[curpos + 1])) {
		// expected +HH or -HH
		return false;
	}
	hour_offset = (str[curpos] - '0') * 10 + (str[curpos + 1] - '0');
	if (sign_char == '-') {
		hour_offset = -hour_offset;
	}
	curpos += 2;

	// optional minute specifier: expected either "MM" or ":MM"
	if (curpos >= len) {
		// done, nothing left
		pos = curpos;
		return true;
	}
	if (str[curpos] == ':') {
		curpos++;
	}
	if (curpos + 2 > len || !StringUtil::CharacterIsDigit(str[curpos]) ||
	    !StringUtil::CharacterIsDigit(str[curpos + 1])) {
		// no MM specifier
		pos = curpos;
		return true;
	}
	// we have an MM specifier: parse it
	minute_offset = (str[curpos] - '0') * 10 + (str[curpos + 1] - '0');
	if (sign_char == '-') {
		minute_offset = -minute_offset;
	}
	pos = curpos + 2;
	return true;
}

timestamp_t Timestamp::FromString(const string &str) {
	return Timestamp::FromCString(str.c_str(), str.size());
}

string Timestamp::ToString(timestamp_t timestamp) {
	if (timestamp == timestamp_t::infinity()) {
		return Date::PINF;
	} else if (timestamp == timestamp_t::ninfinity()) {
		return Date::NINF;
	}
	date_t date;
	dtime_t time;
	Timestamp::Convert(timestamp, date, time);
	return Date::ToString(date) + " " + Time::ToString(time);
}

date_t Timestamp::GetDate(timestamp_t timestamp) {
	if (timestamp == timestamp_t::infinity()) {
		return date_t::infinity();
	} else if (timestamp == timestamp_t::ninfinity()) {
		return date_t::ninfinity();
	}
	return date_t((timestamp.value + (timestamp.value < 0)) / Interval::MICROS_PER_DAY - (timestamp.value < 0));
}

dtime_t Timestamp::GetTime(timestamp_t timestamp) {
	if (!IsFinite(timestamp)) {
		throw ConversionException("Can't get TIME of infinite TIMESTAMP");
	}
	date_t date = Timestamp::GetDate(timestamp);
	return dtime_t(timestamp.value - (int64_t(date.days) * int64_t(Interval::MICROS_PER_DAY)));
}

bool Timestamp::TryFromDatetime(date_t date, dtime_t time, timestamp_t &result) {
	if (!TryMultiplyOperator::Operation<int64_t, int64_t, int64_t>(date.days, Interval::MICROS_PER_DAY, result.value)) {
		return false;
	}
	if (!TryAddOperator::Operation<int64_t, int64_t, int64_t>(result.value, time.micros, result.value)) {
		return false;
	}
	return Timestamp::IsFinite(result);
}

timestamp_t Timestamp::FromDatetime(date_t date, dtime_t time) {
	timestamp_t result;
	if (!TryFromDatetime(date, time, result)) {
		throw Exception("Overflow exception in date/time -> timestamp conversion");
	}
	return result;
}

void Timestamp::Convert(timestamp_t timestamp, date_t &out_date, dtime_t &out_time) {
	out_date = GetDate(timestamp);
	int64_t days_micros;
	if (!TryMultiplyOperator::Operation<int64_t, int64_t, int64_t>(out_date.days, Interval::MICROS_PER_DAY,
	                                                               days_micros)) {
		throw ConversionException("Date out of range in timestamp conversion");
	}
	out_time = dtime_t(timestamp.value - days_micros);
	D_ASSERT(timestamp == Timestamp::FromDatetime(out_date, out_time));
}

timestamp_t Timestamp::GetCurrentTimestamp() {
	auto now = system_clock::now();
	auto epoch_ms = duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
	return Timestamp::FromEpochMs(epoch_ms);
}

timestamp_t Timestamp::FromEpochSeconds(int64_t sec) {
	int64_t result;
	if (!TryMultiplyOperator::Operation(sec, Interval::MICROS_PER_SEC, result)) {
		throw ConversionException("Could not convert Timestamp(S) to Timestamp(US)");
	}
	return timestamp_t(result);
}

timestamp_t Timestamp::FromEpochMs(int64_t ms) {
	int64_t result;
	if (!TryMultiplyOperator::Operation(ms, Interval::MICROS_PER_MSEC, result)) {
		throw ConversionException("Could not convert Timestamp(MS) to Timestamp(US)");
	}
	return timestamp_t(result);
}

timestamp_t Timestamp::FromEpochMicroSeconds(int64_t micros) {
	return timestamp_t(micros);
}

timestamp_t Timestamp::FromEpochNanoSeconds(int64_t ns) {
	return timestamp_t(ns / 1000);
}

int64_t Timestamp::GetEpochSeconds(timestamp_t timestamp) {
	return timestamp.value / Interval::MICROS_PER_SEC;
}

int64_t Timestamp::GetEpochMs(timestamp_t timestamp) {
	return timestamp.value / Interval::MICROS_PER_MSEC;
}

int64_t Timestamp::GetEpochMicroSeconds(timestamp_t timestamp) {
	return timestamp.value;
}

int64_t Timestamp::GetEpochNanoSeconds(timestamp_t timestamp) {
	int64_t result;
	int64_t ns_in_us = 1000;
	if (!TryMultiplyOperator::Operation(timestamp.value, ns_in_us, result)) {
		throw ConversionException("Could not convert Timestamp(US) to Timestamp(NS)");
	}
	return result;
}

} // namespace duckdb



namespace duckdb {

bool UUID::FromString(string str, hugeint_t &result) {
	auto hex2char = [](char ch) -> unsigned char {
		if (ch >= '0' && ch <= '9') {
			return ch - '0';
		}
		if (ch >= 'a' && ch <= 'f') {
			return 10 + ch - 'a';
		}
		if (ch >= 'A' && ch <= 'F') {
			return 10 + ch - 'A';
		}
		return 0;
	};
	auto is_hex = [](char ch) -> bool {
		return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F');
	};

	if (str.empty()) {
		return false;
	}
	int has_braces = 0;
	if (str.front() == '{') {
		has_braces = 1;
	}
	if (has_braces && str.back() != '}') {
		return false;
	}

	result.lower = 0;
	result.upper = 0;
	size_t count = 0;
	for (size_t i = has_braces; i < str.size() - has_braces; ++i) {
		if (str[i] == '-') {
			continue;
		}
		if (count >= 32 || !is_hex(str[i])) {
			return false;
		}
		if (count >= 16) {
			result.lower = (result.lower << 4) | hex2char(str[i]);
		} else {
			result.upper = (result.upper << 4) | hex2char(str[i]);
		}
		count++;
	}
	// Flip the first bit to make `order by uuid` same as `order by uuid::varchar`
	result.upper ^= (int64_t(1) << 63);
	return count == 32;
}

void UUID::ToString(hugeint_t input, char *buf) {
	auto byte_to_hex = [](char byte_val, char *buf, idx_t &pos) {
		static char const HEX_DIGITS[] = "0123456789abcdef";
		buf[pos++] = HEX_DIGITS[(byte_val >> 4) & 0xf];
		buf[pos++] = HEX_DIGITS[byte_val & 0xf];
	};

	// Flip back before convert to string
	int64_t upper = input.upper ^ (int64_t(1) << 63);
	idx_t pos = 0;
	byte_to_hex(upper >> 56 & 0xFF, buf, pos);
	byte_to_hex(upper >> 48 & 0xFF, buf, pos);
	byte_to_hex(upper >> 40 & 0xFF, buf, pos);
	byte_to_hex(upper >> 32 & 0xFF, buf, pos);
	buf[pos++] = '-';
	byte_to_hex(upper >> 24 & 0xFF, buf, pos);
	byte_to_hex(upper >> 16 & 0xFF, buf, pos);
	buf[pos++] = '-';
	byte_to_hex(upper >> 8 & 0xFF, buf, pos);
	byte_to_hex(upper & 0xFF, buf, pos);
	buf[pos++] = '-';
	byte_to_hex(input.lower >> 56 & 0xFF, buf, pos);
	byte_to_hex(input.lower >> 48 & 0xFF, buf, pos);
	buf[pos++] = '-';
	byte_to_hex(input.lower >> 40 & 0xFF, buf, pos);
	byte_to_hex(input.lower >> 32 & 0xFF, buf, pos);
	byte_to_hex(input.lower >> 24 & 0xFF, buf, pos);
	byte_to_hex(input.lower >> 16 & 0xFF, buf, pos);
	byte_to_hex(input.lower >> 8 & 0xFF, buf, pos);
	byte_to_hex(input.lower & 0xFF, buf, pos);
}

hugeint_t UUID::GenerateRandomUUID(RandomEngine &engine) {
	uint8_t bytes[16];
	for (int i = 0; i < 16; i += 4) {
		*reinterpret_cast<uint32_t *>(bytes + i) = engine.NextRandomInteger();
	}
	// variant must be 10xxxxxx
	bytes[8] &= 0xBF;
	bytes[8] |= 0x80;
	// version must be 0100xxxx
	bytes[6] &= 0x4F;
	bytes[6] |= 0x40;

	hugeint_t result;
	result.upper = 0;
	result.upper |= ((int64_t)bytes[0] << 56);
	result.upper |= ((int64_t)bytes[1] << 48);
	result.upper |= ((int64_t)bytes[2] << 40);
	result.upper |= ((int64_t)bytes[3] << 32);
	result.upper |= ((int64_t)bytes[4] << 24);
	result.upper |= ((int64_t)bytes[5] << 16);
	result.upper |= ((int64_t)bytes[6] << 8);
	result.upper |= bytes[7];
	result.lower = 0;
	result.lower |= ((uint64_t)bytes[8] << 56);
	result.lower |= ((uint64_t)bytes[9] << 48);
	result.lower |= ((uint64_t)bytes[10] << 40);
	result.lower |= ((uint64_t)bytes[11] << 32);
	result.lower |= ((uint64_t)bytes[12] << 24);
	result.lower |= ((uint64_t)bytes[13] << 16);
	result.lower |= ((uint64_t)bytes[14] << 8);
	result.lower |= bytes[15];
	return result;
}

hugeint_t UUID::GenerateRandomUUID() {
	RandomEngine engine;
	return GenerateRandomUUID(engine);
}

} // namespace duckdb


namespace duckdb {

ValidityData::ValidityData(idx_t count) : TemplatedValidityData(count) {
}
ValidityData::ValidityData(const ValidityMask &original, idx_t count)
    : TemplatedValidityData(original.GetData(), count) {
}

void ValidityMask::Combine(const ValidityMask &other, idx_t count) {
	if (other.AllValid()) {
		// X & 1 = X
		return;
	}
	if (AllValid()) {
		// 1 & Y = Y
		Initialize(other);
		return;
	}
	if (validity_mask == other.validity_mask) {
		// X & X == X
		return;
	}
	// have to merge
	// create a new validity mask that contains the combined mask
	auto owned_data = std::move(validity_data);
	auto data = GetData();
	auto other_data = other.GetData();

	Initialize(count);
	auto result_data = GetData();

	auto entry_count = ValidityData::EntryCount(count);
	for (idx_t entry_idx = 0; entry_idx < entry_count; entry_idx++) {
		result_data[entry_idx] = data[entry_idx] & other_data[entry_idx];
	}
}

// LCOV_EXCL_START
string ValidityMask::ToString(idx_t count) const {
	string result = "Validity Mask (" + to_string(count) + ") [";
	for (idx_t i = 0; i < count; i++) {
		result += RowIsValid(i) ? "." : "X";
	}
	result += "]";
	return result;
}
// LCOV_EXCL_STOP

void ValidityMask::Resize(idx_t old_size, idx_t new_size) {
	D_ASSERT(new_size >= old_size);
	if (validity_mask) {
		auto new_size_count = EntryCount(new_size);
		auto old_size_count = EntryCount(old_size);
		auto new_validity_data = make_buffer<ValidityBuffer>(new_size);
		auto new_owned_data = new_validity_data->owned_data.get();
		for (idx_t entry_idx = 0; entry_idx < old_size_count; entry_idx++) {
			new_owned_data[entry_idx] = validity_mask[entry_idx];
		}
		for (idx_t entry_idx = old_size_count; entry_idx < new_size_count; entry_idx++) {
			new_owned_data[entry_idx] = ValidityData::MAX_ENTRY;
		}
		validity_data = std::move(new_validity_data);
		validity_mask = validity_data->owned_data.get();
	} else {
		Initialize(new_size);
	}
}

void ValidityMask::Slice(const ValidityMask &other, idx_t offset, idx_t end) {
	if (other.AllValid()) {
		validity_mask = nullptr;
		validity_data.reset();
		return;
	}
	if (offset == 0) {
		Initialize(other);
		return;
	}
	ValidityMask new_mask(end - offset);

// FIXME THIS NEEDS FIXING!
#if 1
	for (idx_t i = offset; i < end; i++) {
		new_mask.Set(i - offset, other.RowIsValid(i));
	}
	Initialize(new_mask);
#else
	// first shift the "whole" units
	idx_t entire_units = offset / BITS_PER_VALUE;
	idx_t sub_units = offset - entire_units * BITS_PER_VALUE;
	if (entire_units > 0) {
		idx_t validity_idx;
		for (validity_idx = 0; validity_idx + entire_units < STANDARD_ENTRY_COUNT; validity_idx++) {
			new_mask.validity_mask[validity_idx] = other.validity_mask[validity_idx + entire_units];
		}
	}
	// now we shift the remaining sub units
	// this gets a bit more complicated because we have to shift over the borders of the entries
	// e.g. suppose we have 2 entries of length 4 and we left-shift by two
	// 0101|1010
	// a regular left-shift of both gets us:
	// 0100|1000
	// we then OR the overflow (right-shifted by BITS_PER_VALUE - offset) together to get the correct result
	// 0100|1000 ->
	// 0110|1000
	if (sub_units > 0) {
		idx_t validity_idx;
		for (validity_idx = 0; validity_idx + 1 < STANDARD_ENTRY_COUNT; validity_idx++) {
			new_mask.validity_mask[validity_idx] =
			    (other.validity_mask[validity_idx] >> sub_units) |
			    (other.validity_mask[validity_idx + 1] << (BITS_PER_VALUE - sub_units));
		}
		new_mask.validity_mask[validity_idx] >>= sub_units;
	}
#ifdef DEBUG
	for (idx_t i = offset; i < STANDARD_VECTOR_SIZE; i++) {
		D_ASSERT(new_mask.RowIsValid(i - offset) == other.RowIsValid(i));
	}
	Initialize(new_mask);
#endif
#endif
}

} // namespace duckdb
































#include <utility>
#include <cmath>

namespace duckdb {

Value::Value(LogicalType type) : type_(std::move(type)), is_null(true) {
}

Value::Value(int32_t val) : type_(LogicalType::INTEGER), is_null(false) {
	value_.integer = val;
}

Value::Value(int64_t val) : type_(LogicalType::BIGINT), is_null(false) {
	value_.bigint = val;
}

Value::Value(float val) : type_(LogicalType::FLOAT), is_null(false) {
	value_.float_ = val;
}

Value::Value(double val) : type_(LogicalType::DOUBLE), is_null(false) {
	value_.double_ = val;
}

Value::Value(const char *val) : Value(val ? string(val) : string()) {
}

Value::Value(std::nullptr_t val) : Value(LogicalType::VARCHAR) {
}

Value::Value(string_t val) : Value(string(val.GetDataUnsafe(), val.GetSize())) {
}

Value::Value(string val) : type_(LogicalType::VARCHAR), is_null(false), str_value(std::move(val)) {
	if (!Value::StringIsValid(str_value.c_str(), str_value.size())) {
		throw Exception(ErrorManager::InvalidUnicodeError(str_value, "value construction"));
	}
}

Value::~Value() {
}

Value::Value(const Value &other)
    : type_(other.type_), is_null(other.is_null), value_(other.value_), str_value(other.str_value),
      struct_value(other.struct_value), list_value(other.list_value) {
}

Value::Value(Value &&other) noexcept
    : type_(std::move(other.type_)), is_null(other.is_null), value_(other.value_),
      str_value(std::move(other.str_value)), struct_value(std::move(other.struct_value)),
      list_value(std::move(other.list_value)) {
}

Value &Value::operator=(const Value &other) {
	type_ = other.type_;
	is_null = other.is_null;
	value_ = other.value_;
	str_value = other.str_value;
	struct_value = other.struct_value;
	list_value = other.list_value;
	return *this;
}

Value &Value::operator=(Value &&other) noexcept {
	type_ = std::move(other.type_);
	is_null = other.is_null;
	value_ = other.value_;
	str_value = std::move(other.str_value);
	struct_value = std::move(other.struct_value);
	list_value = std::move(other.list_value);
	return *this;
}

Value Value::MinimumValue(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::BOOLEAN:
		return Value::BOOLEAN(false);
	case LogicalTypeId::TINYINT:
		return Value::TINYINT(NumericLimits<int8_t>::Minimum());
	case LogicalTypeId::SMALLINT:
		return Value::SMALLINT(NumericLimits<int16_t>::Minimum());
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::SQLNULL:
		return Value::INTEGER(NumericLimits<int32_t>::Minimum());
	case LogicalTypeId::BIGINT:
		return Value::BIGINT(NumericLimits<int64_t>::Minimum());
	case LogicalTypeId::HUGEINT:
		return Value::HUGEINT(NumericLimits<hugeint_t>::Minimum());
	case LogicalTypeId::UUID:
		return Value::UUID(NumericLimits<hugeint_t>::Minimum());
	case LogicalTypeId::UTINYINT:
		return Value::UTINYINT(NumericLimits<uint8_t>::Minimum());
	case LogicalTypeId::USMALLINT:
		return Value::USMALLINT(NumericLimits<uint16_t>::Minimum());
	case LogicalTypeId::UINTEGER:
		return Value::UINTEGER(NumericLimits<uint32_t>::Minimum());
	case LogicalTypeId::UBIGINT:
		return Value::UBIGINT(NumericLimits<uint64_t>::Minimum());
	case LogicalTypeId::DATE:
		return Value::DATE(Date::FromDate(Date::DATE_MIN_YEAR, Date::DATE_MIN_MONTH, Date::DATE_MIN_DAY));
	case LogicalTypeId::TIME:
		return Value::TIME(dtime_t(0));
	case LogicalTypeId::TIMESTAMP:
		return Value::TIMESTAMP(Date::FromDate(Timestamp::MIN_YEAR, Timestamp::MIN_MONTH, Timestamp::MIN_DAY),
		                        dtime_t(0));
	case LogicalTypeId::TIMESTAMP_SEC:
		return MinimumValue(LogicalType::TIMESTAMP).DefaultCastAs(LogicalType::TIMESTAMP_S);
	case LogicalTypeId::TIMESTAMP_MS:
		return MinimumValue(LogicalType::TIMESTAMP).DefaultCastAs(LogicalType::TIMESTAMP_MS);
	case LogicalTypeId::TIMESTAMP_NS:
		return Value::TIMESTAMPNS(timestamp_t(NumericLimits<int64_t>::Minimum()));
	case LogicalTypeId::TIME_TZ:
		return Value::TIMETZ(dtime_t(0));
	case LogicalTypeId::TIMESTAMP_TZ:
		return Value::TIMESTAMPTZ(Timestamp::FromDatetime(
		    Date::FromDate(Timestamp::MIN_YEAR, Timestamp::MIN_MONTH, Timestamp::MIN_DAY), dtime_t(0)));
	case LogicalTypeId::FLOAT:
		return Value::FLOAT(NumericLimits<float>::Minimum());
	case LogicalTypeId::DOUBLE:
		return Value::DOUBLE(NumericLimits<double>::Minimum());
	case LogicalTypeId::DECIMAL: {
		auto width = DecimalType::GetWidth(type);
		auto scale = DecimalType::GetScale(type);
		switch (type.InternalType()) {
		case PhysicalType::INT16:
			return Value::DECIMAL(int16_t(-NumericHelper::POWERS_OF_TEN[width] + 1), width, scale);
		case PhysicalType::INT32:
			return Value::DECIMAL(int32_t(-NumericHelper::POWERS_OF_TEN[width] + 1), width, scale);
		case PhysicalType::INT64:
			return Value::DECIMAL(int64_t(-NumericHelper::POWERS_OF_TEN[width] + 1), width, scale);
		case PhysicalType::INT128:
			return Value::DECIMAL(-Hugeint::POWERS_OF_TEN[width] + 1, width, scale);
		default:
			throw InternalException("Unknown decimal type");
		}
	}
	case LogicalTypeId::ENUM:
		return Value::ENUM(0, type);
	default:
		throw InvalidTypeException(type, "MinimumValue requires numeric type");
	}
}

Value Value::MaximumValue(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::BOOLEAN:
		return Value::BOOLEAN(true);
	case LogicalTypeId::TINYINT:
		return Value::TINYINT(NumericLimits<int8_t>::Maximum());
	case LogicalTypeId::SMALLINT:
		return Value::SMALLINT(NumericLimits<int16_t>::Maximum());
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::SQLNULL:
		return Value::INTEGER(NumericLimits<int32_t>::Maximum());
	case LogicalTypeId::BIGINT:
		return Value::BIGINT(NumericLimits<int64_t>::Maximum());
	case LogicalTypeId::HUGEINT:
		return Value::HUGEINT(NumericLimits<hugeint_t>::Maximum());
	case LogicalTypeId::UUID:
		return Value::UUID(NumericLimits<hugeint_t>::Maximum());
	case LogicalTypeId::UTINYINT:
		return Value::UTINYINT(NumericLimits<uint8_t>::Maximum());
	case LogicalTypeId::USMALLINT:
		return Value::USMALLINT(NumericLimits<uint16_t>::Maximum());
	case LogicalTypeId::UINTEGER:
		return Value::UINTEGER(NumericLimits<uint32_t>::Maximum());
	case LogicalTypeId::UBIGINT:
		return Value::UBIGINT(NumericLimits<uint64_t>::Maximum());
	case LogicalTypeId::DATE:
		return Value::DATE(Date::FromDate(Date::DATE_MAX_YEAR, Date::DATE_MAX_MONTH, Date::DATE_MAX_DAY));
	case LogicalTypeId::TIME:
		return Value::TIME(dtime_t(Interval::SECS_PER_DAY * Interval::MICROS_PER_SEC - 1));
	case LogicalTypeId::TIMESTAMP:
		return Value::TIMESTAMP(timestamp_t(NumericLimits<int64_t>::Maximum() - 1));
	case LogicalTypeId::TIMESTAMP_MS:
		return MaximumValue(LogicalType::TIMESTAMP).DefaultCastAs(LogicalType::TIMESTAMP_MS);
	case LogicalTypeId::TIMESTAMP_NS:
		return Value::TIMESTAMPNS(timestamp_t(NumericLimits<int64_t>::Maximum() - 1));
	case LogicalTypeId::TIMESTAMP_SEC:
		return MaximumValue(LogicalType::TIMESTAMP).DefaultCastAs(LogicalType::TIMESTAMP_S);
	case LogicalTypeId::TIME_TZ:
		return Value::TIMETZ(dtime_t(Interval::SECS_PER_DAY * Interval::MICROS_PER_SEC - 1));
	case LogicalTypeId::TIMESTAMP_TZ:
		return MaximumValue(LogicalType::TIMESTAMP);
	case LogicalTypeId::FLOAT:
		return Value::FLOAT(NumericLimits<float>::Maximum());
	case LogicalTypeId::DOUBLE:
		return Value::DOUBLE(NumericLimits<double>::Maximum());
	case LogicalTypeId::DECIMAL: {
		auto width = DecimalType::GetWidth(type);
		auto scale = DecimalType::GetScale(type);
		switch (type.InternalType()) {
		case PhysicalType::INT16:
			return Value::DECIMAL(int16_t(NumericHelper::POWERS_OF_TEN[width] - 1), width, scale);
		case PhysicalType::INT32:
			return Value::DECIMAL(int32_t(NumericHelper::POWERS_OF_TEN[width] - 1), width, scale);
		case PhysicalType::INT64:
			return Value::DECIMAL(int64_t(NumericHelper::POWERS_OF_TEN[width] - 1), width, scale);
		case PhysicalType::INT128:
			return Value::DECIMAL(Hugeint::POWERS_OF_TEN[width] - 1, width, scale);
		default:
			throw InternalException("Unknown decimal type");
		}
	}
	case LogicalTypeId::ENUM:
		return Value::ENUM(EnumType::GetSize(type) - 1, type);
	default:
		throw InvalidTypeException(type, "MaximumValue requires numeric type");
	}
}

Value Value::BOOLEAN(int8_t value) {
	Value result(LogicalType::BOOLEAN);
	result.value_.boolean = value ? true : false;
	result.is_null = false;
	return result;
}

Value Value::TINYINT(int8_t value) {
	Value result(LogicalType::TINYINT);
	result.value_.tinyint = value;
	result.is_null = false;
	return result;
}

Value Value::SMALLINT(int16_t value) {
	Value result(LogicalType::SMALLINT);
	result.value_.smallint = value;
	result.is_null = false;
	return result;
}

Value Value::INTEGER(int32_t value) {
	Value result(LogicalType::INTEGER);
	result.value_.integer = value;
	result.is_null = false;
	return result;
}

Value Value::BIGINT(int64_t value) {
	Value result(LogicalType::BIGINT);
	result.value_.bigint = value;
	result.is_null = false;
	return result;
}

Value Value::HUGEINT(hugeint_t value) {
	Value result(LogicalType::HUGEINT);
	result.value_.hugeint = value;
	result.is_null = false;
	return result;
}

Value Value::UUID(hugeint_t value) {
	Value result(LogicalType::UUID);
	result.value_.hugeint = value;
	result.is_null = false;
	return result;
}

Value Value::UUID(const string &value) {
	Value result(LogicalType::UUID);
	result.value_.hugeint = UUID::FromString(value);
	result.is_null = false;
	return result;
}

Value Value::UTINYINT(uint8_t value) {
	Value result(LogicalType::UTINYINT);
	result.value_.utinyint = value;
	result.is_null = false;
	return result;
}

Value Value::USMALLINT(uint16_t value) {
	Value result(LogicalType::USMALLINT);
	result.value_.usmallint = value;
	result.is_null = false;
	return result;
}

Value Value::UINTEGER(uint32_t value) {
	Value result(LogicalType::UINTEGER);
	result.value_.uinteger = value;
	result.is_null = false;
	return result;
}

Value Value::UBIGINT(uint64_t value) {
	Value result(LogicalType::UBIGINT);
	result.value_.ubigint = value;
	result.is_null = false;
	return result;
}

bool Value::FloatIsFinite(float value) {
	return !(std::isnan(value) || std::isinf(value));
}

bool Value::DoubleIsFinite(double value) {
	return !(std::isnan(value) || std::isinf(value));
}

template <>
bool Value::IsNan(float input) {
	return std::isnan(input);
}

template <>
bool Value::IsNan(double input) {
	return std::isnan(input);
}

template <>
bool Value::IsFinite(float input) {
	return Value::FloatIsFinite(input);
}

template <>
bool Value::IsFinite(double input) {
	return Value::DoubleIsFinite(input);
}

template <>
bool Value::IsFinite(date_t input) {
	return Date::IsFinite(input);
}

template <>
bool Value::IsFinite(timestamp_t input) {
	return Timestamp::IsFinite(input);
}

bool Value::StringIsValid(const char *str, idx_t length) {
	auto utf_type = Utf8Proc::Analyze(str, length);
	return utf_type != UnicodeType::INVALID;
}

Value Value::DECIMAL(int16_t value, uint8_t width, uint8_t scale) {
	return Value::DECIMAL(int64_t(value), width, scale);
}

Value Value::DECIMAL(int32_t value, uint8_t width, uint8_t scale) {
	return Value::DECIMAL(int64_t(value), width, scale);
}

Value Value::DECIMAL(int64_t value, uint8_t width, uint8_t scale) {
	auto decimal_type = LogicalType::DECIMAL(width, scale);
	Value result(decimal_type);
	switch (decimal_type.InternalType()) {
	case PhysicalType::INT16:
		result.value_.smallint = value;
		break;
	case PhysicalType::INT32:
		result.value_.integer = value;
		break;
	case PhysicalType::INT64:
		result.value_.bigint = value;
		break;
	default:
		result.value_.hugeint = value;
		break;
	}
	result.type_.Verify();
	result.is_null = false;
	return result;
}

Value Value::DECIMAL(hugeint_t value, uint8_t width, uint8_t scale) {
	D_ASSERT(width >= Decimal::MAX_WIDTH_INT64 && width <= Decimal::MAX_WIDTH_INT128);
	Value result(LogicalType::DECIMAL(width, scale));
	result.value_.hugeint = value;
	result.is_null = false;
	return result;
}

Value Value::FLOAT(float value) {
	Value result(LogicalType::FLOAT);
	result.value_.float_ = value;
	result.is_null = false;
	return result;
}

Value Value::DOUBLE(double value) {
	Value result(LogicalType::DOUBLE);
	result.value_.double_ = value;
	result.is_null = false;
	return result;
}

Value Value::HASH(hash_t value) {
	Value result(LogicalType::HASH);
	result.value_.hash = value;
	result.is_null = false;
	return result;
}

Value Value::POINTER(uintptr_t value) {
	Value result(LogicalType::POINTER);
	result.value_.pointer = value;
	result.is_null = false;
	return result;
}

Value Value::DATE(date_t value) {
	Value result(LogicalType::DATE);
	result.value_.date = value;
	result.is_null = false;
	return result;
}

Value Value::DATE(int32_t year, int32_t month, int32_t day) {
	return Value::DATE(Date::FromDate(year, month, day));
}

Value Value::TIME(dtime_t value) {
	Value result(LogicalType::TIME);
	result.value_.time = value;
	result.is_null = false;
	return result;
}

Value Value::TIMETZ(dtime_t value) {
	Value result(LogicalType::TIME_TZ);
	result.value_.time = value;
	result.is_null = false;
	return result;
}

Value Value::TIME(int32_t hour, int32_t min, int32_t sec, int32_t micros) {
	return Value::TIME(Time::FromTime(hour, min, sec, micros));
}

Value Value::TIMESTAMP(timestamp_t value) {
	Value result(LogicalType::TIMESTAMP);
	result.value_.timestamp = value;
	result.is_null = false;
	return result;
}

Value Value::TIMESTAMPTZ(timestamp_t value) {
	Value result(LogicalType::TIMESTAMP_TZ);
	result.value_.timestamp = value;
	result.is_null = false;
	return result;
}

Value Value::TIMESTAMPNS(timestamp_t timestamp) {
	Value result(LogicalType::TIMESTAMP_NS);
	result.value_.timestamp = timestamp;
	result.is_null = false;
	return result;
}

Value Value::TIMESTAMPMS(timestamp_t timestamp) {
	Value result(LogicalType::TIMESTAMP_MS);
	result.value_.timestamp = timestamp;
	result.is_null = false;
	return result;
}

Value Value::TIMESTAMPSEC(timestamp_t timestamp) {
	Value result(LogicalType::TIMESTAMP_S);
	result.value_.timestamp = timestamp;
	result.is_null = false;
	return result;
}

Value Value::TIMESTAMP(date_t date, dtime_t time) {
	return Value::TIMESTAMP(Timestamp::FromDatetime(date, time));
}

Value Value::TIMESTAMP(int32_t year, int32_t month, int32_t day, int32_t hour, int32_t min, int32_t sec,
                       int32_t micros) {
	auto val = Value::TIMESTAMP(Date::FromDate(year, month, day), Time::FromTime(hour, min, sec, micros));
	val.type_ = LogicalType::TIMESTAMP;
	return val;
}

Value Value::STRUCT(child_list_t<Value> values) {
	Value result;
	child_list_t<LogicalType> child_types;
	for (auto &child : values) {
		child_types.push_back(make_pair(std::move(child.first), child.second.type()));
		result.struct_value.push_back(std::move(child.second));
	}
	result.type_ = LogicalType::STRUCT(std::move(child_types));

	result.is_null = false;
	return result;
}

Value Value::MAP(LogicalType child_type, vector<Value> values) {
	Value result;

	result.type_ = LogicalType::MAP(std::move(child_type));
	result.is_null = false;
	if (values.empty()) {
		return result;
	}
	result.list_value = std::move(values);
	return result;
}

Value Value::UNION(child_list_t<LogicalType> members, uint8_t tag, Value value) {
	D_ASSERT(members.size() > 0);
	D_ASSERT(members.size() <= UnionType::MAX_UNION_MEMBERS);
	D_ASSERT(members.size() > tag);

	D_ASSERT(value.type() == members[tag].second);

	Value result;
	result.is_null = false;
	// add the tag to the front of the struct
	result.struct_value.emplace_back(Value::TINYINT(tag));
	for (idx_t i = 0; i < members.size(); i++) {
		if (i != tag) {
			result.struct_value.emplace_back(members[i].second);
		} else {
			result.struct_value.emplace_back(nullptr);
		}
	}
	result.struct_value[tag + 1] = std::move(value);

	result.type_ = LogicalType::UNION(std::move(members));
	return result;
}

Value Value::LIST(vector<Value> values) {
	if (values.empty()) {
		throw InternalException("Value::LIST without providing a child-type requires a non-empty list of values. Use "
		                        "Value::LIST(child_type, list) instead.");
	}
#ifdef DEBUG
	for (idx_t i = 1; i < values.size(); i++) {
		D_ASSERT(values[i].type() == values[0].type());
	}
#endif
	Value result;
	result.type_ = LogicalType::LIST(values[0].type());
	result.list_value = std::move(values);
	result.is_null = false;
	return result;
}

Value Value::LIST(LogicalType child_type, vector<Value> values) {
	if (values.empty()) {
		return Value::EMPTYLIST(std::move(child_type));
	}
	for (auto &val : values) {
		val = val.DefaultCastAs(child_type);
	}
	return Value::LIST(std::move(values));
}

Value Value::EMPTYLIST(LogicalType child_type) {
	Value result;
	result.type_ = LogicalType::LIST(std::move(child_type));
	result.is_null = false;
	return result;
}

Value Value::BLOB(const_data_ptr_t data, idx_t len) {
	Value result(LogicalType::BLOB);
	result.is_null = false;
	result.str_value = string((const char *)data, len);
	return result;
}

Value Value::BLOB(const string &data) {
	Value result(LogicalType::BLOB);
	result.is_null = false;
	result.str_value = Blob::ToBlob(string_t(data));
	return result;
}

Value Value::BIT(const_data_ptr_t data, idx_t len) {
	Value result(LogicalType::BIT);
	result.is_null = false;
	result.str_value = string((const char *)data, len);
	return result;
}

Value Value::BIT(const string &data) {
	Value result(LogicalType::BIT);
	result.is_null = false;
	result.str_value = Bit::ToBit(string_t(data));
	return result;
}

Value Value::ENUM(uint64_t value, const LogicalType &original_type) {
	D_ASSERT(original_type.id() == LogicalTypeId::ENUM);
	Value result(original_type);
	switch (original_type.InternalType()) {
	case PhysicalType::UINT8:
		result.value_.utinyint = value;
		break;
	case PhysicalType::UINT16:
		result.value_.usmallint = value;
		break;
	case PhysicalType::UINT32:
		result.value_.uinteger = value;
		break;
	default:
		throw InternalException("Incorrect Physical Type for ENUM");
	}
	result.is_null = false;
	return result;
}

Value Value::INTERVAL(int32_t months, int32_t days, int64_t micros) {
	Value result(LogicalType::INTERVAL);
	result.is_null = false;
	result.value_.interval.months = months;
	result.value_.interval.days = days;
	result.value_.interval.micros = micros;
	return result;
}

Value Value::INTERVAL(interval_t interval) {
	return Value::INTERVAL(interval.months, interval.days, interval.micros);
}

//===--------------------------------------------------------------------===//
// CreateValue
//===--------------------------------------------------------------------===//
template <>
Value Value::CreateValue(bool value) {
	return Value::BOOLEAN(value);
}

template <>
Value Value::CreateValue(int8_t value) {
	return Value::TINYINT(value);
}

template <>
Value Value::CreateValue(int16_t value) {
	return Value::SMALLINT(value);
}

template <>
Value Value::CreateValue(int32_t value) {
	return Value::INTEGER(value);
}

template <>
Value Value::CreateValue(int64_t value) {
	return Value::BIGINT(value);
}

template <>
Value Value::CreateValue(uint8_t value) {
	return Value::UTINYINT(value);
}

template <>
Value Value::CreateValue(uint16_t value) {
	return Value::USMALLINT(value);
}

template <>
Value Value::CreateValue(uint32_t value) {
	return Value::UINTEGER(value);
}

template <>
Value Value::CreateValue(uint64_t value) {
	return Value::UBIGINT(value);
}

template <>
Value Value::CreateValue(hugeint_t value) {
	return Value::HUGEINT(value);
}

template <>
Value Value::CreateValue(date_t value) {
	return Value::DATE(value);
}

template <>
Value Value::CreateValue(dtime_t value) {
	return Value::TIME(value);
}

template <>
Value Value::CreateValue(dtime_tz_t value) {
	return Value::TIMETZ(value);
}

template <>
Value Value::CreateValue(timestamp_t value) {
	return Value::TIMESTAMP(value);
}

template <>
Value Value::CreateValue(timestamp_sec_t value) {
	return Value::TIMESTAMPSEC(value);
}

template <>
Value Value::CreateValue(timestamp_ms_t value) {
	return Value::TIMESTAMPMS(value);
}

template <>
Value Value::CreateValue(timestamp_ns_t value) {
	return Value::TIMESTAMPNS(value);
}

template <>
Value Value::CreateValue(timestamp_tz_t value) {
	return Value::TIMESTAMPTZ(value);
}

template <>
Value Value::CreateValue(const char *value) {
	return Value(string(value));
}

template <>
Value Value::CreateValue(string value) { // NOLINT: required for templating
	return Value::BLOB(value);
}

template <>
Value Value::CreateValue(string_t value) {
	return Value(value);
}

template <>
Value Value::CreateValue(float value) {
	return Value::FLOAT(value);
}

template <>
Value Value::CreateValue(double value) {
	return Value::DOUBLE(value);
}

template <>
Value Value::CreateValue(interval_t value) {
	return Value::INTERVAL(value);
}

template <>
Value Value::CreateValue(Value value) {
	return value;
}

//===--------------------------------------------------------------------===//
// GetValue
//===--------------------------------------------------------------------===//
template <class T>
T Value::GetValueInternal() const {
	if (IsNull()) {
		throw InternalException("Calling GetValueInternal on a value that is NULL");
	}
	switch (type_.id()) {
	case LogicalTypeId::BOOLEAN:
		return Cast::Operation<bool, T>(value_.boolean);
	case LogicalTypeId::TINYINT:
		return Cast::Operation<int8_t, T>(value_.tinyint);
	case LogicalTypeId::SMALLINT:
		return Cast::Operation<int16_t, T>(value_.smallint);
	case LogicalTypeId::INTEGER:
		return Cast::Operation<int32_t, T>(value_.integer);
	case LogicalTypeId::BIGINT:
		return Cast::Operation<int64_t, T>(value_.bigint);
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::UUID:
		return Cast::Operation<hugeint_t, T>(value_.hugeint);
	case LogicalTypeId::DATE:
		return Cast::Operation<date_t, T>(value_.date);
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_TZ:
		return Cast::Operation<dtime_t, T>(value_.time);
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ:
		return Cast::Operation<timestamp_t, T>(value_.timestamp);
	case LogicalTypeId::UTINYINT:
		return Cast::Operation<uint8_t, T>(value_.utinyint);
	case LogicalTypeId::USMALLINT:
		return Cast::Operation<uint16_t, T>(value_.usmallint);
	case LogicalTypeId::UINTEGER:
		return Cast::Operation<uint32_t, T>(value_.uinteger);
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIMESTAMP_NS:
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::UBIGINT:
		return Cast::Operation<uint64_t, T>(value_.ubigint);
	case LogicalTypeId::FLOAT:
		return Cast::Operation<float, T>(value_.float_);
	case LogicalTypeId::DOUBLE:
		return Cast::Operation<double, T>(value_.double_);
	case LogicalTypeId::VARCHAR:
		return Cast::Operation<string_t, T>(str_value.c_str());
	case LogicalTypeId::INTERVAL:
		return Cast::Operation<interval_t, T>(value_.interval);
	case LogicalTypeId::DECIMAL:
		return DefaultCastAs(LogicalType::DOUBLE).GetValueInternal<T>();
	case LogicalTypeId::ENUM: {
		switch (type_.InternalType()) {
		case PhysicalType::UINT8:
			return Cast::Operation<uint8_t, T>(value_.utinyint);
		case PhysicalType::UINT16:
			return Cast::Operation<uint16_t, T>(value_.usmallint);
		case PhysicalType::UINT32:
			return Cast::Operation<uint32_t, T>(value_.uinteger);
		default:
			throw InternalException("Invalid Internal Type for ENUMs");
		}
	}
	default:
		throw NotImplementedException("Unimplemented type \"%s\" for GetValue()", type_.ToString());
	}
}

template <>
bool Value::GetValue() const {
	return GetValueInternal<int8_t>();
}
template <>
int8_t Value::GetValue() const {
	return GetValueInternal<int8_t>();
}
template <>
int16_t Value::GetValue() const {
	return GetValueInternal<int16_t>();
}
template <>
int32_t Value::GetValue() const {
	if (type_.id() == LogicalTypeId::DATE) {
		return value_.integer;
	}
	return GetValueInternal<int32_t>();
}
template <>
int64_t Value::GetValue() const {
	if (IsNull()) {
		throw InternalException("Calling GetValue on a value that is NULL");
	}
	switch (type_.id()) {
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::TIMESTAMP_NS:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_TZ:
	case LogicalTypeId::TIMESTAMP_TZ:
		return value_.bigint;
	default:
		return GetValueInternal<int64_t>();
	}
}
template <>
hugeint_t Value::GetValue() const {
	return GetValueInternal<hugeint_t>();
}
template <>
uint8_t Value::GetValue() const {
	return GetValueInternal<uint8_t>();
}
template <>
uint16_t Value::GetValue() const {
	return GetValueInternal<uint16_t>();
}
template <>
uint32_t Value::GetValue() const {
	return GetValueInternal<uint32_t>();
}
template <>
uint64_t Value::GetValue() const {
	return GetValueInternal<uint64_t>();
}
template <>
string Value::GetValue() const {
	return ToString();
}
template <>
float Value::GetValue() const {
	return GetValueInternal<float>();
}
template <>
double Value::GetValue() const {
	return GetValueInternal<double>();
}
template <>
date_t Value::GetValue() const {
	return GetValueInternal<date_t>();
}
template <>
dtime_t Value::GetValue() const {
	return GetValueInternal<dtime_t>();
}
template <>
timestamp_t Value::GetValue() const {
	return GetValueInternal<timestamp_t>();
}

template <>
DUCKDB_API interval_t Value::GetValue() const {
	return GetValueInternal<interval_t>();
}

template <>
DUCKDB_API Value Value::GetValue() const {
	return Value(*this);
}

uintptr_t Value::GetPointer() const {
	D_ASSERT(type() == LogicalType::POINTER);
	return value_.pointer;
}

Value Value::Numeric(const LogicalType &type, int64_t value) {
	switch (type.id()) {
	case LogicalTypeId::BOOLEAN:
		D_ASSERT(value == 0 || value == 1);
		return Value::BOOLEAN(value ? 1 : 0);
	case LogicalTypeId::TINYINT:
		D_ASSERT(value >= NumericLimits<int8_t>::Minimum() && value <= NumericLimits<int8_t>::Maximum());
		return Value::TINYINT((int8_t)value);
	case LogicalTypeId::SMALLINT:
		D_ASSERT(value >= NumericLimits<int16_t>::Minimum() && value <= NumericLimits<int16_t>::Maximum());
		return Value::SMALLINT((int16_t)value);
	case LogicalTypeId::INTEGER:
		D_ASSERT(value >= NumericLimits<int32_t>::Minimum() && value <= NumericLimits<int32_t>::Maximum());
		return Value::INTEGER((int32_t)value);
	case LogicalTypeId::BIGINT:
		return Value::BIGINT(value);
	case LogicalTypeId::UTINYINT:
		D_ASSERT(value >= NumericLimits<uint8_t>::Minimum() && value <= NumericLimits<uint8_t>::Maximum());
		return Value::UTINYINT((uint8_t)value);
	case LogicalTypeId::USMALLINT:
		D_ASSERT(value >= NumericLimits<uint16_t>::Minimum() && value <= NumericLimits<uint16_t>::Maximum());
		return Value::USMALLINT((uint16_t)value);
	case LogicalTypeId::UINTEGER:
		D_ASSERT(value >= NumericLimits<uint32_t>::Minimum() && value <= NumericLimits<uint32_t>::Maximum());
		return Value::UINTEGER((uint32_t)value);
	case LogicalTypeId::UBIGINT:
		D_ASSERT(value >= 0);
		return Value::UBIGINT(value);
	case LogicalTypeId::HUGEINT:
		return Value::HUGEINT(value);
	case LogicalTypeId::DECIMAL:
		return Value::DECIMAL(value, DecimalType::GetWidth(type), DecimalType::GetScale(type));
	case LogicalTypeId::FLOAT:
		return Value((float)value);
	case LogicalTypeId::DOUBLE:
		return Value((double)value);
	case LogicalTypeId::POINTER:
		return Value::POINTER(value);
	case LogicalTypeId::DATE:
		D_ASSERT(value >= NumericLimits<int32_t>::Minimum() && value <= NumericLimits<int32_t>::Maximum());
		return Value::DATE(date_t(value));
	case LogicalTypeId::TIME:
		return Value::TIME(dtime_t(value));
	case LogicalTypeId::TIMESTAMP:
		return Value::TIMESTAMP(timestamp_t(value));
	case LogicalTypeId::TIMESTAMP_NS:
		return Value::TIMESTAMPNS(timestamp_t(value));
	case LogicalTypeId::TIMESTAMP_MS:
		return Value::TIMESTAMPMS(timestamp_t(value));
	case LogicalTypeId::TIMESTAMP_SEC:
		return Value::TIMESTAMPSEC(timestamp_t(value));
	case LogicalTypeId::TIME_TZ:
		return Value::TIMETZ(dtime_t(value));
	case LogicalTypeId::TIMESTAMP_TZ:
		return Value::TIMESTAMPTZ(timestamp_t(value));
	case LogicalTypeId::ENUM:
		switch (type.InternalType()) {
		case PhysicalType::UINT8:
			D_ASSERT(value >= NumericLimits<uint8_t>::Minimum() && value <= NumericLimits<uint8_t>::Maximum());
			return Value::UTINYINT((uint8_t)value);
		case PhysicalType::UINT16:
			D_ASSERT(value >= NumericLimits<uint16_t>::Minimum() && value <= NumericLimits<uint16_t>::Maximum());
			return Value::USMALLINT((uint16_t)value);
		case PhysicalType::UINT32:
			D_ASSERT(value >= NumericLimits<uint32_t>::Minimum() && value <= NumericLimits<uint32_t>::Maximum());
			return Value::UINTEGER((uint32_t)value);
		default:
			throw InternalException("Enum doesn't accept this physical type");
		}
	default:
		throw InvalidTypeException(type, "Numeric requires numeric type");
	}
}

Value Value::Numeric(const LogicalType &type, hugeint_t value) {
#ifdef DEBUG
	// perform a throwing cast to verify that the type fits
	Value::HUGEINT(value).DefaultCastAs(type);
#endif
	switch (type.id()) {
	case LogicalTypeId::HUGEINT:
		return Value::HUGEINT(value);
	case LogicalTypeId::UBIGINT:
		return Value::UBIGINT(Hugeint::Cast<uint64_t>(value));
	default:
		return Value::Numeric(type, Hugeint::Cast<int64_t>(value));
	}
}

//===--------------------------------------------------------------------===//
// GetValueUnsafe
//===--------------------------------------------------------------------===//
template <>
DUCKDB_API bool Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::BOOL);
	return value_.boolean;
}

template <>
int8_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::INT8 || type_.InternalType() == PhysicalType::BOOL);
	return value_.tinyint;
}

template <>
int16_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::INT16);
	return value_.smallint;
}

template <>
int32_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::INT32);
	return value_.integer;
}

template <>
int64_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::INT64);
	return value_.bigint;
}

template <>
hugeint_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::INT128);
	return value_.hugeint;
}

template <>
uint8_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::UINT8);
	return value_.utinyint;
}

template <>
uint16_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::UINT16);
	return value_.usmallint;
}

template <>
uint32_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::UINT32);
	return value_.uinteger;
}

template <>
uint64_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::UINT64);
	return value_.ubigint;
}

template <>
string Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::VARCHAR);
	return str_value;
}

template <>
DUCKDB_API string_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::VARCHAR);
	return string_t(str_value);
}

template <>
float Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::FLOAT);
	return value_.float_;
}

template <>
double Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::DOUBLE);
	return value_.double_;
}

template <>
date_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::INT32);
	return value_.date;
}

template <>
dtime_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::INT64);
	return value_.time;
}

template <>
timestamp_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::INT64);
	return value_.timestamp;
}

template <>
interval_t Value::GetValueUnsafe() const {
	D_ASSERT(type_.InternalType() == PhysicalType::INTERVAL);
	return value_.interval;
}

//===--------------------------------------------------------------------===//
// GetReferenceUnsafe
//===--------------------------------------------------------------------===//
template <>
int8_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::INT8 || type_.InternalType() == PhysicalType::BOOL);
	return value_.tinyint;
}

template <>
int16_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::INT16);
	return value_.smallint;
}

template <>
int32_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::INT32);
	return value_.integer;
}

template <>
int64_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::INT64);
	return value_.bigint;
}

template <>
hugeint_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::INT128);
	return value_.hugeint;
}

template <>
uint8_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::UINT8);
	return value_.utinyint;
}

template <>
uint16_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::UINT16);
	return value_.usmallint;
}

template <>
uint32_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::UINT32);
	return value_.uinteger;
}

template <>
uint64_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::UINT64);
	return value_.ubigint;
}

template <>
float &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::FLOAT);
	return value_.float_;
}

template <>
double &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::DOUBLE);
	return value_.double_;
}

template <>
date_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::INT32);
	return value_.date;
}

template <>
dtime_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::INT64);
	return value_.time;
}

template <>
timestamp_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::INT64);
	return value_.timestamp;
}

template <>
interval_t &Value::GetReferenceUnsafe() {
	D_ASSERT(type_.InternalType() == PhysicalType::INTERVAL);
	return value_.interval;
}

//===--------------------------------------------------------------------===//
// Hash
//===--------------------------------------------------------------------===//
hash_t Value::Hash() const {
	if (IsNull()) {
		return 0;
	}
	Vector input(*this);
	Vector result(LogicalType::HASH);
	VectorOperations::Hash(input, result, 1);

	auto data = FlatVector::GetData<hash_t>(result);
	return data[0];
}

string Value::ToString() const {
	if (IsNull()) {
		return "NULL";
	}
	return DefaultCastAs(LogicalType::VARCHAR).str_value;
}

string Value::ToSQLString() const {
	if (IsNull()) {
		return ToString();
	}
	switch (type_.id()) {
	case LogicalTypeId::UUID:
	case LogicalTypeId::DATE:
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIME_TZ:
	case LogicalTypeId::TIMESTAMP_TZ:
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIMESTAMP_NS:
	case LogicalTypeId::INTERVAL:
	case LogicalTypeId::BLOB:
		return "'" + ToString() + "'::" + type_.ToString();
	case LogicalTypeId::VARCHAR:
		return "'" + StringUtil::Replace(ToString(), "'", "''") + "'";
	case LogicalTypeId::STRUCT: {
		string ret = "{";
		auto &child_types = StructType::GetChildTypes(type_);
		for (size_t i = 0; i < struct_value.size(); i++) {
			auto &name = child_types[i].first;
			auto &child = struct_value[i];
			ret += "'" + name + "': " + child.ToSQLString();
			if (i < struct_value.size() - 1) {
				ret += ", ";
			}
		}
		ret += "}";
		return ret;
	}
	case LogicalTypeId::FLOAT:
		if (!FloatIsFinite(FloatValue::Get(*this))) {
			return "'" + ToString() + "'::" + type_.ToString();
		}
		return ToString();
	case LogicalTypeId::DOUBLE: {
		double val = DoubleValue::Get(*this);
		if (!DoubleIsFinite(val)) {
			if (!Value::IsNan(val)) {
				// to infinity and beyond
				return val < 0 ? "-1e1000" : "1e1000";
			}
			return "'" + ToString() + "'::" + type_.ToString();
		}
		return ToString();
	}
	case LogicalTypeId::LIST: {
		string ret = "[";
		for (size_t i = 0; i < list_value.size(); i++) {
			auto &child = list_value[i];
			ret += child.ToSQLString();
			if (i < list_value.size() - 1) {
				ret += ", ";
			}
		}
		ret += "]";
		return ret;
	}
	default:
		return ToString();
	}
}

//===--------------------------------------------------------------------===//
// Type-specific getters
//===--------------------------------------------------------------------===//
bool BooleanValue::Get(const Value &value) {
	return value.GetValueUnsafe<bool>();
}

int8_t TinyIntValue::Get(const Value &value) {
	return value.GetValueUnsafe<int8_t>();
}

int16_t SmallIntValue::Get(const Value &value) {
	return value.GetValueUnsafe<int16_t>();
}

int32_t IntegerValue::Get(const Value &value) {
	return value.GetValueUnsafe<int32_t>();
}

int64_t BigIntValue::Get(const Value &value) {
	return value.GetValueUnsafe<int64_t>();
}

hugeint_t HugeIntValue::Get(const Value &value) {
	return value.GetValueUnsafe<hugeint_t>();
}

uint8_t UTinyIntValue::Get(const Value &value) {
	return value.GetValueUnsafe<uint8_t>();
}

uint16_t USmallIntValue::Get(const Value &value) {
	return value.GetValueUnsafe<uint16_t>();
}

uint32_t UIntegerValue::Get(const Value &value) {
	return value.GetValueUnsafe<uint32_t>();
}

uint64_t UBigIntValue::Get(const Value &value) {
	return value.GetValueUnsafe<uint64_t>();
}

float FloatValue::Get(const Value &value) {
	return value.GetValueUnsafe<float>();
}

double DoubleValue::Get(const Value &value) {
	return value.GetValueUnsafe<double>();
}

const string &StringValue::Get(const Value &value) {
	D_ASSERT(value.type().InternalType() == PhysicalType::VARCHAR);
	return value.str_value;
}

date_t DateValue::Get(const Value &value) {
	return value.GetValueUnsafe<date_t>();
}

dtime_t TimeValue::Get(const Value &value) {
	return value.GetValueUnsafe<dtime_t>();
}

timestamp_t TimestampValue::Get(const Value &value) {
	return value.GetValueUnsafe<timestamp_t>();
}

interval_t IntervalValue::Get(const Value &value) {
	return value.GetValueUnsafe<interval_t>();
}

const vector<Value> &StructValue::GetChildren(const Value &value) {
	D_ASSERT(value.type().InternalType() == PhysicalType::STRUCT);
	return value.struct_value;
}

const vector<Value> &ListValue::GetChildren(const Value &value) {
	D_ASSERT(value.type().InternalType() == PhysicalType::LIST);
	return value.list_value;
}

const Value &UnionValue::GetValue(const Value &value) {
	D_ASSERT(value.type() == LogicalTypeId::UNION);
	auto &children = StructValue::GetChildren(value);
	auto tag = children[0].GetValueUnsafe<uint8_t>();
	D_ASSERT(tag < children.size() - 1);
	return children[tag + 1];
}

uint8_t UnionValue::GetTag(const Value &value) {
	D_ASSERT(value.type() == LogicalTypeId::UNION);
	auto children = StructValue::GetChildren(value);
	auto tag = children[0].GetValueUnsafe<uint8_t>();
	return tag;
}

hugeint_t IntegralValue::Get(const Value &value) {
	switch (value.type().InternalType()) {
	case PhysicalType::INT8:
		return TinyIntValue::Get(value);
	case PhysicalType::INT16:
		return SmallIntValue::Get(value);
	case PhysicalType::INT32:
		return IntegerValue::Get(value);
	case PhysicalType::INT64:
		return BigIntValue::Get(value);
	case PhysicalType::INT128:
		return HugeIntValue::Get(value);
	case PhysicalType::UINT8:
		return UTinyIntValue::Get(value);
	case PhysicalType::UINT16:
		return USmallIntValue::Get(value);
	case PhysicalType::UINT32:
		return UIntegerValue::Get(value);
	case PhysicalType::UINT64:
		return UBigIntValue::Get(value);
	default:
		throw InternalException("Invalid internal type \"%s\" for IntegralValue::Get", value.type().ToString());
	}
}

//===--------------------------------------------------------------------===//
// Comparison Operators
//===--------------------------------------------------------------------===//
bool Value::operator==(const Value &rhs) const {
	return ValueOperations::Equals(*this, rhs);
}

bool Value::operator!=(const Value &rhs) const {
	return ValueOperations::NotEquals(*this, rhs);
}

bool Value::operator<(const Value &rhs) const {
	return ValueOperations::LessThan(*this, rhs);
}

bool Value::operator>(const Value &rhs) const {
	return ValueOperations::GreaterThan(*this, rhs);
}

bool Value::operator<=(const Value &rhs) const {
	return ValueOperations::LessThanEquals(*this, rhs);
}

bool Value::operator>=(const Value &rhs) const {
	return ValueOperations::GreaterThanEquals(*this, rhs);
}

bool Value::operator==(const int64_t &rhs) const {
	return *this == Value::Numeric(type_, rhs);
}

bool Value::operator!=(const int64_t &rhs) const {
	return *this != Value::Numeric(type_, rhs);
}

bool Value::operator<(const int64_t &rhs) const {
	return *this < Value::Numeric(type_, rhs);
}

bool Value::operator>(const int64_t &rhs) const {
	return *this > Value::Numeric(type_, rhs);
}

bool Value::operator<=(const int64_t &rhs) const {
	return *this <= Value::Numeric(type_, rhs);
}

bool Value::operator>=(const int64_t &rhs) const {
	return *this >= Value::Numeric(type_, rhs);
}

bool Value::TryCastAs(CastFunctionSet &set, GetCastFunctionInput &get_input, const LogicalType &target_type,
                      Value &new_value, string *error_message, bool strict) const {
	if (type_ == target_type) {
		new_value = Copy();
		return true;
	}
	Vector input(*this);
	Vector result(target_type);
	if (!VectorOperations::TryCast(set, get_input, input, result, 1, error_message, strict)) {
		return false;
	}
	new_value = result.GetValue(0);
	return true;
}

bool Value::TryCastAs(ClientContext &context, const LogicalType &target_type, Value &new_value, string *error_message,
                      bool strict) const {
	GetCastFunctionInput get_input(context);
	return TryCastAs(CastFunctionSet::Get(context), get_input, target_type, new_value, error_message, strict);
}

bool Value::DefaultTryCastAs(const LogicalType &target_type, Value &new_value, string *error_message,
                             bool strict) const {
	CastFunctionSet set;
	GetCastFunctionInput get_input;
	return TryCastAs(set, get_input, target_type, new_value, error_message, strict);
}

Value Value::CastAs(CastFunctionSet &set, GetCastFunctionInput &get_input, const LogicalType &target_type,
                    bool strict) const {
	Value new_value;
	string error_message;
	if (!TryCastAs(set, get_input, target_type, new_value, &error_message, strict)) {
		throw InvalidInputException("Failed to cast value: %s", error_message);
	}
	return new_value;
}

Value Value::CastAs(ClientContext &context, const LogicalType &target_type, bool strict) const {
	GetCastFunctionInput get_input(context);
	return CastAs(CastFunctionSet::Get(context), get_input, target_type, strict);
}

Value Value::DefaultCastAs(const LogicalType &target_type, bool strict) const {
	CastFunctionSet set;
	GetCastFunctionInput get_input;
	return CastAs(set, get_input, target_type, strict);
}

bool Value::TryCastAs(CastFunctionSet &set, GetCastFunctionInput &get_input, const LogicalType &target_type,
                      bool strict) {
	Value new_value;
	string error_message;
	if (!TryCastAs(set, get_input, target_type, new_value, &error_message, strict)) {
		return false;
	}
	type_ = target_type;
	is_null = new_value.is_null;
	value_ = new_value.value_;
	str_value = new_value.str_value;
	struct_value = new_value.struct_value;
	list_value = new_value.list_value;
	return true;
}

bool Value::TryCastAs(ClientContext &context, const LogicalType &target_type, bool strict) {
	GetCastFunctionInput get_input(context);
	return TryCastAs(CastFunctionSet::Get(context), get_input, target_type, strict);
}

bool Value::DefaultTryCastAs(const LogicalType &target_type, bool strict) {
	CastFunctionSet set;
	GetCastFunctionInput get_input;
	return TryCastAs(set, get_input, target_type, strict);
}

void Value::Reinterpret(LogicalType new_type) {
	this->type_ = std::move(new_type);
}

void Value::Serialize(Serializer &main_serializer) const {
	FieldWriter writer(main_serializer);
	writer.WriteSerializable(type_);
	writer.WriteField<bool>(IsNull());
	if (!IsNull()) {
		auto &serializer = writer.GetSerializer();
		switch (type_.InternalType()) {
		case PhysicalType::BOOL:
			serializer.Write<int8_t>(value_.boolean);
			break;
		case PhysicalType::INT8:
			serializer.Write<int8_t>(value_.tinyint);
			break;
		case PhysicalType::INT16:
			serializer.Write<int16_t>(value_.smallint);
			break;
		case PhysicalType::INT32:
			serializer.Write<int32_t>(value_.integer);
			break;
		case PhysicalType::INT64:
			serializer.Write<int64_t>(value_.bigint);
			break;
		case PhysicalType::UINT8:
			serializer.Write<uint8_t>(value_.utinyint);
			break;
		case PhysicalType::UINT16:
			serializer.Write<uint16_t>(value_.usmallint);
			break;
		case PhysicalType::UINT32:
			serializer.Write<uint32_t>(value_.uinteger);
			break;
		case PhysicalType::UINT64:
			serializer.Write<uint64_t>(value_.ubigint);
			break;
		case PhysicalType::INT128:
			serializer.Write<hugeint_t>(value_.hugeint);
			break;
		case PhysicalType::FLOAT:
			serializer.Write<float>(value_.float_);
			break;
		case PhysicalType::DOUBLE:
			serializer.Write<double>(value_.double_);
			break;
		case PhysicalType::INTERVAL:
			serializer.Write<interval_t>(value_.interval);
			break;
		case PhysicalType::VARCHAR:
			serializer.WriteString(str_value);
			break;
		default: {
			Vector v(*this);
			v.Serialize(1, serializer);
			break;
		}
		}
	}
	writer.Finalize();
}

Value Value::Deserialize(Deserializer &main_source) {
	FieldReader reader(main_source);
	auto type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto is_null = reader.ReadRequired<bool>();
	Value new_value = Value(type);
	if (is_null) {
		reader.Finalize();
		return new_value;
	}
	new_value.is_null = false;
	auto &source = reader.GetSource();
	switch (type.InternalType()) {
	case PhysicalType::BOOL:
		new_value.value_.boolean = source.Read<int8_t>();
		break;
	case PhysicalType::INT8:
		new_value.value_.tinyint = source.Read<int8_t>();
		break;
	case PhysicalType::INT16:
		new_value.value_.smallint = source.Read<int16_t>();
		break;
	case PhysicalType::INT32:
		new_value.value_.integer = source.Read<int32_t>();
		break;
	case PhysicalType::INT64:
		new_value.value_.bigint = source.Read<int64_t>();
		break;
	case PhysicalType::UINT8:
		new_value.value_.utinyint = source.Read<uint8_t>();
		break;
	case PhysicalType::UINT16:
		new_value.value_.usmallint = source.Read<uint16_t>();
		break;
	case PhysicalType::UINT32:
		new_value.value_.uinteger = source.Read<uint32_t>();
		break;
	case PhysicalType::UINT64:
		new_value.value_.ubigint = source.Read<uint64_t>();
		break;
	case PhysicalType::INT128:
		new_value.value_.hugeint = source.Read<hugeint_t>();
		break;
	case PhysicalType::FLOAT:
		new_value.value_.float_ = source.Read<float>();
		break;
	case PhysicalType::DOUBLE:
		new_value.value_.double_ = source.Read<double>();
		break;
	case PhysicalType::INTERVAL:
		new_value.value_.interval = source.Read<interval_t>();
		break;
	case PhysicalType::VARCHAR:
		new_value.str_value = source.Read<string>();
		break;
	default: {
		Vector v(type);
		v.Deserialize(1, source);
		new_value = v.GetValue(0);
		break;
	}
	}
	reader.Finalize();
	return new_value;
}

void Value::Print() const {
	Printer::Print(ToString());
}

bool Value::NotDistinctFrom(const Value &lvalue, const Value &rvalue) {
	return ValueOperations::NotDistinctFrom(lvalue, rvalue);
}

static string SanitizeValue(string input) {
	// some results might contain padding spaces, e.g. when rendering
	// VARCHAR(10) and the string only has 6 characters, they will be padded
	// with spaces to 10 in the rendering. We don't do that here yet as we
	// are looking at internal structures. So just ignore any extra spaces
	// on the right
	StringUtil::RTrim(input);
	// for result checking code, replace null bytes with their escaped value (\0)
	return StringUtil::Replace(input, string("\0", 1), "\\0");
}

bool Value::ValuesAreEqual(CastFunctionSet &set, GetCastFunctionInput &get_input, const Value &result_value,
                           const Value &value) {
	if (result_value.IsNull() != value.IsNull()) {
		return false;
	}
	if (result_value.IsNull() && value.IsNull()) {
		// NULL = NULL in checking code
		return true;
	}
	switch (value.type_.id()) {
	case LogicalTypeId::FLOAT: {
		auto other = result_value.CastAs(set, get_input, LogicalType::FLOAT);
		float ldecimal = value.value_.float_;
		float rdecimal = other.value_.float_;
		return ApproxEqual(ldecimal, rdecimal);
	}
	case LogicalTypeId::DOUBLE: {
		auto other = result_value.CastAs(set, get_input, LogicalType::DOUBLE);
		double ldecimal = value.value_.double_;
		double rdecimal = other.value_.double_;
		return ApproxEqual(ldecimal, rdecimal);
	}
	case LogicalTypeId::VARCHAR: {
		auto other = result_value.CastAs(set, get_input, LogicalType::VARCHAR);
		string left = SanitizeValue(other.str_value);
		string right = SanitizeValue(value.str_value);
		return left == right;
	}
	default:
		if (result_value.type_.id() == LogicalTypeId::FLOAT || result_value.type_.id() == LogicalTypeId::DOUBLE) {
			return Value::ValuesAreEqual(set, get_input, value, result_value);
		}
		return value == result_value;
	}
}

bool Value::ValuesAreEqual(ClientContext &context, const Value &result_value, const Value &value) {
	GetCastFunctionInput get_input(context);
	return Value::ValuesAreEqual(CastFunctionSet::Get(context), get_input, result_value, value);
}
bool Value::DefaultValuesAreEqual(const Value &result_value, const Value &value) {
	CastFunctionSet set;
	GetCastFunctionInput get_input;
	return Value::ValuesAreEqual(set, get_input, result_value, value);
}

} // namespace duckdb




















#include <cstring> // strlen() on Solaris

namespace duckdb {

Vector::Vector(LogicalType type_p, bool create_data, bool zero_data, idx_t capacity)
    : vector_type(VectorType::FLAT_VECTOR), type(std::move(type_p)), data(nullptr) {
	if (create_data) {
		Initialize(zero_data, capacity);
	}
}

Vector::Vector(LogicalType type_p, idx_t capacity) : Vector(std::move(type_p), true, false, capacity) {
}

Vector::Vector(LogicalType type_p, data_ptr_t dataptr)
    : vector_type(VectorType::FLAT_VECTOR), type(std::move(type_p)), data(dataptr) {
	if (dataptr && !type.IsValid()) {
		throw InternalException("Cannot create a vector of type INVALID!");
	}
}

Vector::Vector(const VectorCache &cache) : type(cache.GetType()) {
	ResetFromCache(cache);
}

Vector::Vector(Vector &other) : type(other.type) {
	Reference(other);
}

Vector::Vector(Vector &other, const SelectionVector &sel, idx_t count) : type(other.type) {
	Slice(other, sel, count);
}

Vector::Vector(Vector &other, idx_t offset, idx_t end) : type(other.type) {
	Slice(other, offset, end);
}

Vector::Vector(const Value &value) : type(value.type()) {
	Reference(value);
}

Vector::Vector(Vector &&other) noexcept
    : vector_type(other.vector_type), type(std::move(other.type)), data(other.data),
      validity(std::move(other.validity)), buffer(std::move(other.buffer)), auxiliary(std::move(other.auxiliary)) {
}

void Vector::Reference(const Value &value) {
	D_ASSERT(GetType().id() == value.type().id());
	this->vector_type = VectorType::CONSTANT_VECTOR;
	buffer = VectorBuffer::CreateConstantVector(value.type());
	auto internal_type = value.type().InternalType();
	if (internal_type == PhysicalType::STRUCT) {
		auto struct_buffer = make_unique<VectorStructBuffer>();
		auto &child_types = StructType::GetChildTypes(value.type());
		auto &child_vectors = struct_buffer->GetChildren();
		auto &value_children = StructValue::GetChildren(value);
		for (idx_t i = 0; i < child_types.size(); i++) {
			auto vector = make_unique<Vector>(value.IsNull() ? Value(child_types[i].second) : value_children[i]);
			child_vectors.push_back(std::move(vector));
		}
		auxiliary = std::move(struct_buffer);
		if (value.IsNull()) {
			SetValue(0, value);
		}
	} else if (internal_type == PhysicalType::LIST) {
		auto list_buffer = make_unique<VectorListBuffer>(value.type());
		auxiliary = std::move(list_buffer);
		data = buffer->GetData();
		SetValue(0, value);
	} else {
		auxiliary.reset();
		data = buffer->GetData();
		SetValue(0, value);
	}
}

void Vector::Reference(Vector &other) {
	D_ASSERT(other.GetType() == GetType());
	Reinterpret(other);
}

void Vector::ReferenceAndSetType(Vector &other) {
	type = other.GetType();
	Reference(other);
}

void Vector::Reinterpret(Vector &other) {
	vector_type = other.vector_type;
	AssignSharedPointer(buffer, other.buffer);
	AssignSharedPointer(auxiliary, other.auxiliary);
	data = other.data;
	validity = other.validity;
}

void Vector::ResetFromCache(const VectorCache &cache) {
	cache.ResetFromCache(*this);
}

void Vector::Slice(Vector &other, idx_t offset, idx_t end) {
	if (other.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		Reference(other);
		return;
	}
	D_ASSERT(other.GetVectorType() == VectorType::FLAT_VECTOR);

	auto internal_type = GetType().InternalType();
	if (internal_type == PhysicalType::STRUCT) {
		Vector new_vector(GetType());
		auto &entries = StructVector::GetEntries(new_vector);
		auto &other_entries = StructVector::GetEntries(other);
		D_ASSERT(entries.size() == other_entries.size());
		for (idx_t i = 0; i < entries.size(); i++) {
			entries[i]->Slice(*other_entries[i], offset, end);
		}
		if (offset > 0) {
			new_vector.validity.Slice(other.validity, offset, end);
		} else {
			new_vector.validity = other.validity;
		}
		Reference(new_vector);
	} else {
		Reference(other);
		if (offset > 0) {
			data = data + GetTypeIdSize(internal_type) * offset;
			validity.Slice(other.validity, offset, end);
		}
	}
}

void Vector::Slice(Vector &other, const SelectionVector &sel, idx_t count) {
	Reference(other);
	Slice(sel, count);
}

void Vector::Slice(const SelectionVector &sel, idx_t count) {
	if (GetVectorType() == VectorType::CONSTANT_VECTOR) {
		// dictionary on a constant is just a constant
		return;
	}
	if (GetVectorType() == VectorType::DICTIONARY_VECTOR) {
		// already a dictionary, slice the current dictionary
		auto &current_sel = DictionaryVector::SelVector(*this);
		auto sliced_dictionary = current_sel.Slice(sel, count);
		buffer = make_buffer<DictionaryBuffer>(std::move(sliced_dictionary));
		if (GetType().InternalType() == PhysicalType::STRUCT) {
			auto &child_vector = DictionaryVector::Child(*this);

			Vector new_child(child_vector);
			new_child.auxiliary = make_buffer<VectorStructBuffer>(new_child, sel, count);
			auxiliary = make_buffer<VectorChildBuffer>(std::move(new_child));
		}
		return;
	}

	if (GetVectorType() == VectorType::FSST_VECTOR) {
		Flatten(sel, count);
		return;
	}

	Vector child_vector(*this);
	auto internal_type = GetType().InternalType();
	if (internal_type == PhysicalType::STRUCT) {
		child_vector.auxiliary = make_buffer<VectorStructBuffer>(*this, sel, count);
	}
	auto child_ref = make_buffer<VectorChildBuffer>(std::move(child_vector));
	auto dict_buffer = make_buffer<DictionaryBuffer>(sel);
	vector_type = VectorType::DICTIONARY_VECTOR;
	buffer = std::move(dict_buffer);
	auxiliary = std::move(child_ref);
}

void Vector::Slice(const SelectionVector &sel, idx_t count, SelCache &cache) {
	if (GetVectorType() == VectorType::DICTIONARY_VECTOR && GetType().InternalType() != PhysicalType::STRUCT) {
		// dictionary vector: need to merge dictionaries
		// check if we have a cached entry
		auto &current_sel = DictionaryVector::SelVector(*this);
		auto target_data = current_sel.data();
		auto entry = cache.cache.find(target_data);
		if (entry != cache.cache.end()) {
			// cached entry exists: use that
			this->buffer = make_buffer<DictionaryBuffer>(((DictionaryBuffer &)*entry->second).GetSelVector());
			vector_type = VectorType::DICTIONARY_VECTOR;
		} else {
			Slice(sel, count);
			cache.cache[target_data] = this->buffer;
		}
	} else {
		Slice(sel, count);
	}
}

void Vector::Initialize(bool zero_data, idx_t capacity) {
	auxiliary.reset();
	validity.Reset();
	auto &type = GetType();
	auto internal_type = type.InternalType();
	if (internal_type == PhysicalType::STRUCT) {
		auto struct_buffer = make_unique<VectorStructBuffer>(type, capacity);
		auxiliary = std::move(struct_buffer);
	} else if (internal_type == PhysicalType::LIST) {
		auto list_buffer = make_unique<VectorListBuffer>(type, capacity);
		auxiliary = std::move(list_buffer);
	}
	auto type_size = GetTypeIdSize(internal_type);
	if (type_size > 0) {
		buffer = VectorBuffer::CreateStandardVector(type, capacity);
		data = buffer->GetData();
		if (zero_data) {
			memset(data, 0, capacity * type_size);
		}
	}
	if (capacity > STANDARD_VECTOR_SIZE) {
		validity.Resize(STANDARD_VECTOR_SIZE, capacity);
	}
}

struct DataArrays {
	Vector &vec;
	data_ptr_t data;
	VectorBuffer *buffer;
	idx_t type_size;
	bool is_nested;
	DataArrays(Vector &vec, data_ptr_t data, VectorBuffer *buffer, idx_t type_size, bool is_nested)
	    : vec(vec), data(data), buffer(buffer), type_size(type_size), is_nested(is_nested) {};
};

void FindChildren(std::vector<DataArrays> &to_resize, VectorBuffer &auxiliary) {
	if (auxiliary.GetBufferType() == VectorBufferType::LIST_BUFFER) {
		auto &buffer = (VectorListBuffer &)auxiliary;
		auto &child = buffer.GetChild();
		auto data = child.GetData();
		if (!data) {
			//! Nested type
			DataArrays arrays(child, data, child.GetBuffer().get(), GetTypeIdSize(child.GetType().InternalType()),
			                  true);
			to_resize.emplace_back(arrays);
			FindChildren(to_resize, *child.GetAuxiliary());
		} else {
			DataArrays arrays(child, data, child.GetBuffer().get(), GetTypeIdSize(child.GetType().InternalType()),
			                  false);
			to_resize.emplace_back(arrays);
		}
	} else if (auxiliary.GetBufferType() == VectorBufferType::STRUCT_BUFFER) {
		auto &buffer = (VectorStructBuffer &)auxiliary;
		auto &children = buffer.GetChildren();
		for (auto &child : children) {
			auto data = child->GetData();
			if (!data) {
				//! Nested type
				DataArrays arrays(*child, data, child->GetBuffer().get(),
				                  GetTypeIdSize(child->GetType().InternalType()), true);
				to_resize.emplace_back(arrays);
				FindChildren(to_resize, *child->GetAuxiliary());
			} else {
				DataArrays arrays(*child, data, child->GetBuffer().get(),
				                  GetTypeIdSize(child->GetType().InternalType()), false);
				to_resize.emplace_back(arrays);
			}
		}
	}
}
void Vector::Resize(idx_t cur_size, idx_t new_size) {
	std::vector<DataArrays> to_resize;
	if (!buffer) {
		buffer = make_unique<VectorBuffer>(0);
	}
	if (!data) {
		//! this is a nested structure
		DataArrays arrays(*this, data, buffer.get(), GetTypeIdSize(GetType().InternalType()), true);
		to_resize.emplace_back(arrays);
		FindChildren(to_resize, *auxiliary);
	} else {
		DataArrays arrays(*this, data, buffer.get(), GetTypeIdSize(GetType().InternalType()), false);
		to_resize.emplace_back(arrays);
	}
	for (auto &data_to_resize : to_resize) {
		if (!data_to_resize.is_nested) {
			auto new_data = unique_ptr<data_t[]>(new data_t[new_size * data_to_resize.type_size]);
			memcpy(new_data.get(), data_to_resize.data, cur_size * data_to_resize.type_size * sizeof(data_t));
			data_to_resize.buffer->SetData(std::move(new_data));
			data_to_resize.vec.data = data_to_resize.buffer->GetData();
		}
		data_to_resize.vec.validity.Resize(cur_size, new_size);
	}
}

void Vector::SetValue(idx_t index, const Value &val) {
	if (GetVectorType() == VectorType::DICTIONARY_VECTOR) {
		// dictionary: apply dictionary and forward to child
		auto &sel_vector = DictionaryVector::SelVector(*this);
		auto &child = DictionaryVector::Child(*this);
		return child.SetValue(sel_vector.get_index(index), val);
	}
	if (val.type() != GetType()) {
		SetValue(index, val.DefaultCastAs(GetType()));
		return;
	}
	D_ASSERT(val.type().InternalType() == GetType().InternalType());

	validity.EnsureWritable();
	validity.Set(index, !val.IsNull());
	if (val.IsNull() && GetType().InternalType() != PhysicalType::STRUCT) {
		// for structs we still need to set the child-entries to NULL
		// so we do not bail out yet
		return;
	}

	switch (GetType().InternalType()) {
	case PhysicalType::BOOL:
		((bool *)data)[index] = val.GetValueUnsafe<bool>();
		break;
	case PhysicalType::INT8:
		((int8_t *)data)[index] = val.GetValueUnsafe<int8_t>();
		break;
	case PhysicalType::INT16:
		((int16_t *)data)[index] = val.GetValueUnsafe<int16_t>();
		break;
	case PhysicalType::INT32:
		((int32_t *)data)[index] = val.GetValueUnsafe<int32_t>();
		break;
	case PhysicalType::INT64:
		((int64_t *)data)[index] = val.GetValueUnsafe<int64_t>();
		break;
	case PhysicalType::INT128:
		((hugeint_t *)data)[index] = val.GetValueUnsafe<hugeint_t>();
		break;
	case PhysicalType::UINT8:
		((uint8_t *)data)[index] = val.GetValueUnsafe<uint8_t>();
		break;
	case PhysicalType::UINT16:
		((uint16_t *)data)[index] = val.GetValueUnsafe<uint16_t>();
		break;
	case PhysicalType::UINT32:
		((uint32_t *)data)[index] = val.GetValueUnsafe<uint32_t>();
		break;
	case PhysicalType::UINT64:
		((uint64_t *)data)[index] = val.GetValueUnsafe<uint64_t>();
		break;
	case PhysicalType::FLOAT:
		((float *)data)[index] = val.GetValueUnsafe<float>();
		break;
	case PhysicalType::DOUBLE:
		((double *)data)[index] = val.GetValueUnsafe<double>();
		break;
	case PhysicalType::INTERVAL:
		((interval_t *)data)[index] = val.GetValueUnsafe<interval_t>();
		break;
	case PhysicalType::VARCHAR:
		((string_t *)data)[index] = StringVector::AddStringOrBlob(*this, StringValue::Get(val));
		break;
	case PhysicalType::STRUCT: {
		D_ASSERT(GetVectorType() == VectorType::CONSTANT_VECTOR || GetVectorType() == VectorType::FLAT_VECTOR);

		auto &children = StructVector::GetEntries(*this);
		auto &val_children = StructValue::GetChildren(val);
		D_ASSERT(val.IsNull() || children.size() == val_children.size());
		for (size_t i = 0; i < children.size(); i++) {
			auto &vec_child = children[i];
			if (!val.IsNull()) {
				auto &struct_child = val_children[i];
				vec_child->SetValue(index, struct_child);
			} else {
				vec_child->SetValue(index, Value());
			}
		}
		break;
	}
	case PhysicalType::LIST: {
		auto offset = ListVector::GetListSize(*this);
		auto &val_children = ListValue::GetChildren(val);
		if (!val_children.empty()) {
			for (idx_t i = 0; i < val_children.size(); i++) {
				ListVector::PushBack(*this, val_children[i]);
			}
		}
		//! now set the pointer
		auto &entry = ((list_entry_t *)data)[index];
		entry.length = val_children.size();
		entry.offset = offset;
		break;
	}
	default:
		throw InternalException("Unimplemented type for Vector::SetValue");
	}
}

Value Vector::GetValueInternal(const Vector &v_p, idx_t index_p) {
	const Vector *vector = &v_p;
	idx_t index = index_p;
	bool finished = false;
	while (!finished) {
		switch (vector->GetVectorType()) {
		case VectorType::CONSTANT_VECTOR:
			index = 0;
			finished = true;
			break;
		case VectorType::FLAT_VECTOR:
			finished = true;
			break;
		case VectorType::FSST_VECTOR:
			finished = true;
			break;
		// dictionary: apply dictionary and forward to child
		case VectorType::DICTIONARY_VECTOR: {
			auto &sel_vector = DictionaryVector::SelVector(*vector);
			auto &child = DictionaryVector::Child(*vector);
			vector = &child;
			index = sel_vector.get_index(index);
			break;
		}
		case VectorType::SEQUENCE_VECTOR: {
			int64_t start, increment;
			SequenceVector::GetSequence(*vector, start, increment);
			return Value::Numeric(vector->GetType(), start + increment * index);
		}
		default:
			throw InternalException("Unimplemented vector type for Vector::GetValue");
		}
	}
	auto data = vector->data;
	auto &validity = vector->validity;
	auto &type = vector->GetType();

	if (!validity.RowIsValid(index)) {
		return Value(vector->GetType());
	}

	if (vector->GetVectorType() == VectorType::FSST_VECTOR) {
		if (vector->GetType().InternalType() != PhysicalType::VARCHAR) {
			throw InternalException("FSST Vector with non-string datatype found!");
		}
		auto str_compressed = ((string_t *)data)[index];
		Value result =
		    FSSTPrimitives::DecompressValue(FSSTVector::GetDecoder(const_cast<Vector &>(*vector)),
		                                    (unsigned char *)str_compressed.GetDataUnsafe(), str_compressed.GetSize());
		return result;
	}

	switch (vector->GetType().id()) {
	case LogicalTypeId::BOOLEAN:
		return Value::BOOLEAN(((bool *)data)[index]);
	case LogicalTypeId::TINYINT:
		return Value::TINYINT(((int8_t *)data)[index]);
	case LogicalTypeId::SMALLINT:
		return Value::SMALLINT(((int16_t *)data)[index]);
	case LogicalTypeId::INTEGER:
		return Value::INTEGER(((int32_t *)data)[index]);
	case LogicalTypeId::DATE:
		return Value::DATE(((date_t *)data)[index]);
	case LogicalTypeId::TIME:
		return Value::TIME(((dtime_t *)data)[index]);
	case LogicalTypeId::TIME_TZ:
		return Value::TIMETZ(((dtime_t *)data)[index]);
	case LogicalTypeId::BIGINT:
		return Value::BIGINT(((int64_t *)data)[index]);
	case LogicalTypeId::UTINYINT:
		return Value::UTINYINT(((uint8_t *)data)[index]);
	case LogicalTypeId::USMALLINT:
		return Value::USMALLINT(((uint16_t *)data)[index]);
	case LogicalTypeId::UINTEGER:
		return Value::UINTEGER(((uint32_t *)data)[index]);
	case LogicalTypeId::UBIGINT:
		return Value::UBIGINT(((uint64_t *)data)[index]);
	case LogicalTypeId::TIMESTAMP:
		return Value::TIMESTAMP(((timestamp_t *)data)[index]);
	case LogicalTypeId::TIMESTAMP_NS:
		return Value::TIMESTAMPNS(((timestamp_t *)data)[index]);
	case LogicalTypeId::TIMESTAMP_MS:
		return Value::TIMESTAMPMS(((timestamp_t *)data)[index]);
	case LogicalTypeId::TIMESTAMP_SEC:
		return Value::TIMESTAMPSEC(((timestamp_t *)data)[index]);
	case LogicalTypeId::TIMESTAMP_TZ:
		return Value::TIMESTAMPTZ(((timestamp_t *)data)[index]);
	case LogicalTypeId::HUGEINT:
		return Value::HUGEINT(((hugeint_t *)data)[index]);
	case LogicalTypeId::UUID:
		return Value::UUID(((hugeint_t *)data)[index]);
	case LogicalTypeId::DECIMAL: {
		auto width = DecimalType::GetWidth(type);
		auto scale = DecimalType::GetScale(type);
		switch (type.InternalType()) {
		case PhysicalType::INT16:
			return Value::DECIMAL(((int16_t *)data)[index], width, scale);
		case PhysicalType::INT32:
			return Value::DECIMAL(((int32_t *)data)[index], width, scale);
		case PhysicalType::INT64:
			return Value::DECIMAL(((int64_t *)data)[index], width, scale);
		case PhysicalType::INT128:
			return Value::DECIMAL(((hugeint_t *)data)[index], width, scale);
		default:
			throw InternalException("Physical type '%s' has a width bigger than 38, which is not supported",
			                        TypeIdToString(type.InternalType()));
		}
	}
	case LogicalTypeId::ENUM: {
		switch (type.InternalType()) {
		case PhysicalType::UINT8:
			return Value::ENUM(((uint8_t *)data)[index], type);
		case PhysicalType::UINT16:
			return Value::ENUM(((uint16_t *)data)[index], type);
		case PhysicalType::UINT32:
			return Value::ENUM(((uint32_t *)data)[index], type);
		default:
			throw InternalException("ENUM can only have unsigned integers as physical types");
		}
	}
	case LogicalTypeId::POINTER:
		return Value::POINTER(((uintptr_t *)data)[index]);
	case LogicalTypeId::FLOAT:
		return Value::FLOAT(((float *)data)[index]);
	case LogicalTypeId::DOUBLE:
		return Value::DOUBLE(((double *)data)[index]);
	case LogicalTypeId::INTERVAL:
		return Value::INTERVAL(((interval_t *)data)[index]);
	case LogicalTypeId::VARCHAR: {
		auto str = ((string_t *)data)[index];
		return Value(str.GetString());
	}
	case LogicalTypeId::AGGREGATE_STATE:
	case LogicalTypeId::BLOB: {
		auto str = ((string_t *)data)[index];
		return Value::BLOB((const_data_ptr_t)str.GetDataUnsafe(), str.GetSize());
	}
	case LogicalTypeId::BIT: {
		auto str = ((string_t *)data)[index];
		return Value::BIT((const_data_ptr_t)str.GetDataUnsafe(), str.GetSize());
	}
	case LogicalTypeId::MAP: {
		auto offlen = ((list_entry_t *)data)[index];
		auto &child_vec = ListVector::GetEntry(*vector);
		std::vector<Value> children;
		for (idx_t i = offlen.offset; i < offlen.offset + offlen.length; i++) {
			children.push_back(child_vec.GetValue(i));
		}
		return Value::MAP(ListType::GetChildType(type), std::move(children));
	}
	case LogicalTypeId::UNION: {
		auto tag = UnionVector::GetTag(*vector, index);
		auto value = UnionVector::GetMember(*vector, tag).GetValue(index);
		auto members = UnionType::CopyMemberTypes(type);
		return Value::UNION(members, tag, std::move(value));
	}
	case LogicalTypeId::STRUCT: {
		// we can derive the value schema from the vector schema
		auto &child_entries = StructVector::GetEntries(*vector);
		child_list_t<Value> children;
		for (idx_t child_idx = 0; child_idx < child_entries.size(); child_idx++) {
			auto &struct_child = child_entries[child_idx];
			children.push_back(make_pair(StructType::GetChildName(type, child_idx), struct_child->GetValue(index_p)));
		}
		return Value::STRUCT(std::move(children));
	}
	case LogicalTypeId::LIST: {
		auto offlen = ((list_entry_t *)data)[index];
		auto &child_vec = ListVector::GetEntry(*vector);
		std::vector<Value> children;
		for (idx_t i = offlen.offset; i < offlen.offset + offlen.length; i++) {
			children.push_back(child_vec.GetValue(i));
		}
		return Value::LIST(ListType::GetChildType(type), std::move(children));
	}
	default:
		throw InternalException("Unimplemented type for value access");
	}
}

Value Vector::GetValue(const Vector &v_p, idx_t index_p) {
	auto value = GetValueInternal(v_p, index_p);
	// set the alias of the type to the correct value, if there is a type alias
	if (v_p.GetType().HasAlias()) {
		value.GetTypeMutable().CopyAuxInfo(v_p.GetType());
	}
	if (v_p.GetType().id() != LogicalTypeId::AGGREGATE_STATE && value.type().id() != LogicalTypeId::AGGREGATE_STATE) {

		D_ASSERT(v_p.GetType() == value.type());
	}
	return value;
}

Value Vector::GetValue(idx_t index) const {
	return GetValue(*this, index);
}

// LCOV_EXCL_START
string VectorTypeToString(VectorType type) {
	switch (type) {
	case VectorType::FLAT_VECTOR:
		return "FLAT";
	case VectorType::FSST_VECTOR:
		return "FSST";
	case VectorType::SEQUENCE_VECTOR:
		return "SEQUENCE";
	case VectorType::DICTIONARY_VECTOR:
		return "DICTIONARY";
	case VectorType::CONSTANT_VECTOR:
		return "CONSTANT";
	default:
		return "UNKNOWN";
	}
}

string Vector::ToString(idx_t count) const {
	string retval =
	    VectorTypeToString(GetVectorType()) + " " + GetType().ToString() + ": " + to_string(count) + " = [ ";
	switch (GetVectorType()) {
	case VectorType::FLAT_VECTOR:
	case VectorType::DICTIONARY_VECTOR:
		for (idx_t i = 0; i < count; i++) {
			retval += GetValue(i).ToString() + (i == count - 1 ? "" : ", ");
		}
		break;
	case VectorType::FSST_VECTOR: {
		for (idx_t i = 0; i < count; i++) {
			string_t compressed_string = ((string_t *)data)[i];
			Value val = FSSTPrimitives::DecompressValue(FSSTVector::GetDecoder(const_cast<Vector &>(*this)),
			                                            (unsigned char *)compressed_string.GetDataUnsafe(),
			                                            compressed_string.GetSize());
			retval += GetValue(i).ToString() + (i == count - 1 ? "" : ", ");
		}
	} break;
	case VectorType::CONSTANT_VECTOR:
		retval += GetValue(0).ToString();
		break;
	case VectorType::SEQUENCE_VECTOR: {
		int64_t start, increment;
		SequenceVector::GetSequence(*this, start, increment);
		for (idx_t i = 0; i < count; i++) {
			retval += to_string(start + increment * i) + (i == count - 1 ? "" : ", ");
		}
		break;
	}
	default:
		retval += "UNKNOWN VECTOR TYPE";
		break;
	}
	retval += "]";
	return retval;
}

void Vector::Print(idx_t count) const {
	Printer::Print(ToString(count));
}

string Vector::ToString() const {
	string retval = VectorTypeToString(GetVectorType()) + " " + GetType().ToString() + ": (UNKNOWN COUNT) [ ";
	switch (GetVectorType()) {
	case VectorType::FLAT_VECTOR:
	case VectorType::DICTIONARY_VECTOR:
		break;
	case VectorType::CONSTANT_VECTOR:
		retval += GetValue(0).ToString();
		break;
	case VectorType::SEQUENCE_VECTOR: {
		break;
	}
	default:
		retval += "UNKNOWN VECTOR TYPE";
		break;
	}
	retval += "]";
	return retval;
}

void Vector::Print() const {
	Printer::Print(ToString());
}
// LCOV_EXCL_STOP

template <class T>
static void TemplatedFlattenConstantVector(data_ptr_t data, data_ptr_t old_data, idx_t count) {
	auto constant = Load<T>(old_data);
	auto output = (T *)data;
	for (idx_t i = 0; i < count; i++) {
		output[i] = constant;
	}
}

void Vector::Flatten(idx_t count) {
	switch (GetVectorType()) {
	case VectorType::FLAT_VECTOR:
		// already a flat vector
		break;
	case VectorType::FSST_VECTOR: {
		// Even though count may only be a part of the vector, we need to flatten the whole thing due to the way
		// ToUnifiedFormat uses flatten
		idx_t total_count = FSSTVector::GetCount(*this);
		// create vector to decompress into
		Vector other(GetType(), total_count);
		// now copy the data of this vector to the other vector, decompressing the strings in the process
		VectorOperations::Copy(*this, other, total_count, 0, 0);
		// create a reference to the data in the other vector
		this->Reference(other);
		break;
	}
	case VectorType::DICTIONARY_VECTOR: {
		// create a new flat vector of this type
		Vector other(GetType(), count);
		// now copy the data of this vector to the other vector, removing the selection vector in the process
		VectorOperations::Copy(*this, other, count, 0, 0);
		// create a reference to the data in the other vector
		this->Reference(other);
		break;
	}
	case VectorType::CONSTANT_VECTOR: {
		bool is_null = ConstantVector::IsNull(*this);
		// allocate a new buffer for the vector
		auto old_buffer = std::move(buffer);
		auto old_data = data;
		buffer = VectorBuffer::CreateStandardVector(type, MaxValue<idx_t>(STANDARD_VECTOR_SIZE, count));
		data = buffer->GetData();
		vector_type = VectorType::FLAT_VECTOR;
		if (is_null) {
			// constant NULL, set nullmask
			validity.EnsureWritable();
			validity.SetAllInvalid(count);
			return;
		}
		// non-null constant: have to repeat the constant
		switch (GetType().InternalType()) {
		case PhysicalType::BOOL:
			TemplatedFlattenConstantVector<bool>(data, old_data, count);
			break;
		case PhysicalType::INT8:
			TemplatedFlattenConstantVector<int8_t>(data, old_data, count);
			break;
		case PhysicalType::INT16:
			TemplatedFlattenConstantVector<int16_t>(data, old_data, count);
			break;
		case PhysicalType::INT32:
			TemplatedFlattenConstantVector<int32_t>(data, old_data, count);
			break;
		case PhysicalType::INT64:
			TemplatedFlattenConstantVector<int64_t>(data, old_data, count);
			break;
		case PhysicalType::UINT8:
			TemplatedFlattenConstantVector<uint8_t>(data, old_data, count);
			break;
		case PhysicalType::UINT16:
			TemplatedFlattenConstantVector<uint16_t>(data, old_data, count);
			break;
		case PhysicalType::UINT32:
			TemplatedFlattenConstantVector<uint32_t>(data, old_data, count);
			break;
		case PhysicalType::UINT64:
			TemplatedFlattenConstantVector<uint64_t>(data, old_data, count);
			break;
		case PhysicalType::INT128:
			TemplatedFlattenConstantVector<hugeint_t>(data, old_data, count);
			break;
		case PhysicalType::FLOAT:
			TemplatedFlattenConstantVector<float>(data, old_data, count);
			break;
		case PhysicalType::DOUBLE:
			TemplatedFlattenConstantVector<double>(data, old_data, count);
			break;
		case PhysicalType::INTERVAL:
			TemplatedFlattenConstantVector<interval_t>(data, old_data, count);
			break;
		case PhysicalType::VARCHAR:
			TemplatedFlattenConstantVector<string_t>(data, old_data, count);
			break;
		case PhysicalType::LIST: {
			TemplatedFlattenConstantVector<list_entry_t>(data, old_data, count);
			break;
		}
		case PhysicalType::STRUCT: {
			auto normalified_buffer = make_unique<VectorStructBuffer>();

			auto &new_children = normalified_buffer->GetChildren();

			auto &child_entries = StructVector::GetEntries(*this);
			for (auto &child : child_entries) {
				D_ASSERT(child->GetVectorType() == VectorType::CONSTANT_VECTOR);
				auto vector = make_unique<Vector>(*child);
				vector->Flatten(count);
				new_children.push_back(std::move(vector));
			}
			auxiliary = std::move(normalified_buffer);
		} break;
		default:
			throw InternalException("Unimplemented type for VectorOperations::Flatten");
		}
		break;
	}
	case VectorType::SEQUENCE_VECTOR: {
		int64_t start, increment, sequence_count;
		SequenceVector::GetSequence(*this, start, increment, sequence_count);

		buffer = VectorBuffer::CreateStandardVector(GetType());
		data = buffer->GetData();
		VectorOperations::GenerateSequence(*this, sequence_count, start, increment);
		break;
	}
	default:
		throw InternalException("Unimplemented type for normalify");
	}
}

void Vector::Flatten(const SelectionVector &sel, idx_t count) {
	switch (GetVectorType()) {
	case VectorType::FLAT_VECTOR:
		// already a flat vector
		break;
	case VectorType::FSST_VECTOR: {
		// create a new flat vector of this type
		Vector other(GetType());
		// copy the data of this vector to the other vector, removing compression and selection vector in the process
		VectorOperations::Copy(*this, other, sel, count, 0, 0);
		// create a reference to the data in the other vector
		this->Reference(other);
		break;
	}
	case VectorType::SEQUENCE_VECTOR: {
		int64_t start, increment;
		SequenceVector::GetSequence(*this, start, increment);

		buffer = VectorBuffer::CreateStandardVector(GetType());
		data = buffer->GetData();
		VectorOperations::GenerateSequence(*this, count, sel, start, increment);
		break;
	}
	default:
		throw InternalException("Unimplemented type for normalify with selection vector");
	}
}

void Vector::ToUnifiedFormat(idx_t count, UnifiedVectorFormat &data) {
	switch (GetVectorType()) {
	case VectorType::DICTIONARY_VECTOR: {
		auto &sel = DictionaryVector::SelVector(*this);
		auto &child = DictionaryVector::Child(*this);
		if (child.GetVectorType() == VectorType::FLAT_VECTOR) {
			data.sel = &sel;
			data.data = FlatVector::GetData(child);
			data.validity = FlatVector::Validity(child);
		} else {
			// dictionary with non-flat child: create a new reference to the child and normalify it
			Vector child_vector(child);
			child_vector.Flatten(sel, count);
			auto new_aux = make_buffer<VectorChildBuffer>(std::move(child_vector));

			data.sel = &sel;
			data.data = FlatVector::GetData(new_aux->data);
			data.validity = FlatVector::Validity(new_aux->data);
			this->auxiliary = std::move(new_aux);
		}
		break;
	}
	case VectorType::CONSTANT_VECTOR:
		data.sel = ConstantVector::ZeroSelectionVector(count, data.owned_sel);
		data.data = ConstantVector::GetData(*this);
		data.validity = ConstantVector::Validity(*this);
		break;
	default:
		Flatten(count);
		data.sel = FlatVector::IncrementalSelectionVector();
		data.data = FlatVector::GetData(*this);
		data.validity = FlatVector::Validity(*this);
		break;
	}
}

void Vector::Sequence(int64_t start, int64_t increment, idx_t count) {
	this->vector_type = VectorType::SEQUENCE_VECTOR;
	this->buffer = make_buffer<VectorBuffer>(sizeof(int64_t) * 3);
	auto data = (int64_t *)buffer->GetData();
	data[0] = start;
	data[1] = increment;
	data[2] = int64_t(count);
	validity.Reset();
	auxiliary.reset();
}

void Vector::Serialize(idx_t count, Serializer &serializer) {
	auto &type = GetType();

	UnifiedVectorFormat vdata;
	ToUnifiedFormat(count, vdata);

	const auto write_validity = (count > 0) && !vdata.validity.AllValid();
	serializer.Write<bool>(write_validity);
	if (write_validity) {
		ValidityMask flat_mask(count);
		for (idx_t i = 0; i < count; ++i) {
			auto row_idx = vdata.sel->get_index(i);
			flat_mask.Set(i, vdata.validity.RowIsValid(row_idx));
		}
		serializer.WriteData((const_data_ptr_t)flat_mask.GetData(), flat_mask.ValidityMaskSize(count));
	}
	if (TypeIsConstantSize(type.InternalType())) {
		// constant size type: simple copy
		idx_t write_size = GetTypeIdSize(type.InternalType()) * count;
		auto ptr = unique_ptr<data_t[]>(new data_t[write_size]);
		VectorOperations::WriteToStorage(*this, count, ptr.get());
		serializer.WriteData(ptr.get(), write_size);
	} else {
		switch (type.InternalType()) {
		case PhysicalType::VARCHAR: {
			auto strings = (string_t *)vdata.data;
			for (idx_t i = 0; i < count; i++) {
				auto idx = vdata.sel->get_index(i);
				auto source = !vdata.validity.RowIsValid(idx) ? NullValue<string_t>() : strings[idx];
				serializer.WriteStringLen((const_data_ptr_t)source.GetDataUnsafe(), source.GetSize());
			}
			break;
		}
		case PhysicalType::STRUCT: {
			Flatten(count);
			auto &entries = StructVector::GetEntries(*this);
			for (auto &entry : entries) {
				entry->Serialize(count, serializer);
			}
			break;
		}
		case PhysicalType::LIST: {
			auto &child = ListVector::GetEntry(*this);
			auto list_size = ListVector::GetListSize(*this);

			// serialize the list entries in a flat array
			auto data = unique_ptr<list_entry_t[]>(new list_entry_t[count]);
			auto source_array = (list_entry_t *)vdata.data;
			for (idx_t i = 0; i < count; i++) {
				auto idx = vdata.sel->get_index(i);
				auto source = source_array[idx];
				data[i].offset = source.offset;
				data[i].length = source.length;
			}

			// write the list size
			serializer.Write<idx_t>(list_size);
			serializer.WriteData((data_ptr_t)data.get(), count * sizeof(list_entry_t));

			child.Serialize(list_size, serializer);
			break;
		}
		default:
			throw InternalException("Unimplemented variable width type for Vector::Serialize!");
		}
	}
}

void Vector::Deserialize(idx_t count, Deserializer &source) {
	auto &type = GetType();

	auto &validity = FlatVector::Validity(*this);
	validity.Reset();
	const auto has_validity = source.Read<bool>();
	if (has_validity) {
		validity.Initialize(count);
		source.ReadData((data_ptr_t)validity.GetData(), validity.ValidityMaskSize(count));
	}

	if (TypeIsConstantSize(type.InternalType())) {
		// constant size type: read fixed amount of data from
		auto column_size = GetTypeIdSize(type.InternalType()) * count;
		auto ptr = unique_ptr<data_t[]>(new data_t[column_size]);
		source.ReadData(ptr.get(), column_size);

		VectorOperations::ReadFromStorage(ptr.get(), count, *this);
	} else {
		switch (type.InternalType()) {
		case PhysicalType::VARCHAR: {
			auto strings = FlatVector::GetData<string_t>(*this);
			for (idx_t i = 0; i < count; i++) {
				// read the strings
				auto str = source.Read<string>();
				// now add the string to the StringHeap of the vector
				// and write the pointer into the vector
				if (validity.RowIsValid(i)) {
					strings[i] = StringVector::AddStringOrBlob(*this, str);
				}
			}
			break;
		}
		case PhysicalType::STRUCT: {
			auto &entries = StructVector::GetEntries(*this);
			for (auto &entry : entries) {
				entry->Deserialize(count, source);
			}
			break;
		}
		case PhysicalType::LIST: {
			// read the list size
			auto list_size = source.Read<idx_t>();
			ListVector::Reserve(*this, list_size);
			ListVector::SetListSize(*this, list_size);

			// read the list entry
			auto list_entries = FlatVector::GetData(*this);
			source.ReadData(list_entries, count * sizeof(list_entry_t));

			// deserialize the child vector
			auto &child = ListVector::GetEntry(*this);
			child.Deserialize(list_size, source);

			break;
		}
		default:
			throw InternalException("Unimplemented variable width type for Vector::Deserialize!");
		}
	}
}

void Vector::SetVectorType(VectorType vector_type_p) {
	this->vector_type = vector_type_p;
	if (TypeIsConstantSize(GetType().InternalType()) &&
	    (GetVectorType() == VectorType::CONSTANT_VECTOR || GetVectorType() == VectorType::FLAT_VECTOR)) {
		auxiliary.reset();
	}
	if (vector_type == VectorType::CONSTANT_VECTOR && GetType().InternalType() == PhysicalType::STRUCT) {
		auto &entries = StructVector::GetEntries(*this);
		for (auto &entry : entries) {
			entry->SetVectorType(vector_type);
		}
	}
}

void Vector::UTFVerify(const SelectionVector &sel, idx_t count) {
#ifdef DEBUG
	if (count == 0) {
		return;
	}
	if (GetType().InternalType() == PhysicalType::VARCHAR) {
		// we just touch all the strings and let the sanitizer figure out if any
		// of them are deallocated/corrupt
		switch (GetVectorType()) {
		case VectorType::CONSTANT_VECTOR: {
			auto string = ConstantVector::GetData<string_t>(*this);
			if (!ConstantVector::IsNull(*this)) {
				string->Verify();
			}
			break;
		}
		case VectorType::FLAT_VECTOR: {
			auto strings = FlatVector::GetData<string_t>(*this);
			for (idx_t i = 0; i < count; i++) {
				auto oidx = sel.get_index(i);
				if (validity.RowIsValid(oidx)) {
					strings[oidx].Verify();
				}
			}
			break;
		}
		default:
			break;
		}
	}
#endif
}

void Vector::UTFVerify(idx_t count) {
	auto flat_sel = FlatVector::IncrementalSelectionVector();

	UTFVerify(*flat_sel, count);
}

void Vector::VerifyMap(Vector &vector_p, const SelectionVector &sel_p, idx_t count) {
#ifdef DEBUG
	D_ASSERT(vector_p.GetType().id() == LogicalTypeId::MAP);
	auto valid_check = CheckMapValidity(vector_p, count, sel_p);
	D_ASSERT(valid_check == MapInvalidReason::VALID);
#endif // DEBUG
}

void Vector::VerifyUnion(Vector &vector_p, const SelectionVector &sel_p, idx_t count) {
#ifdef DEBUG
	D_ASSERT(vector_p.GetType().id() == LogicalTypeId::UNION);
	auto valid_check = CheckUnionValidity(vector_p, count, sel_p);
	D_ASSERT(valid_check == UnionInvalidReason::VALID);
#endif // DEBUG
}

void Vector::Verify(Vector &vector_p, const SelectionVector &sel_p, idx_t count) {
#ifdef DEBUG
	if (count == 0) {
		return;
	}
	Vector *vector = &vector_p;
	const SelectionVector *sel = &sel_p;
	SelectionVector owned_sel;
	auto &type = vector->GetType();
	auto vtype = vector->GetVectorType();
	if (vector->GetVectorType() == VectorType::DICTIONARY_VECTOR) {
		auto &child = DictionaryVector::Child(*vector);
		D_ASSERT(child.GetVectorType() != VectorType::DICTIONARY_VECTOR);
		auto &dict_sel = DictionaryVector::SelVector(*vector);
		// merge the selection vectors and verify the child
		auto new_buffer = dict_sel.Slice(*sel, count);
		owned_sel.Initialize(new_buffer);
		sel = &owned_sel;
		vector = &child;
		vtype = vector->GetVectorType();
	}
	if (TypeIsConstantSize(type.InternalType()) &&
	    (vtype == VectorType::CONSTANT_VECTOR || vtype == VectorType::FLAT_VECTOR)) {
		D_ASSERT(!vector->auxiliary);
	}
	if (type.id() == LogicalTypeId::VARCHAR) {
		// verify that the string is correct unicode
		switch (vtype) {
		case VectorType::FLAT_VECTOR: {
			auto &validity = FlatVector::Validity(*vector);
			auto strings = FlatVector::GetData<string_t>(*vector);
			for (idx_t i = 0; i < count; i++) {
				auto oidx = sel->get_index(i);
				if (validity.RowIsValid(oidx)) {
					strings[oidx].Verify();
				}
			}
			break;
		}
		default:
			break;
		}
	}

	if (type.id() == LogicalTypeId::BIT) {
		switch (vtype) {
		case VectorType::FLAT_VECTOR: {
			auto &validity = FlatVector::Validity(*vector);
			auto strings = FlatVector::GetData<string_t>(*vector);
			for (idx_t i = 0; i < count; i++) {
				auto oidx = sel->get_index(i);
				if (validity.RowIsValid(oidx)) {
					auto buf = strings[oidx].GetDataUnsafe();
					D_ASSERT(*buf >= 0 && *buf < 8);
				}
			}
			break;
		}
		default:
			break;
		}
	}

	if (type.InternalType() == PhysicalType::STRUCT) {
		auto &child_types = StructType::GetChildTypes(type);
		D_ASSERT(!child_types.empty());
		// create a selection vector of the non-null entries of the struct vector
		auto &children = StructVector::GetEntries(*vector);
		D_ASSERT(child_types.size() == children.size());
		for (idx_t child_idx = 0; child_idx < children.size(); child_idx++) {
			D_ASSERT(children[child_idx]->GetType() == child_types[child_idx].second);
			Vector::Verify(*children[child_idx], sel_p, count);
			if (vtype == VectorType::CONSTANT_VECTOR) {
				D_ASSERT(children[child_idx]->GetVectorType() == VectorType::CONSTANT_VECTOR);
				if (ConstantVector::IsNull(*vector)) {
					D_ASSERT(ConstantVector::IsNull(*children[child_idx]));
				}
			}
			if (vtype != VectorType::FLAT_VECTOR) {
				continue;
			}
			ValidityMask *child_validity;
			SelectionVector owned_child_sel;
			const SelectionVector *child_sel = &owned_child_sel;
			if (children[child_idx]->GetVectorType() == VectorType::FLAT_VECTOR) {
				child_sel = FlatVector::IncrementalSelectionVector();
				child_validity = &FlatVector::Validity(*children[child_idx]);
			} else if (children[child_idx]->GetVectorType() == VectorType::DICTIONARY_VECTOR) {
				auto &child = DictionaryVector::Child(*children[child_idx]);
				if (child.GetVectorType() != VectorType::FLAT_VECTOR) {
					continue;
				}
				child_validity = &FlatVector::Validity(child);
				child_sel = &DictionaryVector::SelVector(*children[child_idx]);
			} else if (children[child_idx]->GetVectorType() == VectorType::CONSTANT_VECTOR) {
				child_sel = ConstantVector::ZeroSelectionVector(count, owned_child_sel);
				child_validity = &ConstantVector::Validity(*children[child_idx]);
			} else {
				continue;
			}
			// for any NULL entry in the struct, the child should be NULL as well
			auto &validity = FlatVector::Validity(*vector);
			for (idx_t i = 0; i < count; i++) {
				auto index = sel->get_index(i);
				if (!validity.RowIsValid(index)) {
					auto child_index = child_sel->get_index(sel_p.get_index(i));
					D_ASSERT(!child_validity->RowIsValid(child_index));
				}
			}
		}
		if (vector->GetType().id() == LogicalTypeId::MAP) {
			VerifyMap(*vector, *sel, count);
		}

		if (vector->GetType().id() == LogicalTypeId::UNION) {
			VerifyUnion(*vector, *sel, count);
		}
	}

	if (type.InternalType() == PhysicalType::LIST) {
		if (vtype == VectorType::CONSTANT_VECTOR) {
			if (!ConstantVector::IsNull(*vector)) {
				auto &child = ListVector::GetEntry(*vector);
				SelectionVector child_sel(ListVector::GetListSize(*vector));
				idx_t child_count = 0;
				auto le = ConstantVector::GetData<list_entry_t>(*vector);
				D_ASSERT(le->offset + le->length <= ListVector::GetListSize(*vector));
				for (idx_t k = 0; k < le->length; k++) {
					child_sel.set_index(child_count++, le->offset + k);
				}
				Vector::Verify(child, child_sel, child_count);
			}
		} else if (vtype == VectorType::FLAT_VECTOR) {
			auto &validity = FlatVector::Validity(*vector);
			auto &child = ListVector::GetEntry(*vector);
			auto child_size = ListVector::GetListSize(*vector);
			auto list_data = FlatVector::GetData<list_entry_t>(*vector);
			idx_t total_size = 0;
			for (idx_t i = 0; i < count; i++) {
				auto idx = sel->get_index(i);
				auto &le = list_data[idx];
				if (validity.RowIsValid(idx)) {
					D_ASSERT(le.offset + le.length <= child_size);
					total_size += le.length;
				}
			}
			SelectionVector child_sel(total_size);
			idx_t child_count = 0;
			for (idx_t i = 0; i < count; i++) {
				auto idx = sel->get_index(i);
				auto &le = list_data[idx];
				if (validity.RowIsValid(idx)) {
					D_ASSERT(le.offset + le.length <= child_size);
					for (idx_t k = 0; k < le.length; k++) {
						child_sel.set_index(child_count++, le.offset + k);
					}
				}
			}
			Vector::Verify(child, child_sel, child_count);
		}
	}
#endif
}

void Vector::Verify(idx_t count) {
	auto flat_sel = FlatVector::IncrementalSelectionVector();
	Verify(*this, *flat_sel, count);
}

void FlatVector::SetNull(Vector &vector, idx_t idx, bool is_null) {
	D_ASSERT(vector.GetVectorType() == VectorType::FLAT_VECTOR);
	vector.validity.Set(idx, !is_null);
	if (is_null && vector.GetType().InternalType() == PhysicalType::STRUCT) {
		// set all child entries to null as well
		auto &entries = StructVector::GetEntries(vector);
		for (auto &entry : entries) {
			FlatVector::SetNull(*entry, idx, is_null);
		}
	}
}

void ConstantVector::SetNull(Vector &vector, bool is_null) {
	D_ASSERT(vector.GetVectorType() == VectorType::CONSTANT_VECTOR);
	vector.validity.Set(0, !is_null);
	if (is_null && vector.GetType().InternalType() == PhysicalType::STRUCT) {
		// set all child entries to null as well
		auto &entries = StructVector::GetEntries(vector);
		for (auto &entry : entries) {
			entry->SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(*entry, is_null);
		}
	}
}

const SelectionVector *ConstantVector::ZeroSelectionVector(idx_t count, SelectionVector &owned_sel) {
	if (count <= STANDARD_VECTOR_SIZE) {
		return ConstantVector::ZeroSelectionVector();
	}
	owned_sel.Initialize(count);
	for (idx_t i = 0; i < count; i++) {
		owned_sel.set_index(i, 0);
	}
	return &owned_sel;
}

void ConstantVector::Reference(Vector &vector, Vector &source, idx_t position, idx_t count) {
	auto &source_type = source.GetType();
	switch (source_type.InternalType()) {
	case PhysicalType::LIST: {
		// retrieve the list entry from the source vector
		UnifiedVectorFormat vdata;
		source.ToUnifiedFormat(count, vdata);

		auto list_index = vdata.sel->get_index(position);
		if (!vdata.validity.RowIsValid(list_index)) {
			// list is null: create null value
			Value null_value(source_type);
			vector.Reference(null_value);
			break;
		}

		auto list_data = (list_entry_t *)vdata.data;
		auto list_entry = list_data[list_index];

		// add the list entry as the first element of "vector"
		// FIXME: we only need to allocate space for 1 tuple here
		auto target_data = FlatVector::GetData<list_entry_t>(vector);
		target_data[0] = list_entry;

		// create a reference to the child list of the source vector
		auto &child = ListVector::GetEntry(vector);
		child.Reference(ListVector::GetEntry(source));

		ListVector::SetListSize(vector, ListVector::GetListSize(source));
		vector.SetVectorType(VectorType::CONSTANT_VECTOR);
		break;
	}
	case PhysicalType::STRUCT: {
		UnifiedVectorFormat vdata;
		source.ToUnifiedFormat(count, vdata);

		auto struct_index = vdata.sel->get_index(position);
		if (!vdata.validity.RowIsValid(struct_index)) {
			// null struct: create null value
			Value null_value(source_type);
			vector.Reference(null_value);
			break;
		}

		// struct: pass constant reference into child entries
		auto &source_entries = StructVector::GetEntries(source);
		auto &target_entries = StructVector::GetEntries(vector);
		for (idx_t i = 0; i < source_entries.size(); i++) {
			ConstantVector::Reference(*target_entries[i], *source_entries[i], position, count);
		}
		vector.SetVectorType(VectorType::CONSTANT_VECTOR);
		vector.validity.Set(0, true);
		break;
	}
	default:
		// default behavior: get a value from the vector and reference it
		// this is not that expensive for scalar types
		auto value = source.GetValue(position);
		vector.Reference(value);
		D_ASSERT(vector.GetVectorType() == VectorType::CONSTANT_VECTOR);
		break;
	}
}

string_t StringVector::AddString(Vector &vector, const char *data, idx_t len) {
	return StringVector::AddString(vector, string_t(data, len));
}

string_t StringVector::AddStringOrBlob(Vector &vector, const char *data, idx_t len) {
	return StringVector::AddStringOrBlob(vector, string_t(data, len));
}

string_t StringVector::AddString(Vector &vector, const char *data) {
	return StringVector::AddString(vector, string_t(data, strlen(data)));
}

string_t StringVector::AddString(Vector &vector, const string &data) {
	return StringVector::AddString(vector, string_t(data.c_str(), data.size()));
}

string_t StringVector::AddString(Vector &vector, string_t data) {
	D_ASSERT(vector.GetType().id() == LogicalTypeId::VARCHAR);
	if (data.IsInlined()) {
		// string will be inlined: no need to store in string heap
		return data;
	}
	if (!vector.auxiliary) {
		vector.auxiliary = make_buffer<VectorStringBuffer>();
	}
	D_ASSERT(vector.auxiliary->GetBufferType() == VectorBufferType::STRING_BUFFER);
	auto &string_buffer = (VectorStringBuffer &)*vector.auxiliary;
	return string_buffer.AddString(data);
}

string_t StringVector::AddStringOrBlob(Vector &vector, string_t data) {
	D_ASSERT(vector.GetType().InternalType() == PhysicalType::VARCHAR);
	if (data.IsInlined()) {
		// string will be inlined: no need to store in string heap
		return data;
	}
	if (!vector.auxiliary) {
		vector.auxiliary = make_buffer<VectorStringBuffer>();
	}
	D_ASSERT(vector.auxiliary->GetBufferType() == VectorBufferType::STRING_BUFFER);
	auto &string_buffer = (VectorStringBuffer &)*vector.auxiliary;
	return string_buffer.AddBlob(data);
}

string_t StringVector::EmptyString(Vector &vector, idx_t len) {
	D_ASSERT(vector.GetType().InternalType() == PhysicalType::VARCHAR);
	if (len < string_t::INLINE_LENGTH) {
		return string_t(len);
	}
	if (!vector.auxiliary) {
		vector.auxiliary = make_buffer<VectorStringBuffer>();
	}
	D_ASSERT(vector.auxiliary->GetBufferType() == VectorBufferType::STRING_BUFFER);
	auto &string_buffer = (VectorStringBuffer &)*vector.auxiliary;
	return string_buffer.EmptyString(len);
}

void StringVector::AddHandle(Vector &vector, BufferHandle handle) {
	D_ASSERT(vector.GetType().InternalType() == PhysicalType::VARCHAR);
	if (!vector.auxiliary) {
		vector.auxiliary = make_buffer<VectorStringBuffer>();
	}
	auto &string_buffer = (VectorStringBuffer &)*vector.auxiliary;
	string_buffer.AddHeapReference(make_buffer<ManagedVectorBuffer>(std::move(handle)));
}

void StringVector::AddBuffer(Vector &vector, buffer_ptr<VectorBuffer> buffer) {
	D_ASSERT(vector.GetType().InternalType() == PhysicalType::VARCHAR);
	D_ASSERT(buffer.get() != vector.auxiliary.get());
	if (!vector.auxiliary) {
		vector.auxiliary = make_buffer<VectorStringBuffer>();
	}
	auto &string_buffer = (VectorStringBuffer &)*vector.auxiliary;
	string_buffer.AddHeapReference(std::move(buffer));
}

void StringVector::AddHeapReference(Vector &vector, Vector &other) {
	D_ASSERT(vector.GetType().InternalType() == PhysicalType::VARCHAR);
	D_ASSERT(other.GetType().InternalType() == PhysicalType::VARCHAR);

	if (other.GetVectorType() == VectorType::DICTIONARY_VECTOR) {
		StringVector::AddHeapReference(vector, DictionaryVector::Child(other));
		return;
	}
	if (!other.auxiliary) {
		return;
	}
	StringVector::AddBuffer(vector, other.auxiliary);
}

string_t FSSTVector::AddCompressedString(Vector &vector, const char *data, idx_t len) {
	return FSSTVector::AddCompressedString(vector, string_t(data, len));
}

string_t FSSTVector::AddCompressedString(Vector &vector, string_t data) {
	D_ASSERT(vector.GetType().InternalType() == PhysicalType::VARCHAR);
	if (data.IsInlined()) {
		// string will be inlined: no need to store in string heap
		return data;
	}
	if (!vector.auxiliary) {
		vector.auxiliary = make_buffer<VectorFSSTStringBuffer>();
	}
	D_ASSERT(vector.auxiliary->GetBufferType() == VectorBufferType::FSST_BUFFER);
	auto &fsst_string_buffer = (VectorFSSTStringBuffer &)*vector.auxiliary;
	return fsst_string_buffer.AddBlob(data);
}

void *FSSTVector::GetDecoder(const Vector &vector) {
	D_ASSERT(vector.GetType().InternalType() == PhysicalType::VARCHAR);
	if (!vector.auxiliary) {
		throw InternalException("GetDecoder called on FSST Vector without registered buffer");
	}
	D_ASSERT(vector.auxiliary->GetBufferType() == VectorBufferType::FSST_BUFFER);
	auto &fsst_string_buffer = (VectorFSSTStringBuffer &)*vector.auxiliary;
	return (duckdb_fsst_decoder_t *)fsst_string_buffer.GetDecoder();
}

void FSSTVector::RegisterDecoder(Vector &vector, buffer_ptr<void> &duckdb_fsst_decoder) {
	D_ASSERT(vector.GetType().InternalType() == PhysicalType::VARCHAR);

	if (!vector.auxiliary) {
		vector.auxiliary = make_buffer<VectorFSSTStringBuffer>();
	}
	D_ASSERT(vector.auxiliary->GetBufferType() == VectorBufferType::FSST_BUFFER);

	auto &fsst_string_buffer = (VectorFSSTStringBuffer &)*vector.auxiliary;
	fsst_string_buffer.AddDecoder(duckdb_fsst_decoder);
}

void FSSTVector::SetCount(Vector &vector, idx_t count) {
	D_ASSERT(vector.GetType().InternalType() == PhysicalType::VARCHAR);

	if (!vector.auxiliary) {
		vector.auxiliary = make_buffer<VectorFSSTStringBuffer>();
	}
	D_ASSERT(vector.auxiliary->GetBufferType() == VectorBufferType::FSST_BUFFER);

	auto &fsst_string_buffer = (VectorFSSTStringBuffer &)*vector.auxiliary;
	fsst_string_buffer.SetCount(count);
}

idx_t FSSTVector::GetCount(Vector &vector) {
	D_ASSERT(vector.GetType().InternalType() == PhysicalType::VARCHAR);

	if (!vector.auxiliary) {
		vector.auxiliary = make_buffer<VectorFSSTStringBuffer>();
	}
	D_ASSERT(vector.auxiliary->GetBufferType() == VectorBufferType::FSST_BUFFER);

	auto &fsst_string_buffer = (VectorFSSTStringBuffer &)*vector.auxiliary;
	return fsst_string_buffer.GetCount();
}

void FSSTVector::DecompressVector(const Vector &src, Vector &dst, idx_t src_offset, idx_t dst_offset, idx_t copy_count,
                                  const SelectionVector *sel) {
	D_ASSERT(src.GetVectorType() == VectorType::FSST_VECTOR);
	D_ASSERT(dst.GetVectorType() == VectorType::FLAT_VECTOR);
	auto dst_mask = FlatVector::Validity(dst);
	auto ldata = FSSTVector::GetCompressedData<string_t>(src);
	auto tdata = FlatVector::GetData<string_t>(dst);
	for (idx_t i = 0; i < copy_count; i++) {
		auto source_idx = sel->get_index(src_offset + i);
		auto target_idx = dst_offset + i;
		string_t compressed_string = ldata[source_idx];
		if (dst_mask.RowIsValid(target_idx) && compressed_string.GetSize() > 0) {
			tdata[target_idx] = FSSTPrimitives::DecompressValue(FSSTVector::GetDecoder(src), dst,
			                                                    (unsigned char *)compressed_string.GetDataUnsafe(),
			                                                    compressed_string.GetSize());
		} else {
			tdata[target_idx] = string_t(nullptr, 0);
		}
	}
}

Vector &MapVector::GetKeys(Vector &vector) {
	auto &entries = StructVector::GetEntries(ListVector::GetEntry(vector));
	D_ASSERT(entries.size() == 2);
	return *entries[0];
}
Vector &MapVector::GetValues(Vector &vector) {
	auto &entries = StructVector::GetEntries(ListVector::GetEntry(vector));
	D_ASSERT(entries.size() == 2);
	return *entries[1];
}

const Vector &MapVector::GetKeys(const Vector &vector) {
	return GetKeys((Vector &)vector);
}
const Vector &MapVector::GetValues(const Vector &vector) {
	return GetValues((Vector &)vector);
}

vector<unique_ptr<Vector>> &StructVector::GetEntries(Vector &vector) {
	D_ASSERT(vector.GetType().id() == LogicalTypeId::STRUCT || vector.GetType().id() == LogicalTypeId::UNION);

	if (vector.GetVectorType() == VectorType::DICTIONARY_VECTOR) {
		auto &child = DictionaryVector::Child(vector);
		return StructVector::GetEntries(child);
	}
	D_ASSERT(vector.GetVectorType() == VectorType::FLAT_VECTOR ||
	         vector.GetVectorType() == VectorType::CONSTANT_VECTOR);
	D_ASSERT(vector.auxiliary);
	D_ASSERT(vector.auxiliary->GetBufferType() == VectorBufferType::STRUCT_BUFFER);
	return ((VectorStructBuffer *)vector.auxiliary.get())->GetChildren();
}

const vector<unique_ptr<Vector>> &StructVector::GetEntries(const Vector &vector) {
	return GetEntries((Vector &)vector);
}

const Vector &ListVector::GetEntry(const Vector &vector) {
	D_ASSERT(vector.GetType().id() == LogicalTypeId::LIST || vector.GetType().id() == LogicalTypeId::MAP);
	if (vector.GetVectorType() == VectorType::DICTIONARY_VECTOR) {
		auto &child = DictionaryVector::Child(vector);
		return ListVector::GetEntry(child);
	}
	D_ASSERT(vector.GetVectorType() == VectorType::FLAT_VECTOR ||
	         vector.GetVectorType() == VectorType::CONSTANT_VECTOR);
	D_ASSERT(vector.auxiliary);
	D_ASSERT(vector.auxiliary->GetBufferType() == VectorBufferType::LIST_BUFFER);
	return ((VectorListBuffer *)vector.auxiliary.get())->GetChild();
}

Vector &ListVector::GetEntry(Vector &vector) {
	const Vector &cvector = vector;
	return const_cast<Vector &>(ListVector::GetEntry(cvector));
}

void ListVector::Reserve(Vector &vector, idx_t required_capacity) {
	D_ASSERT(vector.GetType().id() == LogicalTypeId::LIST || vector.GetType().id() == LogicalTypeId::MAP);
	D_ASSERT(vector.GetVectorType() == VectorType::FLAT_VECTOR ||
	         vector.GetVectorType() == VectorType::CONSTANT_VECTOR);
	D_ASSERT(vector.auxiliary);
	D_ASSERT(vector.auxiliary->GetBufferType() == VectorBufferType::LIST_BUFFER);
	auto &child_buffer = *((VectorListBuffer *)vector.auxiliary.get());
	child_buffer.Reserve(required_capacity);
}

template <class T>
void TemplatedSearchInMap(Vector &keys, idx_t count, T key, vector<idx_t> &offsets, bool is_key_null, idx_t offset,
                          idx_t length) {
	UnifiedVectorFormat vector_data;
	keys.ToUnifiedFormat(count, vector_data);
	auto data = (T *)vector_data.data;
	auto validity_mask = vector_data.validity;

	if (is_key_null) {
		for (idx_t i = offset; i < offset + length; i++) {
			if (!validity_mask.RowIsValid(i)) {
				offsets.push_back(i);
			}
		}
	} else {
		for (idx_t i = offset; i < offset + length; i++) {
			if (!validity_mask.RowIsValid(i)) {
				continue;
			}
			if (key == data[i]) {
				offsets.push_back(i);
			}
		}
	}
}

template <class T>
void TemplatedSearchInMap(Vector &keys, idx_t count, const Value &key, vector<idx_t> &offsets, bool is_key_null,
                          idx_t offset, idx_t length) {
	TemplatedSearchInMap<T>(keys, count, key.template GetValueUnsafe<T>(), offsets, is_key_null, offset, length);
}

void SearchStringInMap(Vector &keys, idx_t count, const string &key, vector<idx_t> &offsets, bool is_key_null,
                       idx_t offset, idx_t length) {
	UnifiedVectorFormat vector_data;
	keys.ToUnifiedFormat(count, vector_data);
	auto data = (string_t *)vector_data.data;
	auto validity_mask = vector_data.validity;
	if (is_key_null) {
		for (idx_t i = offset; i < offset + length; i++) {
			if (!validity_mask.RowIsValid(i)) {
				offsets.push_back(i);
			}
		}
	} else {
		string_t key_str_t(key);
		for (idx_t i = offset; i < offset + length; i++) {
			if (!validity_mask.RowIsValid(i)) {
				continue;
			}
			if (Equals::Operation<string_t>(data[i], key_str_t)) {
				offsets.push_back(i);
			}
		}
	}
}

vector<idx_t> MapVector::Search(Vector &keys, idx_t count, const Value &key, list_entry_t &entry) {
	vector<idx_t> offsets;

	switch (keys.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		TemplatedSearchInMap<int8_t>(keys, count, key, offsets, key.IsNull(), entry.offset, entry.length);
		break;
	case PhysicalType::INT16:
		TemplatedSearchInMap<int16_t>(keys, count, key, offsets, key.IsNull(), entry.offset, entry.length);
		break;
	case PhysicalType::INT32:
		TemplatedSearchInMap<int32_t>(keys, count, key, offsets, key.IsNull(), entry.offset, entry.length);
		break;
	case PhysicalType::INT64:
		TemplatedSearchInMap<int64_t>(keys, count, key, offsets, key.IsNull(), entry.offset, entry.length);
		break;
	case PhysicalType::INT128:
		TemplatedSearchInMap<hugeint_t>(keys, count, key, offsets, key.IsNull(), entry.offset, entry.length);
		break;
	case PhysicalType::UINT8:
		TemplatedSearchInMap<uint8_t>(keys, count, key, offsets, key.IsNull(), entry.offset, entry.length);
		break;
	case PhysicalType::UINT16:
		TemplatedSearchInMap<uint16_t>(keys, count, key, offsets, key.IsNull(), entry.offset, entry.length);
		break;
	case PhysicalType::UINT32:
		TemplatedSearchInMap<uint32_t>(keys, count, key, offsets, key.IsNull(), entry.offset, entry.length);
		break;
	case PhysicalType::UINT64:
		TemplatedSearchInMap<uint64_t>(keys, count, key, offsets, key.IsNull(), entry.offset, entry.length);
		break;
	case PhysicalType::FLOAT:
		TemplatedSearchInMap<float>(keys, count, key, offsets, key.IsNull(), entry.offset, entry.length);
		break;
	case PhysicalType::DOUBLE:
		TemplatedSearchInMap<double>(keys, count, key, offsets, key.IsNull(), entry.offset, entry.length);
		break;
	case PhysicalType::INTERVAL:
		TemplatedSearchInMap<interval_t>(keys, count, key, offsets, key.IsNull(), entry.offset, entry.length);
		break;
	case PhysicalType::VARCHAR:
		SearchStringInMap(keys, count, StringValue::Get(key), offsets, key.IsNull(), entry.offset, entry.length);
		break;
	default:
		throw InvalidTypeException(keys.GetType().id(), "Invalid type for List Vector Search");
	}
	return offsets;
}

Value FlatVector::GetValuesFromOffsets(Vector &values, vector<idx_t> &offsets) {
	vector<Value> list_values;
	list_values.reserve(offsets.size());
	for (auto &offset : offsets) {
		list_values.push_back(values.GetValue(offset));
	}
	return Value::LIST(values.GetType(), std::move(list_values));
}

idx_t ListVector::GetListSize(const Vector &vec) {
	if (vec.GetVectorType() == VectorType::DICTIONARY_VECTOR) {
		auto &child = DictionaryVector::Child(vec);
		return ListVector::GetListSize(child);
	}
	D_ASSERT(vec.auxiliary);
	return ((VectorListBuffer &)*vec.auxiliary).size;
}

idx_t ListVector::GetListCapacity(const Vector &vec) {
	if (vec.GetVectorType() == VectorType::DICTIONARY_VECTOR) {
		auto &child = DictionaryVector::Child(vec);
		return ListVector::GetListSize(child);
	}
	D_ASSERT(vec.auxiliary);
	return ((VectorListBuffer &)*vec.auxiliary).capacity;
}

void ListVector::ReferenceEntry(Vector &vector, Vector &other) {
	D_ASSERT(vector.GetType().id() == LogicalTypeId::LIST);
	D_ASSERT(vector.GetVectorType() == VectorType::FLAT_VECTOR ||
	         vector.GetVectorType() == VectorType::CONSTANT_VECTOR);
	D_ASSERT(other.GetType().id() == LogicalTypeId::LIST);
	D_ASSERT(other.GetVectorType() == VectorType::FLAT_VECTOR || other.GetVectorType() == VectorType::CONSTANT_VECTOR);
	vector.auxiliary = other.auxiliary;
}

void ListVector::SetListSize(Vector &vec, idx_t size) {
	if (vec.GetVectorType() == VectorType::DICTIONARY_VECTOR) {
		auto &child = DictionaryVector::Child(vec);
		ListVector::SetListSize(child, size);
	}
	((VectorListBuffer &)*vec.auxiliary).size = size;
}

void ListVector::Append(Vector &target, const Vector &source, idx_t source_size, idx_t source_offset) {
	if (source_size - source_offset == 0) {
		//! Nothing to add
		return;
	}
	auto &target_buffer = (VectorListBuffer &)*target.auxiliary;
	target_buffer.Append(source, source_size, source_offset);
}

void ListVector::Append(Vector &target, const Vector &source, const SelectionVector &sel, idx_t source_size,
                        idx_t source_offset) {
	if (source_size - source_offset == 0) {
		//! Nothing to add
		return;
	}
	auto &target_buffer = (VectorListBuffer &)*target.auxiliary;
	target_buffer.Append(source, sel, source_size, source_offset);
}

void ListVector::PushBack(Vector &target, const Value &insert) {
	auto &target_buffer = (VectorListBuffer &)*target.auxiliary;
	target_buffer.PushBack(insert);
}

idx_t ListVector::GetConsecutiveChildList(Vector &list, Vector &result, idx_t offset, idx_t count) {

	auto info = ListVector::GetConsecutiveChildListInfo(list, offset, count);
	if (info.needs_slicing) {
		SelectionVector sel(info.child_list_info.length);
		ListVector::GetConsecutiveChildSelVector(list, sel, offset, count);

		result.Slice(sel, info.child_list_info.length);
		result.Flatten(info.child_list_info.length);
	}
	return info.child_list_info.length;
}

ConsecutiveChildListInfo ListVector::GetConsecutiveChildListInfo(Vector &list, idx_t offset, idx_t count) {

	ConsecutiveChildListInfo info;
	UnifiedVectorFormat unified_list_data;
	list.ToUnifiedFormat(offset + count, unified_list_data);
	auto list_data = (list_entry_t *)unified_list_data.data;

	// find the first non-NULL entry
	idx_t first_length = 0;
	for (idx_t i = offset; i < offset + count; i++) {
		auto idx = unified_list_data.sel->get_index(i);
		if (!unified_list_data.validity.RowIsValid(idx)) {
			continue;
		}
		info.child_list_info.offset = list_data[idx].offset;
		first_length = list_data[idx].length;
		break;
	}

	// small performance improvement for constant vectors
	// avoids iterating over all their (constant) elements
	if (list.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		info.child_list_info.length = first_length;
		return info;
	}

	// now get the child count and determine whether the children are stored consecutively
	// also determine if a flat vector has pseudo constant values (all offsets + length the same)
	// this can happen e.g. for UNNESTs
	bool is_consecutive = true;
	for (idx_t i = offset; i < offset + count; i++) {
		auto idx = unified_list_data.sel->get_index(i);
		if (!unified_list_data.validity.RowIsValid(idx)) {
			continue;
		}
		if (list_data[idx].offset != info.child_list_info.offset || list_data[idx].length != first_length) {
			info.is_constant = false;
		}
		if (list_data[idx].offset != info.child_list_info.offset + info.child_list_info.length) {
			is_consecutive = false;
		}
		info.child_list_info.length += list_data[idx].length;
	}

	if (info.is_constant) {
		info.child_list_info.length = first_length;
	}
	if (!info.is_constant && !is_consecutive) {
		info.needs_slicing = true;
	}

	return info;
}

void ListVector::GetConsecutiveChildSelVector(Vector &list, SelectionVector &sel, idx_t offset, idx_t count) {

	UnifiedVectorFormat unified_list_data;
	list.ToUnifiedFormat(offset + count, unified_list_data);
	auto list_data = (list_entry_t *)unified_list_data.data;

	//	SelectionVector child_sel(info.second.length);
	idx_t entry = 0;
	for (idx_t i = offset; i < offset + count; i++) {
		auto idx = unified_list_data.sel->get_index(i);
		if (!unified_list_data.validity.RowIsValid(idx)) {
			continue;
		}
		for (idx_t k = 0; k < list_data[idx].length; k++) {
			//			child_sel.set_index(entry++, list_data[idx].offset + k);
			sel.set_index(entry++, list_data[idx].offset + k);
		}
	}
	//
	//	result.Slice(child_sel, info.second.length);
	//	result.Flatten(info.second.length);
	//	info.second.offset = 0;
}

// Union vector
const Vector &UnionVector::GetMember(const Vector &vector, idx_t member_index) {
	D_ASSERT(member_index < UnionType::GetMemberCount(vector.GetType()));
	auto &entries = StructVector::GetEntries(vector);
	return *entries[member_index + 1]; // skip the "tag" entry
}

Vector &UnionVector::GetMember(Vector &vector, idx_t member_index) {
	D_ASSERT(member_index < UnionType::GetMemberCount(vector.GetType()));
	auto &entries = StructVector::GetEntries(vector);
	return *entries[member_index + 1]; // skip the "tag" entry
}

const Vector &UnionVector::GetTags(const Vector &vector) {
	// the tag vector is always the first struct child.
	return *StructVector::GetEntries(vector)[0];
}

Vector &UnionVector::GetTags(Vector &vector) {
	// the tag vector is always the first struct child.
	return *StructVector::GetEntries(vector)[0];
}

void UnionVector::SetToMember(Vector &union_vector, union_tag_t tag, Vector &member_vector, idx_t count,
                              bool keep_tags_for_null) {
	D_ASSERT(union_vector.GetType().id() == LogicalTypeId::UNION);
	D_ASSERT(tag < UnionType::GetMemberCount(union_vector.GetType()));

	// Set the union member to the specified vector
	UnionVector::GetMember(union_vector, tag).Reference(member_vector);
	auto &tag_vector = UnionVector::GetTags(union_vector);

	if (member_vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		// if the member vector is constant, we can set the union to constant as well
		union_vector.SetVectorType(VectorType::CONSTANT_VECTOR);
		ConstantVector::GetData<union_tag_t>(tag_vector)[0] = tag;
		ConstantVector::SetNull(union_vector, ConstantVector::IsNull(member_vector));

	} else {
		// otherwise flatten and set to flatvector
		member_vector.Flatten(count);
		union_vector.SetVectorType(VectorType::FLAT_VECTOR);

		if (member_vector.validity.AllValid()) {
			// if the member vector is all valid, we can set the tag to constant
			tag_vector.SetVectorType(VectorType::CONSTANT_VECTOR);
			auto tag_data = ConstantVector::GetData<union_tag_t>(tag_vector);
			*tag_data = tag;
		} else {
			tag_vector.SetVectorType(VectorType::FLAT_VECTOR);
			if (keep_tags_for_null) {
				FlatVector::Validity(tag_vector).SetAllValid(count);
				FlatVector::Validity(union_vector).SetAllValid(count);
			} else {
				// ensure the tags have the same validity as the member
				FlatVector::Validity(union_vector) = FlatVector::Validity(member_vector);
				FlatVector::Validity(tag_vector) = FlatVector::Validity(member_vector);
			}

			auto tag_data = FlatVector::GetData<union_tag_t>(tag_vector);
			memset(tag_data, tag, count);
		}
	}

	// Set the non-selected members to constant null vectors
	for (idx_t i = 0; i < UnionType::GetMemberCount(union_vector.GetType()); i++) {
		if (i != tag) {
			auto &member = UnionVector::GetMember(union_vector, i);
			member.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(member, true);
		}
	}
}

union_tag_t UnionVector::GetTag(const Vector &vector, idx_t index) {
	// the tag vector is always the first struct child.
	auto &tag_vector = *StructVector::GetEntries(vector)[0];
	if (tag_vector.GetVectorType() == VectorType::DICTIONARY_VECTOR) {
		auto &child = DictionaryVector::Child(tag_vector);
		return FlatVector::GetData<union_tag_t>(child)[index];
	}
	if (tag_vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		return ConstantVector::GetData<union_tag_t>(tag_vector)[0];
	}
	return FlatVector::GetData<union_tag_t>(tag_vector)[index];
}

} // namespace duckdb








namespace duckdb {

buffer_ptr<VectorBuffer> VectorBuffer::CreateStandardVector(PhysicalType type, idx_t capacity) {
	return make_buffer<VectorBuffer>(capacity * GetTypeIdSize(type));
}

buffer_ptr<VectorBuffer> VectorBuffer::CreateConstantVector(PhysicalType type) {
	return make_buffer<VectorBuffer>(GetTypeIdSize(type));
}

buffer_ptr<VectorBuffer> VectorBuffer::CreateConstantVector(const LogicalType &type) {
	return VectorBuffer::CreateConstantVector(type.InternalType());
}

buffer_ptr<VectorBuffer> VectorBuffer::CreateStandardVector(const LogicalType &type, idx_t capacity) {
	return VectorBuffer::CreateStandardVector(type.InternalType(), capacity);
}

VectorStringBuffer::VectorStringBuffer() : VectorBuffer(VectorBufferType::STRING_BUFFER) {
}

VectorStringBuffer::VectorStringBuffer(VectorBufferType type) : VectorBuffer(type) {
}

VectorFSSTStringBuffer::VectorFSSTStringBuffer() : VectorStringBuffer(VectorBufferType::FSST_BUFFER) {
}

VectorStructBuffer::VectorStructBuffer() : VectorBuffer(VectorBufferType::STRUCT_BUFFER) {
}

VectorStructBuffer::VectorStructBuffer(const LogicalType &type, idx_t capacity)
    : VectorBuffer(VectorBufferType::STRUCT_BUFFER) {
	auto &child_types = StructType::GetChildTypes(type);
	for (auto &child_type : child_types) {
		auto vector = make_unique<Vector>(child_type.second, capacity);
		children.push_back(std::move(vector));
	}
}

VectorStructBuffer::VectorStructBuffer(Vector &other, const SelectionVector &sel, idx_t count)
    : VectorBuffer(VectorBufferType::STRUCT_BUFFER) {
	auto &other_vector = StructVector::GetEntries(other);
	for (auto &child_vector : other_vector) {
		auto vector = make_unique<Vector>(*child_vector, sel, count);
		children.push_back(std::move(vector));
	}
}

VectorStructBuffer::~VectorStructBuffer() {
}

VectorListBuffer::VectorListBuffer(unique_ptr<Vector> vector, idx_t initial_capacity)
    : VectorBuffer(VectorBufferType::LIST_BUFFER), capacity(initial_capacity), child(std::move(vector)) {
}

VectorListBuffer::VectorListBuffer(const LogicalType &list_type, idx_t initial_capacity)
    : VectorBuffer(VectorBufferType::LIST_BUFFER), capacity(initial_capacity),
      child(make_unique<Vector>(ListType::GetChildType(list_type), initial_capacity)) {
}

void VectorListBuffer::Reserve(idx_t to_reserve) {
	if (to_reserve > capacity) {
		idx_t new_capacity = NextPowerOfTwo(to_reserve);
		D_ASSERT(new_capacity >= to_reserve);
		child->Resize(capacity, new_capacity);
		capacity = new_capacity;
	}
}

void VectorListBuffer::Append(const Vector &to_append, idx_t to_append_size, idx_t source_offset) {
	Reserve(size + to_append_size - source_offset);
	VectorOperations::Copy(to_append, *child, to_append_size, source_offset, size);
	size += to_append_size - source_offset;
}

void VectorListBuffer::Append(const Vector &to_append, const SelectionVector &sel, idx_t to_append_size,
                              idx_t source_offset) {
	Reserve(size + to_append_size - source_offset);
	VectorOperations::Copy(to_append, *child, sel, to_append_size, source_offset, size);
	size += to_append_size - source_offset;
}

void VectorListBuffer::PushBack(const Value &insert) {
	if (size + 1 > capacity) {
		child->Resize(capacity, capacity * 2);
		capacity *= 2;
	}
	child->SetValue(size++, insert);
}

VectorListBuffer::~VectorListBuffer() {
}

ManagedVectorBuffer::ManagedVectorBuffer(BufferHandle handle)
    : VectorBuffer(VectorBufferType::MANAGED_BUFFER), handle(std::move(handle)) {
}

ManagedVectorBuffer::~ManagedVectorBuffer() {
}

} // namespace duckdb





namespace duckdb {

class VectorCacheBuffer : public VectorBuffer {
public:
	explicit VectorCacheBuffer(Allocator &allocator, const LogicalType &type_p, idx_t capacity_p = STANDARD_VECTOR_SIZE)
	    : VectorBuffer(VectorBufferType::OPAQUE_BUFFER), type(type_p), capacity(capacity_p) {
		auto internal_type = type.InternalType();
		switch (internal_type) {
		case PhysicalType::LIST: {
			// memory for the list offsets
			owned_data = allocator.Allocate(capacity * GetTypeIdSize(internal_type));
			// child data of the list
			auto &child_type = ListType::GetChildType(type);
			child_caches.push_back(make_buffer<VectorCacheBuffer>(allocator, child_type));
			auto child_vector = make_unique<Vector>(child_type, false, false);
			auxiliary = make_unique<VectorListBuffer>(std::move(child_vector));
			break;
		}
		case PhysicalType::STRUCT: {
			auto &child_types = StructType::GetChildTypes(type);
			for (auto &child_type : child_types) {
				child_caches.push_back(make_buffer<VectorCacheBuffer>(allocator, child_type.second));
			}
			auto struct_buffer = make_unique<VectorStructBuffer>(type);
			auxiliary = std::move(struct_buffer);
			break;
		}
		default:
			owned_data = allocator.Allocate(capacity * GetTypeIdSize(internal_type));
			break;
		}
	}

	void ResetFromCache(Vector &result, const buffer_ptr<VectorBuffer> &buffer) {
		D_ASSERT(type == result.GetType());
		auto internal_type = type.InternalType();
		result.vector_type = VectorType::FLAT_VECTOR;
		AssignSharedPointer(result.buffer, buffer);
		result.validity.Reset();
		switch (internal_type) {
		case PhysicalType::LIST: {
			result.data = owned_data.get();
			// reinitialize the VectorListBuffer
			AssignSharedPointer(result.auxiliary, auxiliary);
			// propagate through child
			auto &list_buffer = (VectorListBuffer &)*result.auxiliary;
			list_buffer.capacity = capacity;
			list_buffer.size = 0;
			list_buffer.SetAuxiliaryData(nullptr);

			auto &list_child = list_buffer.GetChild();
			auto &child_cache = (VectorCacheBuffer &)*child_caches[0];
			child_cache.ResetFromCache(list_child, child_caches[0]);
			break;
		}
		case PhysicalType::STRUCT: {
			// struct does not have data
			result.data = nullptr;
			// reinitialize the VectorStructBuffer
			auxiliary->SetAuxiliaryData(nullptr);
			AssignSharedPointer(result.auxiliary, auxiliary);
			// propagate through children
			auto &children = ((VectorStructBuffer &)*result.auxiliary).GetChildren();
			for (idx_t i = 0; i < children.size(); i++) {
				auto &child_cache = (VectorCacheBuffer &)*child_caches[i];
				child_cache.ResetFromCache(*children[i], child_caches[i]);
			}
			break;
		}
		default:
			// regular type: no aux data and reset data to cached data
			result.data = owned_data.get();
			result.auxiliary.reset();
			break;
		}
	}

	const LogicalType &GetType() {
		return type;
	}

private:
	//! The type of the vector cache
	LogicalType type;
	//! Owned data
	AllocatedData owned_data;
	//! Child caches (if any). Used for nested types.
	vector<buffer_ptr<VectorBuffer>> child_caches;
	//! Aux data for the vector (if any)
	buffer_ptr<VectorBuffer> auxiliary;
	//! Capacity of the vector
	idx_t capacity;
};

VectorCache::VectorCache(Allocator &allocator, const LogicalType &type_p, idx_t capacity_p) {
	buffer = make_unique<VectorCacheBuffer>(allocator, type_p, capacity_p);
}

void VectorCache::ResetFromCache(Vector &result) const {
	D_ASSERT(buffer);
	auto &vcache = (VectorCacheBuffer &)*buffer;
	vcache.ResetFromCache(result, buffer);
}

const LogicalType &VectorCache::GetType() const {
	auto &vcache = (VectorCacheBuffer &)*buffer;
	return vcache.GetType();
}

} // namespace duckdb


namespace duckdb {

const SelectionVector *ConstantVector::ZeroSelectionVector() {
	static const SelectionVector ZERO_SELECTION_VECTOR = SelectionVector((sel_t *)ConstantVector::ZERO_VECTOR);
	return &ZERO_SELECTION_VECTOR;
}

const SelectionVector *FlatVector::IncrementalSelectionVector() {
	static const SelectionVector INCREMENTAL_SELECTION_VECTOR;
	return &INCREMENTAL_SELECTION_VECTOR;
}

const sel_t ConstantVector::ZERO_VECTOR[STANDARD_VECTOR_SIZE] = {0};

} // namespace duckdb






















#include <cmath>

namespace duckdb {

LogicalType::LogicalType() : LogicalType(LogicalTypeId::INVALID) {
}

LogicalType::LogicalType(LogicalTypeId id) : id_(id) {
	physical_type_ = GetInternalType();
}
LogicalType::LogicalType(LogicalTypeId id, shared_ptr<ExtraTypeInfo> type_info_p)
    : id_(id), type_info_(std::move(type_info_p)) {
	physical_type_ = GetInternalType();
}

LogicalType::LogicalType(const LogicalType &other)
    : id_(other.id_), physical_type_(other.physical_type_), type_info_(other.type_info_) {
}

LogicalType::LogicalType(LogicalType &&other) noexcept
    : id_(other.id_), physical_type_(other.physical_type_), type_info_(std::move(other.type_info_)) {
}

hash_t LogicalType::Hash() const {
	return duckdb::Hash<uint8_t>((uint8_t)id_);
}

PhysicalType LogicalType::GetInternalType() {
	switch (id_) {
	case LogicalTypeId::BOOLEAN:
		return PhysicalType::BOOL;
	case LogicalTypeId::TINYINT:
		return PhysicalType::INT8;
	case LogicalTypeId::UTINYINT:
		return PhysicalType::UINT8;
	case LogicalTypeId::SMALLINT:
		return PhysicalType::INT16;
	case LogicalTypeId::USMALLINT:
		return PhysicalType::UINT16;
	case LogicalTypeId::SQLNULL:
	case LogicalTypeId::DATE:
	case LogicalTypeId::INTEGER:
		return PhysicalType::INT32;
	case LogicalTypeId::UINTEGER:
		return PhysicalType::UINT32;
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::TIMESTAMP_NS:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIME_TZ:
	case LogicalTypeId::TIMESTAMP_TZ:
		return PhysicalType::INT64;
	case LogicalTypeId::UBIGINT:
		return PhysicalType::UINT64;
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::UUID:
		return PhysicalType::INT128;
	case LogicalTypeId::FLOAT:
		return PhysicalType::FLOAT;
	case LogicalTypeId::DOUBLE:
		return PhysicalType::DOUBLE;
	case LogicalTypeId::DECIMAL: {
		if (!type_info_) {
			return PhysicalType::INVALID;
		}
		auto width = DecimalType::GetWidth(*this);
		if (width <= Decimal::MAX_WIDTH_INT16) {
			return PhysicalType::INT16;
		} else if (width <= Decimal::MAX_WIDTH_INT32) {
			return PhysicalType::INT32;
		} else if (width <= Decimal::MAX_WIDTH_INT64) {
			return PhysicalType::INT64;
		} else if (width <= Decimal::MAX_WIDTH_INT128) {
			return PhysicalType::INT128;
		} else {
			throw InternalException("Decimal has a width of %d which is bigger than the maximum supported width of %d",
			                        width, DecimalType::MaxWidth());
		}
	}
	case LogicalTypeId::VARCHAR:
	case LogicalTypeId::CHAR:
	case LogicalTypeId::BLOB:
	case LogicalTypeId::BIT:
		return PhysicalType::VARCHAR;
	case LogicalTypeId::INTERVAL:
		return PhysicalType::INTERVAL;
	case LogicalTypeId::UNION:
	case LogicalTypeId::STRUCT:
		return PhysicalType::STRUCT;
	case LogicalTypeId::LIST:
	case LogicalTypeId::MAP:
		return PhysicalType::LIST;
	case LogicalTypeId::POINTER:
		// LCOV_EXCL_START
		if (sizeof(uintptr_t) == sizeof(uint32_t)) {
			return PhysicalType::UINT32;
		} else if (sizeof(uintptr_t) == sizeof(uint64_t)) {
			return PhysicalType::UINT64;
		} else {
			throw InternalException("Unsupported pointer size");
		}
		// LCOV_EXCL_STOP
	case LogicalTypeId::VALIDITY:
		return PhysicalType::BIT;
	case LogicalTypeId::ENUM: {
		if (!type_info_) {
			return PhysicalType::INVALID;
		}
		return EnumType::GetPhysicalType(*this);
	}
	case LogicalTypeId::TABLE:
	case LogicalTypeId::LAMBDA:
	case LogicalTypeId::ANY:
	case LogicalTypeId::INVALID:
	case LogicalTypeId::UNKNOWN:
		return PhysicalType::INVALID;
	case LogicalTypeId::USER:
		return PhysicalType::UNKNOWN;
	case LogicalTypeId::AGGREGATE_STATE:
		return PhysicalType::VARCHAR;
	default:
		throw InternalException("Invalid LogicalType %s", ToString());
	}
}

constexpr const LogicalTypeId LogicalType::INVALID;
constexpr const LogicalTypeId LogicalType::SQLNULL;
constexpr const LogicalTypeId LogicalType::BOOLEAN;
constexpr const LogicalTypeId LogicalType::TINYINT;
constexpr const LogicalTypeId LogicalType::UTINYINT;
constexpr const LogicalTypeId LogicalType::SMALLINT;
constexpr const LogicalTypeId LogicalType::USMALLINT;
constexpr const LogicalTypeId LogicalType::INTEGER;
constexpr const LogicalTypeId LogicalType::UINTEGER;
constexpr const LogicalTypeId LogicalType::BIGINT;
constexpr const LogicalTypeId LogicalType::UBIGINT;
constexpr const LogicalTypeId LogicalType::HUGEINT;
constexpr const LogicalTypeId LogicalType::UUID;
constexpr const LogicalTypeId LogicalType::FLOAT;
constexpr const LogicalTypeId LogicalType::DOUBLE;
constexpr const LogicalTypeId LogicalType::DATE;

constexpr const LogicalTypeId LogicalType::TIMESTAMP;
constexpr const LogicalTypeId LogicalType::TIMESTAMP_MS;
constexpr const LogicalTypeId LogicalType::TIMESTAMP_NS;
constexpr const LogicalTypeId LogicalType::TIMESTAMP_S;

constexpr const LogicalTypeId LogicalType::TIME;

constexpr const LogicalTypeId LogicalType::TIME_TZ;
constexpr const LogicalTypeId LogicalType::TIMESTAMP_TZ;

constexpr const LogicalTypeId LogicalType::HASH;
constexpr const LogicalTypeId LogicalType::POINTER;

constexpr const LogicalTypeId LogicalType::VARCHAR;

constexpr const LogicalTypeId LogicalType::BLOB;
constexpr const LogicalTypeId LogicalType::BIT;
constexpr const LogicalTypeId LogicalType::INTERVAL;
constexpr const LogicalTypeId LogicalType::ROW_TYPE;

// TODO these are incomplete and should maybe not exist as such
constexpr const LogicalTypeId LogicalType::TABLE;
constexpr const LogicalTypeId LogicalType::LAMBDA;

constexpr const LogicalTypeId LogicalType::ANY;

const vector<LogicalType> LogicalType::Numeric() {
	vector<LogicalType> types = {LogicalType::TINYINT,   LogicalType::SMALLINT,  LogicalType::INTEGER,
	                             LogicalType::BIGINT,    LogicalType::HUGEINT,   LogicalType::FLOAT,
	                             LogicalType::DOUBLE,    LogicalTypeId::DECIMAL, LogicalType::UTINYINT,
	                             LogicalType::USMALLINT, LogicalType::UINTEGER,  LogicalType::UBIGINT};
	return types;
}

const vector<LogicalType> LogicalType::Integral() {
	vector<LogicalType> types = {LogicalType::TINYINT,   LogicalType::SMALLINT, LogicalType::INTEGER,
	                             LogicalType::BIGINT,    LogicalType::HUGEINT,  LogicalType::UTINYINT,
	                             LogicalType::USMALLINT, LogicalType::UINTEGER, LogicalType::UBIGINT};
	return types;
}

const vector<LogicalType> LogicalType::AllTypes() {
	vector<LogicalType> types = {
	    LogicalType::BOOLEAN,   LogicalType::TINYINT,  LogicalType::SMALLINT,  LogicalType::INTEGER,
	    LogicalType::BIGINT,    LogicalType::DATE,     LogicalType::TIMESTAMP, LogicalType::DOUBLE,
	    LogicalType::FLOAT,     LogicalType::VARCHAR,  LogicalType::BLOB,      LogicalType::BIT,
	    LogicalType::INTERVAL,  LogicalType::HUGEINT,  LogicalTypeId::DECIMAL, LogicalType::UTINYINT,
	    LogicalType::USMALLINT, LogicalType::UINTEGER, LogicalType::UBIGINT,   LogicalType::TIME,
	    LogicalTypeId::LIST,    LogicalTypeId::STRUCT, LogicalType::TIME_TZ,   LogicalType::TIMESTAMP_TZ,
	    LogicalTypeId::MAP,     LogicalTypeId::UNION,  LogicalType::UUID};
	return types;
}

const PhysicalType ROW_TYPE = PhysicalType::INT64;

// LCOV_EXCL_START
string TypeIdToString(PhysicalType type) {
	switch (type) {
	case PhysicalType::BOOL:
		return "BOOL";
	case PhysicalType::INT8:
		return "INT8";
	case PhysicalType::INT16:
		return "INT16";
	case PhysicalType::INT32:
		return "INT32";
	case PhysicalType::INT64:
		return "INT64";
	case PhysicalType::UINT8:
		return "UINT8";
	case PhysicalType::UINT16:
		return "UINT16";
	case PhysicalType::UINT32:
		return "UINT32";
	case PhysicalType::UINT64:
		return "UINT64";
	case PhysicalType::INT128:
		return "INT128";
	case PhysicalType::FLOAT:
		return "FLOAT";
	case PhysicalType::DOUBLE:
		return "DOUBLE";
	case PhysicalType::VARCHAR:
		return "VARCHAR";
	case PhysicalType::INTERVAL:
		return "INTERVAL";
	case PhysicalType::STRUCT:
		return "STRUCT";
	case PhysicalType::LIST:
		return "LIST";
	case PhysicalType::INVALID:
		return "INVALID";
	case PhysicalType::BIT:
		return "BIT";
	case PhysicalType::UNKNOWN:
		return "UNKNOWN";
	}
	return "INVALID";
}
// LCOV_EXCL_STOP

idx_t GetTypeIdSize(PhysicalType type) {
	switch (type) {
	case PhysicalType::BIT:
	case PhysicalType::BOOL:
		return sizeof(bool);
	case PhysicalType::INT8:
		return sizeof(int8_t);
	case PhysicalType::INT16:
		return sizeof(int16_t);
	case PhysicalType::INT32:
		return sizeof(int32_t);
	case PhysicalType::INT64:
		return sizeof(int64_t);
	case PhysicalType::UINT8:
		return sizeof(uint8_t);
	case PhysicalType::UINT16:
		return sizeof(uint16_t);
	case PhysicalType::UINT32:
		return sizeof(uint32_t);
	case PhysicalType::UINT64:
		return sizeof(uint64_t);
	case PhysicalType::INT128:
		return sizeof(hugeint_t);
	case PhysicalType::FLOAT:
		return sizeof(float);
	case PhysicalType::DOUBLE:
		return sizeof(double);
	case PhysicalType::VARCHAR:
		return sizeof(string_t);
	case PhysicalType::INTERVAL:
		return sizeof(interval_t);
	case PhysicalType::STRUCT:
	case PhysicalType::UNKNOWN:
		return 0; // no own payload
	case PhysicalType::LIST:
		return sizeof(list_entry_t); // offset + len
	default:
		throw InternalException("Invalid PhysicalType for GetTypeIdSize");
	}
}

bool TypeIsConstantSize(PhysicalType type) {
	return (type >= PhysicalType::BOOL && type <= PhysicalType::DOUBLE) || type == PhysicalType::INTERVAL ||
	       type == PhysicalType::INT128;
}
bool TypeIsIntegral(PhysicalType type) {
	return (type >= PhysicalType::UINT8 && type <= PhysicalType::INT64) || type == PhysicalType::INT128;
}
bool TypeIsNumeric(PhysicalType type) {
	return (type >= PhysicalType::UINT8 && type <= PhysicalType::DOUBLE) || type == PhysicalType::INT128;
}
bool TypeIsInteger(PhysicalType type) {
	return (type >= PhysicalType::UINT8 && type <= PhysicalType::INT64) || type == PhysicalType::INT128;
}

// LCOV_EXCL_START
string LogicalTypeIdToString(LogicalTypeId id) {
	switch (id) {
	case LogicalTypeId::BOOLEAN:
		return "BOOLEAN";
	case LogicalTypeId::TINYINT:
		return "TINYINT";
	case LogicalTypeId::SMALLINT:
		return "SMALLINT";
	case LogicalTypeId::INTEGER:
		return "INTEGER";
	case LogicalTypeId::BIGINT:
		return "BIGINT";
	case LogicalTypeId::HUGEINT:
		return "HUGEINT";
	case LogicalTypeId::UUID:
		return "UUID";
	case LogicalTypeId::UTINYINT:
		return "UTINYINT";
	case LogicalTypeId::USMALLINT:
		return "USMALLINT";
	case LogicalTypeId::UINTEGER:
		return "UINTEGER";
	case LogicalTypeId::UBIGINT:
		return "UBIGINT";
	case LogicalTypeId::DATE:
		return "DATE";
	case LogicalTypeId::TIME:
		return "TIME";
	case LogicalTypeId::TIMESTAMP:
		return "TIMESTAMP";
	case LogicalTypeId::TIMESTAMP_MS:
		return "TIMESTAMP_MS";
	case LogicalTypeId::TIMESTAMP_NS:
		return "TIMESTAMP_NS";
	case LogicalTypeId::TIMESTAMP_SEC:
		return "TIMESTAMP_S";
	case LogicalTypeId::TIMESTAMP_TZ:
		return "TIMESTAMP WITH TIME ZONE";
	case LogicalTypeId::TIME_TZ:
		return "TIME WITH TIME ZONE";
	case LogicalTypeId::FLOAT:
		return "FLOAT";
	case LogicalTypeId::DOUBLE:
		return "DOUBLE";
	case LogicalTypeId::DECIMAL:
		return "DECIMAL";
	case LogicalTypeId::VARCHAR:
		return "VARCHAR";
	case LogicalTypeId::BLOB:
		return "BLOB";
	case LogicalTypeId::CHAR:
		return "CHAR";
	case LogicalTypeId::INTERVAL:
		return "INTERVAL";
	case LogicalTypeId::SQLNULL:
		return "NULL";
	case LogicalTypeId::ANY:
		return "ANY";
	case LogicalTypeId::VALIDITY:
		return "VALIDITY";
	case LogicalTypeId::STRUCT:
		return "STRUCT";
	case LogicalTypeId::LIST:
		return "LIST";
	case LogicalTypeId::MAP:
		return "MAP";
	case LogicalTypeId::POINTER:
		return "POINTER";
	case LogicalTypeId::TABLE:
		return "TABLE";
	case LogicalTypeId::LAMBDA:
		return "LAMBDA";
	case LogicalTypeId::INVALID:
		return "INVALID";
	case LogicalTypeId::UNION:
		return "UNION";
	case LogicalTypeId::UNKNOWN:
		return "UNKNOWN";
	case LogicalTypeId::ENUM:
		return "ENUM";
	case LogicalTypeId::AGGREGATE_STATE:
		return "AGGREGATE_STATE";
	case LogicalTypeId::USER:
		return "USER";
	case LogicalTypeId::BIT:
		return "BIT";
	}
	return "UNDEFINED";
}

string LogicalType::ToString() const {
	auto alias = GetAlias();
	if (!alias.empty()) {
		return alias;
	}
	switch (id_) {
	case LogicalTypeId::STRUCT: {
		if (!type_info_) {
			return "STRUCT";
		}
		auto &child_types = StructType::GetChildTypes(*this);
		string ret = "STRUCT(";
		for (size_t i = 0; i < child_types.size(); i++) {
			ret += child_types[i].first + " " + child_types[i].second.ToString();
			if (i < child_types.size() - 1) {
				ret += ", ";
			}
		}
		ret += ")";
		return ret;
	}
	case LogicalTypeId::LIST: {
		if (!type_info_) {
			return "LIST";
		}
		return ListType::GetChildType(*this).ToString() + "[]";
	}
	case LogicalTypeId::MAP: {
		if (!type_info_) {
			return "MAP";
		}
		auto &key_type = MapType::KeyType(*this);
		auto &value_type = MapType::ValueType(*this);
		return "MAP(" + key_type.ToString() + ", " + value_type.ToString() + ")";
	}
	case LogicalTypeId::UNION: {
		if (!type_info_) {
			return "UNION";
		}
		string ret = "UNION(";
		size_t count = UnionType::GetMemberCount(*this);
		for (size_t i = 0; i < count; i++) {
			ret += UnionType::GetMemberName(*this, i) + " " + UnionType::GetMemberType(*this, i).ToString();
			if (i < count - 1) {
				ret += ", ";
			}
		}
		ret += ")";
		return ret;
	}
	case LogicalTypeId::DECIMAL: {
		if (!type_info_) {
			return "DECIMAL";
		}
		auto width = DecimalType::GetWidth(*this);
		auto scale = DecimalType::GetScale(*this);
		if (width == 0) {
			return "DECIMAL";
		}
		return StringUtil::Format("DECIMAL(%d,%d)", width, scale);
	}
	case LogicalTypeId::ENUM: {
		return KeywordHelper::WriteOptionallyQuoted(EnumType::GetTypeName(*this));
	}
	case LogicalTypeId::USER: {
		return KeywordHelper::WriteOptionallyQuoted(UserType::GetTypeName(*this));
	}
	case LogicalTypeId::AGGREGATE_STATE: {
		return AggregateStateType::GetTypeName(*this);
	}
	default:
		return LogicalTypeIdToString(id_);
	}
}
// LCOV_EXCL_STOP

LogicalTypeId TransformStringToLogicalTypeId(const string &str) {
	auto type = DefaultTypeGenerator::GetDefaultType(str);
	if (type == LogicalTypeId::INVALID) {
		// This is a User Type, at this point we don't know if its one of the User Defined Types or an error
		// It is checked in the binder
		type = LogicalTypeId::USER;
	}
	return type;
}

LogicalType TransformStringToLogicalType(const string &str) {
	if (StringUtil::Lower(str) == "null") {
		return LogicalType::SQLNULL;
	}
	return Parser::ParseColumnList("dummy " + str).GetColumn(LogicalIndex(0)).Type();
}

LogicalType TransformStringToLogicalType(const string &str, ClientContext &context) {
	auto type = TransformStringToLogicalType(str);
	return type.id() == LogicalTypeId::USER
	           ? Catalog::GetSystemCatalog(context).GetType(context, SYSTEM_CATALOG, DEFAULT_SCHEMA, str)
	           : type;
}

bool LogicalType::IsIntegral() const {
	switch (id_) {
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::HUGEINT:
		return true;
	default:
		return false;
	}
}

bool LogicalType::IsNumeric() const {
	switch (id_) {
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
		return true;
	default:
		return false;
	}
}

bool LogicalType::IsValid() const {
	return id() != LogicalTypeId::INVALID && id() != LogicalTypeId::UNKNOWN;
}

bool LogicalType::GetDecimalProperties(uint8_t &width, uint8_t &scale) const {
	switch (id_) {
	case LogicalTypeId::SQLNULL:
		width = 0;
		scale = 0;
		break;
	case LogicalTypeId::BOOLEAN:
		width = 1;
		scale = 0;
		break;
	case LogicalTypeId::TINYINT:
		// tinyint: [-127, 127] = DECIMAL(3,0)
		width = 3;
		scale = 0;
		break;
	case LogicalTypeId::SMALLINT:
		// smallint: [-32767, 32767] = DECIMAL(5,0)
		width = 5;
		scale = 0;
		break;
	case LogicalTypeId::INTEGER:
		// integer: [-2147483647, 2147483647] = DECIMAL(10,0)
		width = 10;
		scale = 0;
		break;
	case LogicalTypeId::BIGINT:
		// bigint: [-9223372036854775807, 9223372036854775807] = DECIMAL(19,0)
		width = 19;
		scale = 0;
		break;
	case LogicalTypeId::UTINYINT:
		// UInt8 — [0 : 255]
		width = 3;
		scale = 0;
		break;
	case LogicalTypeId::USMALLINT:
		// UInt16 — [0 : 65535]
		width = 5;
		scale = 0;
		break;
	case LogicalTypeId::UINTEGER:
		// UInt32 — [0 : 4294967295]
		width = 10;
		scale = 0;
		break;
	case LogicalTypeId::UBIGINT:
		// UInt64 — [0 : 18446744073709551615]
		width = 20;
		scale = 0;
		break;
	case LogicalTypeId::HUGEINT:
		// hugeint: max size decimal (38, 0)
		// note that a hugeint is not guaranteed to fit in this
		width = 38;
		scale = 0;
		break;
	case LogicalTypeId::DECIMAL:
		width = DecimalType::GetWidth(*this);
		scale = DecimalType::GetScale(*this);
		break;
	default:
		return false;
	}
	return true;
}

//! Grows Decimal width/scale when appropriate
static LogicalType DecimalSizeCheck(const LogicalType &left, const LogicalType &right) {
	D_ASSERT(left.id() == LogicalTypeId::DECIMAL || right.id() == LogicalTypeId::DECIMAL);
	D_ASSERT(left.id() != right.id());

	//! Make sure the 'right' is the DECIMAL type
	if (left.id() == LogicalTypeId::DECIMAL) {
		return DecimalSizeCheck(right, left);
	}
	auto width = DecimalType::GetWidth(right);
	auto scale = DecimalType::GetScale(right);

	uint8_t other_width;
	uint8_t other_scale;
	bool success = left.GetDecimalProperties(other_width, other_scale);
	if (!success) {
		throw InternalException("Type provided to DecimalSizeCheck was not a numeric type");
	}
	D_ASSERT(other_scale == 0);
	const auto effective_width = width - scale;
	if (other_width > effective_width) {
		auto new_width = other_width + scale;
		//! Cap the width at max, if an actual value exceeds this, an exception will be thrown later
		if (new_width > DecimalType::MaxWidth()) {
			new_width = DecimalType::MaxWidth();
		}
		return LogicalType::DECIMAL(new_width, scale);
	}
	return right;
}

static LogicalType CombineNumericTypes(const LogicalType &left, const LogicalType &right) {
	D_ASSERT(left.id() != right.id());
	if (left.id() > right.id()) {
		// this method is symmetric
		// arrange it so the left type is smaller to limit the number of options we need to check
		return CombineNumericTypes(right, left);
	}
	if (CastRules::ImplicitCast(left, right) >= 0) {
		// we can implicitly cast left to right, return right
		//! Depending on the type, we might need to grow the `width` of the DECIMAL type
		if (right.id() == LogicalTypeId::DECIMAL) {
			return DecimalSizeCheck(left, right);
		}
		return right;
	}
	if (CastRules::ImplicitCast(right, left) >= 0) {
		// we can implicitly cast right to left, return left
		//! Depending on the type, we might need to grow the `width` of the DECIMAL type
		if (left.id() == LogicalTypeId::DECIMAL) {
			return DecimalSizeCheck(right, left);
		}
		return left;
	}
	// we can't cast implicitly either way and types are not equal
	// this happens when left is signed and right is unsigned
	// e.g. INTEGER and UINTEGER
	// in this case we need to upcast to make sure the types fit

	if (left.id() == LogicalTypeId::BIGINT || right.id() == LogicalTypeId::UBIGINT) {
		return LogicalType::HUGEINT;
	}
	if (left.id() == LogicalTypeId::INTEGER || right.id() == LogicalTypeId::UINTEGER) {
		return LogicalType::BIGINT;
	}
	if (left.id() == LogicalTypeId::SMALLINT || right.id() == LogicalTypeId::USMALLINT) {
		return LogicalType::INTEGER;
	}
	if (left.id() == LogicalTypeId::TINYINT || right.id() == LogicalTypeId::UTINYINT) {
		return LogicalType::SMALLINT;
	}
	throw InternalException("Cannot combine these numeric types!?");
}

LogicalType LogicalType::MaxLogicalType(const LogicalType &left, const LogicalType &right) {
	// we always prefer aliased types
	if (!left.GetAlias().empty()) {
		return left;
	}
	if (!right.GetAlias().empty()) {
		return right;
	}
	if (left.id() != right.id() && left.IsNumeric() && right.IsNumeric()) {
		return CombineNumericTypes(left, right);
	} else if (left.id() == LogicalTypeId::UNKNOWN) {
		return right;
	} else if (right.id() == LogicalTypeId::UNKNOWN) {
		return left;
	} else if (left.id() < right.id()) {
		return right;
	}
	if (right.id() < left.id()) {
		return left;
	}
	// Since both left and right are equal we get the left type as our type_id for checks
	auto type_id = left.id();
	if (type_id == LogicalTypeId::ENUM) {
		// If both types are different ENUMs we do a string comparison.
		return left == right ? left : LogicalType::VARCHAR;
	}
	if (type_id == LogicalTypeId::VARCHAR) {
		// varchar: use type that has collation (if any)
		if (StringType::GetCollation(right).empty()) {
			return left;
		}
		return right;
	}
	if (type_id == LogicalTypeId::DECIMAL) {
		// unify the width/scale so that the resulting decimal always fits
		// "width - scale" gives us the number of digits on the left side of the decimal point
		// "scale" gives us the number of digits allowed on the right of the decimal point
		// using the max of these of the two types gives us the new decimal size
		auto extra_width_left = DecimalType::GetWidth(left) - DecimalType::GetScale(left);
		auto extra_width_right = DecimalType::GetWidth(right) - DecimalType::GetScale(right);
		auto extra_width = MaxValue<uint8_t>(extra_width_left, extra_width_right);
		auto scale = MaxValue<uint8_t>(DecimalType::GetScale(left), DecimalType::GetScale(right));
		auto width = extra_width + scale;
		if (width > DecimalType::MaxWidth()) {
			// if the resulting decimal does not fit, we truncate the scale
			width = DecimalType::MaxWidth();
			scale = width - extra_width;
		}
		return LogicalType::DECIMAL(width, scale);
	}
	if (type_id == LogicalTypeId::LIST) {
		// list: perform max recursively on child type
		auto new_child = MaxLogicalType(ListType::GetChildType(left), ListType::GetChildType(right));
		return LogicalType::LIST(std::move(new_child));
	}
	if (type_id == LogicalTypeId::MAP) {
		// list: perform max recursively on child type
		auto new_child = MaxLogicalType(ListType::GetChildType(left), ListType::GetChildType(right));
		return LogicalType::MAP(std::move(new_child));
	}
	if (type_id == LogicalTypeId::STRUCT) {
		// struct: perform recursively
		auto &left_child_types = StructType::GetChildTypes(left);
		auto &right_child_types = StructType::GetChildTypes(right);
		if (left_child_types.size() != right_child_types.size()) {
			// child types are not of equal size, we can't cast anyway
			// just return the left child
			return left;
		}
		child_list_t<LogicalType> child_types;
		for (idx_t i = 0; i < left_child_types.size(); i++) {
			auto child_type = MaxLogicalType(left_child_types[i].second, right_child_types[i].second);
			child_types.push_back(make_pair(left_child_types[i].first, std::move(child_type)));
		}

		return LogicalType::STRUCT(std::move(child_types));
	}
	if (type_id == LogicalTypeId::UNION) {
		auto left_member_count = UnionType::GetMemberCount(left);
		auto right_member_count = UnionType::GetMemberCount(right);
		if (left_member_count != right_member_count) {
			// return the "larger" type, with the most members
			return left_member_count > right_member_count ? left : right;
		}
		// otherwise, keep left, dont try to meld the two together.
		return left;
	}
	// types are equal but no extra specifier: just return the type
	return left;
}

void LogicalType::Verify() const {
#ifdef DEBUG
	if (id_ == LogicalTypeId::DECIMAL) {
		D_ASSERT(DecimalType::GetWidth(*this) >= 1 && DecimalType::GetWidth(*this) <= Decimal::MAX_WIDTH_DECIMAL);
		D_ASSERT(DecimalType::GetScale(*this) >= 0 && DecimalType::GetScale(*this) <= DecimalType::GetWidth(*this));
	}
#endif
}

bool ApproxEqual(float ldecimal, float rdecimal) {
	if (Value::IsNan(ldecimal) && Value::IsNan(rdecimal)) {
		return true;
	}
	if (!Value::FloatIsFinite(ldecimal) || !Value::FloatIsFinite(rdecimal)) {
		return ldecimal == rdecimal;
	}
	float epsilon = std::fabs(rdecimal) * 0.01 + 0.00000001;
	return std::fabs(ldecimal - rdecimal) <= epsilon;
}

bool ApproxEqual(double ldecimal, double rdecimal) {
	if (Value::IsNan(ldecimal) && Value::IsNan(rdecimal)) {
		return true;
	}
	if (!Value::DoubleIsFinite(ldecimal) || !Value::DoubleIsFinite(rdecimal)) {
		return ldecimal == rdecimal;
	}
	double epsilon = std::fabs(rdecimal) * 0.01 + 0.00000001;
	return std::fabs(ldecimal - rdecimal) <= epsilon;
}

//===--------------------------------------------------------------------===//
// Extra Type Info
//===--------------------------------------------------------------------===//
enum class ExtraTypeInfoType : uint8_t {
	INVALID_TYPE_INFO = 0,
	GENERIC_TYPE_INFO = 1,
	DECIMAL_TYPE_INFO = 2,
	STRING_TYPE_INFO = 3,
	LIST_TYPE_INFO = 4,
	STRUCT_TYPE_INFO = 5,
	ENUM_TYPE_INFO = 6,
	USER_TYPE_INFO = 7,
	AGGREGATE_STATE_TYPE_INFO = 8
};

struct ExtraTypeInfo {
	explicit ExtraTypeInfo(ExtraTypeInfoType type) : type(type) {
	}
	explicit ExtraTypeInfo(ExtraTypeInfoType type, string alias) : type(type), alias(std::move(alias)) {
	}
	virtual ~ExtraTypeInfo() {
	}

	ExtraTypeInfoType type;
	string alias;
	TypeCatalogEntry *catalog_entry = nullptr;

public:
	bool Equals(ExtraTypeInfo *other_p) const {
		if (type == ExtraTypeInfoType::INVALID_TYPE_INFO || type == ExtraTypeInfoType::STRING_TYPE_INFO ||
		    type == ExtraTypeInfoType::GENERIC_TYPE_INFO) {
			if (!other_p) {
				if (!alias.empty()) {
					return false;
				}
				//! We only need to compare aliases when both types have them in this case
				return true;
			}
			if (alias != other_p->alias) {
				return false;
			}
			return true;
		}
		if (!other_p) {
			return false;
		}
		if (type != other_p->type) {
			return false;
		}
		return alias == other_p->alias && EqualsInternal(other_p);
	}
	//! Serializes a ExtraTypeInfo to a stand-alone binary blob
	virtual void Serialize(FieldWriter &writer) const {};
	//! Serializes a ExtraTypeInfo to a stand-alone binary blob
	static void Serialize(ExtraTypeInfo *info, FieldWriter &writer);
	//! Deserializes a blob back into an ExtraTypeInfo
	static shared_ptr<ExtraTypeInfo> Deserialize(FieldReader &reader);

protected:
	virtual bool EqualsInternal(ExtraTypeInfo *other_p) const {
		// Do nothing
		return true;
	}
};

void LogicalType::SetAlias(string alias) {
	if (!type_info_) {
		type_info_ = make_shared<ExtraTypeInfo>(ExtraTypeInfoType::GENERIC_TYPE_INFO, std::move(alias));
	} else {
		type_info_->alias = std::move(alias);
	}
}

string LogicalType::GetAlias() const {
	if (!type_info_) {
		return string();
	} else {
		return type_info_->alias;
	}
}

bool LogicalType::HasAlias() const {
	if (!type_info_) {
		return false;
	}
	return !type_info_->alias.empty();
}

void LogicalType::SetCatalog(LogicalType &type, TypeCatalogEntry *catalog_entry) {
	auto info = type.AuxInfo();
	D_ASSERT(info);
	((ExtraTypeInfo &)*info).catalog_entry = catalog_entry;
}
TypeCatalogEntry *LogicalType::GetCatalog(const LogicalType &type) {
	auto info = type.AuxInfo();
	if (!info) {
		return nullptr;
	}
	return ((ExtraTypeInfo &)*info).catalog_entry;
}

//===--------------------------------------------------------------------===//
// Decimal Type
//===--------------------------------------------------------------------===//
struct DecimalTypeInfo : public ExtraTypeInfo {
	DecimalTypeInfo(uint8_t width_p, uint8_t scale_p)
	    : ExtraTypeInfo(ExtraTypeInfoType::DECIMAL_TYPE_INFO), width(width_p), scale(scale_p) {
		D_ASSERT(width_p >= scale_p);
	}

	uint8_t width;
	uint8_t scale;

public:
	void Serialize(FieldWriter &writer) const override {
		writer.WriteField<uint8_t>(width);
		writer.WriteField<uint8_t>(scale);
	}

	static shared_ptr<ExtraTypeInfo> Deserialize(FieldReader &reader) {
		auto width = reader.ReadRequired<uint8_t>();
		auto scale = reader.ReadRequired<uint8_t>();
		return make_shared<DecimalTypeInfo>(width, scale);
	}

protected:
	bool EqualsInternal(ExtraTypeInfo *other_p) const override {
		auto &other = (DecimalTypeInfo &)*other_p;
		return width == other.width && scale == other.scale;
	}
};

uint8_t DecimalType::GetWidth(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::DECIMAL);
	auto info = type.AuxInfo();
	D_ASSERT(info);
	return ((DecimalTypeInfo &)*info).width;
}

uint8_t DecimalType::GetScale(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::DECIMAL);
	auto info = type.AuxInfo();
	D_ASSERT(info);
	return ((DecimalTypeInfo &)*info).scale;
}

uint8_t DecimalType::MaxWidth() {
	return DecimalWidth<hugeint_t>::max;
}

LogicalType LogicalType::DECIMAL(int width, int scale) {
	D_ASSERT(width >= scale);
	auto type_info = make_shared<DecimalTypeInfo>(width, scale);
	return LogicalType(LogicalTypeId::DECIMAL, std::move(type_info));
}

//===--------------------------------------------------------------------===//
// String Type
//===--------------------------------------------------------------------===//
struct StringTypeInfo : public ExtraTypeInfo {
	explicit StringTypeInfo(string collation_p)
	    : ExtraTypeInfo(ExtraTypeInfoType::STRING_TYPE_INFO), collation(std::move(collation_p)) {
	}

	string collation;

public:
	void Serialize(FieldWriter &writer) const override {
		writer.WriteString(collation);
	}

	static shared_ptr<ExtraTypeInfo> Deserialize(FieldReader &reader) {
		auto collation = reader.ReadRequired<string>();
		return make_shared<StringTypeInfo>(std::move(collation));
	}

protected:
	bool EqualsInternal(ExtraTypeInfo *other_p) const override {
		// collation info has no impact on equality
		return true;
	}
};

string StringType::GetCollation(const LogicalType &type) {
	if (type.id() != LogicalTypeId::VARCHAR) {
		return string();
	}
	auto info = type.AuxInfo();
	if (!info) {
		return string();
	}
	if (info->type == ExtraTypeInfoType::GENERIC_TYPE_INFO) {
		return string();
	}
	return ((StringTypeInfo &)*info).collation;
}

LogicalType LogicalType::VARCHAR_COLLATION(string collation) { // NOLINT
	auto string_info = make_shared<StringTypeInfo>(std::move(collation));
	return LogicalType(LogicalTypeId::VARCHAR, std::move(string_info));
}

//===--------------------------------------------------------------------===//
// List Type
//===--------------------------------------------------------------------===//
struct ListTypeInfo : public ExtraTypeInfo {
	explicit ListTypeInfo(LogicalType child_type_p)
	    : ExtraTypeInfo(ExtraTypeInfoType::LIST_TYPE_INFO), child_type(std::move(child_type_p)) {
	}

	LogicalType child_type;

public:
	void Serialize(FieldWriter &writer) const override {
		writer.WriteSerializable(child_type);
	}

	static shared_ptr<ExtraTypeInfo> Deserialize(FieldReader &reader) {
		auto child_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
		return make_shared<ListTypeInfo>(std::move(child_type));
	}

protected:
	bool EqualsInternal(ExtraTypeInfo *other_p) const override {
		auto &other = (ListTypeInfo &)*other_p;
		return child_type == other.child_type;
	}
};

const LogicalType &ListType::GetChildType(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::LIST || type.id() == LogicalTypeId::MAP);
	auto info = type.AuxInfo();
	D_ASSERT(info);
	return ((ListTypeInfo &)*info).child_type;
}

LogicalType LogicalType::LIST(LogicalType child) {
	auto info = make_shared<ListTypeInfo>(std::move(child));
	return LogicalType(LogicalTypeId::LIST, std::move(info));
}

//===--------------------------------------------------------------------===//
// Struct Type
//===--------------------------------------------------------------------===//
struct StructTypeInfo : public ExtraTypeInfo {
	explicit StructTypeInfo(child_list_t<LogicalType> child_types_p)
	    : ExtraTypeInfo(ExtraTypeInfoType::STRUCT_TYPE_INFO), child_types(std::move(child_types_p)) {
	}

	child_list_t<LogicalType> child_types;

public:
	void Serialize(FieldWriter &writer) const override {
		writer.WriteField<uint32_t>(child_types.size());
		auto &serializer = writer.GetSerializer();
		for (idx_t i = 0; i < child_types.size(); i++) {
			serializer.WriteString(child_types[i].first);
			child_types[i].second.Serialize(serializer);
		}
	}

	static shared_ptr<ExtraTypeInfo> Deserialize(FieldReader &reader) {
		child_list_t<LogicalType> child_list;
		auto child_types_size = reader.ReadRequired<uint32_t>();
		auto &source = reader.GetSource();
		for (uint32_t i = 0; i < child_types_size; i++) {
			auto name = source.Read<string>();
			auto type = LogicalType::Deserialize(source);
			child_list.push_back(make_pair(std::move(name), std::move(type)));
		}
		return make_shared<StructTypeInfo>(std::move(child_list));
	}

protected:
	bool EqualsInternal(ExtraTypeInfo *other_p) const override {
		auto &other = (StructTypeInfo &)*other_p;
		return child_types == other.child_types;
	}
};

struct AggregateStateTypeInfo : public ExtraTypeInfo {
	explicit AggregateStateTypeInfo(aggregate_state_t state_type_p)
	    : ExtraTypeInfo(ExtraTypeInfoType::AGGREGATE_STATE_TYPE_INFO), state_type(std::move(state_type_p)) {
	}

	aggregate_state_t state_type;

public:
	void Serialize(FieldWriter &writer) const override {
		auto &serializer = writer.GetSerializer();
		writer.WriteString(state_type.function_name);
		state_type.return_type.Serialize(serializer);
		writer.WriteField<uint32_t>(state_type.bound_argument_types.size());
		for (idx_t i = 0; i < state_type.bound_argument_types.size(); i++) {
			state_type.bound_argument_types[i].Serialize(serializer);
		}
	}

	static shared_ptr<ExtraTypeInfo> Deserialize(FieldReader &reader) {
		auto &source = reader.GetSource();

		auto function_name = reader.ReadRequired<string>();
		auto return_type = LogicalType::Deserialize(source);
		auto bound_argument_types_size = reader.ReadRequired<uint32_t>();
		vector<LogicalType> bound_argument_types;

		for (uint32_t i = 0; i < bound_argument_types_size; i++) {
			auto type = LogicalType::Deserialize(source);
			bound_argument_types.push_back(std::move(type));
		}
		return make_shared<AggregateStateTypeInfo>(
		    aggregate_state_t(std::move(function_name), std::move(return_type), std::move(bound_argument_types)));
	}

protected:
	bool EqualsInternal(ExtraTypeInfo *other_p) const override {
		auto &other = (AggregateStateTypeInfo &)*other_p;
		return state_type.function_name == other.state_type.function_name &&
		       state_type.return_type == other.state_type.return_type &&
		       state_type.bound_argument_types == other.state_type.bound_argument_types;
	}
};

const aggregate_state_t &AggregateStateType::GetStateType(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::AGGREGATE_STATE);
	auto info = type.AuxInfo();
	D_ASSERT(info);
	return ((AggregateStateTypeInfo &)*info).state_type;
}

const string AggregateStateType::GetTypeName(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::AGGREGATE_STATE);
	auto info = type.AuxInfo();
	if (!info) {
		return "AGGREGATE_STATE<?>";
	}
	auto aggr_state = ((AggregateStateTypeInfo &)*info).state_type;
	return "AGGREGATE_STATE<" + aggr_state.function_name + "(" +
	       StringUtil::Join(aggr_state.bound_argument_types, aggr_state.bound_argument_types.size(), ", ",
	                        [](const LogicalType &arg_type) { return arg_type.ToString(); }) +
	       ")" + "::" + aggr_state.return_type.ToString() + ">";
}

const child_list_t<LogicalType> &StructType::GetChildTypes(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::STRUCT || type.id() == LogicalTypeId::UNION);

	auto info = type.AuxInfo();
	D_ASSERT(info);
	return ((StructTypeInfo &)*info).child_types;
}

const LogicalType &StructType::GetChildType(const LogicalType &type, idx_t index) {
	auto &child_types = StructType::GetChildTypes(type);
	D_ASSERT(index < child_types.size());
	return child_types[index].second;
}

const string &StructType::GetChildName(const LogicalType &type, idx_t index) {
	auto &child_types = StructType::GetChildTypes(type);
	D_ASSERT(index < child_types.size());
	return child_types[index].first;
}

idx_t StructType::GetChildCount(const LogicalType &type) {
	return StructType::GetChildTypes(type).size();
}

LogicalType LogicalType::STRUCT(child_list_t<LogicalType> children) {
	auto info = make_shared<StructTypeInfo>(std::move(children));
	return LogicalType(LogicalTypeId::STRUCT, std::move(info));
}

LogicalType LogicalType::AGGREGATE_STATE(aggregate_state_t state_type) { // NOLINT
	auto info = make_shared<AggregateStateTypeInfo>(std::move(state_type));
	return LogicalType(LogicalTypeId::AGGREGATE_STATE, std::move(info));
}

//===--------------------------------------------------------------------===//
// Map Type
//===--------------------------------------------------------------------===//
LogicalType LogicalType::MAP(LogicalType child) {
	auto info = make_shared<ListTypeInfo>(std::move(child));
	return LogicalType(LogicalTypeId::MAP, std::move(info));
}

LogicalType LogicalType::MAP(LogicalType key, LogicalType value) {
	child_list_t<LogicalType> child_types;
	child_types.push_back({"key", std::move(key)});
	child_types.push_back({"value", std::move(value)});
	return LogicalType::MAP(LogicalType::STRUCT(std::move(child_types)));
}

const LogicalType &MapType::KeyType(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::MAP);
	return StructType::GetChildTypes(ListType::GetChildType(type))[0].second;
}

const LogicalType &MapType::ValueType(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::MAP);
	return StructType::GetChildTypes(ListType::GetChildType(type))[1].second;
}

//===--------------------------------------------------------------------===//
// Union Type
//===--------------------------------------------------------------------===//

LogicalType LogicalType::UNION(child_list_t<LogicalType> members) {
	D_ASSERT(members.size() > 0);
	D_ASSERT(members.size() <= UnionType::MAX_UNION_MEMBERS);
	// union types always have a hidden "tag" field in front
	members.insert(members.begin(), {"", LogicalType::TINYINT});
	auto info = make_shared<StructTypeInfo>(std::move(members));
	return LogicalType(LogicalTypeId::UNION, std::move(info));
}

const LogicalType &UnionType::GetMemberType(const LogicalType &type, idx_t index) {
	auto &child_types = StructType::GetChildTypes(type);
	D_ASSERT(index < child_types.size());
	// skip the "tag" field
	return child_types[index + 1].second;
}

const string &UnionType::GetMemberName(const LogicalType &type, idx_t index) {
	auto &child_types = StructType::GetChildTypes(type);
	D_ASSERT(index < child_types.size());
	// skip the "tag" field
	return child_types[index + 1].first;
}

idx_t UnionType::GetMemberCount(const LogicalType &type) {
	// dont count the "tag" field
	return StructType::GetChildTypes(type).size() - 1;
}
const child_list_t<LogicalType> UnionType::CopyMemberTypes(const LogicalType &type) {
	auto child_types = StructType::GetChildTypes(type);
	child_types.erase(child_types.begin());
	return child_types;
}

//===--------------------------------------------------------------------===//
// User Type
//===--------------------------------------------------------------------===//
struct UserTypeInfo : public ExtraTypeInfo {
	explicit UserTypeInfo(string name_p)
	    : ExtraTypeInfo(ExtraTypeInfoType::USER_TYPE_INFO), user_type_name(std::move(name_p)) {
	}

	string user_type_name;

public:
	void Serialize(FieldWriter &writer) const override {
		writer.WriteString(user_type_name);
	}

	static shared_ptr<ExtraTypeInfo> Deserialize(FieldReader &reader) {
		auto enum_name = reader.ReadRequired<string>();
		return make_shared<UserTypeInfo>(std::move(enum_name));
	}

protected:
	bool EqualsInternal(ExtraTypeInfo *other_p) const override {
		auto &other = (UserTypeInfo &)*other_p;
		return other.user_type_name == user_type_name;
	}
};

const string &UserType::GetTypeName(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::USER);
	auto info = type.AuxInfo();
	D_ASSERT(info);
	return ((UserTypeInfo &)*info).user_type_name;
}

LogicalType LogicalType::USER(const string &user_type_name) {
	auto info = make_shared<UserTypeInfo>(user_type_name);
	return LogicalType(LogicalTypeId::USER, std::move(info));
}

//===--------------------------------------------------------------------===//
// Enum Type
//===--------------------------------------------------------------------===//

enum EnumDictType : uint8_t { INVALID = 0, VECTOR_DICT = 1 };

struct EnumTypeInfo : public ExtraTypeInfo {
	explicit EnumTypeInfo(string enum_name_p, Vector &values_insert_order_p, idx_t dict_size_p)
	    : ExtraTypeInfo(ExtraTypeInfoType::ENUM_TYPE_INFO), dict_type(EnumDictType::VECTOR_DICT),
	      enum_name(std::move(enum_name_p)), values_insert_order(values_insert_order_p), dict_size(dict_size_p) {
	}
	EnumDictType dict_type;
	string enum_name;
	Vector values_insert_order;
	idx_t dict_size;

protected:
	// Equalities are only used in enums with different catalog entries
	bool EqualsInternal(ExtraTypeInfo *other_p) const override {
		auto &other = (EnumTypeInfo &)*other_p;
		if (dict_type != other.dict_type) {
			return false;
		}
		D_ASSERT(dict_type == EnumDictType::VECTOR_DICT);
		// We must check if both enums have the same size
		if (other.dict_size != dict_size) {
			return false;
		}
		auto other_vector_ptr = FlatVector::GetData<string_t>(other.values_insert_order);
		auto this_vector_ptr = FlatVector::GetData<string_t>(values_insert_order);

		// Now we must check if all strings are the same
		for (idx_t i = 0; i < dict_size; i++) {
			if (!Equals::Operation(other_vector_ptr[i], this_vector_ptr[i])) {
				return false;
			}
		}
		return true;
	}

	void Serialize(FieldWriter &writer) const override {
		if (dict_type != EnumDictType::VECTOR_DICT) {
			throw InternalException("Cannot serialize non-vector dictionary ENUM types");
		}
		writer.WriteField<uint32_t>(dict_size);
		writer.WriteString(enum_name);
		((Vector &)values_insert_order).Serialize(dict_size, writer.GetSerializer());
	}
};

template <class T>
struct EnumTypeInfoTemplated : public EnumTypeInfo {
	explicit EnumTypeInfoTemplated(const string &enum_name_p, Vector &values_insert_order_p, idx_t size_p)
	    : EnumTypeInfo(enum_name_p, values_insert_order_p, size_p) {
		D_ASSERT(values_insert_order_p.GetType().InternalType() == PhysicalType::VARCHAR);

		UnifiedVectorFormat vdata;
		values_insert_order.ToUnifiedFormat(size_p, vdata);

		auto data = (string_t *)vdata.data;
		for (idx_t i = 0; i < size_p; i++) {
			auto idx = vdata.sel->get_index(i);
			if (!vdata.validity.RowIsValid(idx)) {
				throw InternalException("Attempted to create ENUM type with NULL value");
			}
			if (values.count(data[idx]) > 0) {
				throw InvalidInputException("Attempted to create ENUM type with duplicate value %s",
				                            data[idx].GetString());
			}
			values[data[idx]] = i;
		}
	}

	static shared_ptr<EnumTypeInfoTemplated> Deserialize(FieldReader &reader, uint32_t size) {
		auto enum_name = reader.ReadRequired<string>();
		Vector values_insert_order(LogicalType::VARCHAR, size);
		values_insert_order.Deserialize(size, reader.GetSource());
		return make_shared<EnumTypeInfoTemplated>(std::move(enum_name), values_insert_order, size);
	}

	string_map_t<T> values;
};

const string &EnumType::GetTypeName(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::ENUM);
	auto info = type.AuxInfo();
	D_ASSERT(info);
	return ((EnumTypeInfo &)*info).enum_name;
}

static PhysicalType EnumVectorDictType(idx_t size) {
	if (size <= NumericLimits<uint8_t>::Maximum()) {
		return PhysicalType::UINT8;
	} else if (size <= NumericLimits<uint16_t>::Maximum()) {
		return PhysicalType::UINT16;
	} else if (size <= NumericLimits<uint32_t>::Maximum()) {
		return PhysicalType::UINT32;
	} else {
		throw InternalException("Enum size must be lower than " + std::to_string(NumericLimits<uint32_t>::Maximum()));
	}
}

LogicalType LogicalType::ENUM(const string &enum_name, Vector &ordered_data, idx_t size) {
	// Generate EnumTypeInfo
	shared_ptr<ExtraTypeInfo> info;
	auto enum_internal_type = EnumVectorDictType(size);
	switch (enum_internal_type) {
	case PhysicalType::UINT8:
		info = make_shared<EnumTypeInfoTemplated<uint8_t>>(enum_name, ordered_data, size);
		break;
	case PhysicalType::UINT16:
		info = make_shared<EnumTypeInfoTemplated<uint16_t>>(enum_name, ordered_data, size);
		break;
	case PhysicalType::UINT32:
		info = make_shared<EnumTypeInfoTemplated<uint32_t>>(enum_name, ordered_data, size);
		break;
	default:
		throw InternalException("Invalid Physical Type for ENUMs");
	}
	// Generate Actual Enum Type
	return LogicalType(LogicalTypeId::ENUM, info);
}

template <class T>
int64_t TemplatedGetPos(string_map_t<T> &map, const string_t &key) {
	auto it = map.find(key);
	if (it == map.end()) {
		return -1;
	}
	return it->second;
}

int64_t EnumType::GetPos(const LogicalType &type, const string_t &key) {
	auto info = type.AuxInfo();
	switch (type.InternalType()) {
	case PhysicalType::UINT8:
		return TemplatedGetPos(((EnumTypeInfoTemplated<uint8_t> &)*info).values, key);
	case PhysicalType::UINT16:
		return TemplatedGetPos(((EnumTypeInfoTemplated<uint16_t> &)*info).values, key);
	case PhysicalType::UINT32:
		return TemplatedGetPos(((EnumTypeInfoTemplated<uint32_t> &)*info).values, key);
	default:
		throw InternalException("ENUM can only have unsigned integers (except UINT64) as physical types");
	}
}

const string EnumType::GetValue(const Value &val) {
	auto info = val.type().AuxInfo();
	auto &values_insert_order = ((EnumTypeInfo &)*info).values_insert_order;
	return StringValue::Get(values_insert_order.GetValue(val.GetValue<uint32_t>()));
}

Vector &EnumType::GetValuesInsertOrder(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::ENUM);
	auto info = type.AuxInfo();
	D_ASSERT(info);
	return ((EnumTypeInfo &)*info).values_insert_order;
}

idx_t EnumType::GetSize(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::ENUM);
	auto info = type.AuxInfo();
	D_ASSERT(info);
	return ((EnumTypeInfo &)*info).dict_size;
}

void EnumType::SetCatalog(LogicalType &type, TypeCatalogEntry *catalog_entry) {
	D_ASSERT(type.id() == LogicalTypeId::ENUM);
	auto info = type.AuxInfo();
	D_ASSERT(info);
	((EnumTypeInfo &)*info).catalog_entry = catalog_entry;
}
TypeCatalogEntry *EnumType::GetCatalog(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::ENUM);
	auto info = type.AuxInfo();
	D_ASSERT(info);
	return ((EnumTypeInfo &)*info).catalog_entry;
}

PhysicalType EnumType::GetPhysicalType(const LogicalType &type) {
	D_ASSERT(type.id() == LogicalTypeId::ENUM);
	auto aux_info = type.AuxInfo();
	D_ASSERT(aux_info);
	auto &info = (EnumTypeInfo &)*aux_info;
	D_ASSERT(info.dict_type == EnumDictType::VECTOR_DICT);
	return EnumVectorDictType(info.dict_size);
}

//===--------------------------------------------------------------------===//
// Extra Type Info
//===--------------------------------------------------------------------===//
void ExtraTypeInfo::Serialize(ExtraTypeInfo *info, FieldWriter &writer) {
	if (!info) {
		writer.WriteField<ExtraTypeInfoType>(ExtraTypeInfoType::INVALID_TYPE_INFO);
		writer.WriteString(string());
	} else {
		writer.WriteField<ExtraTypeInfoType>(info->type);
		info->Serialize(writer);
		writer.WriteString(info->alias);
	}
}
shared_ptr<ExtraTypeInfo> ExtraTypeInfo::Deserialize(FieldReader &reader) {
	auto type = reader.ReadRequired<ExtraTypeInfoType>();
	shared_ptr<ExtraTypeInfo> extra_info;
	switch (type) {
	case ExtraTypeInfoType::INVALID_TYPE_INFO: {
		auto alias = reader.ReadField<string>(string());
		if (!alias.empty()) {
			return make_shared<ExtraTypeInfo>(type, alias);
		}
		return nullptr;
	}
	case ExtraTypeInfoType::GENERIC_TYPE_INFO: {
		extra_info = make_shared<ExtraTypeInfo>(type);
	} break;
	case ExtraTypeInfoType::DECIMAL_TYPE_INFO:
		extra_info = DecimalTypeInfo::Deserialize(reader);
		break;
	case ExtraTypeInfoType::STRING_TYPE_INFO:
		extra_info = StringTypeInfo::Deserialize(reader);
		break;
	case ExtraTypeInfoType::LIST_TYPE_INFO:
		extra_info = ListTypeInfo::Deserialize(reader);
		break;
	case ExtraTypeInfoType::STRUCT_TYPE_INFO:
		extra_info = StructTypeInfo::Deserialize(reader);
		break;
	case ExtraTypeInfoType::USER_TYPE_INFO:
		extra_info = UserTypeInfo::Deserialize(reader);
		break;
	case ExtraTypeInfoType::ENUM_TYPE_INFO: {
		auto enum_size = reader.ReadRequired<uint32_t>();
		auto enum_internal_type = EnumVectorDictType(enum_size);
		switch (enum_internal_type) {
		case PhysicalType::UINT8:
			extra_info = EnumTypeInfoTemplated<uint8_t>::Deserialize(reader, enum_size);
			break;
		case PhysicalType::UINT16:
			extra_info = EnumTypeInfoTemplated<uint16_t>::Deserialize(reader, enum_size);
			break;
		case PhysicalType::UINT32:
			extra_info = EnumTypeInfoTemplated<uint32_t>::Deserialize(reader, enum_size);
			break;
		default:
			throw InternalException("Invalid Physical Type for ENUMs");
		}
	} break;
	case ExtraTypeInfoType::AGGREGATE_STATE_TYPE_INFO:
		extra_info = AggregateStateTypeInfo::Deserialize(reader);
		break;

	default:
		throw InternalException("Unimplemented type info in ExtraTypeInfo::Deserialize");
	}
	auto alias = reader.ReadField<string>(string());
	extra_info->alias = alias;
	return extra_info;
}

//===--------------------------------------------------------------------===//
// Logical Type
//===--------------------------------------------------------------------===//

// the destructor needs to know about the extra type info
LogicalType::~LogicalType() {
}

void LogicalType::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteField<LogicalTypeId>(id_);
	ExtraTypeInfo::Serialize(type_info_.get(), writer);
	writer.Finalize();
}

LogicalType LogicalType::Deserialize(Deserializer &source) {
	FieldReader reader(source);
	auto id = reader.ReadRequired<LogicalTypeId>();
	auto info = ExtraTypeInfo::Deserialize(reader);
	reader.Finalize();

	return LogicalType(id, std::move(info));
}

bool LogicalType::EqualTypeInfo(const LogicalType &rhs) const {
	if (type_info_.get() == rhs.type_info_.get()) {
		return true;
	}
	if (type_info_) {
		return type_info_->Equals(rhs.type_info_.get());
	} else {
		D_ASSERT(rhs.type_info_);
		return rhs.type_info_->Equals(type_info_.get());
	}
}

bool LogicalType::operator==(const LogicalType &rhs) const {
	if (id_ != rhs.id_) {
		return false;
	}
	return EqualTypeInfo(rhs);
}

} // namespace duckdb





namespace duckdb {

//===--------------------------------------------------------------------===//
// Comparison Operations
//===--------------------------------------------------------------------===//

struct ValuePositionComparator {
	// Return true if the positional Values definitely match.
	// Default to the same as the final value
	template <typename OP>
	static inline bool Definite(const Value &lhs, const Value &rhs) {
		return Final<OP>(lhs, rhs);
	}

	// Select the positional Values that need further testing.
	// Usually this means Is Not Distinct, as those are the semantics used by Postges
	template <typename OP>
	static inline bool Possible(const Value &lhs, const Value &rhs) {
		return ValueOperations::NotDistinctFrom(lhs, rhs);
	}

	// Return true if the positional Values definitely match in the final position
	// This needs to be specialised.
	template <typename OP>
	static inline bool Final(const Value &lhs, const Value &rhs) {
		return false;
	}

	// Tie-break based on length when one of the sides has been exhausted, returning true if the LHS matches.
	// This essentially means that the existing positions compare equal.
	// Default to the same semantics as the OP for idx_t. This works in most cases.
	template <typename OP>
	static inline bool TieBreak(const idx_t lpos, const idx_t rpos) {
		return OP::Operation(lpos, rpos);
	}
};

// Equals must always check every column
template <>
inline bool ValuePositionComparator::Definite<duckdb::Equals>(const Value &lhs, const Value &rhs) {
	return false;
}

template <>
inline bool ValuePositionComparator::Final<duckdb::Equals>(const Value &lhs, const Value &rhs) {
	return ValueOperations::NotDistinctFrom(lhs, rhs);
}

// NotEquals must check everything that matched
template <>
inline bool ValuePositionComparator::Possible<duckdb::NotEquals>(const Value &lhs, const Value &rhs) {
	return true;
}

template <>
inline bool ValuePositionComparator::Final<duckdb::NotEquals>(const Value &lhs, const Value &rhs) {
	return ValueOperations::NotDistinctFrom(lhs, rhs);
}

// Non-strict inequalities must use strict comparisons for Definite
template <>
bool ValuePositionComparator::Definite<duckdb::LessThanEquals>(const Value &lhs, const Value &rhs) {
	return ValueOperations::DistinctLessThan(lhs, rhs);
}

template <>
bool ValuePositionComparator::Final<duckdb::LessThanEquals>(const Value &lhs, const Value &rhs) {
	return ValueOperations::DistinctLessThanEquals(lhs, rhs);
}

template <>
bool ValuePositionComparator::Definite<duckdb::GreaterThanEquals>(const Value &lhs, const Value &rhs) {
	return ValueOperations::DistinctGreaterThan(lhs, rhs);
}

template <>
bool ValuePositionComparator::Final<duckdb::GreaterThanEquals>(const Value &lhs, const Value &rhs) {
	return ValueOperations::DistinctGreaterThanEquals(lhs, rhs);
}

// Strict inequalities just use strict for both Definite and Final
template <>
bool ValuePositionComparator::Final<duckdb::LessThan>(const Value &lhs, const Value &rhs) {
	return ValueOperations::DistinctLessThan(lhs, rhs);
}

template <>
bool ValuePositionComparator::Final<duckdb::GreaterThan>(const Value &lhs, const Value &rhs) {
	return ValueOperations::DistinctGreaterThan(lhs, rhs);
}

template <class OP>
static bool TemplatedBooleanOperation(const Value &left, const Value &right) {
	const auto &left_type = left.type();
	const auto &right_type = right.type();
	if (left_type != right_type) {
		Value left_copy = left;
		Value right_copy = right;

		LogicalType comparison_type = BoundComparisonExpression::BindComparison(left_type, right_type);
		if (!left_copy.DefaultTryCastAs(comparison_type) || !right_copy.DefaultTryCastAs(comparison_type)) {
			return false;
		}
		D_ASSERT(left_copy.type() == right_copy.type());
		return TemplatedBooleanOperation<OP>(left_copy, right_copy);
	}
	switch (left_type.InternalType()) {
	case PhysicalType::BOOL:
		return OP::Operation(left.GetValueUnsafe<bool>(), right.GetValueUnsafe<bool>());
	case PhysicalType::INT8:
		return OP::Operation(left.GetValueUnsafe<int8_t>(), right.GetValueUnsafe<int8_t>());
	case PhysicalType::INT16:
		return OP::Operation(left.GetValueUnsafe<int16_t>(), right.GetValueUnsafe<int16_t>());
	case PhysicalType::INT32:
		return OP::Operation(left.GetValueUnsafe<int32_t>(), right.GetValueUnsafe<int32_t>());
	case PhysicalType::INT64:
		return OP::Operation(left.GetValueUnsafe<int64_t>(), right.GetValueUnsafe<int64_t>());
	case PhysicalType::UINT8:
		return OP::Operation(left.GetValueUnsafe<uint8_t>(), right.GetValueUnsafe<uint8_t>());
	case PhysicalType::UINT16:
		return OP::Operation(left.GetValueUnsafe<uint16_t>(), right.GetValueUnsafe<uint16_t>());
	case PhysicalType::UINT32:
		return OP::Operation(left.GetValueUnsafe<uint32_t>(), right.GetValueUnsafe<uint32_t>());
	case PhysicalType::UINT64:
		return OP::Operation(left.GetValueUnsafe<uint64_t>(), right.GetValueUnsafe<uint64_t>());
	case PhysicalType::INT128:
		return OP::Operation(left.GetValueUnsafe<hugeint_t>(), right.GetValueUnsafe<hugeint_t>());
	case PhysicalType::FLOAT:
		return OP::Operation(left.GetValueUnsafe<float>(), right.GetValueUnsafe<float>());
	case PhysicalType::DOUBLE:
		return OP::Operation(left.GetValueUnsafe<double>(), right.GetValueUnsafe<double>());
	case PhysicalType::INTERVAL:
		return OP::Operation(left.GetValueUnsafe<interval_t>(), right.GetValueUnsafe<interval_t>());
	case PhysicalType::VARCHAR:
		return OP::Operation(StringValue::Get(left), StringValue::Get(right));
	case PhysicalType::STRUCT: {
		auto &left_children = StructValue::GetChildren(left);
		auto &right_children = StructValue::GetChildren(right);
		// this should be enforced by the type
		D_ASSERT(left_children.size() == right_children.size());
		idx_t i = 0;
		for (; i < left_children.size() - 1; ++i) {
			if (ValuePositionComparator::Definite<OP>(left_children[i], right_children[i])) {
				return true;
			}
			if (!ValuePositionComparator::Possible<OP>(left_children[i], right_children[i])) {
				return false;
			}
		}
		return ValuePositionComparator::Final<OP>(left_children[i], right_children[i]);
	}
	case PhysicalType::LIST: {
		auto &left_children = ListValue::GetChildren(left);
		auto &right_children = ListValue::GetChildren(right);
		for (idx_t pos = 0;; ++pos) {
			if (pos == left_children.size() || pos == right_children.size()) {
				return ValuePositionComparator::TieBreak<OP>(left_children.size(), right_children.size());
			}
			if (ValuePositionComparator::Definite<OP>(left_children[pos], right_children[pos])) {
				return true;
			}
			if (!ValuePositionComparator::Possible<OP>(left_children[pos], right_children[pos])) {
				return false;
			}
		}
		return false;
	}
	default:
		throw InternalException("Unimplemented type for value comparison");
	}
}

bool ValueOperations::Equals(const Value &left, const Value &right) {
	if (left.IsNull() || right.IsNull()) {
		throw InternalException("Comparison on NULL values");
	}
	return TemplatedBooleanOperation<duckdb::Equals>(left, right);
}

bool ValueOperations::NotEquals(const Value &left, const Value &right) {
	return !ValueOperations::Equals(left, right);
}

bool ValueOperations::GreaterThan(const Value &left, const Value &right) {
	if (left.IsNull() || right.IsNull()) {
		throw InternalException("Comparison on NULL values");
	}
	return TemplatedBooleanOperation<duckdb::GreaterThan>(left, right);
}

bool ValueOperations::GreaterThanEquals(const Value &left, const Value &right) {
	if (left.IsNull() || right.IsNull()) {
		throw InternalException("Comparison on NULL values");
	}
	return TemplatedBooleanOperation<duckdb::GreaterThanEquals>(left, right);
}

bool ValueOperations::LessThan(const Value &left, const Value &right) {
	return ValueOperations::GreaterThan(right, left);
}

bool ValueOperations::LessThanEquals(const Value &left, const Value &right) {
	return ValueOperations::GreaterThanEquals(right, left);
}

bool ValueOperations::NotDistinctFrom(const Value &left, const Value &right) {
	if (left.IsNull() && right.IsNull()) {
		return true;
	}
	if (left.IsNull() != right.IsNull()) {
		return false;
	}
	return TemplatedBooleanOperation<duckdb::Equals>(left, right);
}

bool ValueOperations::DistinctFrom(const Value &left, const Value &right) {
	return !ValueOperations::NotDistinctFrom(left, right);
}

bool ValueOperations::DistinctGreaterThan(const Value &left, const Value &right) {
	if (left.IsNull() && right.IsNull()) {
		return false;
	} else if (right.IsNull()) {
		return false;
	} else if (left.IsNull()) {
		return true;
	}
	return TemplatedBooleanOperation<duckdb::GreaterThan>(left, right);
}

bool ValueOperations::DistinctGreaterThanEquals(const Value &left, const Value &right) {
	if (left.IsNull()) {
		return true;
	} else if (right.IsNull()) {
		return false;
	}
	return TemplatedBooleanOperation<duckdb::GreaterThanEquals>(left, right);
}

bool ValueOperations::DistinctLessThan(const Value &left, const Value &right) {
	return ValueOperations::DistinctGreaterThan(right, left);
}

bool ValueOperations::DistinctLessThanEquals(const Value &left, const Value &right) {
	return ValueOperations::DistinctGreaterThanEquals(right, left);
}

} // namespace duckdb
//===--------------------------------------------------------------------===//
// boolean_operators.cpp
// Description: This file contains the implementation of the boolean
// operations AND OR !
//===--------------------------------------------------------------------===//





namespace duckdb {

//===--------------------------------------------------------------------===//
// AND/OR
//===--------------------------------------------------------------------===//
template <class OP>
static void TemplatedBooleanNullmask(Vector &left, Vector &right, Vector &result, idx_t count) {
	D_ASSERT(left.GetType().id() == LogicalTypeId::BOOLEAN && right.GetType().id() == LogicalTypeId::BOOLEAN &&
	         result.GetType().id() == LogicalTypeId::BOOLEAN);

	if (left.GetVectorType() == VectorType::CONSTANT_VECTOR && right.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		// operation on two constants, result is constant vector
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		auto ldata = ConstantVector::GetData<uint8_t>(left);
		auto rdata = ConstantVector::GetData<uint8_t>(right);
		auto result_data = ConstantVector::GetData<bool>(result);

		bool is_null = OP::Operation(*ldata > 0, *rdata > 0, ConstantVector::IsNull(left),
		                             ConstantVector::IsNull(right), *result_data);
		ConstantVector::SetNull(result, is_null);
	} else {
		// perform generic loop
		UnifiedVectorFormat ldata, rdata;
		left.ToUnifiedFormat(count, ldata);
		right.ToUnifiedFormat(count, rdata);

		result.SetVectorType(VectorType::FLAT_VECTOR);
		auto left_data = (uint8_t *)ldata.data; // we use uint8 to avoid load of gunk bools
		auto right_data = (uint8_t *)rdata.data;
		auto result_data = FlatVector::GetData<bool>(result);
		auto &result_mask = FlatVector::Validity(result);
		if (!ldata.validity.AllValid() || !rdata.validity.AllValid()) {
			for (idx_t i = 0; i < count; i++) {
				auto lidx = ldata.sel->get_index(i);
				auto ridx = rdata.sel->get_index(i);
				bool is_null =
				    OP::Operation(left_data[lidx] > 0, right_data[ridx] > 0, !ldata.validity.RowIsValid(lidx),
				                  !rdata.validity.RowIsValid(ridx), result_data[i]);
				result_mask.Set(i, !is_null);
			}
		} else {
			for (idx_t i = 0; i < count; i++) {
				auto lidx = ldata.sel->get_index(i);
				auto ridx = rdata.sel->get_index(i);
				result_data[i] = OP::SimpleOperation(left_data[lidx], right_data[ridx]);
			}
		}
	}
}

/*
SQL AND Rules:

TRUE  AND TRUE   = TRUE
TRUE  AND FALSE  = FALSE
TRUE  AND NULL   = NULL
FALSE AND TRUE   = FALSE
FALSE AND FALSE  = FALSE
FALSE AND NULL   = FALSE
NULL  AND TRUE   = NULL
NULL  AND FALSE  = FALSE
NULL  AND NULL   = NULL

Basically:
- Only true if both are true
- False if either is false (regardless of NULLs)
- NULL otherwise
*/
struct TernaryAnd {
	static bool SimpleOperation(bool left, bool right) {
		return left && right;
	}
	static bool Operation(bool left, bool right, bool left_null, bool right_null, bool &result) {
		if (left_null && right_null) {
			// both NULL:
			// result is NULL
			return true;
		} else if (left_null) {
			// left is NULL:
			// result is FALSE if right is false
			// result is NULL if right is true
			result = right;
			return right;
		} else if (right_null) {
			// right is NULL:
			// result is FALSE if left is false
			// result is NULL if left is true
			result = left;
			return left;
		} else {
			// no NULL: perform the AND
			result = left && right;
			return false;
		}
	}
};

void VectorOperations::And(Vector &left, Vector &right, Vector &result, idx_t count) {
	TemplatedBooleanNullmask<TernaryAnd>(left, right, result, count);
}

/*
SQL OR Rules:

OR
TRUE  OR TRUE  = TRUE
TRUE  OR FALSE = TRUE
TRUE  OR NULL  = TRUE
FALSE OR TRUE  = TRUE
FALSE OR FALSE = FALSE
FALSE OR NULL  = NULL
NULL  OR TRUE  = TRUE
NULL  OR FALSE = NULL
NULL  OR NULL  = NULL

Basically:
- Only false if both are false
- True if either is true (regardless of NULLs)
- NULL otherwise
*/

struct TernaryOr {
	static bool SimpleOperation(bool left, bool right) {
		return left || right;
	}
	static bool Operation(bool left, bool right, bool left_null, bool right_null, bool &result) {
		if (left_null && right_null) {
			// both NULL:
			// result is NULL
			return true;
		} else if (left_null) {
			// left is NULL:
			// result is TRUE if right is true
			// result is NULL if right is false
			result = right;
			return !right;
		} else if (right_null) {
			// right is NULL:
			// result is TRUE if left is true
			// result is NULL if left is false
			result = left;
			return !left;
		} else {
			// no NULL: perform the OR
			result = left || right;
			return false;
		}
	}
};

void VectorOperations::Or(Vector &left, Vector &right, Vector &result, idx_t count) {
	TemplatedBooleanNullmask<TernaryOr>(left, right, result, count);
}

struct NotOperator {
	template <class TA, class TR>
	static inline TR Operation(TA left) {
		return !left;
	}
};

void VectorOperations::Not(Vector &input, Vector &result, idx_t count) {
	D_ASSERT(input.GetType() == LogicalType::BOOLEAN && result.GetType() == LogicalType::BOOLEAN);
	UnaryExecutor::Execute<bool, bool, NotOperator>(input, result, count);
}

} // namespace duckdb
//===--------------------------------------------------------------------===//
// comparison_operators.cpp
// Description: This file contains the implementation of the comparison
// operations == != >= <= > <
//===--------------------------------------------------------------------===//








namespace duckdb {

template <class T>
bool EqualsFloat(T left, T right) {
	if (DUCKDB_UNLIKELY(Value::IsNan(left) && Value::IsNan(right))) {
		return true;
	}
	return left == right;
}

template <>
bool Equals::Operation(float left, float right) {
	return EqualsFloat<float>(left, right);
}

template <>
bool Equals::Operation(double left, double right) {
	return EqualsFloat<double>(left, right);
}

template <class T>
bool GreaterThanFloat(T left, T right) {
	// handle nans
	// nan is always bigger than everything else
	bool left_is_nan = Value::IsNan(left);
	bool right_is_nan = Value::IsNan(right);
	// if right is nan, there is no number that is bigger than right
	if (DUCKDB_UNLIKELY(right_is_nan)) {
		return false;
	}
	// if left is nan, but right is not, left is always bigger
	if (DUCKDB_UNLIKELY(left_is_nan)) {
		return true;
	}
	return left > right;
}

template <>
bool GreaterThan::Operation(float left, float right) {
	return GreaterThanFloat<float>(left, right);
}

template <>
bool GreaterThan::Operation(double left, double right) {
	return GreaterThanFloat<double>(left, right);
}

template <class T>
bool GreaterThanEqualsFloat(T left, T right) {
	// handle nans
	// nan is always bigger than everything else
	bool left_is_nan = Value::IsNan(left);
	bool right_is_nan = Value::IsNan(right);
	// if right is nan, there is no bigger number
	// we only return true if left is also nan (in which case the numbers are equal)
	if (DUCKDB_UNLIKELY(right_is_nan)) {
		return left_is_nan;
	}
	// if left is nan, but right is not, left is always bigger
	if (DUCKDB_UNLIKELY(left_is_nan)) {
		return true;
	}
	return left >= right;
}

template <>
bool GreaterThanEquals::Operation(float left, float right) {
	return GreaterThanEqualsFloat<float>(left, right);
}

template <>
bool GreaterThanEquals::Operation(double left, double right) {
	return GreaterThanEqualsFloat<double>(left, right);
}

struct ComparisonSelector {
	template <typename OP>
	static idx_t Select(Vector &left, Vector &right, const SelectionVector *sel, idx_t count, SelectionVector *true_sel,
	                    SelectionVector *false_sel) {
		throw NotImplementedException("Unknown comparison operation!");
	}
};

template <>
inline idx_t ComparisonSelector::Select<duckdb::Equals>(Vector &left, Vector &right, const SelectionVector *sel,
                                                        idx_t count, SelectionVector *true_sel,
                                                        SelectionVector *false_sel) {
	return VectorOperations::Equals(left, right, sel, count, true_sel, false_sel);
}

template <>
inline idx_t ComparisonSelector::Select<duckdb::NotEquals>(Vector &left, Vector &right, const SelectionVector *sel,
                                                           idx_t count, SelectionVector *true_sel,
                                                           SelectionVector *false_sel) {
	return VectorOperations::NotEquals(left, right, sel, count, true_sel, false_sel);
}

template <>
inline idx_t ComparisonSelector::Select<duckdb::GreaterThan>(Vector &left, Vector &right, const SelectionVector *sel,
                                                             idx_t count, SelectionVector *true_sel,
                                                             SelectionVector *false_sel) {
	return VectorOperations::GreaterThan(left, right, sel, count, true_sel, false_sel);
}

template <>
inline idx_t ComparisonSelector::Select<duckdb::GreaterThanEquals>(Vector &left, Vector &right,
                                                                   const SelectionVector *sel, idx_t count,
                                                                   SelectionVector *true_sel,
                                                                   SelectionVector *false_sel) {
	return VectorOperations::GreaterThanEquals(left, right, sel, count, true_sel, false_sel);
}

template <>
inline idx_t ComparisonSelector::Select<duckdb::LessThan>(Vector &left, Vector &right, const SelectionVector *sel,
                                                          idx_t count, SelectionVector *true_sel,
                                                          SelectionVector *false_sel) {
	return VectorOperations::LessThan(left, right, sel, count, true_sel, false_sel);
}

template <>
inline idx_t ComparisonSelector::Select<duckdb::LessThanEquals>(Vector &left, Vector &right, const SelectionVector *sel,
                                                                idx_t count, SelectionVector *true_sel,
                                                                SelectionVector *false_sel) {
	return VectorOperations::LessThanEquals(left, right, sel, count, true_sel, false_sel);
}

static void ComparesNotNull(UnifiedVectorFormat &ldata, UnifiedVectorFormat &rdata, ValidityMask &vresult,
                            idx_t count) {
	for (idx_t i = 0; i < count; ++i) {
		auto lidx = ldata.sel->get_index(i);
		auto ridx = rdata.sel->get_index(i);
		if (!ldata.validity.RowIsValid(lidx) || !rdata.validity.RowIsValid(ridx)) {
			vresult.SetInvalid(i);
		}
	}
}

template <typename OP>
static void NestedComparisonExecutor(Vector &left, Vector &right, Vector &result, idx_t count) {
	const auto left_constant = left.GetVectorType() == VectorType::CONSTANT_VECTOR;
	const auto right_constant = right.GetVectorType() == VectorType::CONSTANT_VECTOR;

	if ((left_constant && ConstantVector::IsNull(left)) || (right_constant && ConstantVector::IsNull(right))) {
		// either left or right is constant NULL: result is constant NULL
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		ConstantVector::SetNull(result, true);
		return;
	}

	if (left_constant && right_constant) {
		// both sides are constant, and neither is NULL so just compare one element.
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		SelectionVector true_sel(1);
		auto match_count = ComparisonSelector::Select<OP>(left, right, nullptr, 1, &true_sel, nullptr);
		auto result_data = ConstantVector::GetData<bool>(result);
		result_data[0] = match_count > 0;
		return;
	}

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<bool>(result);
	auto &result_validity = FlatVector::Validity(result);

	UnifiedVectorFormat leftv, rightv;
	left.ToUnifiedFormat(count, leftv);
	right.ToUnifiedFormat(count, rightv);
	if (!leftv.validity.AllValid() || !rightv.validity.AllValid()) {
		ComparesNotNull(leftv, rightv, result_validity, count);
	}
	SelectionVector true_sel(count);
	SelectionVector false_sel(count);
	idx_t match_count = ComparisonSelector::Select<OP>(left, right, nullptr, count, &true_sel, &false_sel);

	for (idx_t i = 0; i < match_count; ++i) {
		const auto idx = true_sel.get_index(i);
		result_data[idx] = true;
	}

	const idx_t no_match_count = count - match_count;
	for (idx_t i = 0; i < no_match_count; ++i) {
		const auto idx = false_sel.get_index(i);
		result_data[idx] = false;
	}
}

struct ComparisonExecutor {
private:
	template <class T, class OP>
	static inline void TemplatedExecute(Vector &left, Vector &right, Vector &result, idx_t count) {
		BinaryExecutor::Execute<T, T, bool, OP>(left, right, result, count);
	}

public:
	template <class OP>
	static inline void Execute(Vector &left, Vector &right, Vector &result, idx_t count) {
		D_ASSERT(left.GetType() == right.GetType() && result.GetType() == LogicalType::BOOLEAN);
		// the inplace loops take the result as the last parameter
		switch (left.GetType().InternalType()) {
		case PhysicalType::BOOL:
		case PhysicalType::INT8:
			TemplatedExecute<int8_t, OP>(left, right, result, count);
			break;
		case PhysicalType::INT16:
			TemplatedExecute<int16_t, OP>(left, right, result, count);
			break;
		case PhysicalType::INT32:
			TemplatedExecute<int32_t, OP>(left, right, result, count);
			break;
		case PhysicalType::INT64:
			TemplatedExecute<int64_t, OP>(left, right, result, count);
			break;
		case PhysicalType::UINT8:
			TemplatedExecute<uint8_t, OP>(left, right, result, count);
			break;
		case PhysicalType::UINT16:
			TemplatedExecute<uint16_t, OP>(left, right, result, count);
			break;
		case PhysicalType::UINT32:
			TemplatedExecute<uint32_t, OP>(left, right, result, count);
			break;
		case PhysicalType::UINT64:
			TemplatedExecute<uint64_t, OP>(left, right, result, count);
			break;
		case PhysicalType::INT128:
			TemplatedExecute<hugeint_t, OP>(left, right, result, count);
			break;
		case PhysicalType::FLOAT:
			TemplatedExecute<float, OP>(left, right, result, count);
			break;
		case PhysicalType::DOUBLE:
			TemplatedExecute<double, OP>(left, right, result, count);
			break;
		case PhysicalType::INTERVAL:
			TemplatedExecute<interval_t, OP>(left, right, result, count);
			break;
		case PhysicalType::VARCHAR:
			TemplatedExecute<string_t, OP>(left, right, result, count);
			break;
		case PhysicalType::LIST:
		case PhysicalType::STRUCT:
			NestedComparisonExecutor<OP>(left, right, result, count);
			break;
		default:
			throw InternalException("Invalid type for comparison");
		}
	}
};

void VectorOperations::Equals(Vector &left, Vector &right, Vector &result, idx_t count) {
	ComparisonExecutor::Execute<duckdb::Equals>(left, right, result, count);
}

void VectorOperations::NotEquals(Vector &left, Vector &right, Vector &result, idx_t count) {
	ComparisonExecutor::Execute<duckdb::NotEquals>(left, right, result, count);
}

void VectorOperations::GreaterThanEquals(Vector &left, Vector &right, Vector &result, idx_t count) {
	ComparisonExecutor::Execute<duckdb::GreaterThanEquals>(left, right, result, count);
}

void VectorOperations::LessThanEquals(Vector &left, Vector &right, Vector &result, idx_t count) {
	ComparisonExecutor::Execute<duckdb::LessThanEquals>(left, right, result, count);
}

void VectorOperations::GreaterThan(Vector &left, Vector &right, Vector &result, idx_t count) {
	ComparisonExecutor::Execute<duckdb::GreaterThan>(left, right, result, count);
}

void VectorOperations::LessThan(Vector &left, Vector &right, Vector &result, idx_t count) {
	ComparisonExecutor::Execute<duckdb::LessThan>(left, right, result, count);
}

} // namespace duckdb
//===--------------------------------------------------------------------===//
// generators.cpp
// Description: This file contains the implementation of different generators
//===--------------------------------------------------------------------===//





namespace duckdb {

template <class T>
void TemplatedGenerateSequence(Vector &result, idx_t count, int64_t start, int64_t increment) {
	D_ASSERT(result.GetType().IsNumeric());
	if (start > NumericLimits<T>::Maximum() || increment > NumericLimits<T>::Maximum()) {
		throw Exception("Sequence start or increment out of type range");
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<T>(result);
	auto value = (T)start;
	for (idx_t i = 0; i < count; i++) {
		if (i > 0) {
			value += increment;
		}
		result_data[i] = value;
	}
}

void VectorOperations::GenerateSequence(Vector &result, idx_t count, int64_t start, int64_t increment) {
	if (!result.GetType().IsNumeric()) {
		throw InvalidTypeException(result.GetType(), "Can only generate sequences for numeric values!");
	}
	switch (result.GetType().InternalType()) {
	case PhysicalType::INT8:
		TemplatedGenerateSequence<int8_t>(result, count, start, increment);
		break;
	case PhysicalType::INT16:
		TemplatedGenerateSequence<int16_t>(result, count, start, increment);
		break;
	case PhysicalType::INT32:
		TemplatedGenerateSequence<int32_t>(result, count, start, increment);
		break;
	case PhysicalType::INT64:
		TemplatedGenerateSequence<int64_t>(result, count, start, increment);
		break;
	case PhysicalType::FLOAT:
		TemplatedGenerateSequence<float>(result, count, start, increment);
		break;
	case PhysicalType::DOUBLE:
		TemplatedGenerateSequence<double>(result, count, start, increment);
		break;
	default:
		throw NotImplementedException("Unimplemented type for generate sequence");
	}
}

template <class T>
void TemplatedGenerateSequence(Vector &result, idx_t count, const SelectionVector &sel, int64_t start,
                               int64_t increment) {
	D_ASSERT(result.GetType().IsNumeric());
	if (start > NumericLimits<T>::Maximum() || increment > NumericLimits<T>::Maximum()) {
		throw Exception("Sequence start or increment out of type range");
	}
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<T>(result);
	auto value = (T)start;
	for (idx_t i = 0; i < count; i++) {
		auto idx = sel.get_index(i);
		result_data[idx] = value + increment * idx;
	}
}

void VectorOperations::GenerateSequence(Vector &result, idx_t count, const SelectionVector &sel, int64_t start,
                                        int64_t increment) {
	if (!result.GetType().IsNumeric()) {
		throw InvalidTypeException(result.GetType(), "Can only generate sequences for numeric values!");
	}
	switch (result.GetType().InternalType()) {
	case PhysicalType::INT8:
		TemplatedGenerateSequence<int8_t>(result, count, sel, start, increment);
		break;
	case PhysicalType::INT16:
		TemplatedGenerateSequence<int16_t>(result, count, sel, start, increment);
		break;
	case PhysicalType::INT32:
		TemplatedGenerateSequence<int32_t>(result, count, sel, start, increment);
		break;
	case PhysicalType::INT64:
		TemplatedGenerateSequence<int64_t>(result, count, sel, start, increment);
		break;
	case PhysicalType::FLOAT:
		TemplatedGenerateSequence<float>(result, count, sel, start, increment);
		break;
	case PhysicalType::DOUBLE:
		TemplatedGenerateSequence<double>(result, count, sel, start, increment);
		break;
	default:
		throw NotImplementedException("Unimplemented type for generate sequence");
	}
}

} // namespace duckdb



namespace duckdb {

struct DistinctBinaryLambdaWrapper {
	template <class OP, class LEFT_TYPE, class RIGHT_TYPE, class RESULT_TYPE>
	static inline RESULT_TYPE Operation(LEFT_TYPE left, RIGHT_TYPE right, bool is_left_null, bool is_right_null) {
		return OP::template Operation<LEFT_TYPE>(left, right, is_left_null, is_right_null);
	}
};

template <class LEFT_TYPE, class RIGHT_TYPE, class RESULT_TYPE, class OP>
static void DistinctExecuteGenericLoop(LEFT_TYPE *__restrict ldata, RIGHT_TYPE *__restrict rdata,
                                       RESULT_TYPE *__restrict result_data, const SelectionVector *__restrict lsel,
                                       const SelectionVector *__restrict rsel, idx_t count, ValidityMask &lmask,
                                       ValidityMask &rmask, ValidityMask &result_mask) {
	for (idx_t i = 0; i < count; i++) {
		auto lindex = lsel->get_index(i);
		auto rindex = rsel->get_index(i);
		auto lentry = ldata[lindex];
		auto rentry = rdata[rindex];
		result_data[i] =
		    OP::template Operation<LEFT_TYPE>(lentry, rentry, !lmask.RowIsValid(lindex), !rmask.RowIsValid(rindex));
	}
}

template <class LEFT_TYPE, class RIGHT_TYPE, class RESULT_TYPE, class OP>
static void DistinctExecuteConstant(Vector &left, Vector &right, Vector &result) {
	result.SetVectorType(VectorType::CONSTANT_VECTOR);

	auto ldata = ConstantVector::GetData<LEFT_TYPE>(left);
	auto rdata = ConstantVector::GetData<RIGHT_TYPE>(right);
	auto result_data = ConstantVector::GetData<RESULT_TYPE>(result);
	*result_data =
	    OP::template Operation<LEFT_TYPE>(*ldata, *rdata, ConstantVector::IsNull(left), ConstantVector::IsNull(right));
}

template <class LEFT_TYPE, class RIGHT_TYPE, class RESULT_TYPE, class OP>
static void DistinctExecuteGeneric(Vector &left, Vector &right, Vector &result, idx_t count) {
	if (left.GetVectorType() == VectorType::CONSTANT_VECTOR && right.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		DistinctExecuteConstant<LEFT_TYPE, RIGHT_TYPE, RESULT_TYPE, OP>(left, right, result);
	} else {
		UnifiedVectorFormat ldata, rdata;

		left.ToUnifiedFormat(count, ldata);
		right.ToUnifiedFormat(count, rdata);

		result.SetVectorType(VectorType::FLAT_VECTOR);
		auto result_data = FlatVector::GetData<RESULT_TYPE>(result);
		DistinctExecuteGenericLoop<LEFT_TYPE, RIGHT_TYPE, RESULT_TYPE, OP>(
		    (LEFT_TYPE *)ldata.data, (RIGHT_TYPE *)rdata.data, result_data, ldata.sel, rdata.sel, count, ldata.validity,
		    rdata.validity, FlatVector::Validity(result));
	}
}

template <class LEFT_TYPE, class RIGHT_TYPE, class RESULT_TYPE, class OP>
static void DistinctExecuteSwitch(Vector &left, Vector &right, Vector &result, idx_t count) {
	DistinctExecuteGeneric<LEFT_TYPE, RIGHT_TYPE, RESULT_TYPE, OP>(left, right, result, count);
}

template <class LEFT_TYPE, class RIGHT_TYPE, class RESULT_TYPE, class OP>
static void DistinctExecute(Vector &left, Vector &right, Vector &result, idx_t count) {
	DistinctExecuteSwitch<LEFT_TYPE, RIGHT_TYPE, RESULT_TYPE, OP>(left, right, result, count);
}

template <class LEFT_TYPE, class RIGHT_TYPE, class OP, bool NO_NULL, bool HAS_TRUE_SEL, bool HAS_FALSE_SEL>
static inline idx_t
DistinctSelectGenericLoop(LEFT_TYPE *__restrict ldata, RIGHT_TYPE *__restrict rdata,
                          const SelectionVector *__restrict lsel, const SelectionVector *__restrict rsel,
                          const SelectionVector *__restrict result_sel, idx_t count, ValidityMask &lmask,
                          ValidityMask &rmask, SelectionVector *true_sel, SelectionVector *false_sel) {
	idx_t true_count = 0, false_count = 0;
	for (idx_t i = 0; i < count; i++) {
		auto result_idx = result_sel->get_index(i);
		auto lindex = lsel->get_index(i);
		auto rindex = rsel->get_index(i);
		if (NO_NULL) {
			if (OP::Operation(ldata[lindex], rdata[rindex], false, false)) {
				if (HAS_TRUE_SEL) {
					true_sel->set_index(true_count++, result_idx);
				}
			} else {
				if (HAS_FALSE_SEL) {
					false_sel->set_index(false_count++, result_idx);
				}
			}
		} else {
			if (OP::Operation(ldata[lindex], rdata[rindex], !lmask.RowIsValid(lindex), !rmask.RowIsValid(rindex))) {
				if (HAS_TRUE_SEL) {
					true_sel->set_index(true_count++, result_idx);
				}
			} else {
				if (HAS_FALSE_SEL) {
					false_sel->set_index(false_count++, result_idx);
				}
			}
		}
	}
	if (HAS_TRUE_SEL) {
		return true_count;
	} else {
		return count - false_count;
	}
}
template <class LEFT_TYPE, class RIGHT_TYPE, class OP, bool NO_NULL>
static inline idx_t
DistinctSelectGenericLoopSelSwitch(LEFT_TYPE *__restrict ldata, RIGHT_TYPE *__restrict rdata,
                                   const SelectionVector *__restrict lsel, const SelectionVector *__restrict rsel,
                                   const SelectionVector *__restrict result_sel, idx_t count, ValidityMask &lmask,
                                   ValidityMask &rmask, SelectionVector *true_sel, SelectionVector *false_sel) {
	if (true_sel && false_sel) {
		return DistinctSelectGenericLoop<LEFT_TYPE, RIGHT_TYPE, OP, NO_NULL, true, true>(
		    ldata, rdata, lsel, rsel, result_sel, count, lmask, rmask, true_sel, false_sel);
	} else if (true_sel) {
		return DistinctSelectGenericLoop<LEFT_TYPE, RIGHT_TYPE, OP, NO_NULL, true, false>(
		    ldata, rdata, lsel, rsel, result_sel, count, lmask, rmask, true_sel, false_sel);
	} else {
		D_ASSERT(false_sel);
		return DistinctSelectGenericLoop<LEFT_TYPE, RIGHT_TYPE, OP, NO_NULL, false, true>(
		    ldata, rdata, lsel, rsel, result_sel, count, lmask, rmask, true_sel, false_sel);
	}
}

template <class LEFT_TYPE, class RIGHT_TYPE, class OP>
static inline idx_t
DistinctSelectGenericLoopSwitch(LEFT_TYPE *__restrict ldata, RIGHT_TYPE *__restrict rdata,
                                const SelectionVector *__restrict lsel, const SelectionVector *__restrict rsel,
                                const SelectionVector *__restrict result_sel, idx_t count, ValidityMask &lmask,
                                ValidityMask &rmask, SelectionVector *true_sel, SelectionVector *false_sel) {
	if (!lmask.AllValid() || !rmask.AllValid()) {
		return DistinctSelectGenericLoopSelSwitch<LEFT_TYPE, RIGHT_TYPE, OP, false>(
		    ldata, rdata, lsel, rsel, result_sel, count, lmask, rmask, true_sel, false_sel);
	} else {
		return DistinctSelectGenericLoopSelSwitch<LEFT_TYPE, RIGHT_TYPE, OP, true>(
		    ldata, rdata, lsel, rsel, result_sel, count, lmask, rmask, true_sel, false_sel);
	}
}

template <class LEFT_TYPE, class RIGHT_TYPE, class OP>
static idx_t DistinctSelectGeneric(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                   SelectionVector *true_sel, SelectionVector *false_sel) {
	UnifiedVectorFormat ldata, rdata;

	left.ToUnifiedFormat(count, ldata);
	right.ToUnifiedFormat(count, rdata);

	return DistinctSelectGenericLoopSwitch<LEFT_TYPE, RIGHT_TYPE, OP>((LEFT_TYPE *)ldata.data, (RIGHT_TYPE *)rdata.data,
	                                                                  ldata.sel, rdata.sel, sel, count, ldata.validity,
	                                                                  rdata.validity, true_sel, false_sel);
}
template <class LEFT_TYPE, class RIGHT_TYPE, class OP, bool LEFT_CONSTANT, bool RIGHT_CONSTANT, bool NO_NULL,
          bool HAS_TRUE_SEL, bool HAS_FALSE_SEL>
static inline idx_t DistinctSelectFlatLoop(LEFT_TYPE *__restrict ldata, RIGHT_TYPE *__restrict rdata,
                                           const SelectionVector *sel, idx_t count, ValidityMask &lmask,
                                           ValidityMask &rmask, SelectionVector *true_sel, SelectionVector *false_sel) {
	idx_t true_count = 0, false_count = 0;
	for (idx_t i = 0; i < count; i++) {
		idx_t result_idx = sel->get_index(i);
		idx_t lidx = LEFT_CONSTANT ? 0 : i;
		idx_t ridx = RIGHT_CONSTANT ? 0 : i;
		const bool lnull = !lmask.RowIsValid(lidx);
		const bool rnull = !rmask.RowIsValid(ridx);
		bool comparison_result = OP::Operation(ldata[lidx], rdata[ridx], lnull, rnull);
		if (HAS_TRUE_SEL) {
			true_sel->set_index(true_count, result_idx);
			true_count += comparison_result;
		}
		if (HAS_FALSE_SEL) {
			false_sel->set_index(false_count, result_idx);
			false_count += !comparison_result;
		}
	}
	if (HAS_TRUE_SEL) {
		return true_count;
	} else {
		return count - false_count;
	}
}

template <class LEFT_TYPE, class RIGHT_TYPE, class OP, bool LEFT_CONSTANT, bool RIGHT_CONSTANT, bool NO_NULL>
static inline idx_t DistinctSelectFlatLoopSelSwitch(LEFT_TYPE *__restrict ldata, RIGHT_TYPE *__restrict rdata,
                                                    const SelectionVector *sel, idx_t count, ValidityMask &lmask,
                                                    ValidityMask &rmask, SelectionVector *true_sel,
                                                    SelectionVector *false_sel) {
	if (true_sel && false_sel) {
		return DistinctSelectFlatLoop<LEFT_TYPE, RIGHT_TYPE, OP, LEFT_CONSTANT, RIGHT_CONSTANT, NO_NULL, true, true>(
		    ldata, rdata, sel, count, lmask, rmask, true_sel, false_sel);
	} else if (true_sel) {
		return DistinctSelectFlatLoop<LEFT_TYPE, RIGHT_TYPE, OP, LEFT_CONSTANT, RIGHT_CONSTANT, NO_NULL, true, false>(
		    ldata, rdata, sel, count, lmask, rmask, true_sel, false_sel);
	} else {
		D_ASSERT(false_sel);
		return DistinctSelectFlatLoop<LEFT_TYPE, RIGHT_TYPE, OP, LEFT_CONSTANT, RIGHT_CONSTANT, NO_NULL, false, true>(
		    ldata, rdata, sel, count, lmask, rmask, true_sel, false_sel);
	}
}

template <class LEFT_TYPE, class RIGHT_TYPE, class OP, bool LEFT_CONSTANT, bool RIGHT_CONSTANT>
static inline idx_t DistinctSelectFlatLoopSwitch(LEFT_TYPE *__restrict ldata, RIGHT_TYPE *__restrict rdata,
                                                 const SelectionVector *sel, idx_t count, ValidityMask &lmask,
                                                 ValidityMask &rmask, SelectionVector *true_sel,
                                                 SelectionVector *false_sel) {
	return DistinctSelectFlatLoopSelSwitch<LEFT_TYPE, RIGHT_TYPE, OP, LEFT_CONSTANT, RIGHT_CONSTANT, true>(
	    ldata, rdata, sel, count, lmask, rmask, true_sel, false_sel);
}
template <class LEFT_TYPE, class RIGHT_TYPE, class OP, bool LEFT_CONSTANT, bool RIGHT_CONSTANT>
static idx_t DistinctSelectFlat(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                SelectionVector *true_sel, SelectionVector *false_sel) {
	auto ldata = FlatVector::GetData<LEFT_TYPE>(left);
	auto rdata = FlatVector::GetData<RIGHT_TYPE>(right);
	if (LEFT_CONSTANT) {
		ValidityMask validity;
		if (ConstantVector::IsNull(left)) {
			validity.SetAllInvalid(1);
		}
		return DistinctSelectFlatLoopSwitch<LEFT_TYPE, RIGHT_TYPE, OP, LEFT_CONSTANT, RIGHT_CONSTANT>(
		    ldata, rdata, sel, count, validity, FlatVector::Validity(right), true_sel, false_sel);
	} else if (RIGHT_CONSTANT) {
		ValidityMask validity;
		if (ConstantVector::IsNull(right)) {
			validity.SetAllInvalid(1);
		}
		return DistinctSelectFlatLoopSwitch<LEFT_TYPE, RIGHT_TYPE, OP, LEFT_CONSTANT, RIGHT_CONSTANT>(
		    ldata, rdata, sel, count, FlatVector::Validity(left), validity, true_sel, false_sel);
	} else {
		return DistinctSelectFlatLoopSwitch<LEFT_TYPE, RIGHT_TYPE, OP, LEFT_CONSTANT, RIGHT_CONSTANT>(
		    ldata, rdata, sel, count, FlatVector::Validity(left), FlatVector::Validity(right), true_sel, false_sel);
	}
}
template <class LEFT_TYPE, class RIGHT_TYPE, class OP>
static idx_t DistinctSelectConstant(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                    SelectionVector *true_sel, SelectionVector *false_sel) {
	auto ldata = ConstantVector::GetData<LEFT_TYPE>(left);
	auto rdata = ConstantVector::GetData<RIGHT_TYPE>(right);

	// both sides are constant, return either 0 or the count
	// in this case we do not fill in the result selection vector at all
	if (!OP::Operation(*ldata, *rdata, ConstantVector::IsNull(left), ConstantVector::IsNull(right))) {
		if (false_sel) {
			for (idx_t i = 0; i < count; i++) {
				false_sel->set_index(i, sel->get_index(i));
			}
		}
		return 0;
	} else {
		if (true_sel) {
			for (idx_t i = 0; i < count; i++) {
				true_sel->set_index(i, sel->get_index(i));
			}
		}
		return count;
	}
}

template <class LEFT_TYPE, class RIGHT_TYPE, class OP>
static idx_t DistinctSelect(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                            SelectionVector *true_sel, SelectionVector *false_sel) {
	if (!sel) {
		sel = FlatVector::IncrementalSelectionVector();
	}
	if (left.GetVectorType() == VectorType::CONSTANT_VECTOR && right.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		return DistinctSelectConstant<LEFT_TYPE, RIGHT_TYPE, OP>(left, right, sel, count, true_sel, false_sel);
	} else if (left.GetVectorType() == VectorType::CONSTANT_VECTOR &&
	           right.GetVectorType() == VectorType::FLAT_VECTOR) {
		return DistinctSelectFlat<LEFT_TYPE, RIGHT_TYPE, OP, true, false>(left, right, sel, count, true_sel, false_sel);
	} else if (left.GetVectorType() == VectorType::FLAT_VECTOR &&
	           right.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		return DistinctSelectFlat<LEFT_TYPE, RIGHT_TYPE, OP, false, true>(left, right, sel, count, true_sel, false_sel);
	} else if (left.GetVectorType() == VectorType::FLAT_VECTOR && right.GetVectorType() == VectorType::FLAT_VECTOR) {
		return DistinctSelectFlat<LEFT_TYPE, RIGHT_TYPE, OP, false, false>(left, right, sel, count, true_sel,
		                                                                   false_sel);
	} else {
		return DistinctSelectGeneric<LEFT_TYPE, RIGHT_TYPE, OP>(left, right, sel, count, true_sel, false_sel);
	}
}

template <class OP>
static idx_t DistinctSelectNotNull(Vector &left, Vector &right, const idx_t count, idx_t &true_count,
                                   const SelectionVector &sel, SelectionVector &maybe_vec, OptionalSelection &true_opt,
                                   OptionalSelection &false_opt) {
	UnifiedVectorFormat lvdata, rvdata;
	left.ToUnifiedFormat(count, lvdata);
	right.ToUnifiedFormat(count, rvdata);

	auto &lmask = lvdata.validity;
	auto &rmask = rvdata.validity;

	idx_t remaining = 0;
	if (lmask.AllValid() && rmask.AllValid()) {
		//	None are NULL, distinguish values.
		for (idx_t i = 0; i < count; ++i) {
			const auto idx = sel.get_index(i);
			maybe_vec.set_index(remaining++, idx);
		}
		return remaining;
	}

	// Slice the Vectors down to the rows that are not determined (i.e., neither is NULL)
	SelectionVector slicer(count);
	true_count = 0;
	idx_t false_count = 0;
	for (idx_t i = 0; i < count; ++i) {
		const auto result_idx = sel.get_index(i);
		const auto lidx = lvdata.sel->get_index(i);
		const auto ridx = rvdata.sel->get_index(i);
		const auto lnull = !lmask.RowIsValid(lidx);
		const auto rnull = !rmask.RowIsValid(ridx);
		if (lnull || rnull) {
			// If either is NULL then we can major distinguish them
			if (!OP::Operation(false, false, lnull, rnull)) {
				false_opt.Append(false_count, result_idx);
			} else {
				true_opt.Append(true_count, result_idx);
			}
		} else {
			//	Neither is NULL, distinguish values.
			slicer.set_index(remaining, i);
			maybe_vec.set_index(remaining++, result_idx);
		}
	}

	true_opt.Advance(true_count);
	false_opt.Advance(false_count);

	if (remaining && remaining < count) {
		left.Slice(slicer, remaining);
		right.Slice(slicer, remaining);
	}

	return remaining;
}

struct PositionComparator {
	// Select the rows that definitely match.
	// Default to the same as the final row
	template <typename OP>
	static idx_t Definite(Vector &left, Vector &right, const SelectionVector &sel, idx_t count,
	                      SelectionVector *true_sel, SelectionVector &false_sel) {
		return Final<OP>(left, right, sel, count, true_sel, &false_sel);
	}

	// Select the possible rows that need further testing.
	// Usually this means Is Not Distinct, as those are the semantics used by Postges
	template <typename OP>
	static idx_t Possible(Vector &left, Vector &right, const SelectionVector &sel, idx_t count,
	                      SelectionVector &true_sel, SelectionVector *false_sel) {
		return VectorOperations::NestedEquals(left, right, sel, count, &true_sel, false_sel);
	}

	// Select the matching rows for the final position.
	// This needs to be specialised.
	template <typename OP>
	static idx_t Final(Vector &left, Vector &right, const SelectionVector &sel, idx_t count, SelectionVector *true_sel,
	                   SelectionVector *false_sel) {
		return 0;
	}

	// Tie-break based on length when one of the sides has been exhausted, returning true if the LHS matches.
	// This essentially means that the existing positions compare equal.
	// Default to the same semantics as the OP for idx_t. This works in most cases.
	template <typename OP>
	static bool TieBreak(const idx_t lpos, const idx_t rpos) {
		return OP::Operation(lpos, rpos, false, false);
	}
};

// NotDistinctFrom must always check every column
template <>
idx_t PositionComparator::Definite<duckdb::NotDistinctFrom>(Vector &left, Vector &right, const SelectionVector &sel,
                                                            idx_t count, SelectionVector *true_sel,
                                                            SelectionVector &false_sel) {
	return 0;
}

template <>
idx_t PositionComparator::Final<duckdb::NotDistinctFrom>(Vector &left, Vector &right, const SelectionVector &sel,
                                                         idx_t count, SelectionVector *true_sel,
                                                         SelectionVector *false_sel) {
	return VectorOperations::NestedEquals(left, right, sel, count, true_sel, false_sel);
}

// DistinctFrom must check everything that matched
template <>
idx_t PositionComparator::Possible<duckdb::DistinctFrom>(Vector &left, Vector &right, const SelectionVector &sel,
                                                         idx_t count, SelectionVector &true_sel,
                                                         SelectionVector *false_sel) {
	return count;
}

template <>
idx_t PositionComparator::Final<duckdb::DistinctFrom>(Vector &left, Vector &right, const SelectionVector &sel,
                                                      idx_t count, SelectionVector *true_sel,
                                                      SelectionVector *false_sel) {
	return VectorOperations::NestedNotEquals(left, right, sel, count, true_sel, false_sel);
}

// Non-strict inequalities must use strict comparisons for Definite
template <>
idx_t PositionComparator::Definite<duckdb::DistinctLessThanEquals>(Vector &left, Vector &right,
                                                                   const SelectionVector &sel, idx_t count,
                                                                   SelectionVector *true_sel,
                                                                   SelectionVector &false_sel) {
	return VectorOperations::DistinctLessThan(left, right, &sel, count, true_sel, &false_sel);
}

template <>
idx_t PositionComparator::Final<duckdb::DistinctLessThanEquals>(Vector &left, Vector &right, const SelectionVector &sel,
                                                                idx_t count, SelectionVector *true_sel,
                                                                SelectionVector *false_sel) {
	return VectorOperations::DistinctLessThanEquals(left, right, &sel, count, true_sel, false_sel);
}

template <>
idx_t PositionComparator::Definite<duckdb::DistinctGreaterThanEquals>(Vector &left, Vector &right,
                                                                      const SelectionVector &sel, idx_t count,
                                                                      SelectionVector *true_sel,
                                                                      SelectionVector &false_sel) {
	return VectorOperations::DistinctGreaterThan(left, right, &sel, count, true_sel, &false_sel);
}

template <>
idx_t PositionComparator::Final<duckdb::DistinctGreaterThanEquals>(Vector &left, Vector &right,
                                                                   const SelectionVector &sel, idx_t count,
                                                                   SelectionVector *true_sel,
                                                                   SelectionVector *false_sel) {
	return VectorOperations::DistinctGreaterThanEquals(left, right, &sel, count, true_sel, false_sel);
}

// Strict inequalities just use strict for both Definite and Final
template <>
idx_t PositionComparator::Final<duckdb::DistinctLessThan>(Vector &left, Vector &right, const SelectionVector &sel,
                                                          idx_t count, SelectionVector *true_sel,
                                                          SelectionVector *false_sel) {
	return VectorOperations::DistinctLessThan(left, right, &sel, count, true_sel, false_sel);
}

template <>
idx_t PositionComparator::Final<duckdb::DistinctGreaterThan>(Vector &left, Vector &right, const SelectionVector &sel,
                                                             idx_t count, SelectionVector *true_sel,
                                                             SelectionVector *false_sel) {
	return VectorOperations::DistinctGreaterThan(left, right, &sel, count, true_sel, false_sel);
}

using StructEntries = vector<unique_ptr<Vector>>;

static void ExtractNestedSelection(const SelectionVector &slice_sel, const idx_t count, const SelectionVector &sel,
                                   OptionalSelection &opt) {

	for (idx_t i = 0; i < count;) {
		const auto slice_idx = slice_sel.get_index(i);
		const auto result_idx = sel.get_index(slice_idx);
		opt.Append(i, result_idx);
	}
	opt.Advance(count);
}

static void DensifyNestedSelection(const SelectionVector &dense_sel, const idx_t count, SelectionVector &slice_sel) {
	for (idx_t i = 0; i < count; ++i) {
		slice_sel.set_index(i, dense_sel.get_index(i));
	}
}

template <class OP>
static idx_t DistinctSelectStruct(Vector &left, Vector &right, idx_t count, const SelectionVector &sel,
                                  OptionalSelection &true_opt, OptionalSelection &false_opt) {
	if (count == 0) {
		return 0;
	}

	// Avoid allocating in the 99% of the cases where we don't need to.
	StructEntries lsliced, rsliced;
	auto &lchildren = StructVector::GetEntries(left);
	auto &rchildren = StructVector::GetEntries(right);
	D_ASSERT(lchildren.size() == rchildren.size());

	// In order to reuse the comparators, we have to track what passed and failed internally.
	// To do that, we need local SVs that we then merge back into the real ones after every pass.
	const auto vcount = count;
	SelectionVector slice_sel(count);
	for (idx_t i = 0; i < count; ++i) {
		slice_sel.set_index(i, i);
	}

	SelectionVector true_sel(count);
	SelectionVector false_sel(count);

	idx_t match_count = 0;
	for (idx_t col_no = 0; col_no < lchildren.size(); ++col_no) {
		// Slice the children to maintain density
		Vector lchild(*lchildren[col_no]);
		lchild.Flatten(vcount);
		lchild.Slice(slice_sel, count);

		Vector rchild(*rchildren[col_no]);
		rchild.Flatten(vcount);
		rchild.Slice(slice_sel, count);

		// Find everything that definitely matches
		auto true_count = PositionComparator::Definite<OP>(lchild, rchild, slice_sel, count, &true_sel, false_sel);
		if (true_count > 0) {
			auto false_count = count - true_count;

			// Extract the definite matches into the true result
			ExtractNestedSelection(false_count ? true_sel : slice_sel, true_count, sel, true_opt);

			// Remove the definite matches from the slicing vector
			DensifyNestedSelection(false_sel, false_count, slice_sel);

			match_count += true_count;
			count -= true_count;
		}

		if (col_no != lchildren.size() - 1) {
			// Find what might match on the next position
			true_count = PositionComparator::Possible<OP>(lchild, rchild, slice_sel, count, true_sel, &false_sel);
			auto false_count = count - true_count;

			// Extract the definite failures into the false result
			ExtractNestedSelection(true_count ? false_sel : slice_sel, false_count, sel, false_opt);

			// Remove any definite failures from the slicing vector
			if (false_count) {
				DensifyNestedSelection(true_sel, true_count, slice_sel);
			}

			count = true_count;
		} else {
			true_count = PositionComparator::Final<OP>(lchild, rchild, slice_sel, count, &true_sel, &false_sel);
			auto false_count = count - true_count;

			// Extract the definite matches into the true result
			ExtractNestedSelection(false_count ? true_sel : slice_sel, true_count, sel, true_opt);

			// Extract the definite failures into the false result
			ExtractNestedSelection(true_count ? false_sel : slice_sel, false_count, sel, false_opt);

			match_count += true_count;
		}
	}
	return match_count;
}

static void PositionListCursor(SelectionVector &cursor, UnifiedVectorFormat &vdata, const idx_t pos,
                               const SelectionVector &slice_sel, const idx_t count) {
	const auto data = (const list_entry_t *)vdata.data;
	for (idx_t i = 0; i < count; ++i) {
		const auto slice_idx = slice_sel.get_index(i);

		const auto lidx = vdata.sel->get_index(slice_idx);
		const auto &entry = data[lidx];
		cursor.set_index(i, entry.offset + pos);
	}
}

template <class OP>
static idx_t DistinctSelectList(Vector &left, Vector &right, idx_t count, const SelectionVector &sel,
                                OptionalSelection &true_opt, OptionalSelection &false_opt) {
	if (count == 0) {
		return count;
	}

	// Create dictionary views of the children so we can vectorise the positional comparisons.
	SelectionVector lcursor(count);
	SelectionVector rcursor(count);

	ListVector::GetEntry(left).Flatten(ListVector::GetListSize(left));
	ListVector::GetEntry(right).Flatten(ListVector::GetListSize(right));
	Vector lchild(ListVector::GetEntry(left), lcursor, count);
	Vector rchild(ListVector::GetEntry(right), rcursor, count);

	// To perform the positional comparison, we use a vectorisation of the following algorithm:
	// bool CompareLists(T *left, idx_t nleft, T *right, nright) {
	// 	for (idx_t pos = 0; ; ++pos) {
	// 		if (nleft == pos || nright == pos)
	// 			return OP::TieBreak(nleft, nright);
	// 		if (OP::Definite(*left, *right))
	// 			return true;
	// 		if (!OP::Maybe(*left, *right))
	// 			return false;
	// 		}
	//	 	++left, ++right;
	// 	}
	// }

	// Get pointers to the list entries
	UnifiedVectorFormat lvdata;
	left.ToUnifiedFormat(count, lvdata);
	const auto ldata = (const list_entry_t *)lvdata.data;

	UnifiedVectorFormat rvdata;
	right.ToUnifiedFormat(count, rvdata);
	const auto rdata = (const list_entry_t *)rvdata.data;

	// In order to reuse the comparators, we have to track what passed and failed internally.
	// To do that, we need local SVs that we then merge back into the real ones after every pass.
	SelectionVector slice_sel(count);
	for (idx_t i = 0; i < count; ++i) {
		slice_sel.set_index(i, i);
	}

	SelectionVector true_sel(count);
	SelectionVector false_sel(count);

	idx_t match_count = 0;
	for (idx_t pos = 0; count > 0; ++pos) {
		// Set up the cursors for the current position
		PositionListCursor(lcursor, lvdata, pos, slice_sel, count);
		PositionListCursor(rcursor, rvdata, pos, slice_sel, count);

		// Tie-break the pairs where one of the LISTs is exhausted.
		idx_t true_count = 0;
		idx_t false_count = 0;
		idx_t maybe_count = 0;
		for (idx_t i = 0; i < count; ++i) {
			const auto slice_idx = slice_sel.get_index(i);
			const auto lidx = lvdata.sel->get_index(slice_idx);
			const auto &lentry = ldata[lidx];
			const auto ridx = rvdata.sel->get_index(slice_idx);
			const auto &rentry = rdata[ridx];
			if (lentry.length == pos || rentry.length == pos) {
				const auto idx = sel.get_index(slice_idx);
				if (PositionComparator::TieBreak<OP>(lentry.length, rentry.length)) {
					true_opt.Append(true_count, idx);
				} else {
					false_opt.Append(false_count, idx);
				}
			} else {
				true_sel.set_index(maybe_count++, slice_idx);
			}
		}
		true_opt.Advance(true_count);
		false_opt.Advance(false_count);
		match_count += true_count;

		// Redensify the list cursors
		if (maybe_count < count) {
			count = maybe_count;
			DensifyNestedSelection(true_sel, count, slice_sel);
			PositionListCursor(lcursor, lvdata, pos, slice_sel, count);
			PositionListCursor(rcursor, rvdata, pos, slice_sel, count);
		}

		// Find everything that definitely matches
		true_count = PositionComparator::Definite<OP>(lchild, rchild, slice_sel, count, &true_sel, false_sel);
		if (true_count) {
			false_count = count - true_count;
			ExtractNestedSelection(false_count ? true_sel : slice_sel, true_count, sel, true_opt);
			match_count += true_count;

			// Redensify the list cursors
			count -= true_count;
			DensifyNestedSelection(false_sel, count, slice_sel);
			PositionListCursor(lcursor, lvdata, pos, slice_sel, count);
			PositionListCursor(rcursor, rvdata, pos, slice_sel, count);
		}

		// Find what might match on the next position
		true_count = PositionComparator::Possible<OP>(lchild, rchild, slice_sel, count, true_sel, &false_sel);
		false_count = count - true_count;
		ExtractNestedSelection(true_count ? false_sel : slice_sel, false_count, sel, false_opt);

		if (false_count) {
			DensifyNestedSelection(true_sel, true_count, slice_sel);
		}
		count = true_count;
	}

	return match_count;
}

template <class OP, class OPNESTED>
static idx_t DistinctSelectNested(Vector &left, Vector &right, const SelectionVector *sel, const idx_t count,
                                  SelectionVector *true_sel, SelectionVector *false_sel) {
	// The Select operations all use a dense pair of input vectors to partition
	// a selection vector in a single pass. But to implement progressive comparisons,
	// we have to make multiple passes, so we need to keep track of the original input positions
	// and then scatter the output selections when we are done.
	if (!sel) {
		sel = FlatVector::IncrementalSelectionVector();
	}

	// Make buffered selections for progressive comparisons
	// TODO: Remove unnecessary allocations
	SelectionVector true_vec(count);
	OptionalSelection true_opt(&true_vec);

	SelectionVector false_vec(count);
	OptionalSelection false_opt(&false_vec);

	SelectionVector maybe_vec(count);

	// Handle NULL nested values
	Vector l_not_null(left);
	Vector r_not_null(right);

	idx_t match_count = 0;
	auto unknown =
	    DistinctSelectNotNull<OP>(l_not_null, r_not_null, count, match_count, *sel, maybe_vec, true_opt, false_opt);

	if (PhysicalType::LIST == left.GetType().InternalType()) {
		match_count += DistinctSelectList<OPNESTED>(l_not_null, r_not_null, unknown, maybe_vec, true_opt, false_opt);
	} else {
		match_count += DistinctSelectStruct<OPNESTED>(l_not_null, r_not_null, unknown, maybe_vec, true_opt, false_opt);
	}

	// Copy the buffered selections to the output selections
	if (true_sel) {
		DensifyNestedSelection(true_vec, match_count, *true_sel);
	}

	if (false_sel) {
		DensifyNestedSelection(false_vec, count - match_count, *false_sel);
	}

	return match_count;
}

template <typename OP>
static void NestedDistinctExecute(Vector &left, Vector &right, Vector &result, idx_t count);

template <class T, class OP>
static inline void TemplatedDistinctExecute(Vector &left, Vector &right, Vector &result, idx_t count) {
	DistinctExecute<T, T, bool, OP>(left, right, result, count);
}
template <class OP>
static void ExecuteDistinct(Vector &left, Vector &right, Vector &result, idx_t count) {
	D_ASSERT(left.GetType() == right.GetType() && result.GetType() == LogicalType::BOOLEAN);
	// the inplace loops take the result as the last parameter
	switch (left.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		TemplatedDistinctExecute<int8_t, OP>(left, right, result, count);
		break;
	case PhysicalType::INT16:
		TemplatedDistinctExecute<int16_t, OP>(left, right, result, count);
		break;
	case PhysicalType::INT32:
		TemplatedDistinctExecute<int32_t, OP>(left, right, result, count);
		break;
	case PhysicalType::INT64:
		TemplatedDistinctExecute<int64_t, OP>(left, right, result, count);
		break;
	case PhysicalType::UINT8:
		TemplatedDistinctExecute<uint8_t, OP>(left, right, result, count);
		break;
	case PhysicalType::UINT16:
		TemplatedDistinctExecute<uint16_t, OP>(left, right, result, count);
		break;
	case PhysicalType::UINT32:
		TemplatedDistinctExecute<uint32_t, OP>(left, right, result, count);
		break;
	case PhysicalType::UINT64:
		TemplatedDistinctExecute<uint64_t, OP>(left, right, result, count);
		break;
	case PhysicalType::INT128:
		TemplatedDistinctExecute<hugeint_t, OP>(left, right, result, count);
		break;
	case PhysicalType::FLOAT:
		TemplatedDistinctExecute<float, OP>(left, right, result, count);
		break;
	case PhysicalType::DOUBLE:
		TemplatedDistinctExecute<double, OP>(left, right, result, count);
		break;
	case PhysicalType::INTERVAL:
		TemplatedDistinctExecute<interval_t, OP>(left, right, result, count);
		break;
	case PhysicalType::VARCHAR:
		TemplatedDistinctExecute<string_t, OP>(left, right, result, count);
		break;
	case PhysicalType::LIST:
	case PhysicalType::STRUCT:
		NestedDistinctExecute<OP>(left, right, result, count);
		break;
	default:
		throw InternalException("Invalid type for distinct comparison");
	}
}

template <class OP, class OPNESTED = OP>
static idx_t TemplatedDistinctSelectOperation(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                              SelectionVector *true_sel, SelectionVector *false_sel) {
	// the inplace loops take the result as the last parameter
	switch (left.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		return DistinctSelect<int8_t, int8_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::INT16:
		return DistinctSelect<int16_t, int16_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::INT32:
		return DistinctSelect<int32_t, int32_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::INT64:
		return DistinctSelect<int64_t, int64_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::UINT8:
		return DistinctSelect<uint8_t, uint8_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::UINT16:
		return DistinctSelect<uint16_t, uint16_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::UINT32:
		return DistinctSelect<uint32_t, uint32_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::UINT64:
		return DistinctSelect<uint64_t, uint64_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::INT128:
		return DistinctSelect<hugeint_t, hugeint_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::FLOAT:
		return DistinctSelect<float, float, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::DOUBLE:
		return DistinctSelect<double, double, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::INTERVAL:
		return DistinctSelect<interval_t, interval_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::VARCHAR:
		return DistinctSelect<string_t, string_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::STRUCT:
	case PhysicalType::LIST:
		return DistinctSelectNested<OP, OPNESTED>(left, right, sel, count, true_sel, false_sel);
	default:
		throw InternalException("Invalid type for distinct selection");
	}
}

template <typename OP>
static void NestedDistinctExecute(Vector &left, Vector &right, Vector &result, idx_t count) {
	const auto left_constant = left.GetVectorType() == VectorType::CONSTANT_VECTOR;
	const auto right_constant = right.GetVectorType() == VectorType::CONSTANT_VECTOR;

	if (left_constant && right_constant) {
		// both sides are constant, so just compare one element.
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		auto result_data = ConstantVector::GetData<bool>(result);
		SelectionVector true_sel(1);
		auto match_count = TemplatedDistinctSelectOperation<OP>(left, right, nullptr, 1, &true_sel, nullptr);
		result_data[0] = match_count > 0;
		return;
	}

	SelectionVector true_sel(count);
	SelectionVector false_sel(count);

	// DISTINCT is either true or false
	idx_t match_count = TemplatedDistinctSelectOperation<OP>(left, right, nullptr, count, &true_sel, &false_sel);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<bool>(result);

	for (idx_t i = 0; i < match_count; ++i) {
		const auto idx = true_sel.get_index(i);
		result_data[idx] = true;
	}

	const idx_t no_match_count = count - match_count;
	for (idx_t i = 0; i < no_match_count; ++i) {
		const auto idx = false_sel.get_index(i);
		result_data[idx] = false;
	}
}

void VectorOperations::DistinctFrom(Vector &left, Vector &right, Vector &result, idx_t count) {
	ExecuteDistinct<duckdb::DistinctFrom>(left, right, result, count);
}

void VectorOperations::NotDistinctFrom(Vector &left, Vector &right, Vector &result, idx_t count) {
	ExecuteDistinct<duckdb::NotDistinctFrom>(left, right, result, count);
}

// true := A != B with nulls being equal
idx_t VectorOperations::DistinctFrom(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                     SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedDistinctSelectOperation<duckdb::DistinctFrom>(left, right, sel, count, true_sel, false_sel);
}
// true := A == B with nulls being equal
idx_t VectorOperations::NotDistinctFrom(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                        SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedDistinctSelectOperation<duckdb::NotDistinctFrom>(left, right, sel, count, true_sel, false_sel);
}

// true := A > B with nulls being maximal
idx_t VectorOperations::DistinctGreaterThan(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                            SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedDistinctSelectOperation<duckdb::DistinctGreaterThan>(left, right, sel, count, true_sel, false_sel);
}

// true := A > B with nulls being minimal
idx_t VectorOperations::DistinctGreaterThanNullsFirst(Vector &left, Vector &right, const SelectionVector *sel,
                                                      idx_t count, SelectionVector *true_sel,
                                                      SelectionVector *false_sel) {
	return TemplatedDistinctSelectOperation<duckdb::DistinctGreaterThanNullsFirst, duckdb::DistinctGreaterThan>(
	    left, right, sel, count, true_sel, false_sel);
}
// true := A >= B with nulls being maximal
idx_t VectorOperations::DistinctGreaterThanEquals(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                                  SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedDistinctSelectOperation<duckdb::DistinctGreaterThanEquals>(left, right, sel, count, true_sel,
	                                                                           false_sel);
}
// true := A < B with nulls being maximal
idx_t VectorOperations::DistinctLessThan(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                         SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedDistinctSelectOperation<duckdb::DistinctLessThan>(left, right, sel, count, true_sel, false_sel);
}

// true := A < B with nulls being minimal
idx_t VectorOperations::DistinctLessThanNullsFirst(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                                   SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedDistinctSelectOperation<duckdb::DistinctLessThanNullsFirst, duckdb::DistinctLessThan>(
	    left, right, sel, count, true_sel, false_sel);
}

// true := A <= B with nulls being maximal
idx_t VectorOperations::DistinctLessThanEquals(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                               SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedDistinctSelectOperation<duckdb::DistinctLessThanEquals>(left, right, sel, count, true_sel,
	                                                                        false_sel);
}

// true := A != B with nulls being equal, inputs selected
idx_t VectorOperations::NestedNotEquals(Vector &left, Vector &right, const SelectionVector &sel, idx_t count,
                                        SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedDistinctSelectOperation<duckdb::DistinctFrom>(left, right, &sel, count, true_sel, false_sel);
}
// true := A == B with nulls being equal, inputs selected
idx_t VectorOperations::NestedEquals(Vector &left, Vector &right, const SelectionVector &sel, idx_t count,
                                     SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedDistinctSelectOperation<duckdb::NotDistinctFrom>(left, right, &sel, count, true_sel, false_sel);
}

} // namespace duckdb
//===--------------------------------------------------------------------===//
// null_operators.cpp
// Description: This file contains the implementation of the
// IS NULL/NOT IS NULL operators
//===--------------------------------------------------------------------===//




namespace duckdb {

template <bool INVERSE>
void IsNullLoop(Vector &input, Vector &result, idx_t count) {
	D_ASSERT(result.GetType() == LogicalType::BOOLEAN);

	if (input.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		auto result_data = ConstantVector::GetData<bool>(result);
		*result_data = INVERSE ? !ConstantVector::IsNull(input) : ConstantVector::IsNull(input);
	} else {
		UnifiedVectorFormat data;
		input.ToUnifiedFormat(count, data);

		result.SetVectorType(VectorType::FLAT_VECTOR);
		auto result_data = FlatVector::GetData<bool>(result);
		for (idx_t i = 0; i < count; i++) {
			auto idx = data.sel->get_index(i);
			result_data[i] = INVERSE ? data.validity.RowIsValid(idx) : !data.validity.RowIsValid(idx);
		}
	}
}

void VectorOperations::IsNotNull(Vector &input, Vector &result, idx_t count) {
	IsNullLoop<true>(input, result, count);
}

void VectorOperations::IsNull(Vector &input, Vector &result, idx_t count) {
	IsNullLoop<false>(input, result, count);
}

bool VectorOperations::HasNotNull(Vector &input, idx_t count) {
	if (count == 0) {
		return false;
	}
	if (input.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		return !ConstantVector::IsNull(input);
	} else {
		UnifiedVectorFormat data;
		input.ToUnifiedFormat(count, data);

		if (data.validity.AllValid()) {
			return true;
		}
		for (idx_t i = 0; i < count; i++) {
			auto idx = data.sel->get_index(i);
			if (data.validity.RowIsValid(idx)) {
				return true;
			}
		}
		return false;
	}
}

bool VectorOperations::HasNull(Vector &input, idx_t count) {
	if (count == 0) {
		return false;
	}
	if (input.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		return ConstantVector::IsNull(input);
	} else {
		UnifiedVectorFormat data;
		input.ToUnifiedFormat(count, data);

		if (data.validity.AllValid()) {
			return false;
		}
		for (idx_t i = 0; i < count; i++) {
			auto idx = data.sel->get_index(i);
			if (!data.validity.RowIsValid(idx)) {
				return true;
			}
		}
		return false;
	}
}

idx_t VectorOperations::CountNotNull(Vector &input, const idx_t count) {
	idx_t valid = 0;

	UnifiedVectorFormat vdata;
	input.ToUnifiedFormat(count, vdata);
	if (vdata.validity.AllValid()) {
		return count;
	}
	switch (input.GetVectorType()) {
	case VectorType::FLAT_VECTOR:
		valid += vdata.validity.CountValid(count);
		break;
	case VectorType::CONSTANT_VECTOR:
		valid += vdata.validity.CountValid(1) * count;
		break;
	default:
		for (idx_t i = 0; i < count; ++i) {
			const auto row_idx = vdata.sel->get_index(i);
			valid += int(vdata.validity.RowIsValid(row_idx));
		}
		break;
	}

	return valid;
}

} // namespace duckdb
//===--------------------------------------------------------------------===//
// numeric_inplace_operators.cpp
// Description: This file contains the implementation of numeric inplace ops
// += *= /= -= %=
//===--------------------------------------------------------------------===//



#include <algorithm>

namespace duckdb {

//===--------------------------------------------------------------------===//
// In-Place Addition
//===--------------------------------------------------------------------===//

void VectorOperations::AddInPlace(Vector &input, int64_t right, idx_t count) {
	D_ASSERT(input.GetType().id() == LogicalTypeId::POINTER);
	if (right == 0) {
		return;
	}
	switch (input.GetVectorType()) {
	case VectorType::CONSTANT_VECTOR: {
		D_ASSERT(!ConstantVector::IsNull(input));
		auto data = ConstantVector::GetData<uintptr_t>(input);
		*data += right;
		break;
	}
	default: {
		D_ASSERT(input.GetVectorType() == VectorType::FLAT_VECTOR);
		auto data = FlatVector::GetData<uintptr_t>(input);
		for (idx_t i = 0; i < count; i++) {
			data[i] += right;
		}
		break;
	}
	}
}

} // namespace duckdb






namespace duckdb {

bool VectorOperations::TryCast(CastFunctionSet &set, GetCastFunctionInput &input, Vector &source, Vector &result,
                               idx_t count, string *error_message, bool strict) {
	auto cast_function = set.GetCastFunction(source.GetType(), result.GetType(), input);
	unique_ptr<FunctionLocalState> local_state;
	if (cast_function.init_local_state) {
		local_state = cast_function.init_local_state(*input.context);
	}
	CastParameters parameters(cast_function.cast_data.get(), strict, error_message, local_state.get());
	return cast_function.function(source, result, count, parameters);
}

bool VectorOperations::DefaultTryCast(Vector &source, Vector &result, idx_t count, string *error_message, bool strict) {
	CastFunctionSet set;
	GetCastFunctionInput input;
	return VectorOperations::TryCast(set, input, source, result, count, error_message, strict);
}

void VectorOperations::DefaultCast(Vector &source, Vector &result, idx_t count, bool strict) {
	VectorOperations::DefaultTryCast(source, result, count, nullptr, strict);
}

bool VectorOperations::TryCast(ClientContext &context, Vector &source, Vector &result, idx_t count,
                               string *error_message, bool strict) {
	auto &config = DBConfig::GetConfig(context);
	auto &set = config.GetCastFunctions();
	GetCastFunctionInput get_input(context);
	return VectorOperations::TryCast(set, get_input, source, result, count, error_message, strict);
}

void VectorOperations::Cast(ClientContext &context, Vector &source, Vector &result, idx_t count, bool strict) {
	VectorOperations::TryCast(context, source, result, count, nullptr, strict);
}

} // namespace duckdb
