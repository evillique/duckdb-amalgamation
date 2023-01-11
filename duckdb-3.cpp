// See https://raw.githubusercontent.com/duckdb/duckdb/master/LICENSE for licensing information

#include "duckdb.hpp"
#include "duckdb-internal.hpp"
#ifndef DUCKDB_AMALGAMATION
#error header mismatch
#endif
//===--------------------------------------------------------------------===//
// copy.cpp
// Description: This file contains the implementation of the different copy
// functions
//===--------------------------------------------------------------------===//







namespace duckdb {

template <class T>
static void TemplatedCopy(const Vector &source, const SelectionVector &sel, Vector &target, idx_t source_offset,
                          idx_t target_offset, idx_t copy_count) {
	auto ldata = FlatVector::GetData<T>(source);
	auto tdata = FlatVector::GetData<T>(target);
	for (idx_t i = 0; i < copy_count; i++) {
		auto source_idx = sel.get_index(source_offset + i);
		tdata[target_offset + i] = ldata[source_idx];
	}
}

void VectorOperations::Copy(const Vector &source_p, Vector &target, const SelectionVector &sel_p, idx_t source_count,
                            idx_t source_offset, idx_t target_offset) {
	D_ASSERT(source_offset <= source_count);
	D_ASSERT(source_p.GetType() == target.GetType());
	idx_t copy_count = source_count - source_offset;

	SelectionVector owned_sel;
	const SelectionVector *sel = &sel_p;

	const Vector *source = &source_p;
	bool finished = false;
	while (!finished) {
		switch (source->GetVectorType()) {
		case VectorType::DICTIONARY_VECTOR: {
			// dictionary vector: merge selection vectors
			auto &child = DictionaryVector::Child(*source);
			auto &dict_sel = DictionaryVector::SelVector(*source);
			// merge the selection vectors and verify the child
			auto new_buffer = dict_sel.Slice(*sel, source_count);
			owned_sel.Initialize(new_buffer);
			sel = &owned_sel;
			source = &child;
			break;
		}
		case VectorType::SEQUENCE_VECTOR: {
			int64_t start, increment;
			Vector seq(source->GetType());
			SequenceVector::GetSequence(*source, start, increment);
			VectorOperations::GenerateSequence(seq, source_count, *sel, start, increment);
			VectorOperations::Copy(seq, target, *sel, source_count, source_offset, target_offset);
			return;
		}
		case VectorType::CONSTANT_VECTOR:
			sel = ConstantVector::ZeroSelectionVector(copy_count, owned_sel);
			finished = true;
			break;
		case VectorType::FSST_VECTOR:
			finished = true;
			break;
		case VectorType::FLAT_VECTOR:
			finished = true;
			break;
		default:
			throw NotImplementedException("FIXME unimplemented vector type for VectorOperations::Copy");
		}
	}

	if (copy_count == 0) {
		return;
	}

	// Allow copying of a single value to constant vectors
	const auto target_vector_type = target.GetVectorType();
	if (copy_count == 1 && target_vector_type == VectorType::CONSTANT_VECTOR) {
		target_offset = 0;
		target.SetVectorType(VectorType::FLAT_VECTOR);
	}
	D_ASSERT(target.GetVectorType() == VectorType::FLAT_VECTOR);

	// first copy the nullmask
	auto &tmask = FlatVector::Validity(target);
	if (source->GetVectorType() == VectorType::CONSTANT_VECTOR) {
		const bool valid = !ConstantVector::IsNull(*source);
		for (idx_t i = 0; i < copy_count; i++) {
			tmask.Set(target_offset + i, valid);
		}
	} else {
		const ValidityMask *smask;
		if (source->GetVectorType() == VectorType::FLAT_VECTOR) {
			smask = &(FlatVector::Validity(*source));
		} else if (source->GetVectorType() == VectorType::FSST_VECTOR) {
			smask = &(FSSTVector::Validity(*source));
		} else {
			throw InternalException("Unsupported vector type in vector copy");
		}

		if (smask->IsMaskSet()) {
			for (idx_t i = 0; i < copy_count; i++) {
				auto idx = sel->get_index(source_offset + i);

				if (smask->RowIsValid(idx)) {
					// set valid
					if (!tmask.AllValid()) {
						tmask.SetValidUnsafe(target_offset + i);
					}
				} else {
					// set invalid
					if (tmask.AllValid()) {
						auto init_size = MaxValue<idx_t>(STANDARD_VECTOR_SIZE, target_offset + copy_count);
						tmask.Initialize(init_size);
					}
					tmask.SetInvalidUnsafe(target_offset + i);
				}
			}
		}
	}

	D_ASSERT(sel);

	// For FSST Vectors we decompress instead of copying.
	if (source->GetVectorType() == VectorType::FSST_VECTOR) {
		FSSTVector::DecompressVector(*source, target, source_offset, target_offset, copy_count, sel);
		return;
	}

	// now copy over the data
	switch (source->GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		TemplatedCopy<int8_t>(*source, *sel, target, source_offset, target_offset, copy_count);
		break;
	case PhysicalType::INT16:
		TemplatedCopy<int16_t>(*source, *sel, target, source_offset, target_offset, copy_count);
		break;
	case PhysicalType::INT32:
		TemplatedCopy<int32_t>(*source, *sel, target, source_offset, target_offset, copy_count);
		break;
	case PhysicalType::INT64:
		TemplatedCopy<int64_t>(*source, *sel, target, source_offset, target_offset, copy_count);
		break;
	case PhysicalType::UINT8:
		TemplatedCopy<uint8_t>(*source, *sel, target, source_offset, target_offset, copy_count);
		break;
	case PhysicalType::UINT16:
		TemplatedCopy<uint16_t>(*source, *sel, target, source_offset, target_offset, copy_count);
		break;
	case PhysicalType::UINT32:
		TemplatedCopy<uint32_t>(*source, *sel, target, source_offset, target_offset, copy_count);
		break;
	case PhysicalType::UINT64:
		TemplatedCopy<uint64_t>(*source, *sel, target, source_offset, target_offset, copy_count);
		break;
	case PhysicalType::INT128:
		TemplatedCopy<hugeint_t>(*source, *sel, target, source_offset, target_offset, copy_count);
		break;
	case PhysicalType::FLOAT:
		TemplatedCopy<float>(*source, *sel, target, source_offset, target_offset, copy_count);
		break;
	case PhysicalType::DOUBLE:
		TemplatedCopy<double>(*source, *sel, target, source_offset, target_offset, copy_count);
		break;
	case PhysicalType::INTERVAL:
		TemplatedCopy<interval_t>(*source, *sel, target, source_offset, target_offset, copy_count);
		break;
	case PhysicalType::VARCHAR: {
		auto ldata = FlatVector::GetData<string_t>(*source);
		auto tdata = FlatVector::GetData<string_t>(target);
		for (idx_t i = 0; i < copy_count; i++) {
			auto source_idx = sel->get_index(source_offset + i);
			auto target_idx = target_offset + i;
			if (tmask.RowIsValid(target_idx)) {
				tdata[target_idx] = StringVector::AddStringOrBlob(target, ldata[source_idx]);
			}
		}
		break;
	}
	case PhysicalType::STRUCT: {
		auto &source_children = StructVector::GetEntries(*source);
		auto &target_children = StructVector::GetEntries(target);
		D_ASSERT(source_children.size() == target_children.size());
		for (idx_t i = 0; i < source_children.size(); i++) {
			VectorOperations::Copy(*source_children[i], *target_children[i], sel_p, source_count, source_offset,
			                       target_offset);
		}
		break;
	}
	case PhysicalType::LIST: {
		D_ASSERT(target.GetType().InternalType() == PhysicalType::LIST);

		auto &source_child = ListVector::GetEntry(*source);
		auto sdata = FlatVector::GetData<list_entry_t>(*source);
		auto tdata = FlatVector::GetData<list_entry_t>(target);

		if (target_vector_type == VectorType::CONSTANT_VECTOR) {
			// If we are only writing one value, then the copied values (if any) are contiguous
			// and we can just Append from the offset position
			if (!tmask.RowIsValid(target_offset)) {
				break;
			}
			auto source_idx = sel->get_index(source_offset);
			auto &source_entry = sdata[source_idx];
			const idx_t source_child_size = source_entry.length + source_entry.offset;

			//! overwrite constant target vectors.
			ListVector::SetListSize(target, 0);
			ListVector::Append(target, source_child, source_child_size, source_entry.offset);

			auto &target_entry = tdata[target_offset];
			target_entry.length = source_entry.length;
			target_entry.offset = 0;
		} else {
			//! if the source has list offsets, we need to append them to the target
			//! build a selection vector for the copied child elements
			vector<sel_t> child_rows;
			for (idx_t i = 0; i < copy_count; ++i) {
				if (tmask.RowIsValid(target_offset + i)) {
					auto source_idx = sel->get_index(source_offset + i);
					auto &source_entry = sdata[source_idx];
					for (idx_t j = 0; j < source_entry.length; ++j) {
						child_rows.emplace_back(source_entry.offset + j);
					}
				}
			}
			idx_t source_child_size = child_rows.size();
			SelectionVector child_sel(child_rows.data());

			idx_t old_target_child_len = ListVector::GetListSize(target);

			//! append to list itself
			ListVector::Append(target, source_child, child_sel, source_child_size);

			//! now write the list offsets
			for (idx_t i = 0; i < copy_count; i++) {
				auto source_idx = sel->get_index(source_offset + i);
				auto &source_entry = sdata[source_idx];
				auto &target_entry = tdata[target_offset + i];

				target_entry.length = source_entry.length;
				target_entry.offset = old_target_child_len;
				if (tmask.RowIsValid(target_offset + i)) {
					old_target_child_len += target_entry.length;
				}
			}
		}
		break;
	}
	default:
		throw NotImplementedException("Unimplemented type '%s' for copy!",
		                              TypeIdToString(source->GetType().InternalType()));
	}

	if (target_vector_type != VectorType::FLAT_VECTOR) {
		target.SetVectorType(target_vector_type);
	}
}

void VectorOperations::Copy(const Vector &source, Vector &target, idx_t source_count, idx_t source_offset,
                            idx_t target_offset) {
	VectorOperations::Copy(source, target, *FlatVector::IncrementalSelectionVector(), source_count, source_offset,
	                       target_offset);
}

} // namespace duckdb
//===--------------------------------------------------------------------===//
// hash.cpp
// Description: This file contains the vectorized hash implementations
//===--------------------------------------------------------------------===//







namespace duckdb {

struct HashOp {
	static const hash_t NULL_HASH = 0xbf58476d1ce4e5b9;

	template <class T>
	static inline hash_t Operation(T input, bool is_null) {
		return is_null ? NULL_HASH : duckdb::Hash<T>(input);
	}
};

static inline hash_t CombineHashScalar(hash_t a, hash_t b) {
	return (a * UINT64_C(0xbf58476d1ce4e5b9)) ^ b;
}

template <bool HAS_RSEL, class T>
static inline void TightLoopHash(T *__restrict ldata, hash_t *__restrict result_data, const SelectionVector *rsel,
                                 idx_t count, const SelectionVector *__restrict sel_vector, ValidityMask &mask) {
	if (!mask.AllValid()) {
		for (idx_t i = 0; i < count; i++) {
			auto ridx = HAS_RSEL ? rsel->get_index(i) : i;
			auto idx = sel_vector->get_index(ridx);
			result_data[ridx] = HashOp::Operation(ldata[idx], !mask.RowIsValid(idx));
		}
	} else {
		for (idx_t i = 0; i < count; i++) {
			auto ridx = HAS_RSEL ? rsel->get_index(i) : i;
			auto idx = sel_vector->get_index(ridx);
			result_data[ridx] = duckdb::Hash<T>(ldata[idx]);
		}
	}
}

template <bool HAS_RSEL, class T>
static inline void TemplatedLoopHash(Vector &input, Vector &result, const SelectionVector *rsel, idx_t count) {
	if (input.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);

		auto ldata = ConstantVector::GetData<T>(input);
		auto result_data = ConstantVector::GetData<hash_t>(result);
		*result_data = HashOp::Operation(*ldata, ConstantVector::IsNull(input));
	} else {
		result.SetVectorType(VectorType::FLAT_VECTOR);

		UnifiedVectorFormat idata;
		input.ToUnifiedFormat(count, idata);

		TightLoopHash<HAS_RSEL, T>((T *)idata.data, FlatVector::GetData<hash_t>(result), rsel, count, idata.sel,
		                           idata.validity);
	}
}

template <bool HAS_RSEL, bool FIRST_HASH>
static inline void StructLoopHash(Vector &input, Vector &hashes, const SelectionVector *rsel, idx_t count) {
	auto &children = StructVector::GetEntries(input);

	D_ASSERT(!children.empty());
	idx_t col_no = 0;
	if (HAS_RSEL) {
		if (FIRST_HASH) {
			VectorOperations::Hash(*children[col_no++], hashes, *rsel, count);
		} else {
			VectorOperations::CombineHash(hashes, *children[col_no++], *rsel, count);
		}
		while (col_no < children.size()) {
			VectorOperations::CombineHash(hashes, *children[col_no++], *rsel, count);
		}
	} else {
		if (FIRST_HASH) {
			VectorOperations::Hash(*children[col_no++], hashes, count);
		} else {
			VectorOperations::CombineHash(hashes, *children[col_no++], count);
		}
		while (col_no < children.size()) {
			VectorOperations::CombineHash(hashes, *children[col_no++], count);
		}
	}
}

template <bool HAS_RSEL, bool FIRST_HASH>
static inline void ListLoopHash(Vector &input, Vector &hashes, const SelectionVector *rsel, idx_t count) {
	auto hdata = FlatVector::GetData<hash_t>(hashes);

	UnifiedVectorFormat idata;
	input.ToUnifiedFormat(count, idata);
	const auto ldata = (const list_entry_t *)idata.data;

	// Hash the children into a temporary
	auto &child = ListVector::GetEntry(input);
	const auto child_count = ListVector::GetListSize(input);

	Vector child_hashes(LogicalType::HASH, child_count);
	if (child_count > 0) {
		VectorOperations::Hash(child, child_hashes, child_count);
	}
	auto chdata = FlatVector::GetData<hash_t>(child_hashes);

	// Reduce the number of entries to check to the non-empty ones
	SelectionVector unprocessed(count);
	SelectionVector cursor(HAS_RSEL ? STANDARD_VECTOR_SIZE : count);
	idx_t remaining = 0;
	for (idx_t i = 0; i < count; ++i) {
		const idx_t ridx = HAS_RSEL ? rsel->get_index(i) : i;
		const auto lidx = idata.sel->get_index(ridx);
		const auto &entry = ldata[lidx];
		if (idata.validity.RowIsValid(lidx) && entry.length > 0) {
			unprocessed.set_index(remaining++, ridx);
			cursor.set_index(ridx, entry.offset);
		} else if (FIRST_HASH) {
			hdata[ridx] = HashOp::NULL_HASH;
		}
		// Empty or NULL non-first elements have no effect.
	}

	count = remaining;
	if (count == 0) {
		return;
	}

	// Merge the first position hash into the main hash
	idx_t position = 1;
	if (FIRST_HASH) {
		remaining = 0;
		for (idx_t i = 0; i < count; ++i) {
			const auto ridx = unprocessed.get_index(i);
			const auto cidx = cursor.get_index(ridx);
			hdata[ridx] = chdata[cidx];

			const auto lidx = idata.sel->get_index(ridx);
			const auto &entry = ldata[lidx];
			if (entry.length > position) {
				// Entry still has values to hash
				unprocessed.set_index(remaining++, ridx);
				cursor.set_index(ridx, cidx + 1);
			}
		}
		count = remaining;
		if (count == 0) {
			return;
		}
		++position;
	}

	// Combine the hashes for the remaining positions until there are none left
	for (;; ++position) {
		remaining = 0;
		for (idx_t i = 0; i < count; ++i) {
			const auto ridx = unprocessed.get_index(i);
			const auto cidx = cursor.get_index(ridx);
			hdata[ridx] = CombineHashScalar(hdata[ridx], chdata[cidx]);

			const auto lidx = idata.sel->get_index(ridx);
			const auto &entry = ldata[lidx];
			if (entry.length > position) {
				// Entry still has values to hash
				unprocessed.set_index(remaining++, ridx);
				cursor.set_index(ridx, cidx + 1);
			}
		}

		count = remaining;
		if (count == 0) {
			break;
		}
	}
}

template <bool HAS_RSEL>
static inline void HashTypeSwitch(Vector &input, Vector &result, const SelectionVector *rsel, idx_t count) {
	D_ASSERT(result.GetType().id() == LogicalType::HASH);
	switch (input.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		TemplatedLoopHash<HAS_RSEL, int8_t>(input, result, rsel, count);
		break;
	case PhysicalType::INT16:
		TemplatedLoopHash<HAS_RSEL, int16_t>(input, result, rsel, count);
		break;
	case PhysicalType::INT32:
		TemplatedLoopHash<HAS_RSEL, int32_t>(input, result, rsel, count);
		break;
	case PhysicalType::INT64:
		TemplatedLoopHash<HAS_RSEL, int64_t>(input, result, rsel, count);
		break;
	case PhysicalType::UINT8:
		TemplatedLoopHash<HAS_RSEL, uint8_t>(input, result, rsel, count);
		break;
	case PhysicalType::UINT16:
		TemplatedLoopHash<HAS_RSEL, uint16_t>(input, result, rsel, count);
		break;
	case PhysicalType::UINT32:
		TemplatedLoopHash<HAS_RSEL, uint32_t>(input, result, rsel, count);
		break;
	case PhysicalType::UINT64:
		TemplatedLoopHash<HAS_RSEL, uint64_t>(input, result, rsel, count);
		break;
	case PhysicalType::INT128:
		TemplatedLoopHash<HAS_RSEL, hugeint_t>(input, result, rsel, count);
		break;
	case PhysicalType::FLOAT:
		TemplatedLoopHash<HAS_RSEL, float>(input, result, rsel, count);
		break;
	case PhysicalType::DOUBLE:
		TemplatedLoopHash<HAS_RSEL, double>(input, result, rsel, count);
		break;
	case PhysicalType::INTERVAL:
		TemplatedLoopHash<HAS_RSEL, interval_t>(input, result, rsel, count);
		break;
	case PhysicalType::VARCHAR:
		TemplatedLoopHash<HAS_RSEL, string_t>(input, result, rsel, count);
		break;
	case PhysicalType::STRUCT:
		StructLoopHash<HAS_RSEL, true>(input, result, rsel, count);
		break;
	case PhysicalType::LIST:
		ListLoopHash<HAS_RSEL, true>(input, result, rsel, count);
		break;
	default:
		throw InvalidTypeException(input.GetType(), "Invalid type for hash");
	}
}

void VectorOperations::Hash(Vector &input, Vector &result, idx_t count) {
	HashTypeSwitch<false>(input, result, nullptr, count);
}

void VectorOperations::Hash(Vector &input, Vector &result, const SelectionVector &sel, idx_t count) {
	HashTypeSwitch<true>(input, result, &sel, count);
}

template <bool HAS_RSEL, class T>
static inline void TightLoopCombineHashConstant(T *__restrict ldata, hash_t constant_hash, hash_t *__restrict hash_data,
                                                const SelectionVector *rsel, idx_t count,
                                                const SelectionVector *__restrict sel_vector, ValidityMask &mask) {
	if (!mask.AllValid()) {
		for (idx_t i = 0; i < count; i++) {
			auto ridx = HAS_RSEL ? rsel->get_index(i) : i;
			auto idx = sel_vector->get_index(ridx);
			auto other_hash = HashOp::Operation(ldata[idx], !mask.RowIsValid(idx));
			hash_data[ridx] = CombineHashScalar(constant_hash, other_hash);
		}
	} else {
		for (idx_t i = 0; i < count; i++) {
			auto ridx = HAS_RSEL ? rsel->get_index(i) : i;
			auto idx = sel_vector->get_index(ridx);
			auto other_hash = duckdb::Hash<T>(ldata[idx]);
			hash_data[ridx] = CombineHashScalar(constant_hash, other_hash);
		}
	}
}

template <bool HAS_RSEL, class T>
static inline void TightLoopCombineHash(T *__restrict ldata, hash_t *__restrict hash_data, const SelectionVector *rsel,
                                        idx_t count, const SelectionVector *__restrict sel_vector, ValidityMask &mask) {
	if (!mask.AllValid()) {
		for (idx_t i = 0; i < count; i++) {
			auto ridx = HAS_RSEL ? rsel->get_index(i) : i;
			auto idx = sel_vector->get_index(ridx);
			auto other_hash = HashOp::Operation(ldata[idx], !mask.RowIsValid(idx));
			hash_data[ridx] = CombineHashScalar(hash_data[ridx], other_hash);
		}
	} else {
		for (idx_t i = 0; i < count; i++) {
			auto ridx = HAS_RSEL ? rsel->get_index(i) : i;
			auto idx = sel_vector->get_index(ridx);
			auto other_hash = duckdb::Hash<T>(ldata[idx]);
			hash_data[ridx] = CombineHashScalar(hash_data[ridx], other_hash);
		}
	}
}

template <bool HAS_RSEL, class T>
void TemplatedLoopCombineHash(Vector &input, Vector &hashes, const SelectionVector *rsel, idx_t count) {
	if (input.GetVectorType() == VectorType::CONSTANT_VECTOR && hashes.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		auto ldata = ConstantVector::GetData<T>(input);
		auto hash_data = ConstantVector::GetData<hash_t>(hashes);

		auto other_hash = HashOp::Operation(*ldata, ConstantVector::IsNull(input));
		*hash_data = CombineHashScalar(*hash_data, other_hash);
	} else {
		UnifiedVectorFormat idata;
		input.ToUnifiedFormat(count, idata);
		if (hashes.GetVectorType() == VectorType::CONSTANT_VECTOR) {
			// mix constant with non-constant, first get the constant value
			auto constant_hash = *ConstantVector::GetData<hash_t>(hashes);
			// now re-initialize the hashes vector to an empty flat vector
			hashes.SetVectorType(VectorType::FLAT_VECTOR);
			TightLoopCombineHashConstant<HAS_RSEL, T>((T *)idata.data, constant_hash,
			                                          FlatVector::GetData<hash_t>(hashes), rsel, count, idata.sel,
			                                          idata.validity);
		} else {
			D_ASSERT(hashes.GetVectorType() == VectorType::FLAT_VECTOR);
			TightLoopCombineHash<HAS_RSEL, T>((T *)idata.data, FlatVector::GetData<hash_t>(hashes), rsel, count,
			                                  idata.sel, idata.validity);
		}
	}
}

template <bool HAS_RSEL>
static inline void CombineHashTypeSwitch(Vector &hashes, Vector &input, const SelectionVector *rsel, idx_t count) {
	D_ASSERT(hashes.GetType().id() == LogicalType::HASH);
	switch (input.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		TemplatedLoopCombineHash<HAS_RSEL, int8_t>(input, hashes, rsel, count);
		break;
	case PhysicalType::INT16:
		TemplatedLoopCombineHash<HAS_RSEL, int16_t>(input, hashes, rsel, count);
		break;
	case PhysicalType::INT32:
		TemplatedLoopCombineHash<HAS_RSEL, int32_t>(input, hashes, rsel, count);
		break;
	case PhysicalType::INT64:
		TemplatedLoopCombineHash<HAS_RSEL, int64_t>(input, hashes, rsel, count);
		break;
	case PhysicalType::UINT8:
		TemplatedLoopCombineHash<HAS_RSEL, uint8_t>(input, hashes, rsel, count);
		break;
	case PhysicalType::UINT16:
		TemplatedLoopCombineHash<HAS_RSEL, uint16_t>(input, hashes, rsel, count);
		break;
	case PhysicalType::UINT32:
		TemplatedLoopCombineHash<HAS_RSEL, uint32_t>(input, hashes, rsel, count);
		break;
	case PhysicalType::UINT64:
		TemplatedLoopCombineHash<HAS_RSEL, uint64_t>(input, hashes, rsel, count);
		break;
	case PhysicalType::INT128:
		TemplatedLoopCombineHash<HAS_RSEL, hugeint_t>(input, hashes, rsel, count);
		break;
	case PhysicalType::FLOAT:
		TemplatedLoopCombineHash<HAS_RSEL, float>(input, hashes, rsel, count);
		break;
	case PhysicalType::DOUBLE:
		TemplatedLoopCombineHash<HAS_RSEL, double>(input, hashes, rsel, count);
		break;
	case PhysicalType::INTERVAL:
		TemplatedLoopCombineHash<HAS_RSEL, interval_t>(input, hashes, rsel, count);
		break;
	case PhysicalType::VARCHAR:
		TemplatedLoopCombineHash<HAS_RSEL, string_t>(input, hashes, rsel, count);
		break;
	case PhysicalType::STRUCT:
		StructLoopHash<HAS_RSEL, false>(input, hashes, rsel, count);
		break;
	case PhysicalType::LIST:
		ListLoopHash<HAS_RSEL, false>(input, hashes, rsel, count);
		break;
	default:
		throw InvalidTypeException(input.GetType(), "Invalid type for hash");
	}
}

void VectorOperations::CombineHash(Vector &hashes, Vector &input, idx_t count) {
	CombineHashTypeSwitch<false>(hashes, input, nullptr, count);
}

void VectorOperations::CombineHash(Vector &hashes, Vector &input, const SelectionVector &rsel, idx_t count) {
	CombineHashTypeSwitch<true>(hashes, input, &rsel, count);
}

} // namespace duckdb




namespace duckdb {

template <class T>
static void CopyToStorageLoop(UnifiedVectorFormat &vdata, idx_t count, data_ptr_t target) {
	auto ldata = (T *)vdata.data;
	auto result_data = (T *)target;
	for (idx_t i = 0; i < count; i++) {
		auto idx = vdata.sel->get_index(i);
		if (!vdata.validity.RowIsValid(idx)) {
			result_data[i] = NullValue<T>();
		} else {
			result_data[i] = ldata[idx];
		}
	}
}

void VectorOperations::WriteToStorage(Vector &source, idx_t count, data_ptr_t target) {
	if (count == 0) {
		return;
	}
	UnifiedVectorFormat vdata;
	source.ToUnifiedFormat(count, vdata);

	switch (source.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		CopyToStorageLoop<int8_t>(vdata, count, target);
		break;
	case PhysicalType::INT16:
		CopyToStorageLoop<int16_t>(vdata, count, target);
		break;
	case PhysicalType::INT32:
		CopyToStorageLoop<int32_t>(vdata, count, target);
		break;
	case PhysicalType::INT64:
		CopyToStorageLoop<int64_t>(vdata, count, target);
		break;
	case PhysicalType::UINT8:
		CopyToStorageLoop<uint8_t>(vdata, count, target);
		break;
	case PhysicalType::UINT16:
		CopyToStorageLoop<uint16_t>(vdata, count, target);
		break;
	case PhysicalType::UINT32:
		CopyToStorageLoop<uint32_t>(vdata, count, target);
		break;
	case PhysicalType::UINT64:
		CopyToStorageLoop<uint64_t>(vdata, count, target);
		break;
	case PhysicalType::INT128:
		CopyToStorageLoop<hugeint_t>(vdata, count, target);
		break;
	case PhysicalType::FLOAT:
		CopyToStorageLoop<float>(vdata, count, target);
		break;
	case PhysicalType::DOUBLE:
		CopyToStorageLoop<double>(vdata, count, target);
		break;
	case PhysicalType::INTERVAL:
		CopyToStorageLoop<interval_t>(vdata, count, target);
		break;
	default:
		throw NotImplementedException("Unimplemented type for WriteToStorage");
	}
}

template <class T>
static void ReadFromStorageLoop(data_ptr_t source, idx_t count, Vector &result) {
	auto ldata = (T *)source;
	auto result_data = FlatVector::GetData<T>(result);
	for (idx_t i = 0; i < count; i++) {
		result_data[i] = ldata[i];
	}
}

void VectorOperations::ReadFromStorage(data_ptr_t source, idx_t count, Vector &result) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	switch (result.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		ReadFromStorageLoop<int8_t>(source, count, result);
		break;
	case PhysicalType::INT16:
		ReadFromStorageLoop<int16_t>(source, count, result);
		break;
	case PhysicalType::INT32:
		ReadFromStorageLoop<int32_t>(source, count, result);
		break;
	case PhysicalType::INT64:
		ReadFromStorageLoop<int64_t>(source, count, result);
		break;
	case PhysicalType::UINT8:
		ReadFromStorageLoop<uint8_t>(source, count, result);
		break;
	case PhysicalType::UINT16:
		ReadFromStorageLoop<uint16_t>(source, count, result);
		break;
	case PhysicalType::UINT32:
		ReadFromStorageLoop<uint32_t>(source, count, result);
		break;
	case PhysicalType::UINT64:
		ReadFromStorageLoop<uint64_t>(source, count, result);
		break;
	case PhysicalType::INT128:
		ReadFromStorageLoop<hugeint_t>(source, count, result);
		break;
	case PhysicalType::FLOAT:
		ReadFromStorageLoop<float>(source, count, result);
		break;
	case PhysicalType::DOUBLE:
		ReadFromStorageLoop<double>(source, count, result);
		break;
	case PhysicalType::INTERVAL:
		ReadFromStorageLoop<interval_t>(source, count, result);
		break;
	default:
		throw NotImplementedException("Unimplemented type for ReadFromStorage");
	}
}

} // namespace duckdb






namespace duckdb {

VirtualFileSystem::VirtualFileSystem() : default_fs(FileSystem::CreateLocal()) {
	VirtualFileSystem::RegisterSubSystem(FileCompressionType::GZIP, make_unique<GZipFileSystem>());
}

unique_ptr<FileHandle> VirtualFileSystem::OpenFile(const string &path, uint8_t flags, FileLockType lock,
                                                   FileCompressionType compression, FileOpener *opener) {
	if (compression == FileCompressionType::AUTO_DETECT) {
		// auto detect compression settings based on file name
		auto lower_path = StringUtil::Lower(path);
		if (StringUtil::EndsWith(lower_path, ".gz")) {
			compression = FileCompressionType::GZIP;
		} else if (StringUtil::EndsWith(lower_path, ".zst")) {
			compression = FileCompressionType::ZSTD;
		} else {
			compression = FileCompressionType::UNCOMPRESSED;
		}
	}
	// open the base file handle
	auto file_handle = FindFileSystem(path)->OpenFile(path, flags, lock, FileCompressionType::UNCOMPRESSED, opener);
	if (file_handle->GetType() == FileType::FILE_TYPE_FIFO) {
		file_handle = PipeFileSystem::OpenPipe(std::move(file_handle));
	} else if (compression != FileCompressionType::UNCOMPRESSED) {
		auto entry = compressed_fs.find(compression);
		if (entry == compressed_fs.end()) {
			throw NotImplementedException(
			    "Attempting to open a compressed file, but the compression type is not supported");
		}
		file_handle = entry->second->OpenCompressedFile(std::move(file_handle), flags & FileFlags::FILE_FLAGS_WRITE);
	}
	return file_handle;
}

} // namespace duckdb


namespace duckdb {

#ifdef DUCKDB_WINDOWS

std::wstring WindowsUtil::UTF8ToUnicode(const char *input) {
	idx_t result_size;

	result_size = MultiByteToWideChar(CP_UTF8, 0, input, -1, nullptr, 0);
	if (result_size == 0) {
		throw IOException("Failure in MultiByteToWideChar");
	}
	auto buffer = unique_ptr<wchar_t[]>(new wchar_t[result_size]);
	result_size = MultiByteToWideChar(CP_UTF8, 0, input, -1, buffer.get(), result_size);
	if (result_size == 0) {
		throw IOException("Failure in MultiByteToWideChar");
	}
	return std::wstring(buffer.get(), result_size);
}

static string WideCharToMultiByteWrapper(LPCWSTR input, uint32_t code_page) {
	idx_t result_size;

	result_size = WideCharToMultiByte(code_page, 0, input, -1, 0, 0, 0, 0);
	if (result_size == 0) {
		throw IOException("Failure in WideCharToMultiByte");
	}
	auto buffer = unique_ptr<char[]>(new char[result_size]);
	result_size = WideCharToMultiByte(code_page, 0, input, -1, buffer.get(), result_size, 0, 0);
	if (result_size == 0) {
		throw IOException("Failure in WideCharToMultiByte");
	}
	return string(buffer.get(), result_size - 1);
}

string WindowsUtil::UnicodeToUTF8(LPCWSTR input) {
	return WideCharToMultiByteWrapper(input, CP_UTF8);
}

static string WindowsUnicodeToMBCS(LPCWSTR unicode_text, int use_ansi) {
	uint32_t code_page = use_ansi ? CP_ACP : CP_OEMCP;
	return WideCharToMultiByteWrapper(unicode_text, code_page);
}

string WindowsUtil::UTF8ToMBCS(const char *input, bool use_ansi) {
	auto unicode = WindowsUtil::UTF8ToUnicode(input);
	return WindowsUnicodeToMBCS(unicode.c_str(), use_ansi);
}

#endif

} // namespace duckdb



#include <vector>

namespace duckdb {

AdaptiveFilter::AdaptiveFilter(const Expression &expr)
    : iteration_count(0), observe_interval(10), execute_interval(20), warmup(true) {
	auto &conj_expr = (const BoundConjunctionExpression &)expr;
	D_ASSERT(conj_expr.children.size() > 1);
	for (idx_t idx = 0; idx < conj_expr.children.size(); idx++) {
		permutation.push_back(idx);
		if (idx != conj_expr.children.size() - 1) {
			swap_likeliness.push_back(100);
		}
	}
	right_random_border = 100 * (conj_expr.children.size() - 1);
}

AdaptiveFilter::AdaptiveFilter(TableFilterSet *table_filters)
    : iteration_count(0), observe_interval(10), execute_interval(20), warmup(true) {
	for (auto &table_filter : table_filters->filters) {
		permutation.push_back(table_filter.first);
		swap_likeliness.push_back(100);
	}
	swap_likeliness.pop_back();
	right_random_border = 100 * (table_filters->filters.size() - 1);
}
void AdaptiveFilter::AdaptRuntimeStatistics(double duration) {
	iteration_count++;
	runtime_sum += duration;

	if (!warmup) {
		// the last swap was observed
		if (observe && iteration_count == observe_interval) {
			// keep swap if runtime decreased, else reverse swap
			if (prev_mean - (runtime_sum / iteration_count) <= 0) {
				// reverse swap because runtime didn't decrease
				std::swap(permutation[swap_idx], permutation[swap_idx + 1]);

				// decrease swap likeliness, but make sure there is always a small likeliness left
				if (swap_likeliness[swap_idx] > 1) {
					swap_likeliness[swap_idx] /= 2;
				}
			} else {
				// keep swap because runtime decreased, reset likeliness
				swap_likeliness[swap_idx] = 100;
			}
			observe = false;

			// reset values
			iteration_count = 0;
			runtime_sum = 0.0;
		} else if (!observe && iteration_count == execute_interval) {
			// save old mean to evaluate swap
			prev_mean = runtime_sum / iteration_count;

			// get swap index and swap likeliness
			std::uniform_int_distribution<int> distribution(1, right_random_border); // a <= i <= b
			idx_t random_number = distribution(generator) - 1;

			swap_idx = random_number / 100;                    // index to be swapped
			idx_t likeliness = random_number - 100 * swap_idx; // random number between [0, 100)

			// check if swap is going to happen
			if (swap_likeliness[swap_idx] > likeliness) { // always true for the first swap of an index
				// swap
				std::swap(permutation[swap_idx], permutation[swap_idx + 1]);

				// observe whether swap will be applied
				observe = true;
			}

			// reset values
			iteration_count = 0;
			runtime_sum = 0.0;
		}
	} else {
		if (iteration_count == 5) {
			// initially set all values
			iteration_count = 0;
			runtime_sum = 0.0;
			observe = false;
			warmup = false;
		}
	}
}

} // namespace duckdb














#include <cmath>

namespace duckdb {

using ValidityBytes = RowLayout::ValidityBytes;

GroupedAggregateHashTable::GroupedAggregateHashTable(ClientContext &context, Allocator &allocator,
                                                     vector<LogicalType> group_types, vector<LogicalType> payload_types,
                                                     const vector<BoundAggregateExpression *> &bindings,
                                                     HtEntryType entry_type)
    : GroupedAggregateHashTable(context, allocator, std::move(group_types), std::move(payload_types),
                                AggregateObject::CreateAggregateObjects(bindings), entry_type) {
}

GroupedAggregateHashTable::GroupedAggregateHashTable(ClientContext &context, Allocator &allocator,
                                                     vector<LogicalType> group_types)
    : GroupedAggregateHashTable(context, allocator, std::move(group_types), {}, vector<AggregateObject>()) {
}

GroupedAggregateHashTable::GroupedAggregateHashTable(ClientContext &context, Allocator &allocator,
                                                     vector<LogicalType> group_types_p,
                                                     vector<LogicalType> payload_types_p,
                                                     vector<AggregateObject> aggregate_objects_p,
                                                     HtEntryType entry_type)
    : BaseAggregateHashTable(context, allocator, aggregate_objects_p, std::move(payload_types_p)),
      entry_type(entry_type), capacity(0), entries(0), payload_page_offset(0), is_finalized(false),
      ht_offsets(LogicalTypeId::BIGINT), hash_salts(LogicalTypeId::SMALLINT),
      group_compare_vector(STANDARD_VECTOR_SIZE), no_match_vector(STANDARD_VECTOR_SIZE),
      empty_vector(STANDARD_VECTOR_SIZE) {

	// Append hash column to the end and initialise the row layout
	group_types_p.emplace_back(LogicalType::HASH);
	layout.Initialize(std::move(group_types_p), std::move(aggregate_objects_p));

	// HT layout
	hash_offset = layout.GetOffsets()[layout.ColumnCount() - 1];

	tuple_size = layout.GetRowWidth();

	D_ASSERT(tuple_size <= Storage::BLOCK_SIZE);
	tuples_per_block = Storage::BLOCK_SIZE / tuple_size;
	hashes_hdl = buffer_manager.Allocate(Storage::BLOCK_SIZE);
	hashes_hdl_ptr = hashes_hdl.Ptr();

	switch (entry_type) {
	case HtEntryType::HT_WIDTH_64: {
		hash_prefix_shift = (HASH_WIDTH - sizeof(aggr_ht_entry_64::salt)) * 8;
		Resize<aggr_ht_entry_64>(STANDARD_VECTOR_SIZE * 2L);
		break;
	}
	case HtEntryType::HT_WIDTH_32: {
		hash_prefix_shift = (HASH_WIDTH - sizeof(aggr_ht_entry_32::salt)) * 8;
		Resize<aggr_ht_entry_32>(STANDARD_VECTOR_SIZE * 2L);
		break;
	}
	default:
		throw InternalException("Unknown HT entry width");
	}

	predicates.resize(layout.ColumnCount() - 1, ExpressionType::COMPARE_EQUAL);
	string_heap = make_unique<RowDataCollection>(buffer_manager, (idx_t)Storage::BLOCK_SIZE, 1, true);
}

GroupedAggregateHashTable::~GroupedAggregateHashTable() {
	Destroy();
}

template <class FUNC>
void GroupedAggregateHashTable::PayloadApply(FUNC fun) {
	if (entries == 0) {
		return;
	}
	idx_t apply_entries = entries;
	idx_t page_nr = 0;
	idx_t page_offset = 0;

	for (auto &payload_chunk_ptr : payload_hds_ptrs) {
		auto this_entries = MinValue(tuples_per_block, apply_entries);
		page_offset = 0;
		for (data_ptr_t ptr = payload_chunk_ptr, end = payload_chunk_ptr + this_entries * tuple_size; ptr < end;
		     ptr += tuple_size) {
			fun(page_nr, page_offset++, ptr);
		}
		apply_entries -= this_entries;
		page_nr++;
	}
	D_ASSERT(apply_entries == 0);
}

void GroupedAggregateHashTable::NewBlock() {
	auto pin = buffer_manager.Allocate(Storage::BLOCK_SIZE);
	payload_hds.push_back(std::move(pin));
	payload_hds_ptrs.push_back(payload_hds.back().Ptr());
	payload_page_offset = 0;
}

void GroupedAggregateHashTable::Destroy() {
	// check if there is a destructor
	bool has_destructor = false;
	for (auto &aggr : layout.GetAggregates()) {
		if (aggr.function.destructor) {
			has_destructor = true;
		}
	}
	if (!has_destructor) {
		return;
	}
	// there are aggregates with destructors: loop over the hash table
	// and call the destructor method for each of the aggregates
	data_ptr_t data_pointers[STANDARD_VECTOR_SIZE];
	Vector state_vector(LogicalType::POINTER, (data_ptr_t)data_pointers);
	idx_t count = 0;

	PayloadApply([&](idx_t page_nr, idx_t page_offset, data_ptr_t ptr) {
		data_pointers[count++] = ptr;
		if (count == STANDARD_VECTOR_SIZE) {
			RowOperations::DestroyStates(layout, state_vector, count);
			count = 0;
		}
	});
	RowOperations::DestroyStates(layout, state_vector, count);
}

template <class ENTRY>
void GroupedAggregateHashTable::VerifyInternal() {
	auto hashes_ptr = (ENTRY *)hashes_hdl_ptr;
	D_ASSERT(payload_hds.size() == payload_hds_ptrs.size());
	idx_t count = 0;
	for (idx_t i = 0; i < capacity; i++) {
		if (hashes_ptr[i].page_nr > 0) {
			D_ASSERT(hashes_ptr[i].page_offset < tuples_per_block);
			D_ASSERT(hashes_ptr[i].page_nr <= payload_hds.size());
			auto ptr = payload_hds_ptrs[hashes_ptr[i].page_nr - 1] + ((hashes_ptr[i].page_offset) * tuple_size);
			auto hash = Load<hash_t>(ptr + hash_offset);
			D_ASSERT((hashes_ptr[i].salt) == (hash >> hash_prefix_shift));

			count++;
		}
	}
	(void)count;
	D_ASSERT(count == entries);
}

idx_t GroupedAggregateHashTable::MaxCapacity() {
	idx_t max_pages = 0;
	idx_t max_tuples = 0;

	switch (entry_type) {
	case HtEntryType::HT_WIDTH_32:
		max_pages = NumericLimits<uint8_t>::Maximum();
		max_tuples = NumericLimits<uint16_t>::Maximum();
		break;
	default:
		D_ASSERT(entry_type == HtEntryType::HT_WIDTH_64);
		max_pages = NumericLimits<uint32_t>::Maximum();
		max_tuples = NumericLimits<uint16_t>::Maximum();
		break;
	}

	return max_pages * MinValue(max_tuples, (idx_t)Storage::BLOCK_SIZE / tuple_size);
}

void GroupedAggregateHashTable::Verify() {
#ifdef DEBUG
	switch (entry_type) {
	case HtEntryType::HT_WIDTH_32:
		VerifyInternal<aggr_ht_entry_32>();
		break;
	case HtEntryType::HT_WIDTH_64:
		VerifyInternal<aggr_ht_entry_64>();
		break;
	}
#endif
}

template <class ENTRY>
void GroupedAggregateHashTable::Resize(idx_t size) {
	Verify();

	D_ASSERT(!is_finalized);

	if (size <= capacity) {
		throw InternalException("Cannot downsize a hash table!");
	}
	D_ASSERT(size >= STANDARD_VECTOR_SIZE);

	// size needs to be a power of 2
	D_ASSERT((size & (size - 1)) == 0);
	bitmask = size - 1;

	auto byte_size = size * sizeof(ENTRY);
	if (byte_size > (idx_t)Storage::BLOCK_SIZE) {
		hashes_hdl = buffer_manager.Allocate(byte_size);
		hashes_hdl_ptr = hashes_hdl.Ptr();
	}
	memset(hashes_hdl_ptr, 0, byte_size);
	hashes_end_ptr = hashes_hdl_ptr + byte_size;
	capacity = size;

	auto hashes_arr = (ENTRY *)hashes_hdl_ptr;

	PayloadApply([&](idx_t page_nr, idx_t page_offset, data_ptr_t ptr) {
		auto hash = Load<hash_t>(ptr + hash_offset);
		D_ASSERT((hash & bitmask) == (hash % capacity));
		auto entry_idx = (idx_t)hash & bitmask;
		while (hashes_arr[entry_idx].page_nr > 0) {
			entry_idx++;
			if (entry_idx >= capacity) {
				entry_idx = 0;
			}
		}

		D_ASSERT(!hashes_arr[entry_idx].page_nr);
		D_ASSERT(hash >> hash_prefix_shift <= NumericLimits<uint16_t>::Maximum());

		hashes_arr[entry_idx].salt = hash >> hash_prefix_shift;
		hashes_arr[entry_idx].page_nr = page_nr + 1;
		hashes_arr[entry_idx].page_offset = page_offset;
	});

	Verify();
}

idx_t GroupedAggregateHashTable::AddChunk(DataChunk &groups, DataChunk &payload, AggregateType filter) {
	vector<idx_t> aggregate_filter;

	auto &aggregates = layout.GetAggregates();
	for (idx_t i = 0; i < aggregates.size(); i++) {
		auto &aggregate = aggregates[i];
		if (aggregate.aggr_type == filter) {
			aggregate_filter.push_back(i);
		}
	}
	return AddChunk(groups, payload, aggregate_filter);
}

idx_t GroupedAggregateHashTable::AddChunk(DataChunk &groups, DataChunk &payload, const vector<idx_t> &filter) {
	Vector hashes(LogicalType::HASH);
	groups.Hash(hashes);

	return AddChunk(groups, hashes, payload, filter);
}

idx_t GroupedAggregateHashTable::AddChunk(DataChunk &groups, Vector &group_hashes, DataChunk &payload,
                                          const vector<idx_t> &filter) {
	D_ASSERT(!is_finalized);

	if (groups.size() == 0) {
		return 0;
	}
	// dummy
	SelectionVector new_groups(STANDARD_VECTOR_SIZE);

	D_ASSERT(groups.ColumnCount() + 1 == layout.ColumnCount());
	for (idx_t i = 0; i < groups.ColumnCount(); i++) {
		D_ASSERT(groups.GetTypes()[i] == layout.GetTypes()[i]);
	}

	Vector addresses(LogicalType::POINTER);
	auto new_group_count = FindOrCreateGroups(groups, group_hashes, addresses, new_groups);
	VectorOperations::AddInPlace(addresses, layout.GetAggrOffset(), payload.size());

	// now every cell has an entry
	// update the aggregates
	idx_t payload_idx = 0;

	auto &aggregates = layout.GetAggregates();
	idx_t filter_idx = 0;
	for (idx_t i = 0; i < aggregates.size(); i++) {
		auto &aggr = aggregates[i];
		if (filter_idx >= filter.size() || i < filter[filter_idx]) {
			// Skip all the aggregates that are not in the filter
			payload_idx += aggr.child_count;
			VectorOperations::AddInPlace(addresses, aggr.payload_size, payload.size());
			continue;
		}
		D_ASSERT(i == filter[filter_idx]);

		if (aggr.aggr_type != AggregateType::DISTINCT && aggr.filter) {
			RowOperations::UpdateFilteredStates(filter_set.GetFilterData(i), aggr, addresses, payload, payload_idx);
		} else {
			RowOperations::UpdateStates(aggr, addresses, payload, payload_idx, payload.size());
		}

		// move to the next aggregate
		payload_idx += aggr.child_count;
		VectorOperations::AddInPlace(addresses, aggr.payload_size, payload.size());
		filter_idx++;
	}

	Verify();
	return new_group_count;
}

void GroupedAggregateHashTable::FetchAggregates(DataChunk &groups, DataChunk &result) {
	groups.Verify();
	D_ASSERT(groups.ColumnCount() + 1 == layout.ColumnCount());
	for (idx_t i = 0; i < result.ColumnCount(); i++) {
		D_ASSERT(result.data[i].GetType() == payload_types[i]);
	}
	result.SetCardinality(groups);
	if (groups.size() == 0) {
		return;
	}
	// find the groups associated with the addresses
	// FIXME: this should not use the FindOrCreateGroups, creating them is unnecessary
	Vector addresses(LogicalType::POINTER);
	FindOrCreateGroups(groups, addresses);
	// now fetch the aggregates
	RowOperations::FinalizeStates(layout, addresses, result, 0);
}

template <class ENTRY>
idx_t GroupedAggregateHashTable::FindOrCreateGroupsInternal(DataChunk &groups, Vector &group_hashes, Vector &addresses,
                                                            SelectionVector &new_groups_out) {
	D_ASSERT(!is_finalized);

	if (entries + groups.size() > MaxCapacity()) {
		throw InternalException("Hash table capacity reached");
	}

	// resize at 50% capacity, also need to fit the entire vector
	if (capacity - entries <= groups.size() || entries > capacity / LOAD_FACTOR) {
		Resize<ENTRY>(capacity * 2);
	}

	D_ASSERT(capacity - entries >= groups.size());
	D_ASSERT(groups.ColumnCount() + 1 == layout.ColumnCount());
	// we need to be able to fit at least one vector of data
	D_ASSERT(capacity - entries >= groups.size());
	D_ASSERT(group_hashes.GetType() == LogicalType::HASH);

	group_hashes.Flatten(groups.size());
	auto group_hashes_ptr = FlatVector::GetData<hash_t>(group_hashes);

	D_ASSERT(ht_offsets.GetVectorType() == VectorType::FLAT_VECTOR);
	D_ASSERT(ht_offsets.GetType() == LogicalType::BIGINT);

	D_ASSERT(addresses.GetType() == LogicalType::POINTER);
	addresses.Flatten(groups.size());
	auto addresses_ptr = FlatVector::GetData<data_ptr_t>(addresses);

	// now compute the entry in the table based on the hash using a modulo
	UnaryExecutor::Execute<hash_t, uint64_t>(group_hashes, ht_offsets, groups.size(), [&](hash_t element) {
		D_ASSERT((element & bitmask) == (element % capacity));
		return (element & bitmask);
	});
	auto ht_offsets_ptr = FlatVector::GetData<uint64_t>(ht_offsets);

	// precompute the hash salts for faster comparison below
	D_ASSERT(hash_salts.GetType() == LogicalType::SMALLINT);
	UnaryExecutor::Execute<hash_t, uint16_t>(group_hashes, hash_salts, groups.size(),
	                                         [&](hash_t element) { return (element >> hash_prefix_shift); });
	auto hash_salts_ptr = FlatVector::GetData<uint16_t>(hash_salts);

	// we start out with all entries [0, 1, 2, ..., groups.size()]
	const SelectionVector *sel_vector = FlatVector::IncrementalSelectionVector();

	idx_t remaining_entries = groups.size();

	// make a chunk that references the groups and the hashes
	DataChunk group_chunk;
	group_chunk.InitializeEmpty(layout.GetTypes());
	for (idx_t grp_idx = 0; grp_idx < groups.ColumnCount(); grp_idx++) {
		group_chunk.data[grp_idx].Reference(groups.data[grp_idx]);
	}
	group_chunk.data[groups.ColumnCount()].Reference(group_hashes);
	group_chunk.SetCardinality(groups);

	// convert all vectors to unified format
	auto group_data = group_chunk.ToUnifiedFormat();

	idx_t new_group_count = 0;
	while (remaining_entries > 0) {
		idx_t new_entry_count = 0;
		idx_t need_compare_count = 0;
		idx_t no_match_count = 0;

		// first figure out for each remaining whether or not it belongs to a full or empty group
		for (idx_t i = 0; i < remaining_entries; i++) {
			const idx_t index = sel_vector->get_index(i);
			const auto ht_entry_ptr = ((ENTRY *)this->hashes_hdl_ptr) + ht_offsets_ptr[index];
			if (ht_entry_ptr->page_nr == 0) { // we use page number 0 as a "unused marker"
				// cell is empty; setup the new entry
				if (payload_page_offset == tuples_per_block || payload_hds.empty()) {
					NewBlock();
				}

				auto entry_payload_ptr = payload_hds_ptrs.back() + (payload_page_offset * tuple_size);

				D_ASSERT(group_hashes_ptr[index] >> hash_prefix_shift <= NumericLimits<uint16_t>::Maximum());
				D_ASSERT(payload_page_offset < tuples_per_block);
				D_ASSERT(payload_hds.size() < NumericLimits<uint32_t>::Maximum());
				D_ASSERT(payload_page_offset + 1 < NumericLimits<uint16_t>::Maximum());

				ht_entry_ptr->salt = group_hashes_ptr[index] >> hash_prefix_shift;

				// page numbers start at one so we can use 0 as empty flag
				// GetPtr undoes this
				ht_entry_ptr->page_nr = payload_hds.size();
				ht_entry_ptr->page_offset = payload_page_offset++;

				// update selection lists for outer loops
				empty_vector.set_index(new_entry_count++, index);
				new_groups_out.set_index(new_group_count++, index);
				entries++;

				addresses_ptr[index] = entry_payload_ptr;

			} else {
				// cell is occupied: add to check list
				// only need to check if hash salt in ptr == prefix of hash in payload
				if (ht_entry_ptr->salt == hash_salts_ptr[index]) {
					group_compare_vector.set_index(need_compare_count++, index);

					auto page_ptr = payload_hds_ptrs[ht_entry_ptr->page_nr - 1];
					auto page_offset = ht_entry_ptr->page_offset * tuple_size;
					addresses_ptr[index] = page_ptr + page_offset;

				} else {
					no_match_vector.set_index(no_match_count++, index);
				}
			}
		}

		// for each of the locations that are empty, serialize the group columns to the locations
		RowOperations::Scatter(group_chunk, group_data.get(), layout, addresses, *string_heap, empty_vector,
		                       new_entry_count);
		RowOperations::InitializeStates(layout, addresses, empty_vector, new_entry_count);

		// now we have only the tuples remaining that might match to an existing group
		// start performing comparisons with each of the groups
		RowOperations::Match(group_chunk, group_data.get(), layout, addresses, predicates, group_compare_vector,
		                     need_compare_count, &no_match_vector, no_match_count);

		// each of the entries that do not match we move them to the next entry in the HT
		for (idx_t i = 0; i < no_match_count; i++) {
			idx_t index = no_match_vector.get_index(i);
			ht_offsets_ptr[index]++;
			if (ht_offsets_ptr[index] >= capacity) {
				ht_offsets_ptr[index] = 0;
			}
		}
		sel_vector = &no_match_vector;
		remaining_entries = no_match_count;
	}

	return new_group_count;
}

// this is to support distinct aggregations where we need to record whether we
// have already seen a value for a group
idx_t GroupedAggregateHashTable::FindOrCreateGroups(DataChunk &groups, Vector &group_hashes, Vector &addresses_out,
                                                    SelectionVector &new_groups_out) {
	switch (entry_type) {
	case HtEntryType::HT_WIDTH_64:
		return FindOrCreateGroupsInternal<aggr_ht_entry_64>(groups, group_hashes, addresses_out, new_groups_out);
	case HtEntryType::HT_WIDTH_32:
		return FindOrCreateGroupsInternal<aggr_ht_entry_32>(groups, group_hashes, addresses_out, new_groups_out);
	default:
		throw InternalException("Unknown HT entry width");
	}
}

void GroupedAggregateHashTable::FindOrCreateGroups(DataChunk &groups, Vector &addresses) {
	// create a dummy new_groups sel vector
	SelectionVector new_groups(STANDARD_VECTOR_SIZE);
	FindOrCreateGroups(groups, addresses, new_groups);
}

idx_t GroupedAggregateHashTable::FindOrCreateGroups(DataChunk &groups, Vector &addresses_out,
                                                    SelectionVector &new_groups_out) {
	Vector hashes(LogicalType::HASH);
	groups.Hash(hashes);
	return FindOrCreateGroups(groups, hashes, addresses_out, new_groups_out);
}

struct FlushMoveState {
	FlushMoveState(Allocator &allocator, RowLayout &layout)
	    : new_groups(STANDARD_VECTOR_SIZE), group_addresses(LogicalType::POINTER),
	      new_groups_sel(STANDARD_VECTOR_SIZE) {
		vector<LogicalType> group_types(layout.GetTypes().begin(), layout.GetTypes().end() - 1);
		groups.Initialize(allocator, group_types);
	}

	DataChunk groups;
	SelectionVector new_groups;
	Vector group_addresses;
	SelectionVector new_groups_sel;
};

void GroupedAggregateHashTable::FlushMove(FlushMoveState &state, Vector &source_addresses, Vector &source_hashes,
                                          idx_t count) {
	D_ASSERT(source_addresses.GetType() == LogicalType::POINTER);
	D_ASSERT(source_hashes.GetType() == LogicalType::HASH);

	state.groups.Reset();
	state.groups.SetCardinality(count);
	for (idx_t col_no = 0; col_no < state.groups.ColumnCount(); col_no++) {
		auto &column = state.groups.data[col_no];
		RowOperations::Gather(source_addresses, *FlatVector::IncrementalSelectionVector(), column,
		                      *FlatVector::IncrementalSelectionVector(), count, layout, col_no);
	}

	FindOrCreateGroups(state.groups, source_hashes, state.group_addresses, state.new_groups_sel);

	RowOperations::CombineStates(layout, source_addresses, state.group_addresses, count);
}

void GroupedAggregateHashTable::Combine(GroupedAggregateHashTable &other) {
	D_ASSERT(!is_finalized);

	D_ASSERT(other.layout.GetAggrWidth() == layout.GetAggrWidth());
	D_ASSERT(other.layout.GetDataWidth() == layout.GetDataWidth());
	D_ASSERT(other.layout.GetRowWidth() == layout.GetRowWidth());
	D_ASSERT(other.tuples_per_block == tuples_per_block);

	if (other.entries == 0) {
		return;
	}

	Vector addresses(LogicalType::POINTER);
	auto addresses_ptr = FlatVector::GetData<data_ptr_t>(addresses);

	Vector hashes(LogicalType::HASH);
	auto hashes_ptr = FlatVector::GetData<hash_t>(hashes);

	idx_t group_idx = 0;

	FlushMoveState state(allocator, layout);
	other.PayloadApply([&](idx_t page_nr, idx_t page_offset, data_ptr_t ptr) {
		auto hash = Load<hash_t>(ptr + hash_offset);

		hashes_ptr[group_idx] = hash;
		addresses_ptr[group_idx] = ptr;
		group_idx++;
		if (group_idx == STANDARD_VECTOR_SIZE) {
			FlushMove(state, addresses, hashes, group_idx);
			group_idx = 0;
		}
	});
	FlushMove(state, addresses, hashes, group_idx);
	string_heap->Merge(*other.string_heap);
	Verify();
}

struct PartitionInfo {
	PartitionInfo() : addresses(LogicalType::POINTER), hashes(LogicalType::HASH), group_count(0) {
		addresses_ptr = FlatVector::GetData<data_ptr_t>(addresses);
		hashes_ptr = FlatVector::GetData<hash_t>(hashes);
	};
	Vector addresses;
	Vector hashes;
	idx_t group_count;
	data_ptr_t *addresses_ptr;
	hash_t *hashes_ptr;
};

void GroupedAggregateHashTable::Partition(vector<GroupedAggregateHashTable *> &partition_hts, hash_t mask,
                                          idx_t shift) {
	D_ASSERT(partition_hts.size() > 1);
	vector<PartitionInfo> partition_info(partition_hts.size());

	FlushMoveState state(allocator, layout);
	PayloadApply([&](idx_t page_nr, idx_t page_offset, data_ptr_t ptr) {
		auto hash = Load<hash_t>(ptr + hash_offset);

		idx_t partition = (hash & mask) >> shift;
		D_ASSERT(partition < partition_hts.size());

		auto &info = partition_info[partition];

		info.hashes_ptr[info.group_count] = hash;
		info.addresses_ptr[info.group_count] = ptr;
		info.group_count++;
		if (info.group_count == STANDARD_VECTOR_SIZE) {
			D_ASSERT(partition_hts[partition]);
			partition_hts[partition]->FlushMove(state, info.addresses, info.hashes, info.group_count);
			info.group_count = 0;
		}
	});

	idx_t info_idx = 0;
	idx_t total_count = 0;
	for (auto &partition_entry : partition_hts) {
		auto &info = partition_info[info_idx++];
		partition_entry->FlushMove(state, info.addresses, info.hashes, info.group_count);

		partition_entry->string_heap->Merge(*string_heap);
		partition_entry->Verify();
		total_count += partition_entry->Size();
	}
	(void)total_count;
	D_ASSERT(total_count == entries);
}

idx_t GroupedAggregateHashTable::Scan(AggregateHTScanState &scan_state, DataChunk &result) {
	idx_t this_n;
	Vector addresses(LogicalType::POINTER);
	auto data_pointers = FlatVector::GetData<data_ptr_t>(addresses);
	{
		lock_guard<mutex> l(scan_state.lock);
		if (scan_state.scan_position >= entries) {
			return 0;
		}
		auto remaining = entries - scan_state.scan_position;
		this_n = MinValue((idx_t)STANDARD_VECTOR_SIZE, remaining);

		auto chunk_idx = scan_state.scan_position / tuples_per_block;
		auto chunk_offset = (scan_state.scan_position % tuples_per_block) * tuple_size;
		D_ASSERT(chunk_offset + tuple_size <= Storage::BLOCK_SIZE);

		auto read_ptr = payload_hds_ptrs[chunk_idx++];
		for (idx_t i = 0; i < this_n; i++) {
			data_pointers[i] = read_ptr + chunk_offset;
			chunk_offset += tuple_size;
			if (chunk_offset >= tuples_per_block * tuple_size) {
				read_ptr = payload_hds_ptrs[chunk_idx++];
				chunk_offset = 0;
			}
		}
		scan_state.scan_position += this_n;
	}

	result.SetCardinality(this_n);
	// fetch the group columns (ignoring the final hash column
	const auto group_cols = layout.ColumnCount() - 1;
	for (idx_t col_no = 0; col_no < group_cols; col_no++) {
		auto &column = result.data[col_no];
		RowOperations::Gather(addresses, *FlatVector::IncrementalSelectionVector(), column,
		                      *FlatVector::IncrementalSelectionVector(), result.size(), layout, col_no);
	}

	RowOperations::FinalizeStates(layout, addresses, result, group_cols);
	return this_n;
}

void GroupedAggregateHashTable::Finalize() {
	if (is_finalized) {
		return;
	}

	// early release hashes, not needed for partition/scan
	hashes_hdl.Destroy();
	is_finalized = true;
}

} // namespace duckdb




namespace duckdb {

BaseAggregateHashTable::BaseAggregateHashTable(ClientContext &context, Allocator &allocator,
                                               const vector<AggregateObject> &aggregates,
                                               vector<LogicalType> payload_types_p)
    : allocator(allocator), buffer_manager(BufferManager::GetBufferManager(context)),
      payload_types(std::move(payload_types_p)) {
	filter_set.Initialize(context, aggregates, payload_types);
}

} // namespace duckdb













namespace duckdb {

ColumnBindingResolver::ColumnBindingResolver() {
}

void ColumnBindingResolver::VisitOperator(LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN || op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
		// special case: comparison join
		auto &comp_join = (LogicalComparisonJoin &)op;
		// first get the bindings of the LHS and resolve the LHS expressions
		VisitOperator(*comp_join.children[0]);
		for (auto &cond : comp_join.conditions) {
			VisitExpression(&cond.left);
		}
		if (op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
			// visit the duplicate eliminated columns on the LHS, if any
			auto &delim_join = (LogicalDelimJoin &)op;
			for (auto &expr : delim_join.duplicate_eliminated_columns) {
				VisitExpression(&expr);
			}
		}
		// then get the bindings of the RHS and resolve the RHS expressions
		VisitOperator(*comp_join.children[1]);
		for (auto &cond : comp_join.conditions) {
			VisitExpression(&cond.right);
		}
		// finally update the bindings with the result bindings of the join
		bindings = op.GetColumnBindings();
		return;
	} else if (op.type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
		// ANY join, this join is different because we evaluate the expression on the bindings of BOTH join sides at
		// once i.e. we set the bindings first to the bindings of the entire join, and then resolve the expressions of
		// this operator
		VisitOperatorChildren(op);
		bindings = op.GetColumnBindings();
		VisitOperatorExpressions(op);
		return;
	} else if (op.type == LogicalOperatorType::LOGICAL_CREATE_INDEX) {
		// CREATE INDEX statement, add the columns of the table with table index 0 to the binding set
		// afterwards bind the expressions of the CREATE INDEX statement
		auto &create_index = (LogicalCreateIndex &)op;
		bindings = LogicalOperator::GenerateColumnBindings(0, create_index.table.GetColumns().LogicalColumnCount());
		VisitOperatorExpressions(op);
		return;
	} else if (op.type == LogicalOperatorType::LOGICAL_GET) {
		//! We first need to update the current set of bindings and then visit operator expressions
		bindings = op.GetColumnBindings();
		VisitOperatorExpressions(op);
		return;
	} else if (op.type == LogicalOperatorType::LOGICAL_INSERT) {
		//! We want to execute the normal path, but also add a dummy 'excluded' binding if there is a
		// ON CONFLICT DO UPDATE clause
		auto &insert_op = (LogicalInsert &)op;
		if (insert_op.action_type != OnConflictAction::THROW) {
			VisitOperatorChildren(op);
			auto dummy_bindings = LogicalOperator::GenerateColumnBindings(
			    insert_op.excluded_table_index, insert_op.table->GetColumns().PhysicalColumnCount());
			bindings.insert(bindings.begin(), dummy_bindings.begin(), dummy_bindings.end());
			if (insert_op.on_conflict_condition) {
				VisitExpression(&insert_op.on_conflict_condition);
			}
			if (insert_op.do_update_condition) {
				VisitExpression(&insert_op.do_update_condition);
			}
			VisitOperatorExpressions(op);
			bindings = op.GetColumnBindings();
			return;
		}
	}
	// general case
	// first visit the children of this operator
	VisitOperatorChildren(op);
	// now visit the expressions of this operator to resolve any bound column references
	VisitOperatorExpressions(op);
	// finally update the current set of bindings to the current set of column bindings
	bindings = op.GetColumnBindings();
}

unique_ptr<Expression> ColumnBindingResolver::VisitReplace(BoundColumnRefExpression &expr,
                                                           unique_ptr<Expression> *expr_ptr) {
	D_ASSERT(expr.depth == 0);
	// check the current set of column bindings to see which index corresponds to the column reference
	for (idx_t i = 0; i < bindings.size(); i++) {
		if (expr.binding == bindings[i]) {
			return make_unique<BoundReferenceExpression>(expr.alias, expr.return_type, i);
		}
	}
	// LCOV_EXCL_START
	// could not bind the column reference, this should never happen and indicates a bug in the code
	// generate an error message
	string bound_columns = "[";
	for (idx_t i = 0; i < bindings.size(); i++) {
		if (i != 0) {
			bound_columns += " ";
		}
		bound_columns += to_string(bindings[i].table_index) + "." + to_string(bindings[i].column_index);
	}
	bound_columns += "]";

	throw InternalException("Failed to bind column reference \"%s\" [%d.%d] (bindings: %s)", expr.alias,
	                        expr.binding.table_index, expr.binding.column_index, bound_columns);
	// LCOV_EXCL_STOP
}

unordered_set<idx_t> ColumnBindingResolver::VerifyInternal(LogicalOperator &op) {
	unordered_set<idx_t> result;
	for (auto &child : op.children) {
		auto child_indexes = VerifyInternal(*child);
		for (auto index : child_indexes) {
			D_ASSERT(index != DConstants::INVALID_INDEX);
			if (result.find(index) != result.end()) {
				throw InternalException("Duplicate table index \"%lld\" found", index);
			}
			result.insert(index);
		}
	}
	auto indexes = op.GetTableIndex();
	for (auto index : indexes) {
		D_ASSERT(index != DConstants::INVALID_INDEX);
		if (result.find(index) != result.end()) {
			throw InternalException("Duplicate table index \"%lld\" found", index);
		}
		result.insert(index);
	}
	return result;
}

void ColumnBindingResolver::Verify(LogicalOperator &op) {
#ifdef DEBUG
	VerifyInternal(op);
#endif
}

} // namespace duckdb






namespace duckdb {

struct BothInclusiveBetweenOperator {
	template <class T>
	static inline bool Operation(T input, T lower, T upper) {
		return GreaterThanEquals::Operation<T>(input, lower) && LessThanEquals::Operation<T>(input, upper);
	}
};

struct LowerInclusiveBetweenOperator {
	template <class T>
	static inline bool Operation(T input, T lower, T upper) {
		return GreaterThanEquals::Operation<T>(input, lower) && LessThan::Operation<T>(input, upper);
	}
};

struct UpperInclusiveBetweenOperator {
	template <class T>
	static inline bool Operation(T input, T lower, T upper) {
		return GreaterThan::Operation<T>(input, lower) && LessThanEquals::Operation<T>(input, upper);
	}
};

struct ExclusiveBetweenOperator {
	template <class T>
	static inline bool Operation(T input, T lower, T upper) {
		return GreaterThan::Operation<T>(input, lower) && LessThan::Operation<T>(input, upper);
	}
};

template <class OP>
static idx_t BetweenLoopTypeSwitch(Vector &input, Vector &lower, Vector &upper, const SelectionVector *sel, idx_t count,
                                   SelectionVector *true_sel, SelectionVector *false_sel) {
	switch (input.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		return TernaryExecutor::Select<int8_t, int8_t, int8_t, OP>(input, lower, upper, sel, count, true_sel,
		                                                           false_sel);
	case PhysicalType::INT16:
		return TernaryExecutor::Select<int16_t, int16_t, int16_t, OP>(input, lower, upper, sel, count, true_sel,
		                                                              false_sel);
	case PhysicalType::INT32:
		return TernaryExecutor::Select<int32_t, int32_t, int32_t, OP>(input, lower, upper, sel, count, true_sel,
		                                                              false_sel);
	case PhysicalType::INT64:
		return TernaryExecutor::Select<int64_t, int64_t, int64_t, OP>(input, lower, upper, sel, count, true_sel,
		                                                              false_sel);
	case PhysicalType::INT128:
		return TernaryExecutor::Select<hugeint_t, hugeint_t, hugeint_t, OP>(input, lower, upper, sel, count, true_sel,
		                                                                    false_sel);
	case PhysicalType::UINT8:
		return TernaryExecutor::Select<uint8_t, uint8_t, uint8_t, OP>(input, lower, upper, sel, count, true_sel,
		                                                              false_sel);
	case PhysicalType::UINT16:
		return TernaryExecutor::Select<uint16_t, uint16_t, uint16_t, OP>(input, lower, upper, sel, count, true_sel,
		                                                                 false_sel);
	case PhysicalType::UINT32:
		return TernaryExecutor::Select<uint32_t, uint32_t, uint32_t, OP>(input, lower, upper, sel, count, true_sel,
		                                                                 false_sel);
	case PhysicalType::UINT64:
		return TernaryExecutor::Select<uint64_t, uint64_t, uint64_t, OP>(input, lower, upper, sel, count, true_sel,
		                                                                 false_sel);
	case PhysicalType::FLOAT:
		return TernaryExecutor::Select<float, float, float, OP>(input, lower, upper, sel, count, true_sel, false_sel);
	case PhysicalType::DOUBLE:
		return TernaryExecutor::Select<double, double, double, OP>(input, lower, upper, sel, count, true_sel,
		                                                           false_sel);
	case PhysicalType::VARCHAR:
		return TernaryExecutor::Select<string_t, string_t, string_t, OP>(input, lower, upper, sel, count, true_sel,
		                                                                 false_sel);
	default:
		throw InvalidTypeException(input.GetType(), "Invalid type for BETWEEN");
	}
}

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundBetweenExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_unique<ExpressionState>(expr, root);
	result->AddChild(expr.input.get());
	result->AddChild(expr.lower.get());
	result->AddChild(expr.upper.get());
	result->Finalize();
	return result;
}

void ExpressionExecutor::Execute(const BoundBetweenExpression &expr, ExpressionState *state, const SelectionVector *sel,
                                 idx_t count, Vector &result) {
	// resolve the children
	state->intermediate_chunk.Reset();

	auto &input = state->intermediate_chunk.data[0];
	auto &lower = state->intermediate_chunk.data[1];
	auto &upper = state->intermediate_chunk.data[2];

	Execute(*expr.input, state->child_states[0].get(), sel, count, input);
	Execute(*expr.lower, state->child_states[1].get(), sel, count, lower);
	Execute(*expr.upper, state->child_states[2].get(), sel, count, upper);

	Vector intermediate1(LogicalType::BOOLEAN);
	Vector intermediate2(LogicalType::BOOLEAN);

	if (expr.upper_inclusive && expr.lower_inclusive) {
		VectorOperations::GreaterThanEquals(input, lower, intermediate1, count);
		VectorOperations::LessThanEquals(input, upper, intermediate2, count);
	} else if (expr.lower_inclusive) {
		VectorOperations::GreaterThanEquals(input, lower, intermediate1, count);
		VectorOperations::LessThan(input, upper, intermediate2, count);
	} else if (expr.upper_inclusive) {
		VectorOperations::GreaterThan(input, lower, intermediate1, count);
		VectorOperations::LessThanEquals(input, upper, intermediate2, count);
	} else {
		VectorOperations::GreaterThan(input, lower, intermediate1, count);
		VectorOperations::LessThan(input, upper, intermediate2, count);
	}
	VectorOperations::And(intermediate1, intermediate2, result, count);
}

idx_t ExpressionExecutor::Select(const BoundBetweenExpression &expr, ExpressionState *state, const SelectionVector *sel,
                                 idx_t count, SelectionVector *true_sel, SelectionVector *false_sel) {
	// resolve the children
	Vector input(state->intermediate_chunk.data[0]);
	Vector lower(state->intermediate_chunk.data[1]);
	Vector upper(state->intermediate_chunk.data[2]);

	Execute(*expr.input, state->child_states[0].get(), sel, count, input);
	Execute(*expr.lower, state->child_states[1].get(), sel, count, lower);
	Execute(*expr.upper, state->child_states[2].get(), sel, count, upper);

	if (expr.upper_inclusive && expr.lower_inclusive) {
		return BetweenLoopTypeSwitch<BothInclusiveBetweenOperator>(input, lower, upper, sel, count, true_sel,
		                                                           false_sel);
	} else if (expr.lower_inclusive) {
		return BetweenLoopTypeSwitch<LowerInclusiveBetweenOperator>(input, lower, upper, sel, count, true_sel,
		                                                            false_sel);
	} else if (expr.upper_inclusive) {
		return BetweenLoopTypeSwitch<UpperInclusiveBetweenOperator>(input, lower, upper, sel, count, true_sel,
		                                                            false_sel);
	} else {
		return BetweenLoopTypeSwitch<ExclusiveBetweenOperator>(input, lower, upper, sel, count, true_sel, false_sel);
	}
}

} // namespace duckdb




namespace duckdb {

struct CaseExpressionState : public ExpressionState {
	CaseExpressionState(const Expression &expr, ExpressionExecutorState &root)
	    : ExpressionState(expr, root), true_sel(STANDARD_VECTOR_SIZE), false_sel(STANDARD_VECTOR_SIZE) {
	}

	SelectionVector true_sel;
	SelectionVector false_sel;
};

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundCaseExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_unique<CaseExpressionState>(expr, root);
	for (auto &case_check : expr.case_checks) {
		result->AddChild(case_check.when_expr.get());
		result->AddChild(case_check.then_expr.get());
	}
	result->AddChild(expr.else_expr.get());
	result->Finalize();
	return std::move(result);
}

void ExpressionExecutor::Execute(const BoundCaseExpression &expr, ExpressionState *state_p, const SelectionVector *sel,
                                 idx_t count, Vector &result) {
	auto state = (CaseExpressionState *)state_p;

	state->intermediate_chunk.Reset();

	// first execute the check expression
	auto current_true_sel = &state->true_sel;
	auto current_false_sel = &state->false_sel;
	auto current_sel = sel;
	idx_t current_count = count;
	for (idx_t i = 0; i < expr.case_checks.size(); i++) {
		auto &case_check = expr.case_checks[i];
		auto &intermediate_result = state->intermediate_chunk.data[i * 2 + 1];
		auto check_state = state->child_states[i * 2].get();
		auto then_state = state->child_states[i * 2 + 1].get();

		idx_t tcount =
		    Select(*case_check.when_expr, check_state, current_sel, current_count, current_true_sel, current_false_sel);
		if (tcount == 0) {
			// everything is false: do nothing
			continue;
		}
		idx_t fcount = current_count - tcount;
		if (fcount == 0 && current_count == count) {
			// everything is true in the first CHECK statement
			// we can skip the entire case and only execute the TRUE side
			Execute(*case_check.then_expr, then_state, sel, count, result);
			return;
		} else {
			// we need to execute and then fill in the desired tuples in the result
			Execute(*case_check.then_expr, then_state, current_true_sel, tcount, intermediate_result);
			FillSwitch(intermediate_result, result, *current_true_sel, tcount);
		}
		// continue with the false tuples
		current_sel = current_false_sel;
		current_count = fcount;
		if (fcount == 0) {
			// everything is true: we are done
			break;
		}
	}
	if (current_count > 0) {
		auto else_state = state->child_states.back().get();
		if (current_count == count) {
			// everything was false, we can just evaluate the else expression directly
			Execute(*expr.else_expr, else_state, sel, count, result);
			return;
		} else {
			auto &intermediate_result = state->intermediate_chunk.data[expr.case_checks.size() * 2];

			D_ASSERT(current_sel);
			Execute(*expr.else_expr, else_state, current_sel, current_count, intermediate_result);
			FillSwitch(intermediate_result, result, *current_sel, current_count);
		}
	}
	if (sel) {
		result.Slice(*sel, count);
	}
}

template <class T>
void TemplatedFillLoop(Vector &vector, Vector &result, const SelectionVector &sel, sel_t count) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto res = FlatVector::GetData<T>(result);
	auto &result_mask = FlatVector::Validity(result);
	if (vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		auto data = ConstantVector::GetData<T>(vector);
		if (ConstantVector::IsNull(vector)) {
			for (idx_t i = 0; i < count; i++) {
				result_mask.SetInvalid(sel.get_index(i));
			}
		} else {
			for (idx_t i = 0; i < count; i++) {
				res[sel.get_index(i)] = *data;
			}
		}
	} else {
		UnifiedVectorFormat vdata;
		vector.ToUnifiedFormat(count, vdata);
		auto data = (T *)vdata.data;
		for (idx_t i = 0; i < count; i++) {
			auto source_idx = vdata.sel->get_index(i);
			auto res_idx = sel.get_index(i);

			res[res_idx] = data[source_idx];
			result_mask.Set(res_idx, vdata.validity.RowIsValid(source_idx));
		}
	}
}

void ValidityFillLoop(Vector &vector, Vector &result, const SelectionVector &sel, sel_t count) {
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &result_mask = FlatVector::Validity(result);
	if (vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		if (ConstantVector::IsNull(vector)) {
			for (idx_t i = 0; i < count; i++) {
				result_mask.SetInvalid(sel.get_index(i));
			}
		}
	} else {
		UnifiedVectorFormat vdata;
		vector.ToUnifiedFormat(count, vdata);
		if (vdata.validity.AllValid()) {
			return;
		}
		for (idx_t i = 0; i < count; i++) {
			auto source_idx = vdata.sel->get_index(i);
			if (!vdata.validity.RowIsValid(source_idx)) {
				result_mask.SetInvalid(sel.get_index(i));
			}
		}
	}
}

void ExpressionExecutor::FillSwitch(Vector &vector, Vector &result, const SelectionVector &sel, sel_t count) {
	switch (result.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		TemplatedFillLoop<int8_t>(vector, result, sel, count);
		break;
	case PhysicalType::INT16:
		TemplatedFillLoop<int16_t>(vector, result, sel, count);
		break;
	case PhysicalType::INT32:
		TemplatedFillLoop<int32_t>(vector, result, sel, count);
		break;
	case PhysicalType::INT64:
		TemplatedFillLoop<int64_t>(vector, result, sel, count);
		break;
	case PhysicalType::UINT8:
		TemplatedFillLoop<uint8_t>(vector, result, sel, count);
		break;
	case PhysicalType::UINT16:
		TemplatedFillLoop<uint16_t>(vector, result, sel, count);
		break;
	case PhysicalType::UINT32:
		TemplatedFillLoop<uint32_t>(vector, result, sel, count);
		break;
	case PhysicalType::UINT64:
		TemplatedFillLoop<uint64_t>(vector, result, sel, count);
		break;
	case PhysicalType::INT128:
		TemplatedFillLoop<hugeint_t>(vector, result, sel, count);
		break;
	case PhysicalType::FLOAT:
		TemplatedFillLoop<float>(vector, result, sel, count);
		break;
	case PhysicalType::DOUBLE:
		TemplatedFillLoop<double>(vector, result, sel, count);
		break;
	case PhysicalType::INTERVAL:
		TemplatedFillLoop<interval_t>(vector, result, sel, count);
		break;
	case PhysicalType::VARCHAR:
		TemplatedFillLoop<string_t>(vector, result, sel, count);
		StringVector::AddHeapReference(result, vector);
		break;
	case PhysicalType::STRUCT: {
		auto &vector_entries = StructVector::GetEntries(vector);
		auto &result_entries = StructVector::GetEntries(result);
		ValidityFillLoop(vector, result, sel, count);
		D_ASSERT(vector_entries.size() == result_entries.size());
		for (idx_t i = 0; i < vector_entries.size(); i++) {
			FillSwitch(*vector_entries[i], *result_entries[i], sel, count);
		}
		break;
	}
	case PhysicalType::LIST: {
		idx_t offset = ListVector::GetListSize(result);
		auto &list_child = ListVector::GetEntry(vector);
		ListVector::Append(result, list_child, ListVector::GetListSize(vector));

		// all the false offsets need to be incremented by true_child.count
		TemplatedFillLoop<list_entry_t>(vector, result, sel, count);
		if (offset == 0) {
			break;
		}

		auto result_data = FlatVector::GetData<list_entry_t>(result);
		for (idx_t i = 0; i < count; i++) {
			auto result_idx = sel.get_index(i);
			result_data[result_idx].offset += offset;
		}

		Vector::Verify(result, sel, count);
		break;
	}
	default:
		throw NotImplementedException("Unimplemented type for case expression: %s", result.GetType().ToString());
	}
}

} // namespace duckdb





namespace duckdb {

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundCastExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_unique<ExecuteFunctionState>(expr, root);
	result->AddChild(expr.child.get());
	result->Finalize();
	if (expr.bound_cast.init_local_state) {
		result->local_state = expr.bound_cast.init_local_state(root.executor->GetContext());
	}
	return std::move(result);
}

void ExpressionExecutor::Execute(const BoundCastExpression &expr, ExpressionState *state, const SelectionVector *sel,
                                 idx_t count, Vector &result) {
	auto lstate = ExecuteFunctionState::GetFunctionState(*state);

	// resolve the child
	state->intermediate_chunk.Reset();

	auto &child = state->intermediate_chunk.data[0];
	auto child_state = state->child_states[0].get();

	Execute(*expr.child, child_state, sel, count, child);
	if (expr.try_cast) {
		string error_message;
		CastParameters parameters(expr.bound_cast.cast_data.get(), false, &error_message, lstate);
		expr.bound_cast.function(child, result, count, parameters);
	} else {
		// cast it to the type specified by the cast expression
		D_ASSERT(result.GetType() == expr.return_type);
		CastParameters parameters(expr.bound_cast.cast_data.get(), false, nullptr, lstate);
		expr.bound_cast.function(child, result, count, parameters);
	}
}

} // namespace duckdb






#include <algorithm>

namespace duckdb {

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundComparisonExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_unique<ExpressionState>(expr, root);
	result->AddChild(expr.left.get());
	result->AddChild(expr.right.get());
	result->Finalize();
	return result;
}

void ExpressionExecutor::Execute(const BoundComparisonExpression &expr, ExpressionState *state,
                                 const SelectionVector *sel, idx_t count, Vector &result) {
	// resolve the children
	state->intermediate_chunk.Reset();
	auto &left = state->intermediate_chunk.data[0];
	auto &right = state->intermediate_chunk.data[1];

	Execute(*expr.left, state->child_states[0].get(), sel, count, left);
	Execute(*expr.right, state->child_states[1].get(), sel, count, right);

	switch (expr.type) {
	case ExpressionType::COMPARE_EQUAL:
		VectorOperations::Equals(left, right, result, count);
		break;
	case ExpressionType::COMPARE_NOTEQUAL:
		VectorOperations::NotEquals(left, right, result, count);
		break;
	case ExpressionType::COMPARE_LESSTHAN:
		VectorOperations::LessThan(left, right, result, count);
		break;
	case ExpressionType::COMPARE_GREATERTHAN:
		VectorOperations::GreaterThan(left, right, result, count);
		break;
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		VectorOperations::LessThanEquals(left, right, result, count);
		break;
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		VectorOperations::GreaterThanEquals(left, right, result, count);
		break;
	case ExpressionType::COMPARE_DISTINCT_FROM:
		VectorOperations::DistinctFrom(left, right, result, count);
		break;
	case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
		VectorOperations::NotDistinctFrom(left, right, result, count);
		break;
	default:
		throw InternalException("Unknown comparison type!");
	}
}

template <typename OP>
static idx_t NestedSelectOperation(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                   SelectionVector *true_sel, SelectionVector *false_sel);

template <class OP>
static idx_t TemplatedSelectOperation(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                      SelectionVector *true_sel, SelectionVector *false_sel) {
	// the inplace loops take the result as the last parameter
	switch (left.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		return BinaryExecutor::Select<int8_t, int8_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::INT16:
		return BinaryExecutor::Select<int16_t, int16_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::INT32:
		return BinaryExecutor::Select<int32_t, int32_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::INT64:
		return BinaryExecutor::Select<int64_t, int64_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::UINT8:
		return BinaryExecutor::Select<uint8_t, uint8_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::UINT16:
		return BinaryExecutor::Select<uint16_t, uint16_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::UINT32:
		return BinaryExecutor::Select<uint32_t, uint32_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::UINT64:
		return BinaryExecutor::Select<uint64_t, uint64_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::INT128:
		return BinaryExecutor::Select<hugeint_t, hugeint_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::FLOAT:
		return BinaryExecutor::Select<float, float, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::DOUBLE:
		return BinaryExecutor::Select<double, double, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::INTERVAL:
		return BinaryExecutor::Select<interval_t, interval_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::VARCHAR:
		return BinaryExecutor::Select<string_t, string_t, OP>(left, right, sel, count, true_sel, false_sel);
	case PhysicalType::LIST:
	case PhysicalType::STRUCT:
		return NestedSelectOperation<OP>(left, right, sel, count, true_sel, false_sel);
	default:
		throw InternalException("Invalid type for comparison");
	}
}

struct NestedSelector {
	// Select the matching rows for the values of a nested type that are not both NULL.
	// Those semantics are the same as the corresponding non-distinct comparator
	template <typename OP>
	static idx_t Select(Vector &left, Vector &right, const SelectionVector &sel, idx_t count, SelectionVector *true_sel,
	                    SelectionVector *false_sel) {
		throw InvalidTypeException(left.GetType(), "Invalid operation for nested SELECT");
	}
};

template <>
idx_t NestedSelector::Select<duckdb::Equals>(Vector &left, Vector &right, const SelectionVector &sel, idx_t count,
                                             SelectionVector *true_sel, SelectionVector *false_sel) {
	return VectorOperations::NestedEquals(left, right, sel, count, true_sel, false_sel);
}

template <>
idx_t NestedSelector::Select<duckdb::NotEquals>(Vector &left, Vector &right, const SelectionVector &sel, idx_t count,
                                                SelectionVector *true_sel, SelectionVector *false_sel) {
	return VectorOperations::NestedNotEquals(left, right, sel, count, true_sel, false_sel);
}

template <>
idx_t NestedSelector::Select<duckdb::LessThan>(Vector &left, Vector &right, const SelectionVector &sel, idx_t count,
                                               SelectionVector *true_sel, SelectionVector *false_sel) {
	return VectorOperations::DistinctLessThan(left, right, &sel, count, true_sel, false_sel);
}

template <>
idx_t NestedSelector::Select<duckdb::LessThanEquals>(Vector &left, Vector &right, const SelectionVector &sel,
                                                     idx_t count, SelectionVector *true_sel,
                                                     SelectionVector *false_sel) {
	return VectorOperations::DistinctLessThanEquals(left, right, &sel, count, true_sel, false_sel);
}

template <>
idx_t NestedSelector::Select<duckdb::GreaterThan>(Vector &left, Vector &right, const SelectionVector &sel, idx_t count,
                                                  SelectionVector *true_sel, SelectionVector *false_sel) {
	return VectorOperations::DistinctGreaterThan(left, right, &sel, count, true_sel, false_sel);
}

template <>
idx_t NestedSelector::Select<duckdb::GreaterThanEquals>(Vector &left, Vector &right, const SelectionVector &sel,
                                                        idx_t count, SelectionVector *true_sel,
                                                        SelectionVector *false_sel) {
	return VectorOperations::DistinctGreaterThanEquals(left, right, &sel, count, true_sel, false_sel);
}

static inline idx_t SelectNotNull(Vector &left, Vector &right, const idx_t count, const SelectionVector &sel,
                                  SelectionVector &maybe_vec, OptionalSelection &false_opt) {

	UnifiedVectorFormat lvdata, rvdata;
	left.ToUnifiedFormat(count, lvdata);
	right.ToUnifiedFormat(count, rvdata);

	auto &lmask = lvdata.validity;
	auto &rmask = rvdata.validity;

	// For top-level comparisons, NULL semantics are in effect,
	// so filter out any NULLs
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
	idx_t false_count = 0;
	for (idx_t i = 0; i < count; ++i) {
		const auto result_idx = sel.get_index(i);
		const auto lidx = lvdata.sel->get_index(i);
		const auto ridx = rvdata.sel->get_index(i);
		if (!lmask.RowIsValid(lidx) || !rmask.RowIsValid(ridx)) {
			false_opt.Append(false_count, result_idx);
		} else {
			//	Neither is NULL, distinguish values.
			slicer.set_index(remaining, i);
			maybe_vec.set_index(remaining++, result_idx);
		}
	}
	false_opt.Advance(false_count);

	if (remaining && remaining < count) {
		left.Slice(slicer, remaining);
		right.Slice(slicer, remaining);
	}

	return remaining;
}

static void ScatterSelection(SelectionVector *target, const idx_t count, const SelectionVector &dense_vec) {
	if (target) {
		for (idx_t i = 0; i < count; ++i) {
			target->set_index(i, dense_vec.get_index(i));
		}
	}
}

template <typename OP>
static idx_t NestedSelectOperation(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
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

	auto match_count = SelectNotNull(l_not_null, r_not_null, count, *sel, maybe_vec, false_opt);
	auto no_match_count = count - match_count;
	count = match_count;

	//	Now that we have handled the NULLs, we can use the recursive nested comparator for the rest.
	match_count = NestedSelector::Select<OP>(l_not_null, r_not_null, maybe_vec, count, true_opt, false_opt);
	no_match_count += (count - match_count);

	// Copy the buffered selections to the output selections
	ScatterSelection(true_sel, match_count, true_vec);
	ScatterSelection(false_sel, no_match_count, false_vec);

	return match_count;
}

idx_t VectorOperations::Equals(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                               SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedSelectOperation<duckdb::Equals>(left, right, sel, count, true_sel, false_sel);
}

idx_t VectorOperations::NotEquals(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                  SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedSelectOperation<duckdb::NotEquals>(left, right, sel, count, true_sel, false_sel);
}

idx_t VectorOperations::GreaterThan(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                    SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedSelectOperation<duckdb::GreaterThan>(left, right, sel, count, true_sel, false_sel);
}

idx_t VectorOperations::GreaterThanEquals(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                          SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedSelectOperation<duckdb::GreaterThanEquals>(left, right, sel, count, true_sel, false_sel);
}

idx_t VectorOperations::LessThan(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                 SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedSelectOperation<duckdb::LessThan>(left, right, sel, count, true_sel, false_sel);
}

idx_t VectorOperations::LessThanEquals(Vector &left, Vector &right, const SelectionVector *sel, idx_t count,
                                       SelectionVector *true_sel, SelectionVector *false_sel) {
	return TemplatedSelectOperation<duckdb::LessThanEquals>(left, right, sel, count, true_sel, false_sel);
}

idx_t ExpressionExecutor::Select(const BoundComparisonExpression &expr, ExpressionState *state,
                                 const SelectionVector *sel, idx_t count, SelectionVector *true_sel,
                                 SelectionVector *false_sel) {
	// resolve the children
	state->intermediate_chunk.Reset();
	auto &left = state->intermediate_chunk.data[0];
	auto &right = state->intermediate_chunk.data[1];

	Execute(*expr.left, state->child_states[0].get(), sel, count, left);
	Execute(*expr.right, state->child_states[1].get(), sel, count, right);

	switch (expr.type) {
	case ExpressionType::COMPARE_EQUAL:
		return VectorOperations::Equals(left, right, sel, count, true_sel, false_sel);
	case ExpressionType::COMPARE_NOTEQUAL:
		return VectorOperations::NotEquals(left, right, sel, count, true_sel, false_sel);
	case ExpressionType::COMPARE_LESSTHAN:
		return VectorOperations::LessThan(left, right, sel, count, true_sel, false_sel);
	case ExpressionType::COMPARE_GREATERTHAN:
		return VectorOperations::GreaterThan(left, right, sel, count, true_sel, false_sel);
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return VectorOperations::LessThanEquals(left, right, sel, count, true_sel, false_sel);
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return VectorOperations::GreaterThanEquals(left, right, sel, count, true_sel, false_sel);
	case ExpressionType::COMPARE_DISTINCT_FROM:
		return VectorOperations::DistinctFrom(left, right, sel, count, true_sel, false_sel);
	case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
		return VectorOperations::NotDistinctFrom(left, right, sel, count, true_sel, false_sel);
	default:
		throw InternalException("Unknown comparison type!");
	}
}

} // namespace duckdb






#include <random>

namespace duckdb {

struct ConjunctionState : public ExpressionState {
	ConjunctionState(const Expression &expr, ExpressionExecutorState &root) : ExpressionState(expr, root) {
		adaptive_filter = make_unique<AdaptiveFilter>(expr);
	}
	unique_ptr<AdaptiveFilter> adaptive_filter;
};

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundConjunctionExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_unique<ConjunctionState>(expr, root);
	for (auto &child : expr.children) {
		result->AddChild(child.get());
	}
	result->Finalize();
	return std::move(result);
}

void ExpressionExecutor::Execute(const BoundConjunctionExpression &expr, ExpressionState *state,
                                 const SelectionVector *sel, idx_t count, Vector &result) {
	// execute the children
	state->intermediate_chunk.Reset();
	for (idx_t i = 0; i < expr.children.size(); i++) {
		auto &current_result = state->intermediate_chunk.data[i];
		Execute(*expr.children[i], state->child_states[i].get(), sel, count, current_result);
		if (i == 0) {
			// move the result
			result.Reference(current_result);
		} else {
			Vector intermediate(LogicalType::BOOLEAN);
			// AND/OR together
			switch (expr.type) {
			case ExpressionType::CONJUNCTION_AND:
				VectorOperations::And(current_result, result, intermediate, count);
				break;
			case ExpressionType::CONJUNCTION_OR:
				VectorOperations::Or(current_result, result, intermediate, count);
				break;
			default:
				throw InternalException("Unknown conjunction type!");
			}
			result.Reference(intermediate);
		}
	}
}

idx_t ExpressionExecutor::Select(const BoundConjunctionExpression &expr, ExpressionState *state_p,
                                 const SelectionVector *sel, idx_t count, SelectionVector *true_sel,
                                 SelectionVector *false_sel) {
	auto state = (ConjunctionState *)state_p;

	if (expr.type == ExpressionType::CONJUNCTION_AND) {
		// get runtime statistics
		auto start_time = high_resolution_clock::now();

		const SelectionVector *current_sel = sel;
		idx_t current_count = count;
		idx_t false_count = 0;

		unique_ptr<SelectionVector> temp_true, temp_false;
		if (false_sel) {
			temp_false = make_unique<SelectionVector>(STANDARD_VECTOR_SIZE);
		}
		if (!true_sel) {
			temp_true = make_unique<SelectionVector>(STANDARD_VECTOR_SIZE);
			true_sel = temp_true.get();
		}
		for (idx_t i = 0; i < expr.children.size(); i++) {
			idx_t tcount = Select(*expr.children[state->adaptive_filter->permutation[i]],
			                      state->child_states[state->adaptive_filter->permutation[i]].get(), current_sel,
			                      current_count, true_sel, temp_false.get());
			idx_t fcount = current_count - tcount;
			if (fcount > 0 && false_sel) {
				// move failing tuples into the false_sel
				// tuples passed, move them into the actual result vector
				for (idx_t i = 0; i < fcount; i++) {
					false_sel->set_index(false_count++, temp_false->get_index(i));
				}
			}
			current_count = tcount;
			if (current_count == 0) {
				break;
			}
			if (current_count < count) {
				// tuples were filtered out: move on to using the true_sel to only evaluate passing tuples in subsequent
				// iterations
				current_sel = true_sel;
			}
		}

		// adapt runtime statistics
		auto end_time = high_resolution_clock::now();
		state->adaptive_filter->AdaptRuntimeStatistics(duration_cast<duration<double>>(end_time - start_time).count());
		return current_count;
	} else {
		// get runtime statistics
		auto start_time = high_resolution_clock::now();

		const SelectionVector *current_sel = sel;
		idx_t current_count = count;
		idx_t result_count = 0;

		unique_ptr<SelectionVector> temp_true, temp_false;
		if (true_sel) {
			temp_true = make_unique<SelectionVector>(STANDARD_VECTOR_SIZE);
		}
		if (!false_sel) {
			temp_false = make_unique<SelectionVector>(STANDARD_VECTOR_SIZE);
			false_sel = temp_false.get();
		}
		for (idx_t i = 0; i < expr.children.size(); i++) {
			idx_t tcount = Select(*expr.children[state->adaptive_filter->permutation[i]],
			                      state->child_states[state->adaptive_filter->permutation[i]].get(), current_sel,
			                      current_count, temp_true.get(), false_sel);
			if (tcount > 0) {
				if (true_sel) {
					// tuples passed, move them into the actual result vector
					for (idx_t i = 0; i < tcount; i++) {
						true_sel->set_index(result_count++, temp_true->get_index(i));
					}
				}
				// now move on to check only the non-passing tuples
				current_count -= tcount;
				current_sel = false_sel;
			}
		}

		// adapt runtime statistics
		auto end_time = high_resolution_clock::now();
		state->adaptive_filter->AdaptRuntimeStatistics(duration_cast<duration<double>>(end_time - start_time).count());
		return result_count;
	}
}

} // namespace duckdb




namespace duckdb {

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundConstantExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_unique<ExpressionState>(expr, root);
	result->Finalize();
	return result;
}

void ExpressionExecutor::Execute(const BoundConstantExpression &expr, ExpressionState *state,
                                 const SelectionVector *sel, idx_t count, Vector &result) {
	D_ASSERT(expr.value.type() == expr.return_type);
	result.Reference(expr.value);
}

} // namespace duckdb



namespace duckdb {

ExecuteFunctionState::ExecuteFunctionState(const Expression &expr, ExpressionExecutorState &root)
    : ExpressionState(expr, root) {
}

ExecuteFunctionState::~ExecuteFunctionState() {
}

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundFunctionExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_unique<ExecuteFunctionState>(expr, root);
	for (auto &child : expr.children) {
		result->AddChild(child.get());
	}
	result->Finalize();
	if (expr.function.init_local_state) {
		result->local_state = expr.function.init_local_state(*result, expr, expr.bind_info.get());
	}
	return std::move(result);
}

static void VerifyNullHandling(const BoundFunctionExpression &expr, DataChunk &args, Vector &result) {
#ifdef DEBUG
	if (args.data.empty() || expr.function.null_handling != FunctionNullHandling::DEFAULT_NULL_HANDLING) {
		return;
	}

	// Combine all the argument validity masks into a flat validity mask
	idx_t count = args.size();
	ValidityMask combined_mask(count);
	for (auto &arg : args.data) {
		UnifiedVectorFormat arg_data;
		arg.ToUnifiedFormat(count, arg_data);

		for (idx_t i = 0; i < count; i++) {
			auto idx = arg_data.sel->get_index(i);
			if (!arg_data.validity.RowIsValid(idx)) {
				combined_mask.SetInvalid(i);
			}
		}
	}

	// Default is that if any of the arguments are NULL, the result is also NULL
	UnifiedVectorFormat result_data;
	result.ToUnifiedFormat(count, result_data);
	for (idx_t i = 0; i < count; i++) {
		if (!combined_mask.RowIsValid(i)) {
			auto idx = result_data.sel->get_index(i);
			D_ASSERT(!result_data.validity.RowIsValid(idx));
		}
	}
#endif
}

void ExpressionExecutor::Execute(const BoundFunctionExpression &expr, ExpressionState *state,
                                 const SelectionVector *sel, idx_t count, Vector &result) {
	state->intermediate_chunk.Reset();
	auto &arguments = state->intermediate_chunk;
	if (!state->types.empty()) {
		for (idx_t i = 0; i < expr.children.size(); i++) {
			D_ASSERT(state->types[i] == expr.children[i]->return_type);
			Execute(*expr.children[i], state->child_states[i].get(), sel, count, arguments.data[i]);
#ifdef DEBUG
			if (expr.children[i]->return_type.id() == LogicalTypeId::VARCHAR) {
				arguments.data[i].UTFVerify(count);
			}
#endif
		}
		arguments.Verify();
	}
	arguments.SetCardinality(count);

	state->profiler.BeginSample();
	D_ASSERT(expr.function.function);
	expr.function.function(arguments, *state, result);
	state->profiler.EndSample(count);

	VerifyNullHandling(expr, arguments, result);
	D_ASSERT(result.GetType() == expr.return_type);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundOperatorExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_unique<ExpressionState>(expr, root);
	for (auto &child : expr.children) {
		result->AddChild(child.get());
	}
	result->Finalize();
	return result;
}

void ExpressionExecutor::Execute(const BoundOperatorExpression &expr, ExpressionState *state,
                                 const SelectionVector *sel, idx_t count, Vector &result) {
	// special handling for special snowflake 'IN'
	// IN has n children
	if (expr.type == ExpressionType::COMPARE_IN || expr.type == ExpressionType::COMPARE_NOT_IN) {
		if (expr.children.size() < 2) {
			throw InvalidInputException("IN needs at least two children");
		}

		Vector left(expr.children[0]->return_type);
		// eval left side
		Execute(*expr.children[0], state->child_states[0].get(), sel, count, left);

		// init result to false
		Vector intermediate(LogicalType::BOOLEAN);
		Value false_val = Value::BOOLEAN(false);
		intermediate.Reference(false_val);

		// in rhs is a list of constants
		// for every child, OR the result of the comparision with the left
		// to get the overall result.
		for (idx_t child = 1; child < expr.children.size(); child++) {
			Vector vector_to_check(expr.children[child]->return_type);
			Vector comp_res(LogicalType::BOOLEAN);

			Execute(*expr.children[child], state->child_states[child].get(), sel, count, vector_to_check);
			VectorOperations::Equals(left, vector_to_check, comp_res, count);

			if (child == 1) {
				// first child: move to result
				intermediate.Reference(comp_res);
			} else {
				// otherwise OR together
				Vector new_result(LogicalType::BOOLEAN, true, false);
				VectorOperations::Or(intermediate, comp_res, new_result, count);
				intermediate.Reference(new_result);
			}
		}
		if (expr.type == ExpressionType::COMPARE_NOT_IN) {
			// NOT IN: invert result
			VectorOperations::Not(intermediate, result, count);
		} else {
			// directly use the result
			result.Reference(intermediate);
		}
	} else if (expr.type == ExpressionType::OPERATOR_COALESCE) {
		SelectionVector sel_a(count);
		SelectionVector sel_b(count);
		SelectionVector slice_sel(count);
		SelectionVector result_sel(count);
		SelectionVector *next_sel = &sel_a;
		const SelectionVector *current_sel = sel;
		idx_t remaining_count = count;
		idx_t next_count;
		for (idx_t child = 0; child < expr.children.size(); child++) {
			Vector vector_to_check(expr.children[child]->return_type);
			Execute(*expr.children[child], state->child_states[child].get(), current_sel, remaining_count,
			        vector_to_check);

			UnifiedVectorFormat vdata;
			vector_to_check.ToUnifiedFormat(remaining_count, vdata);

			idx_t result_count = 0;
			next_count = 0;
			for (idx_t i = 0; i < remaining_count; i++) {
				auto base_idx = current_sel ? current_sel->get_index(i) : i;
				auto idx = vdata.sel->get_index(i);
				if (vdata.validity.RowIsValid(idx)) {
					slice_sel.set_index(result_count, i);
					result_sel.set_index(result_count++, base_idx);
				} else {
					next_sel->set_index(next_count++, base_idx);
				}
			}
			if (result_count > 0) {
				vector_to_check.Slice(slice_sel, result_count);
				FillSwitch(vector_to_check, result, result_sel, result_count);
			}
			current_sel = next_sel;
			next_sel = next_sel == &sel_a ? &sel_b : &sel_a;
			remaining_count = next_count;
			if (next_count == 0) {
				break;
			}
		}
		if (remaining_count > 0) {
			for (idx_t i = 0; i < remaining_count; i++) {
				FlatVector::SetNull(result, current_sel->get_index(i), true);
			}
		}
		if (sel) {
			result.Slice(*sel, count);
		} else if (count == 1) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
		}
	} else if (expr.children.size() == 1) {
		state->intermediate_chunk.Reset();
		auto &child = state->intermediate_chunk.data[0];

		Execute(*expr.children[0], state->child_states[0].get(), sel, count, child);
		switch (expr.type) {
		case ExpressionType::OPERATOR_NOT: {
			VectorOperations::Not(child, result, count);
			break;
		}
		case ExpressionType::OPERATOR_IS_NULL: {
			VectorOperations::IsNull(child, result, count);
			break;
		}
		case ExpressionType::OPERATOR_IS_NOT_NULL: {
			VectorOperations::IsNotNull(child, result, count);
			break;
		}
		default:
			throw NotImplementedException("Unsupported operator type with 1 child!");
		}
	} else {
		throw NotImplementedException("operator");
	}
}

} // namespace duckdb




namespace duckdb {

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundParameterExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_unique<ExpressionState>(expr, root);
	result->Finalize();
	return result;
}

void ExpressionExecutor::Execute(const BoundParameterExpression &expr, ExpressionState *state,
                                 const SelectionVector *sel, idx_t count, Vector &result) {
	D_ASSERT(expr.parameter_data);
	D_ASSERT(expr.parameter_data->return_type == expr.return_type);
	D_ASSERT(expr.parameter_data->value.type() == expr.return_type);
	result.Reference(expr.parameter_data->value);
}

} // namespace duckdb



namespace duckdb {

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const BoundReferenceExpression &expr,
                                                                ExpressionExecutorState &root) {
	auto result = make_unique<ExpressionState>(expr, root);
	result->Finalize();
	return result;
}

void ExpressionExecutor::Execute(const BoundReferenceExpression &expr, ExpressionState *state,
                                 const SelectionVector *sel, idx_t count, Vector &result) {
	D_ASSERT(expr.index != DConstants::INVALID_INDEX);
	D_ASSERT(expr.index < chunk->ColumnCount());

	if (sel) {
		result.Slice(chunk->data[expr.index], *sel, count);
	} else {
		result.Reference(chunk->data[expr.index]);
	}
}

} // namespace duckdb






namespace duckdb {

ExpressionExecutor::ExpressionExecutor(ClientContext &context) : context(&context) {
}

ExpressionExecutor::ExpressionExecutor(ClientContext &context, const Expression *expression)
    : ExpressionExecutor(context) {
	D_ASSERT(expression);
	AddExpression(*expression);
}

ExpressionExecutor::ExpressionExecutor(ClientContext &context, const Expression &expression)
    : ExpressionExecutor(context) {
	AddExpression(expression);
}

ExpressionExecutor::ExpressionExecutor(ClientContext &context, const vector<unique_ptr<Expression>> &exprs)
    : ExpressionExecutor(context) {
	D_ASSERT(exprs.size() > 0);
	for (auto &expr : exprs) {
		AddExpression(*expr);
	}
}

ExpressionExecutor::ExpressionExecutor(const vector<unique_ptr<Expression>> &exprs) : context(nullptr) {
	D_ASSERT(exprs.size() > 0);
	for (auto &expr : exprs) {
		AddExpression(*expr);
	}
}

ExpressionExecutor::ExpressionExecutor() : context(nullptr) {
}

bool ExpressionExecutor::HasContext() {
	return context;
}

ClientContext &ExpressionExecutor::GetContext() {
	if (!context) {
		throw InternalException("Calling ExpressionExecutor::GetContext on an expression executor without a context");
	}
	return *context;
}

Allocator &ExpressionExecutor::GetAllocator() {
	return context ? Allocator::Get(*context) : Allocator::DefaultAllocator();
}

void ExpressionExecutor::AddExpression(const Expression &expr) {
	expressions.push_back(&expr);
	auto state = make_unique<ExpressionExecutorState>(expr.ToString());
	Initialize(expr, *state);
	state->Verify();
	states.push_back(std::move(state));
}

void ExpressionExecutor::Initialize(const Expression &expression, ExpressionExecutorState &state) {
	state.executor = this;
	state.root_state = InitializeState(expression, state);
}

void ExpressionExecutor::Execute(DataChunk *input, DataChunk &result) {
	SetChunk(input);
	D_ASSERT(expressions.size() == result.ColumnCount());
	D_ASSERT(!expressions.empty());

	for (idx_t i = 0; i < expressions.size(); i++) {
		ExecuteExpression(i, result.data[i]);
	}
	result.SetCardinality(input ? input->size() : 1);
	result.Verify();
}

void ExpressionExecutor::ExecuteExpression(DataChunk &input, Vector &result) {
	SetChunk(&input);
	ExecuteExpression(result);
}

idx_t ExpressionExecutor::SelectExpression(DataChunk &input, SelectionVector &sel) {
	D_ASSERT(expressions.size() == 1);
	SetChunk(&input);
	states[0]->profiler.BeginSample();
	idx_t selected_tuples = Select(*expressions[0], states[0]->root_state.get(), nullptr, input.size(), &sel, nullptr);
	states[0]->profiler.EndSample(chunk ? chunk->size() : 0);
	return selected_tuples;
}

void ExpressionExecutor::ExecuteExpression(Vector &result) {
	D_ASSERT(expressions.size() == 1);
	ExecuteExpression(0, result);
}

void ExpressionExecutor::ExecuteExpression(idx_t expr_idx, Vector &result) {
	D_ASSERT(expr_idx < expressions.size());
	D_ASSERT(result.GetType().id() == expressions[expr_idx]->return_type.id());
	states[expr_idx]->profiler.BeginSample();
	Execute(*expressions[expr_idx], states[expr_idx]->root_state.get(), nullptr, chunk ? chunk->size() : 1, result);
	states[expr_idx]->profiler.EndSample(chunk ? chunk->size() : 0);
}

Value ExpressionExecutor::EvaluateScalar(ClientContext &context, const Expression &expr, bool allow_unfoldable) {
	D_ASSERT(allow_unfoldable || expr.IsFoldable());
	D_ASSERT(expr.IsScalar());
	// use an ExpressionExecutor to execute the expression
	ExpressionExecutor executor(context, expr);

	Vector result(expr.return_type);
	executor.ExecuteExpression(result);

	D_ASSERT(allow_unfoldable || result.GetVectorType() == VectorType::CONSTANT_VECTOR);
	auto result_value = result.GetValue(0);
	D_ASSERT(result_value.type().InternalType() == expr.return_type.InternalType());
	return result_value;
}

bool ExpressionExecutor::TryEvaluateScalar(ClientContext &context, const Expression &expr, Value &result) {
	try {
		result = EvaluateScalar(context, expr);
		return true;
	} catch (InternalException &ex) {
		throw ex;
	} catch (...) {
		return false;
	}
}

void ExpressionExecutor::Verify(const Expression &expr, Vector &vector, idx_t count) {
	D_ASSERT(expr.return_type.id() == vector.GetType().id());
	vector.Verify(count);
	if (expr.verification_stats) {
		expr.verification_stats->Verify(vector, count);
	}
}

unique_ptr<ExpressionState> ExpressionExecutor::InitializeState(const Expression &expr,
                                                                ExpressionExecutorState &state) {
	switch (expr.expression_class) {
	case ExpressionClass::BOUND_REF:
		return InitializeState((const BoundReferenceExpression &)expr, state);
	case ExpressionClass::BOUND_BETWEEN:
		return InitializeState((const BoundBetweenExpression &)expr, state);
	case ExpressionClass::BOUND_CASE:
		return InitializeState((const BoundCaseExpression &)expr, state);
	case ExpressionClass::BOUND_CAST:
		return InitializeState((const BoundCastExpression &)expr, state);
	case ExpressionClass::BOUND_COMPARISON:
		return InitializeState((const BoundComparisonExpression &)expr, state);
	case ExpressionClass::BOUND_CONJUNCTION:
		return InitializeState((const BoundConjunctionExpression &)expr, state);
	case ExpressionClass::BOUND_CONSTANT:
		return InitializeState((const BoundConstantExpression &)expr, state);
	case ExpressionClass::BOUND_FUNCTION:
		return InitializeState((const BoundFunctionExpression &)expr, state);
	case ExpressionClass::BOUND_OPERATOR:
		return InitializeState((const BoundOperatorExpression &)expr, state);
	case ExpressionClass::BOUND_PARAMETER:
		return InitializeState((const BoundParameterExpression &)expr, state);
	default:
		throw InternalException("Attempting to initialize state of expression of unknown type!");
	}
}

void ExpressionExecutor::Execute(const Expression &expr, ExpressionState *state, const SelectionVector *sel,
                                 idx_t count, Vector &result) {
#ifdef DEBUG
	//! The result Vector must be "clean"
	if (result.GetVectorType() == VectorType::FLAT_VECTOR) {
		D_ASSERT(FlatVector::Validity(result).CheckAllValid(count));
	}
#endif

	if (count == 0) {
		return;
	}
	switch (expr.expression_class) {
	case ExpressionClass::BOUND_BETWEEN:
		Execute((const BoundBetweenExpression &)expr, state, sel, count, result);
		break;
	case ExpressionClass::BOUND_REF:
		Execute((const BoundReferenceExpression &)expr, state, sel, count, result);
		break;
	case ExpressionClass::BOUND_CASE:
		Execute((const BoundCaseExpression &)expr, state, sel, count, result);
		break;
	case ExpressionClass::BOUND_CAST:
		Execute((const BoundCastExpression &)expr, state, sel, count, result);
		break;
	case ExpressionClass::BOUND_COMPARISON:
		Execute((const BoundComparisonExpression &)expr, state, sel, count, result);
		break;
	case ExpressionClass::BOUND_CONJUNCTION:
		Execute((const BoundConjunctionExpression &)expr, state, sel, count, result);
		break;
	case ExpressionClass::BOUND_CONSTANT:
		Execute((const BoundConstantExpression &)expr, state, sel, count, result);
		break;
	case ExpressionClass::BOUND_FUNCTION:
		Execute((const BoundFunctionExpression &)expr, state, sel, count, result);
		break;
	case ExpressionClass::BOUND_OPERATOR:
		Execute((const BoundOperatorExpression &)expr, state, sel, count, result);
		break;
	case ExpressionClass::BOUND_PARAMETER:
		Execute((const BoundParameterExpression &)expr, state, sel, count, result);
		break;
	default:
		throw InternalException("Attempting to execute expression of unknown type!");
	}
	Verify(expr, result, count);
}

idx_t ExpressionExecutor::Select(const Expression &expr, ExpressionState *state, const SelectionVector *sel,
                                 idx_t count, SelectionVector *true_sel, SelectionVector *false_sel) {
	if (count == 0) {
		return 0;
	}
	D_ASSERT(true_sel || false_sel);
	D_ASSERT(expr.return_type.id() == LogicalTypeId::BOOLEAN);
	switch (expr.expression_class) {
	case ExpressionClass::BOUND_BETWEEN:
		return Select((BoundBetweenExpression &)expr, state, sel, count, true_sel, false_sel);
	case ExpressionClass::BOUND_COMPARISON:
		return Select((BoundComparisonExpression &)expr, state, sel, count, true_sel, false_sel);
	case ExpressionClass::BOUND_CONJUNCTION:
		return Select((BoundConjunctionExpression &)expr, state, sel, count, true_sel, false_sel);
	default:
		return DefaultSelect(expr, state, sel, count, true_sel, false_sel);
	}
}

template <bool NO_NULL, bool HAS_TRUE_SEL, bool HAS_FALSE_SEL>
static inline idx_t DefaultSelectLoop(const SelectionVector *bsel, uint8_t *__restrict bdata, ValidityMask &mask,
                                      const SelectionVector *sel, idx_t count, SelectionVector *true_sel,
                                      SelectionVector *false_sel) {
	idx_t true_count = 0, false_count = 0;
	for (idx_t i = 0; i < count; i++) {
		auto bidx = bsel->get_index(i);
		auto result_idx = sel->get_index(i);
		if (bdata[bidx] > 0 && (NO_NULL || mask.RowIsValid(bidx))) {
			if (HAS_TRUE_SEL) {
				true_sel->set_index(true_count++, result_idx);
			}
		} else {
			if (HAS_FALSE_SEL) {
				false_sel->set_index(false_count++, result_idx);
			}
		}
	}
	if (HAS_TRUE_SEL) {
		return true_count;
	} else {
		return count - false_count;
	}
}

template <bool NO_NULL>
static inline idx_t DefaultSelectSwitch(UnifiedVectorFormat &idata, const SelectionVector *sel, idx_t count,
                                        SelectionVector *true_sel, SelectionVector *false_sel) {
	if (true_sel && false_sel) {
		return DefaultSelectLoop<NO_NULL, true, true>(idata.sel, (uint8_t *)idata.data, idata.validity, sel, count,
		                                              true_sel, false_sel);
	} else if (true_sel) {
		return DefaultSelectLoop<NO_NULL, true, false>(idata.sel, (uint8_t *)idata.data, idata.validity, sel, count,
		                                               true_sel, false_sel);
	} else {
		D_ASSERT(false_sel);
		return DefaultSelectLoop<NO_NULL, false, true>(idata.sel, (uint8_t *)idata.data, idata.validity, sel, count,
		                                               true_sel, false_sel);
	}
}

idx_t ExpressionExecutor::DefaultSelect(const Expression &expr, ExpressionState *state, const SelectionVector *sel,
                                        idx_t count, SelectionVector *true_sel, SelectionVector *false_sel) {
	// generic selection of boolean expression:
	// resolve the true/false expression first
	// then use that to generate the selection vector
	bool intermediate_bools[STANDARD_VECTOR_SIZE];
	Vector intermediate(LogicalType::BOOLEAN, (data_ptr_t)intermediate_bools);
	Execute(expr, state, sel, count, intermediate);

	UnifiedVectorFormat idata;
	intermediate.ToUnifiedFormat(count, idata);

	if (!sel) {
		sel = FlatVector::IncrementalSelectionVector();
	}
	if (!idata.validity.AllValid()) {
		return DefaultSelectSwitch<false>(idata, sel, count, true_sel, false_sel);
	} else {
		return DefaultSelectSwitch<true>(idata, sel, count, true_sel, false_sel);
	}
}

vector<unique_ptr<ExpressionExecutorState>> &ExpressionExecutor::GetStates() {
	return states;
}

} // namespace duckdb





namespace duckdb {

void ExpressionState::AddChild(Expression *expr) {
	types.push_back(expr->return_type);
	child_states.push_back(ExpressionExecutor::InitializeState(*expr, root));
}

void ExpressionState::Finalize() {
	if (!types.empty()) {
		intermediate_chunk.Initialize(GetAllocator(), types);
	}
}

Allocator &ExpressionState::GetAllocator() {
	return root.executor->GetAllocator();
}

bool ExpressionState::HasContext() {
	return root.executor->HasContext();
}

ClientContext &ExpressionState::GetContext() {
	if (!HasContext()) {
		throw BinderException("Cannot use %s in this context", ((BoundFunctionExpression &)expr).function.name);
	}
	return root.executor->GetContext();
}

ExpressionState::ExpressionState(const Expression &expr, ExpressionExecutorState &root)
    : expr(expr), root(root), name(expr.ToString()) {
}

ExpressionExecutorState::ExpressionExecutorState(const string &name) : profiler(), name(name) {
}

void ExpressionState::Verify(ExpressionExecutorState &root_executor) {
	D_ASSERT(&root_executor == &root);
	for (auto &entry : child_states) {
		entry->Verify(root_executor);
	}
}

void ExpressionExecutorState::Verify() {
	D_ASSERT(executor);
	root_state->Verify(*this);
}

} // namespace duckdb









#include <algorithm>
#include <cstring>
#include <ctgmath>

namespace duckdb {

ART::ART(const vector<column_t> &column_ids, TableIOManager &table_io_manager,
         const vector<unique_ptr<Expression>> &unbound_expressions, IndexConstraintType constraint_type,
         AttachedDatabase &db, bool track_memory, idx_t block_id, idx_t block_offset)

    : Index(db, IndexType::ART, table_io_manager, column_ids, unbound_expressions, constraint_type, track_memory) {

	if (!Radix::IsLittleEndian()) {
		throw NotImplementedException("ART indexes are not supported on big endian architectures");
	}

	// set the root node of the tree
	tree = nullptr;
	if (block_id != DConstants::INVALID_INDEX) {
		tree = Node::Deserialize(*this, block_id, block_offset);
		ART::Verify();
	}
	serialized_data_pointer = BlockPointer(block_id, block_offset);

	// validate the types of the key columns
	for (idx_t i = 0; i < types.size(); i++) {
		switch (types[i]) {
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
		case PhysicalType::VARCHAR:
			break;
		default:
			throw InvalidTypeException(logical_types[i], "Invalid type for index key.");
		}
	}
}

ART::~ART() {
	if (!tree) {
		return;
	}
	ART::Verify();
	if (track_memory) {
		buffer_manager.DecreaseUsedMemory(memory_size);
	}
	Node::Delete(tree);
	tree = nullptr;
}

//===--------------------------------------------------------------------===//
// Initialize Predicate Scans
//===--------------------------------------------------------------------===//

unique_ptr<IndexScanState> ART::InitializeScanSinglePredicate(const Transaction &transaction, const Value &value,
                                                              ExpressionType expression_type) {
	auto result = make_unique<ARTIndexScanState>();
	result->values[0] = value;
	result->expressions[0] = expression_type;
	return std::move(result);
}

unique_ptr<IndexScanState> ART::InitializeScanTwoPredicates(Transaction &transaction, const Value &low_value,
                                                            ExpressionType low_expression_type, const Value &high_value,
                                                            ExpressionType high_expression_type) {
	auto result = make_unique<ARTIndexScanState>();
	result->values[0] = low_value;
	result->expressions[0] = low_expression_type;
	result->values[1] = high_value;
	result->expressions[1] = high_expression_type;
	return std::move(result);
}

//===--------------------------------------------------------------------===//
// Keys
//===--------------------------------------------------------------------===//

template <class T>
static void TemplatedGenerateKeys(ArenaAllocator &allocator, Vector &input, idx_t count, vector<Key> &keys) {
	UnifiedVectorFormat idata;
	input.ToUnifiedFormat(count, idata);

	D_ASSERT(keys.size() >= count);
	auto input_data = (T *)idata.data;
	for (idx_t i = 0; i < count; i++) {
		auto idx = idata.sel->get_index(i);
		if (idata.validity.RowIsValid(idx)) {
			Key::CreateKey<T>(allocator, keys[i], input_data[idx]);
		}
	}
}

template <class T>
static void ConcatenateKeys(ArenaAllocator &allocator, Vector &input, idx_t count, vector<Key> &keys) {
	UnifiedVectorFormat idata;
	input.ToUnifiedFormat(count, idata);

	auto input_data = (T *)idata.data;
	for (idx_t i = 0; i < count; i++) {
		auto idx = idata.sel->get_index(i);

		// key is not NULL (no previous column entry was NULL)
		if (!keys[i].Empty()) {
			if (!idata.validity.RowIsValid(idx)) {
				// this column entry is NULL, set whole key to NULL
				keys[i] = Key();
			} else {
				auto other_key = Key::CreateKey<T>(allocator, input_data[idx]);
				keys[i].ConcatenateKey(allocator, other_key);
			}
		}
	}
}

void ART::GenerateKeys(ArenaAllocator &allocator, DataChunk &input, vector<Key> &keys) {
	// generate keys for the first input column
	switch (input.data[0].GetType().InternalType()) {
	case PhysicalType::BOOL:
		TemplatedGenerateKeys<bool>(allocator, input.data[0], input.size(), keys);
		break;
	case PhysicalType::INT8:
		TemplatedGenerateKeys<int8_t>(allocator, input.data[0], input.size(), keys);
		break;
	case PhysicalType::INT16:
		TemplatedGenerateKeys<int16_t>(allocator, input.data[0], input.size(), keys);
		break;
	case PhysicalType::INT32:
		TemplatedGenerateKeys<int32_t>(allocator, input.data[0], input.size(), keys);
		break;
	case PhysicalType::INT64:
		TemplatedGenerateKeys<int64_t>(allocator, input.data[0], input.size(), keys);
		break;
	case PhysicalType::INT128:
		TemplatedGenerateKeys<hugeint_t>(allocator, input.data[0], input.size(), keys);
		break;
	case PhysicalType::UINT8:
		TemplatedGenerateKeys<uint8_t>(allocator, input.data[0], input.size(), keys);
		break;
	case PhysicalType::UINT16:
		TemplatedGenerateKeys<uint16_t>(allocator, input.data[0], input.size(), keys);
		break;
	case PhysicalType::UINT32:
		TemplatedGenerateKeys<uint32_t>(allocator, input.data[0], input.size(), keys);
		break;
	case PhysicalType::UINT64:
		TemplatedGenerateKeys<uint64_t>(allocator, input.data[0], input.size(), keys);
		break;
	case PhysicalType::FLOAT:
		TemplatedGenerateKeys<float>(allocator, input.data[0], input.size(), keys);
		break;
	case PhysicalType::DOUBLE:
		TemplatedGenerateKeys<double>(allocator, input.data[0], input.size(), keys);
		break;
	case PhysicalType::VARCHAR:
		TemplatedGenerateKeys<string_t>(allocator, input.data[0], input.size(), keys);
		break;
	default:
		throw InternalException("Invalid type for index");
	}

	for (idx_t i = 1; i < input.ColumnCount(); i++) {
		// for each of the remaining columns, concatenate
		switch (input.data[i].GetType().InternalType()) {
		case PhysicalType::BOOL:
			ConcatenateKeys<bool>(allocator, input.data[i], input.size(), keys);
			break;
		case PhysicalType::INT8:
			ConcatenateKeys<int8_t>(allocator, input.data[i], input.size(), keys);
			break;
		case PhysicalType::INT16:
			ConcatenateKeys<int16_t>(allocator, input.data[i], input.size(), keys);
			break;
		case PhysicalType::INT32:
			ConcatenateKeys<int32_t>(allocator, input.data[i], input.size(), keys);
			break;
		case PhysicalType::INT64:
			ConcatenateKeys<int64_t>(allocator, input.data[i], input.size(), keys);
			break;
		case PhysicalType::INT128:
			ConcatenateKeys<hugeint_t>(allocator, input.data[i], input.size(), keys);
			break;
		case PhysicalType::UINT8:
			ConcatenateKeys<uint8_t>(allocator, input.data[i], input.size(), keys);
			break;
		case PhysicalType::UINT16:
			ConcatenateKeys<uint16_t>(allocator, input.data[i], input.size(), keys);
			break;
		case PhysicalType::UINT32:
			ConcatenateKeys<uint32_t>(allocator, input.data[i], input.size(), keys);
			break;
		case PhysicalType::UINT64:
			ConcatenateKeys<uint64_t>(allocator, input.data[i], input.size(), keys);
			break;
		case PhysicalType::FLOAT:
			ConcatenateKeys<float>(allocator, input.data[i], input.size(), keys);
			break;
		case PhysicalType::DOUBLE:
			ConcatenateKeys<double>(allocator, input.data[i], input.size(), keys);
			break;
		case PhysicalType::VARCHAR:
			ConcatenateKeys<string_t>(allocator, input.data[i], input.size(), keys);
			break;
		default:
			throw InternalException("Invalid type for index");
		}
	}
}

//===--------------------------------------------------------------------===//
// Construct from sorted data
//===--------------------------------------------------------------------===//

struct KeySection {
	KeySection(idx_t start_p, idx_t end_p, idx_t depth_p, data_t key_byte_p)
	    : start(start_p), end(end_p), depth(depth_p), key_byte(key_byte_p) {};
	KeySection(idx_t start_p, idx_t end_p, vector<Key> &keys, KeySection &key_section)
	    : start(start_p), end(end_p), depth(key_section.depth + 1), key_byte(keys[end_p].data[key_section.depth]) {};
	idx_t start;
	idx_t end;
	idx_t depth;
	data_t key_byte;
};

void GetChildSections(vector<KeySection> &child_sections, vector<Key> &keys, KeySection &key_section) {

	idx_t child_start_idx = key_section.start;
	for (idx_t i = key_section.start + 1; i <= key_section.end; i++) {
		if (keys[i - 1].data[key_section.depth] != keys[i].data[key_section.depth]) {
			child_sections.emplace_back(child_start_idx, i - 1, keys, key_section);
			child_start_idx = i;
		}
	}
	child_sections.emplace_back(child_start_idx, key_section.end, keys, key_section);
}

bool Construct(ART &art, vector<Key> &keys, row_t *row_ids, Node *&node, KeySection &key_section,
               bool &has_constraint) {

	D_ASSERT(key_section.start < keys.size());
	D_ASSERT(key_section.end < keys.size());
	D_ASSERT(key_section.start <= key_section.end);

	auto &start_key = keys[key_section.start];
	auto &end_key = keys[key_section.end];

	// increment the depth until we reach a leaf or find a mismatching byte
	auto prefix_start = key_section.depth;
	while (start_key.len != key_section.depth && start_key.ByteMatches(end_key, key_section.depth)) {
		key_section.depth++;
	}

	// we reached a leaf, i.e. all the bytes of start_key and end_key match
	if (start_key.len == key_section.depth) {
		// end_idx is inclusive
		auto num_row_ids = key_section.end - key_section.start + 1;

		// check for possible constraint violation
		auto single_row_id = num_row_ids == 1;
		if (has_constraint && !single_row_id) {
			return false;
		}

		if (single_row_id) {
			node = Leaf::New(start_key, prefix_start, row_ids[key_section.start]);
		} else {
			node = Leaf::New(start_key, prefix_start, row_ids + key_section.start, num_row_ids);
		}
		art.memory_size += node->MemorySize(art, false);
		return true;
	}
	// create a new node and recurse

	// we will find at least two child entries of this node, otherwise we'd have reached a leaf
	vector<KeySection> child_sections;
	GetChildSections(child_sections, keys, key_section);

	auto node_type = Node::GetTypeBySize(child_sections.size());
	Node::New(node_type, node);

	auto prefix_length = key_section.depth - prefix_start;
	node->prefix = Prefix(start_key, prefix_start, prefix_length);
	art.memory_size += node->MemorySize(art, false);

	// recurse on each child section
	for (auto &child_section : child_sections) {
		Node *new_child = nullptr;
		auto no_violation = Construct(art, keys, row_ids, new_child, child_section, has_constraint);
		Node::InsertChild(art, node, child_section.key_byte, new_child);
		if (!no_violation) {
			return false;
		}
	}
	return true;
}

bool ART::ConstructFromSorted(idx_t count, vector<Key> &keys, Vector &row_identifiers) {

	// prepare the row_identifiers
	row_identifiers.Flatten(count);
	auto row_ids = FlatVector::GetData<row_t>(row_identifiers);

	auto key_section = KeySection(0, count - 1, 0, 0);
	auto has_constraint = IsUnique();
	return Construct(*this, keys, row_ids, this->tree, key_section, has_constraint);
}

//===--------------------------------------------------------------------===//
// Insert
//===--------------------------------------------------------------------===//

bool ART::Insert(IndexLock &lock, DataChunk &input, Vector &row_ids) {

	D_ASSERT(row_ids.GetType().InternalType() == ROW_TYPE);
	D_ASSERT(logical_types[0] == input.data[0].GetType());

	// generate the keys for the given input
	ArenaAllocator arena_allocator(BufferAllocator::Get(db));
	vector<Key> keys(input.size());
	GenerateKeys(arena_allocator, input, keys);

	auto old_memory_size = this->memory_size;

	// get the corresponding row IDs
	row_ids.Flatten(input.size());
	auto row_identifiers = FlatVector::GetData<row_t>(row_ids);

	// now insert the elements into the index
	idx_t failed_index = DConstants::INVALID_INDEX;
	for (idx_t i = 0; i < input.size(); i++) {
		if (keys[i].Empty()) {
			continue;
		}

		row_t row_id = row_identifiers[i];
		if (!Insert(tree, keys[i], 0, row_id)) {
			// failed to insert because of constraint violation
			failed_index = i;
			break;
		}
	}
	if (failed_index != DConstants::INVALID_INDEX) {

		// failed to insert because of constraint violation: remove previously inserted entries
		for (idx_t i = 0; i < failed_index; i++) {
			if (keys[i].Empty()) {
				continue;
			}
			row_t row_id = row_identifiers[i];
			Erase(tree, keys[i], 0, row_id);
		}
		// nothing changed, no need to update the buffer memory size
		return false;
	}

	D_ASSERT(old_memory_size <= memory_size);
	Verify();
	if (track_memory) {
		buffer_manager.IncreaseUsedMemory(memory_size - old_memory_size);
	}
	return true;
}

bool ART::Append(IndexLock &lock, DataChunk &appended_data, Vector &row_identifiers) {
	DataChunk expression_result;
	expression_result.Initialize(Allocator::DefaultAllocator(), logical_types);

	// first resolve the expressions for the index
	ExecuteExpressions(appended_data, expression_result);

	// now insert into the index
	return Insert(lock, expression_result, row_identifiers);
}

void ART::VerifyAppend(DataChunk &chunk) {
	ConflictManager conflict_manager(VerifyExistenceType::APPEND, chunk.size());
	LookupValues(chunk, conflict_manager);
}

void ART::VerifyAppend(DataChunk &chunk, ConflictManager &conflict_manager) {
	D_ASSERT(conflict_manager.LookupType() == VerifyExistenceType::APPEND);
	LookupValues(chunk, conflict_manager);
}

void ART::VerifyAppendForeignKey(DataChunk &chunk) {
	ConflictManager conflict_manager(VerifyExistenceType::APPEND_FK, chunk.size());
	LookupValues(chunk, conflict_manager);
}

void ART::VerifyDeleteForeignKey(DataChunk &chunk) {
	if (!IsUnique()) {
		return;
	}
	ConflictManager conflict_manager(VerifyExistenceType::DELETE_FK, chunk.size());
	LookupValues(chunk, conflict_manager);
}

bool ART::InsertToLeaf(Leaf &leaf, row_t row_id) {
#ifdef DEBUG
	for (idx_t k = 0; k < leaf.count; k++) {
		D_ASSERT(leaf.GetRowId(k) != row_id);
	}
#endif
	if (IsUnique() && leaf.count != 0) {
		return false;
	}
	leaf.Insert(*this, row_id);
	return true;
}

bool ART::Insert(Node *&node, Key &key, idx_t depth, row_t row_id) {

	if (!node) {
		// node is currently empty, create a leaf here with the key
		node = Leaf::New(key, depth, row_id);
		this->memory_size += node->MemorySize(*this, false);
		return true;
	}

	if (node->type == NodeType::NLeaf) {
		// replace leaf with Node4 and store both leaves in it
		// or add a row ID to a leaf, if they have the same key
		auto leaf = (Leaf *)node;
		uint32_t new_prefix_length = 0;

		// FIXME: this code (if and while) can be optimized, less branching, see Construct
		// leaf node is already there (its key matches the current key), update row_id vector
		if (new_prefix_length == leaf->prefix.Size() && depth + leaf->prefix.Size() == key.len) {
			return InsertToLeaf(*leaf, row_id);
		}
		while (leaf->prefix[new_prefix_length] == key[depth + new_prefix_length]) {
			new_prefix_length++;
			// leaf node is already there (its key matches the current key), update row_id vector
			if (new_prefix_length == leaf->prefix.Size() && depth + leaf->prefix.Size() == key.len) {
				return InsertToLeaf(*leaf, row_id);
			}
		}

		Node *new_node = Node4::New();
		new_node->prefix = Prefix(key, depth, new_prefix_length);
		this->memory_size += new_node->MemorySize(*this, false);

		auto key_byte = node->prefix.Reduce(*this, new_prefix_length);
		Node4::InsertChild(*this, new_node, key_byte, node);

		Node *leaf_node = Leaf::New(key, depth + new_prefix_length + 1, row_id);
		Node4::InsertChild(*this, new_node, key[depth + new_prefix_length], leaf_node);
		this->memory_size += leaf_node->MemorySize(*this, false);

		node = new_node;
		return true;
	}

	// handle prefix of inner node
	if (node->prefix.Size()) {

		uint32_t mismatch_pos = node->prefix.KeyMismatchPosition(key, depth);
		if (mismatch_pos != node->prefix.Size()) {
			// prefix differs, create new node
			Node *new_node = Node4::New();
			new_node->prefix = Prefix(key, depth, mismatch_pos);
			this->memory_size += new_node->MemorySize(*this, false);

			// break up prefix
			auto key_byte = node->prefix.Reduce(*this, mismatch_pos);
			Node4::InsertChild(*this, new_node, key_byte, node);

			Node *leaf_node = Leaf::New(key, depth + mismatch_pos + 1, row_id);
			Node4::InsertChild(*this, new_node, key[depth + mismatch_pos], leaf_node);
			this->memory_size += leaf_node->MemorySize(*this, false);

			node = new_node;
			return true;
		}
		depth += node->prefix.Size();
	}

	// recurse
	D_ASSERT(depth < key.len);
	idx_t pos = node->GetChildPos(key[depth]);
	if (pos != DConstants::INVALID_INDEX) {
		auto child = node->GetChild(*this, pos);
		bool insertion_result = Insert(child, key, depth + 1, row_id);
		node->ReplaceChildPointer(pos, child);
		return insertion_result;
	}

	Node *leaf_node = Leaf::New(key, depth + 1, row_id);
	Node::InsertChild(*this, node, key[depth], leaf_node);
	this->memory_size += leaf_node->MemorySize(*this, false);
	return true;
}

//===--------------------------------------------------------------------===//
// Delete
//===--------------------------------------------------------------------===//

void ART::Delete(IndexLock &state, DataChunk &input, Vector &row_ids) {

	DataChunk expression;
	expression.Initialize(Allocator::DefaultAllocator(), logical_types);

	// first resolve the expressions
	ExecuteExpressions(input, expression);

	// then generate the keys for the given input
	ArenaAllocator arena_allocator(BufferAllocator::Get(db));
	vector<Key> keys(expression.size());
	GenerateKeys(arena_allocator, expression, keys);

	auto old_memory_size = this->memory_size;

	// now erase the elements from the database
	row_ids.Flatten(input.size());
	auto row_identifiers = FlatVector::GetData<row_t>(row_ids);

	for (idx_t i = 0; i < input.size(); i++) {
		if (keys[i].Empty()) {
			continue;
		}
		Erase(tree, keys[i], 0, row_identifiers[i]);
#ifdef DEBUG
		auto node = Lookup(tree, keys[i], 0);
		if (node) {
			auto leaf = (Leaf *)node;
			for (idx_t k = 0; k < leaf->count; k++) {
				D_ASSERT(leaf->GetRowId(k) != row_identifiers[i]);
			}
		}
#endif
	}

	D_ASSERT(old_memory_size >= memory_size);
	Verify();
	if (track_memory) {
		buffer_manager.DecreaseUsedMemory(old_memory_size - memory_size);
	}
}

void ART::Erase(Node *&node, Key &key, idx_t depth, row_t row_id) {

	if (!node) {
		return;
	}

	// delete a leaf from a tree
	if (node->type == NodeType::NLeaf) {
		auto leaf = (Leaf *)node;
		leaf->Remove(*this, row_id);

		if (leaf->count == 0) {
			D_ASSERT(this->memory_size >= leaf->MemorySize(*this, false));
			this->memory_size -= leaf->MemorySize(*this, false);
			Node::Delete(node);
			node = nullptr;
		}
		return;
	}

	// handle prefix
	if (node->prefix.Size()) {
		if (node->prefix.KeyMismatchPosition(key, depth) != node->prefix.Size()) {
			return;
		}
		depth += node->prefix.Size();
	}

	idx_t pos = node->GetChildPos(key[depth]);
	if (pos != DConstants::INVALID_INDEX) {
		auto child = node->GetChild(*this, pos);
		D_ASSERT(child);

		if (child->type == NodeType::NLeaf) {
			// leaf found, remove entry
			auto leaf = (Leaf *)child;
			leaf->Remove(*this, row_id);

			if (leaf->count == 0) {
				// leaf is empty, delete leaf, decrement node counter and maybe shrink node
				Node::EraseChild(*this, node, pos);
			}

		} else {
			// recurse
			Erase(child, key, depth + 1, row_id);
			node->ReplaceChildPointer(pos, child);
		}
	}
}

//===--------------------------------------------------------------------===//
// Point Query
//===--------------------------------------------------------------------===//

static Key CreateKey(ArenaAllocator &allocator, PhysicalType type, Value &value) {
	D_ASSERT(type == value.type().InternalType());
	switch (type) {
	case PhysicalType::BOOL:
		return Key::CreateKey<bool>(allocator, value);
	case PhysicalType::INT8:
		return Key::CreateKey<int8_t>(allocator, value);
	case PhysicalType::INT16:
		return Key::CreateKey<int16_t>(allocator, value);
	case PhysicalType::INT32:
		return Key::CreateKey<int32_t>(allocator, value);
	case PhysicalType::INT64:
		return Key::CreateKey<int64_t>(allocator, value);
	case PhysicalType::UINT8:
		return Key::CreateKey<uint8_t>(allocator, value);
	case PhysicalType::UINT16:
		return Key::CreateKey<uint16_t>(allocator, value);
	case PhysicalType::UINT32:
		return Key::CreateKey<uint32_t>(allocator, value);
	case PhysicalType::UINT64:
		return Key::CreateKey<uint64_t>(allocator, value);
	case PhysicalType::INT128:
		return Key::CreateKey<hugeint_t>(allocator, value);
	case PhysicalType::FLOAT:
		return Key::CreateKey<float>(allocator, value);
	case PhysicalType::DOUBLE:
		return Key::CreateKey<double>(allocator, value);
	case PhysicalType::VARCHAR:
		return Key::CreateKey<string_t>(allocator, value);
	default:
		throw InternalException("Invalid type for index");
	}
}

bool ART::SearchEqual(Key &key, idx_t max_count, vector<row_t> &result_ids) {

	auto leaf = (Leaf *)(Lookup(tree, key, 0));
	if (!leaf) {
		return true;
	}
	if (leaf->count > max_count) {
		return false;
	}
	for (idx_t i = 0; i < leaf->count; i++) {
		row_t row_id = leaf->GetRowId(i);
		result_ids.push_back(row_id);
	}
	return true;
}

void ART::SearchEqualJoinNoFetch(Key &key, idx_t &result_size) {

	// we need to look for a leaf
	auto leaf = Lookup(tree, key, 0);
	if (!leaf) {
		return;
	}
	result_size = leaf->count;
}

Leaf *ART::Lookup(Node *node, Key &key, idx_t depth) {
	while (node) {
		if (node->type == NodeType::NLeaf) {
			auto leaf = (Leaf *)node;
			auto &leaf_prefix = leaf->prefix;
			//! Check leaf
			for (idx_t i = 0; i < leaf->prefix.Size(); i++) {
				if (leaf_prefix[i] != key[i + depth]) {
					return nullptr;
				}
			}
			return (Leaf *)node;
		}
		if (node->prefix.Size()) {
			for (idx_t pos = 0; pos < node->prefix.Size(); pos++) {
				if (key[depth + pos] != node->prefix[pos]) {
					return nullptr;
				}
			}
			depth += node->prefix.Size();
		}
		idx_t pos = node->GetChildPos(key[depth]);
		if (pos == DConstants::INVALID_INDEX) {
			return nullptr;
		}
		node = node->GetChild(*this, pos);
		D_ASSERT(node);
		depth++;
	}
	return nullptr;
}

//===--------------------------------------------------------------------===//
// Greater Than
// Returns: True (If found leaf >= key)
//          False (Otherwise)
//===--------------------------------------------------------------------===//

bool ART::SearchGreater(ARTIndexScanState *state, Key &key, bool inclusive, idx_t max_count,
                        vector<row_t> &result_ids) {

	Iterator *it = &state->iterator;

	// greater than scan: first set the iterator to the node at which we will start our scan by finding the lowest node
	// that satisfies our requirement
	if (!it->art) {
		it->art = this;
		bool found = it->LowerBound(tree, key, inclusive);
		if (!found) {
			return true;
		}
	}
	// after that we continue the scan; we don't need to check the bounds as any value following this value is
	// automatically bigger and hence satisfies our predicate
	Key empty_key = Key();
	return it->Scan(empty_key, max_count, result_ids, false);
}

//===--------------------------------------------------------------------===//
// Less Than
//===--------------------------------------------------------------------===//

bool ART::SearchLess(ARTIndexScanState *state, Key &upper_bound, bool inclusive, idx_t max_count,
                     vector<row_t> &result_ids) {

	if (!tree) {
		return true;
	}

	Iterator *it = &state->iterator;

	if (!it->art) {
		it->art = this;
		// first find the minimum value in the ART: we start scanning from this value
		it->FindMinimum(*tree);
		// early out min value higher than upper bound query
		if (it->cur_key > upper_bound) {
			return true;
		}
	}
	// now continue the scan until we reach the upper bound
	return it->Scan(upper_bound, max_count, result_ids, inclusive);
}

//===--------------------------------------------------------------------===//
// Closed Range Query
//===--------------------------------------------------------------------===//

bool ART::SearchCloseRange(ARTIndexScanState *state, Key &lower_bound, Key &upper_bound, bool left_inclusive,
                           bool right_inclusive, idx_t max_count, vector<row_t> &result_ids) {

	Iterator *it = &state->iterator;

	// first find the first node that satisfies the left predicate
	if (!it->art) {
		it->art = this;
		bool found = it->LowerBound(tree, lower_bound, left_inclusive);
		if (!found) {
			return true;
		}
	}
	// now continue the scan until we reach the upper bound
	return it->Scan(upper_bound, max_count, result_ids, right_inclusive);
}

bool ART::Scan(Transaction &transaction, DataTable &table, IndexScanState &table_state, idx_t max_count,
               vector<row_t> &result_ids) {

	auto state = (ARTIndexScanState *)&table_state;
	vector<row_t> row_ids;
	bool success;

	// FIXME: the key directly owning the data for a single key might be more efficient
	D_ASSERT(state->values[0].type().InternalType() == types[0]);
	ArenaAllocator arena_allocator(Allocator::Get(db));
	auto key = CreateKey(arena_allocator, types[0], state->values[0]);

	if (state->values[1].IsNull()) {

		// single predicate
		lock_guard<mutex> l(lock);
		switch (state->expressions[0]) {
		case ExpressionType::COMPARE_EQUAL:
			success = SearchEqual(key, max_count, row_ids);
			break;
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			success = SearchGreater(state, key, true, max_count, row_ids);
			break;
		case ExpressionType::COMPARE_GREATERTHAN:
			success = SearchGreater(state, key, false, max_count, row_ids);
			break;
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			success = SearchLess(state, key, true, max_count, row_ids);
			break;
		case ExpressionType::COMPARE_LESSTHAN:
			success = SearchLess(state, key, false, max_count, row_ids);
			break;
		default:
			throw InternalException("Operation not implemented");
		}

	} else {

		// two predicates
		lock_guard<mutex> l(lock);

		D_ASSERT(state->values[1].type().InternalType() == types[0]);
		auto upper_bound = CreateKey(arena_allocator, types[0], state->values[1]);

		bool left_inclusive = state->expressions[0] == ExpressionType ::COMPARE_GREATERTHANOREQUALTO;
		bool right_inclusive = state->expressions[1] == ExpressionType ::COMPARE_LESSTHANOREQUALTO;
		success = SearchCloseRange(state, key, upper_bound, left_inclusive, right_inclusive, max_count, row_ids);
	}

	if (!success) {
		return false;
	}
	if (row_ids.empty()) {
		return true;
	}

	// sort the row ids
	sort(row_ids.begin(), row_ids.end());
	// duplicate eliminate the row ids and append them to the row ids of the state
	result_ids.reserve(row_ids.size());

	result_ids.push_back(row_ids[0]);
	for (idx_t i = 1; i < row_ids.size(); i++) {
		if (row_ids[i] != row_ids[i - 1]) {
			result_ids.push_back(row_ids[i]);
		}
	}
	return true;
}

string ART::GenerateErrorKeyName(DataChunk &input, idx_t row) {
	// re-executing the expressions is not very fast, but we're going to throw anyways, so we don't care
	DataChunk expression_chunk;
	expression_chunk.Initialize(Allocator::DefaultAllocator(), logical_types);
	ExecuteExpressions(input, expression_chunk);

	string key_name;
	for (idx_t k = 0; k < expression_chunk.ColumnCount(); k++) {
		if (k > 0) {
			key_name += ", ";
		}
		key_name += unbound_expressions[k]->GetName() + ": " + expression_chunk.data[k].GetValue(row).ToString();
	}
	return key_name;
}

string ART::GenerateConstraintErrorMessage(VerifyExistenceType verify_type, const string &key_name) {
	switch (verify_type) {
	case VerifyExistenceType::APPEND: {
		// APPEND to PK/UNIQUE table, but node/key already exists in PK/UNIQUE table
		string type = IsPrimary() ? "primary key" : "unique";
		return StringUtil::Format("Duplicate key \"%s\" violates %s constraint", key_name, type);
	}
	case VerifyExistenceType::APPEND_FK: {
		// APPEND_FK to FK table, node/key does not exist in PK/UNIQUE table
		return StringUtil::Format(
		    "Violates foreign key constraint because key \"%s\" does not exist in the referenced table", key_name);
	}
	case VerifyExistenceType::DELETE_FK: {
		// DELETE_FK that still exists in a FK table, i.e., not a valid delete
		return StringUtil::Format("Violates foreign key constraint because key \"%s\" is still referenced by a foreign "
		                          "key in a different table",
		                          key_name);
	}
	default:
		throw NotImplementedException("Type not implemented for VerifyExistenceType");
	}
}

void ART::LookupValues(DataChunk &input, ConflictManager &conflict_manager) {

	// don't alter the index during constraint checking
	lock_guard<mutex> l(lock);

	// first resolve the expressions for the index
	DataChunk expression_chunk;
	expression_chunk.Initialize(Allocator::DefaultAllocator(), logical_types);
	ExecuteExpressions(input, expression_chunk);

	// generate the keys for the given input
	ArenaAllocator arena_allocator(BufferAllocator::Get(db));
	vector<Key> keys(expression_chunk.size());
	GenerateKeys(arena_allocator, expression_chunk, keys);

	idx_t found_conflict = DConstants::INVALID_INDEX;
	for (idx_t i = 0; found_conflict == DConstants::INVALID_INDEX && i < input.size(); i++) {
		if (keys[i].Empty()) {
			if (conflict_manager.AddNull(i)) {
				found_conflict = i;
			}
			continue;
		}
		Leaf *leaf_ptr = Lookup(tree, keys[i], 0);
		if (leaf_ptr == nullptr) {
			if (conflict_manager.AddMiss(i)) {
				found_conflict = i;
			}
			continue;
		}
		// When we find a node, we need to update the 'matches' and 'row_ids'
		// NOTE: Leafs can have more than one row_id, but for UNIQUE/PRIMARY KEY they will only have one
		D_ASSERT(leaf_ptr->count == 1);
		auto row_id = leaf_ptr->GetRowId(0);
		if (conflict_manager.AddHit(i, row_id)) {
			found_conflict = i;
		}
	}
	conflict_manager.FinishLookup();
	if (found_conflict == DConstants::INVALID_INDEX) {
		// No conflicts detected
		return;
	}
	auto key_name = GenerateErrorKeyName(input, found_conflict);
	auto exception_msg = GenerateConstraintErrorMessage(conflict_manager.LookupType(), key_name);
	throw ConstraintException(exception_msg);
}

//===--------------------------------------------------------------------===//
// Serialization
//===--------------------------------------------------------------------===//

BlockPointer ART::Serialize(duckdb::MetaBlockWriter &writer) {
	lock_guard<mutex> l(lock);
	if (tree) {
		serialized_data_pointer = tree->Serialize(*this, writer);
	} else {
		serialized_data_pointer = {(block_id_t)DConstants::INVALID_INDEX, (uint32_t)DConstants::INVALID_INDEX};
	}
	return serialized_data_pointer;
}

//===--------------------------------------------------------------------===//
// Merging
//===--------------------------------------------------------------------===//

bool ART::MergeIndexes(IndexLock &state, Index *other_index) {

	auto other_art = (ART *)other_index;

	if (!this->tree) {
		this->memory_size += other_art->memory_size;
		this->tree = other_art->tree;
		other_art->tree = nullptr;
		return true;
	}

	return Node::MergeARTs(this, other_art);
}

//===--------------------------------------------------------------------===//
// Utility
//===--------------------------------------------------------------------===//

string ART::ToString() {
	if (tree) {
		return tree->ToString(*this);
	}
	return "[empty]";
}

void ART::Verify() {
#ifdef DEBUG
	idx_t current_mem_size = 0;
	if (tree) {
		current_mem_size = tree->MemorySize(*this, true);
	}
	if (memory_size != current_mem_size) {
		throw InternalException("Memory_size value (%d) does not match actual memory size (%d).", memory_size,
		                        current_mem_size);
	}
#endif
}

} // namespace duckdb




namespace duckdb {

Key::Key() : len(0) {
}

Key::Key(data_ptr_t data, idx_t len) : len(len), data(data) {
}

Key::Key(ArenaAllocator &allocator, idx_t len) : len(len) {
	data = allocator.Allocate(len);
}

template <>
Key Key::CreateKey(ArenaAllocator &allocator, string_t value) {
	idx_t len = value.GetSize() + 1;
	auto data = allocator.Allocate(len);
	memcpy(data, value.GetDataUnsafe(), len - 1);

	if (len > 1 && data[len - 2] == '\0') {
		// FIXME: rethink this
		throw NotImplementedException("Indexes cannot have contain null-terminated decoded BLOBs.");
	}
	data[len - 1] = '\0';
	return Key(data, len);
}

template <>
Key Key::CreateKey(ArenaAllocator &allocator, const char *value) {
	return Key::CreateKey(allocator, string_t(value, strlen(value)));
}

template <>
void Key::CreateKey(ArenaAllocator &allocator, Key &key, string_t value) {
	key.len = value.GetSize() + 1;
	key.data = allocator.Allocate(key.len);
	memcpy(key.data, value.GetDataUnsafe(), key.len - 1);

	if (key.len > 1 && key.data[key.len - 2] == '\0') {
		// FIXME: rethink this
		throw NotImplementedException("Indexes cannot have contain null-terminated decoded BLOBs.");
	}
	key.data[key.len - 1] = '\0';
}

template <>
void Key::CreateKey(ArenaAllocator &allocator, Key &key, const char *value) {
	Key::CreateKey(allocator, key, string_t(value, strlen(value)));
}

bool Key::operator>(const Key &k) const {
	for (idx_t i = 0; i < MinValue<idx_t>(len, k.len); i++) {
		if (data[i] > k.data[i]) {
			return true;
		} else if (data[i] < k.data[i]) {
			return false;
		}
	}
	return len > k.len;
}

bool Key::operator<(const Key &k) const {
	for (idx_t i = 0; i < MinValue<idx_t>(len, k.len); i++) {
		if (data[i] < k.data[i]) {
			return true;
		} else if (data[i] > k.data[i]) {
			return false;
		}
	}
	return len < k.len;
}

bool Key::operator>=(const Key &k) const {
	for (idx_t i = 0; i < MinValue<idx_t>(len, k.len); i++) {
		if (data[i] > k.data[i]) {
			return true;
		} else if (data[i] < k.data[i]) {
			return false;
		}
	}
	return len >= k.len;
}

bool Key::operator==(const Key &k) const {
	if (len != k.len) {
		return false;
	}
	for (idx_t i = 0; i < len; i++) {
		if (data[i] != k.data[i]) {
			return false;
		}
	}
	return true;
}

bool Key::ByteMatches(Key &other, idx_t &depth) {
	return data[depth] == other[depth];
}

bool Key::Empty() {
	return len == 0;
}

void Key::ConcatenateKey(ArenaAllocator &allocator, Key &other_key) {

	auto compound_data = allocator.Allocate(len + other_key.len);
	memcpy(compound_data, data, len);
	memcpy(compound_data + len, other_key.data, other_key.len);
	len += other_key.len;
	data = compound_data;
}
} // namespace duckdb




namespace duckdb {
uint8_t &IteratorCurrentKey::operator[](idx_t idx) {
	if (idx >= key.size()) {
		key.push_back(0);
	}
	D_ASSERT(idx < key.size());
	return key[idx];
}

//! Push Byte
void IteratorCurrentKey::Push(uint8_t byte) {
	if (cur_key_pos == key.size()) {
		key.push_back(byte);
	}
	D_ASSERT(cur_key_pos < key.size());
	key[cur_key_pos++] = byte;
}
//! Pops n elements
void IteratorCurrentKey::Pop(idx_t n) {
	cur_key_pos -= n;
	D_ASSERT(cur_key_pos <= key.size());
}

bool IteratorCurrentKey::operator>(const Key &k) const {
	for (idx_t i = 0; i < MinValue<idx_t>(cur_key_pos, k.len); i++) {
		if (key[i] > k.data[i]) {
			return true;
		} else if (key[i] < k.data[i]) {
			return false;
		}
	}
	return cur_key_pos > k.len;
}

bool IteratorCurrentKey::operator>=(const Key &k) const {
	for (idx_t i = 0; i < MinValue<idx_t>(cur_key_pos, k.len); i++) {
		if (key[i] > k.data[i]) {
			return true;
		} else if (key[i] < k.data[i]) {
			return false;
		}
	}
	return cur_key_pos >= k.len;
}

bool IteratorCurrentKey::operator==(const Key &k) const {
	if (cur_key_pos != k.len) {
		return false;
	}
	for (idx_t i = 0; i < cur_key_pos; i++) {
		if (key[i] != k.data[i]) {
			return false;
		}
	}
	return true;
}

void Iterator::FindMinimum(Node &node) {
	Node *next = nullptr;
	idx_t pos = 0;
	// reconstruct the prefix
	for (idx_t i = 0; i < node.prefix.Size(); i++) {
		cur_key.Push(node.prefix[i]);
	}
	switch (node.type) {
	case NodeType::NLeaf:
		last_leaf = (Leaf *)&node;
		return;
	case NodeType::N4: {
		next = ((Node4 &)node).children[0].Unswizzle(*art);
		cur_key.Push(((Node4 &)node).key[0]);
		break;
	}
	case NodeType::N16: {
		next = ((Node16 &)node).children[0].Unswizzle(*art);
		cur_key.Push(((Node16 &)node).key[0]);
		break;
	}
	case NodeType::N48: {
		auto &n48 = (Node48 &)node;
		while (n48.child_index[pos] == Node::EMPTY_MARKER) {
			pos++;
		}
		cur_key.Push(pos);
		next = n48.children[n48.child_index[pos]].Unswizzle(*art);
		break;
	}
	case NodeType::N256: {
		auto &n256 = (Node256 &)node;
		while (!n256.children[pos]) {
			pos++;
		}
		cur_key.Push(pos);
		next = (Node *)n256.children[pos].Unswizzle(*art);
		break;
	}
	}
	nodes.push(IteratorEntry(&node, pos));
	FindMinimum(*next);
}

void Iterator::PushKey(Node *cur_node, uint16_t pos) {
	switch (cur_node->type) {
	case NodeType::N4:
		cur_key.Push(((Node4 *)cur_node)->key[pos]);
		break;
	case NodeType::N16:
		cur_key.Push(((Node16 *)cur_node)->key[pos]);
		break;
	case NodeType::N48:
	case NodeType::N256:
		cur_key.Push(pos);
		break;
	case NodeType::NLeaf:
		break;
	}
}

bool Iterator::Scan(Key &bound, idx_t max_count, vector<row_t> &result_ids, bool is_inclusive) {
	bool has_next;
	do {
		if (!bound.Empty()) {
			if (is_inclusive) {
				if (cur_key > bound) {
					break;
				}
			} else {
				if (cur_key >= bound) {
					break;
				}
			}
		}
		if (result_ids.size() + last_leaf->count > max_count) {
			// adding these elements would exceed the max count
			return false;
		}
		for (idx_t i = 0; i < last_leaf->count; i++) {
			row_t row_id = last_leaf->GetRowId(i);
			result_ids.push_back(row_id);
		}
		has_next = Next();
	} while (has_next);
	return true;
}

void Iterator::PopNode() {
	auto cur_node = nodes.top();
	idx_t elements_to_pop = cur_node.node->prefix.Size() + (nodes.size() != 1);
	cur_key.Pop(elements_to_pop);
	nodes.pop();
}

bool Iterator::Next() {
	if (!nodes.empty()) {
		auto cur_node = nodes.top().node;
		if (cur_node->type == NodeType::NLeaf) {
			// Pop Leaf (We must pop the prefix size + the key to the node (unless we are popping the root)
			PopNode();
		}
	}

	// Look for the next leaf
	while (!nodes.empty()) {
		// cur_node
		auto &top = nodes.top();
		Node *node = top.node;
		if (node->type == NodeType::NLeaf) {
			// found a leaf: move to next node
			last_leaf = (Leaf *)node;
			return true;
		}
		// Find next node
		top.pos = node->GetNextPos(top.pos);
		if (top.pos != DConstants::INVALID_INDEX) {
			// add key-byte of the new node
			PushKey(node, top.pos);
			auto next_node = node->GetChild(*art, top.pos);
			// add prefix of new node
			for (idx_t i = 0; i < next_node->prefix.Size(); i++) {
				cur_key.Push(next_node->prefix[i]);
			}
			// next node found: push it
			nodes.push(IteratorEntry(next_node, DConstants::INVALID_INDEX));
		} else {
			// no node found: move up the tree and Pop prefix and key of current node
			PopNode();
		}
	}
	return false;
}

bool Iterator::LowerBound(Node *node, Key &key, bool inclusive) {
	bool equal = true;
	if (!node) {
		return false;
	}
	idx_t depth = 0;
	while (true) {
		nodes.push(IteratorEntry(node, 0));
		auto &top = nodes.top();
		// reconstruct the prefix
		for (idx_t i = 0; i < top.node->prefix.Size(); i++) {
			cur_key.Push(top.node->prefix[i]);
		}
		// greater case: find leftmost leaf node directly
		if (!equal) {
			while (node->type != NodeType::NLeaf) {
				auto min_pos = node->GetMin();
				PushKey(node, min_pos);
				nodes.push(IteratorEntry(node, min_pos));
				node = node->GetChild(*art, min_pos);
				// reconstruct the prefix
				for (idx_t i = 0; i < node->prefix.Size(); i++) {
					cur_key.Push(node->prefix[i]);
				}
				auto &c_top = nodes.top();
				c_top.node = node;
			}
		}
		if (node->type == NodeType::NLeaf) {
			// found a leaf node: check if it is bigger or equal than the current key
			auto leaf = static_cast<Leaf *>(node);
			last_leaf = leaf;
			// if the search is not inclusive the leaf node could still be equal to the current value
			// check if leaf is equal to the current key
			if (cur_key == key) {
				// if it's not inclusive check if there is a next leaf
				if (!inclusive && !Next()) {
					return false;
				} else {
					return true;
				}
			}

			if (cur_key > key) {
				return true;
			}
			// Case1: When the ART has only one leaf node, the Next() will return false
			// Case2: This means the previous node prefix(if any) + a_key(one element of of key array of previous node)
			// == key[q..=w].
			// But key[w+1..=z] maybe greater than leaf node prefix.
			// One fact is key[w] is alawys equal to a_key and the next element
			// of key array of previous node is always > a_key So we just call Next() once.

			return Next();
		}
		// equal case:
		uint32_t mismatch_pos = node->prefix.KeyMismatchPosition(key, depth);
		if (mismatch_pos != node->prefix.Size()) {
			if (node->prefix[mismatch_pos] < key[depth + mismatch_pos]) {
				// Less
				PopNode();
				return Next();
			} else {
				// Greater
				top.pos = DConstants::INVALID_INDEX;
				return Next();
			}
		}

		// prefix matches, search inside the child for the key
		depth += node->prefix.Size();

		top.pos = node->GetChildGreaterEqual(key[depth], equal);
		// The maximum key byte of the current node is less than the key
		// So fall back to the previous node
		if (top.pos == DConstants::INVALID_INDEX) {
			PopNode();
			return Next();
		}
		PushKey(node, top.pos);
		node = node->GetChild(*art, top.pos);
		// This means all children of this node qualify as geq
		depth++;
	}
}

} // namespace duckdb






#include <cstring>

namespace duckdb {
idx_t Leaf::GetCapacity() const {
	return IsInlined() ? 1 : rowids.ptr[0];
}

bool Leaf::IsInlined() const {
	return count <= 1;
}

row_t Leaf::GetRowId(idx_t index) {
	D_ASSERT(index < count);
	if (IsInlined()) {
		return rowids.inlined;
	} else {
		D_ASSERT(rowids.ptr[0] >= count);
		return rowids.ptr[index + 1];
	}
}

row_t *Leaf::GetRowIds() {
	if (IsInlined()) {
		return &rowids.inlined;
	} else {
		return rowids.ptr + 1;
	}
}

Leaf::Leaf() : Node(NodeType::NLeaf) {
}

Leaf::Leaf(Key &value, uint32_t depth, row_t row_id) : Node(NodeType::NLeaf) {
	count = 1;
	rowids.inlined = row_id;
	D_ASSERT(value.len >= depth);
	prefix = Prefix(value, depth, value.len - depth);
}

Leaf::Leaf(Key &value, uint32_t depth, row_t *row_ids_p, idx_t num_elements_p) : Node(NodeType::NLeaf) {
	D_ASSERT(num_elements_p >= 1);
	if (num_elements_p == 1) {
		// we can inline the row ids
		rowids.inlined = row_ids_p[0];
	} else {
		// new row ids of this leaf
		count = 0;
		Resize(row_ids_p, num_elements_p, num_elements_p);
	}
	count = num_elements_p;
	D_ASSERT(value.len >= depth);
	prefix = Prefix(value, depth, value.len - depth);
}

Leaf::Leaf(row_t *row_ids_p, idx_t num_elements_p, Prefix &prefix_p) : Node(NodeType::NLeaf) {
	D_ASSERT(num_elements_p > 1);
	D_ASSERT(row_ids_p[0] == row_t(num_elements_p)); // first element should contain capacity
	rowids.ptr = row_ids_p;
	count = num_elements_p;
	prefix = prefix_p;
}

Leaf::Leaf(row_t row_id, Prefix &prefix_p) : Node(NodeType::NLeaf) {
	rowids.inlined = row_id;
	count = 1;
	prefix = prefix_p;
}

Leaf::~Leaf() {
	if (!IsInlined()) {
		DeleteArray<row_t>(rowids.ptr, rowids.ptr[0] + 1);
		count = 0;
	}
}

idx_t Leaf::MemorySize(ART &, const bool &) {
	if (IsInlined()) {
		return prefix.MemorySize() + sizeof(*this) + sizeof(row_t);
	}
	return prefix.MemorySize() + sizeof(*this) + sizeof(row_t) * (GetCapacity() + 1);
}

row_t *Leaf::Resize(row_t *current_row_ids, uint32_t current_count, idx_t new_capacity) {
	D_ASSERT(new_capacity >= current_count);
	auto new_allocation = AllocateArray<row_t>(new_capacity + 1);
	new_allocation[0] = new_capacity;
	auto new_row_ids = new_allocation + 1;
	memcpy(new_row_ids, current_row_ids, current_count * sizeof(row_t));
	if (!IsInlined()) {
		// delete the old data
		DeleteArray<row_t>(rowids.ptr, rowids.ptr[0] + 1);
	}
	// set up the new pointers
	rowids.ptr = new_allocation;
	return new_row_ids;
}

void Leaf::Insert(ART &art, row_t row_id) {
	auto capacity = GetCapacity();
	row_t *row_ids = GetRowIds();
	D_ASSERT(count <= capacity);

	if (count == capacity) {
		// grow array
		if (IsInlined()) {
			art.memory_size += (capacity + 1) * sizeof(row_t);
		} else {
			art.memory_size += capacity * sizeof(row_t);
		}
		row_ids = Resize(row_ids, count, capacity * 2);
	}
	// insert new row ID
	row_ids[count++] = row_id;
}

void Leaf::Remove(ART &art, row_t row_id) {
	idx_t entry_offset = DConstants::INVALID_INDEX;
	row_t *row_ids = GetRowIds();

	// find the row ID in the leaf
	for (idx_t i = 0; i < count; i++) {
		if (row_ids[i] == row_id) {
			entry_offset = i;
			break;
		}
	}

	// didn't find the row ID
	if (entry_offset == DConstants::INVALID_INDEX) {
		return;
	}

	// now empty leaf
	if (IsInlined()) {
		D_ASSERT(count == 1);
		count--;
		return;
	}

	count--;
	if (count == 1) {
		// after erasing we can now inline the leaf
		// delete the pointer and inline the remaining rowid
		auto remaining_row_id = row_ids[0] == row_id ? row_ids[1] : row_ids[0];
		DeleteArray<row_t>(rowids.ptr, rowids.ptr[0] + 1);
		rowids.inlined = remaining_row_id;
		D_ASSERT(art.memory_size >= sizeof(row_t));
		art.memory_size -= 2 * sizeof(row_t);
		return;
	}

	// shrink array, if less than half full
	auto capacity = GetCapacity();
	if (capacity > 2 && count < capacity / 2) {

		auto new_capacity = capacity / 2;
		D_ASSERT(art.memory_size >= (capacity - new_capacity) * sizeof(row_t));
		art.memory_size -= (capacity - new_capacity) * sizeof(row_t);

		auto new_allocation = AllocateArray<row_t>(new_capacity + 1);
		new_allocation[0] = new_capacity;

		auto new_row_ids = new_allocation + 1;
		memcpy(new_row_ids, row_ids, entry_offset * sizeof(row_t));
		memcpy(new_row_ids + entry_offset, row_ids + entry_offset + 1, (count - entry_offset) * sizeof(row_t));

		DeleteArray<row_t>(rowids.ptr, rowids.ptr[0] + 1);
		rowids.ptr = new_allocation;

	} else {
		// move the trailing row IDs (after entry_offset)
		memmove(row_ids + entry_offset, row_ids + entry_offset + 1, (count - entry_offset) * sizeof(row_t));
	}
}

string Leaf::ToString(Node *node) {
	Leaf *leaf = (Leaf *)node;
	string str = "Leaf: [";
	auto row_ids = leaf->GetRowIds();
	for (idx_t i = 0; i < leaf->count; i++) {
		str += i == 0 ? to_string(row_ids[i]) : ", " + to_string(row_ids[i]);
	}
	return str + "]";
}

void Leaf::Merge(ART &art, Node *&l_node, Node *&r_node) {
	Leaf *l_n = (Leaf *)l_node;
	Leaf *r_n = (Leaf *)r_node;

	auto l_capacity = l_n->GetCapacity();
	auto l_row_ids = l_n->GetRowIds();
	auto r_row_ids = r_n->GetRowIds();

	if (l_n->count + r_n->count > l_capacity) {
		auto capacity = l_n->GetCapacity();
		auto new_capacity = NextPowerOfTwo(l_n->count + r_n->count);
		art.memory_size += sizeof(row_t) * (new_capacity - capacity);
		l_row_ids = l_n->Resize(l_row_ids, l_n->count, new_capacity);
	}

	// append row_ids to l_n
	memcpy(l_row_ids + l_n->count, r_row_ids, r_n->count * sizeof(row_t));
	l_n->count += r_n->count;
}

BlockPointer Leaf::Serialize(duckdb::MetaBlockWriter &writer) {

	auto ptr = writer.GetBlockPointer();
	writer.Write(type);
	prefix.Serialize(writer);
	writer.Write<uint16_t>(count);

	auto row_ids = GetRowIds();
	for (idx_t i = 0; i < count; i++) {
		writer.Write(row_ids[i]);
	}
	return ptr;
}

void Leaf::Deserialize(ART &art, MetaBlockReader &reader) {

	prefix.Deserialize(reader);
	count = reader.Read<uint16_t>();
	if (count == 1) {
		// inlined
		auto row_id = reader.Read<row_t>();
		rowids.inlined = row_id;

	} else {
		// non-inlined
		auto row_ids = AllocateArray<row_t>(count + 1);
		row_ids[0] = count;
		for (idx_t i = 0; i < count; i++) {
			row_ids[i + 1] = reader.Read<row_t>();
		}
		rowids.ptr = row_ids;
	}
}

} // namespace duckdb








namespace duckdb {

InternalType::InternalType(Node *n) {
	switch (n->type) {
	case NodeType::N4: {
		auto n4 = (Node4 *)n;
		Set(n4->key, 4, n4->children, 4);
		break;
	}
	case NodeType::N16: {
		auto n16 = (Node16 *)n;
		Set(n16->key, 16, n16->children, 16);
		break;
	}
	case NodeType::N48: {
		auto n48 = (Node48 *)n;
		Set(n48->child_index, 256, n48->children, 48);
		break;
	}
	case NodeType::N256: {
		auto n256 = (Node256 *)n;
		Set(nullptr, 0, n256->children, 256);
		break;
	}
	default:
		throw InternalException("This is not an Internal ART Node Type");
	}
}

void InternalType::Set(uint8_t *key_p, uint16_t key_size_p, ARTPointer *children_p, uint16_t children_size_p) {
	key = key_p;
	key_size = key_size_p;
	children = children_p;
	children_size = children_size_p;
}

Node::Node(NodeType type) : count(0), type(type) {
}

// LCOV_EXCL_START
idx_t Node::MemorySize(ART &, const bool &) {
	throw InternalException("MemorySize not implemented for the specific node type.");
}

idx_t Node::GetMin() {
	throw InternalException("GetMin not implemented for the specific node type.");
}

Node *Node::GetChild(ART &art, idx_t pos) {
	throw InternalException("GetChild not implemented for the specific node type.");
}

void Node::ReplaceChildPointer(idx_t pos, Node *node) {
	throw InternalException("ReplaceChildPointer not implemented for the specific node type.");
}

bool Node::ChildIsInMemory(idx_t) {
	throw InternalException("ChildIsInMemory not implemented for the specific node type.");
}
// LCOV_EXCL_STOP

void Node::InsertChild(ART &art, Node *&node, uint8_t key_byte, Node *new_child) {
	switch (node->type) {
	case NodeType::N4:
		Node4::InsertChild(art, node, key_byte, new_child);
		break;
	case NodeType::N16:
		Node16::InsertChild(art, node, key_byte, new_child);
		break;
	case NodeType::N48:
		Node48::InsertChild(art, node, key_byte, new_child);
		break;
	case NodeType::N256:
		Node256::InsertChild(art, node, key_byte, new_child);
		break;
	default:
		throw InternalException("Unrecognized node type for insert.");
	}
}

void Node::EraseChild(ART &art, Node *&node, idx_t pos) {
	switch (node->type) {
	case NodeType::N4: {
		Node4::EraseChild(art, node, pos);
		break;
	}
	case NodeType::N16: {
		Node16::EraseChild(art, node, pos);
		break;
	}
	case NodeType::N48: {
		Node48::EraseChild(art, node, pos);
		break;
	}
	case NodeType::N256:
		Node256::EraseChild(art, node, pos);
		break;
	default:
		throw InternalException("Unrecognized node type for erase.");
	}
}

NodeType Node::GetTypeBySize(idx_t size) {

	if (size <= Node4::GetSize()) {
		return NodeType::N4;
	} else if (size <= Node16::GetSize()) {
		return NodeType::N16;
	} else if (size <= Node48::GetSize()) {
		return NodeType::N48;
	}
	D_ASSERT(size <= Node256::GetSize());
	return NodeType::N256;
}

void Node::New(const NodeType &type, Node *&node) {
	switch (type) {
	case NodeType::N4:
		node = (Node *)Node4::New();
		return;
	case NodeType::N16:
		node = (Node *)Node16::New();
		return;
	case NodeType::N48:
		node = (Node *)Node48::New();
		return;
	case NodeType::N256:
		node = (Node *)Node256::New();
		return;
	default:
		throw InternalException("Unrecognized node type for new node creation.");
	}
}

Node4 *Node4::New() {
	return AllocateObject<Node4>();
}

Node16 *Node16::New() {
	return AllocateObject<Node16>();
}

Node48 *Node48::New() {
	return AllocateObject<Node48>();
}

Node256 *Node256::New() {
	return AllocateObject<Node256>();
}

Leaf *Leaf::New() {
	return AllocateObject<Leaf>();
}

Leaf *Leaf::New(Key &value, uint32_t depth, row_t row_id) {
	return AllocateObject<Leaf>(value, depth, row_id);
}

Leaf *Leaf::New(Key &value, uint32_t depth, row_t *row_ids, idx_t num_elements) {
	return AllocateObject<Leaf>(value, depth, row_ids, num_elements);
}

Leaf *Leaf::New(row_t *row_ids, idx_t num_elements, Prefix &prefix) {
	return AllocateObject<Leaf>(row_ids, num_elements, prefix);
}

Leaf *Leaf::New(row_t row_id, Prefix &prefix) {
	return AllocateObject<Leaf>(row_id, prefix);
}

void Node::Delete(Node *ptr) {
	switch (ptr->type) {
	case NodeType::NLeaf:
		DestroyObject((Leaf *)ptr);
		break;
	case NodeType::N4:
		DestroyObject((Node4 *)ptr);
		break;
	case NodeType::N16:
		DestroyObject((Node16 *)ptr);
		break;
	case NodeType::N48:
		DestroyObject((Node48 *)ptr);
		break;
	case NodeType::N256:
		DestroyObject((Node256 *)ptr);
		break;
	default:
		throw InternalException("Invalid node type for delete.");
	}
}

string Node::ToString(ART &art) {

	string str = "Node";
	switch (this->type) {
	case NodeType::NLeaf:
		return Leaf::ToString(this);
	case NodeType::N4:
		str += to_string(Node4::GetSize());
		break;
	case NodeType::N16:
		str += to_string(Node16::GetSize());
		break;
	case NodeType::N48:
		str += to_string(Node48::GetSize());
		break;
	case NodeType::N256:
		str += to_string(Node256::GetSize());
		break;
	}

	str += ": [";
	auto next_pos = GetNextPos(DConstants::INVALID_INDEX);
	while (next_pos != DConstants::INVALID_INDEX) {
		auto child = GetChild(art, next_pos);
		str += "(" + to_string(next_pos) + ", " + child->ToString(art) + ")";
		next_pos = GetNextPos(next_pos);
	}
	return str + "]";
}

BlockPointer Node::SerializeInternal(ART &art, duckdb::MetaBlockWriter &writer, InternalType &internal_type) {

	// iterate through children and annotate their offsets
	vector<BlockPointer> child_offsets;
	for (idx_t i = 0; i < internal_type.children_size; i++) {
		child_offsets.emplace_back(internal_type.children[i].Serialize(art, writer));
	}
	auto ptr = writer.GetBlockPointer();

	writer.Write(type);
	writer.Write<uint16_t>(count);
	prefix.Serialize(writer);

	// write key values
	for (idx_t i = 0; i < internal_type.key_size; i++) {
		writer.Write(internal_type.key[i]);
	}

	// write child offsets
	for (auto &offsets : child_offsets) {
		writer.Write(offsets.block_id);
		writer.Write(offsets.offset);
	}
	return ptr;
}

BlockPointer Node::Serialize(ART &art, duckdb::MetaBlockWriter &writer) {

	switch (type) {
	case NodeType::N4:
	case NodeType::N16:
	case NodeType::N48:
	case NodeType::N256: {
		InternalType internal_type(this);
		return SerializeInternal(art, writer, internal_type);
	}
	case NodeType::NLeaf: {
		auto leaf = (Leaf *)this;
		return leaf->Serialize(writer);
	}
	default:
		throw InternalException("Invalid ART node for serialize.");
	}
}

void Node::DeserializeInternal(ART &art, duckdb::MetaBlockReader &reader) {

	InternalType internal_type(this);
	count = reader.Read<uint16_t>();
	prefix.Deserialize(reader);

	// read key values
	for (idx_t i = 0; i < internal_type.key_size; i++) {
		internal_type.key[i] = reader.Read<uint8_t>();
	}

	// read child offsets
	for (idx_t i = 0; i < internal_type.children_size; i++) {
		internal_type.children[i] = ARTPointer(reader);
	}
}

Node *Node::Deserialize(ART &art, idx_t block_id, idx_t offset) {

	MetaBlockReader reader(art.table_io_manager.GetIndexBlockManager(), block_id);
	reader.offset = offset;

	auto n = reader.Read<uint8_t>();
	NodeType node_type((NodeType)(n));

	Node *deserialized_node = nullptr;
	auto old_memory_size = art.memory_size;
	switch (node_type) {
	case NodeType::NLeaf: {
		auto leaf = Leaf::New();
		leaf->Deserialize(art, reader);
		art.memory_size += leaf->MemorySize(art, false);
		D_ASSERT(art.memory_size >= old_memory_size);
		if (art.track_memory) {
			art.buffer_manager.IncreaseUsedMemory(art.memory_size - old_memory_size);
		}
		return leaf;
	}
	case NodeType::N4: {
		deserialized_node = (Node *)Node4::New();
		break;
	}
	case NodeType::N16: {
		deserialized_node = (Node *)Node16::New();
		break;
	}
	case NodeType::N48: {
		deserialized_node = (Node *)Node48::New();
		break;
	}
	case NodeType::N256: {
		deserialized_node = (Node *)Node256::New();
		break;
	}
	default:
		throw InternalException("Unrecognized node type");
	}
	deserialized_node->DeserializeInternal(art, reader);
	art.memory_size += deserialized_node->MemorySize(art, false);
	D_ASSERT(art.memory_size >= old_memory_size);
	if (art.track_memory) {
		art.buffer_manager.IncreaseUsedMemory(art.memory_size - old_memory_size);
	}
	return deserialized_node;
}

void UpdateParentsOfNodes(Node *&l_node, Node *&r_node, ParentsOfNodes &parents) {
	if (parents.l_parent) {
		parents.l_parent->ReplaceChildPointer(parents.l_pos, l_node);
	}
	if (parents.r_parent) {
		parents.r_parent->ReplaceChildPointer(parents.r_pos, r_node);
	}
}

// forward declaration
bool ResolvePrefixesAndMerge(MergeInfo &info, ParentsOfNodes &parents);

void SwapNodes(MergeInfo &info, ParentsOfNodes &parents) {
	// adjust the memory sizes
	auto l_node_memory_size = info.l_node->MemorySize(*info.l_art, true);
	auto r_node_memory_size = info.r_node->MemorySize(*info.r_art, true);

	D_ASSERT(info.root_l_art->memory_size >= l_node_memory_size);
	D_ASSERT(info.root_r_art->memory_size >= r_node_memory_size);
	info.root_l_art->memory_size -= l_node_memory_size;
	info.root_r_art->memory_size -= r_node_memory_size;
	info.root_l_art->memory_size += r_node_memory_size;
	info.root_r_art->memory_size += l_node_memory_size;

	// actual swap
	swap(info.l_art, info.r_art);
	swap(info.l_node, info.r_node);
	UpdateParentsOfNodes(info.l_node, info.r_node, parents);
}

bool Merge(MergeInfo &info, ParentsOfNodes &parents) {

	D_ASSERT(info.l_node);
	D_ASSERT(info.r_node);

	// always try to merge the smaller node into the bigger node
	// because maybe there is enough free space in the bigger node to fit the smaller one
	// without too much recursion

	if (info.l_node->type < info.r_node->type) {
		// swap subtrees to ensure that l_node has the bigger node type
		SwapNodes(info, parents);
	}

	if (info.r_node->type == NodeType::NLeaf) {
		D_ASSERT(info.l_node->type == NodeType::NLeaf);
		D_ASSERT(info.r_node->type == NodeType::NLeaf);
		if (info.l_art->IsUnique()) {
			return false;
		}
		Leaf::Merge(*info.root_l_art, info.l_node, info.r_node);
		return true;
	}

	uint8_t key_byte;
	idx_t r_child_pos = DConstants::INVALID_INDEX;

	while (true) {
		r_child_pos = info.r_node->GetNextPosAndByte(r_child_pos, key_byte);
		if (r_child_pos == DConstants::INVALID_INDEX) {
			break;
		}
		auto r_child = info.r_node->GetChild(*info.r_art, r_child_pos);
		auto l_child_pos = info.l_node->GetChildPos(key_byte);

		if (l_child_pos == DConstants::INVALID_INDEX) {
			// insert child at empty position
			auto r_memory_size = r_child->MemorySize(*info.r_art, true);
			Node::InsertChild(*info.root_l_art, info.l_node, key_byte, r_child);

			info.root_l_art->memory_size += r_memory_size;
			D_ASSERT(info.root_r_art->memory_size >= r_memory_size);
			info.root_r_art->memory_size -= r_memory_size;
			if (parents.l_parent) {
				parents.l_parent->ReplaceChildPointer(parents.l_pos, info.l_node);
			}
			info.r_node->ReplaceChildPointer(r_child_pos, nullptr);

		} else {
			// recurse
			auto l_child = info.l_node->GetChild(*info.l_art, l_child_pos);
			MergeInfo child_info(info.l_art, info.r_art, info.root_l_art, info.root_r_art, l_child, r_child);
			ParentsOfNodes child_parents(info.l_node, l_child_pos, info.r_node, r_child_pos);
			if (!ResolvePrefixesAndMerge(child_info, child_parents)) {
				return false;
			}
		}
	}
	return true;
}

bool ResolvePrefixesAndMerge(MergeInfo &info, ParentsOfNodes &parents) {
	// NOTE: we always merge into the left ART

	D_ASSERT(info.l_node);
	D_ASSERT(info.r_node);

	// make sure that r_node has the longer (or equally long) prefix
	if (info.l_node->prefix.Size() > info.r_node->prefix.Size()) {
		SwapNodes(info, parents);
	}

	Node *null_parent = nullptr;
	auto &l_node = info.l_node;
	auto &r_node = info.r_node;
	auto l_prefix_size = l_node->prefix.Size();
	auto r_prefix_size = r_node->prefix.Size();

	auto mismatch_pos = l_node->prefix.MismatchPosition(r_node->prefix);

	// both nodes have no prefix or the same prefix
	if (mismatch_pos == l_prefix_size && l_prefix_size == r_prefix_size) {
		return Merge(info, parents);
	}

	if (mismatch_pos == l_prefix_size) {
		// r_node's prefix contains l_node's prefix
		// l_node cannot be a leaf, otherwise the key represented by l_node would be a subset of another key
		// which is not possible by our construction
		D_ASSERT(l_node->type != NodeType::NLeaf);

		// test if the next byte (mismatch_pos) in r_node (longer prefix) exists in l_node
		auto mismatch_byte = r_node->prefix[mismatch_pos];
		auto child_pos = l_node->GetChildPos(mismatch_byte);

		// update the prefix of r_node to only consist of the bytes after mismatch_pos
		r_node->prefix.Reduce(*info.root_r_art, mismatch_pos);

		// insert r_node as a child of l_node at empty position
		if (child_pos == DConstants::INVALID_INDEX) {

			auto r_memory_size = r_node->MemorySize(*info.r_art, true);
			Node::InsertChild(*info.root_l_art, l_node, mismatch_byte, r_node);

			info.root_l_art->memory_size += r_memory_size;
			D_ASSERT(info.root_r_art->memory_size >= r_memory_size);
			info.root_r_art->memory_size -= r_memory_size;
			UpdateParentsOfNodes(l_node, null_parent, parents);
			r_node = nullptr;
			return true;
		}

		// recurse
		auto child_node = l_node->GetChild(*info.l_art, child_pos);
		MergeInfo child_info(info.l_art, info.r_art, info.root_l_art, info.root_r_art, child_node, r_node);
		ParentsOfNodes child_parents(l_node, child_pos, parents.r_parent, parents.r_pos);
		return ResolvePrefixesAndMerge(child_info, child_parents);
	}

	// prefixes differ, create new node and insert both nodes as children

	// create new node
	Node *new_node = Node4::New();
	new_node->prefix = Prefix(l_node->prefix, mismatch_pos);
	info.root_l_art->memory_size += new_node->MemorySize(*info.l_art, false);

	// insert l_node, break up prefix of l_node
	auto key_byte = l_node->prefix.Reduce(*info.root_l_art, mismatch_pos);
	Node4::InsertChild(*info.root_l_art, new_node, key_byte, l_node);

	// insert r_node, break up prefix of r_node
	key_byte = r_node->prefix.Reduce(*info.root_r_art, mismatch_pos);
	auto r_memory_size = r_node->MemorySize(*info.r_art, true);
	Node4::InsertChild(*info.root_l_art, new_node, key_byte, r_node);

	info.root_l_art->memory_size += r_memory_size;
	D_ASSERT(info.root_r_art->memory_size >= r_memory_size);
	info.root_r_art->memory_size -= r_memory_size;

	l_node = new_node;
	UpdateParentsOfNodes(l_node, null_parent, parents);
	r_node = nullptr;
	return true;
}

bool Node::MergeARTs(ART *l_art, ART *r_art) {

	Node *null_parent = nullptr;
	MergeInfo info(l_art, r_art, l_art, r_art, l_art->tree, r_art->tree);
	ParentsOfNodes parents(null_parent, 0, null_parent, 0);
	return ResolvePrefixesAndMerge(info, parents);
}

idx_t Node::RecursiveMemorySize(ART &art) {

	// get the size of all children
	auto memory_size_children = 0;

	auto next_pos = GetNextPos(DConstants::INVALID_INDEX);
	while (next_pos != DConstants::INVALID_INDEX) {
		if (ChildIsInMemory(next_pos)) {
			auto child = GetChild(art, next_pos);
			memory_size_children += child->MemorySize(art, true);
		}
		next_pos = GetNextPos(next_pos);
	}

	return memory_size_children;
}

} // namespace duckdb






#include <cstring>

namespace duckdb {

Node16::Node16() : Node(NodeType::N16) {
	memset(key, 16, sizeof(key));
}

idx_t Node16::MemorySize(ART &art, const bool &recurse) {
	if (recurse) {
		return prefix.MemorySize() + sizeof(*this) + RecursiveMemorySize(art);
	}
	return prefix.MemorySize() + sizeof(*this);
}

idx_t Node16::GetChildPos(uint8_t k) {
	for (idx_t pos = 0; pos < count; pos++) {
		if (key[pos] == k) {
			return pos;
		}
	}
	return Node::GetChildPos(k);
}

idx_t Node16::GetChildGreaterEqual(uint8_t k, bool &equal) {
	for (idx_t pos = 0; pos < count; pos++) {
		if (key[pos] >= k) {
			if (key[pos] == k) {
				equal = true;
			} else {
				equal = false;
			}

			return pos;
		}
	}
	return DConstants::INVALID_INDEX;
}

idx_t Node16::GetMin() {
	return 0;
}

idx_t Node16::GetNextPos(idx_t pos) {
	if (pos == DConstants::INVALID_INDEX) {
		return 0;
	}
	pos++;
	return pos < count ? pos : DConstants::INVALID_INDEX;
}

idx_t Node16::GetNextPosAndByte(idx_t pos, uint8_t &byte) {
	if (pos == DConstants::INVALID_INDEX) {
		byte = key[0];
		return 0;
	}
	pos++;
	if (pos < count) {
		byte = key[pos];
		return pos;
	}
	return DConstants::INVALID_INDEX;
}

Node *Node16::GetChild(ART &art, idx_t pos) {
	D_ASSERT(pos < count);
	return children[pos].Unswizzle(art);
}

void Node16::ReplaceChildPointer(idx_t pos, Node *node) {
	children[pos] = node;
}

bool Node16::ChildIsInMemory(idx_t pos) {
	return children[pos] && !children[pos].IsSwizzled();
}

void Node16::InsertChild(ART &art, Node *&node, uint8_t key_byte, Node *new_child) {
	Node16 *n = (Node16 *)node;

	// insert new child node into node
	if (n->count < Node16::GetSize()) {
		// still space, just insert the child
		idx_t pos = 0;
		while (pos < node->count && n->key[pos] < key_byte) {
			pos++;
		}
		if (n->children[pos]) {
			for (idx_t i = n->count; i > pos; i--) {
				n->key[i] = n->key[i - 1];
				n->children[i] = n->children[i - 1];
			}
		}
		n->key[pos] = key_byte;
		n->children[pos] = new_child;
		n->count++;

	} else {
		// node is full, grow to Node48
		auto new_node = Node48::New();
		art.memory_size += new_node->MemorySize(art, false);
		new_node->count = node->count;
		new_node->prefix = std::move(n->prefix);

		for (idx_t i = 0; i < node->count; i++) {
			new_node->child_index[n->key[i]] = i;
			new_node->children[i] = n->children[i];
			n->children[i] = nullptr;
		}

		D_ASSERT(art.memory_size >= node->MemorySize(art, false));
		art.memory_size -= node->MemorySize(art, false);
		Node::Delete(node);
		node = new_node;
		Node48::InsertChild(art, node, key_byte, new_child);
	}
}

void Node16::EraseChild(ART &art, Node *&node, idx_t pos) {

	auto n = (Node16 *)node;
	D_ASSERT(pos < n->count);

	// adjust the ART size
	if (n->ChildIsInMemory(pos)) {
		auto child = n->GetChild(art, pos);
		D_ASSERT(art.memory_size >= child->MemorySize(art, true));
		art.memory_size -= child->MemorySize(art, true);
	}

	// erase the child and decrease the count
	n->children[pos].Reset();
	n->count--;

	// potentially move any children backwards
	for (; pos < n->count; pos++) {
		n->key[pos] = n->key[pos + 1];
		n->children[pos] = n->children[pos + 1];
	}
	// set any remaining nodes as nullptr
	for (; pos < Node16::GetSize(); pos++) {
		if (!n->children[pos]) {
			break;
		}
		n->children[pos] = nullptr;
	}

	// shrink node to Node4
	if (node->count < Node4::GetSize()) {

		auto new_node = Node4::New();
		art.memory_size += new_node->MemorySize(art, false);
		new_node->prefix = std::move(n->prefix);

		for (idx_t i = 0; i < n->count; i++) {
			new_node->key[new_node->count] = n->key[i];
			new_node->children[new_node->count++] = n->children[i];
			n->children[i] = nullptr;
		}

		D_ASSERT(art.memory_size >= node->MemorySize(art, false));
		art.memory_size -= node->MemorySize(art, false);
		Node::Delete(node);
		node = new_node;
	}
}
} // namespace duckdb





namespace duckdb {

Node256::Node256() : Node(NodeType::N256) {
}

idx_t Node256::MemorySize(ART &art, const bool &recurse) {
	if (recurse) {
		return prefix.MemorySize() + sizeof(*this) + RecursiveMemorySize(art);
	}
	return prefix.MemorySize() + sizeof(*this);
}

idx_t Node256::GetChildPos(uint8_t k) {
	if (children[k]) {
		return k;
	} else {
		return DConstants::INVALID_INDEX;
	}
}

idx_t Node256::GetChildGreaterEqual(uint8_t k, bool &equal) {
	for (idx_t pos = k; pos < Node256::GetSize(); pos++) {
		if (children[pos]) {
			if (pos == k) {
				equal = true;
			} else {
				equal = false;
			}
			return pos;
		}
	}
	return DConstants::INVALID_INDEX;
}

idx_t Node256::GetMin() {
	for (idx_t i = 0; i < Node256::GetSize(); i++) {
		if (children[i]) {
			return i;
		}
	}
	return DConstants::INVALID_INDEX;
}

idx_t Node256::GetNextPos(idx_t pos) {
	pos == DConstants::INVALID_INDEX ? pos = 0 : pos++;
	for (; pos < Node256::GetSize(); pos++) {
		if (children[pos]) {
			return pos;
		}
	}
	return Node::GetNextPos(pos);
}

idx_t Node256::GetNextPosAndByte(idx_t pos, uint8_t &byte) {
	pos == DConstants::INVALID_INDEX ? pos = 0 : pos++;
	for (; pos < Node256::GetSize(); pos++) {
		if (children[pos]) {
			byte = uint8_t(pos);
			return pos;
		}
	}
	return Node::GetNextPos(pos);
}

Node *Node256::GetChild(ART &art, idx_t pos) {
	return children[pos].Unswizzle(art);
}

void Node256::ReplaceChildPointer(idx_t pos, Node *node) {
	children[pos] = node;
}

bool Node256::ChildIsInMemory(idx_t pos) {
	return children[pos] && !children[pos].IsSwizzled();
}

void Node256::InsertChild(ART &, Node *&node, uint8_t key_byte, Node *new_child) {
	auto n = (Node256 *)(node);

	n->count++;
	n->children[key_byte] = new_child;
}

void Node256::EraseChild(ART &art, Node *&node, idx_t pos) {
	auto n = (Node256 *)(node);

	// adjust the ART size
	if (n->ChildIsInMemory(pos)) {
		auto child = n->GetChild(art, pos);
		D_ASSERT(art.memory_size >= child->MemorySize(art, true));
		art.memory_size -= child->MemorySize(art, true);
	}

	// erase the child and decrease the count
	n->children[pos].Reset();
	n->count--;

	// shrink node to Node48
	if (node->count <= NODE_256_SHRINK_THRESHOLD) {

		auto new_node = Node48::New();
		art.memory_size += new_node->MemorySize(art, false);
		new_node->prefix = std::move(n->prefix);

		for (idx_t i = 0; i < Node256::GetSize(); i++) {
			if (n->children[i]) {
				new_node->child_index[i] = new_node->count;
				new_node->children[new_node->count++] = n->children[i];
				n->children[i] = nullptr;
			}
		}

		D_ASSERT(art.memory_size >= node->MemorySize(art, false));
		art.memory_size -= node->MemorySize(art, false);
		Node::Delete(node);
		node = new_node;
	}
}
} // namespace duckdb






namespace duckdb {

Node4::Node4() : Node(NodeType::N4) {
	memset(key, 0, sizeof(key));
}

idx_t Node4::MemorySize(ART &art, const bool &recurse) {
	if (recurse) {
		return prefix.MemorySize() + sizeof(*this) + RecursiveMemorySize(art);
	}
	return prefix.MemorySize() + sizeof(*this);
}

idx_t Node4::GetChildPos(uint8_t k) {
	for (idx_t pos = 0; pos < count; pos++) {
		if (key[pos] == k) {
			return pos;
		}
	}
	return Node::GetChildPos(k);
}

idx_t Node4::GetChildGreaterEqual(uint8_t k, bool &equal) {
	for (idx_t pos = 0; pos < count; pos++) {
		if (key[pos] >= k) {
			if (key[pos] == k) {
				equal = true;
			} else {
				equal = false;
			}
			return pos;
		}
	}
	return DConstants::INVALID_INDEX;
}

idx_t Node4::GetMin() {
	return 0;
}

idx_t Node4::GetNextPos(idx_t pos) {
	if (pos == DConstants::INVALID_INDEX) {
		return 0;
	}
	pos++;
	return pos < count ? pos : DConstants::INVALID_INDEX;
}

idx_t Node4::GetNextPosAndByte(idx_t pos, uint8_t &byte) {
	if (pos == DConstants::INVALID_INDEX) {
		byte = key[0];
		return 0;
	}
	pos++;
	if (pos < count) {
		byte = key[pos];
		return pos;
	}
	return DConstants::INVALID_INDEX;
}

Node *Node4::GetChild(ART &art, idx_t pos) {
	D_ASSERT(pos < count);
	return children[pos].Unswizzle(art);
}

void Node4::ReplaceChildPointer(idx_t pos, Node *node) {
	children[pos] = node;
}

bool Node4::ChildIsInMemory(idx_t pos) {
	return children[pos] && !children[pos].IsSwizzled();
}

void Node4::InsertChild(ART &art, Node *&node, uint8_t key_byte, Node *new_child) {
	Node4 *n = (Node4 *)node;

	// insert new child node into node
	if (node->count < Node4::GetSize()) {
		// still space, just insert the child
		idx_t pos = 0;
		while ((pos < node->count) && (n->key[pos] < key_byte)) {
			pos++;
		}
		if (n->children[pos]) {
			for (idx_t i = n->count; i > pos; i--) {
				n->key[i] = n->key[i - 1];
				n->children[i] = n->children[i - 1];
			}
		}
		n->key[pos] = key_byte;
		n->children[pos] = new_child;
		n->count++;

	} else {
		// node is full, grow to Node16
		auto new_node = Node16::New();
		art.memory_size += new_node->MemorySize(art, false);
		new_node->count = n->count;
		new_node->prefix = std::move(node->prefix);

		for (idx_t i = 0; i < n->count; i++) {
			new_node->key[i] = n->key[i];
			new_node->children[i] = n->children[i];
			n->children[i] = nullptr;
		}
		n->count = 0;

		D_ASSERT(art.memory_size >= node->MemorySize(art, false));
		art.memory_size -= node->MemorySize(art, false);
		Node::Delete(node);
		node = new_node;
		Node16::InsertChild(art, node, key_byte, new_child);
	}
}

void Node4::EraseChild(ART &art, Node *&node, idx_t pos) {

	Node4 *n = (Node4 *)node;
	D_ASSERT(pos < n->count);
	D_ASSERT(n->count > 1);

	// adjust the ART size
	if (n->ChildIsInMemory(pos)) {
		auto child = n->GetChild(art, pos);
		D_ASSERT(art.memory_size >= child->MemorySize(art, true));
		art.memory_size -= child->MemorySize(art, true);
	}

	// erase the child and decrease the count
	n->children[pos].Reset();
	n->count--;
	D_ASSERT(n->count >= 1);

	// potentially move any children backwards
	for (; pos < n->count; pos++) {
		n->key[pos] = n->key[pos + 1];
		n->children[pos] = n->children[pos + 1];
	}
	// set any remaining nodes as nullptr
	for (; pos < Node4::GetSize(); pos++) {
		n->children[pos] = nullptr;
	}

	// this is a one way node, compress
	if (n->count == 1) {

		// get only child and concatenate prefixes
		auto child_ref = n->GetChild(art, 0);
		// concatenate prefixes
		child_ref->prefix.Concatenate(art, n->key[0], node->prefix);
		// ensure that when deleting the node, we do not delete the child (because we move it)
		n->children[0] = nullptr;

		D_ASSERT(art.memory_size >= n->MemorySize(art, false));
		art.memory_size -= n->MemorySize(art, false);
		Node::Delete(node);
		node = child_ref;
	}
}
} // namespace duckdb






namespace duckdb {

Node48::Node48() : Node(NodeType::N48) {
	for (idx_t i = 0; i < 256; i++) {
		child_index[i] = Node::EMPTY_MARKER;
	}
}

idx_t Node48::MemorySize(ART &art, const bool &recurse) {
	if (recurse) {
		return prefix.MemorySize() + sizeof(*this) + RecursiveMemorySize(art);
	}
	return prefix.MemorySize() + sizeof(*this);
}

idx_t Node48::GetChildPos(uint8_t k) {
	if (child_index[k] == Node::EMPTY_MARKER) {
		return DConstants::INVALID_INDEX;
	} else {
		return k;
	}
}

idx_t Node48::GetChildGreaterEqual(uint8_t k, bool &equal) {
	for (idx_t pos = k; pos < Node256::GetSize(); pos++) {
		if (child_index[pos] != Node::EMPTY_MARKER) {
			if (pos == k) {
				equal = true;
			} else {
				equal = false;
			}
			return pos;
		}
	}
	return DConstants::INVALID_INDEX;
}

idx_t Node48::GetMin() {
	for (idx_t i = 0; i < Node256::GetSize(); i++) {
		if (child_index[i] != Node::EMPTY_MARKER) {
			return i;
		}
	}
	return DConstants::INVALID_INDEX;
}

idx_t Node48::GetNextPos(idx_t pos) {
	pos == DConstants::INVALID_INDEX ? pos = 0 : pos++;
	for (; pos < Node256::GetSize(); pos++) {
		if (child_index[pos] != Node::EMPTY_MARKER) {
			return pos;
		}
	}
	return Node::GetNextPos(pos);
}

idx_t Node48::GetNextPosAndByte(idx_t pos, uint8_t &byte) {
	pos == DConstants::INVALID_INDEX ? pos = 0 : pos++;
	for (; pos < Node256::GetSize(); pos++) {
		if (child_index[pos] != Node::EMPTY_MARKER) {
			byte = uint8_t(pos);
			return pos;
		}
	}
	return Node::GetNextPos(pos);
}

Node *Node48::GetChild(ART &art, idx_t pos) {
	D_ASSERT(child_index[pos] != Node::EMPTY_MARKER);
	return children[child_index[pos]].Unswizzle(art);
}

void Node48::ReplaceChildPointer(idx_t pos, Node *node) {
	children[child_index[pos]] = node;
}

bool Node48::ChildIsInMemory(idx_t pos) {
	return children[child_index[pos]] && !children[child_index[pos]].IsSwizzled();
}

void Node48::InsertChild(ART &art, Node *&node, uint8_t key_byte, Node *new_child) {
	auto n = (Node48 *)node;

	// insert new child node into node
	if (node->count < Node48::GetSize()) {
		// still space, just insert the child
		idx_t pos = n->count;
		if (n->children[pos]) {
			// find an empty position in the node list if the current position is occupied
			pos = 0;
			while (n->children[pos]) {
				pos++;
			}
		}
		n->children[pos] = new_child;
		n->child_index[key_byte] = pos;
		n->count++;

	} else {
		// node is full, grow to Node256
		auto new_node = Node256::New();
		art.memory_size += new_node->MemorySize(art, false);
		new_node->count = n->count;
		new_node->prefix = std::move(n->prefix);

		for (idx_t i = 0; i < Node256::GetSize(); i++) {
			if (n->child_index[i] != Node::EMPTY_MARKER) {
				new_node->children[i] = n->children[n->child_index[i]];
				n->children[n->child_index[i]] = nullptr;
			}
		}

		D_ASSERT(art.memory_size >= node->MemorySize(art, false));
		art.memory_size -= node->MemorySize(art, false);
		Node::Delete(node);
		node = new_node;
		Node256::InsertChild(art, node, key_byte, new_child);
	}
}

void Node48::EraseChild(ART &art, Node *&node, idx_t pos) {
	auto n = (Node48 *)(node);

	// adjust the ART size
	if (n->ChildIsInMemory(pos)) {
		auto child = n->GetChild(art, pos);
		D_ASSERT(art.memory_size >= child->MemorySize(art, true));
		art.memory_size -= child->MemorySize(art, true);
	}

	// erase the child and decrease the count
	n->children[n->child_index[pos]].Reset();
	n->child_index[pos] = Node::EMPTY_MARKER;
	n->count--;

	// shrink node to Node16
	if (node->count < NODE_48_SHRINK_THRESHOLD) {

		auto new_node = Node16::New();
		art.memory_size += new_node->MemorySize(art, false);
		new_node->prefix = std::move(n->prefix);

		for (idx_t i = 0; i < Node256::GetSize(); i++) {
			if (n->child_index[i] != Node::EMPTY_MARKER) {
				new_node->key[new_node->count] = i;
				new_node->children[new_node->count++] = n->children[n->child_index[i]];
				n->children[n->child_index[i]] = nullptr;
			}
		}

		D_ASSERT(art.memory_size >= node->MemorySize(art, false));
		art.memory_size -= node->MemorySize(art, false);
		Node::Delete(node);
		node = new_node;
	}
}
} // namespace duckdb




namespace duckdb {

bool Prefix::IsInlined() const {
	return size <= PREFIX_INLINE_BYTES;
}

uint8_t *Prefix::GetPrefixData() {
	return IsInlined() ? &value.inlined[0] : value.ptr;
}

const uint8_t *Prefix::GetPrefixData() const {
	return IsInlined() ? &value.inlined[0] : value.ptr;
}

uint8_t *Prefix::AllocatePrefix(uint32_t size) {
	Destroy();

	this->size = size;
	uint8_t *prefix;
	if (IsInlined()) {
		prefix = &value.inlined[0];
	} else {
		// allocate new prefix
		value.ptr = AllocateArray<uint8_t>(size);
		prefix = value.ptr;
	}
	return prefix;
}

Prefix::Prefix() : size(0) {
}

Prefix::Prefix(Key &key, uint32_t depth, uint32_t size) : size(0) {
	auto prefix = AllocatePrefix(size);

	// copy key to prefix
	idx_t prefix_idx = 0;
	for (idx_t i = depth; i < size + depth; i++) {
		prefix[prefix_idx++] = key.data[i];
	}
}

Prefix::Prefix(Prefix &other_prefix, uint32_t size) : size(0) {
	auto prefix = AllocatePrefix(size);

	// copy key to Prefix
	auto other_data = other_prefix.GetPrefixData();
	for (idx_t i = 0; i < size; i++) {
		prefix[i] = other_data[i];
	}
}

Prefix::~Prefix() {
	Destroy();
}

idx_t Prefix::MemorySize() {
	return sizeof(*this) + sizeof(uint8_t) * size;
}

void Prefix::Destroy() {
	if (!IsInlined()) {
		DeleteArray<uint8_t>(value.ptr, size);
		size = 0;
	}
}

uint8_t &Prefix::operator[](idx_t idx) {
	D_ASSERT(idx < Size());
	return GetPrefixData()[idx];
}

Prefix &Prefix::operator=(const Prefix &src) {
	auto prefix = AllocatePrefix(src.size);

	// copy prefix
	auto src_prefix = src.GetPrefixData();
	for (idx_t i = 0; i < src.size; i++) {
		prefix[i] = src_prefix[i];
	}
	size = src.size;
	return *this;
}

Prefix &Prefix::operator=(Prefix &&other) noexcept {
	std::swap(size, other.size);
	std::swap(value, other.value);
	return *this;
}

void Prefix::Overwrite(uint32_t new_size, uint8_t *data) {
	if (new_size <= PREFIX_INLINE_BYTES) {
		// new entry would be inlined
		// inline the data and destroy the pointer
		auto prefix = AllocatePrefix(new_size);
		for (idx_t i = 0; i < new_size; i++) {
			prefix[i] = data[i];
		}
		DeleteArray<uint8_t>(data, new_size);

	} else {
		// new entry would not be inlined
		// take over the data directly
		Destroy();
		size = new_size;
		value.ptr = data;
	}
}

void Prefix::Concatenate(ART &art, uint8_t key, Prefix &other) {
	auto new_size = size + 1 + other.size;
	art.memory_size += (new_size - size) * sizeof(uint8_t);
	// have to allocate space in our prefix array
	auto new_prefix = AllocateArray<uint8_t>(new_size);
	idx_t new_prefix_idx = 0;

	// 1) add the to-be deleted node's prefix
	for (uint32_t i = 0; i < other.size; i++) {
		new_prefix[new_prefix_idx++] = other[i];
	}

	// 2) now move the current partial key byte as part of the prefix
	new_prefix[new_prefix_idx++] = key;

	// 3) move the existing prefix (if any)
	auto prefix = GetPrefixData();
	for (uint32_t i = 0; i < size; i++) {
		new_prefix[new_prefix_idx++] = prefix[i];
	}
	Overwrite(new_size, new_prefix);
}

uint8_t Prefix::Reduce(ART &art, uint32_t n) {
	auto new_size = size - n - 1;
	D_ASSERT(art.memory_size >= (size - new_size) * sizeof(uint8_t));
	art.memory_size -= (size - new_size) * sizeof(uint8_t);
	auto prefix = GetPrefixData();
	auto partial_key = prefix[n];

	if (new_size == 0) {
		Destroy();
		size = 0;
		return partial_key;
	}
	auto new_prefix = AllocateArray<uint8_t>(new_size);
	for (idx_t i = 0; i < new_size; i++) {
		new_prefix[i] = prefix[i + n + 1];
	}
	Overwrite(new_size, new_prefix);
	return partial_key;
}

void Prefix::Serialize(duckdb::MetaBlockWriter &writer) {
	writer.Write(size);
	auto prefix = GetPrefixData();
	writer.WriteData(prefix, size);
}

void Prefix::Deserialize(duckdb::MetaBlockReader &reader) {
	auto prefix_size = reader.Read<uint32_t>();
	auto prefix = AllocatePrefix(prefix_size);
	this->size = prefix_size;
	reader.ReadData(prefix, size);
}

uint32_t Prefix::KeyMismatchPosition(Key &key, uint32_t depth) {
	uint64_t pos;
	auto prefix = GetPrefixData();
	for (pos = 0; pos < size; pos++) {
		if (key[depth + pos] != prefix[pos]) {
			return pos;
		}
	}
	return pos;
}

uint32_t Prefix::MismatchPosition(Prefix &other) {
	auto prefix = GetPrefixData();
	auto other_data = other.GetPrefixData();
	for (idx_t i = 0; i < size; i++) {
		if (prefix[i] != other_data[i]) {
			return i;
		}
	}
	return size;
}

} // namespace duckdb




namespace duckdb {
SwizzleablePointer::~SwizzleablePointer() {
	if (pointer) {
		if (!IsSwizzled()) {
			Node::Delete((Node *)pointer);
		}
	}
}

SwizzleablePointer::SwizzleablePointer(duckdb::MetaBlockReader &reader) {
	idx_t block_id = reader.Read<block_id_t>();
	uint32_t offset = reader.Read<uint32_t>();
	if (block_id == DConstants::INVALID_INDEX || offset == (uint32_t)DConstants::INVALID_INDEX) {
		pointer = 0;
		return;
	}
	idx_t pointer_size = sizeof(pointer) * 8;
	pointer = block_id;
	// This assumes high 32 bits of pointer are zero.
	pointer = pointer << (pointer_size / 2);
	D_ASSERT((pointer >> (pointer_size / 2)) == block_id);
	pointer += offset;
	// Set the left most bit to indicate this is a swizzled pointer and send it back to the mother-ship
	uint64_t mask = 1;
	mask = mask << (pointer_size - 1);
	// This assumes the 33rd most significant bit of the block_id is zero.
	pointer |= mask;
}

SwizzleablePointer &SwizzleablePointer::operator=(const Node *ptr) {
	// If the object already has a non-swizzled pointer, this will leak memory.
	//
	// TODO: If enabled, this assert will fire, indicating a possible leak. If an exception
	// is thrown here, it will cause a double-free. There is some work to do to make all this safer.
	// D_ASSERT(empty() || IsSwizzled());
	if (sizeof(ptr) == 4) {
		pointer = (uint32_t)(size_t)ptr;
	} else {
		pointer = (uint64_t)ptr;
	}
	return *this;
}

bool operator!=(const SwizzleablePointer &s_ptr, const uint64_t &ptr) {
	return (s_ptr.pointer != ptr);
}

BlockPointer SwizzleablePointer::GetSwizzledBlockInfo() {
	D_ASSERT(IsSwizzled());
	idx_t pointer_size = sizeof(pointer) * 8;
	// This is destructive. Pointer will be invalid after this operation.
	// That's okay because this is only ever called from Unswizzle.
	pointer = pointer & ~(1ULL << (pointer_size - 1));
	uint32_t block_id = pointer >> (pointer_size / 2);
	uint32_t offset = pointer & 0xffffffff;
	return {block_id, offset};
}

bool SwizzleablePointer::IsSwizzled() {
	idx_t pointer_size = sizeof(pointer) * 8;
	return (pointer >> (pointer_size - 1)) & 1;
}

void SwizzleablePointer::Reset() {
	if (pointer) {
		if (!IsSwizzled()) {
			Node::Delete((Node *)pointer);
		}
	}
	*this = nullptr;
}

Node *SwizzleablePointer::Unswizzle(ART &art) {
	if (IsSwizzled()) {
		// This means our pointer is not yet in memory, gotta deserialize this
		// first we unset the bae
		auto block_info = GetSwizzledBlockInfo();
		*this = Node::Deserialize(art, block_info.block_id, block_info.offset);
		art.Verify();
	}
	return (Node *)pointer;
}

BlockPointer SwizzleablePointer::Serialize(ART &art, duckdb::MetaBlockWriter &writer) {
	if (pointer) {
		Unswizzle(art);
		return ((Node *)pointer)->Serialize(art, writer);
	} else {
		return {(block_id_t)DConstants::INVALID_INDEX, (uint32_t)DConstants::INVALID_INDEX};
	}
}

} // namespace duckdb











namespace duckdb {

using ValidityBytes = JoinHashTable::ValidityBytes;
using ScanStructure = JoinHashTable::ScanStructure;
using ProbeSpill = JoinHashTable::ProbeSpill;
using ProbeSpillLocalState = JoinHashTable::ProbeSpillLocalAppendState;

JoinHashTable::JoinHashTable(BufferManager &buffer_manager, const vector<JoinCondition> &conditions,
                             vector<LogicalType> btypes, JoinType type)
    : buffer_manager(buffer_manager), conditions(conditions), build_types(std::move(btypes)), entry_size(0),
      tuple_size(0), vfound(Value::BOOLEAN(false)), join_type(type), finalized(false), has_null(false), external(false),
      radix_bits(4), tuples_per_round(0), partition_start(0), partition_end(0) {
	for (auto &condition : conditions) {
		D_ASSERT(condition.left->return_type == condition.right->return_type);
		auto type = condition.left->return_type;
		if (condition.comparison == ExpressionType::COMPARE_EQUAL ||
		    condition.comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM ||
		    condition.comparison == ExpressionType::COMPARE_DISTINCT_FROM) {
			// all equality conditions should be at the front
			// all other conditions at the back
			// this assert checks that
			D_ASSERT(equality_types.size() == condition_types.size());
			equality_types.push_back(type);
		}

		predicates.push_back(condition.comparison);
		null_values_are_equal.push_back(condition.comparison == ExpressionType::COMPARE_DISTINCT_FROM ||
		                                condition.comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM);

		condition_types.push_back(type);
	}
	// at least one equality is necessary
	D_ASSERT(!equality_types.empty());

	// Types for the layout
	vector<LogicalType> layout_types(condition_types);
	layout_types.insert(layout_types.end(), build_types.begin(), build_types.end());
	if (IsRightOuterJoin(join_type)) {
		// full/right outer joins need an extra bool to keep track of whether or not a tuple has found a matching entry
		// we place the bool before the NEXT pointer
		layout_types.emplace_back(LogicalType::BOOLEAN);
	}
	layout_types.emplace_back(LogicalType::HASH);
	layout.Initialize(layout_types, false);

	const auto &offsets = layout.GetOffsets();
	tuple_size = offsets[condition_types.size() + build_types.size()];
	pointer_offset = offsets.back();
	entry_size = layout.GetRowWidth();

	// compute the per-block capacity of this HT
	idx_t block_capacity = Storage::BLOCK_SIZE / entry_size;
	block_collection = make_unique<RowDataCollection>(buffer_manager, block_capacity, entry_size);
	string_heap = make_unique<RowDataCollection>(buffer_manager, (idx_t)Storage::BLOCK_SIZE, 1, true);
	swizzled_block_collection = block_collection->CloneEmpty();
	swizzled_string_heap = string_heap->CloneEmpty();
}

JoinHashTable::~JoinHashTable() {
}

void JoinHashTable::Merge(JoinHashTable &other) {
	block_collection->Merge(*other.block_collection);
	swizzled_block_collection->Merge(*other.swizzled_block_collection);
	if (!layout.AllConstant()) {
		string_heap->Merge(*other.string_heap);
		swizzled_string_heap->Merge(*other.swizzled_string_heap);
	}

	if (join_type == JoinType::MARK) {
		auto &info = correlated_mark_join_info;
		lock_guard<mutex> mj_lock(info.mj_lock);
		has_null = has_null || other.has_null;
		if (!info.correlated_types.empty()) {
			auto &other_info = other.correlated_mark_join_info;
			info.correlated_counts->Combine(*other_info.correlated_counts);
		}
	}

	lock_guard<mutex> lock(partitioned_data_lock);
	if (partition_block_collections.empty()) {
		D_ASSERT(partition_string_heaps.empty());
		// Move partitions to this HT
		for (idx_t p = 0; p < other.partition_block_collections.size(); p++) {
			partition_block_collections.push_back(std::move(other.partition_block_collections[p]));
			if (!layout.AllConstant()) {
				partition_string_heaps.push_back(std::move(other.partition_string_heaps[p]));
			}
		}
		return;
	}

	// Should have same number of partitions
	D_ASSERT(partition_block_collections.size() == other.partition_block_collections.size());
	D_ASSERT(partition_string_heaps.size() == other.partition_string_heaps.size());
	for (idx_t idx = 0; idx < other.partition_block_collections.size(); idx++) {
		partition_block_collections[idx]->Merge(*other.partition_block_collections[idx]);
		if (!layout.AllConstant()) {
			partition_string_heaps[idx]->Merge(*other.partition_string_heaps[idx]);
		}
	}
}

void JoinHashTable::ApplyBitmask(Vector &hashes, idx_t count) {
	if (hashes.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		D_ASSERT(!ConstantVector::IsNull(hashes));
		auto indices = ConstantVector::GetData<hash_t>(hashes);
		*indices = *indices & bitmask;
	} else {
		hashes.Flatten(count);
		auto indices = FlatVector::GetData<hash_t>(hashes);
		for (idx_t i = 0; i < count; i++) {
			indices[i] &= bitmask;
		}
	}
}

void JoinHashTable::ApplyBitmask(Vector &hashes, const SelectionVector &sel, idx_t count, Vector &pointers) {
	UnifiedVectorFormat hdata;
	hashes.ToUnifiedFormat(count, hdata);

	auto hash_data = (hash_t *)hdata.data;
	auto result_data = FlatVector::GetData<data_ptr_t *>(pointers);
	auto main_ht = (data_ptr_t *)hash_map.get();
	for (idx_t i = 0; i < count; i++) {
		auto rindex = sel.get_index(i);
		auto hindex = hdata.sel->get_index(rindex);
		auto hash = hash_data[hindex];
		result_data[rindex] = main_ht + (hash & bitmask);
	}
}

void JoinHashTable::Hash(DataChunk &keys, const SelectionVector &sel, idx_t count, Vector &hashes) {
	if (count == keys.size()) {
		// no null values are filtered: use regular hash functions
		VectorOperations::Hash(keys.data[0], hashes, keys.size());
		for (idx_t i = 1; i < equality_types.size(); i++) {
			VectorOperations::CombineHash(hashes, keys.data[i], keys.size());
		}
	} else {
		// null values were filtered: use selection vector
		VectorOperations::Hash(keys.data[0], hashes, sel, count);
		for (idx_t i = 1; i < equality_types.size(); i++) {
			VectorOperations::CombineHash(hashes, keys.data[i], sel, count);
		}
	}
}

static idx_t FilterNullValues(UnifiedVectorFormat &vdata, const SelectionVector &sel, idx_t count,
                              SelectionVector &result) {
	idx_t result_count = 0;
	for (idx_t i = 0; i < count; i++) {
		auto idx = sel.get_index(i);
		auto key_idx = vdata.sel->get_index(idx);
		if (vdata.validity.RowIsValid(key_idx)) {
			result.set_index(result_count++, idx);
		}
	}
	return result_count;
}

idx_t JoinHashTable::PrepareKeys(DataChunk &keys, unique_ptr<UnifiedVectorFormat[]> &key_data,
                                 const SelectionVector *&current_sel, SelectionVector &sel, bool build_side) {
	key_data = keys.ToUnifiedFormat();

	// figure out which keys are NULL, and create a selection vector out of them
	current_sel = FlatVector::IncrementalSelectionVector();
	idx_t added_count = keys.size();
	if (build_side && IsRightOuterJoin(join_type)) {
		// in case of a right or full outer join, we cannot remove NULL keys from the build side
		return added_count;
	}
	for (idx_t i = 0; i < keys.ColumnCount(); i++) {
		if (!null_values_are_equal[i]) {
			if (key_data[i].validity.AllValid()) {
				continue;
			}
			added_count = FilterNullValues(key_data[i], *current_sel, added_count, sel);
			// null values are NOT equal for this column, filter them out
			current_sel = &sel;
		}
	}
	return added_count;
}

void JoinHashTable::Build(DataChunk &keys, DataChunk &payload) {
	D_ASSERT(!finalized);
	D_ASSERT(keys.size() == payload.size());
	if (keys.size() == 0) {
		return;
	}
	// special case: correlated mark join
	if (join_type == JoinType::MARK && !correlated_mark_join_info.correlated_types.empty()) {
		auto &info = correlated_mark_join_info;
		lock_guard<mutex> mj_lock(info.mj_lock);
		// Correlated MARK join
		// for the correlated mark join we need to keep track of COUNT(*) and COUNT(COLUMN) for each of the correlated
		// columns push into the aggregate hash table
		D_ASSERT(info.correlated_counts);
		info.group_chunk.SetCardinality(keys);
		for (idx_t i = 0; i < info.correlated_types.size(); i++) {
			info.group_chunk.data[i].Reference(keys.data[i]);
		}
		if (info.correlated_payload.data.empty()) {
			vector<LogicalType> types;
			types.push_back(keys.data[info.correlated_types.size()].GetType());
			info.correlated_payload.InitializeEmpty(types);
		}
		info.correlated_payload.SetCardinality(keys);
		info.correlated_payload.data[0].Reference(keys.data[info.correlated_types.size()]);
		info.correlated_counts->AddChunk(info.group_chunk, info.correlated_payload, AggregateType::NON_DISTINCT);
	}

	// prepare the keys for processing
	unique_ptr<UnifiedVectorFormat[]> key_data;
	const SelectionVector *current_sel;
	SelectionVector sel(STANDARD_VECTOR_SIZE);
	idx_t added_count = PrepareKeys(keys, key_data, current_sel, sel, true);
	if (added_count < keys.size()) {
		has_null = true;
	}
	if (added_count == 0) {
		return;
	}

	// build out the buffer space
	Vector addresses(LogicalType::POINTER);
	auto key_locations = FlatVector::GetData<data_ptr_t>(addresses);
	auto handles = block_collection->Build(added_count, key_locations, nullptr, current_sel);

	// hash the keys and obtain an entry in the list
	// note that we only hash the keys used in the equality comparison
	Vector hash_values(LogicalType::HASH);
	Hash(keys, *current_sel, added_count, hash_values);

	// build a chunk so we can handle nested types that need more than Orrification
	DataChunk source_chunk;
	source_chunk.InitializeEmpty(layout.GetTypes());

	vector<UnifiedVectorFormat> source_data;
	source_data.reserve(layout.ColumnCount());

	// serialize the keys to the key locations
	for (idx_t i = 0; i < keys.ColumnCount(); i++) {
		source_chunk.data[i].Reference(keys.data[i]);
		source_data.emplace_back(std::move(key_data[i]));
	}
	// now serialize the payload
	D_ASSERT(build_types.size() == payload.ColumnCount());
	for (idx_t i = 0; i < payload.ColumnCount(); i++) {
		source_chunk.data[source_data.size()].Reference(payload.data[i]);
		UnifiedVectorFormat pdata;
		payload.data[i].ToUnifiedFormat(payload.size(), pdata);
		source_data.emplace_back(std::move(pdata));
	}
	if (IsRightOuterJoin(join_type)) {
		// for FULL/RIGHT OUTER joins initialize the "found" boolean to false
		source_chunk.data[source_data.size()].Reference(vfound);
		UnifiedVectorFormat fdata;
		vfound.ToUnifiedFormat(keys.size(), fdata);
		source_data.emplace_back(std::move(fdata));
	}

	// serialise the hashes at the end
	source_chunk.data[source_data.size()].Reference(hash_values);
	UnifiedVectorFormat hdata;
	hash_values.ToUnifiedFormat(keys.size(), hdata);
	source_data.emplace_back(std::move(hdata));

	source_chunk.SetCardinality(keys);

	RowOperations::Scatter(source_chunk, source_data.data(), layout, addresses, *string_heap, *current_sel,
	                       added_count);
}

template <bool PARALLEL>
static inline void InsertHashesLoop(atomic<data_ptr_t> pointers[], const hash_t indices[], const idx_t count,
                                    const data_ptr_t key_locations[], const idx_t pointer_offset) {
	for (idx_t i = 0; i < count; i++) {
		auto index = indices[i];
		if (PARALLEL) {
			data_ptr_t head;
			do {
				head = pointers[index];
				Store<data_ptr_t>(head, key_locations[i] + pointer_offset);
			} while (!std::atomic_compare_exchange_weak(&pointers[index], &head, key_locations[i]));
		} else {
			// set prev in current key to the value (NOTE: this will be nullptr if there is none)
			Store<data_ptr_t>(pointers[index], key_locations[i] + pointer_offset);

			// set pointer to current tuple
			pointers[index] = key_locations[i];
		}
	}
}

void JoinHashTable::InsertHashes(Vector &hashes, idx_t count, data_ptr_t key_locations[], bool parallel) {
	D_ASSERT(hashes.GetType().id() == LogicalType::HASH);

	// use bitmask to get position in array
	ApplyBitmask(hashes, count);

	hashes.Flatten(count);
	D_ASSERT(hashes.GetVectorType() == VectorType::FLAT_VECTOR);

	auto pointers = (atomic<data_ptr_t> *)hash_map.get();
	auto indices = FlatVector::GetData<hash_t>(hashes);

	if (parallel) {
		InsertHashesLoop<true>(pointers, indices, count, key_locations, pointer_offset);
	} else {
		InsertHashesLoop<false>(pointers, indices, count, key_locations, pointer_offset);
	}
}

void JoinHashTable::InitializePointerTable() {
	idx_t count = external ? MaxValue<idx_t>(tuples_per_round, Count()) : Count();
	idx_t capacity = PointerTableCapacity(count);
	// size needs to be a power of 2
	D_ASSERT((capacity & (capacity - 1)) == 0);
	bitmask = capacity - 1;

	if (!hash_map.get()) {
		// allocate the HT if not yet done
		hash_map = buffer_manager.GetBufferAllocator().Allocate(capacity * sizeof(data_ptr_t));
	}
	D_ASSERT(hash_map.GetSize() == capacity * sizeof(data_ptr_t));

	// initialize HT with all-zero entries
	memset(hash_map.get(), 0, capacity * sizeof(data_ptr_t));
}

void JoinHashTable::Finalize(idx_t block_idx_start, idx_t block_idx_end, bool parallel) {
	// Pointer table should be allocated
	D_ASSERT(hash_map.get());

	const auto unswizzle = external && !layout.AllConstant();
	vector<BufferHandle> local_pinned_handles;

	Vector hashes(LogicalType::HASH);
	auto hash_data = FlatVector::GetData<hash_t>(hashes);
	data_ptr_t key_locations[STANDARD_VECTOR_SIZE];
	// now construct the actual hash table; scan the nodes
	// as we scan the nodes we pin all the blocks of the HT and keep them pinned until the HT is destroyed
	// this is so that we can keep pointers around to the blocks
	for (idx_t block_idx = block_idx_start; block_idx < block_idx_end; block_idx++) {
		auto &block = block_collection->blocks[block_idx];
		auto handle = buffer_manager.Pin(block->block);
		data_ptr_t dataptr = handle.Ptr();

		data_ptr_t heap_ptr = nullptr;
		if (unswizzle) {
			auto &heap_block = string_heap->blocks[block_idx];
			auto heap_handle = buffer_manager.Pin(heap_block->block);
			heap_ptr = heap_handle.Ptr();
			local_pinned_handles.push_back(std::move(heap_handle));
		}

		idx_t entry = 0;
		while (entry < block->count) {
			idx_t next = MinValue<idx_t>(STANDARD_VECTOR_SIZE, block->count - entry);

			if (unswizzle) {
				RowOperations::UnswizzlePointers(layout, dataptr, heap_ptr, next);
			}

			// fetch the next vector of entries from the blocks
			for (idx_t i = 0; i < next; i++) {
				hash_data[i] = Load<hash_t>((data_ptr_t)(dataptr + pointer_offset));
				key_locations[i] = dataptr;
				dataptr += entry_size;
			}
			// now insert into the hash table
			InsertHashes(hashes, next, key_locations, parallel);

			entry += next;
		}
		local_pinned_handles.push_back(std::move(handle));
	}

	lock_guard<mutex> lock(pinned_handles_lock);
	for (auto &local_pinned_handle : local_pinned_handles) {
		pinned_handles.push_back(std::move(local_pinned_handle));
	}
}

unique_ptr<ScanStructure> JoinHashTable::InitializeScanStructure(DataChunk &keys, const SelectionVector *&current_sel) {
	D_ASSERT(Count() > 0); // should be handled before
	D_ASSERT(finalized);

	// set up the scan structure
	auto ss = make_unique<ScanStructure>(*this);

	if (join_type != JoinType::INNER) {
		ss->found_match = unique_ptr<bool[]>(new bool[STANDARD_VECTOR_SIZE]);
		memset(ss->found_match.get(), 0, sizeof(bool) * STANDARD_VECTOR_SIZE);
	}

	// first prepare the keys for probing
	ss->count = PrepareKeys(keys, ss->key_data, current_sel, ss->sel_vector, false);
	return ss;
}

unique_ptr<ScanStructure> JoinHashTable::Probe(DataChunk &keys, Vector *precomputed_hashes) {
	const SelectionVector *current_sel;
	auto ss = InitializeScanStructure(keys, current_sel);
	if (ss->count == 0) {
		return ss;
	}

	if (precomputed_hashes) {
		ApplyBitmask(*precomputed_hashes, *current_sel, ss->count, ss->pointers);
	} else {
		// hash all the keys
		Vector hashes(LogicalType::HASH);
		Hash(keys, *current_sel, ss->count, hashes);

		// now initialize the pointers of the scan structure based on the hashes
		ApplyBitmask(hashes, *current_sel, ss->count, ss->pointers);
	}

	// create the selection vector linking to only non-empty entries
	ss->InitializeSelectionVector(current_sel);

	return ss;
}

ScanStructure::ScanStructure(JoinHashTable &ht)
    : pointers(LogicalType::POINTER), sel_vector(STANDARD_VECTOR_SIZE), ht(ht), finished(false) {
}

void ScanStructure::Next(DataChunk &keys, DataChunk &left, DataChunk &result) {
	if (finished) {
		return;
	}
	switch (ht.join_type) {
	case JoinType::INNER:
	case JoinType::RIGHT:
		NextInnerJoin(keys, left, result);
		break;
	case JoinType::SEMI:
		NextSemiJoin(keys, left, result);
		break;
	case JoinType::MARK:
		NextMarkJoin(keys, left, result);
		break;
	case JoinType::ANTI:
		NextAntiJoin(keys, left, result);
		break;
	case JoinType::OUTER:
	case JoinType::LEFT:
		NextLeftJoin(keys, left, result);
		break;
	case JoinType::SINGLE:
		NextSingleJoin(keys, left, result);
		break;
	default:
		throw InternalException("Unhandled join type in JoinHashTable");
	}
}

idx_t ScanStructure::ResolvePredicates(DataChunk &keys, SelectionVector &match_sel, SelectionVector *no_match_sel) {
	// Start with the scan selection
	for (idx_t i = 0; i < this->count; ++i) {
		match_sel.set_index(i, this->sel_vector.get_index(i));
	}
	idx_t no_match_count = 0;

	return RowOperations::Match(keys, key_data.get(), ht.layout, pointers, ht.predicates, match_sel, this->count,
	                            no_match_sel, no_match_count);
}

idx_t ScanStructure::ScanInnerJoin(DataChunk &keys, SelectionVector &result_vector) {
	while (true) {
		// resolve the predicates for this set of keys
		idx_t result_count = ResolvePredicates(keys, result_vector, nullptr);

		// after doing all the comparisons set the found_match vector
		if (found_match) {
			for (idx_t i = 0; i < result_count; i++) {
				auto idx = result_vector.get_index(i);
				found_match[idx] = true;
			}
		}
		if (result_count > 0) {
			return result_count;
		}
		// no matches found: check the next set of pointers
		AdvancePointers();
		if (this->count == 0) {
			return 0;
		}
	}
}

void ScanStructure::AdvancePointers(const SelectionVector &sel, idx_t sel_count) {
	// now for all the pointers, we move on to the next set of pointers
	idx_t new_count = 0;
	auto ptrs = FlatVector::GetData<data_ptr_t>(this->pointers);
	for (idx_t i = 0; i < sel_count; i++) {
		auto idx = sel.get_index(i);
		ptrs[idx] = Load<data_ptr_t>(ptrs[idx] + ht.pointer_offset);
		if (ptrs[idx]) {
			this->sel_vector.set_index(new_count++, idx);
		}
	}
	this->count = new_count;
}

void ScanStructure::InitializeSelectionVector(const SelectionVector *&current_sel) {
	idx_t non_empty_count = 0;
	auto ptrs = FlatVector::GetData<data_ptr_t>(pointers);
	auto cnt = count;
	for (idx_t i = 0; i < cnt; i++) {
		const auto idx = current_sel->get_index(i);
		ptrs[idx] = Load<data_ptr_t>(ptrs[idx]);
		if (ptrs[idx]) {
			sel_vector.set_index(non_empty_count++, idx);
		}
	}
	count = non_empty_count;
}

void ScanStructure::AdvancePointers() {
	AdvancePointers(this->sel_vector, this->count);
}

void ScanStructure::GatherResult(Vector &result, const SelectionVector &result_vector,
                                 const SelectionVector &sel_vector, const idx_t count, const idx_t col_no) {
	RowOperations::Gather(pointers, sel_vector, result, result_vector, count, ht.layout, col_no);
}

void ScanStructure::GatherResult(Vector &result, const SelectionVector &sel_vector, const idx_t count,
                                 const idx_t col_idx) {
	GatherResult(result, *FlatVector::IncrementalSelectionVector(), sel_vector, count, col_idx);
}

void ScanStructure::NextInnerJoin(DataChunk &keys, DataChunk &left, DataChunk &result) {
	D_ASSERT(result.ColumnCount() == left.ColumnCount() + ht.build_types.size());
	if (this->count == 0) {
		// no pointers left to chase
		return;
	}

	SelectionVector result_vector(STANDARD_VECTOR_SIZE);

	idx_t result_count = ScanInnerJoin(keys, result_vector);
	if (result_count > 0) {
		if (IsRightOuterJoin(ht.join_type)) {
			// full/right outer join: mark join matches as FOUND in the HT
			auto ptrs = FlatVector::GetData<data_ptr_t>(pointers);
			for (idx_t i = 0; i < result_count; i++) {
				auto idx = result_vector.get_index(i);
				// NOTE: threadsan reports this as a data race because this can be set concurrently by separate threads
				// Technically it is, but it does not matter, since the only value that can be written is "true"
				Store<bool>(true, ptrs[idx] + ht.tuple_size);
			}
		}
		// matches were found
		// construct the result
		// on the LHS, we create a slice using the result vector
		result.Slice(left, result_vector, result_count);

		// on the RHS, we need to fetch the data from the hash table
		for (idx_t i = 0; i < ht.build_types.size(); i++) {
			auto &vector = result.data[left.ColumnCount() + i];
			D_ASSERT(vector.GetType() == ht.build_types[i]);
			GatherResult(vector, result_vector, result_count, i + ht.condition_types.size());
		}
		AdvancePointers();
	}
}

void ScanStructure::ScanKeyMatches(DataChunk &keys) {
	// the semi-join, anti-join and mark-join we handle a differently from the inner join
	// since there can be at most STANDARD_VECTOR_SIZE results
	// we handle the entire chunk in one call to Next().
	// for every pointer, we keep chasing pointers and doing comparisons.
	// this results in a boolean array indicating whether or not the tuple has a match
	SelectionVector match_sel(STANDARD_VECTOR_SIZE), no_match_sel(STANDARD_VECTOR_SIZE);
	while (this->count > 0) {
		// resolve the predicates for the current set of pointers
		idx_t match_count = ResolvePredicates(keys, match_sel, &no_match_sel);
		idx_t no_match_count = this->count - match_count;

		// mark each of the matches as found
		for (idx_t i = 0; i < match_count; i++) {
			found_match[match_sel.get_index(i)] = true;
		}
		// continue searching for the ones where we did not find a match yet
		AdvancePointers(no_match_sel, no_match_count);
	}
}

template <bool MATCH>
void ScanStructure::NextSemiOrAntiJoin(DataChunk &keys, DataChunk &left, DataChunk &result) {
	D_ASSERT(left.ColumnCount() == result.ColumnCount());
	D_ASSERT(keys.size() == left.size());
	// create the selection vector from the matches that were found
	SelectionVector sel(STANDARD_VECTOR_SIZE);
	idx_t result_count = 0;
	for (idx_t i = 0; i < keys.size(); i++) {
		if (found_match[i] == MATCH) {
			// part of the result
			sel.set_index(result_count++, i);
		}
	}
	// construct the final result
	if (result_count > 0) {
		// we only return the columns on the left side
		// reference the columns of the left side from the result
		result.Slice(left, sel, result_count);
	} else {
		D_ASSERT(result.size() == 0);
	}
}

void ScanStructure::NextSemiJoin(DataChunk &keys, DataChunk &left, DataChunk &result) {
	// first scan for key matches
	ScanKeyMatches(keys);
	// then construct the result from all tuples with a match
	NextSemiOrAntiJoin<true>(keys, left, result);

	finished = true;
}

void ScanStructure::NextAntiJoin(DataChunk &keys, DataChunk &left, DataChunk &result) {
	// first scan for key matches
	ScanKeyMatches(keys);
	// then construct the result from all tuples that did not find a match
	NextSemiOrAntiJoin<false>(keys, left, result);

	finished = true;
}

void ScanStructure::ConstructMarkJoinResult(DataChunk &join_keys, DataChunk &child, DataChunk &result) {
	// for the initial set of columns we just reference the left side
	result.SetCardinality(child);
	for (idx_t i = 0; i < child.ColumnCount(); i++) {
		result.data[i].Reference(child.data[i]);
	}
	auto &mark_vector = result.data.back();
	mark_vector.SetVectorType(VectorType::FLAT_VECTOR);
	// first we set the NULL values from the join keys
	// if there is any NULL in the keys, the result is NULL
	auto bool_result = FlatVector::GetData<bool>(mark_vector);
	auto &mask = FlatVector::Validity(mark_vector);
	for (idx_t col_idx = 0; col_idx < join_keys.ColumnCount(); col_idx++) {
		if (ht.null_values_are_equal[col_idx]) {
			continue;
		}
		UnifiedVectorFormat jdata;
		join_keys.data[col_idx].ToUnifiedFormat(join_keys.size(), jdata);
		if (!jdata.validity.AllValid()) {
			for (idx_t i = 0; i < join_keys.size(); i++) {
				auto jidx = jdata.sel->get_index(i);
				mask.Set(i, jdata.validity.RowIsValidUnsafe(jidx));
			}
		}
	}
	// now set the remaining entries to either true or false based on whether a match was found
	if (found_match) {
		for (idx_t i = 0; i < child.size(); i++) {
			bool_result[i] = found_match[i];
		}
	} else {
		memset(bool_result, 0, sizeof(bool) * child.size());
	}
	// if the right side contains NULL values, the result of any FALSE becomes NULL
	if (ht.has_null) {
		for (idx_t i = 0; i < child.size(); i++) {
			if (!bool_result[i]) {
				mask.SetInvalid(i);
			}
		}
	}
}

void ScanStructure::NextMarkJoin(DataChunk &keys, DataChunk &input, DataChunk &result) {
	D_ASSERT(result.ColumnCount() == input.ColumnCount() + 1);
	D_ASSERT(result.data.back().GetType() == LogicalType::BOOLEAN);
	// this method should only be called for a non-empty HT
	D_ASSERT(ht.Count() > 0);

	ScanKeyMatches(keys);
	if (ht.correlated_mark_join_info.correlated_types.empty()) {
		ConstructMarkJoinResult(keys, input, result);
	} else {
		auto &info = ht.correlated_mark_join_info;
		// there are correlated columns
		// first we fetch the counts from the aggregate hashtable corresponding to these entries
		D_ASSERT(keys.ColumnCount() == info.group_chunk.ColumnCount() + 1);
		info.group_chunk.SetCardinality(keys);
		for (idx_t i = 0; i < info.group_chunk.ColumnCount(); i++) {
			info.group_chunk.data[i].Reference(keys.data[i]);
		}
		info.correlated_counts->FetchAggregates(info.group_chunk, info.result_chunk);

		// for the initial set of columns we just reference the left side
		result.SetCardinality(input);
		for (idx_t i = 0; i < input.ColumnCount(); i++) {
			result.data[i].Reference(input.data[i]);
		}
		// create the result matching vector
		auto &last_key = keys.data.back();
		auto &result_vector = result.data.back();
		// first set the nullmask based on whether or not there were NULL values in the join key
		result_vector.SetVectorType(VectorType::FLAT_VECTOR);
		auto bool_result = FlatVector::GetData<bool>(result_vector);
		auto &mask = FlatVector::Validity(result_vector);
		switch (last_key.GetVectorType()) {
		case VectorType::CONSTANT_VECTOR:
			if (ConstantVector::IsNull(last_key)) {
				mask.SetAllInvalid(input.size());
			}
			break;
		case VectorType::FLAT_VECTOR:
			mask.Copy(FlatVector::Validity(last_key), input.size());
			break;
		default: {
			UnifiedVectorFormat kdata;
			last_key.ToUnifiedFormat(keys.size(), kdata);
			for (idx_t i = 0; i < input.size(); i++) {
				auto kidx = kdata.sel->get_index(i);
				mask.Set(i, kdata.validity.RowIsValid(kidx));
			}
			break;
		}
		}

		auto count_star = FlatVector::GetData<int64_t>(info.result_chunk.data[0]);
		auto count = FlatVector::GetData<int64_t>(info.result_chunk.data[1]);
		// set the entries to either true or false based on whether a match was found
		for (idx_t i = 0; i < input.size(); i++) {
			D_ASSERT(count_star[i] >= count[i]);
			bool_result[i] = found_match ? found_match[i] : false;
			if (!bool_result[i] && count_star[i] > count[i]) {
				// RHS has NULL value and result is false: set to null
				mask.SetInvalid(i);
			}
			if (count_star[i] == 0) {
				// count == 0, set nullmask to false (we know the result is false now)
				mask.SetValid(i);
			}
		}
	}
	finished = true;
}

void ScanStructure::NextLeftJoin(DataChunk &keys, DataChunk &left, DataChunk &result) {
	// a LEFT OUTER JOIN is identical to an INNER JOIN except all tuples that do
	// not have a match must return at least one tuple (with the right side set
	// to NULL in every column)
	NextInnerJoin(keys, left, result);
	if (result.size() == 0) {
		// no entries left from the normal join
		// fill in the result of the remaining left tuples
		// together with NULL values on the right-hand side
		idx_t remaining_count = 0;
		SelectionVector sel(STANDARD_VECTOR_SIZE);
		for (idx_t i = 0; i < left.size(); i++) {
			if (!found_match[i]) {
				sel.set_index(remaining_count++, i);
			}
		}
		if (remaining_count > 0) {
			// have remaining tuples
			// slice the left side with tuples that did not find a match
			result.Slice(left, sel, remaining_count);

			// now set the right side to NULL
			for (idx_t i = left.ColumnCount(); i < result.ColumnCount(); i++) {
				Vector &vec = result.data[i];
				vec.SetVectorType(VectorType::CONSTANT_VECTOR);
				ConstantVector::SetNull(vec, true);
			}
		}
		finished = true;
	}
}

void ScanStructure::NextSingleJoin(DataChunk &keys, DataChunk &input, DataChunk &result) {
	// single join
	// this join is similar to the semi join except that
	// (1) we actually return data from the RHS and
	// (2) we return NULL for that data if there is no match
	idx_t result_count = 0;
	SelectionVector result_sel(STANDARD_VECTOR_SIZE);
	SelectionVector match_sel(STANDARD_VECTOR_SIZE), no_match_sel(STANDARD_VECTOR_SIZE);
	while (this->count > 0) {
		// resolve the predicates for the current set of pointers
		idx_t match_count = ResolvePredicates(keys, match_sel, &no_match_sel);
		idx_t no_match_count = this->count - match_count;

		// mark each of the matches as found
		for (idx_t i = 0; i < match_count; i++) {
			// found a match for this index
			auto index = match_sel.get_index(i);
			found_match[index] = true;
			result_sel.set_index(result_count++, index);
		}
		// continue searching for the ones where we did not find a match yet
		AdvancePointers(no_match_sel, no_match_count);
	}
	// reference the columns of the left side from the result
	D_ASSERT(input.ColumnCount() > 0);
	for (idx_t i = 0; i < input.ColumnCount(); i++) {
		result.data[i].Reference(input.data[i]);
	}
	// now fetch the data from the RHS
	for (idx_t i = 0; i < ht.build_types.size(); i++) {
		auto &vector = result.data[input.ColumnCount() + i];
		// set NULL entries for every entry that was not found
		for (idx_t j = 0; j < input.size(); j++) {
			if (!found_match[j]) {
				FlatVector::SetNull(vector, j, true);
			}
		}
		// for the remaining values we fetch the values
		GatherResult(vector, result_sel, result_sel, result_count, i + ht.condition_types.size());
	}
	result.SetCardinality(input.size());

	// like the SEMI, ANTI and MARK join types, the SINGLE join only ever does one pass over the HT per input chunk
	finished = true;
}

idx_t JoinHashTable::ScanFullOuter(JoinHTScanState &state, Vector &addresses) {
	// scan the HT starting from the current position and check which rows from the build side did not find a match
	auto key_locations = FlatVector::GetData<data_ptr_t>(addresses);
	idx_t found_entries = 0;
	for (; state.block_position < block_collection->blocks.size(); state.block_position++, state.position = 0) {
		auto &block = block_collection->blocks[state.block_position];
		auto handle = buffer_manager.Pin(block->block);
		auto baseptr = handle.Ptr();
		for (; state.position < block->count; state.position++, state.scan_index++) {
			auto tuple_base = baseptr + state.position * entry_size;
			auto found_match = Load<bool>(tuple_base + tuple_size);
			if (!found_match) {
				key_locations[found_entries++] = tuple_base;
				if (found_entries == STANDARD_VECTOR_SIZE) {
					state.position++;
					state.scan_index++;
					break;
				}
			}
		}
		if (found_entries == STANDARD_VECTOR_SIZE) {
			break;
		}
	}
	return found_entries;
}

void JoinHashTable::GatherFullOuter(DataChunk &result, Vector &addresses, idx_t found_entries) {
	if (found_entries == 0) {
		return;
	}
	result.SetCardinality(found_entries);
	idx_t left_column_count = result.ColumnCount() - build_types.size();
	const auto &sel_vector = *FlatVector::IncrementalSelectionVector();
	// set the left side as a constant NULL
	for (idx_t i = 0; i < left_column_count; i++) {
		Vector &vec = result.data[i];
		vec.SetVectorType(VectorType::CONSTANT_VECTOR);
		ConstantVector::SetNull(vec, true);
	}
	// gather the values from the RHS
	for (idx_t i = 0; i < build_types.size(); i++) {
		auto &vector = result.data[left_column_count + i];
		D_ASSERT(vector.GetType() == build_types[i]);
		const auto col_no = condition_types.size() + i;
		RowOperations::Gather(addresses, sel_vector, vector, sel_vector, found_entries, layout, col_no);
	}
}

idx_t JoinHashTable::FillWithHTOffsets(data_ptr_t *key_locations, JoinHTScanState &state) {
	// iterate over blocks
	idx_t key_count = 0;
	while (state.block_position < block_collection->blocks.size()) {
		auto &block = block_collection->blocks[state.block_position];
		auto handle = buffer_manager.Pin(block->block);
		auto base_ptr = handle.Ptr();
		// go through all the tuples within this block
		while (state.position < block->count) {
			auto tuple_base = base_ptr + state.position * entry_size;
			// store its locations
			key_locations[key_count++] = tuple_base;
			state.position++;
		}
		state.block_position++;
		state.position = 0;
	}
	return key_count;
}

void JoinHashTable::PinAllBlocks() {
	for (auto &block : block_collection->blocks) {
		pinned_handles.push_back(buffer_manager.Pin(block->block));
	}
}

void JoinHashTable::SwizzleBlocks() {
	if (block_collection->count == 0) {
		return;
	}

	if (layout.AllConstant()) {
		// No heap blocks! Just merge fixed-size data
		swizzled_block_collection->Merge(*block_collection);
		return;
	}

	// We create one heap block per data block and swizzle the pointers
	auto &heap_blocks = string_heap->blocks;
	idx_t heap_block_idx = 0;
	idx_t heap_block_remaining = heap_blocks[heap_block_idx]->count;
	for (auto &data_block : block_collection->blocks) {
		if (heap_block_remaining == 0) {
			heap_block_remaining = heap_blocks[++heap_block_idx]->count;
		}

		// Pin the data block and swizzle the pointers within the rows
		auto data_handle = buffer_manager.Pin(data_block->block);
		auto data_ptr = data_handle.Ptr();
		RowOperations::SwizzleColumns(layout, data_ptr, data_block->count);

		// We want to copy as little of the heap data as possible, check how the data and heap blocks line up
		if (heap_block_remaining >= data_block->count) {
			// Easy: current heap block contains all strings for this data block, just copy (reference) the block
			swizzled_string_heap->blocks.emplace_back(heap_blocks[heap_block_idx]->Copy());
			swizzled_string_heap->blocks.back()->count = data_block->count;

			// Swizzle the heap pointer
			auto heap_handle = buffer_manager.Pin(swizzled_string_heap->blocks.back()->block);
			auto heap_ptr = Load<data_ptr_t>(data_ptr + layout.GetHeapOffset());
			auto heap_offset = heap_ptr - heap_handle.Ptr();
			RowOperations::SwizzleHeapPointer(layout, data_ptr, heap_ptr, data_block->count, heap_offset);

			// Update counter
			heap_block_remaining -= data_block->count;
		} else {
			// Strings for this data block are spread over the current heap block and the next (and possibly more)
			idx_t data_block_remaining = data_block->count;
			vector<std::pair<data_ptr_t, idx_t>> ptrs_and_sizes;
			idx_t total_size = 0;
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
			swizzled_string_heap->blocks.emplace_back(
			    make_unique<RowDataBlock>(buffer_manager, MaxValue<idx_t>(total_size, (idx_t)Storage::BLOCK_SIZE), 1));
			auto new_heap_handle = buffer_manager.Pin(swizzled_string_heap->blocks.back()->block);
			auto new_heap_ptr = new_heap_handle.Ptr();
			for (auto &ptr_and_size : ptrs_and_sizes) {
				memcpy(new_heap_ptr, ptr_and_size.first, ptr_and_size.second);
				new_heap_ptr += ptr_and_size.second;
			}
		}
	}

	// We're done with variable-sized data, now just merge the fixed-size data
	swizzled_block_collection->Merge(*block_collection);
	D_ASSERT(swizzled_block_collection->blocks.size() == swizzled_string_heap->blocks.size());

	// Update counts and cleanup
	swizzled_string_heap->count = string_heap->count;
	string_heap->Clear();
}

void JoinHashTable::ComputePartitionSizes(ClientConfig &config, vector<unique_ptr<JoinHashTable>> &local_hts,
                                          idx_t max_ht_size) {
	external = true;

	// First set the number of tuples in the HT per partitioned round
	total_count = 0;
	idx_t total_size = 0;
	for (auto &ht : local_hts) {
		// TODO: SizeInBytes / SwizzledSize overestimates size by a lot because we make extra references of heap blocks
		//  Need to compute this more accurately
		total_count += ht->Count() + ht->SwizzledCount();
		total_size += ht->SizeInBytes() + ht->SwizzledSize();
	}

	if (total_count == 0) {
		return;
	}

	total_size += PointerTableCapacity(total_count) * sizeof(data_ptr_t);
	double avg_tuple_size = double(total_size) / double(total_count);
	tuples_per_round = double(max_ht_size) / avg_tuple_size;

	if (config.force_external) {
		// For force_external we do at least three rounds to test all code paths
		tuples_per_round = MinValue<idx_t>((total_count + 2) / 3, tuples_per_round);
	}

	// Set the number of radix bits (minimum 4, maximum 8)
	for (; radix_bits < 8; radix_bits++) {
		auto num_partitions = RadixPartitioning::NumberOfPartitions(radix_bits);
		auto avg_partition_size = total_size / num_partitions;

		// We aim for at least 8 partitions per probe round (tweaked experimentally)
		if (avg_partition_size * 8 < max_ht_size) {
			break;
		}
	}
}

void JoinHashTable::Partition(JoinHashTable &global_ht) {
#ifdef DEBUG
	D_ASSERT(layout.ColumnCount() == global_ht.layout.ColumnCount());
	for (idx_t col_idx = 0; col_idx < layout.ColumnCount(); col_idx++) {
		D_ASSERT(layout.GetTypes()[col_idx] == global_ht.layout.GetTypes()[col_idx]);
	}
#endif

	// Swizzle and Partition
	SwizzleBlocks();
	RadixPartitioning::PartitionRowData(global_ht.buffer_manager, global_ht.layout, global_ht.pointer_offset,
	                                    *swizzled_block_collection, *swizzled_string_heap, partition_block_collections,
	                                    partition_string_heaps, global_ht.radix_bits);

	// Add to global HT
	global_ht.Merge(*this);
}

void JoinHashTable::Reset() {
	pinned_handles.clear();
	block_collection->Clear();
	string_heap->Clear();
	finalized = false;
}

bool JoinHashTable::PrepareExternalFinalize() {
	idx_t num_partitions = RadixPartitioning::NumberOfPartitions(radix_bits);
	if (partition_block_collections.empty() || partition_end == num_partitions) {
		return false;
	}

	if (finalized) {
		Reset();
	}

	// Determine how many partitions we can do next (at least one)
	idx_t next = 0;
	idx_t count = 0;
	partition_start = partition_end;
	for (idx_t p = partition_start; p < num_partitions; p++) {
		auto partition_count = partition_block_collections[p]->count;
		if (partition_count != 0 && count != 0 && count + partition_count > tuples_per_round) {
			// We skip over empty partitions (partition_count != 0),
			// and need to have at least one partition (count != 0)
			break;
		}
		next++;
		count += partition_count;
	}
	partition_end += next;

	// Move specific partitions to the swizzled_... collections so they can be unswizzled
	D_ASSERT(SwizzledCount() == 0);
	for (idx_t p = partition_start; p < partition_end; p++) {
		auto &p_block_collection = *partition_block_collections[p];
		if (!layout.AllConstant()) {
			auto &p_string_heap = *partition_string_heaps[p];
			D_ASSERT(p_block_collection.count == p_string_heap.count);
			swizzled_string_heap->Merge(p_string_heap);
			// Remove after merging
			partition_string_heaps[p] = nullptr;
		}
		swizzled_block_collection->Merge(p_block_collection);
		// Remove after merging
		partition_block_collections[p] = nullptr;
	}
	D_ASSERT(count == SwizzledCount());

	// Unswizzle them
	D_ASSERT(Count() == 0);
	// Move swizzled data to regular data (will be unswizzled in 'Finalize()')
	block_collection->Merge(*swizzled_block_collection);
	string_heap->Merge(*swizzled_string_heap);
	D_ASSERT(count == Count());

	return true;
}

static void CreateSpillChunk(DataChunk &spill_chunk, DataChunk &keys, DataChunk &payload, Vector &hashes) {
	spill_chunk.Reset();
	idx_t spill_col_idx = 0;
	for (idx_t col_idx = 0; col_idx < keys.ColumnCount(); col_idx++) {
		spill_chunk.data[col_idx].Reference(keys.data[col_idx]);
	}
	spill_col_idx += keys.ColumnCount();
	for (idx_t col_idx = 0; col_idx < payload.data.size(); col_idx++) {
		spill_chunk.data[spill_col_idx + col_idx].Reference(payload.data[col_idx]);
	}
	spill_col_idx += payload.ColumnCount();
	spill_chunk.data[spill_col_idx].Reference(hashes);
}

unique_ptr<ScanStructure> JoinHashTable::ProbeAndSpill(DataChunk &keys, DataChunk &payload, ProbeSpill &probe_spill,
                                                       ProbeSpillLocalAppendState &spill_state,
                                                       DataChunk &spill_chunk) {
	// hash all the keys
	Vector hashes(LogicalType::HASH);
	Hash(keys, *FlatVector::IncrementalSelectionVector(), keys.size(), hashes);

	// find out which keys we can match with the current pinned partitions
	SelectionVector true_sel;
	SelectionVector false_sel;
	true_sel.Initialize();
	false_sel.Initialize();
	auto true_count = RadixPartitioning::Select(hashes, FlatVector::IncrementalSelectionVector(), keys.size(),
	                                            radix_bits, partition_end, &true_sel, &false_sel);
	auto false_count = keys.size() - true_count;

	CreateSpillChunk(spill_chunk, keys, payload, hashes);

	// can't probe these values right now, append to spill
	spill_chunk.Slice(false_sel, false_count);
	spill_chunk.Verify();
	probe_spill.Append(spill_chunk, spill_state);

	// slice the stuff we CAN probe right now
	hashes.Slice(true_sel, true_count);
	keys.Slice(true_sel, true_count);
	payload.Slice(true_sel, true_count);

	const SelectionVector *current_sel;
	auto ss = InitializeScanStructure(keys, current_sel);
	if (ss->count == 0) {
		return ss;
	}

	// now initialize the pointers of the scan structure based on the hashes
	ApplyBitmask(hashes, *current_sel, ss->count, ss->pointers);

	// create the selection vector linking to only non-empty entries
	ss->InitializeSelectionVector(current_sel);

	return ss;
}

ProbeSpill::ProbeSpill(JoinHashTable &ht, ClientContext &context, const vector<LogicalType> &probe_types)
    : ht(ht), context(context), probe_types(probe_types) {
	if (ht.total_count - ht.Count() <= ht.tuples_per_round) {
		// No need to partition as we will only have one more probe round
		partitioned = false;
	} else {
		// More than one probe round to go, so we need to partition
		partitioned = true;
		global_partitions =
		    make_unique<RadixPartitionedColumnData>(context, probe_types, ht.radix_bits, probe_types.size() - 1);
	}
	column_ids.reserve(probe_types.size());
	for (column_t column_id = 0; column_id < probe_types.size(); column_id++) {
		column_ids.emplace_back(column_id);
	}
}

ProbeSpillLocalState ProbeSpill::RegisterThread() {
	ProbeSpillLocalAppendState result;
	lock_guard<mutex> guard(lock);
	if (partitioned) {
		local_partitions.emplace_back(global_partitions->CreateShared());
		local_partition_append_states.emplace_back(make_unique<PartitionedColumnDataAppendState>());
		local_partitions.back()->InitializeAppendState(*local_partition_append_states.back());

		result.local_partition = local_partitions.back().get();
		result.local_partition_append_state = local_partition_append_states.back().get();
	} else {
		local_spill_collections.emplace_back(
		    make_unique<ColumnDataCollection>(BufferManager::GetBufferManager(context), probe_types));
		local_spill_append_states.emplace_back(make_unique<ColumnDataAppendState>());
		local_spill_collections.back()->InitializeAppend(*local_spill_append_states.back());

		result.local_spill_collection = local_spill_collections.back().get();
		result.local_spill_append_state = local_spill_append_states.back().get();
	}
	return result;
}

void ProbeSpill::Append(DataChunk &chunk, ProbeSpillLocalAppendState &local_state) {
	if (partitioned) {
		local_state.local_partition->Append(*local_state.local_partition_append_state, chunk);
	} else {
		local_state.local_spill_collection->Append(*local_state.local_spill_append_state, chunk);
	}
}

void ProbeSpill::Finalize() {
	if (partitioned) {
		D_ASSERT(local_partitions.size() == local_partition_append_states.size());
		for (idx_t i = 0; i < local_partition_append_states.size(); i++) {
			local_partitions[i]->FlushAppendState(*local_partition_append_states[i]);
		}
		for (auto &local_partition : local_partitions) {
			global_partitions->Combine(*local_partition);
		}
		local_partitions.clear();
		local_partition_append_states.clear();
	} else {
		if (local_spill_collections.empty()) {
			global_spill_collection =
			    make_unique<ColumnDataCollection>(BufferManager::GetBufferManager(context), probe_types);
		} else {
			global_spill_collection = std::move(local_spill_collections[0]);
			for (idx_t i = 1; i < local_spill_collections.size(); i++) {
				global_spill_collection->Combine(*local_spill_collections[i]);
			}
		}
		local_spill_collections.clear();
		local_spill_append_states.clear();
	}
}

void ProbeSpill::PrepareNextProbe() {
	if (partitioned) {
		auto &partitions = global_partitions->GetPartitions();
		if (partitions.empty() || ht.partition_start == partitions.size()) {
			// Can't probe, just make an empty one
			global_spill_collection =
			    make_unique<ColumnDataCollection>(BufferManager::GetBufferManager(context), probe_types);
		} else {
			// Move specific partitions to the global spill collection
			global_spill_collection = std::move(partitions[ht.partition_start]);
			for (idx_t i = ht.partition_start + 1; i < ht.partition_end; i++) {
				global_spill_collection->Combine(*partitions[i]);
			}
		}
	}
	consumer = make_unique<ColumnDataConsumer>(*global_spill_collection, column_ids);
	consumer->InitializeScan();
}

} // namespace duckdb



namespace duckdb {

template <class OP>
struct ComparisonOperationWrapper {
	template <class T>
	static inline bool Operation(T left, T right, bool left_is_null, bool right_is_null) {
		if (left_is_null || right_is_null) {
			return false;
		}
		return OP::Operation(left, right);
	}
};

struct InitialNestedLoopJoin {
	template <class T, class OP>
	static idx_t Operation(Vector &left, Vector &right, idx_t left_size, idx_t right_size, idx_t &lpos, idx_t &rpos,
	                       SelectionVector &lvector, SelectionVector &rvector, idx_t current_match_count) {
		// initialize phase of nested loop join
		// fill lvector and rvector with matches from the base vectors
		UnifiedVectorFormat left_data, right_data;
		left.ToUnifiedFormat(left_size, left_data);
		right.ToUnifiedFormat(right_size, right_data);

		auto ldata = (T *)left_data.data;
		auto rdata = (T *)right_data.data;
		idx_t result_count = 0;
		for (; rpos < right_size; rpos++) {
			idx_t right_position = right_data.sel->get_index(rpos);
			bool right_is_valid = right_data.validity.RowIsValid(right_position);
			for (; lpos < left_size; lpos++) {
				if (result_count == STANDARD_VECTOR_SIZE) {
					// out of space!
					return result_count;
				}
				idx_t left_position = left_data.sel->get_index(lpos);
				bool left_is_valid = left_data.validity.RowIsValid(left_position);
				if (OP::Operation(ldata[left_position], rdata[right_position], !left_is_valid, !right_is_valid)) {
					// emit tuple
					lvector.set_index(result_count, lpos);
					rvector.set_index(result_count, rpos);
					result_count++;
				}
			}
			lpos = 0;
		}
		return result_count;
	}
};

struct RefineNestedLoopJoin {
	template <class T, class OP>
	static idx_t Operation(Vector &left, Vector &right, idx_t left_size, idx_t right_size, idx_t &lpos, idx_t &rpos,
	                       SelectionVector &lvector, SelectionVector &rvector, idx_t current_match_count) {
		UnifiedVectorFormat left_data, right_data;
		left.ToUnifiedFormat(left_size, left_data);
		right.ToUnifiedFormat(right_size, right_data);

		// refine phase of the nested loop join
		// refine lvector and rvector based on matches of subsequent conditions (in case there are multiple conditions
		// in the join)
		D_ASSERT(current_match_count > 0);
		auto ldata = (T *)left_data.data;
		auto rdata = (T *)right_data.data;
		idx_t result_count = 0;
		for (idx_t i = 0; i < current_match_count; i++) {
			auto lidx = lvector.get_index(i);
			auto ridx = rvector.get_index(i);
			auto left_idx = left_data.sel->get_index(lidx);
			auto right_idx = right_data.sel->get_index(ridx);
			bool left_is_valid = left_data.validity.RowIsValid(left_idx);
			bool right_is_valid = right_data.validity.RowIsValid(right_idx);
			if (OP::Operation(ldata[left_idx], rdata[right_idx], !left_is_valid, !right_is_valid)) {
				lvector.set_index(result_count, lidx);
				rvector.set_index(result_count, ridx);
				result_count++;
			}
		}
		return result_count;
	}
};

template <class NLTYPE, class OP>
static idx_t NestedLoopJoinTypeSwitch(Vector &left, Vector &right, idx_t left_size, idx_t right_size, idx_t &lpos,
                                      idx_t &rpos, SelectionVector &lvector, SelectionVector &rvector,
                                      idx_t current_match_count) {
	switch (left.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		return NLTYPE::template Operation<int8_t, OP>(left, right, left_size, right_size, lpos, rpos, lvector, rvector,
		                                              current_match_count);
	case PhysicalType::INT16:
		return NLTYPE::template Operation<int16_t, OP>(left, right, left_size, right_size, lpos, rpos, lvector, rvector,
		                                               current_match_count);
	case PhysicalType::INT32:
		return NLTYPE::template Operation<int32_t, OP>(left, right, left_size, right_size, lpos, rpos, lvector, rvector,
		                                               current_match_count);
	case PhysicalType::INT64:
		return NLTYPE::template Operation<int64_t, OP>(left, right, left_size, right_size, lpos, rpos, lvector, rvector,
		                                               current_match_count);
	case PhysicalType::UINT8:
		return NLTYPE::template Operation<uint8_t, OP>(left, right, left_size, right_size, lpos, rpos, lvector, rvector,
		                                               current_match_count);
	case PhysicalType::UINT16:
		return NLTYPE::template Operation<uint16_t, OP>(left, right, left_size, right_size, lpos, rpos, lvector,
		                                                rvector, current_match_count);
	case PhysicalType::UINT32:
		return NLTYPE::template Operation<uint32_t, OP>(left, right, left_size, right_size, lpos, rpos, lvector,
		                                                rvector, current_match_count);
	case PhysicalType::UINT64:
		return NLTYPE::template Operation<uint64_t, OP>(left, right, left_size, right_size, lpos, rpos, lvector,
		                                                rvector, current_match_count);
	case PhysicalType::INT128:
		return NLTYPE::template Operation<hugeint_t, OP>(left, right, left_size, right_size, lpos, rpos, lvector,
		                                                 rvector, current_match_count);
	case PhysicalType::FLOAT:
		return NLTYPE::template Operation<float, OP>(left, right, left_size, right_size, lpos, rpos, lvector, rvector,
		                                             current_match_count);
	case PhysicalType::DOUBLE:
		return NLTYPE::template Operation<double, OP>(left, right, left_size, right_size, lpos, rpos, lvector, rvector,
		                                              current_match_count);
	case PhysicalType::INTERVAL:
		return NLTYPE::template Operation<interval_t, OP>(left, right, left_size, right_size, lpos, rpos, lvector,
		                                                  rvector, current_match_count);
	case PhysicalType::VARCHAR:
		return NLTYPE::template Operation<string_t, OP>(left, right, left_size, right_size, lpos, rpos, lvector,
		                                                rvector, current_match_count);
	default:
		throw InternalException("Unimplemented type for join!");
	}
}

template <class NLTYPE>
idx_t NestedLoopJoinComparisonSwitch(Vector &left, Vector &right, idx_t left_size, idx_t right_size, idx_t &lpos,
                                     idx_t &rpos, SelectionVector &lvector, SelectionVector &rvector,
                                     idx_t current_match_count, ExpressionType comparison_type) {
	D_ASSERT(left.GetType() == right.GetType());
	switch (comparison_type) {
	case ExpressionType::COMPARE_EQUAL:
		return NestedLoopJoinTypeSwitch<NLTYPE, ComparisonOperationWrapper<duckdb::Equals>>(
		    left, right, left_size, right_size, lpos, rpos, lvector, rvector, current_match_count);
	case ExpressionType::COMPARE_NOTEQUAL:
		return NestedLoopJoinTypeSwitch<NLTYPE, ComparisonOperationWrapper<duckdb::NotEquals>>(
		    left, right, left_size, right_size, lpos, rpos, lvector, rvector, current_match_count);
	case ExpressionType::COMPARE_LESSTHAN:
		return NestedLoopJoinTypeSwitch<NLTYPE, ComparisonOperationWrapper<duckdb::LessThan>>(
		    left, right, left_size, right_size, lpos, rpos, lvector, rvector, current_match_count);
	case ExpressionType::COMPARE_GREATERTHAN:
		return NestedLoopJoinTypeSwitch<NLTYPE, ComparisonOperationWrapper<duckdb::GreaterThan>>(
		    left, right, left_size, right_size, lpos, rpos, lvector, rvector, current_match_count);
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return NestedLoopJoinTypeSwitch<NLTYPE, ComparisonOperationWrapper<duckdb::LessThanEquals>>(
		    left, right, left_size, right_size, lpos, rpos, lvector, rvector, current_match_count);
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return NestedLoopJoinTypeSwitch<NLTYPE, ComparisonOperationWrapper<duckdb::GreaterThanEquals>>(
		    left, right, left_size, right_size, lpos, rpos, lvector, rvector, current_match_count);
	case ExpressionType::COMPARE_DISTINCT_FROM:
		return NestedLoopJoinTypeSwitch<NLTYPE, duckdb::DistinctFrom>(left, right, left_size, right_size, lpos, rpos,
		                                                              lvector, rvector, current_match_count);
	default:
		throw NotImplementedException("Unimplemented comparison type for join!");
	}
}

idx_t NestedLoopJoinInner::Perform(idx_t &lpos, idx_t &rpos, DataChunk &left_conditions, DataChunk &right_conditions,
                                   SelectionVector &lvector, SelectionVector &rvector,
                                   const vector<JoinCondition> &conditions) {
	D_ASSERT(left_conditions.ColumnCount() == right_conditions.ColumnCount());
	if (lpos >= left_conditions.size() || rpos >= right_conditions.size()) {
		return 0;
	}
	// for the first condition, lvector and rvector are not set yet
	// we initialize them using the InitialNestedLoopJoin
	idx_t match_count = NestedLoopJoinComparisonSwitch<InitialNestedLoopJoin>(
	    left_conditions.data[0], right_conditions.data[0], left_conditions.size(), right_conditions.size(), lpos, rpos,
	    lvector, rvector, 0, conditions[0].comparison);
	// now resolve the rest of the conditions
	for (idx_t i = 1; i < conditions.size(); i++) {
		// check if we have run out of tuples to compare
		if (match_count == 0) {
			return 0;
		}
		// if not, get the vectors to compare
		Vector &l = left_conditions.data[i];
		Vector &r = right_conditions.data[i];
		// then we refine the currently obtained results using the RefineNestedLoopJoin
		match_count = NestedLoopJoinComparisonSwitch<RefineNestedLoopJoin>(
		    l, r, left_conditions.size(), right_conditions.size(), lpos, rpos, lvector, rvector, match_count,
		    conditions[i].comparison);
	}
	return match_count;
}

} // namespace duckdb




namespace duckdb {

template <class T, class OP>
static void TemplatedMarkJoin(Vector &left, Vector &right, idx_t lcount, idx_t rcount, bool found_match[]) {
	UnifiedVectorFormat left_data, right_data;
	left.ToUnifiedFormat(lcount, left_data);
	right.ToUnifiedFormat(rcount, right_data);

	auto ldata = (T *)left_data.data;
	auto rdata = (T *)right_data.data;
	for (idx_t i = 0; i < lcount; i++) {
		if (found_match[i]) {
			continue;
		}
		auto lidx = left_data.sel->get_index(i);
		if (!left_data.validity.RowIsValid(lidx)) {
			continue;
		}
		for (idx_t j = 0; j < rcount; j++) {
			auto ridx = right_data.sel->get_index(j);
			if (!right_data.validity.RowIsValid(ridx)) {
				continue;
			}
			if (OP::Operation(ldata[lidx], rdata[ridx])) {
				found_match[i] = true;
				break;
			}
		}
	}
}

static void MarkJoinNested(Vector &left, Vector &right, idx_t lcount, idx_t rcount, bool found_match[],
                           ExpressionType comparison_type) {
	Vector left_reference(left.GetType());
	SelectionVector true_sel(rcount);
	for (idx_t i = 0; i < lcount; i++) {
		if (found_match[i]) {
			continue;
		}
		ConstantVector::Reference(left_reference, left, i, rcount);
		idx_t count;
		switch (comparison_type) {
		case ExpressionType::COMPARE_EQUAL:
			count = VectorOperations::Equals(left_reference, right, nullptr, rcount, nullptr, nullptr);
			break;
		case ExpressionType::COMPARE_NOTEQUAL:
			count = VectorOperations::NotEquals(left_reference, right, nullptr, rcount, nullptr, nullptr);
			break;
		case ExpressionType::COMPARE_LESSTHAN:
			count = VectorOperations::LessThan(left_reference, right, nullptr, rcount, nullptr, nullptr);
			break;
		case ExpressionType::COMPARE_GREATERTHAN:
			count = VectorOperations::GreaterThan(left_reference, right, nullptr, rcount, nullptr, nullptr);
			break;
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			count = VectorOperations::LessThanEquals(left_reference, right, nullptr, rcount, nullptr, nullptr);
			break;
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			count = VectorOperations::GreaterThanEquals(left_reference, right, nullptr, rcount, nullptr, nullptr);
			break;
		default:
			throw InternalException("Unsupported comparison type for MarkJoinNested");
		}
		if (count > 0) {
			found_match[i] = true;
		}
	}
}

template <class OP>
static void MarkJoinSwitch(Vector &left, Vector &right, idx_t lcount, idx_t rcount, bool found_match[]) {
	switch (left.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		return TemplatedMarkJoin<int8_t, OP>(left, right, lcount, rcount, found_match);
	case PhysicalType::INT16:
		return TemplatedMarkJoin<int16_t, OP>(left, right, lcount, rcount, found_match);
	case PhysicalType::INT32:
		return TemplatedMarkJoin<int32_t, OP>(left, right, lcount, rcount, found_match);
	case PhysicalType::INT64:
		return TemplatedMarkJoin<int64_t, OP>(left, right, lcount, rcount, found_match);
	case PhysicalType::INT128:
		return TemplatedMarkJoin<hugeint_t, OP>(left, right, lcount, rcount, found_match);
	case PhysicalType::UINT8:
		return TemplatedMarkJoin<uint8_t, OP>(left, right, lcount, rcount, found_match);
	case PhysicalType::UINT16:
		return TemplatedMarkJoin<uint16_t, OP>(left, right, lcount, rcount, found_match);
	case PhysicalType::UINT32:
		return TemplatedMarkJoin<uint32_t, OP>(left, right, lcount, rcount, found_match);
	case PhysicalType::UINT64:
		return TemplatedMarkJoin<uint64_t, OP>(left, right, lcount, rcount, found_match);
	case PhysicalType::FLOAT:
		return TemplatedMarkJoin<float, OP>(left, right, lcount, rcount, found_match);
	case PhysicalType::DOUBLE:
		return TemplatedMarkJoin<double, OP>(left, right, lcount, rcount, found_match);
	case PhysicalType::VARCHAR:
		return TemplatedMarkJoin<string_t, OP>(left, right, lcount, rcount, found_match);
	default:
		throw NotImplementedException("Unimplemented type for mark join!");
	}
}

static void MarkJoinComparisonSwitch(Vector &left, Vector &right, idx_t lcount, idx_t rcount, bool found_match[],
                                     ExpressionType comparison_type) {
	switch (left.GetType().InternalType()) {
	case PhysicalType::STRUCT:
	case PhysicalType::LIST:
		return MarkJoinNested(left, right, lcount, rcount, found_match, comparison_type);
	default:
		break;
	}
	D_ASSERT(left.GetType() == right.GetType());
	switch (comparison_type) {
	case ExpressionType::COMPARE_EQUAL:
		return MarkJoinSwitch<duckdb::Equals>(left, right, lcount, rcount, found_match);
	case ExpressionType::COMPARE_NOTEQUAL:
		return MarkJoinSwitch<duckdb::NotEquals>(left, right, lcount, rcount, found_match);
	case ExpressionType::COMPARE_LESSTHAN:
		return MarkJoinSwitch<duckdb::LessThan>(left, right, lcount, rcount, found_match);
	case ExpressionType::COMPARE_GREATERTHAN:
		return MarkJoinSwitch<duckdb::GreaterThan>(left, right, lcount, rcount, found_match);
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		return MarkJoinSwitch<duckdb::LessThanEquals>(left, right, lcount, rcount, found_match);
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return MarkJoinSwitch<duckdb::GreaterThanEquals>(left, right, lcount, rcount, found_match);
	default:
		throw NotImplementedException("Unimplemented comparison type for join!");
	}
}

void NestedLoopJoinMark::Perform(DataChunk &left, ColumnDataCollection &right, bool found_match[],
                                 const vector<JoinCondition> &conditions) {
	// initialize a new temporary selection vector for the left chunk
	// loop over all chunks in the RHS
	ColumnDataScanState scan_state;
	right.InitializeScan(scan_state);

	DataChunk scan_chunk;
	right.InitializeScanChunk(scan_chunk);

	while (right.Scan(scan_state, scan_chunk)) {
		for (idx_t i = 0; i < conditions.size(); i++) {
			MarkJoinComparisonSwitch(left.data[i], scan_chunk.data[i], left.size(), scan_chunk.size(), found_match,
			                         conditions[i].comparison);
		}
	}
}

} // namespace duckdb



namespace duckdb {

AggregateObject::AggregateObject(AggregateFunction function, FunctionData *bind_data, idx_t child_count,
                                 idx_t payload_size, AggregateType aggr_type, PhysicalType return_type,
                                 Expression *filter)
    : function(std::move(function)), bind_data(bind_data), child_count(child_count), payload_size(payload_size),
      aggr_type(aggr_type), return_type(return_type), filter(filter) {
}

AggregateObject::AggregateObject(BoundAggregateExpression *aggr)
    : AggregateObject(aggr->function, aggr->bind_info.get(), aggr->children.size(),
                      AlignValue(aggr->function.state_size()), aggr->aggr_type, aggr->return_type.InternalType(),
                      aggr->filter.get()) {
}

vector<AggregateObject> AggregateObject::CreateAggregateObjects(const vector<BoundAggregateExpression *> &bindings) {
	vector<AggregateObject> aggregates;
	aggregates.reserve(aggregates.size());
	for (auto &binding : bindings) {
		aggregates.emplace_back(binding);
	}
	return aggregates;
}

AggregateFilterData::AggregateFilterData(ClientContext &context, Expression &filter_expr,
                                         const vector<LogicalType> &payload_types)
    : filter_executor(context, &filter_expr), true_sel(STANDARD_VECTOR_SIZE) {
	if (payload_types.empty()) {
		return;
	}
	filtered_payload.Initialize(Allocator::Get(context), payload_types);
}

idx_t AggregateFilterData::ApplyFilter(DataChunk &payload) {
	filtered_payload.Reset();

	auto count = filter_executor.SelectExpression(payload, true_sel);
	filtered_payload.Slice(payload, true_sel, count);
	return count;
}

AggregateFilterDataSet::AggregateFilterDataSet() {
}

void AggregateFilterDataSet::Initialize(ClientContext &context, const vector<AggregateObject> &aggregates,
                                        const vector<LogicalType> &payload_types) {
	bool has_filters = false;
	for (auto &aggregate : aggregates) {
		if (aggregate.filter) {
			has_filters = true;
			break;
		}
	}
	if (!has_filters) {
		// no filters: nothing to do
		return;
	}
	filter_data.resize(aggregates.size());
	for (idx_t aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
		auto &aggr = aggregates[aggr_idx];
		if (aggr.filter) {
			filter_data[aggr_idx] = make_unique<AggregateFilterData>(context, *aggr.filter, payload_types);
		}
	}
}

AggregateFilterData &AggregateFilterDataSet::GetFilterData(idx_t aggr_idx) {
	D_ASSERT(aggr_idx < filter_data.size());
	D_ASSERT(filter_data[aggr_idx]);
	return *filter_data[aggr_idx];
}
} // namespace duckdb






namespace duckdb {

//! Shared information about a collection of distinct aggregates
DistinctAggregateCollectionInfo::DistinctAggregateCollectionInfo(const vector<unique_ptr<Expression>> &aggregates,
                                                                 vector<idx_t> indices)
    : indices(std::move(indices)), aggregates(aggregates) {
	table_count = CreateTableIndexMap();

	const idx_t aggregate_count = aggregates.size();

	total_child_count = 0;
	for (idx_t i = 0; i < aggregate_count; i++) {
		auto &aggregate = (BoundAggregateExpression &)*aggregates[i];

		if (!aggregate.IsDistinct()) {
			continue;
		}
		total_child_count += aggregate.children.size();
	}
}

//! Stateful data for the distinct aggregates

DistinctAggregateState::DistinctAggregateState(const DistinctAggregateData &data, ClientContext &client)
    : child_executor(client) {

	radix_states.resize(data.info.table_count);
	distinct_output_chunks.resize(data.info.table_count);

	idx_t aggregate_count = data.info.aggregates.size();
	for (idx_t i = 0; i < aggregate_count; i++) {
		auto &aggregate = (BoundAggregateExpression &)*data.info.aggregates[i];

		// Initialize the child executor and get the payload types for every aggregate
		for (auto &child : aggregate.children) {
			child_executor.AddExpression(*child);
		}
		if (!aggregate.IsDistinct()) {
			continue;
		}
		D_ASSERT(data.info.table_map.count(i));
		idx_t table_idx = data.info.table_map.at(i);
		if (data.radix_tables[table_idx] == nullptr) {
			//! This table is unused because the aggregate shares its data with another
			continue;
		}

		// Get the global sinkstate for the aggregate
		auto &radix_table = *data.radix_tables[table_idx];
		radix_states[table_idx] = radix_table.GetGlobalSinkState(client);

		// Fill the chunk_types (group_by + children)
		vector<LogicalType> chunk_types;
		for (auto &group_type : data.grouped_aggregate_data[table_idx]->group_types) {
			chunk_types.push_back(group_type);
		}

		// This is used in Finalize to get the data from the radix table
		distinct_output_chunks[table_idx] = make_unique<DataChunk>();
		distinct_output_chunks[table_idx]->Initialize(client, chunk_types);
	}
}

//! Persistent + shared (read-only) data for the distinct aggregates
DistinctAggregateData::DistinctAggregateData(const DistinctAggregateCollectionInfo &info)
    : DistinctAggregateData(info, {}, nullptr) {
}

DistinctAggregateData::DistinctAggregateData(const DistinctAggregateCollectionInfo &info, const GroupingSet &groups,
                                             const vector<unique_ptr<Expression>> *group_expressions)
    : info(info) {
	grouped_aggregate_data.resize(info.table_count);
	radix_tables.resize(info.table_count);
	grouping_sets.resize(info.table_count);

	for (auto &i : info.indices) {
		auto &aggregate = (BoundAggregateExpression &)*info.aggregates[i];

		D_ASSERT(info.table_map.count(i));
		idx_t table_idx = info.table_map.at(i);
		if (radix_tables[table_idx] != nullptr) {
			//! This aggregate shares a table with another aggregate, and the table is already initialized
			continue;
		}
		// The grouping set contains the indices of the chunk that correspond to the data vector
		// that will be used to figure out in which bucket the payload should be put
		auto &grouping_set = grouping_sets[table_idx];
		//! Populate the group with the children of the aggregate
		for (auto &group : groups) {
			grouping_set.insert(group);
		}
		idx_t group_by_size = group_expressions ? group_expressions->size() : 0;
		for (idx_t set_idx = 0; set_idx < aggregate.children.size(); set_idx++) {
			grouping_set.insert(set_idx + group_by_size);
		}
		// Create the hashtable for the aggregate
		grouped_aggregate_data[table_idx] = make_unique<GroupedAggregateData>();
		grouped_aggregate_data[table_idx]->InitializeDistinct(info.aggregates[i], group_expressions);
		radix_tables[table_idx] =
		    make_unique<RadixPartitionedHashTable>(grouping_set, *grouped_aggregate_data[table_idx]);

		// Fill the chunk_types (only contains the payload of the distinct aggregates)
		vector<LogicalType> chunk_types;
		for (auto &child_p : aggregate.children) {
			chunk_types.push_back(child_p->return_type);
		}
	}
}

using aggr_ref_t = std::reference_wrapper<BoundAggregateExpression>;

struct FindMatchingAggregate {
	explicit FindMatchingAggregate(const aggr_ref_t &aggr) : aggr_r(aggr) {
	}
	bool operator()(const aggr_ref_t other_r) {
		auto &other = other_r.get();
		auto &aggr = aggr_r.get();
		if (other.children.size() != aggr.children.size()) {
			return false;
		}
		if (!Expression::Equals(aggr.filter.get(), other.filter.get())) {
			return false;
		}
		for (idx_t i = 0; i < aggr.children.size(); i++) {
			auto &other_child = (BoundReferenceExpression &)*other.children[i];
			auto &aggr_child = (BoundReferenceExpression &)*aggr.children[i];
			if (other_child.index != aggr_child.index) {
				return false;
			}
		}
		return true;
	}
	const aggr_ref_t aggr_r;
};

idx_t DistinctAggregateCollectionInfo::CreateTableIndexMap() {
	vector<aggr_ref_t> table_inputs;

	D_ASSERT(table_map.empty());
	for (auto &agg_idx : indices) {
		D_ASSERT(agg_idx < aggregates.size());
		auto &aggregate = (BoundAggregateExpression &)*aggregates[agg_idx];

		auto matching_inputs =
		    std::find_if(table_inputs.begin(), table_inputs.end(), FindMatchingAggregate(std::ref(aggregate)));
		if (matching_inputs != table_inputs.end()) {
			//! Assign the existing table to the aggregate
			idx_t found_idx = std::distance(table_inputs.begin(), matching_inputs);
			table_map[agg_idx] = found_idx;
			continue;
		}
		//! Create a new table and assign its index to the aggregate
		table_map[agg_idx] = table_inputs.size();
		table_inputs.push_back(std::ref(aggregate));
	}
	//! Every distinct aggregate needs to be assigned an index
	D_ASSERT(table_map.size() == indices.size());
	//! There can not be more tables than there are distinct aggregates
	D_ASSERT(table_inputs.size() <= indices.size());

	return table_inputs.size();
}

bool DistinctAggregateCollectionInfo::AnyDistinct() const {
	return !indices.empty();
}

const vector<idx_t> &DistinctAggregateCollectionInfo::Indices() const {
	return this->indices;
}

static vector<idx_t> GetDistinctIndices(vector<unique_ptr<Expression>> &aggregates) {
	vector<idx_t> distinct_indices;
	for (idx_t i = 0; i < aggregates.size(); i++) {
		auto &aggregate = aggregates[i];
		auto &aggr = (BoundAggregateExpression &)*aggregate;
		if (aggr.IsDistinct()) {
			distinct_indices.push_back(i);
		}
	}
	return distinct_indices;
}

unique_ptr<DistinctAggregateCollectionInfo>
DistinctAggregateCollectionInfo::Create(vector<unique_ptr<Expression>> &aggregates) {
	vector<idx_t> indices = GetDistinctIndices(aggregates);
	if (indices.empty()) {
		return nullptr;
	}
	return make_unique<DistinctAggregateCollectionInfo>(aggregates, std::move(indices));
}

bool DistinctAggregateData::IsDistinct(idx_t index) const {
	bool is_distinct = !radix_tables.empty() && info.table_map.count(index);
#ifdef DEBUG
	//! Make sure that if it is distinct, it's also in the indices
	//! And if it's not distinct, that it's also not in the indices
	bool found = false;
	for (auto &idx : info.indices) {
		if (idx == index) {
			found = true;
			break;
		}
	}
	D_ASSERT(found == is_distinct);
#endif
	return is_distinct;
}

} // namespace duckdb


namespace duckdb {

idx_t GroupedAggregateData::GroupCount() const {
	return groups.size();
}

const vector<vector<idx_t>> &GroupedAggregateData::GetGroupingFunctions() const {
	return grouping_functions;
}

void GroupedAggregateData::InitializeGroupby(vector<unique_ptr<Expression>> groups,
                                             vector<unique_ptr<Expression>> expressions,
                                             vector<vector<idx_t>> grouping_functions) {
	InitializeGroupbyGroups(std::move(groups));
	vector<LogicalType> payload_types_filters;

	SetGroupingFunctions(grouping_functions);

	filter_count = 0;
	for (auto &expr : expressions) {
		D_ASSERT(expr->expression_class == ExpressionClass::BOUND_AGGREGATE);
		D_ASSERT(expr->IsAggregate());
		auto &aggr = (BoundAggregateExpression &)*expr;
		bindings.push_back(&aggr);

		aggregate_return_types.push_back(aggr.return_type);
		for (auto &child : aggr.children) {
			payload_types.push_back(child->return_type);
		}
		if (aggr.filter) {
			filter_count++;
			payload_types_filters.push_back(aggr.filter->return_type);
		}
		if (!aggr.function.combine) {
			throw InternalException("Aggregate function %s is missing a combine method", aggr.function.name);
		}
		aggregates.push_back(std::move(expr));
	}
	for (const auto &pay_filters : payload_types_filters) {
		payload_types.push_back(pay_filters);
	}
}

void GroupedAggregateData::InitializeDistinct(const unique_ptr<Expression> &aggregate,
                                              const vector<unique_ptr<Expression>> *groups_p) {
	auto &aggr = (BoundAggregateExpression &)*aggregate;
	D_ASSERT(aggr.IsDistinct());

	// Add the (empty in ungrouped case) groups of the aggregates
	InitializeDistinctGroups(groups_p);

	// bindings.push_back(&aggr);
	filter_count = 0;
	aggregate_return_types.push_back(aggr.return_type);
	for (idx_t i = 0; i < aggr.children.size(); i++) {
		auto &child = aggr.children[i];
		group_types.push_back(child->return_type);
		groups.push_back(child->Copy());
		payload_types.push_back(child->return_type);
		if (aggr.filter) {
			filter_count++;
		}
	}
	if (!aggr.function.combine) {
		throw InternalException("Aggregate function %s is missing a combine method", aggr.function.name);
	}
}

void GroupedAggregateData::InitializeDistinctGroups(const vector<unique_ptr<Expression>> *groups_p) {
	if (!groups_p) {
		return;
	}
	for (auto &expr : *groups_p) {
		group_types.push_back(expr->return_type);
		groups.push_back(expr->Copy());
	}
}

void GroupedAggregateData::InitializeGroupbyGroups(vector<unique_ptr<Expression>> groups) {
	// Add all the expressions of the group by clause
	for (auto &expr : groups) {
		group_types.push_back(expr->return_type);
	}
	this->groups = std::move(groups);
}

void GroupedAggregateData::SetGroupingFunctions(vector<vector<idx_t>> &functions) {
	grouping_functions.reserve(functions.size());
	for (idx_t i = 0; i < functions.size(); i++) {
		grouping_functions.push_back(std::move(functions[i]));
	}
}

} // namespace duckdb
















namespace duckdb {

HashAggregateGroupingData::HashAggregateGroupingData(GroupingSet &grouping_set_p,
                                                     const GroupedAggregateData &grouped_aggregate_data,
                                                     unique_ptr<DistinctAggregateCollectionInfo> &info)
    : table_data(grouping_set_p, grouped_aggregate_data) {
	if (info) {
		distinct_data = make_unique<DistinctAggregateData>(*info, grouping_set_p, &grouped_aggregate_data.groups);
	}
}

bool HashAggregateGroupingData::HasDistinct() const {
	return distinct_data != nullptr;
}

HashAggregateGroupingGlobalState::HashAggregateGroupingGlobalState(const HashAggregateGroupingData &data,
                                                                   ClientContext &context) {
	table_state = data.table_data.GetGlobalSinkState(context);
	if (data.HasDistinct()) {
		distinct_state = make_unique<DistinctAggregateState>(*data.distinct_data, context);
	}
}

HashAggregateGroupingLocalState::HashAggregateGroupingLocalState(const PhysicalHashAggregate &op,
                                                                 const HashAggregateGroupingData &data,
                                                                 ExecutionContext &context) {
	table_state = data.table_data.GetLocalSinkState(context);
	if (!data.HasDistinct()) {
		return;
	}
	auto &distinct_data = *data.distinct_data;

	auto &distinct_indices = op.distinct_collection_info->Indices();
	D_ASSERT(!distinct_indices.empty());

	distinct_states.resize(op.distinct_collection_info->aggregates.size());
	auto &table_map = op.distinct_collection_info->table_map;

	for (auto &idx : distinct_indices) {
		idx_t table_idx = table_map[idx];
		auto &radix_table = distinct_data.radix_tables[table_idx];
		if (radix_table == nullptr) {
			// This aggregate has identical input as another aggregate, so no table is created for it
			continue;
		}
		// Initialize the states of the radix tables used for the distinct aggregates
		distinct_states[table_idx] = radix_table->GetLocalSinkState(context);
	}
}

static vector<LogicalType> CreateGroupChunkTypes(vector<unique_ptr<Expression>> &groups) {
	set<idx_t> group_indices;

	if (groups.empty()) {
		return {};
	}

	for (auto &group : groups) {
		D_ASSERT(group->type == ExpressionType::BOUND_REF);
		auto &bound_ref = (BoundReferenceExpression &)*group;
		group_indices.insert(bound_ref.index);
	}
	idx_t highest_index = *group_indices.rbegin();
	vector<LogicalType> types(highest_index + 1, LogicalType::SQLNULL);
	for (auto &group : groups) {
		auto &bound_ref = (BoundReferenceExpression &)*group;
		types[bound_ref.index] = bound_ref.return_type;
	}
	return types;
}

bool PhysicalHashAggregate::CanSkipRegularSink() const {
	if (!filter_indexes.empty()) {
		// If we have filters, we can't skip the regular sink, because we might lose groups otherwise.
		return false;
	}
	if (grouped_aggregate_data.aggregates.empty()) {
		// When there are no aggregates, we have to add to the main ht right away
		return false;
	}
	if (!non_distinct_filter.empty()) {
		return false;
	}
	return true;
}

PhysicalHashAggregate::PhysicalHashAggregate(ClientContext &context, vector<LogicalType> types,
                                             vector<unique_ptr<Expression>> expressions, idx_t estimated_cardinality)
    : PhysicalHashAggregate(context, std::move(types), std::move(expressions), {}, estimated_cardinality) {
}

PhysicalHashAggregate::PhysicalHashAggregate(ClientContext &context, vector<LogicalType> types,
                                             vector<unique_ptr<Expression>> expressions,
                                             vector<unique_ptr<Expression>> groups_p, idx_t estimated_cardinality)
    : PhysicalHashAggregate(context, std::move(types), std::move(expressions), std::move(groups_p), {}, {},
                            estimated_cardinality) {
}

PhysicalHashAggregate::PhysicalHashAggregate(ClientContext &context, vector<LogicalType> types,
                                             vector<unique_ptr<Expression>> expressions,
                                             vector<unique_ptr<Expression>> groups_p,
                                             vector<GroupingSet> grouping_sets_p,
                                             vector<vector<idx_t>> grouping_functions_p, idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::HASH_GROUP_BY, std::move(types), estimated_cardinality),
      grouping_sets(std::move(grouping_sets_p)) {
	// get a list of all aggregates to be computed
	const idx_t group_count = groups_p.size();
	if (grouping_sets.empty()) {
		GroupingSet set;
		for (idx_t i = 0; i < group_count; i++) {
			set.insert(i);
		}
		grouping_sets.push_back(std::move(set));
	}
	input_group_types = CreateGroupChunkTypes(groups_p);

	grouped_aggregate_data.InitializeGroupby(std::move(groups_p), std::move(expressions),
	                                         std::move(grouping_functions_p));

	auto &aggregates = grouped_aggregate_data.aggregates;
	// filter_indexes must be pre-built, not lazily instantiated in parallel...
	// Because everything that lives in this class should be read-only at execution time
	idx_t aggregate_input_idx = 0;
	for (idx_t i = 0; i < aggregates.size(); i++) {
		auto &aggregate = aggregates[i];
		auto &aggr = (BoundAggregateExpression &)*aggregate;
		aggregate_input_idx += aggr.children.size();
		if (aggr.aggr_type == AggregateType::DISTINCT) {
			distinct_filter.push_back(i);
		} else if (aggr.aggr_type == AggregateType::NON_DISTINCT) {
			non_distinct_filter.push_back(i);
		} else { // LCOV_EXCL_START
			throw NotImplementedException("AggregateType not implemented in PhysicalHashAggregate");
		} // LCOV_EXCL_STOP
	}

	for (idx_t i = 0; i < aggregates.size(); i++) {
		auto &aggregate = aggregates[i];
		auto &aggr = (BoundAggregateExpression &)*aggregate;
		if (aggr.filter) {
			auto &bound_ref_expr = (BoundReferenceExpression &)*aggr.filter;
			if (!filter_indexes.count(aggr.filter.get())) {
				// Replace the bound reference expression's index with the corresponding index of the payload chunk
				filter_indexes[aggr.filter.get()] = bound_ref_expr.index;
				bound_ref_expr.index = aggregate_input_idx;
			}
			aggregate_input_idx++;
		}
	}

	distinct_collection_info = DistinctAggregateCollectionInfo::Create(grouped_aggregate_data.aggregates);

	for (idx_t i = 0; i < grouping_sets.size(); i++) {
		groupings.emplace_back(grouping_sets[i], grouped_aggregate_data, distinct_collection_info);
	}
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class HashAggregateGlobalState : public GlobalSinkState {
public:
	HashAggregateGlobalState(const PhysicalHashAggregate &op, ClientContext &context) {
		grouping_states.reserve(op.groupings.size());
		for (idx_t i = 0; i < op.groupings.size(); i++) {
			auto &grouping = op.groupings[i];
			grouping_states.emplace_back(grouping, context);
		}
		vector<LogicalType> filter_types;
		for (auto &aggr : op.grouped_aggregate_data.aggregates) {
			auto &aggregate = (BoundAggregateExpression &)*aggr;
			for (auto &child : aggregate.children) {
				payload_types.push_back(child->return_type);
			}
			if (aggregate.filter) {
				filter_types.push_back(aggregate.filter->return_type);
			}
		}
		payload_types.reserve(payload_types.size() + filter_types.size());
		payload_types.insert(payload_types.end(), filter_types.begin(), filter_types.end());
	}

	vector<HashAggregateGroupingGlobalState> grouping_states;
	vector<LogicalType> payload_types;
	//! Whether or not the aggregate is finished
	bool finished = false;
};

class HashAggregateLocalState : public LocalSinkState {
public:
	HashAggregateLocalState(const PhysicalHashAggregate &op, ExecutionContext &context) {

		auto &payload_types = op.grouped_aggregate_data.payload_types;
		if (!payload_types.empty()) {
			aggregate_input_chunk.InitializeEmpty(payload_types);
		}

		grouping_states.reserve(op.groupings.size());
		for (auto &grouping : op.groupings) {
			grouping_states.emplace_back(op, grouping, context);
		}
		// The filter set is only needed here for the distinct aggregates
		// the filtering of data for the regular aggregates is done within the hashtable
		vector<AggregateObject> aggregate_objects;
		for (auto &aggregate : op.grouped_aggregate_data.aggregates) {
			auto &aggr = (BoundAggregateExpression &)*aggregate;
			aggregate_objects.emplace_back(&aggr);
		}

		filter_set.Initialize(context.client, aggregate_objects, payload_types);
	}

	DataChunk aggregate_input_chunk;
	vector<HashAggregateGroupingLocalState> grouping_states;
	AggregateFilterDataSet filter_set;
};

void PhysicalHashAggregate::SetMultiScan(GlobalSinkState &state) {
	auto &gstate = (HashAggregateGlobalState &)state;
	for (auto &grouping_state : gstate.grouping_states) {
		auto &radix_state = grouping_state.table_state;
		RadixPartitionedHashTable::SetMultiScan(*radix_state);
		if (!grouping_state.distinct_state) {
			continue;
		}
	}
}

unique_ptr<GlobalSinkState> PhysicalHashAggregate::GetGlobalSinkState(ClientContext &context) const {
	return make_unique<HashAggregateGlobalState>(*this, context);
}

unique_ptr<LocalSinkState> PhysicalHashAggregate::GetLocalSinkState(ExecutionContext &context) const {
	return make_unique<HashAggregateLocalState>(*this, context);
}

void PhysicalHashAggregate::SinkDistinctGrouping(ExecutionContext &context, GlobalSinkState &state,
                                                 LocalSinkState &lstate, DataChunk &input, idx_t grouping_idx) const {
	auto &sink = (HashAggregateLocalState &)lstate;
	auto &global_sink = (HashAggregateGlobalState &)state;

	auto &grouping_gstate = global_sink.grouping_states[grouping_idx];
	auto &grouping_lstate = sink.grouping_states[grouping_idx];
	auto &distinct_info = *distinct_collection_info;

	auto &distinct_state = grouping_gstate.distinct_state;
	auto &distinct_data = groupings[grouping_idx].distinct_data;

	DataChunk empty_chunk;

	// Create an empty filter for Sink, since we don't need to update any aggregate states here
	vector<idx_t> empty_filter;

	for (idx_t &idx : distinct_info.indices) {
		auto &aggregate = (BoundAggregateExpression &)*grouped_aggregate_data.aggregates[idx];

		D_ASSERT(distinct_info.table_map.count(idx));
		idx_t table_idx = distinct_info.table_map[idx];
		if (!distinct_data->radix_tables[table_idx]) {
			continue;
		}
		D_ASSERT(distinct_data->radix_tables[table_idx]);
		auto &radix_table = *distinct_data->radix_tables[table_idx];
		auto &radix_global_sink = *distinct_state->radix_states[table_idx];
		auto &radix_local_sink = *grouping_lstate.distinct_states[table_idx];

		if (aggregate.filter) {
			DataChunk filter_chunk;
			auto &filtered_data = sink.filter_set.GetFilterData(idx);
			filter_chunk.InitializeEmpty(filtered_data.filtered_payload.GetTypes());

			// Add the filter Vector (BOOL)
			auto it = filter_indexes.find(aggregate.filter.get());
			D_ASSERT(it != filter_indexes.end());
			D_ASSERT(it->second < input.data.size());
			auto &filter_bound_ref = (BoundReferenceExpression &)*aggregate.filter;
			filter_chunk.data[filter_bound_ref.index].Reference(input.data[it->second]);
			filter_chunk.SetCardinality(input.size());

			// We cant use the AggregateFilterData::ApplyFilter method, because the chunk we need to
			// apply the filter to also has the groups, and the filtered_data.filtered_payload does not have those.
			SelectionVector sel_vec(STANDARD_VECTOR_SIZE);
			idx_t count = filtered_data.filter_executor.SelectExpression(filter_chunk, sel_vec);

			if (count == 0) {
				continue;
			}

			// Because the 'input' chunk needs to be re-used after this, we need to create
			// a duplicate of it, that we can apply the filter to
			DataChunk filtered_input;
			filtered_input.InitializeEmpty(input.GetTypes());

			for (idx_t group_idx = 0; group_idx < grouped_aggregate_data.groups.size(); group_idx++) {
				auto &group = grouped_aggregate_data.groups[group_idx];
				auto &bound_ref = (BoundReferenceExpression &)*group;
				filtered_input.data[bound_ref.index].Reference(input.data[bound_ref.index]);
			}
			for (idx_t child_idx = 0; child_idx < aggregate.children.size(); child_idx++) {
				auto &child = aggregate.children[child_idx];
				auto &bound_ref = (BoundReferenceExpression &)*child;

				filtered_input.data[bound_ref.index].Reference(input.data[bound_ref.index]);
			}
			filtered_input.Slice(sel_vec, count);
			filtered_input.SetCardinality(count);

			radix_table.Sink(context, radix_global_sink, radix_local_sink, filtered_input, empty_chunk, empty_filter);
		} else {
			radix_table.Sink(context, radix_global_sink, radix_local_sink, input, empty_chunk, empty_filter);
		}
	}
}

void PhysicalHashAggregate::SinkDistinct(ExecutionContext &context, GlobalSinkState &state, LocalSinkState &lstate,
                                         DataChunk &input) const {
	for (idx_t i = 0; i < groupings.size(); i++) {
		SinkDistinctGrouping(context, state, lstate, input, i);
	}
}

SinkResultType PhysicalHashAggregate::Sink(ExecutionContext &context, GlobalSinkState &state, LocalSinkState &lstate,
                                           DataChunk &input) const {
	auto &llstate = (HashAggregateLocalState &)lstate;
	auto &gstate = (HashAggregateGlobalState &)state;

	if (distinct_collection_info) {
		SinkDistinct(context, state, lstate, input);
	}

	if (CanSkipRegularSink()) {
		return SinkResultType::NEED_MORE_INPUT;
	}

	DataChunk &aggregate_input_chunk = llstate.aggregate_input_chunk;

	auto &aggregates = grouped_aggregate_data.aggregates;
	idx_t aggregate_input_idx = 0;

	// Populate the aggregate child vectors
	for (auto &aggregate : aggregates) {
		auto &aggr = (BoundAggregateExpression &)*aggregate;
		for (auto &child_expr : aggr.children) {
			D_ASSERT(child_expr->type == ExpressionType::BOUND_REF);
			auto &bound_ref_expr = (BoundReferenceExpression &)*child_expr;
			D_ASSERT(bound_ref_expr.index < input.data.size());
			aggregate_input_chunk.data[aggregate_input_idx++].Reference(input.data[bound_ref_expr.index]);
		}
	}
	// Populate the filter vectors
	for (auto &aggregate : aggregates) {
		auto &aggr = (BoundAggregateExpression &)*aggregate;
		if (aggr.filter) {
			auto it = filter_indexes.find(aggr.filter.get());
			D_ASSERT(it != filter_indexes.end());
			D_ASSERT(it->second < input.data.size());
			aggregate_input_chunk.data[aggregate_input_idx++].Reference(input.data[it->second]);
		}
	}

	aggregate_input_chunk.SetCardinality(input.size());
	aggregate_input_chunk.Verify();

	// For every grouping set there is one radix_table
	for (idx_t i = 0; i < groupings.size(); i++) {
		auto &grouping_gstate = gstate.grouping_states[i];
		auto &grouping_lstate = llstate.grouping_states[i];

		auto &grouping = groupings[i];
		auto &table = grouping.table_data;
		table.Sink(context, *grouping_gstate.table_state, *grouping_lstate.table_state, input, aggregate_input_chunk,
		           non_distinct_filter);
	}

	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalHashAggregate::CombineDistinct(ExecutionContext &context, GlobalSinkState &state,
                                            LocalSinkState &lstate) const {
	auto &global_sink = (HashAggregateGlobalState &)state;
	auto &sink = (HashAggregateLocalState &)lstate;

	if (!distinct_collection_info) {
		return;
	}
	for (idx_t i = 0; i < groupings.size(); i++) {
		auto &grouping_gstate = global_sink.grouping_states[i];
		auto &grouping_lstate = sink.grouping_states[i];

		auto &distinct_data = groupings[i].distinct_data;
		auto &distinct_state = grouping_gstate.distinct_state;

		const auto table_count = distinct_data->radix_tables.size();
		for (idx_t table_idx = 0; table_idx < table_count; table_idx++) {
			if (!distinct_data->radix_tables[table_idx]) {
				continue;
			}
			auto &radix_table = *distinct_data->radix_tables[table_idx];
			auto &radix_global_sink = *distinct_state->radix_states[table_idx];
			auto &radix_local_sink = *grouping_lstate.distinct_states[table_idx];

			radix_table.Combine(context, radix_global_sink, radix_local_sink);
		}
	}
}

void PhysicalHashAggregate::Combine(ExecutionContext &context, GlobalSinkState &state, LocalSinkState &lstate) const {
	auto &gstate = (HashAggregateGlobalState &)state;
	auto &llstate = (HashAggregateLocalState &)lstate;

	CombineDistinct(context, state, lstate);

	if (CanSkipRegularSink()) {
		return;
	}
	for (idx_t i = 0; i < groupings.size(); i++) {
		auto &grouping_gstate = gstate.grouping_states[i];
		auto &grouping_lstate = llstate.grouping_states[i];

		auto &grouping = groupings[i];
		auto &table = grouping.table_data;
		table.Combine(context, *grouping_gstate.table_state, *grouping_lstate.table_state);
	}
}

//! REGULAR FINALIZE EVENT

class HashAggregateMergeEvent : public BasePipelineEvent {
public:
	HashAggregateMergeEvent(const PhysicalHashAggregate &op_p, HashAggregateGlobalState &gstate_p, Pipeline *pipeline_p)
	    : BasePipelineEvent(*pipeline_p), op(op_p), gstate(gstate_p) {
	}

	const PhysicalHashAggregate &op;
	HashAggregateGlobalState &gstate;

public:
	void Schedule() override {
		vector<unique_ptr<Task>> tasks;
		for (idx_t i = 0; i < op.groupings.size(); i++) {
			auto &grouping_gstate = gstate.grouping_states[i];

			auto &grouping = op.groupings[i];
			auto &table = grouping.table_data;
			table.ScheduleTasks(pipeline->executor, shared_from_this(), *grouping_gstate.table_state, tasks);
		}
		D_ASSERT(!tasks.empty());
		SetTasks(std::move(tasks));
	}
};

//! REGULAR FINALIZE FROM DISTINCT FINALIZE

class HashAggregateFinalizeTask : public ExecutorTask {
public:
	HashAggregateFinalizeTask(Pipeline &pipeline, shared_ptr<Event> event_p, HashAggregateGlobalState &state_p,
	                          ClientContext &context, const PhysicalHashAggregate &op)
	    : ExecutorTask(pipeline.executor), pipeline(pipeline), event(std::move(event_p)), gstate(state_p),
	      context(context), op(op) {
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		op.FinalizeInternal(pipeline, *event, context, gstate, false);
		D_ASSERT(!gstate.finished);
		gstate.finished = true;
		event->FinishTask();
		return TaskExecutionResult::TASK_FINISHED;
	}

private:
	Pipeline &pipeline;
	shared_ptr<Event> event;
	HashAggregateGlobalState &gstate;
	ClientContext &context;
	const PhysicalHashAggregate &op;
};

class HashAggregateFinalizeEvent : public BasePipelineEvent {
public:
	HashAggregateFinalizeEvent(const PhysicalHashAggregate &op_p, HashAggregateGlobalState &gstate_p,
	                           Pipeline *pipeline_p, ClientContext &context)
	    : BasePipelineEvent(*pipeline_p), op(op_p), gstate(gstate_p), context(context) {
	}

	const PhysicalHashAggregate &op;
	HashAggregateGlobalState &gstate;
	ClientContext &context;

public:
	void Schedule() override {
		vector<unique_ptr<Task>> tasks;
		tasks.push_back(make_unique<HashAggregateFinalizeTask>(*pipeline, shared_from_this(), gstate, context, op));
		D_ASSERT(!tasks.empty());
		SetTasks(std::move(tasks));
	}
};

//! DISTINCT FINALIZE TASK

class HashDistinctAggregateFinalizeTask : public ExecutorTask {
public:
	HashDistinctAggregateFinalizeTask(Pipeline &pipeline, shared_ptr<Event> event_p, HashAggregateGlobalState &state_p,
	                                  ClientContext &context, const PhysicalHashAggregate &op,
	                                  vector<vector<unique_ptr<GlobalSourceState>>> &global_sources_p)
	    : ExecutorTask(pipeline.executor), pipeline(pipeline), event(std::move(event_p)), gstate(state_p),
	      context(context), op(op), global_sources(global_sources_p) {
	}

	void AggregateDistinctGrouping(DistinctAggregateCollectionInfo &info,
	                               const HashAggregateGroupingData &grouping_data,
	                               HashAggregateGroupingGlobalState &grouping_state, idx_t grouping_idx) {
		auto &aggregates = info.aggregates;
		auto &data = *grouping_data.distinct_data;
		auto &state = *grouping_state.distinct_state;
		auto &table_state = *grouping_state.table_state;

		ThreadContext temp_thread_context(context);
		ExecutionContext temp_exec_context(context, temp_thread_context, &pipeline);

		auto temp_local_state = grouping_data.table_data.GetLocalSinkState(temp_exec_context);

		// Create a chunk that mimics the 'input' chunk in Sink, for storing the group vectors
		DataChunk group_chunk;
		if (!op.input_group_types.empty()) {
			group_chunk.Initialize(context, op.input_group_types);
		}

		auto &groups = op.grouped_aggregate_data.groups;
		const idx_t group_by_size = groups.size();

		DataChunk aggregate_input_chunk;
		if (!gstate.payload_types.empty()) {
			aggregate_input_chunk.Initialize(context, gstate.payload_types);
		}

		idx_t payload_idx;
		idx_t next_payload_idx = 0;

		for (idx_t i = 0; i < op.grouped_aggregate_data.aggregates.size(); i++) {
			auto &aggregate = (BoundAggregateExpression &)*aggregates[i];

			// Forward the payload idx
			payload_idx = next_payload_idx;
			next_payload_idx = payload_idx + aggregate.children.size();

			// If aggregate is not distinct, skip it
			if (!data.IsDistinct(i)) {
				continue;
			}
			D_ASSERT(data.info.table_map.count(i));
			auto table_idx = data.info.table_map.at(i);
			auto &radix_table_p = data.radix_tables[table_idx];

			// Create a duplicate of the output_chunk, because of multi-threading we cant alter the original
			DataChunk output_chunk;
			output_chunk.Initialize(context, state.distinct_output_chunks[table_idx]->GetTypes());

			auto &global_source = global_sources[grouping_idx][i];
			auto local_source = radix_table_p->GetLocalSourceState(temp_exec_context);

			// Fetch all the data from the aggregate ht, and Sink it into the main ht
			while (true) {
				output_chunk.Reset();
				group_chunk.Reset();
				aggregate_input_chunk.Reset();
				radix_table_p->GetData(temp_exec_context, output_chunk, *state.radix_states[table_idx], *global_source,
				                       *local_source);

				if (output_chunk.size() == 0) {
					break;
				}

				auto &grouped_aggregate_data = *data.grouped_aggregate_data[table_idx];

				for (idx_t group_idx = 0; group_idx < group_by_size; group_idx++) {
					auto &group = grouped_aggregate_data.groups[group_idx];
					auto &bound_ref_expr = (BoundReferenceExpression &)*group;
					group_chunk.data[bound_ref_expr.index].Reference(output_chunk.data[group_idx]);
				}
				group_chunk.SetCardinality(output_chunk);

				for (idx_t child_idx = 0; child_idx < grouped_aggregate_data.groups.size() - group_by_size;
				     child_idx++) {
					aggregate_input_chunk.data[payload_idx + child_idx].Reference(
					    output_chunk.data[group_by_size + child_idx]);
				}
				aggregate_input_chunk.SetCardinality(output_chunk);

				// Sink it into the main ht
				grouping_data.table_data.Sink(temp_exec_context, table_state, *temp_local_state, group_chunk,
				                              aggregate_input_chunk, {i});
			}
		}
		grouping_data.table_data.Combine(temp_exec_context, table_state, *temp_local_state);
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		D_ASSERT(op.distinct_collection_info);
		auto &info = *op.distinct_collection_info;
		for (idx_t i = 0; i < op.groupings.size(); i++) {
			auto &grouping = op.groupings[i];
			auto &grouping_state = gstate.grouping_states[i];
			AggregateDistinctGrouping(info, grouping, grouping_state, i);
		}
		event->FinishTask();
		return TaskExecutionResult::TASK_FINISHED;
	}

private:
	Pipeline &pipeline;
	shared_ptr<Event> event;
	HashAggregateGlobalState &gstate;
	ClientContext &context;
	const PhysicalHashAggregate &op;
	vector<vector<unique_ptr<GlobalSourceState>>> &global_sources;
};

//! DISTINCT FINALIZE EVENT

// TODO: Create tasks and run these in parallel instead of doing this all in Schedule, single threaded
class HashDistinctAggregateFinalizeEvent : public BasePipelineEvent {
public:
	HashDistinctAggregateFinalizeEvent(const PhysicalHashAggregate &op_p, HashAggregateGlobalState &gstate_p,
	                                   Pipeline &pipeline_p, ClientContext &context)
	    : BasePipelineEvent(pipeline_p), op(op_p), gstate(gstate_p), context(context) {
	}
	const PhysicalHashAggregate &op;
	HashAggregateGlobalState &gstate;
	ClientContext &context;
	//! The GlobalSourceStates for all the radix tables of the distinct aggregates
	vector<vector<unique_ptr<GlobalSourceState>>> global_sources;

public:
	void Schedule() override {
		global_sources = CreateGlobalSources();

		vector<unique_ptr<Task>> tasks;
		auto &scheduler = TaskScheduler::GetScheduler(context);
		auto number_of_threads = scheduler.NumberOfThreads();
		tasks.reserve(number_of_threads);
		for (int32_t i = 0; i < number_of_threads; i++) {
			tasks.push_back(make_unique<HashDistinctAggregateFinalizeTask>(*pipeline, shared_from_this(), gstate,
			                                                               context, op, global_sources));
		}
		D_ASSERT(!tasks.empty());
		SetTasks(std::move(tasks));
	}

	void FinishEvent() override {
		//! Now that everything is added to the main ht, we can actually finalize
		auto new_event = make_shared<HashAggregateFinalizeEvent>(op, gstate, pipeline.get(), context);
		this->InsertEvent(std::move(new_event));
	}

private:
	vector<vector<unique_ptr<GlobalSourceState>>> CreateGlobalSources() {
		vector<vector<unique_ptr<GlobalSourceState>>> grouping_sources;
		grouping_sources.reserve(op.groupings.size());
		for (idx_t grouping_idx = 0; grouping_idx < op.groupings.size(); grouping_idx++) {
			auto &grouping = op.groupings[grouping_idx];
			auto &data = *grouping.distinct_data;

			vector<unique_ptr<GlobalSourceState>> aggregate_sources;
			aggregate_sources.reserve(op.grouped_aggregate_data.aggregates.size());

			for (idx_t i = 0; i < op.grouped_aggregate_data.aggregates.size(); i++) {
				auto &aggregate = op.grouped_aggregate_data.aggregates[i];
				auto &aggr = (BoundAggregateExpression &)*aggregate;

				if (!aggr.IsDistinct()) {
					aggregate_sources.push_back(nullptr);
					continue;
				}

				D_ASSERT(data.info.table_map.count(i));
				auto table_idx = data.info.table_map.at(i);
				auto &radix_table_p = data.radix_tables[table_idx];
				aggregate_sources.push_back(radix_table_p->GetGlobalSourceState(context));
			}
			grouping_sources.push_back(std::move(aggregate_sources));
		}
		return grouping_sources;
	}
};

//! DISTINCT COMBINE EVENT

class HashDistinctCombineFinalizeEvent : public BasePipelineEvent {
public:
	HashDistinctCombineFinalizeEvent(const PhysicalHashAggregate &op_p, HashAggregateGlobalState &gstate_p,
	                                 Pipeline &pipeline_p, ClientContext &client)
	    : BasePipelineEvent(pipeline_p), op(op_p), gstate(gstate_p), client(client) {
	}

	const PhysicalHashAggregate &op;
	HashAggregateGlobalState &gstate;
	ClientContext &client;

public:
	void Schedule() override {
		vector<unique_ptr<Task>> tasks;
		for (idx_t i = 0; i < op.groupings.size(); i++) {
			auto &grouping = op.groupings[i];
			auto &distinct_data = *grouping.distinct_data;
			auto &distinct_state = *gstate.grouping_states[i].distinct_state;
			for (idx_t table_idx = 0; table_idx < distinct_data.radix_tables.size(); table_idx++) {
				if (!distinct_data.radix_tables[table_idx]) {
					continue;
				}
				distinct_data.radix_tables[table_idx]->ScheduleTasks(pipeline->executor, shared_from_this(),
				                                                     *distinct_state.radix_states[table_idx], tasks);
			}
		}

		D_ASSERT(!tasks.empty());
		SetTasks(std::move(tasks));
	}

	void FinishEvent() override {
		//! Now that all tables are combined, it's time to do the distinct aggregations
		auto new_event = make_shared<HashDistinctAggregateFinalizeEvent>(op, gstate, *pipeline, client);
		this->InsertEvent(std::move(new_event));
	}
};

//! FINALIZE

SinkFinalizeType PhysicalHashAggregate::FinalizeDistinct(Pipeline &pipeline, Event &event, ClientContext &context,
                                                         GlobalSinkState &gstate_p) const {
	auto &gstate = (HashAggregateGlobalState &)gstate_p;
	D_ASSERT(distinct_collection_info);

	bool any_partitioned = false;
	for (idx_t i = 0; i < groupings.size(); i++) {
		auto &grouping = groupings[i];
		auto &distinct_data = *grouping.distinct_data;
		auto &distinct_state = *gstate.grouping_states[i].distinct_state;

		for (idx_t table_idx = 0; table_idx < distinct_data.radix_tables.size(); table_idx++) {
			if (!distinct_data.radix_tables[table_idx]) {
				continue;
			}
			auto &radix_table = distinct_data.radix_tables[table_idx];
			auto &radix_state = *distinct_state.radix_states[table_idx];
			bool partitioned = radix_table->Finalize(context, radix_state);
			if (partitioned) {
				any_partitioned = true;
			}
		}
	}
	if (any_partitioned) {
		// If any of the groupings are partitioned then we first need to combine those, then aggregate
		auto new_event = make_shared<HashDistinctCombineFinalizeEvent>(*this, gstate, pipeline, context);
		event.InsertEvent(std::move(new_event));
	} else {
		// Hashtables aren't partitioned, they dont need to be joined first
		// so we can already compute the aggregate
		auto new_event = make_shared<HashDistinctAggregateFinalizeEvent>(*this, gstate, pipeline, context);
		event.InsertEvent(std::move(new_event));
	}
	return SinkFinalizeType::READY;
}

SinkFinalizeType PhysicalHashAggregate::FinalizeInternal(Pipeline &pipeline, Event &event, ClientContext &context,
                                                         GlobalSinkState &gstate_p, bool check_distinct) const {
	auto &gstate = (HashAggregateGlobalState &)gstate_p;

	if (check_distinct && distinct_collection_info) {
		// There are distinct aggregates
		// If these are partitioned those need to be combined first
		// Then we Finalize again, skipping this step
		return FinalizeDistinct(pipeline, event, context, gstate_p);
	}

	bool any_partitioned = false;
	for (idx_t i = 0; i < groupings.size(); i++) {
		auto &grouping = groupings[i];
		auto &grouping_gstate = gstate.grouping_states[i];

		bool is_partitioned = grouping.table_data.Finalize(context, *grouping_gstate.table_state);
		if (is_partitioned) {
			any_partitioned = true;
		}
	}
	if (any_partitioned) {
		auto new_event = make_shared<HashAggregateMergeEvent>(*this, gstate, &pipeline);
		event.InsertEvent(std::move(new_event));
	}
	return SinkFinalizeType::READY;
}

SinkFinalizeType PhysicalHashAggregate::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                 GlobalSinkState &gstate_p) const {
	return FinalizeInternal(pipeline, event, context, gstate_p, true);
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class PhysicalHashAggregateGlobalSourceState : public GlobalSourceState {
public:
	PhysicalHashAggregateGlobalSourceState(ClientContext &context, const PhysicalHashAggregate &op)
	    : op(op), state_index(0) {
		for (auto &grouping : op.groupings) {
			auto &rt = grouping.table_data;
			radix_states.push_back(rt.GetGlobalSourceState(context));
		}
	}

	const PhysicalHashAggregate &op;
	mutex lock;
	atomic<idx_t> state_index;

	vector<unique_ptr<GlobalSourceState>> radix_states;

public:
	idx_t MaxThreads() override {
		// If there are no tables, we only need one thread.
		if (op.groupings.empty()) {
			return 1;
		}

		auto &ht_state = (HashAggregateGlobalState &)*op.sink_state;
		idx_t count = 0;
		for (size_t sidx = 0; sidx < op.groupings.size(); ++sidx) {
			auto &grouping = op.groupings[sidx];
			auto &grouping_gstate = ht_state.grouping_states[sidx];
			count += grouping.table_data.Size(*grouping_gstate.table_state);
		}
		return MaxValue<idx_t>(1, count / RowGroup::ROW_GROUP_SIZE);
	}
};

unique_ptr<GlobalSourceState> PhysicalHashAggregate::GetGlobalSourceState(ClientContext &context) const {
	return make_unique<PhysicalHashAggregateGlobalSourceState>(context, *this);
}

class PhysicalHashAggregateLocalSourceState : public LocalSourceState {
public:
	explicit PhysicalHashAggregateLocalSourceState(ExecutionContext &context, const PhysicalHashAggregate &op) {
		for (auto &grouping : op.groupings) {
			auto &rt = grouping.table_data;
			radix_states.push_back(rt.GetLocalSourceState(context));
		}
	}

	vector<unique_ptr<LocalSourceState>> radix_states;
};

unique_ptr<LocalSourceState> PhysicalHashAggregate::GetLocalSourceState(ExecutionContext &context,
                                                                        GlobalSourceState &gstate) const {
	return make_unique<PhysicalHashAggregateLocalSourceState>(context, *this);
}

void PhysicalHashAggregate::GetData(ExecutionContext &context, DataChunk &chunk, GlobalSourceState &gstate_p,
                                    LocalSourceState &lstate_p) const {
	auto &sink_gstate = (HashAggregateGlobalState &)*sink_state;
	auto &gstate = (PhysicalHashAggregateGlobalSourceState &)gstate_p;
	auto &lstate = (PhysicalHashAggregateLocalSourceState &)lstate_p;
	while (true) {
		idx_t radix_idx = gstate.state_index;
		if (radix_idx >= groupings.size()) {
			break;
		}
		auto &grouping = groupings[radix_idx];
		auto &radix_table = grouping.table_data;
		auto &grouping_gstate = sink_gstate.grouping_states[radix_idx];
		radix_table.GetData(context, chunk, *grouping_gstate.table_state, *gstate.radix_states[radix_idx],
		                    *lstate.radix_states[radix_idx]);
		if (chunk.size() != 0) {
			return;
		}
		// move to the next table
		lock_guard<mutex> l(gstate.lock);
		radix_idx++;
		if (radix_idx > gstate.state_index) {
			// we have not yet worked on the table
			// move the global index forwards
			gstate.state_index = radix_idx;
		}
	}
}

string PhysicalHashAggregate::ParamsToString() const {
	string result;
	auto &groups = grouped_aggregate_data.groups;
	auto &aggregates = grouped_aggregate_data.aggregates;
	for (idx_t i = 0; i < groups.size(); i++) {
		if (i > 0) {
			result += "\n";
		}
		result += groups[i]->GetName();
	}
	for (idx_t i = 0; i < aggregates.size(); i++) {
		auto &aggregate = (BoundAggregateExpression &)*aggregates[i];
		if (i > 0 || !groups.empty()) {
			result += "\n";
		}
		result += aggregates[i]->GetName();
		if (aggregate.filter) {
			result += " Filter: " + aggregate.filter->GetName();
		}
	}
	return result;
}

} // namespace duckdb








namespace duckdb {

PhysicalPerfectHashAggregate::PhysicalPerfectHashAggregate(ClientContext &context, vector<LogicalType> types_p,
                                                           vector<unique_ptr<Expression>> aggregates_p,
                                                           vector<unique_ptr<Expression>> groups_p,
                                                           vector<unique_ptr<BaseStatistics>> group_stats,
                                                           vector<idx_t> required_bits_p, idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::PERFECT_HASH_GROUP_BY, std::move(types_p), estimated_cardinality),
      groups(std::move(groups_p)), aggregates(std::move(aggregates_p)), required_bits(std::move(required_bits_p)) {
	D_ASSERT(groups.size() == group_stats.size());
	group_minima.reserve(group_stats.size());
	for (auto &stats : group_stats) {
		D_ASSERT(stats);
		auto &nstats = (NumericStatistics &)*stats;
		D_ASSERT(!nstats.min.IsNull());
		group_minima.push_back(std::move(nstats.min));
	}
	for (auto &expr : groups) {
		group_types.push_back(expr->return_type);
	}

	vector<BoundAggregateExpression *> bindings;
	vector<LogicalType> payload_types_filters;
	for (auto &expr : aggregates) {
		D_ASSERT(expr->expression_class == ExpressionClass::BOUND_AGGREGATE);
		D_ASSERT(expr->IsAggregate());
		auto &aggr = (BoundAggregateExpression &)*expr;
		bindings.push_back(&aggr);

		D_ASSERT(!aggr.IsDistinct());
		D_ASSERT(aggr.function.combine);
		for (auto &child : aggr.children) {
			payload_types.push_back(child->return_type);
		}
		if (aggr.filter) {
			payload_types_filters.push_back(aggr.filter->return_type);
		}
	}
	for (const auto &pay_filters : payload_types_filters) {
		payload_types.push_back(pay_filters);
	}
	aggregate_objects = AggregateObject::CreateAggregateObjects(bindings);

	// filter_indexes must be pre-built, not lazily instantiated in parallel...
	idx_t aggregate_input_idx = 0;
	for (auto &aggregate : aggregates) {
		auto &aggr = (BoundAggregateExpression &)*aggregate;
		aggregate_input_idx += aggr.children.size();
	}
	for (auto &aggregate : aggregates) {
		auto &aggr = (BoundAggregateExpression &)*aggregate;
		if (aggr.filter) {
			auto &bound_ref_expr = (BoundReferenceExpression &)*aggr.filter;
			auto it = filter_indexes.find(aggr.filter.get());
			if (it == filter_indexes.end()) {
				filter_indexes[aggr.filter.get()] = bound_ref_expr.index;
				bound_ref_expr.index = aggregate_input_idx++;
			} else {
				++aggregate_input_idx;
			}
		}
	}
}

unique_ptr<PerfectAggregateHashTable> PhysicalPerfectHashAggregate::CreateHT(Allocator &allocator,
                                                                             ClientContext &context) const {
	return make_unique<PerfectAggregateHashTable>(context, allocator, group_types, payload_types, aggregate_objects,
	                                              group_minima, required_bits);
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class PerfectHashAggregateGlobalState : public GlobalSinkState {
public:
	PerfectHashAggregateGlobalState(const PhysicalPerfectHashAggregate &op, ClientContext &context)
	    : ht(op.CreateHT(Allocator::Get(context), context)) {
	}

	//! The lock for updating the global aggregate state
	mutex lock;
	//! The global aggregate hash table
	unique_ptr<PerfectAggregateHashTable> ht;
};

class PerfectHashAggregateLocalState : public LocalSinkState {
public:
	PerfectHashAggregateLocalState(const PhysicalPerfectHashAggregate &op, ExecutionContext &context)
	    : ht(op.CreateHT(Allocator::Get(context.client), context.client)) {
		group_chunk.InitializeEmpty(op.group_types);
		if (!op.payload_types.empty()) {
			aggregate_input_chunk.InitializeEmpty(op.payload_types);
		}
	}

	//! The local aggregate hash table
	unique_ptr<PerfectAggregateHashTable> ht;
	DataChunk group_chunk;
	DataChunk aggregate_input_chunk;
};

unique_ptr<GlobalSinkState> PhysicalPerfectHashAggregate::GetGlobalSinkState(ClientContext &context) const {
	return make_unique<PerfectHashAggregateGlobalState>(*this, context);
}

unique_ptr<LocalSinkState> PhysicalPerfectHashAggregate::GetLocalSinkState(ExecutionContext &context) const {
	return make_unique<PerfectHashAggregateLocalState>(*this, context);
}

SinkResultType PhysicalPerfectHashAggregate::Sink(ExecutionContext &context, GlobalSinkState &state,
                                                  LocalSinkState &lstate_p, DataChunk &input) const {
	auto &lstate = (PerfectHashAggregateLocalState &)lstate_p;
	DataChunk &group_chunk = lstate.group_chunk;
	DataChunk &aggregate_input_chunk = lstate.aggregate_input_chunk;

	for (idx_t group_idx = 0; group_idx < groups.size(); group_idx++) {
		auto &group = groups[group_idx];
		D_ASSERT(group->type == ExpressionType::BOUND_REF);
		auto &bound_ref_expr = (BoundReferenceExpression &)*group;
		group_chunk.data[group_idx].Reference(input.data[bound_ref_expr.index]);
	}
	idx_t aggregate_input_idx = 0;
	for (auto &aggregate : aggregates) {
		auto &aggr = (BoundAggregateExpression &)*aggregate;
		for (auto &child_expr : aggr.children) {
			D_ASSERT(child_expr->type == ExpressionType::BOUND_REF);
			auto &bound_ref_expr = (BoundReferenceExpression &)*child_expr;
			aggregate_input_chunk.data[aggregate_input_idx++].Reference(input.data[bound_ref_expr.index]);
		}
	}
	for (auto &aggregate : aggregates) {
		auto &aggr = (BoundAggregateExpression &)*aggregate;
		if (aggr.filter) {
			auto it = filter_indexes.find(aggr.filter.get());
			D_ASSERT(it != filter_indexes.end());
			aggregate_input_chunk.data[aggregate_input_idx++].Reference(input.data[it->second]);
		}
	}

	group_chunk.SetCardinality(input.size());

	aggregate_input_chunk.SetCardinality(input.size());

	group_chunk.Verify();
	aggregate_input_chunk.Verify();
	D_ASSERT(aggregate_input_chunk.ColumnCount() == 0 || group_chunk.size() == aggregate_input_chunk.size());

	lstate.ht->AddChunk(group_chunk, aggregate_input_chunk);
	return SinkResultType::NEED_MORE_INPUT;
}

//===--------------------------------------------------------------------===//
// Combine
//===--------------------------------------------------------------------===//
void PhysicalPerfectHashAggregate::Combine(ExecutionContext &context, GlobalSinkState &gstate_p,
                                           LocalSinkState &lstate_p) const {
	auto &lstate = (PerfectHashAggregateLocalState &)lstate_p;
	auto &gstate = (PerfectHashAggregateGlobalState &)gstate_p;

	lock_guard<mutex> l(gstate.lock);
	gstate.ht->Combine(*lstate.ht);
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class PerfectHashAggregateState : public GlobalSourceState {
public:
	PerfectHashAggregateState() : ht_scan_position(0) {
	}

	//! The current position to scan the HT for output tuples
	idx_t ht_scan_position;
};

unique_ptr<GlobalSourceState> PhysicalPerfectHashAggregate::GetGlobalSourceState(ClientContext &context) const {
	return make_unique<PerfectHashAggregateState>();
}

void PhysicalPerfectHashAggregate::GetData(ExecutionContext &context, DataChunk &chunk, GlobalSourceState &gstate_p,
                                           LocalSourceState &lstate) const {
	auto &state = (PerfectHashAggregateState &)gstate_p;
	auto &gstate = (PerfectHashAggregateGlobalState &)*sink_state;

	gstate.ht->Scan(state.ht_scan_position, chunk);
}

string PhysicalPerfectHashAggregate::ParamsToString() const {
	string result;
	for (idx_t i = 0; i < groups.size(); i++) {
		if (i > 0) {
			result += "\n";
		}
		result += groups[i]->GetName();
	}
	for (idx_t i = 0; i < aggregates.size(); i++) {
		if (i > 0 || !groups.empty()) {
			result += "\n";
		}
		result += aggregates[i]->GetName();
		auto &aggregate = (BoundAggregateExpression &)*aggregates[i];
		if (aggregate.filter) {
			result += " Filter: " + aggregate.filter->GetName();
		}
	}
	return result;
}

} // namespace duckdb








namespace duckdb {

PhysicalStreamingWindow::PhysicalStreamingWindow(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list,
                                                 idx_t estimated_cardinality, PhysicalOperatorType type)
    : PhysicalOperator(type, std::move(types), estimated_cardinality), select_list(std::move(select_list)) {
}

class StreamingWindowGlobalState : public GlobalOperatorState {
public:
	StreamingWindowGlobalState() : row_number(1) {
	}

	//! The next row number.
	std::atomic<int64_t> row_number;
};

class StreamingWindowState : public OperatorState {
public:
	using StateBuffer = vector<data_t>;

	StreamingWindowState() : initialized(false), statev(LogicalType::POINTER, (data_ptr_t)&state_ptr) {
	}

	~StreamingWindowState() override {
		for (size_t i = 0; i < aggregate_dtors.size(); ++i) {
			auto dtor = aggregate_dtors[i];
			if (dtor) {
				state_ptr = aggregate_states[i].data();
				dtor(statev, 1);
			}
		}
	}

	void Initialize(ClientContext &context, DataChunk &input, const vector<unique_ptr<Expression>> &expressions) {
		const_vectors.resize(expressions.size());
		aggregate_states.resize(expressions.size());
		aggregate_dtors.resize(expressions.size(), nullptr);

		for (idx_t expr_idx = 0; expr_idx < expressions.size(); expr_idx++) {
			auto &expr = *expressions[expr_idx];
			auto &wexpr = (BoundWindowExpression &)expr;
			switch (expr.GetExpressionType()) {
			case ExpressionType::WINDOW_AGGREGATE: {
				auto &aggregate = *wexpr.aggregate;
				auto &state = aggregate_states[expr_idx];
				aggregate_dtors[expr_idx] = aggregate.destructor;
				state.resize(aggregate.state_size());
				aggregate.initialize(state.data());
				break;
			}
			case ExpressionType::WINDOW_FIRST_VALUE: {
				// Just execute the expression once
				ExpressionExecutor executor(context);
				executor.AddExpression(*wexpr.children[0]);
				DataChunk result;
				result.Initialize(Allocator::Get(context), {wexpr.children[0]->return_type});
				executor.Execute(input, result);

				const_vectors[expr_idx] = make_unique<Vector>(result.GetValue(0, 0));
				break;
			}
			case ExpressionType::WINDOW_PERCENT_RANK: {
				const_vectors[expr_idx] = make_unique<Vector>(Value((double)0));
				break;
			}
			case ExpressionType::WINDOW_RANK:
			case ExpressionType::WINDOW_RANK_DENSE: {
				const_vectors[expr_idx] = make_unique<Vector>(Value((int64_t)1));
				break;
			}
			default:
				break;
			}
		}
		initialized = true;
	}

public:
	bool initialized;
	vector<unique_ptr<Vector>> const_vectors;

	// Aggregation
	vector<StateBuffer> aggregate_states;
	vector<aggregate_destructor_t> aggregate_dtors;
	data_ptr_t state_ptr;
	Vector statev;
};

unique_ptr<GlobalOperatorState> PhysicalStreamingWindow::GetGlobalOperatorState(ClientContext &context) const {
	return make_unique<StreamingWindowGlobalState>();
}

unique_ptr<OperatorState> PhysicalStreamingWindow::GetOperatorState(ExecutionContext &context) const {
	return make_unique<StreamingWindowState>();
}

OperatorResultType PhysicalStreamingWindow::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                    GlobalOperatorState &gstate_p, OperatorState &state_p) const {
	auto &gstate = (StreamingWindowGlobalState &)gstate_p;
	auto &state = (StreamingWindowState &)state_p;
	if (!state.initialized) {
		state.Initialize(context.client, input, select_list);
	}
	// Put payload columns in place
	for (idx_t col_idx = 0; col_idx < input.data.size(); col_idx++) {
		chunk.data[col_idx].Reference(input.data[col_idx]);
	}
	// Compute window function
	const idx_t count = input.size();
	for (idx_t expr_idx = 0; expr_idx < select_list.size(); expr_idx++) {
		idx_t col_idx = input.data.size() + expr_idx;
		auto &expr = *select_list[expr_idx];
		auto &result = chunk.data[col_idx];
		switch (expr.GetExpressionType()) {
		case ExpressionType::WINDOW_AGGREGATE: {
			//	Establish the aggregation environment
			auto &wexpr = (BoundWindowExpression &)expr;
			auto &aggregate = *wexpr.aggregate;
			auto &statev = state.statev;
			state.state_ptr = state.aggregate_states[expr_idx].data();
			AggregateInputData aggr_input_data(wexpr.bind_info.get(), Allocator::DefaultAllocator());

			// Check for COUNT(*)
			if (wexpr.children.empty()) {
				D_ASSERT(GetTypeIdSize(result.GetType().InternalType()) == sizeof(int64_t));
				auto data = FlatVector::GetData<int64_t>(result);
				int64_t start_row = gstate.row_number;
				for (idx_t i = 0; i < input.size(); ++i) {
					data[i] = start_row + i;
				}
				break;
			}

			// Compute the arguments
			auto &allocator = Allocator::Get(context.client);
			ExpressionExecutor executor(context.client);
			vector<LogicalType> payload_types;
			for (auto &child : wexpr.children) {
				payload_types.push_back(child->return_type);
				executor.AddExpression(*child);
			}

			DataChunk payload;
			payload.Initialize(allocator, payload_types);
			executor.Execute(input, payload);

			// Iterate through them using a single SV
			payload.Flatten();
			DataChunk row;
			row.Initialize(allocator, payload_types);
			sel_t s = 0;
			SelectionVector sel(&s);
			row.Slice(sel, 1);
			for (size_t col_idx = 0; col_idx < payload.ColumnCount(); ++col_idx) {
				DictionaryVector::Child(row.data[col_idx]).Reference(payload.data[col_idx]);
			}

			// Update the state and finalize it one row at a time.
			for (idx_t i = 0; i < input.size(); ++i) {
				sel.set_index(0, i);
				aggregate.update(row.data.data(), aggr_input_data, row.ColumnCount(), statev, 1);
				aggregate.finalize(statev, aggr_input_data, result, 1, i);
			}
			break;
		}
		case ExpressionType::WINDOW_FIRST_VALUE:
		case ExpressionType::WINDOW_PERCENT_RANK:
		case ExpressionType::WINDOW_RANK:
		case ExpressionType::WINDOW_RANK_DENSE: {
			// Reference constant vector
			chunk.data[col_idx].Reference(*state.const_vectors[expr_idx]);
			break;
		}
		case ExpressionType::WINDOW_ROW_NUMBER: {
			// Set row numbers
			int64_t start_row = gstate.row_number;
			auto rdata = FlatVector::GetData<int64_t>(chunk.data[col_idx]);
			for (idx_t i = 0; i < count; i++) {
				rdata[i] = start_row + i;
			}
			break;
		}
		default:
			throw NotImplementedException("%s for StreamingWindow", ExpressionTypeToString(expr.GetExpressionType()));
		}
	}
	gstate.row_number += count;
	chunk.SetCardinality(count);
	return OperatorResultType::NEED_MORE_INPUT;
}

string PhysicalStreamingWindow::ParamsToString() const {
	string result;
	for (idx_t i = 0; i < select_list.size(); i++) {
		if (i > 0) {
			result += "\n";
		}
		result += select_list[i]->GetName();
	}
	return result;
}

} // namespace duckdb














#include <functional>


namespace duckdb {

PhysicalUngroupedAggregate::PhysicalUngroupedAggregate(vector<LogicalType> types,
                                                       vector<unique_ptr<Expression>> expressions,
                                                       idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::UNGROUPED_AGGREGATE, std::move(types), estimated_cardinality),
      aggregates(std::move(expressions)) {

	distinct_collection_info = DistinctAggregateCollectionInfo::Create(aggregates);
	if (!distinct_collection_info) {
		return;
	}
	distinct_data = make_unique<DistinctAggregateData>(*distinct_collection_info);
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
struct AggregateState {
	explicit AggregateState(const vector<unique_ptr<Expression>> &aggregate_expressions) {
		for (auto &aggregate : aggregate_expressions) {
			D_ASSERT(aggregate->GetExpressionClass() == ExpressionClass::BOUND_AGGREGATE);
			auto &aggr = (BoundAggregateExpression &)*aggregate;
			auto state = unique_ptr<data_t[]>(new data_t[aggr.function.state_size()]);
			aggr.function.initialize(state.get());
			aggregates.push_back(std::move(state));
			destructors.push_back(aggr.function.destructor);
#ifdef DEBUG
			counts.push_back(0);
#endif
		}
	}
	~AggregateState() {
		D_ASSERT(destructors.size() == aggregates.size());
		for (idx_t i = 0; i < destructors.size(); i++) {
			if (!destructors[i]) {
				continue;
			}
			Vector state_vector(Value::POINTER((uintptr_t)aggregates[i].get()));
			state_vector.SetVectorType(VectorType::FLAT_VECTOR);

			destructors[i](state_vector, 1);
		}
	}

	void Move(AggregateState &other) {
		other.aggregates = std::move(aggregates);
		other.destructors = std::move(destructors);
	}

	//! The aggregate values
	vector<unique_ptr<data_t[]>> aggregates;
	//! The destructors
	vector<aggregate_destructor_t> destructors;
	//! Counts (used for verification)
	vector<idx_t> counts;
};

class UngroupedAggregateGlobalState : public GlobalSinkState {
public:
	UngroupedAggregateGlobalState(const PhysicalUngroupedAggregate &op, ClientContext &client)
	    : state(op.aggregates), finished(false) {
		if (op.distinct_data) {
			distinct_state = make_unique<DistinctAggregateState>(*op.distinct_data, client);
		}
	}

	//! The lock for updating the global aggregate state
	mutex lock;
	//! The global aggregate state
	AggregateState state;
	//! Whether or not the aggregate is finished
	bool finished;
	//! The data related to the distinct aggregates (if there are any)
	unique_ptr<DistinctAggregateState> distinct_state;
};

class UngroupedAggregateLocalState : public LocalSinkState {
public:
	UngroupedAggregateLocalState(const PhysicalUngroupedAggregate &op, const vector<LogicalType> &child_types,
	                             GlobalSinkState &gstate_p, ExecutionContext &context)
	    : state(op.aggregates), child_executor(context.client), aggregate_input_chunk(), filter_set() {
		auto &gstate = (UngroupedAggregateGlobalState &)gstate_p;

		auto &allocator = Allocator::Get(context.client);
		InitializeDistinctAggregates(op, gstate, context);

		vector<LogicalType> payload_types;
		vector<AggregateObject> aggregate_objects;
		for (auto &aggregate : op.aggregates) {
			D_ASSERT(aggregate->GetExpressionClass() == ExpressionClass::BOUND_AGGREGATE);
			auto &aggr = (BoundAggregateExpression &)*aggregate;
			// initialize the payload chunk
			for (auto &child : aggr.children) {
				payload_types.push_back(child->return_type);
				child_executor.AddExpression(*child);
			}
			aggregate_objects.emplace_back(&aggr);
		}
		if (!payload_types.empty()) { // for select count(*) from t; there is no payload at all
			aggregate_input_chunk.Initialize(allocator, payload_types);
		}
		filter_set.Initialize(context.client, aggregate_objects, child_types);
	}

	//! The local aggregate state
	AggregateState state;
	//! The executor
	ExpressionExecutor child_executor;
	//! The payload chunk, containing all the Vectors for the aggregates
	DataChunk aggregate_input_chunk;
	//! Aggregate filter data set
	AggregateFilterDataSet filter_set;
	//! The local sink states of the distinct aggregates hash tables
	vector<unique_ptr<LocalSinkState>> radix_states;

public:
	void Reset() {
		aggregate_input_chunk.Reset();
	}
	void InitializeDistinctAggregates(const PhysicalUngroupedAggregate &op, const UngroupedAggregateGlobalState &gstate,
	                                  ExecutionContext &context) {

		if (!op.distinct_data) {
			return;
		}
		auto &data = *op.distinct_data;
		auto &state = *gstate.distinct_state;
		D_ASSERT(!data.radix_tables.empty());

		const idx_t aggregate_count = state.radix_states.size();
		radix_states.resize(aggregate_count);

		auto &distinct_info = *op.distinct_collection_info;

		for (auto &idx : distinct_info.indices) {
			idx_t table_idx = distinct_info.table_map[idx];
			if (data.radix_tables[table_idx] == nullptr) {
				// This aggregate has identical input as another aggregate, so no table is created for it
				continue;
			}
			auto &radix_table = *data.radix_tables[table_idx];
			radix_states[table_idx] = radix_table.GetLocalSinkState(context);
		}
	}
};

unique_ptr<GlobalSinkState> PhysicalUngroupedAggregate::GetGlobalSinkState(ClientContext &context) const {
	return make_unique<UngroupedAggregateGlobalState>(*this, context);
}

unique_ptr<LocalSinkState> PhysicalUngroupedAggregate::GetLocalSinkState(ExecutionContext &context) const {
	D_ASSERT(sink_state);
	auto &gstate = *sink_state;
	return make_unique<UngroupedAggregateLocalState>(*this, children[0]->GetTypes(), gstate, context);
}

void PhysicalUngroupedAggregate::SinkDistinct(ExecutionContext &context, GlobalSinkState &state, LocalSinkState &lstate,
                                              DataChunk &input) const {
	auto &sink = (UngroupedAggregateLocalState &)lstate;
	auto &global_sink = (UngroupedAggregateGlobalState &)state;
	D_ASSERT(distinct_data);
	auto &distinct_state = *global_sink.distinct_state;
	auto &distinct_info = *distinct_collection_info;
	auto &distinct_indices = distinct_info.Indices();

	DataChunk empty_chunk;

	auto &distinct_filter = distinct_info.Indices();

	for (auto &idx : distinct_indices) {
		auto &aggregate = (BoundAggregateExpression &)*aggregates[idx];

		idx_t table_idx = distinct_info.table_map[idx];
		if (!distinct_data->radix_tables[table_idx]) {
			// This distinct aggregate shares its data with another
			continue;
		}
		D_ASSERT(distinct_data->radix_tables[table_idx]);
		auto &radix_table = *distinct_data->radix_tables[table_idx];
		auto &radix_global_sink = *distinct_state.radix_states[table_idx];
		auto &radix_local_sink = *sink.radix_states[table_idx];

		if (aggregate.filter) {
			// The hashtable can apply a filter, but only on the payload
			// And in our case, we need to filter the groups (the distinct aggr children)

			// Apply the filter before inserting into the hashtable
			auto &filtered_data = sink.filter_set.GetFilterData(idx);
			idx_t count = filtered_data.ApplyFilter(input);
			filtered_data.filtered_payload.SetCardinality(count);

			radix_table.Sink(context, radix_global_sink, radix_local_sink, filtered_data.filtered_payload, empty_chunk,
			                 distinct_filter);
		} else {
			radix_table.Sink(context, radix_global_sink, radix_local_sink, input, empty_chunk, distinct_filter);
		}
	}
}

SinkResultType PhysicalUngroupedAggregate::Sink(ExecutionContext &context, GlobalSinkState &state,
                                                LocalSinkState &lstate, DataChunk &input) const {
	auto &sink = (UngroupedAggregateLocalState &)lstate;

	// perform the aggregation inside the local state
	sink.Reset();

	if (distinct_data) {
		SinkDistinct(context, state, lstate, input);
	}

	DataChunk &payload_chunk = sink.aggregate_input_chunk;

	idx_t payload_idx = 0;
	idx_t next_payload_idx = 0;

	for (idx_t aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
		auto &aggregate = (BoundAggregateExpression &)*aggregates[aggr_idx];

		payload_idx = next_payload_idx;
		next_payload_idx = payload_idx + aggregate.children.size();

		if (aggregate.IsDistinct()) {
			continue;
		}

		idx_t payload_cnt = 0;
		// resolve the filter (if any)
		if (aggregate.filter) {
			auto &filtered_data = sink.filter_set.GetFilterData(aggr_idx);
			auto count = filtered_data.ApplyFilter(input);

			sink.child_executor.SetChunk(filtered_data.filtered_payload);
			payload_chunk.SetCardinality(count);
		} else {
			sink.child_executor.SetChunk(input);
			payload_chunk.SetCardinality(input);
		}

#ifdef DEBUG
		sink.state.counts[aggr_idx] += payload_chunk.size();
#endif

		// resolve the child expressions of the aggregate (if any)
		for (idx_t i = 0; i < aggregate.children.size(); ++i) {
			sink.child_executor.ExecuteExpression(payload_idx + payload_cnt,
			                                      payload_chunk.data[payload_idx + payload_cnt]);
			payload_cnt++;
		}

		auto start_of_input = payload_cnt == 0 ? nullptr : &payload_chunk.data[payload_idx];
		AggregateInputData aggr_input_data(aggregate.bind_info.get(), Allocator::DefaultAllocator());
		aggregate.function.simple_update(start_of_input, aggr_input_data, payload_cnt,
		                                 sink.state.aggregates[aggr_idx].get(), payload_chunk.size());
	}
	return SinkResultType::NEED_MORE_INPUT;
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//

void PhysicalUngroupedAggregate::CombineDistinct(ExecutionContext &context, GlobalSinkState &state,
                                                 LocalSinkState &lstate) const {
	auto &global_sink = (UngroupedAggregateGlobalState &)state;
	auto &source = (UngroupedAggregateLocalState &)lstate;

	if (!distinct_data) {
		return;
	}
	auto &distinct_state = global_sink.distinct_state;
	auto table_count = distinct_data->radix_tables.size();
	for (idx_t table_idx = 0; table_idx < table_count; table_idx++) {
		D_ASSERT(distinct_data->radix_tables[table_idx]);
		auto &radix_table = *distinct_data->radix_tables[table_idx];
		auto &radix_global_sink = *distinct_state->radix_states[table_idx];
		auto &radix_local_sink = *source.radix_states[table_idx];

		radix_table.Combine(context, radix_global_sink, radix_local_sink);
	}
}

void PhysicalUngroupedAggregate::Combine(ExecutionContext &context, GlobalSinkState &state,
                                         LocalSinkState &lstate) const {
	auto &gstate = (UngroupedAggregateGlobalState &)state;
	auto &source = (UngroupedAggregateLocalState &)lstate;
	D_ASSERT(!gstate.finished);

	// finalize: combine the local state into the global state
	// all aggregates are combinable: we might be doing a parallel aggregate
	// use the combine method to combine the partial aggregates
	lock_guard<mutex> glock(gstate.lock);

	CombineDistinct(context, state, lstate);

	for (idx_t aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
		auto &aggregate = (BoundAggregateExpression &)*aggregates[aggr_idx];

		if (aggregate.IsDistinct()) {
			continue;
		}

		Vector source_state(Value::POINTER((uintptr_t)source.state.aggregates[aggr_idx].get()));
		Vector dest_state(Value::POINTER((uintptr_t)gstate.state.aggregates[aggr_idx].get()));

		AggregateInputData aggr_input_data(aggregate.bind_info.get(), Allocator::DefaultAllocator());
		aggregate.function.combine(source_state, dest_state, aggr_input_data, 1);
#ifdef DEBUG
		gstate.state.counts[aggr_idx] += source.state.counts[aggr_idx];
#endif
	}

	auto &client_profiler = QueryProfiler::Get(context.client);
	context.thread.profiler.Flush(this, &source.child_executor, "child_executor", 0);
	client_profiler.Flush(context.thread.profiler);
}

class UngroupedDistinctAggregateFinalizeTask : public ExecutorTask {
public:
	UngroupedDistinctAggregateFinalizeTask(Executor &executor, shared_ptr<Event> event_p,
	                                       UngroupedAggregateGlobalState &state_p, ClientContext &context,
	                                       const PhysicalUngroupedAggregate &op)
	    : ExecutorTask(executor), event(std::move(event_p)), gstate(state_p), context(context), op(op) {
	}

	void AggregateDistinct() {
		D_ASSERT(gstate.distinct_state);
		auto &aggregates = op.aggregates;
		auto &distinct_state = *gstate.distinct_state;
		auto &distinct_data = *op.distinct_data;

		ThreadContext temp_thread_context(context);
		ExecutionContext temp_exec_context(context, temp_thread_context, nullptr);

		idx_t payload_idx = 0;
		idx_t next_payload_idx = 0;

		for (idx_t i = 0; i < aggregates.size(); i++) {
			auto &aggregate = (BoundAggregateExpression &)*aggregates[i];

			// Forward the payload idx
			payload_idx = next_payload_idx;
			next_payload_idx = payload_idx + aggregate.children.size();

			// If aggregate is not distinct, skip it
			if (!distinct_data.IsDistinct(i)) {
				continue;
			}

			DataChunk payload_chunk;

			D_ASSERT(distinct_data.info.table_map.count(i));
			auto table_idx = distinct_data.info.table_map.at(i);
			auto &radix_table_p = distinct_data.radix_tables[table_idx];
			auto &output_chunk = *distinct_state.distinct_output_chunks[table_idx];
			auto &grouped_aggregate_data = *distinct_data.grouped_aggregate_data[table_idx];

			payload_chunk.InitializeEmpty(grouped_aggregate_data.group_types);
			payload_chunk.SetCardinality(0);

			//! Create global and local state for the hashtable
			auto global_source_state = radix_table_p->GetGlobalSourceState(context);
			auto local_source_state = radix_table_p->GetLocalSourceState(temp_exec_context);

			//! Retrieve the stored data from the hashtable
			while (true) {
				output_chunk.Reset();
				radix_table_p->GetData(temp_exec_context, output_chunk, *distinct_state.radix_states[table_idx],
				                       *global_source_state, *local_source_state);
				if (output_chunk.size() == 0) {
					break;
				}

				// We dont need to resolve the filter, we already did this in Sink
				idx_t payload_cnt = aggregate.children.size();
				for (idx_t i = 0; i < payload_cnt; i++) {
					payload_chunk.data[i].Reference(output_chunk.data[i]);
				}
				payload_chunk.SetCardinality(output_chunk);
#ifdef DEBUG
				gstate.state.counts[i] += payload_chunk.size();
#endif

				auto start_of_input = payload_cnt ? &payload_chunk.data[0] : nullptr;
				//! Update the aggregate state
				AggregateInputData aggr_input_data(aggregate.bind_info.get(), Allocator::DefaultAllocator());
				aggregate.function.simple_update(start_of_input, aggr_input_data, payload_cnt,
				                                 gstate.state.aggregates[i].get(), payload_chunk.size());
			}
		}
		D_ASSERT(!gstate.finished);
		gstate.finished = true;
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		AggregateDistinct();
		event->FinishTask();
		return TaskExecutionResult::TASK_FINISHED;
	}

private:
	shared_ptr<Event> event;
	UngroupedAggregateGlobalState &gstate;
	ClientContext &context;
	const PhysicalUngroupedAggregate &op;
};

// TODO: Create tasks and run these in parallel instead of doing this all in Schedule, single threaded
class UngroupedDistinctAggregateFinalizeEvent : public BasePipelineEvent {
public:
	UngroupedDistinctAggregateFinalizeEvent(const PhysicalUngroupedAggregate &op_p,
	                                        UngroupedAggregateGlobalState &gstate_p, Pipeline &pipeline_p,
	                                        ClientContext &context)
	    : BasePipelineEvent(pipeline_p), op(op_p), gstate(gstate_p), context(context) {
	}
	const PhysicalUngroupedAggregate &op;
	UngroupedAggregateGlobalState &gstate;
	ClientContext &context;

public:
	void Schedule() override {
		vector<unique_ptr<Task>> tasks;
		tasks.push_back(make_unique<UngroupedDistinctAggregateFinalizeTask>(pipeline->executor, shared_from_this(),
		                                                                    gstate, context, op));
		D_ASSERT(!tasks.empty());
		SetTasks(std::move(tasks));
	}
};

class UngroupedDistinctCombineFinalizeEvent : public BasePipelineEvent {
public:
	UngroupedDistinctCombineFinalizeEvent(const PhysicalUngroupedAggregate &op_p,
	                                      UngroupedAggregateGlobalState &gstate_p, Pipeline &pipeline_p,
	                                      ClientContext &client)
	    : BasePipelineEvent(pipeline_p), op(op_p), gstate(gstate_p), client(client) {
	}

	const PhysicalUngroupedAggregate &op;
	UngroupedAggregateGlobalState &gstate;
	ClientContext &client;

public:
	void Schedule() override {
		auto &distinct_state = *gstate.distinct_state;
		auto &distinct_data = *op.distinct_data;
		vector<unique_ptr<Task>> tasks;
		for (idx_t table_idx = 0; table_idx < distinct_data.radix_tables.size(); table_idx++) {
			distinct_data.radix_tables[table_idx]->ScheduleTasks(pipeline->executor, shared_from_this(),
			                                                     *distinct_state.radix_states[table_idx], tasks);
		}
		D_ASSERT(!tasks.empty());
		SetTasks(std::move(tasks));
	}

	void FinishEvent() override {
		//! Now that all tables are combined, it's time to do the distinct aggregations
		auto new_event = make_shared<UngroupedDistinctAggregateFinalizeEvent>(op, gstate, *pipeline, client);
		this->InsertEvent(std::move(new_event));
	}
};

SinkFinalizeType PhysicalUngroupedAggregate::FinalizeDistinct(Pipeline &pipeline, Event &event, ClientContext &context,
                                                              GlobalSinkState &gstate_p) const {
	auto &gstate = (UngroupedAggregateGlobalState &)gstate_p;
	D_ASSERT(distinct_data);
	auto &distinct_state = *gstate.distinct_state;

	bool any_partitioned = false;
	for (idx_t table_idx = 0; table_idx < distinct_data->radix_tables.size(); table_idx++) {
		auto &radix_table_p = distinct_data->radix_tables[table_idx];
		auto &radix_state = *distinct_state.radix_states[table_idx];
		bool partitioned = radix_table_p->Finalize(context, radix_state);
		if (partitioned) {
			any_partitioned = true;
		}
	}
	if (any_partitioned) {
		auto new_event = make_shared<UngroupedDistinctCombineFinalizeEvent>(*this, gstate, pipeline, context);
		event.InsertEvent(std::move(new_event));
	} else {
		//! Hashtables aren't partitioned, they dont need to be joined first
		//! So we can compute the aggregate already
		auto new_event = make_shared<UngroupedDistinctAggregateFinalizeEvent>(*this, gstate, pipeline, context);
		event.InsertEvent(std::move(new_event));
	}
	return SinkFinalizeType::READY;
}

SinkFinalizeType PhysicalUngroupedAggregate::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                      GlobalSinkState &gstate_p) const {
	auto &gstate = (UngroupedAggregateGlobalState &)gstate_p;

	if (distinct_data) {
		return FinalizeDistinct(pipeline, event, context, gstate_p);
	}

	D_ASSERT(!gstate.finished);
	gstate.finished = true;
	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class UngroupedAggregateState : public GlobalSourceState {
public:
	UngroupedAggregateState() : finished(false) {
	}

	bool finished;
};

unique_ptr<GlobalSourceState> PhysicalUngroupedAggregate::GetGlobalSourceState(ClientContext &context) const {
	return make_unique<UngroupedAggregateState>();
}

void VerifyNullHandling(DataChunk &chunk, AggregateState &state, const vector<unique_ptr<Expression>> &aggregates) {
#ifdef DEBUG
	for (idx_t aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
		auto &aggr = (BoundAggregateExpression &)*aggregates[aggr_idx];
		if (state.counts[aggr_idx] == 0 && aggr.function.null_handling == FunctionNullHandling::DEFAULT_NULL_HANDLING) {
			// Default is when 0 values go in, NULL comes out
			UnifiedVectorFormat vdata;
			chunk.data[aggr_idx].ToUnifiedFormat(1, vdata);
			D_ASSERT(!vdata.validity.RowIsValid(vdata.sel->get_index(0)));
		}
	}
#endif
}

void PhysicalUngroupedAggregate::GetData(ExecutionContext &context, DataChunk &chunk, GlobalSourceState &gstate_p,
                                         LocalSourceState &lstate) const {
	auto &gstate = (UngroupedAggregateGlobalState &)*sink_state;
	auto &state = (UngroupedAggregateState &)gstate_p;
	D_ASSERT(gstate.finished);
	if (state.finished) {
		return;
	}

	// initialize the result chunk with the aggregate values
	chunk.SetCardinality(1);
	for (idx_t aggr_idx = 0; aggr_idx < aggregates.size(); aggr_idx++) {
		auto &aggregate = (BoundAggregateExpression &)*aggregates[aggr_idx];

		Vector state_vector(Value::POINTER((uintptr_t)gstate.state.aggregates[aggr_idx].get()));
		AggregateInputData aggr_input_data(aggregate.bind_info.get(), Allocator::DefaultAllocator());
		aggregate.function.finalize(state_vector, aggr_input_data, chunk.data[aggr_idx], 1, 0);
	}
	VerifyNullHandling(chunk, gstate.state, aggregates);
	state.finished = true;
}

string PhysicalUngroupedAggregate::ParamsToString() const {
	string result;
	for (idx_t i = 0; i < aggregates.size(); i++) {
		auto &aggregate = (BoundAggregateExpression &)*aggregates[i];
		if (i > 0) {
			result += "\n";
		}
		result += aggregates[i]->GetName();
		if (aggregate.filter) {
			result += " Filter: " + aggregate.filter->GetName();
		}
	}
	return result;
}

} // namespace duckdb





















#include <algorithm>
#include <cmath>
#include <numeric>

namespace duckdb {

class WindowGlobalHashGroup {
public:
	using GlobalSortStatePtr = unique_ptr<GlobalSortState>;
	using LocalSortStatePtr = unique_ptr<LocalSortState>;
	using Orders = vector<BoundOrderByNode>;
	using Types = vector<LogicalType>;

	WindowGlobalHashGroup(BufferManager &buffer_manager, const Orders &partitions, const Orders &orders,
	                      const Types &payload_types, bool external)
	    : count(0) {

		RowLayout payload_layout;
		payload_layout.Initialize(payload_types);
		global_sort = make_unique<GlobalSortState>(buffer_manager, orders, payload_layout);
		global_sort->external = external;

		partition_layout = global_sort->sort_layout.GetPrefixComparisonLayout(partitions.size());
	}

	void ComputeMasks(ValidityMask &partition_mask, ValidityMask &order_mask);

	GlobalSortStatePtr global_sort;
	atomic<idx_t> count;

	// Mask computation
	SortLayout partition_layout;
};

void WindowGlobalHashGroup::ComputeMasks(ValidityMask &partition_mask, ValidityMask &order_mask) {
	D_ASSERT(count > 0);

	//	Set up a comparator for the partition subset
	const auto partition_size = partition_layout.comparison_size;

	SBIterator prev(*global_sort, ExpressionType::COMPARE_LESSTHAN);
	SBIterator curr(*global_sort, ExpressionType::COMPARE_LESSTHAN);

	partition_mask.SetValidUnsafe(0);
	order_mask.SetValidUnsafe(0);
	for (++curr; curr.GetIndex() < count; ++curr) {
		//	Compare the partition subset first because if that differs, then so does the full ordering
		int part_cmp = 0;
		if (partition_layout.all_constant) {
			part_cmp = FastMemcmp(prev.entry_ptr, curr.entry_ptr, partition_size);
		} else {
			part_cmp = Comparators::CompareTuple(prev.scan, curr.scan, prev.entry_ptr, curr.entry_ptr, partition_layout,
			                                     prev.external);
		}

		if (part_cmp) {
			partition_mask.SetValidUnsafe(curr.GetIndex());
			order_mask.SetValidUnsafe(curr.GetIndex());
		} else if (prev.Compare(curr)) {
			order_mask.SetValidUnsafe(curr.GetIndex());
		}
		++prev;
	}
}

//	Global sink state
class WindowGlobalSinkState : public GlobalSinkState {
public:
	using HashGroupPtr = unique_ptr<WindowGlobalHashGroup>;
	using Orders = vector<BoundOrderByNode>;
	using Types = vector<LogicalType>;

	using GroupingPartition = unique_ptr<PartitionedColumnData>;
	using GroupingAppend = unique_ptr<PartitionedColumnDataAppendState>;

	WindowGlobalSinkState(const PhysicalWindow &op_p, ClientContext &context)
	    : op(op_p), context(context), buffer_manager(BufferManager::GetBufferManager(context)),
	      allocator(Allocator::Get(context)), payload_types(op.children[0]->types), memory_per_thread(0), count(0),
	      mode(DBConfig::GetConfig(context).options.window_mode) {

		D_ASSERT(op.select_list[0]->GetExpressionClass() == ExpressionClass::BOUND_WINDOW);
		auto wexpr = reinterpret_cast<BoundWindowExpression *>(op.select_list[0].get());

		// we sort by both 1) partition by expression list and 2) order by expressions
		const auto partition_cols = wexpr->partitions.size();
		for (idx_t prt_idx = 0; prt_idx < partition_cols; prt_idx++) {
			auto &pexpr = wexpr->partitions[prt_idx];

			if (wexpr->partitions_stats.empty() || !wexpr->partitions_stats[prt_idx]) {
				orders.emplace_back(OrderType::ASCENDING, OrderByNullType::NULLS_FIRST, pexpr->Copy(), nullptr);
			} else {
				orders.emplace_back(OrderType::ASCENDING, OrderByNullType::NULLS_FIRST, pexpr->Copy(),
				                    wexpr->partitions_stats[prt_idx]->Copy());
			}
			partitions.emplace_back(orders.back().Copy());
		}

		for (const auto &order : wexpr->orders) {
			orders.emplace_back(order.Copy());
		}

		memory_per_thread = op.GetMaxThreadMemory(context);
		external = ClientConfig::GetConfig(context).force_external;

		if (!orders.empty()) {
			grouping_types = payload_types;
			grouping_types.push_back(LogicalType::HASH);

			ResizeGroupingData(op.estimated_cardinality);
		}
	}

	void UpdateLocalPartition(GroupingPartition &local_partition, GroupingAppend &local_append);
	void CombineLocalPartition(GroupingPartition &local_partition, GroupingAppend &local_append);

	void BuildSortState(ColumnDataCollection &group_data, WindowGlobalHashGroup &global_sort);

	const PhysicalWindow &op;
	ClientContext &context;
	BufferManager &buffer_manager;
	Allocator &allocator;
	mutex lock;

	// OVER(PARTITION BY...) (hash grouping)
	unique_ptr<RadixPartitionedColumnData> grouping_data;
	//! Payload plus hash column
	Types grouping_types;

	// OVER(...) (sorting)
	Orders partitions;
	Orders orders;
	const Types payload_types;
	vector<HashGroupPtr> hash_groups;
	bool external;

	// OVER() (no sorting)
	unique_ptr<RowDataCollection> rows;
	unique_ptr<RowDataCollection> strings;

	// Threading
	idx_t memory_per_thread;
	atomic<idx_t> count;
	WindowAggregationMode mode;

private:
	void ResizeGroupingData(idx_t cardinality);
	void SyncLocalPartition(GroupingPartition &local_partition, GroupingAppend &local_append);
};

void WindowGlobalSinkState::ResizeGroupingData(idx_t cardinality) {
	//	Is the average partition size too large?
	const idx_t partition_size = STANDARD_ROW_GROUPS_SIZE;
	const auto bits = grouping_data ? grouping_data->GetRadixBits() : 0;
	auto new_bits = bits ? bits : 4;
	while (new_bits < 10 && (cardinality / RadixPartitioning::NumberOfPartitions(new_bits)) > partition_size) {
		++new_bits;
	}

	// Repartition the grouping data
	if (new_bits != bits) {
		const auto hash_col_idx = payload_types.size();
		auto new_grouping_data =
		    make_unique<RadixPartitionedColumnData>(context, grouping_types, new_bits, hash_col_idx);

		// We have to append to a shared copy for some reason
		if (grouping_data) {
			auto new_shared = new_grouping_data->CreateShared();
			PartitionedColumnDataAppendState shared_append;
			new_shared->InitializeAppendState(shared_append);

			auto &partitions = grouping_data->GetPartitions();
			for (auto &partition : partitions) {
				ColumnDataScanState scanner;
				partition->InitializeScan(scanner);

				DataChunk scan_chunk;
				partition->InitializeScanChunk(scan_chunk);
				for (scan_chunk.Reset(); partition->Scan(scanner, scan_chunk); scan_chunk.Reset()) {
					new_shared->Append(shared_append, scan_chunk);
				}
			}
			new_shared->FlushAppendState(shared_append);
			new_grouping_data->Combine(*new_shared);
		}

		grouping_data = std::move(new_grouping_data);
	}
}

void WindowGlobalSinkState::SyncLocalPartition(GroupingPartition &local_partition, GroupingAppend &local_append) {
	// We are done if the local_partition is right sized.
	auto local_radix = (RadixPartitionedColumnData *)local_partition.get();
	if (local_radix->GetRadixBits() == grouping_data->GetRadixBits()) {
		return;
	}

	// If the local partition is now too small, flush it and reallocate
	auto new_partition = grouping_data->CreateShared();
	auto new_append = make_unique<PartitionedColumnDataAppendState>();
	new_partition->InitializeAppendState(*new_append);

	local_partition->FlushAppendState(*local_append);
	auto &local_groups = local_partition->GetPartitions();
	for (auto &local_group : local_groups) {
		ColumnDataScanState scanner;
		local_group->InitializeScan(scanner);

		DataChunk scan_chunk;
		local_group->InitializeScanChunk(scan_chunk);
		for (scan_chunk.Reset(); local_group->Scan(scanner, scan_chunk); scan_chunk.Reset()) {
			new_partition->Append(*new_append, scan_chunk);
		}
	}

	// The append state has stale pointers to the old local partition, so nuke it from orbit.
	new_partition->FlushAppendState(*new_append);

	local_partition = std::move(new_partition);
	local_append = make_unique<PartitionedColumnDataAppendState>();
	local_partition->InitializeAppendState(*local_append);
}

void WindowGlobalSinkState::UpdateLocalPartition(GroupingPartition &local_partition, GroupingAppend &local_append) {
	// Make sure grouping_data doesn't change under us.
	lock_guard<mutex> guard(lock);

	if (!local_partition) {
		local_partition = grouping_data->CreateShared();
		local_append = make_unique<PartitionedColumnDataAppendState>();
		local_partition->InitializeAppendState(*local_append);
		return;
	}

	// 	Grow the groups if they are too big
	ResizeGroupingData(count);

	//	Sync local partition to have the same bit count
	SyncLocalPartition(local_partition, local_append);
}

void WindowGlobalSinkState::CombineLocalPartition(GroupingPartition &local_partition, GroupingAppend &local_append) {
	if (!local_partition) {
		return;
	}
	local_partition->FlushAppendState(*local_append);

	// Make sure grouping_data doesn't change under us.
	// Combine has an internal mutex, so this is single-threaded anyway.
	lock_guard<mutex> guard(lock);
	SyncLocalPartition(local_partition, local_append);
	grouping_data->Combine(*local_partition);
}

void WindowGlobalSinkState::BuildSortState(ColumnDataCollection &group_data, WindowGlobalHashGroup &hash_group) {
	auto &global_sort = *hash_group.global_sort;

	//	 Set up the sort expression computation.
	vector<LogicalType> sort_types;
	ExpressionExecutor executor(context);
	for (auto &order : orders) {
		auto &oexpr = order.expression;
		sort_types.emplace_back(oexpr->return_type);
		executor.AddExpression(*oexpr);
	}
	DataChunk sort_chunk;
	sort_chunk.Initialize(allocator, sort_types);

	// Copy the data from the group into the sort code.
	LocalSortState local_sort;
	local_sort.Initialize(global_sort, global_sort.buffer_manager);

	//	Strip hash column
	DataChunk payload_chunk;
	payload_chunk.Initialize(allocator, payload_types);

	vector<column_t> column_ids;
	column_ids.reserve(payload_types.size());
	for (column_t i = 0; i < payload_types.size(); ++i) {
		column_ids.emplace_back(i);
	}
	ColumnDataConsumer scanner(group_data, column_ids);
	ColumnDataConsumerScanState chunk_state;
	chunk_state.current_chunk_state.properties = ColumnDataScanProperties::ALLOW_ZERO_COPY;
	scanner.InitializeScan();
	for (auto chunk_idx = scanner.ChunkCount(); chunk_idx-- > 0;) {
		if (!scanner.AssignChunk(chunk_state)) {
			break;
		}
		scanner.ScanChunk(chunk_state, payload_chunk);

		sort_chunk.Reset();
		executor.Execute(payload_chunk, sort_chunk);

		local_sort.SinkChunk(sort_chunk, payload_chunk);
		if (local_sort.SizeInBytes() > memory_per_thread) {
			local_sort.Sort(global_sort, true);
		}
		scanner.FinishChunk(chunk_state);
	}

	global_sort.AddLocalState(local_sort);

	hash_group.count += group_data.Count();
}

//	Per-thread sink state
class WindowLocalSinkState : public LocalSinkState {
public:
	WindowLocalSinkState(ClientContext &context, const PhysicalWindow &op_p)
	    : op(op_p), allocator(Allocator::Get(context)), executor(context) {
		D_ASSERT(op.select_list[0]->GetExpressionClass() == ExpressionClass::BOUND_WINDOW);
		auto wexpr = reinterpret_cast<BoundWindowExpression *>(op.select_list[0].get());

		vector<LogicalType> group_types;
		for (idx_t prt_idx = 0; prt_idx < wexpr->partitions.size(); prt_idx++) {
			auto &pexpr = wexpr->partitions[prt_idx];
			group_types.push_back(pexpr->return_type);
			executor.AddExpression(*pexpr);
		}
		sort_cols = wexpr->orders.size() + group_types.size();

		if (sort_cols) {
			if (!group_types.empty()) {
				// OVER(PARTITION BY...)
				group_chunk.Initialize(allocator, group_types);
			}
			// OVER(...)
			auto payload_types = op.children[0]->types;
			payload_types.emplace_back(LogicalType::HASH);
			payload_chunk.Initialize(allocator, payload_types);
		} else {
			// OVER()
			payload_layout.Initialize(op.children[0]->types);
		}
	}

	// Global state
	const PhysicalWindow &op;
	Allocator &allocator;

	// OVER(PARTITION BY...) (hash grouping)
	ExpressionExecutor executor;
	DataChunk group_chunk;
	DataChunk payload_chunk;
	unique_ptr<PartitionedColumnData> local_partition;
	unique_ptr<PartitionedColumnDataAppendState> local_append;

	// OVER(...) (sorting)
	size_t sort_cols;

	// OVER() (no sorting)
	RowLayout payload_layout;
	unique_ptr<RowDataCollection> rows;
	unique_ptr<RowDataCollection> strings;

	//! Compute the hash values
	void Hash(DataChunk &input_chunk, Vector &hash_vector);
	//! Sink an input chunk
	void Sink(DataChunk &input_chunk, WindowGlobalSinkState &gstate);
	//! Merge the state into the global state.
	void Combine(WindowGlobalSinkState &gstate);
};

void WindowLocalSinkState::Hash(DataChunk &input_chunk, Vector &hash_vector) {
	const auto count = input_chunk.size();
	if (group_chunk.ColumnCount() > 0) {
		// OVER(PARTITION BY...) (hash grouping)
		group_chunk.Reset();
		executor.Execute(input_chunk, group_chunk);
		VectorOperations::Hash(group_chunk.data[0], hash_vector, count);
		for (idx_t prt_idx = 1; prt_idx < group_chunk.ColumnCount(); ++prt_idx) {
			VectorOperations::CombineHash(hash_vector, group_chunk.data[prt_idx], count);
		}
	} else {
		// OVER(...) (sorting)
		// Single partition => single hash value
		hash_vector.SetVectorType(VectorType::CONSTANT_VECTOR);
		auto hashes = ConstantVector::GetData<hash_t>(hash_vector);
		hashes[0] = 0;
	}
}

void WindowLocalSinkState::Sink(DataChunk &input_chunk, WindowGlobalSinkState &gstate) {
	gstate.count += input_chunk.size();

	// OVER()
	if (sort_cols == 0) {
		//	No sorts, so build paged row chunks
		if (!rows) {
			const auto entry_size = payload_layout.GetRowWidth();
			const auto capacity = MaxValue<idx_t>(STANDARD_VECTOR_SIZE, (Storage::BLOCK_SIZE / entry_size) + 1);
			rows = make_unique<RowDataCollection>(gstate.buffer_manager, capacity, entry_size);
			strings = make_unique<RowDataCollection>(gstate.buffer_manager, (idx_t)Storage::BLOCK_SIZE, 1, true);
		}
		const auto row_count = input_chunk.size();
		const auto row_sel = FlatVector::IncrementalSelectionVector();
		Vector addresses(LogicalType::POINTER);
		auto key_locations = FlatVector::GetData<data_ptr_t>(addresses);
		const auto prev_rows_blocks = rows->blocks.size();
		auto handles = rows->Build(row_count, key_locations, nullptr, row_sel);
		auto input_data = input_chunk.ToUnifiedFormat();
		RowOperations::Scatter(input_chunk, input_data.get(), payload_layout, addresses, *strings, *row_sel, row_count);
		// Mark that row blocks contain pointers (heap blocks are pinned)
		if (!payload_layout.AllConstant()) {
			D_ASSERT(strings->keep_pinned);
			for (size_t i = prev_rows_blocks; i < rows->blocks.size(); ++i) {
				rows->blocks[i]->block->SetSwizzling("WindowLocalSinkState::Sink");
			}
		}
		return;
	}

	// OVER(...)
	gstate.UpdateLocalPartition(local_partition, local_append);

	payload_chunk.Reset();
	auto &hash_vector = payload_chunk.data.back();
	Hash(input_chunk, hash_vector);
	for (idx_t col_idx = 0; col_idx < input_chunk.ColumnCount(); ++col_idx) {
		payload_chunk.data[col_idx].Reference(input_chunk.data[col_idx]);
	}
	payload_chunk.SetCardinality(input_chunk);

	local_partition->Append(*local_append, payload_chunk);
}

void WindowLocalSinkState::Combine(WindowGlobalSinkState &gstate) {
	// OVER()
	if (sort_cols == 0) {
		// Only one partition again, so need a global lock.
		lock_guard<mutex> glock(gstate.lock);
		if (gstate.rows) {
			if (rows) {
				gstate.rows->Merge(*rows);
				gstate.strings->Merge(*strings);
				rows.reset();
				strings.reset();
			}
		} else {
			gstate.rows = std::move(rows);
			gstate.strings = std::move(strings);
		}
		return;
	}

	// OVER(...)
	gstate.CombineLocalPartition(local_partition, local_append);
}

// this implements a sorted window functions variant
PhysicalWindow::PhysicalWindow(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list_p,
                               idx_t estimated_cardinality, PhysicalOperatorType type)
    : PhysicalOperator(type, std::move(types), estimated_cardinality), select_list(std::move(select_list_p)) {
	is_order_dependent = false;
	for (auto &expr : select_list) {
		D_ASSERT(expr->expression_class == ExpressionClass::BOUND_WINDOW);
		auto &bound_window = (BoundWindowExpression &)*expr;
		if (bound_window.partitions.empty() && bound_window.orders.empty()) {
			is_order_dependent = true;
		}
	}
}

static idx_t FindNextStart(const ValidityMask &mask, idx_t l, const idx_t r, idx_t &n) {
	if (mask.AllValid()) {
		auto start = MinValue(l + n - 1, r);
		n -= MinValue(n, r - l);
		return start;
	}

	while (l < r) {
		//	If l is aligned with the start of a block, and the block is blank, then skip forward one block.
		idx_t entry_idx;
		idx_t shift;
		mask.GetEntryIndex(l, entry_idx, shift);

		const auto block = mask.GetValidityEntry(entry_idx);
		if (mask.NoneValid(block) && !shift) {
			l += ValidityMask::BITS_PER_VALUE;
			continue;
		}

		// Loop over the block
		for (; shift < ValidityMask::BITS_PER_VALUE && l < r; ++shift, ++l) {
			if (mask.RowIsValid(block, shift) && --n == 0) {
				return MinValue(l, r);
			}
		}
	}

	//	Didn't find a start so return the end of the range
	return r;
}

static idx_t FindPrevStart(const ValidityMask &mask, const idx_t l, idx_t r, idx_t &n) {
	if (mask.AllValid()) {
		auto start = (r <= l + n) ? l : r - n;
		n -= r - start;
		return start;
	}

	while (l < r) {
		// If r is aligned with the start of a block, and the previous block is blank,
		// then skip backwards one block.
		idx_t entry_idx;
		idx_t shift;
		mask.GetEntryIndex(r - 1, entry_idx, shift);

		const auto block = mask.GetValidityEntry(entry_idx);
		if (mask.NoneValid(block) && (shift + 1 == ValidityMask::BITS_PER_VALUE)) {
			// r is nonzero (> l) and word aligned, so this will not underflow.
			r -= ValidityMask::BITS_PER_VALUE;
			continue;
		}

		// Loop backwards over the block
		// shift is probing r-1 >= l >= 0
		for (++shift; shift-- > 0; --r) {
			if (mask.RowIsValid(block, shift) && --n == 0) {
				return MaxValue(l, r - 1);
			}
		}
	}

	//	Didn't find a start so return the start of the range
	return l;
}

static void PrepareInputExpressions(Expression **exprs, idx_t expr_count, ExpressionExecutor &executor,
                                    DataChunk &chunk) {
	if (expr_count == 0) {
		return;
	}

	vector<LogicalType> types;
	for (idx_t expr_idx = 0; expr_idx < expr_count; ++expr_idx) {
		types.push_back(exprs[expr_idx]->return_type);
		executor.AddExpression(*exprs[expr_idx]);
	}

	if (!types.empty()) {
		auto &allocator = executor.GetAllocator();
		chunk.Initialize(allocator, types);
	}
}

static void PrepareInputExpression(Expression *expr, ExpressionExecutor &executor, DataChunk &chunk) {
	PrepareInputExpressions(&expr, 1, executor, chunk);
}

struct WindowInputExpression {
	WindowInputExpression(Expression *expr_p, ClientContext &context)
	    : expr(expr_p), ptype(PhysicalType::INVALID), scalar(true), executor(context) {
		if (expr) {
			PrepareInputExpression(expr, executor, chunk);
			ptype = expr->return_type.InternalType();
			scalar = expr->IsScalar();
		}
	}

	void Execute(DataChunk &input_chunk) {
		if (expr) {
			chunk.Reset();
			executor.Execute(input_chunk, chunk);
			chunk.Verify();
		}
	}

	template <typename T>
	inline T GetCell(idx_t i) const {
		D_ASSERT(!chunk.data.empty());
		const auto data = FlatVector::GetData<T>(chunk.data[0]);
		return data[scalar ? 0 : i];
	}

	inline bool CellIsNull(idx_t i) const {
		D_ASSERT(!chunk.data.empty());
		if (chunk.data[0].GetVectorType() == VectorType::CONSTANT_VECTOR) {
			return ConstantVector::IsNull(chunk.data[0]);
		}
		return FlatVector::IsNull(chunk.data[0], i);
	}

	inline void CopyCell(Vector &target, idx_t target_offset) const {
		D_ASSERT(!chunk.data.empty());
		auto &source = chunk.data[0];
		auto source_offset = scalar ? 0 : target_offset;
		VectorOperations::Copy(source, target, source_offset + 1, source_offset, target_offset);
	}

	Expression *expr;
	PhysicalType ptype;
	bool scalar;
	ExpressionExecutor executor;
	DataChunk chunk;
};

struct WindowInputColumn {
	WindowInputColumn(Expression *expr_p, ClientContext &context, idx_t capacity_p)
	    : input_expr(expr_p, context), count(0), capacity(capacity_p) {
		if (input_expr.expr) {
			target = make_unique<Vector>(input_expr.chunk.data[0].GetType(), capacity);
		}
	}

	void Append(DataChunk &input_chunk) {
		if (input_expr.expr && (!input_expr.scalar || !count)) {
			input_expr.Execute(input_chunk);
			auto &source = input_expr.chunk.data[0];
			const auto source_count = input_expr.chunk.size();
			D_ASSERT(count + source_count <= capacity);
			VectorOperations::Copy(source, *target, source_count, 0, count);
			count += source_count;
		}
	}

	inline bool CellIsNull(idx_t i) {
		D_ASSERT(target);
		D_ASSERT(i < count);
		return FlatVector::IsNull(*target, input_expr.scalar ? 0 : i);
	}

	template <typename T>
	inline T GetCell(idx_t i) {
		D_ASSERT(target);
		D_ASSERT(i < count);
		const auto data = FlatVector::GetData<T>(*target);
		return data[input_expr.scalar ? 0 : i];
	}

	WindowInputExpression input_expr;

private:
	unique_ptr<Vector> target;
	idx_t count;
	idx_t capacity;
};

static inline bool BoundaryNeedsPeer(const WindowBoundary &boundary) {
	switch (boundary) {
	case WindowBoundary::CURRENT_ROW_RANGE:
	case WindowBoundary::EXPR_PRECEDING_RANGE:
	case WindowBoundary::EXPR_FOLLOWING_RANGE:
		return true;
	default:
		return false;
	}
}

struct WindowBoundariesState {
	static inline bool IsScalar(const unique_ptr<Expression> &expr) {
		return expr ? expr->IsScalar() : true;
	}

	WindowBoundariesState(BoundWindowExpression *wexpr, const idx_t input_size)
	    : type(wexpr->type), input_size(input_size), start_boundary(wexpr->start), end_boundary(wexpr->end),
	      partition_count(wexpr->partitions.size()), order_count(wexpr->orders.size()),
	      range_sense(wexpr->orders.empty() ? OrderType::INVALID : wexpr->orders[0].type),
	      has_preceding_range(wexpr->start == WindowBoundary::EXPR_PRECEDING_RANGE ||
	                          wexpr->end == WindowBoundary::EXPR_PRECEDING_RANGE),
	      has_following_range(wexpr->start == WindowBoundary::EXPR_FOLLOWING_RANGE ||
	                          wexpr->end == WindowBoundary::EXPR_FOLLOWING_RANGE),
	      needs_peer(BoundaryNeedsPeer(wexpr->end) || wexpr->type == ExpressionType::WINDOW_CUME_DIST) {
	}

	void Update(const idx_t row_idx, WindowInputColumn &range_collection, const idx_t source_offset,
	            WindowInputExpression &boundary_start, WindowInputExpression &boundary_end,
	            const ValidityMask &partition_mask, const ValidityMask &order_mask);

	// Cached lookups
	const ExpressionType type;
	const idx_t input_size;
	const WindowBoundary start_boundary;
	const WindowBoundary end_boundary;
	const size_t partition_count;
	const size_t order_count;
	const OrderType range_sense;
	const bool has_preceding_range;
	const bool has_following_range;
	const bool needs_peer;

	idx_t partition_start = 0;
	idx_t partition_end = 0;
	idx_t peer_start = 0;
	idx_t peer_end = 0;
	idx_t valid_start = 0;
	idx_t valid_end = 0;
	int64_t window_start = -1;
	int64_t window_end = -1;
	bool is_same_partition = false;
	bool is_peer = false;
};

static bool WindowNeedsRank(BoundWindowExpression *wexpr) {
	return wexpr->type == ExpressionType::WINDOW_PERCENT_RANK || wexpr->type == ExpressionType::WINDOW_RANK ||
	       wexpr->type == ExpressionType::WINDOW_RANK_DENSE || wexpr->type == ExpressionType::WINDOW_CUME_DIST;
}

template <typename T>
static T GetCell(DataChunk &chunk, idx_t column, idx_t index) {
	D_ASSERT(chunk.ColumnCount() > column);
	auto &source = chunk.data[column];
	const auto data = FlatVector::GetData<T>(source);
	return data[index];
}

static bool CellIsNull(DataChunk &chunk, idx_t column, idx_t index) {
	D_ASSERT(chunk.ColumnCount() > column);
	auto &source = chunk.data[column];
	return FlatVector::IsNull(source, index);
}

static void CopyCell(DataChunk &chunk, idx_t column, idx_t index, Vector &target, idx_t target_offset) {
	D_ASSERT(chunk.ColumnCount() > column);
	auto &source = chunk.data[column];
	VectorOperations::Copy(source, target, index + 1, index, target_offset);
}

template <typename T>
struct WindowColumnIterator {
	using iterator = WindowColumnIterator<T>;
	using iterator_category = std::forward_iterator_tag;
	using difference_type = std::ptrdiff_t;
	using value_type = T;
	using reference = T;
	using pointer = idx_t;

	explicit WindowColumnIterator(WindowInputColumn &coll_p, pointer pos_p = 0) : coll(&coll_p), pos(pos_p) {
	}

	inline reference operator*() const {
		return coll->GetCell<T>(pos);
	}
	inline explicit operator pointer() const {
		return pos;
	}

	inline iterator &operator++() {
		++pos;
		return *this;
	}
	inline iterator operator++(int) {
		auto result = *this;
		++(*this);
		return result;
	}

	friend inline bool operator==(const iterator &a, const iterator &b) {
		return a.pos == b.pos;
	}
	friend inline bool operator!=(const iterator &a, const iterator &b) {
		return a.pos != b.pos;
	}

private:
	WindowInputColumn *coll;
	pointer pos;
};

template <typename T, typename OP>
struct OperationCompare : public std::function<bool(T, T)> {
	inline bool operator()(const T &lhs, const T &val) const {
		return OP::template Operation(lhs, val);
	}
};

template <typename T, typename OP, bool FROM>
static idx_t FindTypedRangeBound(WindowInputColumn &over, const idx_t order_begin, const idx_t order_end,
                                 WindowInputExpression &boundary, const idx_t boundary_row) {
	D_ASSERT(!boundary.CellIsNull(boundary_row));
	const auto val = boundary.GetCell<T>(boundary_row);

	OperationCompare<T, OP> comp;
	WindowColumnIterator<T> begin(over, order_begin);
	WindowColumnIterator<T> end(over, order_end);
	if (FROM) {
		return idx_t(std::lower_bound(begin, end, val, comp));
	} else {
		return idx_t(std::upper_bound(begin, end, val, comp));
	}
}

template <typename OP, bool FROM>
static idx_t FindRangeBound(WindowInputColumn &over, const idx_t order_begin, const idx_t order_end,
                            WindowInputExpression &boundary, const idx_t expr_idx) {
	D_ASSERT(boundary.chunk.ColumnCount() == 1);
	D_ASSERT(boundary.chunk.data[0].GetType().InternalType() == over.input_expr.ptype);

	switch (over.input_expr.ptype) {
	case PhysicalType::INT8:
		return FindTypedRangeBound<int8_t, OP, FROM>(over, order_begin, order_end, boundary, expr_idx);
	case PhysicalType::INT16:
		return FindTypedRangeBound<int16_t, OP, FROM>(over, order_begin, order_end, boundary, expr_idx);
	case PhysicalType::INT32:
		return FindTypedRangeBound<int32_t, OP, FROM>(over, order_begin, order_end, boundary, expr_idx);
	case PhysicalType::INT64:
		return FindTypedRangeBound<int64_t, OP, FROM>(over, order_begin, order_end, boundary, expr_idx);
	case PhysicalType::UINT8:
		return FindTypedRangeBound<uint8_t, OP, FROM>(over, order_begin, order_end, boundary, expr_idx);
	case PhysicalType::UINT16:
		return FindTypedRangeBound<uint16_t, OP, FROM>(over, order_begin, order_end, boundary, expr_idx);
	case PhysicalType::UINT32:
		return FindTypedRangeBound<uint32_t, OP, FROM>(over, order_begin, order_end, boundary, expr_idx);
	case PhysicalType::UINT64:
		return FindTypedRangeBound<uint64_t, OP, FROM>(over, order_begin, order_end, boundary, expr_idx);
	case PhysicalType::INT128:
		return FindTypedRangeBound<hugeint_t, OP, FROM>(over, order_begin, order_end, boundary, expr_idx);
	case PhysicalType::FLOAT:
		return FindTypedRangeBound<float, OP, FROM>(over, order_begin, order_end, boundary, expr_idx);
	case PhysicalType::DOUBLE:
		return FindTypedRangeBound<double, OP, FROM>(over, order_begin, order_end, boundary, expr_idx);
	case PhysicalType::INTERVAL:
		return FindTypedRangeBound<interval_t, OP, FROM>(over, order_begin, order_end, boundary, expr_idx);
	default:
		throw InternalException("Unsupported column type for RANGE");
	}
}

template <bool FROM>
static idx_t FindOrderedRangeBound(WindowInputColumn &over, const OrderType range_sense, const idx_t order_begin,
                                   const idx_t order_end, WindowInputExpression &boundary, const idx_t expr_idx) {
	switch (range_sense) {
	case OrderType::ASCENDING:
		return FindRangeBound<LessThan, FROM>(over, order_begin, order_end, boundary, expr_idx);
	case OrderType::DESCENDING:
		return FindRangeBound<GreaterThan, FROM>(over, order_begin, order_end, boundary, expr_idx);
	default:
		throw InternalException("Unsupported ORDER BY sense for RANGE");
	}
}

void WindowBoundariesState::Update(const idx_t row_idx, WindowInputColumn &range_collection, const idx_t expr_idx,
                                   WindowInputExpression &boundary_start, WindowInputExpression &boundary_end,
                                   const ValidityMask &partition_mask, const ValidityMask &order_mask) {

	auto &bounds = *this;
	if (bounds.partition_count + bounds.order_count > 0) {

		// determine partition and peer group boundaries to ultimately figure out window size
		bounds.is_same_partition = !partition_mask.RowIsValidUnsafe(row_idx);
		bounds.is_peer = !order_mask.RowIsValidUnsafe(row_idx);

		// when the partition changes, recompute the boundaries
		if (!bounds.is_same_partition) {
			bounds.partition_start = row_idx;
			bounds.peer_start = row_idx;

			// find end of partition
			bounds.partition_end = bounds.input_size;
			if (bounds.partition_count) {
				idx_t n = 1;
				bounds.partition_end = FindNextStart(partition_mask, bounds.partition_start + 1, bounds.input_size, n);
			}

			// Find valid ordering values for the new partition
			// so we can exclude NULLs from RANGE expression computations
			bounds.valid_start = bounds.partition_start;
			bounds.valid_end = bounds.partition_end;

			if ((bounds.valid_start < bounds.valid_end) && bounds.has_preceding_range) {
				// Exclude any leading NULLs
				if (range_collection.CellIsNull(bounds.valid_start)) {
					idx_t n = 1;
					bounds.valid_start = FindNextStart(order_mask, bounds.valid_start + 1, bounds.valid_end, n);
				}
			}

			if ((bounds.valid_start < bounds.valid_end) && bounds.has_following_range) {
				// Exclude any trailing NULLs
				if (range_collection.CellIsNull(bounds.valid_end - 1)) {
					idx_t n = 1;
					bounds.valid_end = FindPrevStart(order_mask, bounds.valid_start, bounds.valid_end, n);
				}
			}

		} else if (!bounds.is_peer) {
			bounds.peer_start = row_idx;
		}

		if (bounds.needs_peer) {
			bounds.peer_end = bounds.partition_end;
			if (bounds.order_count) {
				idx_t n = 1;
				bounds.peer_end = FindNextStart(order_mask, bounds.peer_start + 1, bounds.partition_end, n);
			}
		}

	} else {
		bounds.is_same_partition = false;
		bounds.is_peer = true;
		bounds.partition_end = bounds.input_size;
		bounds.peer_end = bounds.partition_end;
	}

	// determine window boundaries depending on the type of expression
	bounds.window_start = -1;
	bounds.window_end = -1;

	switch (bounds.start_boundary) {
	case WindowBoundary::UNBOUNDED_PRECEDING:
		bounds.window_start = bounds.partition_start;
		break;
	case WindowBoundary::CURRENT_ROW_ROWS:
		bounds.window_start = row_idx;
		break;
	case WindowBoundary::CURRENT_ROW_RANGE:
		bounds.window_start = bounds.peer_start;
		break;
	case WindowBoundary::EXPR_PRECEDING_ROWS: {
		bounds.window_start = (int64_t)row_idx - boundary_start.GetCell<int64_t>(expr_idx);
		break;
	}
	case WindowBoundary::EXPR_FOLLOWING_ROWS: {
		bounds.window_start = row_idx + boundary_start.GetCell<int64_t>(expr_idx);
		break;
	}
	case WindowBoundary::EXPR_PRECEDING_RANGE: {
		if (boundary_start.CellIsNull(expr_idx)) {
			bounds.window_start = bounds.peer_start;
		} else {
			bounds.window_start = FindOrderedRangeBound<true>(range_collection, bounds.range_sense, bounds.valid_start,
			                                                  row_idx, boundary_start, expr_idx);
		}
		break;
	}
	case WindowBoundary::EXPR_FOLLOWING_RANGE: {
		if (boundary_start.CellIsNull(expr_idx)) {
			bounds.window_start = bounds.peer_start;
		} else {
			bounds.window_start = FindOrderedRangeBound<true>(range_collection, bounds.range_sense, row_idx,
			                                                  bounds.valid_end, boundary_start, expr_idx);
		}
		break;
	}
	default:
		throw InternalException("Unsupported window start boundary");
	}

	switch (bounds.end_boundary) {
	case WindowBoundary::CURRENT_ROW_ROWS:
		bounds.window_end = row_idx + 1;
		break;
	case WindowBoundary::CURRENT_ROW_RANGE:
		bounds.window_end = bounds.peer_end;
		break;
	case WindowBoundary::UNBOUNDED_FOLLOWING:
		bounds.window_end = bounds.partition_end;
		break;
	case WindowBoundary::EXPR_PRECEDING_ROWS:
		bounds.window_end = (int64_t)row_idx - boundary_end.GetCell<int64_t>(expr_idx) + 1;
		break;
	case WindowBoundary::EXPR_FOLLOWING_ROWS:
		bounds.window_end = row_idx + boundary_end.GetCell<int64_t>(expr_idx) + 1;
		break;
	case WindowBoundary::EXPR_PRECEDING_RANGE: {
		if (boundary_end.CellIsNull(expr_idx)) {
			bounds.window_end = bounds.peer_end;
		} else {
			bounds.window_end = FindOrderedRangeBound<false>(range_collection, bounds.range_sense, bounds.valid_start,
			                                                 row_idx, boundary_end, expr_idx);
		}
		break;
	}
	case WindowBoundary::EXPR_FOLLOWING_RANGE: {
		if (boundary_end.CellIsNull(expr_idx)) {
			bounds.window_end = bounds.peer_end;
		} else {
			bounds.window_end = FindOrderedRangeBound<false>(range_collection, bounds.range_sense, row_idx,
			                                                 bounds.valid_end, boundary_end, expr_idx);
		}
		break;
	}
	default:
		throw InternalException("Unsupported window end boundary");
	}

	// clamp windows to partitions if they should exceed
	if (bounds.window_start < (int64_t)bounds.partition_start) {
		bounds.window_start = bounds.partition_start;
	}
	if (bounds.window_start > (int64_t)bounds.partition_end) {
		bounds.window_start = bounds.partition_end;
	}
	if (bounds.window_end < (int64_t)bounds.partition_start) {
		bounds.window_end = bounds.partition_start;
	}
	if (bounds.window_end > (int64_t)bounds.partition_end) {
		bounds.window_end = bounds.partition_end;
	}

	if (bounds.window_start < 0 || bounds.window_end < 0) {
		throw InternalException("Failed to compute window boundaries");
	}
}

struct WindowExecutor {
	WindowExecutor(BoundWindowExpression *wexpr, ClientContext &context, const idx_t count);

	void Sink(DataChunk &input_chunk, const idx_t input_idx, const idx_t total_count);
	void Finalize(WindowAggregationMode mode);

	void Evaluate(idx_t row_idx, DataChunk &input_chunk, Vector &result, const ValidityMask &partition_mask,
	              const ValidityMask &order_mask);

	// The function
	BoundWindowExpression *wexpr;

	// Frame management
	WindowBoundariesState bounds;
	uint64_t dense_rank = 1;
	uint64_t rank_equal = 0;
	uint64_t rank = 1;

	// Expression collections
	DataChunk payload_collection;
	ExpressionExecutor payload_executor;
	DataChunk payload_chunk;

	ExpressionExecutor filter_executor;
	ValidityMask filter_mask;
	vector<validity_t> filter_bits;
	SelectionVector filter_sel;

	// LEAD/LAG Evaluation
	WindowInputExpression leadlag_offset;
	WindowInputExpression leadlag_default;

	// evaluate boundaries if present. Parser has checked boundary types.
	WindowInputExpression boundary_start;
	WindowInputExpression boundary_end;

	// evaluate RANGE expressions, if needed
	WindowInputColumn range;

	// IGNORE NULLS
	ValidityMask ignore_nulls;

	// build a segment tree for frame-adhering aggregates
	// see http://www.vldb.org/pvldb/vol8/p1058-leis.pdf
	unique_ptr<WindowSegmentTree> segment_tree = nullptr;
};

WindowExecutor::WindowExecutor(BoundWindowExpression *wexpr, ClientContext &context, const idx_t count)
    : wexpr(wexpr), bounds(wexpr, count), payload_collection(), payload_executor(context), filter_executor(context),
      leadlag_offset(wexpr->offset_expr.get(), context), leadlag_default(wexpr->default_expr.get(), context),
      boundary_start(wexpr->start_expr.get(), context), boundary_end(wexpr->end_expr.get(), context),
      range((bounds.has_preceding_range || bounds.has_following_range) ? wexpr->orders[0].expression.get() : nullptr,
            context, count)

{
	// TODO we could evaluate those expressions in parallel

	// evaluate the FILTER clause and stuff it into a large mask for compactness and reuse
	if (wexpr->filter_expr) {
		// 	Start with all invalid and set the ones that pass
		filter_bits.resize(ValidityMask::ValidityMaskSize(count), 0);
		filter_mask.Initialize(filter_bits.data());
		filter_executor.AddExpression(*wexpr->filter_expr);
		filter_sel.Initialize(STANDARD_VECTOR_SIZE);
	}

	// TODO: child may be a scalar, don't need to materialize the whole collection then

	// evaluate inner expressions of window functions, could be more complex
	vector<Expression *> exprs;
	exprs.reserve(wexpr->children.size());
	for (auto &child : wexpr->children) {
		exprs.push_back(child.get());
	}
	PrepareInputExpressions(exprs.data(), exprs.size(), payload_executor, payload_chunk);

	auto types = payload_chunk.GetTypes();
	if (!types.empty()) {
		payload_collection.Initialize(Allocator::Get(context), types);
	}
}

void WindowExecutor::Sink(DataChunk &input_chunk, const idx_t input_idx, const idx_t total_count) {
	// Single pass over the input to produce the global data.
	// Vectorisation for the win...

	// Set up a validity mask for IGNORE NULLS
	bool check_nulls = false;
	if (wexpr->ignore_nulls) {
		switch (wexpr->type) {
		case ExpressionType::WINDOW_LEAD:
		case ExpressionType::WINDOW_LAG:
		case ExpressionType::WINDOW_FIRST_VALUE:
		case ExpressionType::WINDOW_LAST_VALUE:
		case ExpressionType::WINDOW_NTH_VALUE:
			check_nulls = true;
			break;
		default:
			break;
		}
	}

	const auto count = input_chunk.size();

	if (!wexpr->children.empty()) {
		payload_chunk.Reset();
		payload_executor.Execute(input_chunk, payload_chunk);
		payload_chunk.Verify();
		payload_collection.Append(payload_chunk, true);

		// process payload chunks while they are still piping hot
		if (check_nulls) {
			UnifiedVectorFormat vdata;
			payload_chunk.data[0].ToUnifiedFormat(count, vdata);
			if (!vdata.validity.AllValid()) {
				//	Lazily materialise the contents when we find the first NULL
				if (ignore_nulls.AllValid()) {
					ignore_nulls.Initialize(total_count);
				}
				// Write to the current position
				if (input_idx % ValidityMask::BITS_PER_VALUE == 0) {
					// If we are at the edge of an output entry, just copy the entries
					auto dst = ignore_nulls.GetData() + ignore_nulls.EntryCount(input_idx);
					auto src = vdata.validity.GetData();
					for (auto entry_count = vdata.validity.EntryCount(count); entry_count-- > 0;) {
						*dst++ = *src++;
					}
				} else {
					// If not, we have ragged data and need to copy one bit at a time.
					for (idx_t i = 0; i < count; ++i) {
						ignore_nulls.Set(input_idx + i, vdata.validity.RowIsValid(i));
					}
				}
			}
		}
	}

	if (wexpr->filter_expr) {
		const auto filtered = filter_executor.SelectExpression(input_chunk, filter_sel);
		for (idx_t f = 0; f < filtered; ++f) {
			filter_mask.SetValid(input_idx + filter_sel[f]);
		}
	}

	range.Append(input_chunk);
}

void WindowExecutor::Finalize(WindowAggregationMode mode) {
	// build a segment tree for frame-adhering aggregates
	// see http://www.vldb.org/pvldb/vol8/p1058-leis.pdf

	if (wexpr->aggregate) {
		segment_tree = make_unique<WindowSegmentTree>(*(wexpr->aggregate), wexpr->bind_info.get(), wexpr->return_type,
		                                              &payload_collection, filter_mask, mode);
	}
}

void WindowExecutor::Evaluate(idx_t row_idx, DataChunk &input_chunk, Vector &result, const ValidityMask &partition_mask,
                              const ValidityMask &order_mask) {
	// Evaluate the row-level arguments
	boundary_start.Execute(input_chunk);
	boundary_end.Execute(input_chunk);

	leadlag_offset.Execute(input_chunk);
	leadlag_default.Execute(input_chunk);

	// this is the main loop, go through all sorted rows and compute window function result
	for (idx_t output_offset = 0; output_offset < input_chunk.size(); ++output_offset, ++row_idx) {
		// special case, OVER (), aggregate over everything
		bounds.Update(row_idx, range, output_offset, boundary_start, boundary_end, partition_mask, order_mask);
		if (WindowNeedsRank(wexpr)) {
			if (!bounds.is_same_partition || row_idx == 0) { // special case for first row, need to init
				dense_rank = 1;
				rank = 1;
				rank_equal = 0;
			} else if (!bounds.is_peer) {
				dense_rank++;
				rank += rank_equal;
				rank_equal = 0;
			}
			rank_equal++;
		}

		// if no values are read for window, result is NULL
		if (bounds.window_start >= bounds.window_end) {
			FlatVector::SetNull(result, output_offset, true);
			continue;
		}

		switch (wexpr->type) {
		case ExpressionType::WINDOW_AGGREGATE: {
			segment_tree->Compute(result, output_offset, bounds.window_start, bounds.window_end);
			break;
		}
		case ExpressionType::WINDOW_ROW_NUMBER: {
			auto rdata = FlatVector::GetData<int64_t>(result);
			rdata[output_offset] = row_idx - bounds.partition_start + 1;
			break;
		}
		case ExpressionType::WINDOW_RANK_DENSE: {
			auto rdata = FlatVector::GetData<int64_t>(result);
			rdata[output_offset] = dense_rank;
			break;
		}
		case ExpressionType::WINDOW_RANK: {
			auto rdata = FlatVector::GetData<int64_t>(result);
			rdata[output_offset] = rank;
			break;
		}
		case ExpressionType::WINDOW_PERCENT_RANK: {
			int64_t denom = (int64_t)bounds.partition_end - bounds.partition_start - 1;
			double percent_rank = denom > 0 ? ((double)rank - 1) / denom : 0;
			auto rdata = FlatVector::GetData<double>(result);
			rdata[output_offset] = percent_rank;
			break;
		}
		case ExpressionType::WINDOW_CUME_DIST: {
			int64_t denom = (int64_t)bounds.partition_end - bounds.partition_start;
			double cume_dist = denom > 0 ? ((double)(bounds.peer_end - bounds.partition_start)) / denom : 0;
			auto rdata = FlatVector::GetData<double>(result);
			rdata[output_offset] = cume_dist;
			break;
		}
		case ExpressionType::WINDOW_NTILE: {
			D_ASSERT(payload_collection.ColumnCount() == 1);
			if (CellIsNull(payload_collection, 0, row_idx)) {
				FlatVector::SetNull(result, output_offset, true);
			} else {
				auto n_param = GetCell<int64_t>(payload_collection, 0, row_idx);
				if (n_param < 1) {
					throw InvalidInputException("Argument for ntile must be greater than zero");
				}
				// With thanks from SQLite's ntileValueFunc()
				int64_t n_total = bounds.partition_end - bounds.partition_start;
				if (n_param > n_total) {
					// more groups allowed than we have values
					// map every entry to a unique group
					n_param = n_total;
				}
				int64_t n_size = (n_total / n_param);
				// find the row idx within the group
				D_ASSERT(row_idx >= bounds.partition_start);
				int64_t adjusted_row_idx = row_idx - bounds.partition_start;
				// now compute the ntile
				int64_t n_large = n_total - n_param * n_size;
				int64_t i_small = n_large * (n_size + 1);
				int64_t result_ntile;

				D_ASSERT((n_large * (n_size + 1) + (n_param - n_large) * n_size) == n_total);

				if (adjusted_row_idx < i_small) {
					result_ntile = 1 + adjusted_row_idx / (n_size + 1);
				} else {
					result_ntile = 1 + n_large + (adjusted_row_idx - i_small) / n_size;
				}
				// result has to be between [1, NTILE]
				D_ASSERT(result_ntile >= 1 && result_ntile <= n_param);
				auto rdata = FlatVector::GetData<int64_t>(result);
				rdata[output_offset] = result_ntile;
			}
			break;
		}
		case ExpressionType::WINDOW_LEAD:
		case ExpressionType::WINDOW_LAG: {
			int64_t offset = 1;
			if (wexpr->offset_expr) {
				offset = leadlag_offset.GetCell<int64_t>(output_offset);
			}
			int64_t val_idx = (int64_t)row_idx;
			if (wexpr->type == ExpressionType::WINDOW_LEAD) {
				val_idx += offset;
			} else {
				val_idx -= offset;
			}

			idx_t delta = 0;
			if (val_idx < (int64_t)row_idx) {
				// Count backwards
				delta = idx_t(row_idx - val_idx);
				val_idx = FindPrevStart(ignore_nulls, bounds.partition_start, row_idx, delta);
			} else if (val_idx > (int64_t)row_idx) {
				delta = idx_t(val_idx - row_idx);
				val_idx = FindNextStart(ignore_nulls, row_idx + 1, bounds.partition_end, delta);
			}
			// else offset is zero, so don't move.

			if (!delta) {
				CopyCell(payload_collection, 0, val_idx, result, output_offset);
			} else if (wexpr->default_expr) {
				leadlag_default.CopyCell(result, output_offset);
			} else {
				FlatVector::SetNull(result, output_offset, true);
			}
			break;
		}
		case ExpressionType::WINDOW_FIRST_VALUE: {
			idx_t n = 1;
			const auto first_idx = FindNextStart(ignore_nulls, bounds.window_start, bounds.window_end, n);
			CopyCell(payload_collection, 0, first_idx, result, output_offset);
			break;
		}
		case ExpressionType::WINDOW_LAST_VALUE: {
			idx_t n = 1;
			CopyCell(payload_collection, 0, FindPrevStart(ignore_nulls, bounds.window_start, bounds.window_end, n),
			         result, output_offset);
			break;
		}
		case ExpressionType::WINDOW_NTH_VALUE: {
			D_ASSERT(payload_collection.ColumnCount() == 2);
			// Returns value evaluated at the row that is the n'th row of the window frame (counting from 1);
			// returns NULL if there is no such row.
			if (CellIsNull(payload_collection, 1, row_idx)) {
				FlatVector::SetNull(result, output_offset, true);
			} else {
				auto n_param = GetCell<int64_t>(payload_collection, 1, row_idx);
				if (n_param < 1) {
					FlatVector::SetNull(result, output_offset, true);
				} else {
					auto n = idx_t(n_param);
					const auto nth_index = FindNextStart(ignore_nulls, bounds.window_start, bounds.window_end, n);
					if (!n) {
						CopyCell(payload_collection, 0, nth_index, result, output_offset);
					} else {
						FlatVector::SetNull(result, output_offset, true);
					}
				}
			}
			break;
		}
		default:
			throw InternalException("Window aggregate type %s", ExpressionTypeToString(wexpr->type));
		}
	}

	result.Verify(input_chunk.size());
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
SinkResultType PhysicalWindow::Sink(ExecutionContext &context, GlobalSinkState &gstate_p, LocalSinkState &lstate_p,
                                    DataChunk &input) const {
	auto &gstate = (WindowGlobalSinkState &)gstate_p;
	auto &lstate = (WindowLocalSinkState &)lstate_p;

	lstate.Sink(input, gstate);

	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalWindow::Combine(ExecutionContext &context, GlobalSinkState &gstate_p, LocalSinkState &lstate_p) const {
	auto &gstate = (WindowGlobalSinkState &)gstate_p;
	auto &lstate = (WindowLocalSinkState &)lstate_p;
	lstate.Combine(gstate);
}

unique_ptr<LocalSinkState> PhysicalWindow::GetLocalSinkState(ExecutionContext &context) const {
	return make_unique<WindowLocalSinkState>(context.client, *this);
}

unique_ptr<GlobalSinkState> PhysicalWindow::GetGlobalSinkState(ClientContext &context) const {
	return make_unique<WindowGlobalSinkState>(*this, context);
}

enum class WindowSortStage : uint8_t { INIT, PREPARE, MERGE, SORTED };

class WindowGlobalMergeState;

class WindowLocalMergeState {
public:
	WindowLocalMergeState() : merge_state(nullptr), stage(WindowSortStage::INIT) {
		finished = true;
	}

	bool TaskFinished() {
		return finished;
	}

	void Prepare();
	void Merge();

	void ExecuteTask();

	WindowGlobalMergeState *merge_state;
	WindowSortStage stage;
	atomic<bool> finished;
};

class WindowGlobalMergeState {
public:
	using GroupDataPtr = unique_ptr<ColumnDataCollection>;

	explicit WindowGlobalMergeState(WindowGlobalSinkState &sink, GroupDataPtr group_data)
	    : sink(sink), group_data(std::move(group_data)), stage(WindowSortStage::INIT), total_tasks(0),
	      tasks_assigned(0), tasks_completed(0) {

		const auto group_idx = sink.hash_groups.size();
		auto new_group = make_unique<WindowGlobalHashGroup>(sink.buffer_manager, sink.partitions, sink.orders,
		                                                    sink.payload_types, sink.external);
		sink.hash_groups.emplace_back(std::move(new_group));

		hash_group = sink.hash_groups[group_idx].get();
		global_sort = sink.hash_groups[group_idx]->global_sort.get();
	}

	bool IsSorted() const {
		lock_guard<mutex> guard(lock);
		return stage == WindowSortStage::SORTED;
	}

	bool AssignTask(WindowLocalMergeState &local_state);
	bool TryPrepareNextStage();
	void CompleteTask();

	WindowGlobalSinkState &sink;
	GroupDataPtr group_data;
	WindowGlobalHashGroup *hash_group;
	GlobalSortState *global_sort;

private:
	mutable mutex lock;
	WindowSortStage stage;
	idx_t total_tasks;
	idx_t tasks_assigned;
	idx_t tasks_completed;
};

void WindowLocalMergeState::Prepare() {
	auto &global_sort = *merge_state->global_sort;
	merge_state->sink.BuildSortState(*merge_state->group_data, *merge_state->hash_group);
	merge_state->group_data.reset();

	global_sort.PrepareMergePhase();
}

void WindowLocalMergeState::Merge() {
	auto &global_sort = *merge_state->global_sort;
	MergeSorter merge_sorter(global_sort, global_sort.buffer_manager);
	merge_sorter.PerformInMergeRound();
}

void WindowLocalMergeState::ExecuteTask() {
	switch (stage) {
	case WindowSortStage::PREPARE:
		Prepare();
		break;
	case WindowSortStage::MERGE:
		Merge();
		break;
	default:
		throw InternalException("Unexpected WindowGlobalMergeState in ExecuteTask!");
	}

	merge_state->CompleteTask();
	finished = true;
}

bool WindowGlobalMergeState::AssignTask(WindowLocalMergeState &local_state) {
	lock_guard<mutex> guard(lock);

	if (tasks_assigned >= total_tasks) {
		return false;
	}

	local_state.merge_state = this;
	local_state.stage = stage;
	local_state.finished = false;
	tasks_assigned++;

	return true;
}

void WindowGlobalMergeState::CompleteTask() {
	lock_guard<mutex> guard(lock);

	++tasks_completed;
}

bool WindowGlobalMergeState::TryPrepareNextStage() {
	lock_guard<mutex> guard(lock);

	if (tasks_completed < total_tasks) {
		return false;
	}

	tasks_assigned = tasks_completed = 0;

	switch (stage) {
	case WindowSortStage::INIT:
		total_tasks = 1;
		stage = WindowSortStage::PREPARE;
		return true;

	case WindowSortStage::PREPARE:
		total_tasks = global_sort->sorted_blocks.size() / 2;
		if (!total_tasks) {
			break;
		}
		stage = WindowSortStage::MERGE;
		global_sort->InitializeMergeRound();
		return true;

	case WindowSortStage::MERGE:
		global_sort->CompleteMergeRound(true);
		total_tasks = global_sort->sorted_blocks.size() / 2;
		if (!total_tasks) {
			break;
		}
		global_sort->InitializeMergeRound();
		return true;

	case WindowSortStage::SORTED:
		break;
	}

	stage = WindowSortStage::SORTED;

	return false;
}

class WindowGlobalMergeStates {
public:
	using WindowGlobalMergeStatePtr = unique_ptr<WindowGlobalMergeState>;

	explicit WindowGlobalMergeStates(WindowGlobalSinkState &sink) {
		// Schedule all the sorts for maximum thread utilisation
		for (auto &group_data : sink.grouping_data->GetPartitions()) {
			// Prepare for merge sort phase
			if (group_data->Count()) {
				auto state = make_unique<WindowGlobalMergeState>(sink, std::move(group_data));
				states.emplace_back(std::move(state));
			}
		}
	}

	vector<WindowGlobalMergeStatePtr> states;
};

class WindowMergeTask : public ExecutorTask {
public:
	WindowMergeTask(shared_ptr<Event> event_p, ClientContext &context_p, WindowGlobalMergeStates &hash_groups_p)
	    : ExecutorTask(context_p), event(std::move(event_p)), hash_groups(hash_groups_p) {
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override;

private:
	shared_ptr<Event> event;
	WindowLocalMergeState local_state;
	WindowGlobalMergeStates &hash_groups;
};

TaskExecutionResult WindowMergeTask::ExecuteTask(TaskExecutionMode mode) {
	// Loop until all hash groups are done
	size_t sorted = 0;
	while (sorted < hash_groups.states.size()) {
		// First check if there is an unfinished task for this thread
		if (!local_state.TaskFinished()) {
			local_state.ExecuteTask();
			continue;
		}

		// Thread is done with its assigned task, try to fetch new work
		for (auto group = sorted; group < hash_groups.states.size(); ++group) {
			auto &global_state = hash_groups.states[group];
			if (global_state->IsSorted()) {
				// This hash group is done
				// Update the high water mark of densely completed groups
				if (sorted == group) {
					++sorted;
				}
				continue;
			}

			// Try to assign work for this hash group to this thread
			if (global_state->AssignTask(local_state)) {
				// We assigned a task to this thread!
				// Break out of this loop to re-enter the top-level loop and execute the task
				break;
			}

			// Hash group global state couldn't assign a task to this thread
			// Try to prepare the next stage
			if (!global_state->TryPrepareNextStage()) {
				// This current hash group is not yet done
				// But we were not able to assign a task for it to this thread
				// See if the next hash group is better
				continue;
			}

			// We were able to prepare the next stage for this hash group!
			// Try to assign a task once more
			if (global_state->AssignTask(local_state)) {
				// We assigned a task to this thread!
				// Break out of this loop to re-enter the top-level loop and execute the task
				break;
			}

			// We were able to prepare the next merge round,
			// but we were not able to assign a task for it to this thread
			// The tasks were assigned to other threads while this thread waited for the lock
			// Go to the next iteration to see if another hash group has a task
		}
	}

	event->FinishTask();
	return TaskExecutionResult::TASK_FINISHED;
}

class WindowMergeEvent : public BasePipelineEvent {
public:
	WindowMergeEvent(WindowGlobalSinkState &gstate_p, Pipeline &pipeline_p)
	    : BasePipelineEvent(pipeline_p), gstate(gstate_p), merge_states(gstate_p) {
	}

	WindowGlobalSinkState &gstate;
	WindowGlobalMergeStates merge_states;

public:
	void Schedule() override {
		auto &context = pipeline->GetClientContext();

		// Schedule tasks equal to the number of threads, which will each merge multiple partitions
		auto &ts = TaskScheduler::GetScheduler(context);
		idx_t num_threads = ts.NumberOfThreads();

		vector<unique_ptr<Task>> merge_tasks;
		for (idx_t tnum = 0; tnum < num_threads; tnum++) {
			merge_tasks.push_back(make_unique<WindowMergeTask>(shared_from_this(), context, merge_states));
		}
		SetTasks(std::move(merge_tasks));
	}
};

SinkFinalizeType PhysicalWindow::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                          GlobalSinkState &gstate_p) const {
	auto &state = (WindowGlobalSinkState &)gstate_p;

	//	Did we get any data?
	if (!state.count) {
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}

	// Do we have any sorting to schedule?
	if (state.rows) {
		D_ASSERT(!state.grouping_data);
		return state.rows->count ? SinkFinalizeType::READY : SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}

	// Find the first group to sort
	auto &groups = state.grouping_data->GetPartitions();
	if (groups.empty()) {
		// Empty input!
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}

	// Schedule all the sorts for maximum thread utilisation
	auto new_event = make_shared<WindowMergeEvent>(state, pipeline);
	event.InsertEvent(std::move(new_event));

	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class WindowGlobalSourceState : public GlobalSourceState {
public:
	explicit WindowGlobalSourceState(const PhysicalWindow &op) : op(op), next_bin(0) {
	}

	const PhysicalWindow &op;
	//! The output read position.
	atomic<idx_t> next_bin;

public:
	idx_t MaxThreads() override {
		auto &state = (WindowGlobalSinkState &)*op.sink_state;

		// If there is only one partition, we have to process it on one thread.
		if (!state.grouping_data) {
			return 1;
		}

		// If there is not a lot of data, process serially.
		if (state.count < STANDARD_ROW_GROUPS_SIZE) {
			return 1;
		}

		return state.hash_groups.size();
	}
};

// Per-thread read state
class WindowLocalSourceState : public LocalSourceState {
public:
	using HashGroupPtr = unique_ptr<WindowGlobalHashGroup>;
	using WindowExecutorPtr = unique_ptr<WindowExecutor>;
	using WindowExecutors = vector<WindowExecutorPtr>;

	WindowLocalSourceState(const PhysicalWindow &op, ExecutionContext &context, WindowGlobalSourceState &gstate)
	    : context(context.client), allocator(Allocator::Get(context.client)) {
		vector<LogicalType> output_types;
		for (idx_t expr_idx = 0; expr_idx < op.select_list.size(); ++expr_idx) {
			D_ASSERT(op.select_list[expr_idx]->GetExpressionClass() == ExpressionClass::BOUND_WINDOW);
			auto wexpr = reinterpret_cast<BoundWindowExpression *>(op.select_list[expr_idx].get());
			output_types.emplace_back(wexpr->return_type);
		}
		output_chunk.Initialize(allocator, output_types);

		const auto &input_types = op.children[0]->types;
		layout.Initialize(input_types);
		input_chunk.Initialize(allocator, input_types);
	}

	void MaterializeSortedData();
	void GeneratePartition(WindowGlobalSinkState &gstate, const idx_t hash_bin);
	void Scan(DataChunk &chunk);

	HashGroupPtr hash_group;
	ClientContext &context;
	Allocator &allocator;

	//! The generated input chunks
	unique_ptr<RowDataCollection> rows;
	unique_ptr<RowDataCollection> heap;
	RowLayout layout;
	//! The partition boundary mask
	vector<validity_t> partition_bits;
	ValidityMask partition_mask;
	//! The order boundary mask
	vector<validity_t> order_bits;
	ValidityMask order_mask;
	//! The current execution functions
	WindowExecutors window_execs;

	//! The read partition
	idx_t hash_bin;
	//! The read cursor
	unique_ptr<RowDataCollectionScanner> scanner;
	//! Buffer for the inputs
	DataChunk input_chunk;
	//! Buffer for window results
	DataChunk output_chunk;
};

void WindowLocalSourceState::MaterializeSortedData() {
	auto &global_sort_state = *hash_group->global_sort;
	if (global_sort_state.sorted_blocks.empty()) {
		return;
	}

	// scan the sorted row data
	D_ASSERT(global_sort_state.sorted_blocks.size() == 1);
	auto &sb = *global_sort_state.sorted_blocks[0];

	// Free up some memory before allocating more
	sb.radix_sorting_data.clear();
	sb.blob_sorting_data = nullptr;

	// Move the sorting row blocks into our RDCs
	auto &buffer_manager = global_sort_state.buffer_manager;
	auto &sd = *sb.payload_data;

	// Data blocks are required
	D_ASSERT(!sd.data_blocks.empty());
	auto &block = sd.data_blocks[0];
	rows = make_unique<RowDataCollection>(buffer_manager, block->capacity, block->entry_size);
	rows->blocks = std::move(sd.data_blocks);
	rows->count = std::accumulate(rows->blocks.begin(), rows->blocks.end(), idx_t(0),
	                              [&](idx_t c, const unique_ptr<RowDataBlock> &b) { return c + b->count; });

	// Heap blocks are optional, but we want both for iteration.
	if (!sd.heap_blocks.empty()) {
		auto &block = sd.heap_blocks[0];
		heap = make_unique<RowDataCollection>(buffer_manager, block->capacity, block->entry_size);
		heap->blocks = std::move(sd.heap_blocks);
		hash_group.reset();
	} else {
		heap = make_unique<RowDataCollection>(buffer_manager, (idx_t)Storage::BLOCK_SIZE, 1, true);
	}
	heap->count = std::accumulate(heap->blocks.begin(), heap->blocks.end(), idx_t(0),
	                              [&](idx_t c, const unique_ptr<RowDataBlock> &b) { return c + b->count; });
}

void WindowLocalSourceState::GeneratePartition(WindowGlobalSinkState &gstate, const idx_t hash_bin_p) {
	auto &op = (PhysicalWindow &)gstate.op;

	//	Get rid of any stale data
	hash_bin = hash_bin_p;

	// There are three types of partitions:
	// 1. No partition (no sorting)
	// 2. One partition (sorting, but no hashing)
	// 3. Multiple partitions (sorting and hashing)

	//	How big is the partition?
	idx_t count = 0;
	if (hash_bin < gstate.hash_groups.size() && gstate.hash_groups[hash_bin]) {
		count = gstate.hash_groups[hash_bin]->count;
	} else if (gstate.rows && !hash_bin) {
		count = gstate.count;
	} else {
		return;
	}

	// Create the executors for each function
	window_execs.clear();
	for (idx_t expr_idx = 0; expr_idx < op.select_list.size(); ++expr_idx) {
		D_ASSERT(op.select_list[expr_idx]->GetExpressionClass() == ExpressionClass::BOUND_WINDOW);
		auto wexpr = reinterpret_cast<BoundWindowExpression *>(op.select_list[expr_idx].get());
		auto wexec = make_unique<WindowExecutor>(wexpr, context, count);
		window_execs.emplace_back(std::move(wexec));
	}

	//	Initialise masks to false
	const auto bit_count = ValidityMask::ValidityMaskSize(count);
	partition_bits.clear();
	partition_bits.resize(bit_count, 0);
	partition_mask.Initialize(partition_bits.data());

	order_bits.clear();
	order_bits.resize(bit_count, 0);
	order_mask.Initialize(order_bits.data());

	// Scan the sorted data into new Collections
	auto external = gstate.external;
	if (gstate.rows && !hash_bin) {
		// Simple mask
		partition_mask.SetValidUnsafe(0);
		order_mask.SetValidUnsafe(0);
		//	No partition - align the heap blocks with the row blocks
		rows = gstate.rows->CloneEmpty(gstate.rows->keep_pinned);
		heap = gstate.strings->CloneEmpty(gstate.strings->keep_pinned);
		RowDataCollectionScanner::AlignHeapBlocks(*rows, *heap, *gstate.rows, *gstate.strings, layout);
		external = true;
	} else if (hash_bin < gstate.hash_groups.size() && gstate.hash_groups[hash_bin]) {
		// Overwrite the collections with the sorted data
		hash_group = std::move(gstate.hash_groups[hash_bin]);
		hash_group->ComputeMasks(partition_mask, order_mask);
		MaterializeSortedData();
	} else {
		return;
	}

	//	First pass over the input without flushing
	//	TODO: Factor out the constructor data as global state
	scanner = make_unique<RowDataCollectionScanner>(*rows, *heap, layout, external, false);
	idx_t input_idx = 0;
	while (true) {
		input_chunk.Reset();
		scanner->Scan(input_chunk);
		if (input_chunk.size() == 0) {
			break;
		}

		//	TODO: Parallelization opportunity
		for (auto &wexec : window_execs) {
			wexec->Sink(input_chunk, input_idx, scanner->Count());
		}
		input_idx += input_chunk.size();
	}

	//	TODO: Parallelization opportunity
	for (auto &wexec : window_execs) {
		wexec->Finalize(gstate.mode);
	}

	// External scanning assumes all blocks are swizzled.
	scanner->ReSwizzle();

	//	Second pass can flush
	scanner->Reset(true);
}

void WindowLocalSourceState::Scan(DataChunk &result) {
	D_ASSERT(scanner);
	if (!scanner->Remaining()) {
		return;
	}

	const auto position = scanner->Scanned();
	input_chunk.Reset();
	scanner->Scan(input_chunk);

	output_chunk.Reset();
	for (idx_t expr_idx = 0; expr_idx < window_execs.size(); ++expr_idx) {
		auto &executor = *window_execs[expr_idx];
		executor.Evaluate(position, input_chunk, output_chunk.data[expr_idx], partition_mask, order_mask);
	}
	output_chunk.SetCardinality(input_chunk);
	output_chunk.Verify();

	idx_t out_idx = 0;
	result.SetCardinality(input_chunk);
	for (idx_t col_idx = 0; col_idx < input_chunk.ColumnCount(); col_idx++) {
		result.data[out_idx++].Reference(input_chunk.data[col_idx]);
	}
	for (idx_t col_idx = 0; col_idx < output_chunk.ColumnCount(); col_idx++) {
		result.data[out_idx++].Reference(output_chunk.data[col_idx]);
	}
	result.Verify();
}

unique_ptr<LocalSourceState> PhysicalWindow::GetLocalSourceState(ExecutionContext &context,
                                                                 GlobalSourceState &gstate_p) const {
	auto &gstate = (WindowGlobalSourceState &)gstate_p;
	return make_unique<WindowLocalSourceState>(*this, context, gstate);
}

unique_ptr<GlobalSourceState> PhysicalWindow::GetGlobalSourceState(ClientContext &context) const {
	return make_unique<WindowGlobalSourceState>(*this);
}

void PhysicalWindow::GetData(ExecutionContext &context, DataChunk &chunk, GlobalSourceState &gstate_p,
                             LocalSourceState &lstate_p) const {
	auto &state = (WindowLocalSourceState &)lstate_p;
	auto &global_source = (WindowGlobalSourceState &)gstate_p;
	auto &gstate = (WindowGlobalSinkState &)*sink_state;

	const auto bin_count = gstate.hash_groups.empty() ? 1 : gstate.hash_groups.size();

	while (chunk.size() == 0) {
		//	Move to the next bin if we are done.
		while (!state.scanner || !state.scanner->Remaining()) {
			state.scanner.reset();
			state.rows.reset();
			state.heap.reset();
			state.hash_group.reset();
			auto hash_bin = global_source.next_bin++;
			if (hash_bin >= bin_count) {
				return;
			}

			for (; hash_bin < gstate.hash_groups.size(); hash_bin = global_source.next_bin++) {
				if (gstate.hash_groups[hash_bin]) {
					break;
				}
			}
			state.GeneratePartition(gstate, hash_bin);
		}

		state.Scan(chunk);
	}
}

string PhysicalWindow::ParamsToString() const {
	string result;
	for (idx_t i = 0; i < select_list.size(); i++) {
		if (i > 0) {
			result += "\n";
		}
		result += select_list[i]->GetName();
	}
	return result;
}

} // namespace duckdb




namespace duckdb {

PhysicalFilter::PhysicalFilter(vector<LogicalType> types, vector<unique_ptr<Expression>> select_list,
                               idx_t estimated_cardinality)
    : CachingPhysicalOperator(PhysicalOperatorType::FILTER, std::move(types), estimated_cardinality) {
	D_ASSERT(select_list.size() > 0);
	if (select_list.size() > 1) {
		// create a big AND out of the expressions
		auto conjunction = make_unique<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
		for (auto &expr : select_list) {
			conjunction->children.push_back(std::move(expr));
		}
		expression = std::move(conjunction);
	} else {
		expression = std::move(select_list[0]);
	}
}

class FilterState : public CachingOperatorState {
public:
	explicit FilterState(ExecutionContext &context, Expression &expr)
	    : executor(context.client, expr), sel(STANDARD_VECTOR_SIZE) {
	}

	ExpressionExecutor executor;
	SelectionVector sel;

public:
	void Finalize(PhysicalOperator *op, ExecutionContext &context) override {
		context.thread.profiler.Flush(op, &executor, "filter", 0);
	}
};

unique_ptr<OperatorState> PhysicalFilter::GetOperatorState(ExecutionContext &context) const {
	return make_unique<FilterState>(context, *expression);
}

OperatorResultType PhysicalFilter::ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                   GlobalOperatorState &gstate, OperatorState &state_p) const {
	auto &state = (FilterState &)state_p;
	idx_t result_count = state.executor.SelectExpression(input, state.sel);
	if (result_count == input.size()) {
		// nothing was filtered: skip adding any selection vectors
		chunk.Reference(input);
	} else {
		chunk.Slice(input, state.sel, result_count);
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

string PhysicalFilter::ParamsToString() const {
	auto result = expression->GetName();
	result += "\n[INFOSEPARATOR]\n";
	result += StringUtil::Format("EC: %llu", estimated_props->GetCardinality<idx_t>());
	return result;
}

} // namespace duckdb





namespace duckdb {

PhysicalBatchCollector::PhysicalBatchCollector(PreparedStatementData &data) : PhysicalResultCollector(data) {
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class BatchCollectorGlobalState : public GlobalSinkState {
public:
	BatchCollectorGlobalState(ClientContext &context, const PhysicalBatchCollector &op) : data(op.types) {
	}

	mutex glock;
	BatchedDataCollection data;
	unique_ptr<MaterializedQueryResult> result;
};

class BatchCollectorLocalState : public LocalSinkState {
public:
	BatchCollectorLocalState(ClientContext &context, const PhysicalBatchCollector &op) : data(op.types) {
	}

	BatchedDataCollection data;
};

SinkResultType PhysicalBatchCollector::Sink(ExecutionContext &context, GlobalSinkState &gstate,
                                            LocalSinkState &lstate_p, DataChunk &input) const {
	auto &state = (BatchCollectorLocalState &)lstate_p;
	state.data.Append(input, state.batch_index);
	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalBatchCollector::Combine(ExecutionContext &context, GlobalSinkState &gstate_p,
                                     LocalSinkState &lstate_p) const {
	auto &gstate = (BatchCollectorGlobalState &)gstate_p;
	auto &state = (BatchCollectorLocalState &)lstate_p;

	lock_guard<mutex> lock(gstate.glock);
	gstate.data.Merge(state.data);
}

SinkFinalizeType PhysicalBatchCollector::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                  GlobalSinkState &gstate_p) const {
	auto &gstate = (BatchCollectorGlobalState &)gstate_p;
	auto collection = gstate.data.FetchCollection();
	D_ASSERT(collection);
	auto result = make_unique<MaterializedQueryResult>(statement_type, properties, names, std::move(collection),
	                                                   context.GetClientProperties());
	gstate.result = std::move(result);
	return SinkFinalizeType::READY;
}

unique_ptr<LocalSinkState> PhysicalBatchCollector::GetLocalSinkState(ExecutionContext &context) const {
	return make_unique<BatchCollectorLocalState>(context.client, *this);
}

unique_ptr<GlobalSinkState> PhysicalBatchCollector::GetGlobalSinkState(ClientContext &context) const {
	return make_unique<BatchCollectorGlobalState>(context, *this);
}

unique_ptr<QueryResult> PhysicalBatchCollector::GetResult(GlobalSinkState &state) {
	auto &gstate = (BatchCollectorGlobalState &)state;
	D_ASSERT(gstate.result);
	return std::move(gstate.result);
}

} // namespace duckdb




namespace duckdb {

PhysicalExecute::PhysicalExecute(PhysicalOperator *plan)
    : PhysicalOperator(PhysicalOperatorType::EXECUTE, plan->types, -1), plan(plan) {
}

vector<PhysicalOperator *> PhysicalExecute::GetChildren() const {
	return {plan};
}

bool PhysicalExecute::AllOperatorsPreserveOrder() const {
	D_ASSERT(plan);
	return plan->AllOperatorsPreserveOrder();
}

void PhysicalExecute::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	// EXECUTE statement: build pipeline on child
	meta_pipeline.Build(plan);
}

} // namespace duckdb




namespace duckdb {

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class ExplainAnalyzeStateGlobalState : public GlobalSinkState {
public:
	string analyzed_plan;
};

SinkResultType PhysicalExplainAnalyze::Sink(ExecutionContext &context, GlobalSinkState &state, LocalSinkState &lstate,
                                            DataChunk &input) const {
	return SinkResultType::NEED_MORE_INPUT;
}

SinkFinalizeType PhysicalExplainAnalyze::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                  GlobalSinkState &gstate_p) const {
	auto &gstate = (ExplainAnalyzeStateGlobalState &)gstate_p;
	auto &profiler = QueryProfiler::Get(context);
	gstate.analyzed_plan = profiler.ToString();
	return SinkFinalizeType::READY;
}

unique_ptr<GlobalSinkState> PhysicalExplainAnalyze::GetGlobalSinkState(ClientContext &context) const {
	return make_unique<ExplainAnalyzeStateGlobalState>();
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class ExplainAnalyzeState : public GlobalSourceState {
public:
	ExplainAnalyzeState() : finished(false) {
	}

	bool finished;
};

unique_ptr<GlobalSourceState> PhysicalExplainAnalyze::GetGlobalSourceState(ClientContext &context) const {
	return make_unique<ExplainAnalyzeState>();
}

void PhysicalExplainAnalyze::GetData(ExecutionContext &context, DataChunk &chunk, GlobalSourceState &source_state,
                                     LocalSourceState &lstate) const {
	auto &state = (ExplainAnalyzeState &)source_state;
	auto &gstate = (ExplainAnalyzeStateGlobalState &)*sink_state;
	if (state.finished) {
		return;
	}
	chunk.SetValue(0, 0, Value("analyzed_plan"));
	chunk.SetValue(1, 0, Value(gstate.analyzed_plan));
	chunk.SetCardinality(1);

	state.finished = true;
}

} // namespace duckdb









namespace duckdb {

PhysicalLimit::PhysicalLimit(vector<LogicalType> types, idx_t limit, idx_t offset,
                             unique_ptr<Expression> limit_expression, unique_ptr<Expression> offset_expression,
                             idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::LIMIT, std::move(types), estimated_cardinality), limit_value(limit),
      offset_value(offset), limit_expression(std::move(limit_expression)),
      offset_expression(std::move(offset_expression)) {
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class LimitGlobalState : public GlobalSinkState {
public:
	explicit LimitGlobalState(ClientContext &context, const PhysicalLimit &op) : data(op.types) {
		limit = 0;
		offset = 0;
	}

	mutex glock;
	idx_t limit;
	idx_t offset;
	BatchedDataCollection data;
};

class LimitLocalState : public LocalSinkState {
public:
	explicit LimitLocalState(ClientContext &context, const PhysicalLimit &op) : current_offset(0), data(op.types) {
		this->limit = op.limit_expression ? DConstants::INVALID_INDEX : op.limit_value;
		this->offset = op.offset_expression ? DConstants::INVALID_INDEX : op.offset_value;
	}

	idx_t current_offset;
	idx_t limit;
	idx_t offset;
	BatchedDataCollection data;
};

unique_ptr<GlobalSinkState> PhysicalLimit::GetGlobalSinkState(ClientContext &context) const {
	return make_unique<LimitGlobalState>(context, *this);
}

unique_ptr<LocalSinkState> PhysicalLimit::GetLocalSinkState(ExecutionContext &context) const {
	return make_unique<LimitLocalState>(context.client, *this);
}

bool PhysicalLimit::ComputeOffset(ExecutionContext &context, DataChunk &input, idx_t &limit, idx_t &offset,
                                  idx_t current_offset, idx_t &max_element, Expression *limit_expression,
                                  Expression *offset_expression) {
	if (limit != DConstants::INVALID_INDEX && offset != DConstants::INVALID_INDEX) {
		max_element = limit + offset;
		if ((limit == 0 || current_offset >= max_element) && !(limit_expression || offset_expression)) {
			return false;
		}
	}

	// get the next chunk from the child
	if (limit == DConstants::INVALID_INDEX) {
		limit = 1ULL << 62ULL;
		Value val = GetDelimiter(context, input, limit_expression);
		if (!val.IsNull()) {
			limit = val.GetValue<idx_t>();
		}
		if (limit > 1ULL << 62ULL) {
			throw BinderException("Max value %lld for LIMIT/OFFSET is %lld", limit, 1ULL << 62ULL);
		}
	}
	if (offset == DConstants::INVALID_INDEX) {
		offset = 0;
		Value val = GetDelimiter(context, input, offset_expression);
		if (!val.IsNull()) {
			offset = val.GetValue<idx_t>();
		}
		if (offset > 1ULL << 62ULL) {
			throw BinderException("Max value %lld for LIMIT/OFFSET is %lld", offset, 1ULL << 62ULL);
		}
	}
	max_element = limit + offset;
	if (limit == 0 || current_offset >= max_element) {
		return false;
	}
	return true;
}

SinkResultType PhysicalLimit::Sink(ExecutionContext &context, GlobalSinkState &gstate, LocalSinkState &lstate,
                                   DataChunk &input) const {

	D_ASSERT(input.size() > 0);
	auto &state = (LimitLocalState &)lstate;
	auto &limit = state.limit;
	auto &offset = state.offset;

	idx_t max_element;
	if (!ComputeOffset(context, input, limit, offset, state.current_offset, max_element, limit_expression.get(),
	                   offset_expression.get())) {
		return SinkResultType::FINISHED;
	}
	auto max_cardinality = max_element - state.current_offset;
	if (max_cardinality < input.size()) {
		input.SetCardinality(max_cardinality);
	}
	state.data.Append(input, lstate.batch_index);
	state.current_offset += input.size();
	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalLimit::Combine(ExecutionContext &context, GlobalSinkState &gstate_p, LocalSinkState &lstate_p) const {
	auto &gstate = (LimitGlobalState &)gstate_p;
	auto &state = (LimitLocalState &)lstate_p;

	lock_guard<mutex> lock(gstate.glock);
	gstate.limit = state.limit;
	gstate.offset = state.offset;
	gstate.data.Merge(state.data);
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class LimitSourceState : public GlobalSourceState {
public:
	LimitSourceState() {
		initialized = false;
		current_offset = 0;
	}

	bool initialized;
	idx_t current_offset;
	BatchedChunkScanState scan_state;
};

unique_ptr<GlobalSourceState> PhysicalLimit::GetGlobalSourceState(ClientContext &context) const {
	return make_unique<LimitSourceState>();
}

void PhysicalLimit::GetData(ExecutionContext &context, DataChunk &chunk, GlobalSourceState &gstate_p,
                            LocalSourceState &lstate) const {
	auto &gstate = (LimitGlobalState &)*sink_state;
	auto &state = (LimitSourceState &)gstate_p;
	while (state.current_offset < gstate.limit + gstate.offset) {
		if (!state.initialized) {
			gstate.data.InitializeScan(state.scan_state);
			state.initialized = true;
		}
		gstate.data.Scan(state.scan_state, chunk);
		if (chunk.size() == 0) {
			break;
		}
		if (HandleOffset(chunk, state.current_offset, gstate.offset, gstate.limit)) {
			break;
		}
	}
}

bool PhysicalLimit::HandleOffset(DataChunk &input, idx_t &current_offset, idx_t offset, idx_t limit) {
	idx_t max_element = limit + offset;
	if (limit == DConstants::INVALID_INDEX) {
		max_element = DConstants::INVALID_INDEX;
	}
	idx_t input_size = input.size();
	if (current_offset < offset) {
		// we are not yet at the offset point
		if (current_offset + input.size() > offset) {
			// however we will reach it in this chunk
			// we have to copy part of the chunk with an offset
			idx_t start_position = offset - current_offset;
			auto chunk_count = MinValue<idx_t>(limit, input.size() - start_position);
			SelectionVector sel(STANDARD_VECTOR_SIZE);
			for (idx_t i = 0; i < chunk_count; i++) {
				sel.set_index(i, start_position + i);
			}
			// set up a slice of the input chunks
			input.Slice(input, sel, chunk_count);
		} else {
			current_offset += input_size;
			return false;
		}
	} else {
		// have to copy either the entire chunk or part of it
		idx_t chunk_count;
		if (current_offset + input.size() >= max_element) {
			// have to limit the count of the chunk
			chunk_count = max_element - current_offset;
		} else {
			// we copy the entire chunk
			chunk_count = input.size();
		}
		// instead of copying we just change the pointer in the current chunk
		input.Reference(input);
		input.SetCardinality(chunk_count);
	}

	current_offset += input_size;
	return true;
}

Value PhysicalLimit::GetDelimiter(ExecutionContext &context, DataChunk &input, Expression *expr) {
	DataChunk limit_chunk;
	vector<LogicalType> types {expr->return_type};
	auto &allocator = Allocator::Get(context.client);
	limit_chunk.Initialize(allocator, types);
	ExpressionExecutor limit_executor(context.client, expr);
	auto input_size = input.size();
	input.SetCardinality(1);
	limit_executor.Execute(input, limit_chunk);
	input.SetCardinality(input_size);
	auto limit_value = limit_chunk.GetValue(0, 0);
	return limit_value;
}

} // namespace duckdb








namespace duckdb {

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class LimitPercentGlobalState : public GlobalSinkState {
public:
	explicit LimitPercentGlobalState(ClientContext &context, const PhysicalLimitPercent &op)
	    : current_offset(0), data(context, op.GetTypes()) {
		if (!op.limit_expression) {
			this->limit_percent = op.limit_percent;
			is_limit_percent_delimited = true;
		} else {
			this->limit_percent = 100.0;
		}

		if (!op.offset_expression) {
			this->offset = op.offset_value;
			is_offset_delimited = true;
		} else {
			this->offset = 0;
		}
	}

	idx_t current_offset;
	double limit_percent;
	idx_t offset;
	ColumnDataCollection data;

	bool is_limit_percent_delimited = false;
	bool is_offset_delimited = false;
};

unique_ptr<GlobalSinkState> PhysicalLimitPercent::GetGlobalSinkState(ClientContext &context) const {
	return make_unique<LimitPercentGlobalState>(context, *this);
}

SinkResultType PhysicalLimitPercent::Sink(ExecutionContext &context, GlobalSinkState &gstate, LocalSinkState &lstate,
                                          DataChunk &input) const {
	D_ASSERT(input.size() > 0);
	auto &state = (LimitPercentGlobalState &)gstate;
	auto &limit_percent = state.limit_percent;
	auto &offset = state.offset;

	// get the next chunk from the child
	if (!state.is_limit_percent_delimited) {
		Value val = PhysicalLimit::GetDelimiter(context, input, limit_expression.get());
		if (!val.IsNull()) {
			limit_percent = val.GetValue<double>();
		}
		if (limit_percent < 0.0) {
			throw BinderException("Percentage value(%f) can't be negative", limit_percent);
		}
		state.is_limit_percent_delimited = true;
	}
	if (!state.is_offset_delimited) {
		Value val = PhysicalLimit::GetDelimiter(context, input, offset_expression.get());
		if (!val.IsNull()) {
			offset = val.GetValue<idx_t>();
		}
		if (offset > 1ULL << 62ULL) {
			throw BinderException("Max value %lld for LIMIT/OFFSET is %lld", offset, 1ULL << 62ULL);
		}
		state.is_offset_delimited = true;
	}

	if (!PhysicalLimit::HandleOffset(input, state.current_offset, offset, DConstants::INVALID_INDEX)) {
		return SinkResultType::NEED_MORE_INPUT;
	}

	state.data.Append(input);
	return SinkResultType::NEED_MORE_INPUT;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class LimitPercentOperatorState : public GlobalSourceState {
public:
	explicit LimitPercentOperatorState(const PhysicalLimitPercent &op)
	    : limit(DConstants::INVALID_INDEX), current_offset(0) {
		D_ASSERT(op.sink_state);
		auto &gstate = (LimitPercentGlobalState &)*op.sink_state;
		gstate.data.InitializeScan(scan_state);
	}

	ColumnDataScanState scan_state;
	idx_t limit;
	idx_t current_offset;
};

unique_ptr<GlobalSourceState> PhysicalLimitPercent::GetGlobalSourceState(ClientContext &context) const {
	return make_unique<LimitPercentOperatorState>(*this);
}

void PhysicalLimitPercent::GetData(ExecutionContext &context, DataChunk &chunk, GlobalSourceState &gstate_p,
                                   LocalSourceState &lstate) const {
	auto &gstate = (LimitPercentGlobalState &)*sink_state;
	auto &state = (LimitPercentOperatorState &)gstate_p;
	auto &percent_limit = gstate.limit_percent;
	auto &offset = gstate.offset;
	auto &limit = state.limit;
	auto &current_offset = state.current_offset;

	if (gstate.is_limit_percent_delimited && limit == DConstants::INVALID_INDEX) {
		idx_t count = gstate.data.Count();
		if (count > 0) {
			count += offset;
		}
		if (Value::IsNan(percent_limit) || percent_limit < 0 || percent_limit > 100) {
			throw OutOfRangeException("Limit percent out of range, should be between 0% and 100%");
		}
		double limit_dbl = percent_limit / 100 * count;
		if (limit_dbl > count) {
			limit = count;
		} else {
			limit = idx_t(limit_dbl);
		}
		if (limit == 0) {
			return;
		}
	}

	if (current_offset >= limit) {
		return;
	}
	if (!gstate.data.Scan(state.scan_state, chunk)) {
		return;
	}

	PhysicalLimit::HandleOffset(chunk, current_offset, 0, limit);
}

} // namespace duckdb



namespace duckdb {

void PhysicalLoad::GetData(ExecutionContext &context, DataChunk &chunk, GlobalSourceState &gstate_p,
                           LocalSourceState &lstate) const {
	if (info->load_type == LoadType::INSTALL || info->load_type == LoadType::FORCE_INSTALL) {
		ExtensionHelper::InstallExtension(context.client, info->filename, info->load_type == LoadType::FORCE_INSTALL);
	} else {
		ExtensionHelper::LoadExternalExtension(context.client, info->filename);
	}
}

} // namespace duckdb





namespace duckdb {

PhysicalMaterializedCollector::PhysicalMaterializedCollector(PreparedStatementData &data, bool parallel)
    : PhysicalResultCollector(data), parallel(parallel) {
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class MaterializedCollectorGlobalState : public GlobalSinkState {
public:
	mutex glock;
	unique_ptr<ColumnDataCollection> collection;
	shared_ptr<ClientContext> context;
};

class MaterializedCollectorLocalState : public LocalSinkState {
public:
	unique_ptr<ColumnDataCollection> collection;
	ColumnDataAppendState append_state;
};

SinkResultType PhysicalMaterializedCollector::Sink(ExecutionContext &context, GlobalSinkState &gstate_p,
                                                   LocalSinkState &lstate_p, DataChunk &input) const {
	auto &lstate = (MaterializedCollectorLocalState &)lstate_p;
	lstate.collection->Append(lstate.append_state, input);
	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalMaterializedCollector::Combine(ExecutionContext &context, GlobalSinkState &gstate_p,
                                            LocalSinkState &lstate_p) const {
	auto &gstate = (MaterializedCollectorGlobalState &)gstate_p;
	auto &lstate = (MaterializedCollectorLocalState &)lstate_p;
	if (lstate.collection->Count() == 0) {
		return;
	}

	lock_guard<mutex> l(gstate.glock);
	if (!gstate.collection) {
		gstate.collection = std::move(lstate.collection);
	} else {
		gstate.collection->Combine(*lstate.collection);
	}
}

unique_ptr<GlobalSinkState> PhysicalMaterializedCollector::GetGlobalSinkState(ClientContext &context) const {
	auto state = make_unique<MaterializedCollectorGlobalState>();
	state->context = context.shared_from_this();
	return std::move(state);
}

unique_ptr<LocalSinkState> PhysicalMaterializedCollector::GetLocalSinkState(ExecutionContext &context) const {
	auto state = make_unique<MaterializedCollectorLocalState>();
	state->collection = make_unique<ColumnDataCollection>(Allocator::DefaultAllocator(), types);
	state->collection->InitializeAppend(state->append_state);
	return std::move(state);
}

unique_ptr<QueryResult> PhysicalMaterializedCollector::GetResult(GlobalSinkState &state) {
	auto &gstate = (MaterializedCollectorGlobalState &)state;
	if (!gstate.collection) {
		gstate.collection = make_unique<ColumnDataCollection>(Allocator::DefaultAllocator(), types);
	}
	auto result = make_unique<MaterializedQueryResult>(statement_type, properties, names, std::move(gstate.collection),
	                                                   gstate.context->GetClientProperties());
	return std::move(result);
}

bool PhysicalMaterializedCollector::ParallelSink() const {
	return parallel;
}

} // namespace duckdb


namespace duckdb {

void PhysicalPragma::GetData(ExecutionContext &context, DataChunk &chunk, GlobalSourceState &gstate_p,
                             LocalSourceState &lstate) const {
	auto &client = context.client;
	FunctionParameters parameters {info.parameters, info.named_parameters};
	function.function(client, parameters);
}

} // namespace duckdb



namespace duckdb {

void PhysicalPrepare::GetData(ExecutionContext &context, DataChunk &chunk, GlobalSourceState &gstate_p,
                              LocalSourceState &lstate) const {
	auto &client = context.client;

	// store the prepared statement in the context
	ClientData::Get(client).prepared_statements[name] = prepared;
}

} // namespace duckdb



namespace duckdb {

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class SampleGlobalSinkState : public GlobalSinkState {
public:
	explicit SampleGlobalSinkState(Allocator &allocator, SampleOptions &options) {
		if (options.is_percentage) {
			auto percentage = options.sample_size.GetValue<double>();
			if (percentage == 0) {
				return;
			}
			sample = make_unique<ReservoirSamplePercentage>(allocator, percentage, options.seed);
		} else {
			auto size = options.sample_size.GetValue<int64_t>();
			if (size == 0) {
				return;
			}
			sample = make_unique<ReservoirSample>(allocator, size, options.seed);
		}
	}

	//! The lock for updating the global aggregate state
	mutex lock;
	//! The reservoir sample
	unique_ptr<BlockingSample> sample;
};

unique_ptr<GlobalSinkState> PhysicalReservoirSample::GetGlobalSinkState(ClientContext &context) const {
	return make_unique<SampleGlobalSinkState>(Allocator::Get(context), *options);
}

SinkResultType PhysicalReservoirSample::Sink(ExecutionContext &context, GlobalSinkState &state, LocalSinkState &lstate,
                                             DataChunk &input) const {
	auto &gstate = (SampleGlobalSinkState &)state;
	if (!gstate.sample) {
		return SinkResultType::FINISHED;
	}
	// we implement reservoir sampling without replacement and exponential jumps here
	// the algorithm is adopted from the paper Weighted random sampling with a reservoir by Pavlos S. Efraimidis et al.
	// note that the original algorithm is about weighted sampling; this is a simplified approach for uniform sampling
	lock_guard<mutex> glock(gstate.lock);
	gstate.sample->AddToReservoir(input);
	return SinkResultType::NEED_MORE_INPUT;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
void PhysicalReservoirSample::GetData(ExecutionContext &context, DataChunk &chunk, GlobalSourceState &gstate,
                                      LocalSourceState &lstate) const {
	auto &sink = (SampleGlobalSinkState &)*this->sink_state;
	if (!sink.sample) {
		return;
	}
	auto sample_chunk = sink.sample->GetChunk();
	if (!sample_chunk) {
		return;
	}
	chunk.Move(*sample_chunk);
}

string PhysicalReservoirSample::ParamsToString() const {
	return options->sample_size.ToString() + (options->is_percentage ? "%" : " rows");
}

} // namespace duckdb






namespace duckdb {

void PhysicalReset::ResetExtensionVariable(ExecutionContext &context, DBConfig &config,
                                           ExtensionOption &extension_option) const {
	if (extension_option.set_function) {
		extension_option.set_function(context.client, scope, extension_option.default_value);
	}
	if (scope == SetScope::GLOBAL) {
		config.ResetOption(name);
	} else {
		auto &client_config = ClientConfig::GetConfig(context.client);
		client_config.set_variables[name] = extension_option.default_value;
	}
}

void PhysicalReset::GetData(ExecutionContext &context, DataChunk &chunk, GlobalSourceState &gstate,
                            LocalSourceState &lstate) const {
	auto option = DBConfig::GetOptionByName(name);
	if (!option) {
		// check if this is an extra extension variable
		auto &config = DBConfig::GetConfig(context.client);
		auto entry = config.extension_parameters.find(name);
		if (entry == config.extension_parameters.end()) {
			// it is not!
			// get a list of all options
			vector<string> potential_names = DBConfig::GetOptionNames();
			for (auto &entry : config.extension_parameters) {
				potential_names.push_back(entry.first);
			}

			throw CatalogException("unrecognized configuration parameter \"%s\"\n%s", name,
			                       StringUtil::CandidatesErrorMessage(potential_names, name, "Did you mean"));
		}
		ResetExtensionVariable(context, config, entry->second);
		return;
	}

	// Transform scope
	SetScope variable_scope = scope;
	if (variable_scope == SetScope::AUTOMATIC) {
		if (option->set_local) {
			variable_scope = SetScope::SESSION;
		} else {
			D_ASSERT(option->set_global);
			variable_scope = SetScope::GLOBAL;
		}
	}

	switch (variable_scope) {
	case SetScope::GLOBAL: {
		if (!option->set_global) {
			throw CatalogException("option \"%s\" cannot be reset globally", name);
		}
		auto &db = DatabaseInstance::GetDatabase(context.client);
		auto &config = DBConfig::GetConfig(context.client);
		config.ResetOption(&db, *option);
		break;
	}
	case SetScope::SESSION:
		if (!option->reset_local) {
			throw CatalogException("option \"%s\" cannot be reset locally", name);
		}
		option->reset_local(context.client);
		break;
	default:
		throw InternalException("Unsupported SetScope for variable");
	}
}

} // namespace duckdb










namespace duckdb {

PhysicalResultCollector::PhysicalResultCollector(PreparedStatementData &data)
    : PhysicalOperator(PhysicalOperatorType::RESULT_COLLECTOR, {LogicalType::BOOLEAN}, 0),
      statement_type(data.statement_type), properties(data.properties), plan(data.plan.get()), names(data.names) {
	this->types = data.types;
}

unique_ptr<PhysicalResultCollector> PhysicalResultCollector::GetResultCollector(ClientContext &context,
                                                                                PreparedStatementData &data) {
	if (!PhysicalPlanGenerator::PreserveInsertionOrder(context, *data.plan)) {
		// the plan is not order preserving, so we just use the parallel materialized collector
		return make_unique_base<PhysicalResultCollector, PhysicalMaterializedCollector>(data, true);
	} else if (!PhysicalPlanGenerator::UseBatchIndex(context, *data.plan)) {
		// the plan is order preserving, but we cannot use the batch index: use a single-threaded result collector
		return make_unique_base<PhysicalResultCollector, PhysicalMaterializedCollector>(data, false);
	} else {
		// we care about maintaining insertion order and the sources all support batch indexes
		// use a batch collector
		return make_unique_base<PhysicalResultCollector, PhysicalBatchCollector>(data);
	}
}

vector<PhysicalOperator *> PhysicalResultCollector::GetChildren() const {
	return {plan};
}

bool PhysicalResultCollector::AllOperatorsPreserveOrder() const {
	D_ASSERT(plan);
	return plan->AllOperatorsPreserveOrder();
}

void PhysicalResultCollector::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	// operator is a sink, build a pipeline
	sink_state.reset();

	D_ASSERT(children.empty());
	D_ASSERT(plan);

	// single operator: the operator becomes the data source of the current pipeline
	auto &state = meta_pipeline.GetState();
	state.SetPipelineSource(current, this);

	// we create a new pipeline starting from the child
	auto child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, this);
	child_meta_pipeline->Build(plan);
}

} // namespace duckdb






namespace duckdb {

void PhysicalSet::SetExtensionVariable(ClientContext &context, ExtensionOption &extension_option, const string &name,
                                       SetScope scope, const Value &value) {
	auto &config = DBConfig::GetConfig(context);
	auto &target_type = extension_option.type;
	Value target_value = value.CastAs(context, target_type);
	if (extension_option.set_function) {
		extension_option.set_function(context, scope, target_value);
	}
	if (scope == SetScope::GLOBAL) {
		config.SetOption(name, std::move(target_value));
	} else {
		auto &client_config = ClientConfig::GetConfig(context);
		client_config.set_variables[name] = std::move(target_value);
	}
}

void PhysicalSet::GetData(ExecutionContext &context, DataChunk &chunk, GlobalSourceState &gstate,
                          LocalSourceState &lstate) const {
	auto option = DBConfig::GetOptionByName(name);
	if (!option) {
		// check if this is an extra extension variable
		auto &config = DBConfig::GetConfig(context.client);
		auto entry = config.extension_parameters.find(name);
		if (entry == config.extension_parameters.end()) {
			// it is not!
			// get a list of all options
			vector<string> potential_names = DBConfig::GetOptionNames();
			for (auto &entry : config.extension_parameters) {
				potential_names.push_back(entry.first);
			}

			throw CatalogException("unrecognized configuration parameter \"%s\"\n%s", name,
			                       StringUtil::CandidatesErrorMessage(potential_names, name, "Did you mean"));
		}
		SetExtensionVariable(context.client, entry->second, name, scope, value);
		return;
	}
	SetScope variable_scope = scope;
	if (variable_scope == SetScope::AUTOMATIC) {
		if (option->set_local) {
			variable_scope = SetScope::SESSION;
		} else {
			D_ASSERT(option->set_global);
			variable_scope = SetScope::GLOBAL;
		}
	}

	Value input = value.CastAs(context.client, option->parameter_type);
	switch (variable_scope) {
	case SetScope::GLOBAL: {
		if (!option->set_global) {
			throw CatalogException("option \"%s\" cannot be set globally", name);
		}
		auto &db = DatabaseInstance::GetDatabase(context.client);
		auto &config = DBConfig::GetConfig(context.client);
		config.SetOption(&db, *option, input);
		break;
	}
	case SetScope::SESSION:
		if (!option->set_local) {
			throw CatalogException("option \"%s\" cannot be set locally", name);
		}
		option->set_local(context.client, input);
		break;
	default:
		throw InternalException("Unsupported SetScope for variable");
	}
}

} // namespace duckdb



namespace duckdb {

PhysicalStreamingLimit::PhysicalStreamingLimit(vector<LogicalType> types, idx_t limit, idx_t offset,
                                               unique_ptr<Expression> limit_expression,
                                               unique_ptr<Expression> offset_expression, idx_t estimated_cardinality,
                                               bool parallel)
    : PhysicalOperator(PhysicalOperatorType::STREAMING_LIMIT, std::move(types), estimated_cardinality),
      limit_value(limit), offset_value(offset), limit_expression(std::move(limit_expression)),
      offset_expression(std::move(offset_expression)), parallel(parallel) {
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
class StreamingLimitOperatorState : public OperatorState {
public:
	explicit StreamingLimitOperatorState(const PhysicalStreamingLimit &op) {
		this->limit = op.limit_expression ? DConstants::INVALID_INDEX : op.limit_value;
		this->offset = op.offset_expression ? DConstants::INVALID_INDEX : op.offset_value;
	}

	idx_t limit;
	idx_t offset;
};

class StreamingLimitGlobalState : public GlobalOperatorState {
public:
	StreamingLimitGlobalState() : current_offset(0) {
	}

	std::atomic<idx_t> current_offset;
};

unique_ptr<OperatorState> PhysicalStreamingLimit::GetOperatorState(ExecutionContext &context) const {
	return make_unique<StreamingLimitOperatorState>(*this);
}

unique_ptr<GlobalOperatorState> PhysicalStreamingLimit::GetGlobalOperatorState(ClientContext &context) const {
	return make_unique<StreamingLimitGlobalState>();
}

OperatorResultType PhysicalStreamingLimit::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                   GlobalOperatorState &gstate_p, OperatorState &state_p) const {
	auto &gstate = (StreamingLimitGlobalState &)gstate_p;
	auto &state = (StreamingLimitOperatorState &)state_p;
	auto &limit = state.limit;
	auto &offset = state.offset;
	idx_t current_offset = gstate.current_offset.fetch_add(input.size());
	idx_t max_element;
	if (!PhysicalLimit::ComputeOffset(context, input, limit, offset, current_offset, max_element,
	                                  limit_expression.get(), offset_expression.get())) {
		return OperatorResultType::FINISHED;
	}
	if (PhysicalLimit::HandleOffset(input, current_offset, offset, limit)) {
		chunk.Reference(input);
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

bool PhysicalStreamingLimit::IsOrderDependent() const {
	return !parallel;
}

bool PhysicalStreamingLimit::ParallelOperator() const {
	return parallel;
}

} // namespace duckdb




namespace duckdb {

PhysicalStreamingSample::PhysicalStreamingSample(vector<LogicalType> types, SampleMethod method, double percentage,
                                                 int64_t seed, idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::STREAMING_SAMPLE, std::move(types), estimated_cardinality), method(method),
      percentage(percentage / 100), seed(seed) {
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
class StreamingSampleOperatorState : public OperatorState {
public:
	explicit StreamingSampleOperatorState(int64_t seed) : random(seed) {
	}

	RandomEngine random;
};

void PhysicalStreamingSample::SystemSample(DataChunk &input, DataChunk &result, OperatorState &state_p) const {
	// system sampling: we throw one dice per chunk
	auto &state = (StreamingSampleOperatorState &)state_p;
	double rand = state.random.NextRandom();
	if (rand <= percentage) {
		// rand is smaller than sample_size: output chunk
		result.Reference(input);
	}
}

void PhysicalStreamingSample::BernoulliSample(DataChunk &input, DataChunk &result, OperatorState &state_p) const {
	// bernoulli sampling: we throw one dice per tuple
	// then slice the result chunk
	auto &state = (StreamingSampleOperatorState &)state_p;
	idx_t result_count = 0;
	SelectionVector sel(STANDARD_VECTOR_SIZE);
	for (idx_t i = 0; i < input.size(); i++) {
		double rand = state.random.NextRandom();
		if (rand <= percentage) {
			sel.set_index(result_count++, i);
		}
	}
	if (result_count > 0) {
		result.Slice(input, sel, result_count);
	}
}

unique_ptr<OperatorState> PhysicalStreamingSample::GetOperatorState(ExecutionContext &context) const {
	return make_unique<StreamingSampleOperatorState>(seed);
}

OperatorResultType PhysicalStreamingSample::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                    GlobalOperatorState &gstate, OperatorState &state) const {
	switch (method) {
	case SampleMethod::BERNOULLI_SAMPLE:
		BernoulliSample(input, chunk, state);
		break;
	case SampleMethod::SYSTEM_SAMPLE:
		SystemSample(input, chunk, state);
		break;
	default:
		throw InternalException("Unsupported sample method for streaming sample");
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

string PhysicalStreamingSample::ParamsToString() const {
	return SampleMethodToString(method) + ": " + to_string(100 * percentage) + "%";
}

} // namespace duckdb




namespace duckdb {

void PhysicalTransaction::GetData(ExecutionContext &context, DataChunk &chunk, GlobalSourceState &gstate,
                                  LocalSourceState &lstate) const {
	auto &client = context.client;

	auto type = info->type;
	if (type == TransactionType::COMMIT && ValidChecker::IsInvalidated(client.ActiveTransaction())) {
		// transaction is invalidated - turn COMMIT into ROLLBACK
		type = TransactionType::ROLLBACK;
	}
	switch (type) {
	case TransactionType::BEGIN_TRANSACTION: {
		if (client.transaction.IsAutoCommit()) {
			// start the active transaction
			// if autocommit is active, we have already called
			// BeginTransaction by setting autocommit to false we
			// prevent it from being closed after this query, hence
			// preserving the transaction context for the next query
			client.transaction.SetAutoCommit(false);
		} else {
			throw TransactionException("cannot start a transaction within a transaction");
		}
		break;
	}
	case TransactionType::COMMIT: {
		if (client.transaction.IsAutoCommit()) {
			throw TransactionException("cannot commit - no transaction is active");
		} else {
			// explicitly commit the current transaction
			client.transaction.Commit();
		}
		break;
	}
	case TransactionType::ROLLBACK: {
		if (client.transaction.IsAutoCommit()) {
			throw TransactionException("cannot rollback - no transaction is active");
		} else {
			// explicitly rollback the current transaction
			client.transaction.Rollback();
		}
		break;
	}
	default:
		throw NotImplementedException("Unrecognized transaction type!");
	}
}

} // namespace duckdb







namespace duckdb {

PhysicalVacuum::PhysicalVacuum(unique_ptr<VacuumInfo> info_p, idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::VACUUM, {LogicalType::BOOLEAN}, estimated_cardinality),
      info(std::move(info_p)) {
}

class VacuumLocalSinkState : public LocalSinkState {
public:
	explicit VacuumLocalSinkState(VacuumInfo &info) {
		for (idx_t col_idx = 0; col_idx < info.columns.size(); col_idx++) {
			column_distinct_stats.push_back(make_unique<DistinctStatistics>());
		}
	};

	vector<unique_ptr<DistinctStatistics>> column_distinct_stats;
};

unique_ptr<LocalSinkState> PhysicalVacuum::GetLocalSinkState(ExecutionContext &context) const {
	return make_unique<VacuumLocalSinkState>(*info);
}

class VacuumGlobalSinkState : public GlobalSinkState {
public:
	explicit VacuumGlobalSinkState(VacuumInfo &info) {
		for (idx_t col_idx = 0; col_idx < info.columns.size(); col_idx++) {
			column_distinct_stats.push_back(make_unique<DistinctStatistics>());
		}
	};

	mutex stats_lock;
	vector<unique_ptr<DistinctStatistics>> column_distinct_stats;
};

unique_ptr<GlobalSinkState> PhysicalVacuum::GetGlobalSinkState(ClientContext &context) const {
	return make_unique<VacuumGlobalSinkState>(*info);
}

SinkResultType PhysicalVacuum::Sink(ExecutionContext &context, GlobalSinkState &gstate_p, LocalSinkState &lstate_p,
                                    DataChunk &input) const {
	auto &lstate = (VacuumLocalSinkState &)lstate_p;
	D_ASSERT(lstate.column_distinct_stats.size() == info->column_id_map.size());

	for (idx_t col_idx = 0; col_idx < input.data.size(); col_idx++) {
		lstate.column_distinct_stats[col_idx]->Update(input.data[col_idx], input.size(), false);
	}

	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalVacuum::Combine(ExecutionContext &context, GlobalSinkState &gstate_p, LocalSinkState &lstate_p) const {
	auto &gstate = (VacuumGlobalSinkState &)gstate_p;
	auto &lstate = (VacuumLocalSinkState &)lstate_p;

	lock_guard<mutex> lock(gstate.stats_lock);
	D_ASSERT(gstate.column_distinct_stats.size() == lstate.column_distinct_stats.size());
	for (idx_t col_idx = 0; col_idx < gstate.column_distinct_stats.size(); col_idx++) {
		gstate.column_distinct_stats[col_idx]->Merge(*lstate.column_distinct_stats[col_idx]);
	}
}

SinkFinalizeType PhysicalVacuum::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                          GlobalSinkState &gstate) const {
	auto &sink = (VacuumGlobalSinkState &)gstate;

	auto table = info->table;
	for (idx_t col_idx = 0; col_idx < sink.column_distinct_stats.size(); col_idx++) {
		table->GetStorage().SetStatistics(info->column_id_map.at(col_idx), [&](BaseStatistics &stats) {
			stats.distinct_stats = std::move(sink.column_distinct_stats[col_idx]);
		});
	}

	return SinkFinalizeType::READY;
}

void PhysicalVacuum::GetData(ExecutionContext &context, DataChunk &chunk, GlobalSourceState &gstate,
                             LocalSourceState &lstate) const {
	// NOP
}

} // namespace duckdb


namespace duckdb {

OuterJoinMarker::OuterJoinMarker(bool enabled_p) : enabled(enabled_p), count(0) {
}

void OuterJoinMarker::Initialize(idx_t count_p) {
	if (!enabled) {
		return;
	}
	this->count = count_p;
	found_match = unique_ptr<bool[]>(new bool[count]);
	Reset();
}

void OuterJoinMarker::Reset() {
	if (!enabled) {
		return;
	}
	memset(found_match.get(), 0, sizeof(bool) * count);
}

void OuterJoinMarker::SetMatch(idx_t position) {
	if (!enabled) {
		return;
	}
	D_ASSERT(position < count);
	found_match[position] = true;
}

void OuterJoinMarker::SetMatches(const SelectionVector &sel, idx_t count, idx_t base_idx) {
	if (!enabled) {
		return;
	}
	for (idx_t i = 0; i < count; i++) {
		auto idx = sel.get_index(i);
		auto pos = base_idx + idx;
		D_ASSERT(pos < this->count);
		found_match[pos] = true;
	}
}

void OuterJoinMarker::ConstructLeftJoinResult(DataChunk &left, DataChunk &result) {
	if (!enabled) {
		return;
	}
	D_ASSERT(count == STANDARD_VECTOR_SIZE);
	SelectionVector remaining_sel(STANDARD_VECTOR_SIZE);
	idx_t remaining_count = 0;
	for (idx_t i = 0; i < left.size(); i++) {
		if (!found_match[i]) {
			remaining_sel.set_index(remaining_count++, i);
		}
	}
	if (remaining_count > 0) {
		result.Slice(left, remaining_sel, remaining_count);
		for (idx_t idx = left.ColumnCount(); idx < result.ColumnCount(); idx++) {
			result.data[idx].SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result.data[idx], true);
		}
	}
}

idx_t OuterJoinMarker::MaxThreads() const {
	return count / (STANDARD_VECTOR_SIZE * 10);
}

void OuterJoinMarker::InitializeScan(ColumnDataCollection &data, OuterJoinGlobalScanState &gstate) {
	gstate.data = &data;
	data.InitializeScan(gstate.global_scan);
}

void OuterJoinMarker::InitializeScan(OuterJoinGlobalScanState &gstate, OuterJoinLocalScanState &lstate) {
	D_ASSERT(gstate.data);
	lstate.match_sel.Initialize(STANDARD_VECTOR_SIZE);
	gstate.data->InitializeScanChunk(lstate.scan_chunk);
}

void OuterJoinMarker::Scan(OuterJoinGlobalScanState &gstate, OuterJoinLocalScanState &lstate, DataChunk &result) {
	D_ASSERT(gstate.data);
	// fill in NULL values for the LHS
	while (gstate.data->Scan(gstate.global_scan, lstate.local_scan, lstate.scan_chunk)) {
		idx_t result_count = 0;
		// figure out which tuples didn't find a match in the RHS
		for (idx_t i = 0; i < lstate.scan_chunk.size(); i++) {
			if (!found_match[lstate.local_scan.current_row_index + i]) {
				lstate.match_sel.set_index(result_count++, i);
			}
		}
		if (result_count > 0) {
			// if there were any tuples that didn't find a match, output them
			idx_t left_column_count = result.ColumnCount() - lstate.scan_chunk.ColumnCount();
			for (idx_t i = 0; i < left_column_count; i++) {
				result.data[i].SetVectorType(VectorType::CONSTANT_VECTOR);
				ConstantVector::SetNull(result.data[i], true);
			}
			for (idx_t col_idx = left_column_count; col_idx < result.ColumnCount(); col_idx++) {
				result.data[col_idx].Slice(lstate.scan_chunk.data[col_idx - left_column_count], lstate.match_sel,
				                           result_count);
			}
			result.SetCardinality(result_count);
			return;
		}
	}
}

} // namespace duckdb





namespace duckdb {

PerfectHashJoinExecutor::PerfectHashJoinExecutor(const PhysicalHashJoin &join_p, JoinHashTable &ht_p,
                                                 PerfectHashJoinStats perfect_join_stats)
    : join(join_p), ht(ht_p), perfect_join_statistics(std::move(perfect_join_stats)) {
}

bool PerfectHashJoinExecutor::CanDoPerfectHashJoin() {
	return perfect_join_statistics.is_build_small;
}

//===--------------------------------------------------------------------===//
// Build
//===--------------------------------------------------------------------===//
bool PerfectHashJoinExecutor::BuildPerfectHashTable(LogicalType &key_type) {
	// First, allocate memory for each build column
	auto build_size = perfect_join_statistics.build_range + 1;
	for (const auto &type : ht.build_types) {
		perfect_hash_table.emplace_back(type, build_size);
	}
	// and for duplicate_checking
	bitmap_build_idx = unique_ptr<bool[]>(new bool[build_size]);
	memset(bitmap_build_idx.get(), 0, sizeof(bool) * build_size); // set false

	// pin all fixed-size blocks (variable-sized should still be pinned)
	ht.PinAllBlocks();

	// Now fill columns with build data
	JoinHTScanState join_ht_state;
	return FullScanHashTable(join_ht_state, key_type);
}

bool PerfectHashJoinExecutor::FullScanHashTable(JoinHTScanState &state, LogicalType &key_type) {
	Vector tuples_addresses(LogicalType::POINTER, ht.Count());              // allocate space for all the tuples
	auto key_locations = FlatVector::GetData<data_ptr_t>(tuples_addresses); // get a pointer to vector data
	// TODO: In a parallel finalize: One should exclusively lock and each thread should do one part of the code below.
	// Go through all the blocks and fill the keys addresses
	auto keys_count = ht.FillWithHTOffsets(key_locations, state);
	// Scan the build keys in the hash table
	Vector build_vector(key_type, keys_count);
	RowOperations::FullScanColumn(ht.layout, tuples_addresses, build_vector, keys_count, 0);
	// Now fill the selection vector using the build keys and create a sequential vector
	// todo: add check for fast pass when probe is part of build domain
	SelectionVector sel_build(keys_count + 1);
	SelectionVector sel_tuples(keys_count + 1);
	bool success = FillSelectionVectorSwitchBuild(build_vector, sel_build, sel_tuples, keys_count);
	// early out
	if (!success) {
		return false;
	}
	if (unique_keys == perfect_join_statistics.build_range + 1 && !ht.has_null) {
		perfect_join_statistics.is_build_dense = true;
	}
	keys_count = unique_keys; // do not consider keys out of the range
	// Full scan the remaining build columns and fill the perfect hash table
	for (idx_t i = 0; i < ht.build_types.size(); i++) {
		auto build_size = perfect_join_statistics.build_range + 1;
		auto &vector = perfect_hash_table[i];
		D_ASSERT(vector.GetType() == ht.build_types[i]);
		const auto col_no = ht.condition_types.size() + i;
		RowOperations::Gather(tuples_addresses, sel_tuples, vector, sel_build, keys_count, ht.layout, col_no,
		                      build_size);
	}
	return true;
}

bool PerfectHashJoinExecutor::FillSelectionVectorSwitchBuild(Vector &source, SelectionVector &sel_vec,
                                                             SelectionVector &seq_sel_vec, idx_t count) {
	switch (source.GetType().InternalType()) {
	case PhysicalType::INT8:
		return TemplatedFillSelectionVectorBuild<int8_t>(source, sel_vec, seq_sel_vec, count);
	case PhysicalType::INT16:
		return TemplatedFillSelectionVectorBuild<int16_t>(source, sel_vec, seq_sel_vec, count);
	case PhysicalType::INT32:
		return TemplatedFillSelectionVectorBuild<int32_t>(source, sel_vec, seq_sel_vec, count);
	case PhysicalType::INT64:
		return TemplatedFillSelectionVectorBuild<int64_t>(source, sel_vec, seq_sel_vec, count);
	case PhysicalType::UINT8:
		return TemplatedFillSelectionVectorBuild<uint8_t>(source, sel_vec, seq_sel_vec, count);
	case PhysicalType::UINT16:
		return TemplatedFillSelectionVectorBuild<uint16_t>(source, sel_vec, seq_sel_vec, count);
	case PhysicalType::UINT32:
		return TemplatedFillSelectionVectorBuild<uint32_t>(source, sel_vec, seq_sel_vec, count);
	case PhysicalType::UINT64:
		return TemplatedFillSelectionVectorBuild<uint64_t>(source, sel_vec, seq_sel_vec, count);
	default:
		throw NotImplementedException("Type not supported for perfect hash join");
	}
}

template <typename T>
bool PerfectHashJoinExecutor::TemplatedFillSelectionVectorBuild(Vector &source, SelectionVector &sel_vec,
                                                                SelectionVector &seq_sel_vec, idx_t count) {
	if (perfect_join_statistics.build_min.IsNull() || perfect_join_statistics.build_max.IsNull()) {
		return false;
	}
	auto min_value = perfect_join_statistics.build_min.GetValueUnsafe<T>();
	auto max_value = perfect_join_statistics.build_max.GetValueUnsafe<T>();
	UnifiedVectorFormat vector_data;
	source.ToUnifiedFormat(count, vector_data);
	auto data = reinterpret_cast<T *>(vector_data.data);
	// generate the selection vector
	for (idx_t i = 0, sel_idx = 0; i < count; ++i) {
		auto data_idx = vector_data.sel->get_index(i);
		auto input_value = data[data_idx];
		// add index to selection vector if value in the range
		if (min_value <= input_value && input_value <= max_value) {
			auto idx = (idx_t)(input_value - min_value); // subtract min value to get the idx position
			sel_vec.set_index(sel_idx, idx);
			if (bitmap_build_idx[idx]) {
				return false;
			} else {
				bitmap_build_idx[idx] = true;
				unique_keys++;
			}
			seq_sel_vec.set_index(sel_idx++, i);
		}
	}
	return true;
}

//===--------------------------------------------------------------------===//
// Probe
//===--------------------------------------------------------------------===//
class PerfectHashJoinState : public OperatorState {
public:
	PerfectHashJoinState(ClientContext &context, const PhysicalHashJoin &join) : probe_executor(context) {
		join_keys.Initialize(Allocator::Get(context), join.condition_types);
		for (auto &cond : join.conditions) {
			probe_executor.AddExpression(*cond.left);
		}
		build_sel_vec.Initialize(STANDARD_VECTOR_SIZE);
		probe_sel_vec.Initialize(STANDARD_VECTOR_SIZE);
		seq_sel_vec.Initialize(STANDARD_VECTOR_SIZE);
	}

	DataChunk join_keys;
	ExpressionExecutor probe_executor;
	SelectionVector build_sel_vec;
	SelectionVector probe_sel_vec;
	SelectionVector seq_sel_vec;
};

unique_ptr<OperatorState> PerfectHashJoinExecutor::GetOperatorState(ExecutionContext &context) {
	auto state = make_unique<PerfectHashJoinState>(context.client, join);
	return std::move(state);
}

OperatorResultType PerfectHashJoinExecutor::ProbePerfectHashTable(ExecutionContext &context, DataChunk &input,
                                                                  DataChunk &result, OperatorState &state_p) {
	auto &state = (PerfectHashJoinState &)state_p;
	// keeps track of how many probe keys have a match
	idx_t probe_sel_count = 0;

	// fetch the join keys from the chunk
	state.join_keys.Reset();
	state.probe_executor.Execute(input, state.join_keys);
	// select the keys that are in the min-max range
	auto &keys_vec = state.join_keys.data[0];
	auto keys_count = state.join_keys.size();
	// todo: add check for fast pass when probe is part of build domain
	FillSelectionVectorSwitchProbe(keys_vec, state.build_sel_vec, state.probe_sel_vec, keys_count, probe_sel_count);

	// If build is dense and probe is in build's domain, just reference probe
	if (perfect_join_statistics.is_build_dense && keys_count == probe_sel_count) {
		result.Reference(input);
	} else {
		// otherwise, filter it out the values that do not match
		result.Slice(input, state.probe_sel_vec, probe_sel_count, 0);
	}
	// on the build side, we need to fetch the data and build dictionary vectors with the sel_vec
	for (idx_t i = 0; i < ht.build_types.size(); i++) {
		auto &result_vector = result.data[input.ColumnCount() + i];
		D_ASSERT(result_vector.GetType() == ht.build_types[i]);
		auto &build_vec = perfect_hash_table[i];
		result_vector.Reference(build_vec);
		result_vector.Slice(state.build_sel_vec, probe_sel_count);
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

void PerfectHashJoinExecutor::FillSelectionVectorSwitchProbe(Vector &source, SelectionVector &build_sel_vec,
                                                             SelectionVector &probe_sel_vec, idx_t count,
                                                             idx_t &probe_sel_count) {
	switch (source.GetType().InternalType()) {
	case PhysicalType::INT8:
		TemplatedFillSelectionVectorProbe<int8_t>(source, build_sel_vec, probe_sel_vec, count, probe_sel_count);
		break;
	case PhysicalType::INT16:
		TemplatedFillSelectionVectorProbe<int16_t>(source, build_sel_vec, probe_sel_vec, count, probe_sel_count);
		break;
	case PhysicalType::INT32:
		TemplatedFillSelectionVectorProbe<int32_t>(source, build_sel_vec, probe_sel_vec, count, probe_sel_count);
		break;
	case PhysicalType::INT64:
		TemplatedFillSelectionVectorProbe<int64_t>(source, build_sel_vec, probe_sel_vec, count, probe_sel_count);
		break;
	case PhysicalType::UINT8:
		TemplatedFillSelectionVectorProbe<uint8_t>(source, build_sel_vec, probe_sel_vec, count, probe_sel_count);
		break;
	case PhysicalType::UINT16:
		TemplatedFillSelectionVectorProbe<uint16_t>(source, build_sel_vec, probe_sel_vec, count, probe_sel_count);
		break;
	case PhysicalType::UINT32:
		TemplatedFillSelectionVectorProbe<uint32_t>(source, build_sel_vec, probe_sel_vec, count, probe_sel_count);
		break;
	case PhysicalType::UINT64:
		TemplatedFillSelectionVectorProbe<uint64_t>(source, build_sel_vec, probe_sel_vec, count, probe_sel_count);
		break;
	default:
		throw NotImplementedException("Type not supported");
	}
}

template <typename T>
void PerfectHashJoinExecutor::TemplatedFillSelectionVectorProbe(Vector &source, SelectionVector &build_sel_vec,
                                                                SelectionVector &probe_sel_vec, idx_t count,
                                                                idx_t &probe_sel_count) {
	auto min_value = perfect_join_statistics.build_min.GetValueUnsafe<T>();
	auto max_value = perfect_join_statistics.build_max.GetValueUnsafe<T>();

	UnifiedVectorFormat vector_data;
	source.ToUnifiedFormat(count, vector_data);
	auto data = reinterpret_cast<T *>(vector_data.data);
	auto validity_mask = &vector_data.validity;
	// build selection vector for non-dense build
	if (validity_mask->AllValid()) {
		for (idx_t i = 0, sel_idx = 0; i < count; ++i) {
			// retrieve value from vector
			auto data_idx = vector_data.sel->get_index(i);
			auto input_value = data[data_idx];
			// add index to selection vector if value in the range
			if (min_value <= input_value && input_value <= max_value) {
				auto idx = (idx_t)(input_value - min_value); // subtract min value to get the idx position
				                                             // check for matches in the build
				if (bitmap_build_idx[idx]) {
					build_sel_vec.set_index(sel_idx, idx);
					probe_sel_vec.set_index(sel_idx++, i);
					probe_sel_count++;
				}
			}
		}
	} else {
		for (idx_t i = 0, sel_idx = 0; i < count; ++i) {
			// retrieve value from vector
			auto data_idx = vector_data.sel->get_index(i);
			if (!validity_mask->RowIsValid(data_idx)) {
				continue;
			}
			auto input_value = data[data_idx];
			// add index to selection vector if value in the range
			if (min_value <= input_value && input_value <= max_value) {
				auto idx = (idx_t)(input_value - min_value); // subtract min value to get the idx position
				                                             // check for matches in the build
				if (bitmap_build_idx[idx]) {
					build_sel_vec.set_index(sel_idx, idx);
					probe_sel_vec.set_index(sel_idx++, i);
					probe_sel_count++;
				}
			}
		}
	}
}

} // namespace duckdb









namespace duckdb {

PhysicalBlockwiseNLJoin::PhysicalBlockwiseNLJoin(LogicalOperator &op, unique_ptr<PhysicalOperator> left,
                                                 unique_ptr<PhysicalOperator> right, unique_ptr<Expression> condition,
                                                 JoinType join_type, idx_t estimated_cardinality)
    : PhysicalJoin(op, PhysicalOperatorType::BLOCKWISE_NL_JOIN, join_type, estimated_cardinality),
      condition(std::move(condition)) {
	children.push_back(std::move(left));
	children.push_back(std::move(right));
	// MARK and SINGLE joins not handled
	D_ASSERT(join_type != JoinType::MARK);
	D_ASSERT(join_type != JoinType::SINGLE);
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class BlockwiseNLJoinLocalState : public LocalSinkState {
public:
	BlockwiseNLJoinLocalState() {
	}
};

class BlockwiseNLJoinGlobalState : public GlobalSinkState {
public:
	explicit BlockwiseNLJoinGlobalState(ClientContext &context, const PhysicalBlockwiseNLJoin &op)
	    : right_chunks(context, op.children[1]->GetTypes()), right_outer(IsRightOuterJoin(op.join_type)) {
	}

	mutex lock;
	ColumnDataCollection right_chunks;
	OuterJoinMarker right_outer;
};

unique_ptr<GlobalSinkState> PhysicalBlockwiseNLJoin::GetGlobalSinkState(ClientContext &context) const {
	return make_unique<BlockwiseNLJoinGlobalState>(context, *this);
}

unique_ptr<LocalSinkState> PhysicalBlockwiseNLJoin::GetLocalSinkState(ExecutionContext &context) const {
	return make_unique<BlockwiseNLJoinLocalState>();
}

SinkResultType PhysicalBlockwiseNLJoin::Sink(ExecutionContext &context, GlobalSinkState &state, LocalSinkState &lstate,
                                             DataChunk &input) const {
	auto &gstate = (BlockwiseNLJoinGlobalState &)state;
	lock_guard<mutex> nl_lock(gstate.lock);
	gstate.right_chunks.Append(input);
	return SinkResultType::NEED_MORE_INPUT;
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//
SinkFinalizeType PhysicalBlockwiseNLJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                   GlobalSinkState &gstate_p) const {
	auto &gstate = (BlockwiseNLJoinGlobalState &)gstate_p;
	gstate.right_outer.Initialize(gstate.right_chunks.Count());

	if (gstate.right_chunks.Count() == 0 && EmptyResultIfRHSIsEmpty()) {
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}
	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
class BlockwiseNLJoinState : public CachingOperatorState {
public:
	explicit BlockwiseNLJoinState(ExecutionContext &context, ColumnDataCollection &rhs,
	                              const PhysicalBlockwiseNLJoin &op)
	    : cross_product(rhs), left_outer(IsLeftOuterJoin(op.join_type)), match_sel(STANDARD_VECTOR_SIZE),
	      executor(context.client, *op.condition) {
		left_outer.Initialize(STANDARD_VECTOR_SIZE);
	}

	CrossProductExecutor cross_product;
	OuterJoinMarker left_outer;
	SelectionVector match_sel;
	ExpressionExecutor executor;
};

unique_ptr<OperatorState> PhysicalBlockwiseNLJoin::GetOperatorState(ExecutionContext &context) const {
	auto &gstate = (BlockwiseNLJoinGlobalState &)*sink_state;
	return make_unique<BlockwiseNLJoinState>(context, gstate.right_chunks, *this);
}

OperatorResultType PhysicalBlockwiseNLJoin::ExecuteInternal(ExecutionContext &context, DataChunk &input,
                                                            DataChunk &chunk, GlobalOperatorState &gstate_p,
                                                            OperatorState &state_p) const {
	D_ASSERT(input.size() > 0);
	auto &state = (BlockwiseNLJoinState &)state_p;
	auto &gstate = (BlockwiseNLJoinGlobalState &)*sink_state;

	if (gstate.right_chunks.Count() == 0) {
		// empty RHS
		if (!EmptyResultIfRHSIsEmpty()) {
			PhysicalComparisonJoin::ConstructEmptyJoinResult(join_type, false, input, chunk);
			return OperatorResultType::NEED_MORE_INPUT;
		} else {
			return OperatorResultType::FINISHED;
		}
	}

	// now perform the actual join
	// we perform a cross product, then execute the expression directly on the cross product' result
	idx_t result_count = 0;
	do {
		auto result = state.cross_product.Execute(input, chunk);
		if (result == OperatorResultType::NEED_MORE_INPUT) {
			// exhausted input, have to pull new LHS chunk
			if (state.left_outer.Enabled()) {
				// left join: before we move to the next chunk, see if we need to output any vectors that didn't
				// have a match found
				state.left_outer.ConstructLeftJoinResult(input, chunk);
				state.left_outer.Reset();
			}
			return OperatorResultType::NEED_MORE_INPUT;
		}

		// now perform the computation
		result_count = state.executor.SelectExpression(chunk, state.match_sel);
		if (result_count > 0) {
			// found a match!
			// check if the cross product is scanning the LHS or the RHS in its entirety
			if (!state.cross_product.ScanLHS()) {
				// set the match flags in the LHS
				state.left_outer.SetMatches(state.match_sel, result_count);
				// set the match flag in the RHS
				gstate.right_outer.SetMatch(state.cross_product.ScanPosition() + state.cross_product.PositionInChunk());
			} else {
				// set the match flag in the LHS
				state.left_outer.SetMatch(state.cross_product.PositionInChunk());
				// set the match flags in the RHS
				gstate.right_outer.SetMatches(state.match_sel, result_count, state.cross_product.ScanPosition());
			}
			chunk.Slice(state.match_sel, result_count);
		} else {
			// no result: reset the chunk
			chunk.Reset();
		}
	} while (result_count == 0);
	return OperatorResultType::HAVE_MORE_OUTPUT;
}

string PhysicalBlockwiseNLJoin::ParamsToString() const {
	string extra_info = JoinTypeToString(join_type) + "\n";
	extra_info += condition->GetName();
	return extra_info;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class BlockwiseNLJoinGlobalScanState : public GlobalSourceState {
public:
	explicit BlockwiseNLJoinGlobalScanState(const PhysicalBlockwiseNLJoin &op) : op(op) {
		D_ASSERT(op.sink_state);
		auto &sink = (BlockwiseNLJoinGlobalState &)*op.sink_state;
		sink.right_outer.InitializeScan(sink.right_chunks, scan_state);
	}

	const PhysicalBlockwiseNLJoin &op;
	OuterJoinGlobalScanState scan_state;

public:
	idx_t MaxThreads() override {
		auto &sink = (BlockwiseNLJoinGlobalState &)*op.sink_state;
		return sink.right_outer.MaxThreads();
	}
};

class BlockwiseNLJoinLocalScanState : public LocalSourceState {
public:
	explicit BlockwiseNLJoinLocalScanState(const PhysicalBlockwiseNLJoin &op, BlockwiseNLJoinGlobalScanState &gstate) {
		D_ASSERT(op.sink_state);
		auto &sink = (BlockwiseNLJoinGlobalState &)*op.sink_state;
		sink.right_outer.InitializeScan(gstate.scan_state, scan_state);
	}

	OuterJoinLocalScanState scan_state;
};

unique_ptr<GlobalSourceState> PhysicalBlockwiseNLJoin::GetGlobalSourceState(ClientContext &context) const {
	return make_unique<BlockwiseNLJoinGlobalScanState>(*this);
}

unique_ptr<LocalSourceState> PhysicalBlockwiseNLJoin::GetLocalSourceState(ExecutionContext &context,
                                                                          GlobalSourceState &gstate) const {
	return make_unique<BlockwiseNLJoinLocalScanState>(*this, (BlockwiseNLJoinGlobalScanState &)gstate);
}

void PhysicalBlockwiseNLJoin::GetData(ExecutionContext &context, DataChunk &chunk, GlobalSourceState &gstate_p,
                                      LocalSourceState &lstate_p) const {
	D_ASSERT(IsRightOuterJoin(join_type));
	// check if we need to scan any unmatched tuples from the RHS for the full/right outer join
	auto &sink = (BlockwiseNLJoinGlobalState &)*sink_state;
	auto &gstate = (BlockwiseNLJoinGlobalScanState &)gstate_p;
	auto &lstate = (BlockwiseNLJoinLocalScanState &)lstate_p;

	// if the LHS is exhausted in a FULL/RIGHT OUTER JOIN, we scan chunks we still need to output
	sink.right_outer.Scan(gstate.scan_state, lstate.scan_state, chunk);
}

} // namespace duckdb



namespace duckdb {

PhysicalComparisonJoin::PhysicalComparisonJoin(LogicalOperator &op, PhysicalOperatorType type,
                                               vector<JoinCondition> conditions_p, JoinType join_type,
                                               idx_t estimated_cardinality)
    : PhysicalJoin(op, type, join_type, estimated_cardinality) {
	conditions.resize(conditions_p.size());
	// we reorder conditions so the ones with COMPARE_EQUAL occur first
	idx_t equal_position = 0;
	idx_t other_position = conditions_p.size() - 1;
	for (idx_t i = 0; i < conditions_p.size(); i++) {
		if (conditions_p[i].comparison == ExpressionType::COMPARE_EQUAL ||
		    conditions_p[i].comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
			// COMPARE_EQUAL and COMPARE_NOT_DISTINCT_FROM, move to the start
			conditions[equal_position++] = std::move(conditions_p[i]);
		} else {
			// other expression, move to the end
			conditions[other_position--] = std::move(conditions_p[i]);
		}
	}
}

string PhysicalComparisonJoin::ParamsToString() const {
	string extra_info = JoinTypeToString(join_type) + "\n";
	for (auto &it : conditions) {
		string op = ExpressionTypeToOperator(it.comparison);
		extra_info += it.left->GetName() + " " + op + " " + it.right->GetName() + "\n";
	}
	extra_info += "\n[INFOSEPARATOR]\n";
	extra_info += StringUtil::Format("EC: %llu\n", estimated_props->GetCardinality<idx_t>());
	extra_info += StringUtil::Format("Cost: %llu", (idx_t)estimated_props->GetCost());
	return extra_info;
}

void PhysicalComparisonJoin::ConstructEmptyJoinResult(JoinType join_type, bool has_null, DataChunk &input,
                                                      DataChunk &result) {
	// empty hash table, special case
	if (join_type == JoinType::ANTI) {
		// anti join with empty hash table, NOP join
		// return the input
		D_ASSERT(input.ColumnCount() == result.ColumnCount());
		result.Reference(input);
	} else if (join_type == JoinType::MARK) {
		// MARK join with empty hash table
		D_ASSERT(join_type == JoinType::MARK);
		D_ASSERT(result.ColumnCount() == input.ColumnCount() + 1);
		auto &result_vector = result.data.back();
		D_ASSERT(result_vector.GetType() == LogicalType::BOOLEAN);
		// for every data vector, we just reference the child chunk
		result.SetCardinality(input);
		for (idx_t i = 0; i < input.ColumnCount(); i++) {
			result.data[i].Reference(input.data[i]);
		}
		// for the MARK vector:
		// if the HT has no NULL values (i.e. empty result set), return a vector that has false for every input
		// entry if the HT has NULL values (i.e. result set had values, but all were NULL), return a vector that
		// has NULL for every input entry
		if (!has_null) {
			auto bool_result = FlatVector::GetData<bool>(result_vector);
			for (idx_t i = 0; i < result.size(); i++) {
				bool_result[i] = false;
			}
		} else {
			FlatVector::Validity(result_vector).SetAllInvalid(result.size());
		}
	} else if (join_type == JoinType::LEFT || join_type == JoinType::OUTER || join_type == JoinType::SINGLE) {
		// LEFT/FULL OUTER/SINGLE join and build side is empty
		// for the LHS we reference the data
		result.SetCardinality(input.size());
		for (idx_t i = 0; i < input.ColumnCount(); i++) {
			result.data[i].Reference(input.data[i]);
		}
		// for the RHS
		for (idx_t k = input.ColumnCount(); k < result.ColumnCount(); k++) {
			result.data[k].SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result.data[k], true);
		}
	}
}
} // namespace duckdb






namespace duckdb {

PhysicalCrossProduct::PhysicalCrossProduct(vector<LogicalType> types, unique_ptr<PhysicalOperator> left,
                                           unique_ptr<PhysicalOperator> right, idx_t estimated_cardinality)
    : CachingPhysicalOperator(PhysicalOperatorType::CROSS_PRODUCT, std::move(types), estimated_cardinality) {
	children.push_back(std::move(left));
	children.push_back(std::move(right));
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class CrossProductGlobalState : public GlobalSinkState {
public:
	explicit CrossProductGlobalState(ClientContext &context, const PhysicalCrossProduct &op)
	    : rhs_materialized(context, op.children[1]->GetTypes()) {
		rhs_materialized.InitializeAppend(append_state);
	}

	ColumnDataCollection rhs_materialized;
	ColumnDataAppendState append_state;
	mutex rhs_lock;
};

unique_ptr<GlobalSinkState> PhysicalCrossProduct::GetGlobalSinkState(ClientContext &context) const {
	return make_unique<CrossProductGlobalState>(context, *this);
}

SinkResultType PhysicalCrossProduct::Sink(ExecutionContext &context, GlobalSinkState &state, LocalSinkState &lstate_p,
                                          DataChunk &input) const {
	auto &sink = (CrossProductGlobalState &)state;
	lock_guard<mutex> client_guard(sink.rhs_lock);
	sink.rhs_materialized.Append(sink.append_state, input);
	return SinkResultType::NEED_MORE_INPUT;
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
CrossProductExecutor::CrossProductExecutor(ColumnDataCollection &rhs)
    : rhs(rhs), position_in_chunk(0), initialized(false), finished(false) {
	rhs.InitializeScanChunk(scan_chunk);
}

void CrossProductExecutor::Reset(DataChunk &input, DataChunk &output) {
	initialized = true;
	finished = false;
	scan_input_chunk = false;
	rhs.InitializeScan(scan_state);
	position_in_chunk = 0;
	scan_chunk.Reset();
}

bool CrossProductExecutor::NextValue(DataChunk &input, DataChunk &output) {
	if (!initialized) {
		// not initialized yet: initialize the scan
		Reset(input, output);
	}
	position_in_chunk++;
	idx_t chunk_size = scan_input_chunk ? input.size() : scan_chunk.size();
	if (position_in_chunk < chunk_size) {
		return true;
	}
	// fetch the next chunk
	rhs.Scan(scan_state, scan_chunk);
	position_in_chunk = 0;
	if (scan_chunk.size() == 0) {
		return false;
	}
	// the way the cross product works is that we keep one chunk constantly referenced
	// while iterating over the other chunk one value at a time
	// the second one is the chunk we are "scanning"

	// for the engine, it is better if we emit larger chunks
	// hence the chunk that we keep constantly referenced should be the larger of the two
	scan_input_chunk = input.size() < scan_chunk.size();
	return true;
}

OperatorResultType CrossProductExecutor::Execute(DataChunk &input, DataChunk &output) {
	if (rhs.Count() == 0) {
		// no RHS: empty result
		return OperatorResultType::FINISHED;
	}
	if (!NextValue(input, output)) {
		// ran out of entries on the RHS
		// reset the RHS and move to the next chunk on the LHS
		initialized = false;
		return OperatorResultType::NEED_MORE_INPUT;
	}

	// set up the constant chunk
	auto &constant_chunk = scan_input_chunk ? scan_chunk : input;
	auto col_count = constant_chunk.ColumnCount();
	auto col_offset = scan_input_chunk ? input.ColumnCount() : 0;
	output.SetCardinality(constant_chunk.size());
	for (idx_t i = 0; i < col_count; i++) {
		output.data[col_offset + i].Reference(constant_chunk.data[i]);
	}

	// for the chunk that we are scanning, scan a single value from that chunk
	auto &scan = scan_input_chunk ? input : scan_chunk;
	col_count = scan.ColumnCount();
	col_offset = scan_input_chunk ? 0 : input.ColumnCount();
	for (idx_t i = 0; i < col_count; i++) {
		ConstantVector::Reference(output.data[col_offset + i], scan.data[i], position_in_chunk, scan.size());
	}
	return OperatorResultType::HAVE_MORE_OUTPUT;
}

class CrossProductOperatorState : public CachingOperatorState {
public:
	explicit CrossProductOperatorState(ColumnDataCollection &rhs) : executor(rhs) {
	}

	CrossProductExecutor executor;
};

unique_ptr<OperatorState> PhysicalCrossProduct::GetOperatorState(ExecutionContext &context) const {
	auto &sink = (CrossProductGlobalState &)*sink_state;
	return make_unique<CrossProductOperatorState>(sink.rhs_materialized);
}

OperatorResultType PhysicalCrossProduct::ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                         GlobalOperatorState &gstate, OperatorState &state_p) const {
	auto &state = (CrossProductOperatorState &)state_p;
	return state.executor.Execute(input, chunk);
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void PhysicalCrossProduct::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	PhysicalJoin::BuildJoinPipelines(current, meta_pipeline, *this);
}

vector<const PhysicalOperator *> PhysicalCrossProduct::GetSources() const {
	return children[0]->GetSources();
}

} // namespace duckdb











namespace duckdb {

PhysicalDelimJoin::PhysicalDelimJoin(vector<LogicalType> types, unique_ptr<PhysicalOperator> original_join,
                                     vector<PhysicalOperator *> delim_scans, idx_t estimated_cardinality)
    : PhysicalOperator(PhysicalOperatorType::DELIM_JOIN, std::move(types), estimated_cardinality),
      join(std::move(original_join)), delim_scans(std::move(delim_scans)) {
	D_ASSERT(join->children.size() == 2);
	// now for the original join
	// we take its left child, this is the side that we will duplicate eliminate
	children.push_back(std::move(join->children[0]));

	// we replace it with a PhysicalColumnDataScan, that scans the ColumnDataCollection that we keep cached
	// the actual chunk collection to scan will be created in the DelimJoinGlobalState
	auto cached_chunk_scan = make_unique<PhysicalColumnDataScan>(
	    children[0]->GetTypes(), PhysicalOperatorType::COLUMN_DATA_SCAN, estimated_cardinality);
	join->children[0] = std::move(cached_chunk_scan);
}

vector<PhysicalOperator *> PhysicalDelimJoin::GetChildren() const {
	vector<PhysicalOperator *> result;
	for (auto &child : children) {
		result.push_back(child.get());
	}
	result.push_back(join.get());
	result.push_back(distinct.get());
	return result;
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class DelimJoinGlobalState : public GlobalSinkState {
public:
	explicit DelimJoinGlobalState(ClientContext &context, const PhysicalDelimJoin &delim_join)
	    : lhs_data(context, delim_join.children[0]->GetTypes()) {
		D_ASSERT(delim_join.delim_scans.size() > 0);
		// set up the delim join chunk to scan in the original join
		auto &cached_chunk_scan = (PhysicalColumnDataScan &)*delim_join.join->children[0];
		cached_chunk_scan.collection = &lhs_data;
	}

	ColumnDataCollection lhs_data;
	mutex lhs_lock;

	void Merge(ColumnDataCollection &input) {
		lock_guard<mutex> guard(lhs_lock);
		lhs_data.Combine(input);
	}
};

class DelimJoinLocalState : public LocalSinkState {
public:
	explicit DelimJoinLocalState(ClientContext &context, const PhysicalDelimJoin &delim_join)
	    : lhs_data(context, delim_join.children[0]->GetTypes()) {
		lhs_data.InitializeAppend(append_state);
	}

	unique_ptr<LocalSinkState> distinct_state;
	ColumnDataCollection lhs_data;
	ColumnDataAppendState append_state;

	void Append(DataChunk &input) {
		lhs_data.Append(input);
	}
};

unique_ptr<GlobalSinkState> PhysicalDelimJoin::GetGlobalSinkState(ClientContext &context) const {
	auto state = make_unique<DelimJoinGlobalState>(context, *this);
	distinct->sink_state = distinct->GetGlobalSinkState(context);
	if (delim_scans.size() > 1) {
		PhysicalHashAggregate::SetMultiScan(*distinct->sink_state);
	}
	return std::move(state);
}

unique_ptr<LocalSinkState> PhysicalDelimJoin::GetLocalSinkState(ExecutionContext &context) const {
	auto state = make_unique<DelimJoinLocalState>(context.client, *this);
	state->distinct_state = distinct->GetLocalSinkState(context);
	return std::move(state);
}

SinkResultType PhysicalDelimJoin::Sink(ExecutionContext &context, GlobalSinkState &state_p, LocalSinkState &lstate_p,
                                       DataChunk &input) const {
	auto &lstate = (DelimJoinLocalState &)lstate_p;
	lstate.lhs_data.Append(lstate.append_state, input);
	distinct->Sink(context, *distinct->sink_state, *lstate.distinct_state, input);
	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalDelimJoin::Combine(ExecutionContext &context, GlobalSinkState &state, LocalSinkState &lstate_p) const {
	auto &lstate = (DelimJoinLocalState &)lstate_p;
	auto &gstate = (DelimJoinGlobalState &)state;
	gstate.Merge(lstate.lhs_data);
	distinct->Combine(context, *distinct->sink_state, *lstate.distinct_state);
}

SinkFinalizeType PhysicalDelimJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &client,
                                             GlobalSinkState &gstate) const {
	// finalize the distinct HT
	D_ASSERT(distinct);
	distinct->Finalize(pipeline, event, client, *distinct->sink_state);
	return SinkFinalizeType::READY;
}

string PhysicalDelimJoin::ParamsToString() const {
	return join->ParamsToString();
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void PhysicalDelimJoin::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	op_state.reset();
	sink_state.reset();

	auto child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, this);
	child_meta_pipeline->Build(children[0].get());

	if (type == PhysicalOperatorType::DELIM_JOIN) {
		// recurse into the actual join
		// any pipelines in there depend on the main pipeline
		// any scan of the duplicate eliminated data on the RHS depends on this pipeline
		// we add an entry to the mapping of (PhysicalOperator*) -> (Pipeline*)
		auto &state = meta_pipeline.GetState();
		for (auto &delim_scan : delim_scans) {
			state.delim_join_dependencies[delim_scan] = child_meta_pipeline->GetBasePipeline().get();
		}
		join->BuildPipelines(current, meta_pipeline);
	}
}

} // namespace duckdb















namespace duckdb {

PhysicalHashJoin::PhysicalHashJoin(LogicalOperator &op, unique_ptr<PhysicalOperator> left,
                                   unique_ptr<PhysicalOperator> right, vector<JoinCondition> cond, JoinType join_type,
                                   const vector<idx_t> &left_projection_map,
                                   const vector<idx_t> &right_projection_map_p, vector<LogicalType> delim_types,
                                   idx_t estimated_cardinality, PerfectHashJoinStats perfect_join_stats)
    : PhysicalComparisonJoin(op, PhysicalOperatorType::HASH_JOIN, std::move(cond), join_type, estimated_cardinality),
      right_projection_map(right_projection_map_p), delim_types(std::move(delim_types)),
      perfect_join_statistics(std::move(perfect_join_stats)) {

	children.push_back(std::move(left));
	children.push_back(std::move(right));

	D_ASSERT(left_projection_map.empty());
	for (auto &condition : conditions) {
		condition_types.push_back(condition.left->return_type);
	}

	// for ANTI, SEMI and MARK join, we only need to store the keys, so for these the build types are empty
	if (join_type != JoinType::ANTI && join_type != JoinType::SEMI && join_type != JoinType::MARK) {
		build_types = LogicalOperator::MapTypes(children[1]->GetTypes(), right_projection_map);
	}
}

PhysicalHashJoin::PhysicalHashJoin(LogicalOperator &op, unique_ptr<PhysicalOperator> left,
                                   unique_ptr<PhysicalOperator> right, vector<JoinCondition> cond, JoinType join_type,
                                   idx_t estimated_cardinality, PerfectHashJoinStats perfect_join_state)
    : PhysicalHashJoin(op, std::move(left), std::move(right), std::move(cond), join_type, {}, {}, {},
                       estimated_cardinality, std::move(perfect_join_state)) {
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class HashJoinGlobalSinkState : public GlobalSinkState {
public:
	HashJoinGlobalSinkState(const PhysicalHashJoin &op, ClientContext &context)
	    : finalized(false), scanned_data(false) {
		hash_table = op.InitializeHashTable(context);

		// for perfect hash join
		perfect_join_executor = make_unique<PerfectHashJoinExecutor>(op, *hash_table, op.perfect_join_statistics);
		// for external hash join
		external = op.can_go_external && ClientConfig::GetConfig(context).force_external;
		// memory usage per thread scales with max mem / num threads
		double max_memory = BufferManager::GetBufferManager(context).GetMaxMemory();
		double num_threads = TaskScheduler::GetScheduler(context).NumberOfThreads();
		// HT may not exceed 60% of memory
		max_ht_size = max_memory * 0.6;
		sink_memory_per_thread = max_ht_size / num_threads;
		// Set probe types
		const auto &payload_types = op.children[0]->types;
		probe_types.insert(probe_types.end(), op.condition_types.begin(), op.condition_types.end());
		probe_types.insert(probe_types.end(), payload_types.begin(), payload_types.end());
		probe_types.emplace_back(LogicalType::HASH);
	}

	void ScheduleFinalize(Pipeline &pipeline, Event &event);
	void InitializeProbeSpill(ClientContext &context);

public:
	//! Global HT used by the join
	unique_ptr<JoinHashTable> hash_table;
	//! The perfect hash join executor (if any)
	unique_ptr<PerfectHashJoinExecutor> perfect_join_executor;
	//! Whether or not the hash table has been finalized
	bool finalized = false;

	//! Whether we are doing an external join
	bool external;
	//! Memory usage per thread during the Sink and Execute phases
	idx_t max_ht_size;
	idx_t sink_memory_per_thread;

	//! Hash tables built by each thread
	mutex lock;
	vector<unique_ptr<JoinHashTable>> local_hash_tables;

	//! Excess probe data gathered during Sink
	vector<LogicalType> probe_types;
	unique_ptr<JoinHashTable::ProbeSpill> probe_spill;

	//! Whether or not we have started scanning data using GetData
	atomic<bool> scanned_data;
};

class HashJoinLocalSinkState : public LocalSinkState {
public:
	HashJoinLocalSinkState(const PhysicalHashJoin &op, ClientContext &context) : build_executor(context) {
		auto &allocator = Allocator::Get(context);
		if (!op.right_projection_map.empty()) {
			build_chunk.Initialize(allocator, op.build_types);
		}
		for (auto &cond : op.conditions) {
			build_executor.AddExpression(*cond.right);
		}
		join_keys.Initialize(allocator, op.condition_types);

		hash_table = op.InitializeHashTable(context);
	}

public:
	DataChunk build_chunk;
	DataChunk join_keys;
	ExpressionExecutor build_executor;

	//! Thread-local HT
	unique_ptr<JoinHashTable> hash_table;
};

unique_ptr<JoinHashTable> PhysicalHashJoin::InitializeHashTable(ClientContext &context) const {
	auto result =
	    make_unique<JoinHashTable>(BufferManager::GetBufferManager(context), conditions, build_types, join_type);
	if (!delim_types.empty() && join_type == JoinType::MARK) {
		// correlated MARK join
		if (delim_types.size() + 1 == conditions.size()) {
			// the correlated MARK join has one more condition than the amount of correlated columns
			// this is the case in a correlated ANY() expression
			// in this case we need to keep track of additional entries, namely:
			// - (1) the total amount of elements per group
			// - (2) the amount of non-null elements per group
			// we need these to correctly deal with the cases of either:
			// - (1) the group being empty [in which case the result is always false, even if the comparison is NULL]
			// - (2) the group containing a NULL value [in which case FALSE becomes NULL]
			auto &info = result->correlated_mark_join_info;

			vector<LogicalType> payload_types;
			vector<BoundAggregateExpression *> correlated_aggregates;
			unique_ptr<BoundAggregateExpression> aggr;

			// jury-rigging the GroupedAggregateHashTable
			// we need a count_star and a count to get counts with and without NULLs

			FunctionBinder function_binder(context);
			aggr = function_binder.BindAggregateFunction(CountStarFun::GetFunction(), {}, nullptr,
			                                             AggregateType::NON_DISTINCT);
			correlated_aggregates.push_back(&*aggr);
			payload_types.push_back(aggr->return_type);
			info.correlated_aggregates.push_back(std::move(aggr));

			auto count_fun = CountFun::GetFunction();
			vector<unique_ptr<Expression>> children;
			// this is a dummy but we need it to make the hash table understand whats going on
			children.push_back(make_unique_base<Expression, BoundReferenceExpression>(count_fun.return_type, 0));
			aggr = function_binder.BindAggregateFunction(count_fun, std::move(children), nullptr,
			                                             AggregateType::NON_DISTINCT);
			correlated_aggregates.push_back(&*aggr);
			payload_types.push_back(aggr->return_type);
			info.correlated_aggregates.push_back(std::move(aggr));

			auto &allocator = Allocator::Get(context);
			info.correlated_counts = make_unique<GroupedAggregateHashTable>(context, allocator, delim_types,
			                                                                payload_types, correlated_aggregates);
			info.correlated_types = delim_types;
			info.group_chunk.Initialize(allocator, delim_types);
			info.result_chunk.Initialize(allocator, payload_types);
		}
	}
	return result;
}

unique_ptr<GlobalSinkState> PhysicalHashJoin::GetGlobalSinkState(ClientContext &context) const {
	return make_unique<HashJoinGlobalSinkState>(*this, context);
}

unique_ptr<LocalSinkState> PhysicalHashJoin::GetLocalSinkState(ExecutionContext &context) const {
	return make_unique<HashJoinLocalSinkState>(*this, context.client);
}

SinkResultType PhysicalHashJoin::Sink(ExecutionContext &context, GlobalSinkState &gstate_p, LocalSinkState &lstate_p,
                                      DataChunk &input) const {
	auto &gstate = (HashJoinGlobalSinkState &)gstate_p;
	auto &lstate = (HashJoinLocalSinkState &)lstate_p;

	// resolve the join keys for the right chunk
	lstate.join_keys.Reset();
	lstate.build_executor.Execute(input, lstate.join_keys);
	// build the HT
	auto &ht = *lstate.hash_table;
	if (!right_projection_map.empty()) {
		// there is a projection map: fill the build chunk with the projected columns
		lstate.build_chunk.Reset();
		lstate.build_chunk.SetCardinality(input);
		for (idx_t i = 0; i < right_projection_map.size(); i++) {
			lstate.build_chunk.data[i].Reference(input.data[right_projection_map[i]]);
		}
		ht.Build(lstate.join_keys, lstate.build_chunk);
	} else if (!build_types.empty()) {
		// there is not a projected map: place the entire right chunk in the HT
		ht.Build(lstate.join_keys, input);
	} else {
		// there are only keys: place an empty chunk in the payload
		lstate.build_chunk.SetCardinality(input.size());
		ht.Build(lstate.join_keys, lstate.build_chunk);
	}

	// swizzle if we reach memory limit
	auto approx_ptr_table_size = ht.Count() * 3 * sizeof(data_ptr_t);
	if (can_go_external && ht.SizeInBytes() + approx_ptr_table_size >= gstate.sink_memory_per_thread) {
		lstate.hash_table->SwizzleBlocks();
		gstate.external = true;
	}

	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalHashJoin::Combine(ExecutionContext &context, GlobalSinkState &gstate_p, LocalSinkState &lstate_p) const {
	auto &gstate = (HashJoinGlobalSinkState &)gstate_p;
	auto &lstate = (HashJoinLocalSinkState &)lstate_p;
	if (lstate.hash_table) {
		lock_guard<mutex> local_ht_lock(gstate.lock);
		gstate.local_hash_tables.push_back(std::move(lstate.hash_table));
	}
	auto &client_profiler = QueryProfiler::Get(context.client);
	context.thread.profiler.Flush(this, &lstate.build_executor, "build_executor", 1);
	client_profiler.Flush(context.thread.profiler);
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//
class HashJoinFinalizeTask : public ExecutorTask {
public:
	HashJoinFinalizeTask(shared_ptr<Event> event_p, ClientContext &context, HashJoinGlobalSinkState &sink,
	                     idx_t block_idx_start, idx_t block_idx_end, bool parallel)
	    : ExecutorTask(context), event(std::move(event_p)), sink(sink), block_idx_start(block_idx_start),
	      block_idx_end(block_idx_end), parallel(parallel) {
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		sink.hash_table->Finalize(block_idx_start, block_idx_end, parallel);
		event->FinishTask();
		return TaskExecutionResult::TASK_FINISHED;
	}

private:
	shared_ptr<Event> event;
	HashJoinGlobalSinkState &sink;
	idx_t block_idx_start;
	idx_t block_idx_end;
	bool parallel;
};

class HashJoinFinalizeEvent : public BasePipelineEvent {
public:
	HashJoinFinalizeEvent(Pipeline &pipeline_p, HashJoinGlobalSinkState &sink)
	    : BasePipelineEvent(pipeline_p), sink(sink) {
	}

	HashJoinGlobalSinkState &sink;

public:
	void Schedule() override {
		auto &context = pipeline->GetClientContext();

		vector<unique_ptr<Task>> finalize_tasks;
		auto &ht = *sink.hash_table;
		const auto &block_collection = ht.GetBlockCollection();
		const auto &blocks = block_collection.blocks;
		const auto num_blocks = blocks.size();
		if (block_collection.count < PARALLEL_CONSTRUCT_THRESHOLD && !context.config.verify_parallelism) {
			// Single-threaded finalize
			finalize_tasks.push_back(
			    make_unique<HashJoinFinalizeTask>(shared_from_this(), context, sink, 0, num_blocks, false));
		} else {
			// Parallel finalize
			idx_t num_threads = TaskScheduler::GetScheduler(context).NumberOfThreads();
			auto blocks_per_thread = MaxValue<idx_t>((num_blocks + num_threads - 1) / num_threads, 1);

			idx_t block_idx = 0;
			for (idx_t thread_idx = 0; thread_idx < num_threads; thread_idx++) {
				auto block_idx_start = block_idx;
				auto block_idx_end = MinValue<idx_t>(block_idx_start + blocks_per_thread, num_blocks);
				finalize_tasks.push_back(make_unique<HashJoinFinalizeTask>(shared_from_this(), context, sink,
				                                                           block_idx_start, block_idx_end, true));
				block_idx = block_idx_end;
				if (block_idx == num_blocks) {
					break;
				}
			}
		}
		SetTasks(std::move(finalize_tasks));
	}

	void FinishEvent() override {
		sink.hash_table->finalized = true;
	}

	static constexpr const idx_t PARALLEL_CONSTRUCT_THRESHOLD = 1048576;
};

void HashJoinGlobalSinkState::ScheduleFinalize(Pipeline &pipeline, Event &event) {
	if (hash_table->Count() == 0) {
		hash_table->finalized = true;
		return;
	}
	hash_table->InitializePointerTable();
	auto new_event = make_shared<HashJoinFinalizeEvent>(pipeline, *this);
	event.InsertEvent(std::move(new_event));
}

void HashJoinGlobalSinkState::InitializeProbeSpill(ClientContext &context) {
	lock_guard<mutex> guard(lock);
	if (!probe_spill) {
		probe_spill = make_unique<JoinHashTable::ProbeSpill>(*hash_table, context, probe_types);
	}
}

class HashJoinPartitionTask : public ExecutorTask {
public:
	HashJoinPartitionTask(shared_ptr<Event> event_p, ClientContext &context, JoinHashTable &global_ht,
	                      JoinHashTable &local_ht)
	    : ExecutorTask(context), event(std::move(event_p)), global_ht(global_ht), local_ht(local_ht) {
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		local_ht.Partition(global_ht);
		event->FinishTask();
		return TaskExecutionResult::TASK_FINISHED;
	}

private:
	shared_ptr<Event> event;

	JoinHashTable &global_ht;
	JoinHashTable &local_ht;
};

class HashJoinPartitionEvent : public BasePipelineEvent {
public:
	HashJoinPartitionEvent(Pipeline &pipeline_p, HashJoinGlobalSinkState &sink,
	                       vector<unique_ptr<JoinHashTable>> &local_hts)
	    : BasePipelineEvent(pipeline_p), sink(sink), local_hts(local_hts) {
	}

	HashJoinGlobalSinkState &sink;
	vector<unique_ptr<JoinHashTable>> &local_hts;

public:
	void Schedule() override {
		auto &context = pipeline->GetClientContext();
		vector<unique_ptr<Task>> partition_tasks;
		partition_tasks.reserve(local_hts.size());
		for (auto &local_ht : local_hts) {
			partition_tasks.push_back(
			    make_unique<HashJoinPartitionTask>(shared_from_this(), context, *sink.hash_table, *local_ht));
		}
		SetTasks(std::move(partition_tasks));
	}

	void FinishEvent() override {
		local_hts.clear();
		sink.hash_table->PrepareExternalFinalize();
		sink.ScheduleFinalize(*pipeline, *this);
	}
};

SinkFinalizeType PhysicalHashJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                            GlobalSinkState &gstate) const {
	auto &sink = (HashJoinGlobalSinkState &)gstate;

	if (sink.external) {
		D_ASSERT(can_go_external);
		// External join - partition HT
		sink.perfect_join_executor.reset();
		sink.hash_table->ComputePartitionSizes(context.config, sink.local_hash_tables, sink.max_ht_size);
		auto new_event = make_shared<HashJoinPartitionEvent>(pipeline, sink, sink.local_hash_tables);
		event.InsertEvent(std::move(new_event));
		sink.finalized = true;
		return SinkFinalizeType::READY;
	} else {
		for (auto &local_ht : sink.local_hash_tables) {
			sink.hash_table->Merge(*local_ht);
		}
		sink.local_hash_tables.clear();
	}

	// check for possible perfect hash table
	auto use_perfect_hash = sink.perfect_join_executor->CanDoPerfectHashJoin();
	if (use_perfect_hash) {
		D_ASSERT(sink.hash_table->equality_types.size() == 1);
		auto key_type = sink.hash_table->equality_types[0];
		use_perfect_hash = sink.perfect_join_executor->BuildPerfectHashTable(key_type);
	}
	// In case of a large build side or duplicates, use regular hash join
	if (!use_perfect_hash) {
		sink.perfect_join_executor.reset();
		sink.ScheduleFinalize(pipeline, event);
	}
	sink.finalized = true;
	if (sink.hash_table->Count() == 0 && EmptyResultIfRHSIsEmpty()) {
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}
	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
class HashJoinOperatorState : public CachingOperatorState {
public:
	explicit HashJoinOperatorState(ClientContext &context) : probe_executor(context), initialized(false) {
	}

	DataChunk join_keys;
	ExpressionExecutor probe_executor;
	unique_ptr<JoinHashTable::ScanStructure> scan_structure;
	unique_ptr<OperatorState> perfect_hash_join_state;

	bool initialized;
	JoinHashTable::ProbeSpillLocalAppendState spill_state;
	//! Chunk to sink data into for external join
	DataChunk spill_chunk;

public:
	void Finalize(PhysicalOperator *op, ExecutionContext &context) override {
		context.thread.profiler.Flush(op, &probe_executor, "probe_executor", 0);
	}
};

unique_ptr<OperatorState> PhysicalHashJoin::GetOperatorState(ExecutionContext &context) const {
	auto &allocator = Allocator::Get(context.client);
	auto &sink = (HashJoinGlobalSinkState &)*sink_state;
	auto state = make_unique<HashJoinOperatorState>(context.client);
	if (sink.perfect_join_executor) {
		state->perfect_hash_join_state = sink.perfect_join_executor->GetOperatorState(context);
	} else {
		state->join_keys.Initialize(allocator, condition_types);
		for (auto &cond : conditions) {
			state->probe_executor.AddExpression(*cond.left);
		}
	}
	if (sink.external) {
		state->spill_chunk.Initialize(allocator, sink.probe_types);
		sink.InitializeProbeSpill(context.client);
	}

	return std::move(state);
}

OperatorResultType PhysicalHashJoin::ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                     GlobalOperatorState &gstate, OperatorState &state_p) const {
	auto &state = (HashJoinOperatorState &)state_p;
	auto &sink = (HashJoinGlobalSinkState &)*sink_state;
	D_ASSERT(sink.finalized);
	D_ASSERT(!sink.scanned_data);

	// some initialization for external hash join
	if (sink.external && !state.initialized) {
		if (!sink.probe_spill) {
			sink.InitializeProbeSpill(context.client);
		}
		state.spill_state = sink.probe_spill->RegisterThread();
		state.initialized = true;
	}

	if (sink.hash_table->Count() == 0 && EmptyResultIfRHSIsEmpty()) {
		return OperatorResultType::FINISHED;
	}

	if (sink.perfect_join_executor) {
		D_ASSERT(!sink.external);
		return sink.perfect_join_executor->ProbePerfectHashTable(context, input, chunk, *state.perfect_hash_join_state);
	}

	if (state.scan_structure) {
		// still have elements remaining (i.e. we got >STANDARD_VECTOR_SIZE elements in the previous probe)
		state.scan_structure->Next(state.join_keys, input, chunk);
		if (chunk.size() > 0) {
			return OperatorResultType::HAVE_MORE_OUTPUT;
		}
		state.scan_structure = nullptr;
		return OperatorResultType::NEED_MORE_INPUT;
	}

	// probe the HT
	if (sink.hash_table->Count() == 0) {
		ConstructEmptyJoinResult(sink.hash_table->join_type, sink.hash_table->has_null, input, chunk);
		return OperatorResultType::NEED_MORE_INPUT;
	}

	// resolve the join keys for the left chunk
	state.join_keys.Reset();
	state.probe_executor.Execute(input, state.join_keys);

	// perform the actual probe
	if (sink.external) {
		state.scan_structure = sink.hash_table->ProbeAndSpill(state.join_keys, input, *sink.probe_spill,
		                                                      state.spill_state, state.spill_chunk);
	} else {
		state.scan_structure = sink.hash_table->Probe(state.join_keys);
	}
	state.scan_structure->Next(state.join_keys, input, chunk);
	return OperatorResultType::HAVE_MORE_OUTPUT;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
enum class HashJoinSourceStage : uint8_t { INIT, BUILD, PROBE, SCAN_HT, DONE };

class HashJoinLocalSourceState;

class HashJoinGlobalSourceState : public GlobalSourceState {
public:
	HashJoinGlobalSourceState(const PhysicalHashJoin &op, ClientContext &context);

	//! Initialize this source state using the info in the sink
	void Initialize(ClientContext &context, HashJoinGlobalSinkState &sink);
	//! Try to prepare the next stage
	void TryPrepareNextStage(HashJoinGlobalSinkState &sink);
	//! Prepare the next build/probe stage for external hash join (must hold lock)
	void PrepareBuild(HashJoinGlobalSinkState &sink);
	void PrepareProbe(HashJoinGlobalSinkState &sink);
	//! Assigns a task to a local source state
	bool AssignTask(HashJoinGlobalSinkState &sink, HashJoinLocalSourceState &lstate);

	idx_t MaxThreads() override {
		return probe_count / ((idx_t)STANDARD_VECTOR_SIZE * parallel_scan_chunk_count);
	}

public:
	const PhysicalHashJoin &op;

	//! For synchronizing the external hash join
	atomic<HashJoinSourceStage> global_stage;
	mutex lock;

	//! For HT build synchronization
	idx_t build_block_idx;
	idx_t build_block_count;
	idx_t build_block_done;
	idx_t build_blocks_per_thread;

	//! For probe synchronization
	idx_t probe_chunk_count;
	idx_t probe_chunk_done;

	//! For full/outer synchronization
	JoinHTScanState full_outer_scan;

	//! To determine the number of threads
	idx_t probe_count;
	idx_t parallel_scan_chunk_count;
};

class HashJoinLocalSourceState : public LocalSourceState {
public:
	HashJoinLocalSourceState(const PhysicalHashJoin &op, Allocator &allocator);

	//! Do the work this thread has been assigned
	void ExecuteTask(HashJoinGlobalSinkState &sink, HashJoinGlobalSourceState &gstate, DataChunk &chunk);
	//! Whether this thread has finished the work it has been assigned
	bool TaskFinished();
	//! Build, probe and scan for external hash join
	void ExternalBuild(HashJoinGlobalSinkState &sink, HashJoinGlobalSourceState &gstate);
	void ExternalProbe(HashJoinGlobalSinkState &sink, HashJoinGlobalSourceState &gstate, DataChunk &chunk);
	void ExternalScanHT(HashJoinGlobalSinkState &sink, HashJoinGlobalSourceState &gstate, DataChunk &chunk);

	//! Scans the HT for full/outer join
	void ScanFullOuter(HashJoinGlobalSinkState &sink, HashJoinGlobalSourceState &gstate);

public:
	//! The stage that this thread was assigned work for
	HashJoinSourceStage local_stage;
	//! Vector with pointers here so we don't have to re-initialize
	Vector addresses;

	//! Blocks assigned to this thread for building the pointer table
	idx_t build_block_idx_start;
	idx_t build_block_idx_end;

	//! Local scan state for probe spill
	ColumnDataConsumerScanState probe_local_scan;
	//! Chunks for holding the scanned probe collection
	DataChunk probe_chunk;
	DataChunk join_keys;
	DataChunk payload;
	//! Column indices to easily reference the join keys/payload columns in probe_chunk
	vector<idx_t> join_key_indices;
	vector<idx_t> payload_indices;
	//! Scan structure for the external probe
	unique_ptr<JoinHashTable::ScanStructure> scan_structure;

	//! Current number of tuples from a full/outer scan that are 'in-flight'
	idx_t full_outer_found_entries;
	idx_t full_outer_in_progress;
};

unique_ptr<GlobalSourceState> PhysicalHashJoin::GetGlobalSourceState(ClientContext &context) const {
	return make_unique<HashJoinGlobalSourceState>(*this, context);
}

unique_ptr<LocalSourceState> PhysicalHashJoin::GetLocalSourceState(ExecutionContext &context,
                                                                   GlobalSourceState &gstate) const {
	return make_unique<HashJoinLocalSourceState>(*this, Allocator::Get(context.client));
}

HashJoinGlobalSourceState::HashJoinGlobalSourceState(const PhysicalHashJoin &op, ClientContext &context)
    : op(op), global_stage(HashJoinSourceStage::INIT), probe_chunk_count(0), probe_chunk_done(0),
      probe_count(op.children[0]->estimated_cardinality),
      parallel_scan_chunk_count(context.config.verify_parallelism ? 1 : 120) {
}

void HashJoinGlobalSourceState::Initialize(ClientContext &context, HashJoinGlobalSinkState &sink) {
	lock_guard<mutex> init_lock(lock);
	if (global_stage != HashJoinSourceStage::INIT) {
		// Another thread initialized
		return;
	}
	full_outer_scan.total = sink.hash_table->Count();

	idx_t num_blocks = sink.hash_table->GetBlockCollection().blocks.size();
	idx_t num_threads = TaskScheduler::GetScheduler(context).NumberOfThreads();
	build_blocks_per_thread = MaxValue<idx_t>((num_blocks + num_threads - 1) / num_threads, 1);

	// Finalize the probe spill too
	if (sink.probe_spill) {
		sink.probe_spill->Finalize();
	}

	global_stage = HashJoinSourceStage::PROBE;
}

void HashJoinGlobalSourceState::TryPrepareNextStage(HashJoinGlobalSinkState &sink) {
	lock_guard<mutex> guard(lock);
	switch (global_stage.load()) {
	case HashJoinSourceStage::BUILD:
		if (build_block_done == build_block_count) {
			sink.hash_table->finalized = true;
			PrepareProbe(sink);
		}
		break;
	case HashJoinSourceStage::PROBE:
		if (probe_chunk_done == probe_chunk_count) {
			if (IsRightOuterJoin(op.join_type)) {
				global_stage = HashJoinSourceStage::SCAN_HT;
			} else {
				PrepareBuild(sink);
			}
		}
		break;
	case HashJoinSourceStage::SCAN_HT:
		if (full_outer_scan.scanned == full_outer_scan.total) {
			PrepareBuild(sink);
		}
		break;
	default:
		break;
	}
}

void HashJoinGlobalSourceState::PrepareBuild(HashJoinGlobalSinkState &sink) {
	D_ASSERT(global_stage != HashJoinSourceStage::BUILD);
	auto &ht = *sink.hash_table;

	// Try to put the next partitions in the block collection of the HT
	if (!ht.PrepareExternalFinalize()) {
		global_stage = HashJoinSourceStage::DONE;
		return;
	}

	auto &block_collection = ht.GetBlockCollection();
	build_block_idx = 0;
	build_block_count = block_collection.blocks.size();
	build_block_done = 0;
	ht.InitializePointerTable();

	global_stage = HashJoinSourceStage::BUILD;
}

void HashJoinGlobalSourceState::PrepareProbe(HashJoinGlobalSinkState &sink) {
	sink.probe_spill->PrepareNextProbe();

	probe_chunk_count = sink.probe_spill->consumer->ChunkCount();
	probe_chunk_done = 0;

	if (IsRightOuterJoin(op.join_type)) {
		full_outer_scan.Reset();
		full_outer_scan.total = sink.hash_table->Count();
	}

	global_stage = HashJoinSourceStage::PROBE;
}

bool HashJoinGlobalSourceState::AssignTask(HashJoinGlobalSinkState &sink, HashJoinLocalSourceState &lstate) {
	D_ASSERT(lstate.TaskFinished());

	lock_guard<mutex> guard(lock);
	switch (global_stage.load()) {
	case HashJoinSourceStage::BUILD:
		if (build_block_idx != build_block_count) {
			lstate.local_stage = global_stage;
			lstate.build_block_idx_start = build_block_idx;
			build_block_idx = MinValue<idx_t>(build_block_count, build_block_idx + build_blocks_per_thread);
			lstate.build_block_idx_end = build_block_idx;
			return true;
		}
		break;
	case HashJoinSourceStage::PROBE:
		if (sink.probe_spill->consumer && sink.probe_spill->consumer->AssignChunk(lstate.probe_local_scan)) {
			lstate.local_stage = global_stage;
			return true;
		}
		break;
	case HashJoinSourceStage::SCAN_HT:
		if (full_outer_scan.scan_index != full_outer_scan.total) {
			lstate.local_stage = global_stage;
			lstate.ScanFullOuter(sink, *this);
			return true;
		}
		break;
	case HashJoinSourceStage::DONE:
		break;
	default:
		throw InternalException("Unexpected HashJoinSourceStage in AssignTask!");
	}
	return false;
}

HashJoinLocalSourceState::HashJoinLocalSourceState(const PhysicalHashJoin &op, Allocator &allocator)
    : local_stage(HashJoinSourceStage::INIT), addresses(LogicalType::POINTER) {
	auto &chunk_state = probe_local_scan.current_chunk_state;
	chunk_state.properties = ColumnDataScanProperties::ALLOW_ZERO_COPY;

	auto &sink = (HashJoinGlobalSinkState &)*op.sink_state;
	probe_chunk.Initialize(allocator, sink.probe_types);
	join_keys.Initialize(allocator, op.condition_types);
	payload.Initialize(allocator, op.children[0]->types);

	// Store the indices of the columns to reference them easily
	idx_t col_idx = 0;
	for (; col_idx < op.condition_types.size(); col_idx++) {
		join_key_indices.push_back(col_idx);
	}
	for (; col_idx < sink.probe_types.size() - 1; col_idx++) {
		payload_indices.push_back(col_idx);
	}
}

void HashJoinLocalSourceState::ExecuteTask(HashJoinGlobalSinkState &sink, HashJoinGlobalSourceState &gstate,
                                           DataChunk &chunk) {
	switch (local_stage) {
	case HashJoinSourceStage::BUILD:
		ExternalBuild(sink, gstate);
		break;
	case HashJoinSourceStage::PROBE:
		ExternalProbe(sink, gstate, chunk);
		break;
	case HashJoinSourceStage::SCAN_HT:
		ExternalScanHT(sink, gstate, chunk);
		break;
	default:
		throw InternalException("Unexpected HashJoinSourceStage in ExecuteTask!");
	}
}

bool HashJoinLocalSourceState::TaskFinished() {
	switch (local_stage) {
	case HashJoinSourceStage::INIT:
	case HashJoinSourceStage::BUILD:
		return true;
	case HashJoinSourceStage::PROBE:
		return scan_structure == nullptr;
	case HashJoinSourceStage::SCAN_HT:
		return full_outer_in_progress == 0;
	default:
		throw InternalException("Unexpected HashJoinSourceStage in TaskFinished!");
	}
}

void HashJoinLocalSourceState::ExternalBuild(HashJoinGlobalSinkState &sink, HashJoinGlobalSourceState &gstate) {
	D_ASSERT(local_stage == HashJoinSourceStage::BUILD);

	auto &ht = *sink.hash_table;
	ht.Finalize(build_block_idx_start, build_block_idx_end, true);

	lock_guard<mutex> guard(gstate.lock);
	gstate.build_block_done += build_block_idx_end - build_block_idx_start;
}

void HashJoinLocalSourceState::ExternalProbe(HashJoinGlobalSinkState &sink, HashJoinGlobalSourceState &gstate,
                                             DataChunk &chunk) {
	D_ASSERT(local_stage == HashJoinSourceStage::PROBE && sink.hash_table->finalized);

	if (scan_structure) {
		// still have elements remaining (i.e. we got >STANDARD_VECTOR_SIZE elements in the previous probe)
		scan_structure->Next(join_keys, payload, chunk);
		if (chunk.size() == 0) {
			scan_structure = nullptr;
			sink.probe_spill->consumer->FinishChunk(probe_local_scan);
			lock_guard<mutex> lock(gstate.lock);
			gstate.probe_chunk_done++;
		}
		return;
	}

	// Scan input chunk for next probe
	sink.probe_spill->consumer->ScanChunk(probe_local_scan, probe_chunk);

	// Get the probe chunk columns/hashes
	join_keys.ReferenceColumns(probe_chunk, join_key_indices);
	payload.ReferenceColumns(probe_chunk, payload_indices);
	auto precomputed_hashes = &probe_chunk.data.back();

	// Perform the probe
	scan_structure = sink.hash_table->Probe(join_keys, precomputed_hashes);
	scan_structure->Next(join_keys, payload, chunk);
}

void HashJoinLocalSourceState::ExternalScanHT(HashJoinGlobalSinkState &sink, HashJoinGlobalSourceState &gstate,
                                              DataChunk &chunk) {
	D_ASSERT(local_stage == HashJoinSourceStage::SCAN_HT && full_outer_in_progress != 0);

	if (full_outer_found_entries != 0) {
		// Just did a scan, now gather
		sink.hash_table->GatherFullOuter(chunk, addresses, full_outer_found_entries);
		full_outer_found_entries = 0;
		return;
	}

	lock_guard<mutex> guard(gstate.lock);
	auto &fo_ss = gstate.full_outer_scan;
	fo_ss.scanned += full_outer_in_progress;
	full_outer_in_progress = 0;
}

void HashJoinLocalSourceState::ScanFullOuter(HashJoinGlobalSinkState &sink, HashJoinGlobalSourceState &gstate) {
	auto &fo_ss = gstate.full_outer_scan;
	idx_t scan_index_before = fo_ss.scan_index;
	full_outer_found_entries = sink.hash_table->ScanFullOuter(fo_ss, addresses);
	idx_t scanned = fo_ss.scan_index - scan_index_before;
	full_outer_in_progress = scanned;
}

void PhysicalHashJoin::GetData(ExecutionContext &context, DataChunk &chunk, GlobalSourceState &gstate_p,
                               LocalSourceState &lstate_p) const {
	auto &sink = (HashJoinGlobalSinkState &)*sink_state;
	auto &gstate = (HashJoinGlobalSourceState &)gstate_p;
	auto &lstate = (HashJoinLocalSourceState &)lstate_p;
	sink.scanned_data = true;

	if (!sink.external) {
		if (IsRightOuterJoin(join_type)) {
			{
				lock_guard<mutex> guard(gstate.lock);
				lstate.ScanFullOuter(sink, gstate);
			}
			sink.hash_table->GatherFullOuter(chunk, lstate.addresses, lstate.full_outer_found_entries);
		}
		return;
	}

	D_ASSERT(can_go_external);
	if (gstate.global_stage == HashJoinSourceStage::INIT) {
		gstate.Initialize(context.client, sink);
	}

	// Any call to GetData must produce tuples, otherwise the pipeline executor thinks that we're done
	// Therefore, we loop until we've produced tuples, or until the operator is actually done
	while (gstate.global_stage != HashJoinSourceStage::DONE && chunk.size() == 0) {
		if (!lstate.TaskFinished() || gstate.AssignTask(sink, lstate)) {
			lstate.ExecuteTask(sink, gstate, chunk);
		} else {
			gstate.TryPrepareNextStage(sink);
		}
	}
}

} // namespace duckdb














#include <thread>

namespace duckdb {

PhysicalIEJoin::PhysicalIEJoin(LogicalOperator &op, unique_ptr<PhysicalOperator> left,
                               unique_ptr<PhysicalOperator> right, vector<JoinCondition> cond, JoinType join_type,
                               idx_t estimated_cardinality)
    : PhysicalRangeJoin(op, PhysicalOperatorType::IE_JOIN, std::move(left), std::move(right), std::move(cond),
                        join_type, estimated_cardinality) {

	// 1. let L1 (resp. L2) be the array of column X (resp. Y)
	D_ASSERT(conditions.size() >= 2);
	lhs_orders.resize(2);
	rhs_orders.resize(2);
	for (idx_t i = 0; i < 2; ++i) {
		auto &cond = conditions[i];
		D_ASSERT(cond.left->return_type == cond.right->return_type);
		join_key_types.push_back(cond.left->return_type);

		// Convert the conditions to sort orders
		auto left = cond.left->Copy();
		auto right = cond.right->Copy();
		auto sense = OrderType::INVALID;

		// 2. if (op1 ∈ {>, ≥}) sort L1 in descending order
		// 3. else if (op1 ∈ {<, ≤}) sort L1 in ascending order
		// 4. if (op2 ∈ {>, ≥}) sort L2 in ascending order
		// 5. else if (op2 ∈ {<, ≤}) sort L2 in descending order
		switch (cond.comparison) {
		case ExpressionType::COMPARE_GREATERTHAN:
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			sense = i ? OrderType::ASCENDING : OrderType::DESCENDING;
			break;
		case ExpressionType::COMPARE_LESSTHAN:
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			sense = i ? OrderType::DESCENDING : OrderType::ASCENDING;
			break;
		default:
			throw NotImplementedException("Unimplemented join type for IEJoin");
		}
		lhs_orders[i].emplace_back(BoundOrderByNode(sense, OrderByNullType::NULLS_LAST, std::move(left)));
		rhs_orders[i].emplace_back(BoundOrderByNode(sense, OrderByNullType::NULLS_LAST, std::move(right)));
	}

	for (idx_t i = 2; i < conditions.size(); ++i) {
		auto &cond = conditions[i];
		D_ASSERT(cond.left->return_type == cond.right->return_type);
		join_key_types.push_back(cond.left->return_type);
	}
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class IEJoinLocalState : public LocalSinkState {
public:
	using LocalSortedTable = PhysicalRangeJoin::LocalSortedTable;

	IEJoinLocalState(ClientContext &context, const PhysicalRangeJoin &op, const idx_t child)
	    : table(context, op, child) {
	}

	//! The local sort state
	LocalSortedTable table;
};

class IEJoinGlobalState : public GlobalSinkState {
public:
	using GlobalSortedTable = PhysicalRangeJoin::GlobalSortedTable;

public:
	IEJoinGlobalState(ClientContext &context, const PhysicalIEJoin &op) : child(0) {
		tables.resize(2);
		RowLayout lhs_layout;
		lhs_layout.Initialize(op.children[0]->types);
		vector<BoundOrderByNode> lhs_order;
		lhs_order.emplace_back(op.lhs_orders[0][0].Copy());
		tables[0] = make_unique<GlobalSortedTable>(context, lhs_order, lhs_layout);

		RowLayout rhs_layout;
		rhs_layout.Initialize(op.children[1]->types);
		vector<BoundOrderByNode> rhs_order;
		rhs_order.emplace_back(op.rhs_orders[0][0].Copy());
		tables[1] = make_unique<GlobalSortedTable>(context, rhs_order, rhs_layout);
	}

	IEJoinGlobalState(IEJoinGlobalState &prev)
	    : GlobalSinkState(prev), tables(std::move(prev.tables)), child(prev.child + 1) {
	}

	void Sink(DataChunk &input, IEJoinLocalState &lstate) {
		auto &table = *tables[child];
		auto &global_sort_state = table.global_sort_state;
		auto &local_sort_state = lstate.table.local_sort_state;

		// Sink the data into the local sort state
		lstate.table.Sink(input, global_sort_state);

		// When sorting data reaches a certain size, we sort it
		if (local_sort_state.SizeInBytes() >= table.memory_per_thread) {
			local_sort_state.Sort(global_sort_state, true);
		}
	}

	vector<unique_ptr<GlobalSortedTable>> tables;
	size_t child;
};

unique_ptr<GlobalSinkState> PhysicalIEJoin::GetGlobalSinkState(ClientContext &context) const {
	D_ASSERT(!sink_state);
	return make_unique<IEJoinGlobalState>(context, *this);
}

unique_ptr<LocalSinkState> PhysicalIEJoin::GetLocalSinkState(ExecutionContext &context) const {
	idx_t sink_child = 0;
	if (sink_state) {
		const auto &ie_sink = (IEJoinGlobalState &)*sink_state;
		sink_child = ie_sink.child;
	}
	return make_unique<IEJoinLocalState>(context.client, *this, sink_child);
}

SinkResultType PhysicalIEJoin::Sink(ExecutionContext &context, GlobalSinkState &gstate_p, LocalSinkState &lstate_p,
                                    DataChunk &input) const {
	auto &gstate = (IEJoinGlobalState &)gstate_p;
	auto &lstate = (IEJoinLocalState &)lstate_p;

	gstate.Sink(input, lstate);

	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalIEJoin::Combine(ExecutionContext &context, GlobalSinkState &gstate_p, LocalSinkState &lstate_p) const {
	auto &gstate = (IEJoinGlobalState &)gstate_p;
	auto &lstate = (IEJoinLocalState &)lstate_p;
	gstate.tables[gstate.child]->Combine(lstate.table);
	auto &client_profiler = QueryProfiler::Get(context.client);

	context.thread.profiler.Flush(this, &lstate.table.executor, gstate.child ? "rhs_executor" : "lhs_executor", 1);
	client_profiler.Flush(context.thread.profiler);
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//
SinkFinalizeType PhysicalIEJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                          GlobalSinkState &gstate_p) const {
	auto &gstate = (IEJoinGlobalState &)gstate_p;
	auto &table = *gstate.tables[gstate.child];
	auto &global_sort_state = table.global_sort_state;

	if ((gstate.child == 1 && IsRightOuterJoin(join_type)) || (gstate.child == 0 && IsLeftOuterJoin(join_type))) {
		// for FULL/LEFT/RIGHT OUTER JOIN, initialize found_match to false for every tuple
		table.IntializeMatches();
	}
	if (gstate.child == 1 && global_sort_state.sorted_blocks.empty() && EmptyResultIfRHSIsEmpty()) {
		// Empty input!
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}

	// Sort the current input child
	table.Finalize(pipeline, event);

	// Move to the next input child
	++gstate.child;

	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
OperatorResultType PhysicalIEJoin::ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                   GlobalOperatorState &gstate, OperatorState &state) const {
	return OperatorResultType::FINISHED;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
struct IEJoinUnion {
	using SortedTable = PhysicalRangeJoin::GlobalSortedTable;

	static idx_t AppendKey(SortedTable &table, ExpressionExecutor &executor, SortedTable &marked, int64_t increment,
	                       int64_t base, const idx_t block_idx);

	static void Sort(SortedTable &table) {
		auto &global_sort_state = table.global_sort_state;
		global_sort_state.PrepareMergePhase();
		while (global_sort_state.sorted_blocks.size() > 1) {
			global_sort_state.InitializeMergeRound();
			MergeSorter merge_sorter(global_sort_state, global_sort_state.buffer_manager);
			merge_sorter.PerformInMergeRound();
			global_sort_state.CompleteMergeRound(true);
		}
	}

	template <typename T>
	static vector<T> ExtractColumn(SortedTable &table, idx_t col_idx) {
		vector<T> result;
		result.reserve(table.count);

		auto &gstate = table.global_sort_state;
		auto &blocks = *gstate.sorted_blocks[0]->payload_data;
		PayloadScanner scanner(blocks, gstate, false);

		DataChunk payload;
		payload.Initialize(Allocator::DefaultAllocator(), gstate.payload_layout.GetTypes());
		for (;;) {
			scanner.Scan(payload);
			const auto count = payload.size();
			if (!count) {
				break;
			}

			const auto data_ptr = FlatVector::GetData<T>(payload.data[col_idx]);
			result.insert(result.end(), data_ptr, data_ptr + count);
		}

		return result;
	}

	IEJoinUnion(ClientContext &context, const PhysicalIEJoin &op, SortedTable &t1, const idx_t b1, SortedTable &t2,
	            const idx_t b2);

	idx_t SearchL1(idx_t pos);
	bool NextRow();

	//! Inverted loop
	idx_t JoinComplexBlocks(SelectionVector &lsel, SelectionVector &rsel);

	//! L1
	unique_ptr<SortedTable> l1;
	//! L2
	unique_ptr<SortedTable> l2;

	//! Li
	vector<int64_t> li;
	//! P
	vector<idx_t> p;

	//! B
	vector<validity_t> bit_array;
	ValidityMask bit_mask;

	//! Bloom Filter
	static constexpr idx_t BLOOM_CHUNK_BITS = 1024;
	idx_t bloom_count;
	vector<validity_t> bloom_array;
	ValidityMask bloom_filter;

	//! Iteration state
	idx_t n;
	idx_t i;
	idx_t j;
	unique_ptr<SBIterator> op1;
	unique_ptr<SBIterator> off1;
	unique_ptr<SBIterator> op2;
	unique_ptr<SBIterator> off2;
	int64_t lrid;
};

idx_t IEJoinUnion::AppendKey(SortedTable &table, ExpressionExecutor &executor, SortedTable &marked, int64_t increment,
                             int64_t base, const idx_t block_idx) {
	LocalSortState local_sort_state;
	local_sort_state.Initialize(marked.global_sort_state, marked.global_sort_state.buffer_manager);

	// Reading
	const auto valid = table.count - table.has_null;
	auto &gstate = table.global_sort_state;
	PayloadScanner scanner(gstate, block_idx);
	auto table_idx = block_idx * gstate.block_capacity;

	DataChunk scanned;
	scanned.Initialize(Allocator::DefaultAllocator(), scanner.GetPayloadTypes());

	// Writing
	auto types = local_sort_state.sort_layout->logical_types;
	const idx_t payload_idx = types.size();

	const auto &payload_types = local_sort_state.payload_layout->GetTypes();
	types.insert(types.end(), payload_types.begin(), payload_types.end());
	const idx_t rid_idx = types.size() - 1;

	DataChunk keys;
	DataChunk payload;
	keys.Initialize(Allocator::DefaultAllocator(), types);

	idx_t inserted = 0;
	for (auto rid = base; table_idx < valid;) {
		scanner.Scan(scanned);

		// NULLs are at the end, so stop when we reach them
		auto scan_count = scanned.size();
		if (table_idx + scan_count > valid) {
			scan_count = valid - table_idx;
			scanned.SetCardinality(scan_count);
		}
		if (scan_count == 0) {
			break;
		}
		table_idx += scan_count;

		// Compute the input columns from the payload
		keys.Reset();
		keys.Split(payload, rid_idx);
		executor.Execute(scanned, keys);

		// Mark the rid column
		payload.data[0].Sequence(rid, increment, scan_count);
		payload.SetCardinality(scan_count);
		keys.Fuse(payload);
		rid += increment * scan_count;

		// Sort on the sort columns (which will no longer be needed)
		keys.Split(payload, payload_idx);
		local_sort_state.SinkChunk(keys, payload);
		inserted += scan_count;
		keys.Fuse(payload);

		// Flush when we have enough data
		if (local_sort_state.SizeInBytes() >= marked.memory_per_thread) {
			local_sort_state.Sort(marked.global_sort_state, true);
		}
	}
	marked.global_sort_state.AddLocalState(local_sort_state);
	marked.count += inserted;

	return inserted;
}

IEJoinUnion::IEJoinUnion(ClientContext &context, const PhysicalIEJoin &op, SortedTable &t1, const idx_t b1,
                         SortedTable &t2, const idx_t b2)
    : n(0), i(0) {
	// input : query Q with 2 join predicates t1.X op1 t2.X' and t1.Y op2 t2.Y', tables T, T' of sizes m and n resp.
	// output: a list of tuple pairs (ti , tj)
	// Note that T/T' are already sorted on X/X' and contain the payload data
	// We only join the two block numbers and use the sizes of the blocks as the counts

	// 0. Filter out tables with no overlap
	if (!t1.BlockSize(b1) || !t2.BlockSize(b2)) {
		return;
	}

	const auto &cmp1 = op.conditions[0].comparison;
	SBIterator bounds1(t1.global_sort_state, cmp1);
	SBIterator bounds2(t2.global_sort_state, cmp1);

	// t1.X[0] op1 t2.X'[-1]
	bounds1.SetIndex(bounds1.block_capacity * b1);
	bounds2.SetIndex(bounds2.block_capacity * b2 + t2.BlockSize(b2) - 1);
	if (!bounds1.Compare(bounds2)) {
		return;
	}

	// 1. let L1 (resp. L2) be the array of column X (resp. Y )
	const auto &order1 = op.lhs_orders[0][0];
	const auto &order2 = op.lhs_orders[1][0];

	// 2. if (op1 ∈ {>, ≥}) sort L1 in descending order
	// 3. else if (op1 ∈ {<, ≤}) sort L1 in ascending order

	// For the union algorithm, we make a unified table with the keys and the rids as the payload:
	//		X/X', Y/Y', R/R'/Li
	// The first position is the sort key.
	vector<LogicalType> types;
	types.emplace_back(order2.expression->return_type);
	types.emplace_back(LogicalType::BIGINT);
	RowLayout payload_layout;
	payload_layout.Initialize(types);

	// Sort on the first expression
	auto ref = make_unique<BoundReferenceExpression>(order1.expression->return_type, 0);
	vector<BoundOrderByNode> orders;
	orders.emplace_back(BoundOrderByNode(order1.type, order1.null_order, std::move(ref)));

	l1 = make_unique<SortedTable>(context, orders, payload_layout);

	// LHS has positive rids
	ExpressionExecutor l_executor(context);
	l_executor.AddExpression(*order1.expression);
	l_executor.AddExpression(*order2.expression);
	AppendKey(t1, l_executor, *l1, 1, 1, b1);

	// RHS has negative rids
	ExpressionExecutor r_executor(context);
	r_executor.AddExpression(*op.rhs_orders[0][0].expression);
	r_executor.AddExpression(*op.rhs_orders[1][0].expression);
	AppendKey(t2, r_executor, *l1, -1, -1, b2);

	Sort(*l1);

	op1 = make_unique<SBIterator>(l1->global_sort_state, cmp1);
	off1 = make_unique<SBIterator>(l1->global_sort_state, cmp1);

	// We don't actually need the L1 column, just its sort key, which is in the sort blocks
	li = ExtractColumn<int64_t>(*l1, types.size() - 1);

	// 4. if (op2 ∈ {>, ≥}) sort L2 in ascending order
	// 5. else if (op2 ∈ {<, ≤}) sort L2 in descending order

	// We sort on Y/Y' to obtain the sort keys and the permutation array.
	// For this we just need a two-column table of Y, P
	types.clear();
	types.emplace_back(LogicalType::BIGINT);
	payload_layout.Initialize(types);

	// Sort on the first expression
	orders.clear();
	ref = make_unique<BoundReferenceExpression>(order2.expression->return_type, 0);
	orders.emplace_back(BoundOrderByNode(order2.type, order2.null_order, std::move(ref)));

	ExpressionExecutor executor(context);
	executor.AddExpression(*orders[0].expression);

	l2 = make_unique<SortedTable>(context, orders, payload_layout);
	for (idx_t base = 0, block_idx = 0; block_idx < l1->BlockCount(); ++block_idx) {
		base += AppendKey(*l1, executor, *l2, 1, base, block_idx);
	}

	Sort(*l2);

	// We don't actually need the L2 column, just its sort key, which is in the sort blocks

	// 6. compute the permutation array P of L2 w.r.t. L1
	p = ExtractColumn<idx_t>(*l2, types.size() - 1);

	// 7. initialize bit-array B (|B| = n), and set all bits to 0
	n = l2->count.load();
	bit_array.resize(ValidityMask::EntryCount(n), 0);
	bit_mask.Initialize(bit_array.data());

	// Bloom filter
	bloom_count = (n + (BLOOM_CHUNK_BITS - 1)) / BLOOM_CHUNK_BITS;
	bloom_array.resize(ValidityMask::EntryCount(bloom_count), 0);
	bloom_filter.Initialize(bloom_array.data());

	// 11. for(i←1 to n) do
	const auto &cmp2 = op.conditions[1].comparison;
	op2 = make_unique<SBIterator>(l2->global_sort_state, cmp2);
	off2 = make_unique<SBIterator>(l2->global_sort_state, cmp2);
	i = 0;
	j = 0;
	(void)NextRow();
}

idx_t IEJoinUnion::SearchL1(idx_t pos) {
	// Perform an exponential search in the appropriate direction
	op1->SetIndex(pos);

	idx_t step = 1;
	auto hi = pos;
	auto lo = pos;
	if (!op1->cmp) {
		// Scan left for loose inequality
		lo -= MinValue(step, lo);
		step *= 2;
		off1->SetIndex(lo);
		while (lo > 0 && op1->Compare(*off1)) {
			hi = lo;
			lo -= MinValue(step, lo);
			step *= 2;
			off1->SetIndex(lo);
		}
	} else {
		// Scan right for strict inequality
		hi += MinValue(step, n - hi);
		step *= 2;
		off1->SetIndex(hi);
		while (hi < n && !op1->Compare(*off1)) {
			lo = hi;
			hi += MinValue(step, n - hi);
			step *= 2;
			off1->SetIndex(hi);
		}
	}

	// Binary search the target area
	while (lo < hi) {
		const auto mid = lo + (hi - lo) / 2;
		off1->SetIndex(mid);
		if (op1->Compare(*off1)) {
			hi = mid;
		} else {
			lo = mid + 1;
		}
	}

	off1->SetIndex(lo);

	return lo;
}

bool IEJoinUnion::NextRow() {
	for (; i < n; ++i) {
		// 12. pos ← P[i]
		auto pos = p[i];
		lrid = li[pos];
		if (lrid < 0) {
			continue;
		}

		// 16. B[pos] ← 1
		op2->SetIndex(i);
		for (; off2->GetIndex() < n; ++(*off2)) {
			if (!off2->Compare(*op2)) {
				break;
			}
			const auto p2 = p[off2->GetIndex()];
			if (li[p2] < 0) {
				// Only mark rhs matches.
				bit_mask.SetValid(p2);
				bloom_filter.SetValid(p2 / BLOOM_CHUNK_BITS);
			}
		}

		// 9.  if (op1 ∈ {≤,≥} and op2 ∈ {≤,≥}) eqOff = 0
		// 10. else eqOff = 1
		// No, because there could be more than one equal value.
		// Find the leftmost off1 where L1[pos] op1 L1[off1..n]
		// These are the rows that satisfy the op1 condition
		// and that is where we should start scanning B from
		j = SearchL1(pos);

		return true;
	}
	return false;
}

static idx_t NextValid(const ValidityMask &bits, idx_t j, const idx_t n) {
	if (j >= n) {
		return n;
	}

	// We can do a first approximation by checking entries one at a time
	// which gives 64:1.
	idx_t entry_idx, idx_in_entry;
	bits.GetEntryIndex(j, entry_idx, idx_in_entry);
	auto entry = bits.GetValidityEntry(entry_idx++);

	// Trim the bits before the start position
	entry &= (ValidityMask::ValidityBuffer::MAX_ENTRY << idx_in_entry);

	// Check the non-ragged entries
	for (const auto entry_count = bits.EntryCount(n); entry_idx < entry_count; ++entry_idx) {
		if (entry) {
			for (; idx_in_entry < bits.BITS_PER_VALUE; ++idx_in_entry, ++j) {
				if (bits.RowIsValid(entry, idx_in_entry)) {
					return j;
				}
			}
		} else {
			j += bits.BITS_PER_VALUE - idx_in_entry;
		}

		entry = bits.GetValidityEntry(entry_idx);
		idx_in_entry = 0;
	}

	// Check the final entry
	for (; j < n; ++idx_in_entry, ++j) {
		if (bits.RowIsValid(entry, idx_in_entry)) {
			return j;
		}
	}

	return j;
}

idx_t IEJoinUnion::JoinComplexBlocks(SelectionVector &lsel, SelectionVector &rsel) {
	// 8. initialize join result as an empty list for tuple pairs
	idx_t result_count = 0;

	// 11. for(i←1 to n) do
	while (i < n) {
		// 13. for (j ← pos+eqOff to n) do
		for (;;) {
			// 14. if B[j] = 1 then

			//	Use the Bloom filter to find candidate blocks
			while (j < n) {
				auto bloom_begin = NextValid(bloom_filter, j / BLOOM_CHUNK_BITS, bloom_count) * BLOOM_CHUNK_BITS;
				auto bloom_end = MinValue<idx_t>(n, bloom_begin + BLOOM_CHUNK_BITS);

				j = MaxValue<idx_t>(j, bloom_begin);
				j = NextValid(bit_mask, j, bloom_end);
				if (j < bloom_end) {
					break;
				}
			}

			if (j >= n) {
				break;
			}

			// Filter out tuples with the same sign (they come from the same table)
			const auto rrid = li[j];
			++j;

			// 15. add tuples w.r.t. (L1[j], L1[i]) to join result
			if (lrid > 0 && rrid < 0) {
				lsel.set_index(result_count, sel_t(+lrid - 1));
				rsel.set_index(result_count, sel_t(-rrid - 1));
				++result_count;
				if (result_count == STANDARD_VECTOR_SIZE) {
					// out of space!
					return result_count;
				}
			}
		}
		++i;

		if (!NextRow()) {
			break;
		}
	}

	return result_count;
}

class IEJoinLocalSourceState : public LocalSourceState {
public:
	explicit IEJoinLocalSourceState(ClientContext &context, const PhysicalIEJoin &op)
	    : op(op), true_sel(STANDARD_VECTOR_SIZE), left_executor(context), right_executor(context),
	      left_matches(nullptr), right_matches(nullptr) {
		auto &allocator = Allocator::Get(context);
		if (op.conditions.size() < 3) {
			return;
		}

		vector<LogicalType> left_types;
		vector<LogicalType> right_types;
		for (idx_t i = 2; i < op.conditions.size(); ++i) {
			const auto &cond = op.conditions[i];

			left_types.push_back(cond.left->return_type);
			left_executor.AddExpression(*cond.left);

			right_types.push_back(cond.left->return_type);
			right_executor.AddExpression(*cond.right);
		}

		left_keys.Initialize(allocator, left_types);
		right_keys.Initialize(allocator, right_types);
	}

	idx_t SelectOuterRows(bool *matches) {
		idx_t count = 0;
		for (; outer_idx < outer_count; ++outer_idx) {
			if (!matches[outer_idx]) {
				true_sel.set_index(count++, outer_idx);
				if (count >= STANDARD_VECTOR_SIZE) {
					break;
				}
			}
		}

		return count;
	}

	const PhysicalIEJoin &op;

	// Joining
	unique_ptr<IEJoinUnion> joiner;

	idx_t left_base;
	idx_t left_block_index;

	idx_t right_base;
	idx_t right_block_index;

	// Trailing predicates
	SelectionVector true_sel;

	ExpressionExecutor left_executor;
	DataChunk left_keys;

	ExpressionExecutor right_executor;
	DataChunk right_keys;

	// Outer joins
	idx_t outer_idx;
	idx_t outer_count;
	bool *left_matches;
	bool *right_matches;
};

void PhysicalIEJoin::ResolveComplexJoin(ExecutionContext &context, DataChunk &chunk, LocalSourceState &state_p) const {
	auto &state = (IEJoinLocalSourceState &)state_p;
	auto &ie_sink = (IEJoinGlobalState &)*sink_state;
	auto &left_table = *ie_sink.tables[0];
	auto &right_table = *ie_sink.tables[1];

	const auto left_cols = children[0]->GetTypes().size();
	do {
		SelectionVector lsel(STANDARD_VECTOR_SIZE);
		SelectionVector rsel(STANDARD_VECTOR_SIZE);
		auto result_count = state.joiner->JoinComplexBlocks(lsel, rsel);
		if (result_count == 0) {
			// exhausted this pair
			return;
		}

		// found matches: extract them
		chunk.Reset();
		SliceSortedPayload(chunk, left_table.global_sort_state, state.left_block_index, lsel, result_count, 0);
		SliceSortedPayload(chunk, right_table.global_sort_state, state.right_block_index, rsel, result_count,
		                   left_cols);
		chunk.SetCardinality(result_count);

		auto sel = FlatVector::IncrementalSelectionVector();
		if (conditions.size() > 2) {
			// If there are more expressions to compute,
			// split the result chunk into the left and right halves
			// so we can compute the values for comparison.
			const auto tail_cols = conditions.size() - 2;

			DataChunk right_chunk;
			chunk.Split(right_chunk, left_cols);
			state.left_executor.SetChunk(chunk);
			state.right_executor.SetChunk(right_chunk);

			auto tail_count = result_count;
			auto true_sel = &state.true_sel;
			for (size_t cmp_idx = 0; cmp_idx < tail_cols; ++cmp_idx) {
				auto &left = state.left_keys.data[cmp_idx];
				state.left_executor.ExecuteExpression(cmp_idx, left);

				auto &right = state.right_keys.data[cmp_idx];
				state.right_executor.ExecuteExpression(cmp_idx, right);

				if (tail_count < result_count) {
					left.Slice(*sel, tail_count);
					right.Slice(*sel, tail_count);
				}
				tail_count = SelectJoinTail(conditions[cmp_idx + 2].comparison, left, right, sel, tail_count, true_sel);
				sel = true_sel;
			}
			chunk.Fuse(right_chunk);

			if (tail_count < result_count) {
				result_count = tail_count;
				chunk.Slice(*sel, result_count);
			}
		}

		// found matches: mark the found matches if required
		if (left_table.found_match) {
			for (idx_t i = 0; i < result_count; i++) {
				left_table.found_match[state.left_base + lsel[sel->get_index(i)]] = true;
			}
		}
		if (right_table.found_match) {
			for (idx_t i = 0; i < result_count; i++) {
				right_table.found_match[state.right_base + rsel[sel->get_index(i)]] = true;
			}
		}
		chunk.Verify();
	} while (chunk.size() == 0);
}

class IEJoinGlobalSourceState : public GlobalSourceState {
public:
	explicit IEJoinGlobalSourceState(const PhysicalIEJoin &op)
	    : op(op), initialized(false), next_pair(0), completed(0), left_outers(0), next_left(0), right_outers(0),
	      next_right(0) {
	}

	void Initialize(IEJoinGlobalState &sink_state) {
		lock_guard<mutex> initializing(lock);
		if (initialized) {
			return;
		}

		// Compute the starting row for reach block
		// (In theory these are all the same size, but you never know...)
		auto &left_table = *sink_state.tables[0];
		const auto left_blocks = left_table.BlockCount();
		idx_t left_base = 0;

		for (size_t lhs = 0; lhs < left_blocks; ++lhs) {
			left_bases.emplace_back(left_base);
			left_base += left_table.BlockSize(lhs);
		}

		auto &right_table = *sink_state.tables[1];
		const auto right_blocks = right_table.BlockCount();
		idx_t right_base = 0;
		for (size_t rhs = 0; rhs < right_blocks; ++rhs) {
			right_bases.emplace_back(right_base);
			right_base += right_table.BlockSize(rhs);
		}

		// Outer join block counts
		if (left_table.found_match) {
			left_outers = left_blocks;
		}

		if (right_table.found_match) {
			right_outers = right_blocks;
		}

		// Ready for action
		initialized = true;
	}

public:
	idx_t MaxThreads() override {
		// We can't leverage any more threads than block pairs.
		const auto &sink_state = ((IEJoinGlobalState &)*op.sink_state);
		return sink_state.tables[0]->BlockCount() * sink_state.tables[1]->BlockCount();
	}

	void GetNextPair(ClientContext &client, IEJoinGlobalState &gstate, IEJoinLocalSourceState &lstate) {
		auto &left_table = *gstate.tables[0];
		auto &right_table = *gstate.tables[1];

		const auto left_blocks = left_table.BlockCount();
		const auto right_blocks = right_table.BlockCount();
		const auto pair_count = left_blocks * right_blocks;

		// Regular block
		const auto i = next_pair++;
		if (i < pair_count) {
			const auto b1 = i / right_blocks;
			const auto b2 = i % right_blocks;

			lstate.left_block_index = b1;
			lstate.left_base = left_bases[b1];

			lstate.right_block_index = b2;
			lstate.right_base = right_bases[b2];

			lstate.joiner = make_unique<IEJoinUnion>(client, op, left_table, b1, right_table, b2);
			return;
		} else {
			--next_pair;
		}

		// Outer joins
		if (!left_outers && !right_outers) {
			return;
		}

		// Spin wait for regular blocks to finish(!)
		while (completed < pair_count) {
			std::this_thread::yield();
		}

		// Left outer blocks
		const auto l = next_left++;
		if (l < left_outers) {
			lstate.left_block_index = l;
			lstate.left_base = left_bases[l];

			lstate.left_matches = left_table.found_match.get() + lstate.left_base;
			lstate.outer_idx = 0;
			lstate.outer_count = left_table.BlockSize(l);
			return;
		} else {
			lstate.left_matches = nullptr;
			--next_left;
		}

		// Right outer block
		const auto r = next_right++;
		if (r < right_outers) {
			lstate.right_block_index = r;
			lstate.right_base = right_bases[r];

			lstate.right_matches = right_table.found_match.get() + lstate.right_base;
			lstate.outer_idx = 0;
			lstate.outer_count = right_table.BlockSize(r);
			return;
		} else {
			lstate.right_matches = nullptr;
			--next_right;
		}
	}

	void PairCompleted(ClientContext &client, IEJoinGlobalState &gstate, IEJoinLocalSourceState &lstate) {
		lstate.joiner.reset();
		++completed;
		GetNextPair(client, gstate, lstate);
	}

	const PhysicalIEJoin &op;

	mutex lock;
	bool initialized;

	// Join queue state
	std::atomic<size_t> next_pair;
	std::atomic<size_t> completed;

	// Block base row number
	vector<idx_t> left_bases;
	vector<idx_t> right_bases;

	// Outer joins
	idx_t left_outers;
	std::atomic<idx_t> next_left;

	idx_t right_outers;
	std::atomic<idx_t> next_right;
};

unique_ptr<GlobalSourceState> PhysicalIEJoin::GetGlobalSourceState(ClientContext &context) const {
	return make_unique<IEJoinGlobalSourceState>(*this);
}

unique_ptr<LocalSourceState> PhysicalIEJoin::GetLocalSourceState(ExecutionContext &context,
                                                                 GlobalSourceState &gstate) const {
	return make_unique<IEJoinLocalSourceState>(context.client, *this);
}

void PhysicalIEJoin::GetData(ExecutionContext &context, DataChunk &result, GlobalSourceState &gstate,
                             LocalSourceState &lstate) const {
	auto &ie_sink = (IEJoinGlobalState &)*sink_state;
	auto &ie_gstate = (IEJoinGlobalSourceState &)gstate;
	auto &ie_lstate = (IEJoinLocalSourceState &)lstate;

	ie_gstate.Initialize(ie_sink);

	if (!ie_lstate.joiner) {
		ie_gstate.GetNextPair(context.client, ie_sink, ie_lstate);
	}

	// Process INNER results
	while (ie_lstate.joiner) {
		ResolveComplexJoin(context, result, ie_lstate);

		if (result.size()) {
			return;
		}

		ie_gstate.PairCompleted(context.client, ie_sink, ie_lstate);
	}

	// Process LEFT OUTER results
	const auto left_cols = children[0]->GetTypes().size();
	while (ie_lstate.left_matches) {
		const idx_t count = ie_lstate.SelectOuterRows(ie_lstate.left_matches);
		if (!count) {
			ie_gstate.GetNextPair(context.client, ie_sink, ie_lstate);
			continue;
		}

		SliceSortedPayload(result, ie_sink.tables[0]->global_sort_state, ie_lstate.left_base, ie_lstate.true_sel,
		                   count);

		// Fill in NULLs to the right
		for (auto col_idx = left_cols; col_idx < result.ColumnCount(); ++col_idx) {
			result.data[col_idx].SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result.data[col_idx], true);
		}

		result.SetCardinality(count);
		result.Verify();

		return;
	}

	// Process RIGHT OUTER results
	while (ie_lstate.right_matches) {
		const idx_t count = ie_lstate.SelectOuterRows(ie_lstate.right_matches);
		if (!count) {
			ie_gstate.GetNextPair(context.client, ie_sink, ie_lstate);
			continue;
		}

		SliceSortedPayload(result, ie_sink.tables[1]->global_sort_state, ie_lstate.right_base, ie_lstate.true_sel,
		                   count, left_cols);

		// Fill in NULLs to the left
		for (idx_t col_idx = 0; col_idx < left_cols; ++col_idx) {
			result.data[col_idx].SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result.data[col_idx], true);
		}

		result.SetCardinality(count);
		result.Verify();

		return;
	}
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void PhysicalIEJoin::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	D_ASSERT(children.size() == 2);
	if (meta_pipeline.HasRecursiveCTE()) {
		throw NotImplementedException("IEJoins are not supported in recursive CTEs yet");
	}

	// becomes a source after both children fully sink their data
	meta_pipeline.GetState().SetPipelineSource(current, this);

	// Create one child meta pipeline that will hold the LHS and RHS pipelines
	auto child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, this);
	auto lhs_pipeline = child_meta_pipeline->GetBasePipeline();
	auto rhs_pipeline = child_meta_pipeline->CreatePipeline();

	// Build out LHS
	children[0]->BuildPipelines(*lhs_pipeline, *child_meta_pipeline);

	// RHS depends on everything in LHS
	child_meta_pipeline->AddDependenciesFrom(rhs_pipeline, lhs_pipeline.get(), true);

	// Build out RHS
	children[1]->BuildPipelines(*rhs_pipeline, *child_meta_pipeline);

	// Despite having the same sink, RHS needs its own PipelineFinishEvent
	child_meta_pipeline->AddFinishEvent(rhs_pipeline);
}

} // namespace duckdb















namespace duckdb {

class IndexJoinOperatorState : public CachingOperatorState {
public:
	IndexJoinOperatorState(ClientContext &context, const PhysicalIndexJoin &op)
	    : probe_executor(context), arena_allocator(BufferAllocator::Get(context)), keys(STANDARD_VECTOR_SIZE) {
		auto &allocator = Allocator::Get(context);
		rhs_rows.resize(STANDARD_VECTOR_SIZE);
		result_sizes.resize(STANDARD_VECTOR_SIZE);

		join_keys.Initialize(allocator, op.condition_types);
		for (auto &cond : op.conditions) {
			probe_executor.AddExpression(*cond.left);
		}
		if (!op.fetch_types.empty()) {
			rhs_chunk.Initialize(allocator, op.fetch_types);
		}
		rhs_sel.Initialize(STANDARD_VECTOR_SIZE);
	}

	bool first_fetch = true;
	idx_t lhs_idx = 0;
	idx_t rhs_idx = 0;
	idx_t result_size = 0;
	vector<idx_t> result_sizes;
	DataChunk join_keys;
	DataChunk rhs_chunk;
	SelectionVector rhs_sel;

	//! Vector of rows that mush be fetched for every LHS key
	vector<vector<row_t>> rhs_rows;
	ExpressionExecutor probe_executor;

	ArenaAllocator arena_allocator;
	vector<Key> keys;
	unique_ptr<ColumnFetchState> fetch_state;

public:
	void Finalize(PhysicalOperator *op, ExecutionContext &context) override {
		context.thread.profiler.Flush(op, &probe_executor, "probe_executor", 0);
	}
};

PhysicalIndexJoin::PhysicalIndexJoin(LogicalOperator &op, unique_ptr<PhysicalOperator> left,
                                     unique_ptr<PhysicalOperator> right, vector<JoinCondition> cond, JoinType join_type,
                                     const vector<idx_t> &left_projection_map_p, vector<idx_t> right_projection_map_p,
                                     vector<column_t> column_ids_p, Index *index_p, bool lhs_first,
                                     idx_t estimated_cardinality)
    : CachingPhysicalOperator(PhysicalOperatorType::INDEX_JOIN, std::move(op.types), estimated_cardinality),
      left_projection_map(left_projection_map_p), right_projection_map(std::move(right_projection_map_p)),
      index(index_p), conditions(std::move(cond)), join_type(join_type), lhs_first(lhs_first) {
	column_ids = std::move(column_ids_p);
	children.push_back(std::move(left));
	children.push_back(std::move(right));
	for (auto &condition : conditions) {
		condition_types.push_back(condition.left->return_type);
	}
	//! Only add to fetch_ids columns that are not indexed
	for (auto &index_id : index->column_ids) {
		index_ids.insert(index_id);
	}
	for (idx_t column_id = 0; column_id < column_ids.size(); column_id++) {
		auto it = index_ids.find(column_ids[column_id]);
		if (it == index_ids.end()) {
			fetch_ids.push_back(column_ids[column_id]);
			fetch_types.push_back(children[1]->types[column_id]);
		}
	}
	if (right_projection_map.empty()) {
		for (column_t i = 0; i < column_ids.size(); i++) {
			right_projection_map.push_back(i);
		}
	}
	if (left_projection_map.empty()) {
		for (column_t i = 0; i < children[0]->types.size(); i++) {
			left_projection_map.push_back(i);
		}
	}
}

unique_ptr<OperatorState> PhysicalIndexJoin::GetOperatorState(ExecutionContext &context) const {
	return make_unique<IndexJoinOperatorState>(context.client, *this);
}

void PhysicalIndexJoin::Output(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                               OperatorState &state_p) const {
	auto &phy_tbl_scan = (PhysicalTableScan &)*children[1];
	auto &bind_tbl = (TableScanBindData &)*phy_tbl_scan.bind_data;
	auto &transaction = DuckTransaction::Get(context.client, *bind_tbl.table->catalog);
	auto &state = (IndexJoinOperatorState &)state_p;

	auto &tbl = bind_tbl.table->GetStorage();
	idx_t output_sel_idx = 0;
	vector<row_t> fetch_rows;

	while (output_sel_idx < STANDARD_VECTOR_SIZE && state.lhs_idx < input.size()) {
		if (state.rhs_idx < state.result_sizes[state.lhs_idx]) {
			state.rhs_sel.set_index(output_sel_idx++, state.lhs_idx);
			if (!fetch_types.empty()) {
				//! We need to collect the rows we want to fetch
				fetch_rows.push_back(state.rhs_rows[state.lhs_idx][state.rhs_idx]);
			}
			state.rhs_idx++;
		} else {
			//! We are done with the matches from this LHS Key
			state.rhs_idx = 0;
			state.lhs_idx++;
		}
	}
	//! Now we fetch the RHS data
	if (!fetch_types.empty()) {
		if (fetch_rows.empty()) {
			return;
		}
		state.rhs_chunk.Reset();
		state.fetch_state = make_unique<ColumnFetchState>();
		Vector row_ids(LogicalType::ROW_TYPE, (data_ptr_t)&fetch_rows[0]);
		tbl.Fetch(transaction, state.rhs_chunk, fetch_ids, row_ids, output_sel_idx, *state.fetch_state);
	}

	//! Now we actually produce our result chunk
	idx_t left_offset = lhs_first ? 0 : right_projection_map.size();
	idx_t right_offset = lhs_first ? left_projection_map.size() : 0;
	idx_t rhs_column_idx = 0;
	for (idx_t i = 0; i < right_projection_map.size(); i++) {
		auto it = index_ids.find(column_ids[right_projection_map[i]]);
		if (it == index_ids.end()) {
			chunk.data[right_offset + i].Reference(state.rhs_chunk.data[rhs_column_idx++]);
		} else {
			chunk.data[right_offset + i].Slice(state.join_keys.data[0], state.rhs_sel, output_sel_idx);
		}
	}
	for (idx_t i = 0; i < left_projection_map.size(); i++) {
		chunk.data[left_offset + i].Slice(input.data[left_projection_map[i]], state.rhs_sel, output_sel_idx);
	}

	state.result_size = output_sel_idx;
	chunk.SetCardinality(state.result_size);
}

void PhysicalIndexJoin::GetRHSMatches(ExecutionContext &context, DataChunk &input, OperatorState &state_p) const {

	auto &state = (IndexJoinOperatorState &)state_p;
	auto &art = (ART &)*index;

	// generate the keys for this chunk
	state.arena_allocator.Reset();
	ART::GenerateKeys(state.arena_allocator, state.join_keys, state.keys);

	for (idx_t i = 0; i < input.size(); i++) {
		state.rhs_rows[i].clear();
		if (!state.keys[i].Empty()) {
			if (fetch_types.empty()) {
				IndexLock lock;
				index->InitializeLock(lock);
				art.SearchEqualJoinNoFetch(state.keys[i], state.result_sizes[i]);
			} else {
				IndexLock lock;
				index->InitializeLock(lock);
				art.SearchEqual(state.keys[i], (idx_t)-1, state.rhs_rows[i]);
				state.result_sizes[i] = state.rhs_rows[i].size();
			}
		} else {
			//! This is null so no matches
			state.result_sizes[i] = 0;
		}
	}
	for (idx_t i = input.size(); i < STANDARD_VECTOR_SIZE; i++) {
		//! No LHS chunk value so result size is empty
		state.result_sizes[i] = 0;
	}
}

OperatorResultType PhysicalIndexJoin::ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                      GlobalOperatorState &gstate, OperatorState &state_p) const {
	auto &state = (IndexJoinOperatorState &)state_p;

	state.result_size = 0;
	if (state.first_fetch) {
		state.probe_executor.Execute(input, state.join_keys);

		//! Fill Matches for the current LHS chunk
		GetRHSMatches(context, input, state_p);
		state.first_fetch = false;
	}
	//! Check if we need to get a new LHS chunk
	if (state.lhs_idx >= input.size()) {
		state.lhs_idx = 0;
		state.rhs_idx = 0;
		state.first_fetch = true;
		return OperatorResultType::NEED_MORE_INPUT;
	}
	//! Output vectors
	if (state.lhs_idx < input.size()) {
		Output(context, input, chunk, state_p);
	}
	return OperatorResultType::HAVE_MORE_OUTPUT;
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void PhysicalIndexJoin::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	// index join: we only continue into the LHS
	// the right side is probed by the index join
	// so we don't need to do anything in the pipeline with this child
	meta_pipeline.GetState().AddPipelineOperator(current, this);
	children[0]->BuildPipelines(current, meta_pipeline);
}

vector<const PhysicalOperator *> PhysicalIndexJoin::GetSources() const {
	return children[0]->GetSources();
}

} // namespace duckdb






namespace duckdb {

PhysicalJoin::PhysicalJoin(LogicalOperator &op, PhysicalOperatorType type, JoinType join_type,
                           idx_t estimated_cardinality)
    : CachingPhysicalOperator(type, op.types, estimated_cardinality), join_type(join_type) {
}

bool PhysicalJoin::EmptyResultIfRHSIsEmpty() const {
	// empty RHS with INNER, RIGHT or SEMI join means empty result set
	switch (join_type) {
	case JoinType::INNER:
	case JoinType::RIGHT:
	case JoinType::SEMI:
		return true;
	default:
		return false;
	}
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void PhysicalJoin::BuildJoinPipelines(Pipeline &current, MetaPipeline &meta_pipeline, PhysicalOperator &op) {
	op.op_state.reset();
	op.sink_state.reset();

	// 'current' is the probe pipeline: add this operator
	auto &state = meta_pipeline.GetState();
	state.AddPipelineOperator(current, &op);

	// save the last added pipeline to set up dependencies later (in case we need to add a child pipeline)
	vector<shared_ptr<Pipeline>> pipelines_so_far;
	meta_pipeline.GetPipelines(pipelines_so_far, false);
	auto last_pipeline = pipelines_so_far.back().get();

	// on the RHS (build side), we construct a child MetaPipeline with this operator as its sink
	auto child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, &op);
	child_meta_pipeline->Build(op.children[1].get());

	// continue building the current pipeline on the LHS (probe side)
	op.children[0]->BuildPipelines(current, meta_pipeline);

	switch (op.type) {
	case PhysicalOperatorType::POSITIONAL_JOIN:
		// Positional joins are always outer
		meta_pipeline.CreateChildPipeline(current, &op, last_pipeline);
		return;
	case PhysicalOperatorType::CROSS_PRODUCT:
		return;
	default:
		break;
	}

	// Join can become a source operator if it's RIGHT/OUTER, or if the hash join goes out-of-core
	bool add_child_pipeline = false;
	auto &join_op = (PhysicalJoin &)op;
	if (IsRightOuterJoin(join_op.join_type)) {
		if (meta_pipeline.HasRecursiveCTE()) {
			throw NotImplementedException("FULL and RIGHT outer joins are not supported in recursive CTEs yet");
		}
		add_child_pipeline = true;
	}

	if (join_op.type == PhysicalOperatorType::HASH_JOIN) {
		auto &hash_join_op = (PhysicalHashJoin &)join_op;
		hash_join_op.can_go_external = !meta_pipeline.HasRecursiveCTE();
		if (hash_join_op.can_go_external) {
			add_child_pipeline = true;
		}
	}

	if (add_child_pipeline) {
		meta_pipeline.CreateChildPipeline(current, &op, last_pipeline);
	}
}

void PhysicalJoin::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	PhysicalJoin::BuildJoinPipelines(current, meta_pipeline, *this);
}

vector<const PhysicalOperator *> PhysicalJoin::GetSources() const {
	auto result = children[0]->GetSources();
	if (IsSource()) {
		result.push_back(this);
	}
	return result;
}

} // namespace duckdb









namespace duckdb {

PhysicalNestedLoopJoin::PhysicalNestedLoopJoin(LogicalOperator &op, unique_ptr<PhysicalOperator> left,
                                               unique_ptr<PhysicalOperator> right, vector<JoinCondition> cond,
                                               JoinType join_type, idx_t estimated_cardinality)
    : PhysicalComparisonJoin(op, PhysicalOperatorType::NESTED_LOOP_JOIN, std::move(cond), join_type,
                             estimated_cardinality) {
	children.push_back(std::move(left));
	children.push_back(std::move(right));
}

bool PhysicalJoin::HasNullValues(DataChunk &chunk) {
	for (idx_t col_idx = 0; col_idx < chunk.ColumnCount(); col_idx++) {
		UnifiedVectorFormat vdata;
		chunk.data[col_idx].ToUnifiedFormat(chunk.size(), vdata);

		if (vdata.validity.AllValid()) {
			continue;
		}
		for (idx_t i = 0; i < chunk.size(); i++) {
			auto idx = vdata.sel->get_index(i);
			if (!vdata.validity.RowIsValid(idx)) {
				return true;
			}
		}
	}
	return false;
}

template <bool MATCH>
static void ConstructSemiOrAntiJoinResult(DataChunk &left, DataChunk &result, bool found_match[]) {
	D_ASSERT(left.ColumnCount() == result.ColumnCount());
	// create the selection vector from the matches that were found
	idx_t result_count = 0;
	SelectionVector sel(STANDARD_VECTOR_SIZE);
	for (idx_t i = 0; i < left.size(); i++) {
		if (found_match[i] == MATCH) {
			sel.set_index(result_count++, i);
		}
	}
	// construct the final result
	if (result_count > 0) {
		// we only return the columns on the left side
		// project them using the result selection vector
		// reference the columns of the left side from the result
		result.Slice(left, sel, result_count);
	} else {
		result.SetCardinality(0);
	}
}

void PhysicalJoin::ConstructSemiJoinResult(DataChunk &left, DataChunk &result, bool found_match[]) {
	ConstructSemiOrAntiJoinResult<true>(left, result, found_match);
}

void PhysicalJoin::ConstructAntiJoinResult(DataChunk &left, DataChunk &result, bool found_match[]) {
	ConstructSemiOrAntiJoinResult<false>(left, result, found_match);
}

void PhysicalJoin::ConstructMarkJoinResult(DataChunk &join_keys, DataChunk &left, DataChunk &result, bool found_match[],
                                           bool has_null) {
	// for the initial set of columns we just reference the left side
	result.SetCardinality(left);
	for (idx_t i = 0; i < left.ColumnCount(); i++) {
		result.data[i].Reference(left.data[i]);
	}
	auto &mark_vector = result.data.back();
	mark_vector.SetVectorType(VectorType::FLAT_VECTOR);
	// first we set the NULL values from the join keys
	// if there is any NULL in the keys, the result is NULL
	auto bool_result = FlatVector::GetData<bool>(mark_vector);
	auto &mask = FlatVector::Validity(mark_vector);
	for (idx_t col_idx = 0; col_idx < join_keys.ColumnCount(); col_idx++) {
		UnifiedVectorFormat jdata;
		join_keys.data[col_idx].ToUnifiedFormat(join_keys.size(), jdata);
		if (!jdata.validity.AllValid()) {
			for (idx_t i = 0; i < join_keys.size(); i++) {
				auto jidx = jdata.sel->get_index(i);
				mask.Set(i, jdata.validity.RowIsValid(jidx));
			}
		}
	}
	// now set the remaining entries to either true or false based on whether a match was found
	if (found_match) {
		for (idx_t i = 0; i < left.size(); i++) {
			bool_result[i] = found_match[i];
		}
	} else {
		memset(bool_result, 0, sizeof(bool) * left.size());
	}
	// if the right side contains NULL values, the result of any FALSE becomes NULL
	if (has_null) {
		for (idx_t i = 0; i < left.size(); i++) {
			if (!bool_result[i]) {
				mask.SetInvalid(i);
			}
		}
	}
}

bool PhysicalNestedLoopJoin::IsSupported(const vector<JoinCondition> &conditions, JoinType join_type) {
	if (join_type == JoinType::MARK) {
		return true;
	}
	for (auto &cond : conditions) {
		if (cond.left->return_type.InternalType() == PhysicalType::STRUCT ||
		    cond.left->return_type.InternalType() == PhysicalType::LIST) {
			return false;
		}
	}
	return true;
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class NestedLoopJoinLocalState : public LocalSinkState {
public:
	explicit NestedLoopJoinLocalState(ClientContext &context, const vector<JoinCondition> &conditions)
	    : rhs_executor(context) {
		vector<LogicalType> condition_types;
		for (auto &cond : conditions) {
			rhs_executor.AddExpression(*cond.right);
			condition_types.push_back(cond.right->return_type);
		}
		right_condition.Initialize(Allocator::Get(context), condition_types);
	}

	//! The chunk holding the right condition
	DataChunk right_condition;
	//! The executor of the RHS condition
	ExpressionExecutor rhs_executor;
};

class NestedLoopJoinGlobalState : public GlobalSinkState {
public:
	explicit NestedLoopJoinGlobalState(ClientContext &context, const PhysicalNestedLoopJoin &op)
	    : right_payload_data(context, op.children[1]->types), right_condition_data(context, op.GetJoinTypes()),
	      has_null(false), right_outer(IsRightOuterJoin(op.join_type)) {
	}

	mutex nj_lock;
	//! Materialized data of the RHS
	ColumnDataCollection right_payload_data;
	//! Materialized join condition of the RHS
	ColumnDataCollection right_condition_data;
	//! Whether or not the RHS of the nested loop join has NULL values
	atomic<bool> has_null;
	//! A bool indicating for each tuple in the RHS if they found a match (only used in FULL OUTER JOIN)
	OuterJoinMarker right_outer;
};

vector<LogicalType> PhysicalNestedLoopJoin::GetJoinTypes() const {
	vector<LogicalType> result;
	for (auto &op : conditions) {
		result.push_back(op.right->return_type);
	}
	return result;
}

SinkResultType PhysicalNestedLoopJoin::Sink(ExecutionContext &context, GlobalSinkState &state, LocalSinkState &lstate,
                                            DataChunk &input) const {
	auto &gstate = (NestedLoopJoinGlobalState &)state;
	auto &nlj_state = (NestedLoopJoinLocalState &)lstate;

	// resolve the join expression of the right side
	nlj_state.right_condition.Reset();
	nlj_state.rhs_executor.Execute(input, nlj_state.right_condition);

	// if we have not seen any NULL values yet, and we are performing a MARK join, check if there are NULL values in
	// this chunk
	if (join_type == JoinType::MARK && !gstate.has_null) {
		if (HasNullValues(nlj_state.right_condition)) {
			gstate.has_null = true;
		}
	}

	// append the payload data and the conditions
	lock_guard<mutex> nj_guard(gstate.nj_lock);
	gstate.right_payload_data.Append(input);
	gstate.right_condition_data.Append(nlj_state.right_condition);
	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalNestedLoopJoin::Combine(ExecutionContext &context, GlobalSinkState &gstate, LocalSinkState &lstate) const {
	auto &state = (NestedLoopJoinLocalState &)lstate;
	auto &client_profiler = QueryProfiler::Get(context.client);

	context.thread.profiler.Flush(this, &state.rhs_executor, "rhs_executor", 1);
	client_profiler.Flush(context.thread.profiler);
}

SinkFinalizeType PhysicalNestedLoopJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                  GlobalSinkState &gstate_p) const {
	auto &gstate = (NestedLoopJoinGlobalState &)gstate_p;
	gstate.right_outer.Initialize(gstate.right_payload_data.Count());
	if (gstate.right_payload_data.Count() == 0 && EmptyResultIfRHSIsEmpty()) {
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}
	return SinkFinalizeType::READY;
}

unique_ptr<GlobalSinkState> PhysicalNestedLoopJoin::GetGlobalSinkState(ClientContext &context) const {
	return make_unique<NestedLoopJoinGlobalState>(context, *this);
}

unique_ptr<LocalSinkState> PhysicalNestedLoopJoin::GetLocalSinkState(ExecutionContext &context) const {
	return make_unique<NestedLoopJoinLocalState>(context.client, conditions);
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
class PhysicalNestedLoopJoinState : public CachingOperatorState {
public:
	PhysicalNestedLoopJoinState(ClientContext &context, const PhysicalNestedLoopJoin &op,
	                            const vector<JoinCondition> &conditions)
	    : fetch_next_left(true), fetch_next_right(false), lhs_executor(context), left_tuple(0), right_tuple(0),
	      left_outer(IsLeftOuterJoin(op.join_type)) {
		vector<LogicalType> condition_types;
		for (auto &cond : conditions) {
			lhs_executor.AddExpression(*cond.left);
			condition_types.push_back(cond.left->return_type);
		}
		auto &allocator = Allocator::Get(context);
		left_condition.Initialize(allocator, condition_types);
		right_condition.Initialize(allocator, condition_types);
		right_payload.Initialize(allocator, op.children[1]->GetTypes());
		left_outer.Initialize(STANDARD_VECTOR_SIZE);
	}

	bool fetch_next_left;
	bool fetch_next_right;
	DataChunk left_condition;
	//! The executor of the LHS condition
	ExpressionExecutor lhs_executor;

	ColumnDataScanState condition_scan_state;
	ColumnDataScanState payload_scan_state;
	DataChunk right_condition;
	DataChunk right_payload;

	idx_t left_tuple;
	idx_t right_tuple;

	OuterJoinMarker left_outer;

public:
	void Finalize(PhysicalOperator *op, ExecutionContext &context) override {
		context.thread.profiler.Flush(op, &lhs_executor, "lhs_executor", 0);
	}
};

unique_ptr<OperatorState> PhysicalNestedLoopJoin::GetOperatorState(ExecutionContext &context) const {
	return make_unique<PhysicalNestedLoopJoinState>(context.client, *this, conditions);
}

OperatorResultType PhysicalNestedLoopJoin::ExecuteInternal(ExecutionContext &context, DataChunk &input,
                                                           DataChunk &chunk, GlobalOperatorState &gstate_p,
                                                           OperatorState &state_p) const {
	auto &gstate = (NestedLoopJoinGlobalState &)*sink_state;

	if (gstate.right_payload_data.Count() == 0) {
		// empty RHS
		if (!EmptyResultIfRHSIsEmpty()) {
			ConstructEmptyJoinResult(join_type, gstate.has_null, input, chunk);
			return OperatorResultType::NEED_MORE_INPUT;
		} else {
			return OperatorResultType::FINISHED;
		}
	}

	switch (join_type) {
	case JoinType::SEMI:
	case JoinType::ANTI:
	case JoinType::MARK:
		// simple joins can have max STANDARD_VECTOR_SIZE matches per chunk
		ResolveSimpleJoin(context, input, chunk, state_p);
		return OperatorResultType::NEED_MORE_INPUT;
	case JoinType::LEFT:
	case JoinType::INNER:
	case JoinType::OUTER:
	case JoinType::RIGHT:
		return ResolveComplexJoin(context, input, chunk, state_p);
	default:
		throw NotImplementedException("Unimplemented type for nested loop join!");
	}
}

void PhysicalNestedLoopJoin::ResolveSimpleJoin(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                               OperatorState &state_p) const {
	auto &state = (PhysicalNestedLoopJoinState &)state_p;
	auto &gstate = (NestedLoopJoinGlobalState &)*sink_state;

	// resolve the left join condition for the current chunk
	state.left_condition.Reset();
	state.lhs_executor.Execute(input, state.left_condition);

	bool found_match[STANDARD_VECTOR_SIZE] = {false};
	NestedLoopJoinMark::Perform(state.left_condition, gstate.right_condition_data, found_match, conditions);
	switch (join_type) {
	case JoinType::MARK:
		// now construct the mark join result from the found matches
		PhysicalJoin::ConstructMarkJoinResult(state.left_condition, input, chunk, found_match, gstate.has_null);
		break;
	case JoinType::SEMI:
		// construct the semi join result from the found matches
		PhysicalJoin::ConstructSemiJoinResult(input, chunk, found_match);
		break;
	case JoinType::ANTI:
		// construct the anti join result from the found matches
		PhysicalJoin::ConstructAntiJoinResult(input, chunk, found_match);
		break;
	default:
		throw NotImplementedException("Unimplemented type for simple nested loop join!");
	}
}

OperatorResultType PhysicalNestedLoopJoin::ResolveComplexJoin(ExecutionContext &context, DataChunk &input,
                                                              DataChunk &chunk, OperatorState &state_p) const {
	auto &state = (PhysicalNestedLoopJoinState &)state_p;
	auto &gstate = (NestedLoopJoinGlobalState &)*sink_state;

	idx_t match_count;
	do {
		if (state.fetch_next_right) {
			// we exhausted the chunk on the right: move to the next chunk on the right
			state.left_tuple = 0;
			state.right_tuple = 0;
			state.fetch_next_right = false;
			// check if we exhausted all chunks on the RHS
			if (gstate.right_condition_data.Scan(state.condition_scan_state, state.right_condition)) {
				if (!gstate.right_payload_data.Scan(state.payload_scan_state, state.right_payload)) {
					throw InternalException("Nested loop join: payload and conditions are unaligned!?");
				}
				if (state.right_condition.size() != state.right_payload.size()) {
					throw InternalException("Nested loop join: payload and conditions are unaligned!?");
				}
			} else {
				// we exhausted all chunks on the right: move to the next chunk on the left
				state.fetch_next_left = true;
				if (state.left_outer.Enabled()) {
					// left join: before we move to the next chunk, see if we need to output any vectors that didn't
					// have a match found
					state.left_outer.ConstructLeftJoinResult(input, chunk);
					state.left_outer.Reset();
				}
				return OperatorResultType::NEED_MORE_INPUT;
			}
		}
		if (state.fetch_next_left) {
			// resolve the left join condition for the current chunk
			state.left_condition.Reset();
			state.lhs_executor.Execute(input, state.left_condition);

			state.left_tuple = 0;
			state.right_tuple = 0;
			gstate.right_condition_data.InitializeScan(state.condition_scan_state);
			gstate.right_condition_data.Scan(state.condition_scan_state, state.right_condition);

			gstate.right_payload_data.InitializeScan(state.payload_scan_state);
			gstate.right_payload_data.Scan(state.payload_scan_state, state.right_payload);
			state.fetch_next_left = false;
		}
		// now we have a left and a right chunk that we can join together
		// note that we only get here in the case of a LEFT, INNER or FULL join
		auto &left_chunk = input;
		auto &right_condition = state.right_condition;
		auto &right_payload = state.right_payload;

		// sanity check
		left_chunk.Verify();
		right_condition.Verify();
		right_payload.Verify();

		// now perform the join
		SelectionVector lvector(STANDARD_VECTOR_SIZE), rvector(STANDARD_VECTOR_SIZE);
		match_count = NestedLoopJoinInner::Perform(state.left_tuple, state.right_tuple, state.left_condition,
		                                           right_condition, lvector, rvector, conditions);
		// we have finished resolving the join conditions
		if (match_count > 0) {
			// we have matching tuples!
			// construct the result
			state.left_outer.SetMatches(lvector, match_count);
			gstate.right_outer.SetMatches(rvector, match_count, state.condition_scan_state.current_row_index);

			chunk.Slice(input, lvector, match_count);
			chunk.Slice(right_payload, rvector, match_count, input.ColumnCount());
		}

		// check if we exhausted the RHS, if we did we need to move to the next right chunk in the next iteration
		if (state.right_tuple >= right_condition.size()) {
			state.fetch_next_right = true;
		}
	} while (match_count == 0);
	return OperatorResultType::HAVE_MORE_OUTPUT;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class NestedLoopJoinGlobalScanState : public GlobalSourceState {
public:
	explicit NestedLoopJoinGlobalScanState(const PhysicalNestedLoopJoin &op) : op(op) {
		D_ASSERT(op.sink_state);
		auto &sink = (NestedLoopJoinGlobalState &)*op.sink_state;
		sink.right_outer.InitializeScan(sink.right_payload_data, scan_state);
	}

	const PhysicalNestedLoopJoin &op;
	OuterJoinGlobalScanState scan_state;

public:
	idx_t MaxThreads() override {
		auto &sink = (NestedLoopJoinGlobalState &)*op.sink_state;
		return sink.right_outer.MaxThreads();
	}
};

class NestedLoopJoinLocalScanState : public LocalSourceState {
public:
	explicit NestedLoopJoinLocalScanState(const PhysicalNestedLoopJoin &op, NestedLoopJoinGlobalScanState &gstate) {
		D_ASSERT(op.sink_state);
		auto &sink = (NestedLoopJoinGlobalState &)*op.sink_state;
		sink.right_outer.InitializeScan(gstate.scan_state, scan_state);
	}

	OuterJoinLocalScanState scan_state;
};

unique_ptr<GlobalSourceState> PhysicalNestedLoopJoin::GetGlobalSourceState(ClientContext &context) const {
	return make_unique<NestedLoopJoinGlobalScanState>(*this);
}

unique_ptr<LocalSourceState> PhysicalNestedLoopJoin::GetLocalSourceState(ExecutionContext &context,
                                                                         GlobalSourceState &gstate) const {
	return make_unique<NestedLoopJoinLocalScanState>(*this, (NestedLoopJoinGlobalScanState &)gstate);
}

void PhysicalNestedLoopJoin::GetData(ExecutionContext &context, DataChunk &chunk, GlobalSourceState &gstate_p,
                                     LocalSourceState &lstate_p) const {
	D_ASSERT(IsRightOuterJoin(join_type));
	// check if we need to scan any unmatched tuples from the RHS for the full/right outer join
	auto &sink = (NestedLoopJoinGlobalState &)*sink_state;
	auto &gstate = (NestedLoopJoinGlobalScanState &)gstate_p;
	auto &lstate = (NestedLoopJoinLocalScanState &)lstate_p;

	// if the LHS is exhausted in a FULL/RIGHT OUTER JOIN, we scan chunks we still need to output
	sink.right_outer.Scan(gstate.scan_state, lstate.scan_state, chunk);
}

} // namespace duckdb














namespace duckdb {

PhysicalPiecewiseMergeJoin::PhysicalPiecewiseMergeJoin(LogicalOperator &op, unique_ptr<PhysicalOperator> left,
                                                       unique_ptr<PhysicalOperator> right, vector<JoinCondition> cond,
                                                       JoinType join_type, idx_t estimated_cardinality)
    : PhysicalRangeJoin(op, PhysicalOperatorType::PIECEWISE_MERGE_JOIN, std::move(left), std::move(right),
                        std::move(cond), join_type, estimated_cardinality) {

	for (auto &cond : conditions) {
		D_ASSERT(cond.left->return_type == cond.right->return_type);
		join_key_types.push_back(cond.left->return_type);

		// Convert the conditions to sort orders
		auto left = cond.left->Copy();
		auto right = cond.right->Copy();
		switch (cond.comparison) {
		case ExpressionType::COMPARE_LESSTHAN:
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			lhs_orders.emplace_back(
			    BoundOrderByNode(OrderType::ASCENDING, OrderByNullType::NULLS_LAST, std::move(left)));
			rhs_orders.emplace_back(
			    BoundOrderByNode(OrderType::ASCENDING, OrderByNullType::NULLS_LAST, std::move(right)));
			break;
		case ExpressionType::COMPARE_GREATERTHAN:
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			lhs_orders.emplace_back(
			    BoundOrderByNode(OrderType::DESCENDING, OrderByNullType::NULLS_LAST, std::move(left)));
			rhs_orders.emplace_back(
			    BoundOrderByNode(OrderType::DESCENDING, OrderByNullType::NULLS_LAST, std::move(right)));
			break;
		case ExpressionType::COMPARE_NOTEQUAL:
		case ExpressionType::COMPARE_DISTINCT_FROM:
			// Allowed in multi-predicate joins, but can't be first/sort.
			D_ASSERT(!lhs_orders.empty());
			lhs_orders.emplace_back(BoundOrderByNode(OrderType::INVALID, OrderByNullType::NULLS_LAST, std::move(left)));
			rhs_orders.emplace_back(
			    BoundOrderByNode(OrderType::INVALID, OrderByNullType::NULLS_LAST, std::move(right)));
			break;

		default:
			// COMPARE EQUAL not supported with merge join
			throw NotImplementedException("Unimplemented join type for merge join");
		}
	}
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
class MergeJoinLocalState : public LocalSinkState {
public:
	explicit MergeJoinLocalState(ClientContext &context, const PhysicalRangeJoin &op, const idx_t child)
	    : table(context, op, child) {
	}

	//! The local sort state
	PhysicalRangeJoin::LocalSortedTable table;
};

class MergeJoinGlobalState : public GlobalSinkState {
public:
	using GlobalSortedTable = PhysicalRangeJoin::GlobalSortedTable;

public:
	MergeJoinGlobalState(ClientContext &context, const PhysicalPiecewiseMergeJoin &op) {
		RowLayout rhs_layout;
		rhs_layout.Initialize(op.children[1]->types);
		vector<BoundOrderByNode> rhs_order;
		rhs_order.emplace_back(op.rhs_orders[0].Copy());
		table = make_unique<GlobalSortedTable>(context, rhs_order, rhs_layout);
	}

	inline idx_t Count() const {
		return table->count;
	}

	void Sink(DataChunk &input, MergeJoinLocalState &lstate) {
		auto &global_sort_state = table->global_sort_state;
		auto &local_sort_state = lstate.table.local_sort_state;

		// Sink the data into the local sort state
		lstate.table.Sink(input, global_sort_state);

		// When sorting data reaches a certain size, we sort it
		if (local_sort_state.SizeInBytes() >= table->memory_per_thread) {
			local_sort_state.Sort(global_sort_state, true);
		}
	}

	unique_ptr<GlobalSortedTable> table;
};

unique_ptr<GlobalSinkState> PhysicalPiecewiseMergeJoin::GetGlobalSinkState(ClientContext &context) const {
	return make_unique<MergeJoinGlobalState>(context, *this);
}

unique_ptr<LocalSinkState> PhysicalPiecewiseMergeJoin::GetLocalSinkState(ExecutionContext &context) const {
	// We only sink the RHS
	return make_unique<MergeJoinLocalState>(context.client, *this, 1);
}

SinkResultType PhysicalPiecewiseMergeJoin::Sink(ExecutionContext &context, GlobalSinkState &gstate_p,
                                                LocalSinkState &lstate_p, DataChunk &input) const {
	auto &gstate = (MergeJoinGlobalState &)gstate_p;
	auto &lstate = (MergeJoinLocalState &)lstate_p;

	gstate.Sink(input, lstate);

	return SinkResultType::NEED_MORE_INPUT;
}

void PhysicalPiecewiseMergeJoin::Combine(ExecutionContext &context, GlobalSinkState &gstate_p,
                                         LocalSinkState &lstate_p) const {
	auto &gstate = (MergeJoinGlobalState &)gstate_p;
	auto &lstate = (MergeJoinLocalState &)lstate_p;
	gstate.table->Combine(lstate.table);
	auto &client_profiler = QueryProfiler::Get(context.client);

	context.thread.profiler.Flush(this, &lstate.table.executor, "rhs_executor", 1);
	client_profiler.Flush(context.thread.profiler);
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//
SinkFinalizeType PhysicalPiecewiseMergeJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                      GlobalSinkState &gstate_p) const {
	auto &gstate = (MergeJoinGlobalState &)gstate_p;
	auto &global_sort_state = gstate.table->global_sort_state;

	if (IsRightOuterJoin(join_type)) {
		// for FULL/RIGHT OUTER JOIN, initialize found_match to false for every tuple
		gstate.table->IntializeMatches();
	}
	if (global_sort_state.sorted_blocks.empty() && EmptyResultIfRHSIsEmpty()) {
		// Empty input!
		return SinkFinalizeType::NO_OUTPUT_POSSIBLE;
	}

	// Sort the current input child
	gstate.table->Finalize(pipeline, event);

	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Operator
//===--------------------------------------------------------------------===//
class PiecewiseMergeJoinState : public CachingOperatorState {
public:
	using LocalSortedTable = PhysicalRangeJoin::LocalSortedTable;

	PiecewiseMergeJoinState(ClientContext &context, const PhysicalPiecewiseMergeJoin &op, bool force_external)
	    : context(context), allocator(Allocator::Get(context)), op(op),
	      buffer_manager(BufferManager::GetBufferManager(context)), force_external(force_external),
	      left_outer(IsLeftOuterJoin(op.join_type)), left_position(0), first_fetch(true), finished(true),
	      right_position(0), right_chunk_index(0), rhs_executor(context) {
		vector<LogicalType> condition_types;
		for (auto &order : op.lhs_orders) {
			condition_types.push_back(order.expression->return_type);
		}
		left_outer.Initialize(STANDARD_VECTOR_SIZE);
		lhs_layout.Initialize(op.children[0]->types);
		lhs_payload.Initialize(allocator, op.children[0]->types);

		lhs_order.emplace_back(op.lhs_orders[0].Copy());

		// Set up shared data for multiple predicates
		sel.Initialize(STANDARD_VECTOR_SIZE);
		condition_types.clear();
		for (auto &order : op.rhs_orders) {
			rhs_executor.AddExpression(*order.expression);
			condition_types.push_back(order.expression->return_type);
		}
		rhs_keys.Initialize(allocator, condition_types);
	}

	ClientContext &context;
	Allocator &allocator;
	const PhysicalPiecewiseMergeJoin &op;
	BufferManager &buffer_manager;
	bool force_external;

	// Block sorting
	DataChunk lhs_payload;
	OuterJoinMarker left_outer;
	vector<BoundOrderByNode> lhs_order;
	RowLayout lhs_layout;
	unique_ptr<LocalSortedTable> lhs_local_table;
	unique_ptr<GlobalSortState> lhs_global_state;
	unique_ptr<PayloadScanner> scanner;

	// Simple scans
	idx_t left_position;

	// Complex scans
	bool first_fetch;
	bool finished;
	idx_t right_position;
	idx_t right_chunk_index;
	idx_t right_base;

	// Secondary predicate shared data
	SelectionVector sel;
	DataChunk rhs_keys;
	DataChunk rhs_input;
	ExpressionExecutor rhs_executor;
	vector<BufferHandle> payload_heap_handles;

public:
	void ResolveJoinKeys(DataChunk &input) {
		// sort by join key
		lhs_global_state = make_unique<GlobalSortState>(buffer_manager, lhs_order, lhs_layout);
		lhs_local_table = make_unique<LocalSortedTable>(context, op, 0);
		lhs_local_table->Sink(input, *lhs_global_state);

		// Set external (can be forced with the PRAGMA)
		lhs_global_state->external = force_external;
		lhs_global_state->AddLocalState(lhs_local_table->local_sort_state);
		lhs_global_state->PrepareMergePhase();
		while (lhs_global_state->sorted_blocks.size() > 1) {
			MergeSorter merge_sorter(*lhs_global_state, buffer_manager);
			merge_sorter.PerformInMergeRound();
			lhs_global_state->CompleteMergeRound();
		}

		// Scan the sorted payload
		D_ASSERT(lhs_global_state->sorted_blocks.size() == 1);

		scanner = make_unique<PayloadScanner>(*lhs_global_state->sorted_blocks[0]->payload_data, *lhs_global_state);
		lhs_payload.Reset();
		scanner->Scan(lhs_payload);

		// Recompute the sorted keys from the sorted input
		lhs_local_table->keys.Reset();
		lhs_local_table->executor.Execute(lhs_payload, lhs_local_table->keys);
	}

	void Finalize(PhysicalOperator *op, ExecutionContext &context) override {
		if (lhs_local_table) {
			context.thread.profiler.Flush(op, &lhs_local_table->executor, "lhs_executor", 0);
		}
	}
};

unique_ptr<OperatorState> PhysicalPiecewiseMergeJoin::GetOperatorState(ExecutionContext &context) const {
	auto &config = ClientConfig::GetConfig(context.client);
	return make_unique<PiecewiseMergeJoinState>(context.client, *this, config.force_external);
}

static inline idx_t SortedBlockNotNull(const idx_t base, const idx_t count, const idx_t not_null) {
	return MinValue(base + count, MaxValue(base, not_null)) - base;
}

static int MergeJoinComparisonValue(ExpressionType comparison) {
	switch (comparison) {
	case ExpressionType::COMPARE_LESSTHAN:
	case ExpressionType::COMPARE_GREATERTHAN:
		return -1;
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return 0;
	default:
		throw InternalException("Unimplemented comparison type for merge join!");
	}
}

struct BlockMergeInfo {
	GlobalSortState &state;
	//! The block being scanned
	const idx_t block_idx;
	//! The number of not-NULL values in the block (they are at the end)
	const idx_t not_null;
	//! The current offset in the block
	idx_t &entry_idx;
	SelectionVector result;

	BlockMergeInfo(GlobalSortState &state, idx_t block_idx, idx_t &entry_idx, idx_t not_null)
	    : state(state), block_idx(block_idx), not_null(not_null), entry_idx(entry_idx), result(STANDARD_VECTOR_SIZE) {
	}
};

static void MergeJoinPinSortingBlock(SBScanState &scan, const idx_t block_idx) {
	scan.SetIndices(block_idx, 0);
	scan.PinRadix(block_idx);

	auto &sd = *scan.sb->blob_sorting_data;
	if (block_idx < sd.data_blocks.size()) {
		scan.PinData(sd);
	}
}

static data_ptr_t MergeJoinRadixPtr(SBScanState &scan, const idx_t entry_idx) {
	scan.entry_idx = entry_idx;
	return scan.RadixPtr();
}

static idx_t MergeJoinSimpleBlocks(PiecewiseMergeJoinState &lstate, MergeJoinGlobalState &rstate, bool *found_match,
                                   const ExpressionType comparison) {
	const auto cmp = MergeJoinComparisonValue(comparison);

	// The sort parameters should all be the same
	auto &lsort = *lstate.lhs_global_state;
	auto &rsort = rstate.table->global_sort_state;
	D_ASSERT(lsort.sort_layout.all_constant == rsort.sort_layout.all_constant);
	const auto all_constant = lsort.sort_layout.all_constant;
	D_ASSERT(lsort.external == rsort.external);
	const auto external = lsort.external;

	// There should only be one sorted block if they have been sorted
	D_ASSERT(lsort.sorted_blocks.size() == 1);
	SBScanState lread(lsort.buffer_manager, lsort);
	lread.sb = lsort.sorted_blocks[0].get();

	const idx_t l_block_idx = 0;
	idx_t l_entry_idx = 0;
	const auto lhs_not_null = lstate.lhs_local_table->count - lstate.lhs_local_table->has_null;
	MergeJoinPinSortingBlock(lread, l_block_idx);
	auto l_ptr = MergeJoinRadixPtr(lread, l_entry_idx);

	D_ASSERT(rsort.sorted_blocks.size() == 1);
	SBScanState rread(rsort.buffer_manager, rsort);
	rread.sb = rsort.sorted_blocks[0].get();

	const auto cmp_size = lsort.sort_layout.comparison_size;
	const auto entry_size = lsort.sort_layout.entry_size;

	idx_t right_base = 0;
	for (idx_t r_block_idx = 0; r_block_idx < rread.sb->radix_sorting_data.size(); r_block_idx++) {
		// we only care about the BIGGEST value in each of the RHS data blocks
		// because we want to figure out if the LHS values are less than [or equal] to ANY value
		// get the biggest value from the RHS chunk
		MergeJoinPinSortingBlock(rread, r_block_idx);

		auto &rblock = *rread.sb->radix_sorting_data[r_block_idx];
		const auto r_not_null =
		    SortedBlockNotNull(right_base, rblock.count, rstate.table->count - rstate.table->has_null);
		if (r_not_null == 0) {
			break;
		}
		const auto r_entry_idx = r_not_null - 1;
		right_base += rblock.count;

		auto r_ptr = MergeJoinRadixPtr(rread, r_entry_idx);

		// now we start from the current lpos value and check if we found a new value that is [<= OR <] the max RHS
		// value
		while (true) {
			int comp_res;
			if (all_constant) {
				comp_res = FastMemcmp(l_ptr, r_ptr, cmp_size);
			} else {
				lread.entry_idx = l_entry_idx;
				rread.entry_idx = r_entry_idx;
				comp_res = Comparators::CompareTuple(lread, rread, l_ptr, r_ptr, lsort.sort_layout, external);
			}

			if (comp_res <= cmp) {
				// found a match for lpos, set it in the found_match vector
				found_match[l_entry_idx] = true;
				l_entry_idx++;
				l_ptr += entry_size;
				if (l_entry_idx >= lhs_not_null) {
					// early out: we exhausted the entire LHS and they all match
					return 0;
				}
			} else {
				// we found no match: any subsequent value from the LHS we scan now will be bigger and thus also not
				// match move to the next RHS chunk
				break;
			}
		}
	}
	return 0;
}

void PhysicalPiecewiseMergeJoin::ResolveSimpleJoin(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                   OperatorState &state_p) const {
	auto &state = (PiecewiseMergeJoinState &)state_p;
	auto &gstate = (MergeJoinGlobalState &)*sink_state;

	state.ResolveJoinKeys(input);
	auto &lhs_table = *state.lhs_local_table;

	// perform the actual join
	bool found_match[STANDARD_VECTOR_SIZE];
	memset(found_match, 0, sizeof(found_match));
	MergeJoinSimpleBlocks(state, gstate, found_match, conditions[0].comparison);

	// use the sorted payload
	const auto lhs_not_null = lhs_table.count - lhs_table.has_null;
	auto &payload = state.lhs_payload;

	// now construct the result based on the join result
	switch (join_type) {
	case JoinType::MARK: {
		// The only part of the join keys that is actually used is the validity mask.
		// Since the payload is sorted, we can just set the tail end of the validity masks to invalid.
		for (auto &key : lhs_table.keys.data) {
			key.Flatten(lhs_table.keys.size());
			auto &mask = FlatVector::Validity(key);
			if (mask.AllValid()) {
				continue;
			}
			mask.SetAllValid(lhs_not_null);
			for (idx_t i = lhs_not_null; i < lhs_table.count; ++i) {
				mask.SetInvalid(i);
			}
		}
		// So we make a set of keys that have the validity mask set for the
		PhysicalJoin::ConstructMarkJoinResult(lhs_table.keys, payload, chunk, found_match, gstate.table->has_null);
		break;
	}
	case JoinType::SEMI:
		PhysicalJoin::ConstructSemiJoinResult(payload, chunk, found_match);
		break;
	case JoinType::ANTI:
		PhysicalJoin::ConstructAntiJoinResult(payload, chunk, found_match);
		break;
	default:
		throw NotImplementedException("Unimplemented join type for merge join");
	}
}

static idx_t MergeJoinComplexBlocks(BlockMergeInfo &l, BlockMergeInfo &r, const ExpressionType comparison) {
	const auto cmp = MergeJoinComparisonValue(comparison);

	// The sort parameters should all be the same
	D_ASSERT(l.state.sort_layout.all_constant == r.state.sort_layout.all_constant);
	const auto all_constant = r.state.sort_layout.all_constant;
	D_ASSERT(l.state.external == r.state.external);
	const auto external = l.state.external;

	// There should only be one sorted block if they have been sorted
	D_ASSERT(l.state.sorted_blocks.size() == 1);
	SBScanState lread(l.state.buffer_manager, l.state);
	lread.sb = l.state.sorted_blocks[0].get();
	D_ASSERT(lread.sb->radix_sorting_data.size() == 1);
	MergeJoinPinSortingBlock(lread, l.block_idx);
	auto l_start = MergeJoinRadixPtr(lread, 0);
	auto l_ptr = MergeJoinRadixPtr(lread, l.entry_idx);

	D_ASSERT(r.state.sorted_blocks.size() == 1);
	SBScanState rread(r.state.buffer_manager, r.state);
	rread.sb = r.state.sorted_blocks[0].get();

	if (r.entry_idx >= r.not_null) {
		return 0;
	}

	MergeJoinPinSortingBlock(rread, r.block_idx);
	auto r_ptr = MergeJoinRadixPtr(rread, r.entry_idx);

	const auto cmp_size = l.state.sort_layout.comparison_size;
	const auto entry_size = l.state.sort_layout.entry_size;

	idx_t result_count = 0;
	while (true) {
		if (l.entry_idx < l.not_null) {
			int comp_res;
			if (all_constant) {
				comp_res = FastMemcmp(l_ptr, r_ptr, cmp_size);
			} else {
				lread.entry_idx = l.entry_idx;
				rread.entry_idx = r.entry_idx;
				comp_res = Comparators::CompareTuple(lread, rread, l_ptr, r_ptr, l.state.sort_layout, external);
			}

			if (comp_res <= cmp) {
				// left side smaller: found match
				l.result.set_index(result_count, sel_t(l.entry_idx));
				r.result.set_index(result_count, sel_t(r.entry_idx));
				result_count++;
				// move left side forward
				l.entry_idx++;
				l_ptr += entry_size;
				if (result_count == STANDARD_VECTOR_SIZE) {
					// out of space!
					break;
				}
				continue;
			}
		}
		// right side smaller or equal, or left side exhausted: move
		// right pointer forward reset left side to start
		r.entry_idx++;
		if (r.entry_idx >= r.not_null) {
			break;
		}
		r_ptr += entry_size;

		l_ptr = l_start;
		l.entry_idx = 0;
	}

	return result_count;
}

OperatorResultType PhysicalPiecewiseMergeJoin::ResolveComplexJoin(ExecutionContext &context, DataChunk &input,
                                                                  DataChunk &chunk, OperatorState &state_p) const {
	auto &state = (PiecewiseMergeJoinState &)state_p;
	auto &gstate = (MergeJoinGlobalState &)*sink_state;
	auto &rsorted = *gstate.table->global_sort_state.sorted_blocks[0];
	const auto left_cols = input.ColumnCount();
	const auto tail_cols = conditions.size() - 1;

	state.payload_heap_handles.clear();
	do {
		if (state.first_fetch) {
			state.ResolveJoinKeys(input);

			state.right_chunk_index = 0;
			state.right_base = 0;
			state.left_position = 0;
			state.right_position = 0;
			state.first_fetch = false;
			state.finished = false;
		}
		if (state.finished) {
			if (state.left_outer.Enabled()) {
				// left join: before we move to the next chunk, see if we need to output any vectors that didn't
				// have a match found
				state.left_outer.ConstructLeftJoinResult(state.lhs_payload, chunk);
				state.left_outer.Reset();
			}
			state.first_fetch = true;
			state.finished = false;
			return OperatorResultType::NEED_MORE_INPUT;
		}

		auto &lhs_table = *state.lhs_local_table;
		const auto lhs_not_null = lhs_table.count - lhs_table.has_null;
		BlockMergeInfo left_info(*state.lhs_global_state, 0, state.left_position, lhs_not_null);

		const auto &rblock = *rsorted.radix_sorting_data[state.right_chunk_index];
		const auto rhs_not_null =
		    SortedBlockNotNull(state.right_base, rblock.count, gstate.table->count - gstate.table->has_null);
		BlockMergeInfo right_info(gstate.table->global_sort_state, state.right_chunk_index, state.right_position,
		                          rhs_not_null);

		idx_t result_count = MergeJoinComplexBlocks(left_info, right_info, conditions[0].comparison);
		if (result_count == 0) {
			// exhausted this chunk on the right side
			// move to the next right chunk
			state.left_position = 0;
			state.right_position = 0;
			state.right_base += rsorted.radix_sorting_data[state.right_chunk_index]->count;
			state.right_chunk_index++;
			if (state.right_chunk_index >= rsorted.radix_sorting_data.size()) {
				state.finished = true;
			}
		} else {
			// found matches: extract them
			chunk.Reset();
			for (idx_t c = 0; c < state.lhs_payload.ColumnCount(); ++c) {
				chunk.data[c].Slice(state.lhs_payload.data[c], left_info.result, result_count);
			}
			state.payload_heap_handles.push_back(SliceSortedPayload(chunk, right_info.state, right_info.block_idx,
			                                                        right_info.result, result_count, left_cols));
			chunk.SetCardinality(result_count);

			auto sel = FlatVector::IncrementalSelectionVector();
			if (tail_cols) {
				// If there are more expressions to compute,
				// split the result chunk into the left and right halves
				// so we can compute the values for comparison.
				chunk.Split(state.rhs_input, left_cols);
				state.rhs_executor.SetChunk(state.rhs_input);
				state.rhs_keys.Reset();

				auto tail_count = result_count;
				for (size_t cmp_idx = 1; cmp_idx < conditions.size(); ++cmp_idx) {
					Vector left(lhs_table.keys.data[cmp_idx]);
					left.Slice(left_info.result, result_count);

					auto &right = state.rhs_keys.data[cmp_idx];
					state.rhs_executor.ExecuteExpression(cmp_idx, right);

					if (tail_count < result_count) {
						left.Slice(*sel, tail_count);
						right.Slice(*sel, tail_count);
					}
					tail_count =
					    SelectJoinTail(conditions[cmp_idx].comparison, left, right, sel, tail_count, &state.sel);
					sel = &state.sel;
				}
				chunk.Fuse(state.rhs_input);

				if (tail_count < result_count) {
					result_count = tail_count;
					chunk.Slice(*sel, result_count);
				}
			}

			// found matches: mark the found matches if required
			if (state.left_outer.Enabled()) {
				for (idx_t i = 0; i < result_count; i++) {
					state.left_outer.SetMatch(left_info.result[sel->get_index(i)]);
				}
			}
			if (gstate.table->found_match) {
				//	Absolute position of the block + start position inside that block
				for (idx_t i = 0; i < result_count; i++) {
					gstate.table->found_match[state.right_base + right_info.result[sel->get_index(i)]] = true;
				}
			}
			chunk.SetCardinality(result_count);
			chunk.Verify();
		}
	} while (chunk.size() == 0);
	return OperatorResultType::HAVE_MORE_OUTPUT;
}

OperatorResultType PhysicalPiecewiseMergeJoin::ExecuteInternal(ExecutionContext &context, DataChunk &input,
                                                               DataChunk &chunk, GlobalOperatorState &gstate_p,
                                                               OperatorState &state) const {
	auto &gstate = (MergeJoinGlobalState &)*sink_state;

	if (gstate.Count() == 0) {
		// empty RHS
		if (!EmptyResultIfRHSIsEmpty()) {
			ConstructEmptyJoinResult(join_type, gstate.table->has_null, input, chunk);
			return OperatorResultType::NEED_MORE_INPUT;
		} else {
			return OperatorResultType::FINISHED;
		}
	}

	input.Verify();
	switch (join_type) {
	case JoinType::SEMI:
	case JoinType::ANTI:
	case JoinType::MARK:
		// simple joins can have max STANDARD_VECTOR_SIZE matches per chunk
		ResolveSimpleJoin(context, input, chunk, state);
		return OperatorResultType::NEED_MORE_INPUT;
	case JoinType::LEFT:
	case JoinType::INNER:
	case JoinType::RIGHT:
	case JoinType::OUTER:
		return ResolveComplexJoin(context, input, chunk, state);
	default:
		throw NotImplementedException("Unimplemented type for piecewise merge loop join!");
	}
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
class PiecewiseJoinScanState : public GlobalSourceState {
public:
	explicit PiecewiseJoinScanState(const PhysicalPiecewiseMergeJoin &op) : op(op), right_outer_position(0) {
	}

	mutex lock;
	const PhysicalPiecewiseMergeJoin &op;
	unique_ptr<PayloadScanner> scanner;
	idx_t right_outer_position;

public:
	idx_t MaxThreads() override {
		auto &sink = (MergeJoinGlobalState &)*op.sink_state;
		return sink.Count() / (STANDARD_VECTOR_SIZE * idx_t(10));
	}
};

unique_ptr<GlobalSourceState> PhysicalPiecewiseMergeJoin::GetGlobalSourceState(ClientContext &context) const {
	return make_unique<PiecewiseJoinScanState>(*this);
}

void PhysicalPiecewiseMergeJoin::GetData(ExecutionContext &context, DataChunk &result, GlobalSourceState &gstate,
                                         LocalSourceState &lstate) const {
	D_ASSERT(IsRightOuterJoin(join_type));
	// check if we need to scan any unmatched tuples from the RHS for the full/right outer join
	auto &sink = (MergeJoinGlobalState &)*sink_state;
	auto &state = (PiecewiseJoinScanState &)gstate;

	lock_guard<mutex> l(state.lock);
	if (!state.scanner) {
		// Initialize scanner (if not yet initialized)
		auto &sort_state = sink.table->global_sort_state;
		if (sort_state.sorted_blocks.empty()) {
			return;
		}
		state.scanner = make_unique<PayloadScanner>(*sort_state.sorted_blocks[0]->payload_data, sort_state);
	}

	// if the LHS is exhausted in a FULL/RIGHT OUTER JOIN, we scan the found_match for any chunks we
	// still need to output
	const auto found_match = sink.table->found_match.get();

	DataChunk rhs_chunk;
	rhs_chunk.Initialize(Allocator::Get(context.client), sink.table->global_sort_state.payload_layout.GetTypes());
	SelectionVector rsel(STANDARD_VECTOR_SIZE);
	for (;;) {
		// Read the next sorted chunk
		state.scanner->Scan(rhs_chunk);

		const auto count = rhs_chunk.size();
		if (count == 0) {
			return;
		}

		idx_t result_count = 0;
		// figure out which tuples didn't find a match in the RHS
		for (idx_t i = 0; i < count; i++) {
			if (!found_match[state.right_outer_position + i]) {
				rsel.set_index(result_count++, i);
			}
		}
		state.right_outer_position += count;

		if (result_count > 0) {
			// if there were any tuples that didn't find a match, output them
			const idx_t left_column_count = children[0]->types.size();
			for (idx_t col_idx = 0; col_idx < left_column_count; ++col_idx) {
				result.data[col_idx].SetVectorType(VectorType::CONSTANT_VECTOR);
				ConstantVector::SetNull(result.data[col_idx], true);
			}
			const idx_t right_column_count = children[1]->types.size();
			;
			for (idx_t col_idx = 0; col_idx < right_column_count; ++col_idx) {
				result.data[left_column_count + col_idx].Slice(rhs_chunk.data[col_idx], rsel, result_count);
			}
			result.SetCardinality(result_count);
			return;
		}
	}
}

} // namespace duckdb
