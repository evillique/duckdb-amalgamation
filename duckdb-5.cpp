// See https://raw.githubusercontent.com/duckdb/duckdb/master/LICENSE for licensing information

#include "duckdb.hpp"
#include "duckdb-internal.hpp"
#ifndef DUCKDB_AMALGAMATION
#error header mismatch
#endif


namespace duckdb {

void BuiltinFunctions::RegisterHolisticAggregates() {
	Register<QuantileFun>();
	Register<ModeFun>();
	Register<ApproximateQuantileFun>();
	Register<ReservoirQuantileFun>();
}

} // namespace duckdb







namespace duckdb {

struct HistogramFunctor {
	template <class T, class MAP_TYPE = map<T, idx_t>>
	static void HistogramUpdate(UnifiedVectorFormat &sdata, UnifiedVectorFormat &input_data, idx_t count) {

		auto states = (HistogramAggState<T, MAP_TYPE> **)sdata.data;
		for (idx_t i = 0; i < count; i++) {
			if (input_data.validity.RowIsValid(input_data.sel->get_index(i))) {
				auto state = states[sdata.sel->get_index(i)];
				if (!state->hist) {
					state->hist = new MAP_TYPE();
				}
				auto value = (T *)input_data.data;
				(*state->hist)[value[input_data.sel->get_index(i)]]++;
			}
		}
	}

	template <class T>
	static Value HistogramFinalize(T first) {
		return Value::CreateValue(first);
	}
};

struct HistogramStringFunctor {
	template <class T, class MAP_TYPE = map<T, idx_t>>
	static void HistogramUpdate(UnifiedVectorFormat &sdata, UnifiedVectorFormat &input_data, idx_t count) {

		auto states = (HistogramAggState<T, MAP_TYPE> **)sdata.data;
		for (idx_t i = 0; i < count; i++) {
			if (input_data.validity.RowIsValid(input_data.sel->get_index(i))) {
				auto state = states[sdata.sel->get_index(i)];
				if (!state->hist) {
					state->hist = new MAP_TYPE();
				}
				auto value = (string_t *)input_data.data;
				(*state->hist)[value[input_data.sel->get_index(i)].GetString()]++;
			}
		}
	}

	template <class T>
	static Value HistogramFinalize(T first) {
		string_t value = first;
		return Value::CreateValue(value);
	}
};

struct HistogramFunction {
	template <class STATE>
	static void Initialize(STATE *state) {
		state->hist = nullptr;
	}

	template <class STATE>
	static void Destroy(STATE *state) {
		if (state->hist) {
			delete state->hist;
		}
	}

	static bool IgnoreNull() {
		return true;
	}
};

template <class OP, class T, class MAP_TYPE>
static void HistogramUpdateFunction(Vector inputs[], AggregateInputData &, idx_t input_count, Vector &state_vector,
                                    idx_t count) {

	D_ASSERT(input_count == 1);

	auto &input = inputs[0];
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	UnifiedVectorFormat input_data;
	input.ToUnifiedFormat(count, input_data);

	OP::template HistogramUpdate<T, MAP_TYPE>(sdata, input_data, count);
}

template <class T, class MAP_TYPE>
static void HistogramCombineFunction(Vector &state, Vector &combined, AggregateInputData &, idx_t count) {

	UnifiedVectorFormat sdata;
	state.ToUnifiedFormat(count, sdata);
	auto states_ptr = (HistogramAggState<T, MAP_TYPE> **)sdata.data;

	auto combined_ptr = FlatVector::GetData<HistogramAggState<T, MAP_TYPE> *>(combined);

	for (idx_t i = 0; i < count; i++) {
		auto state = states_ptr[sdata.sel->get_index(i)];
		if (!state->hist) {
			continue;
		}
		if (!combined_ptr[i]->hist) {
			combined_ptr[i]->hist = new MAP_TYPE();
		}
		D_ASSERT(combined_ptr[i]->hist);
		D_ASSERT(state->hist);
		for (auto &entry : *state->hist) {
			(*combined_ptr[i]->hist)[entry.first] += entry.second;
		}
	}
}

template <class OP, class T, class MAP_TYPE>
static void HistogramFinalizeFunction(Vector &state_vector, AggregateInputData &, Vector &result, idx_t count,
                                      idx_t offset) {

	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (HistogramAggState<T, MAP_TYPE> **)sdata.data;

	auto &mask = FlatVector::Validity(result);
	auto old_len = ListVector::GetListSize(result);

	for (idx_t i = 0; i < count; i++) {
		const auto rid = i + offset;
		auto state = states[sdata.sel->get_index(i)];
		if (!state->hist) {
			mask.SetInvalid(rid);
			continue;
		}

		for (auto &entry : *state->hist) {
			Value bucket_value = OP::template HistogramFinalize<T>(entry.first);
			auto count_value = Value::CreateValue(entry.second);
			auto struct_value =
			    Value::STRUCT({std::make_pair("key", bucket_value), std::make_pair("value", count_value)});
			ListVector::PushBack(result, struct_value);
		}

		auto list_struct_data = ListVector::GetData(result);
		list_struct_data[rid].length = ListVector::GetListSize(result) - old_len;
		list_struct_data[rid].offset = old_len;
		old_len += list_struct_data[rid].length;
	}
}

unique_ptr<FunctionData> HistogramBindFunction(ClientContext &context, AggregateFunction &function,
                                               vector<unique_ptr<Expression>> &arguments) {

	D_ASSERT(arguments.size() == 1);

	if (arguments[0]->return_type.id() == LogicalTypeId::LIST ||
	    arguments[0]->return_type.id() == LogicalTypeId::STRUCT ||
	    arguments[0]->return_type.id() == LogicalTypeId::MAP) {
		throw NotImplementedException("Unimplemented type for histogram %s", arguments[0]->return_type.ToString());
	}

	auto struct_type = LogicalType::MAP(arguments[0]->return_type, LogicalType::UBIGINT);

	function.return_type = struct_type;
	return make_unique<VariableReturnBindData>(function.return_type);
}

template <class OP, class T, class MAP_TYPE = map<T, idx_t>>
static AggregateFunction GetHistogramFunction(const LogicalType &type) {

	using STATE_TYPE = HistogramAggState<T, MAP_TYPE>;

	return AggregateFunction("histogram", {type}, LogicalTypeId::MAP, AggregateFunction::StateSize<STATE_TYPE>,
	                         AggregateFunction::StateInitialize<STATE_TYPE, HistogramFunction>,
	                         HistogramUpdateFunction<OP, T, MAP_TYPE>, HistogramCombineFunction<T, MAP_TYPE>,
	                         HistogramFinalizeFunction<OP, T, MAP_TYPE>, nullptr, HistogramBindFunction,
	                         AggregateFunction::StateDestroy<STATE_TYPE, HistogramFunction>);
}

template <class OP, class T, bool IS_ORDERED>
AggregateFunction GetMapType(const LogicalType &type) {

	if (IS_ORDERED) {
		return GetHistogramFunction<OP, T>(type);
	}
	return GetHistogramFunction<OP, T, unordered_map<T, idx_t>>(type);
}

template <bool IS_ORDERED = true>
AggregateFunction GetHistogramFunction(const LogicalType &type) {

	switch (type.id()) {
	case LogicalType::BOOLEAN:
		return GetMapType<HistogramFunctor, bool, IS_ORDERED>(type);
	case LogicalType::UTINYINT:
		return GetMapType<HistogramFunctor, uint8_t, IS_ORDERED>(type);
	case LogicalType::USMALLINT:
		return GetMapType<HistogramFunctor, uint16_t, IS_ORDERED>(type);
	case LogicalType::UINTEGER:
		return GetMapType<HistogramFunctor, uint32_t, IS_ORDERED>(type);
	case LogicalType::UBIGINT:
		return GetMapType<HistogramFunctor, uint64_t, IS_ORDERED>(type);
	case LogicalType::TINYINT:
		return GetMapType<HistogramFunctor, int8_t, IS_ORDERED>(type);
	case LogicalType::SMALLINT:
		return GetMapType<HistogramFunctor, int16_t, IS_ORDERED>(type);
	case LogicalType::INTEGER:
		return GetMapType<HistogramFunctor, int32_t, IS_ORDERED>(type);
	case LogicalType::BIGINT:
		return GetMapType<HistogramFunctor, int64_t, IS_ORDERED>(type);
	case LogicalType::FLOAT:
		return GetMapType<HistogramFunctor, float, IS_ORDERED>(type);
	case LogicalType::DOUBLE:
		return GetMapType<HistogramFunctor, double, IS_ORDERED>(type);
	case LogicalType::VARCHAR:
		return GetMapType<HistogramStringFunctor, string, IS_ORDERED>(type);
	case LogicalType::TIMESTAMP:
		return GetMapType<HistogramFunctor, timestamp_t, IS_ORDERED>(type);
	case LogicalType::TIMESTAMP_TZ:
		return GetMapType<HistogramFunctor, timestamp_tz_t, IS_ORDERED>(type);
	case LogicalType::TIMESTAMP_S:
		return GetMapType<HistogramFunctor, timestamp_sec_t, IS_ORDERED>(type);
	case LogicalType::TIMESTAMP_MS:
		return GetMapType<HistogramFunctor, timestamp_ms_t, IS_ORDERED>(type);
	case LogicalType::TIMESTAMP_NS:
		return GetMapType<HistogramFunctor, timestamp_ns_t, IS_ORDERED>(type);
	case LogicalType::TIME:
		return GetMapType<HistogramFunctor, dtime_t, IS_ORDERED>(type);
	case LogicalType::TIME_TZ:
		return GetMapType<HistogramFunctor, dtime_tz_t, IS_ORDERED>(type);
	case LogicalType::DATE:
		return GetMapType<HistogramFunctor, date_t, IS_ORDERED>(type);
	default:
		throw InternalException("Unimplemented histogram aggregate");
	}
}

void HistogramFun::RegisterFunction(BuiltinFunctions &set) {
	AggregateFunctionSet fun("histogram");
	fun.AddFunction(GetHistogramFunction<>(LogicalType::BOOLEAN));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::UTINYINT));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::USMALLINT));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::UINTEGER));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::UBIGINT));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::TINYINT));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::SMALLINT));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::INTEGER));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::BIGINT));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::FLOAT));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::DOUBLE));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::VARCHAR));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::TIMESTAMP));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::TIMESTAMP_TZ));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::TIMESTAMP_S));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::TIMESTAMP_MS));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::TIMESTAMP_NS));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::TIME));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::TIME_TZ));
	fun.AddFunction(GetHistogramFunction<>(LogicalType::DATE));
	set.AddFunction(fun);
}

AggregateFunction HistogramFun::GetHistogramUnorderedMap(LogicalType &type) {
	const auto &const_type = type;
	return GetHistogramFunction<false>(const_type);
}

} // namespace duckdb





namespace duckdb {

struct ListSegment {
	uint16_t count;
	uint16_t capacity;
	ListSegment *next;
};
struct LinkedList {
	LinkedList() {};
	LinkedList(idx_t total_capacity_p, ListSegment *first_segment_p, ListSegment *last_segment_p)
	    : total_capacity(total_capacity_p), first_segment(first_segment_p), last_segment(last_segment_p) {
	}

	idx_t total_capacity = 0;
	ListSegment *first_segment = nullptr;
	ListSegment *last_segment = nullptr;
};

// forward declarations
struct WriteDataToSegment;
struct ReadDataFromSegment;
struct CopyDataFromSegment;
typedef ListSegment *(*create_segment_t)(WriteDataToSegment &write_data_to_segment, Allocator &allocator,
                                         vector<AllocatedData> &owning_vector, const uint16_t &capacity);
typedef void (*write_data_to_segment_t)(WriteDataToSegment &write_data_to_segment, Allocator &allocator,
                                        vector<AllocatedData> &owning_vector, ListSegment *segment, Vector &input,
                                        idx_t &entry_idx, idx_t &count);
typedef void (*read_data_from_segment_t)(ReadDataFromSegment &read_data_from_segment, const ListSegment *segment,
                                         Vector &result, idx_t &total_count);
typedef ListSegment *(*copy_data_from_segment_t)(CopyDataFromSegment &copy_data_from_segment, const ListSegment *source,
                                                 Allocator &allocator, vector<AllocatedData> &owning_vector);

struct WriteDataToSegment {
	create_segment_t create_segment;
	write_data_to_segment_t segment_function;
	vector<WriteDataToSegment> child_functions;
};
struct ReadDataFromSegment {
	read_data_from_segment_t segment_function;
	vector<ReadDataFromSegment> child_functions;
};
struct CopyDataFromSegment {
	copy_data_from_segment_t segment_function;
	vector<CopyDataFromSegment> child_functions;
};

// forward declarations
static void AppendRow(WriteDataToSegment &write_data_to_segment, Allocator &allocator,
                      vector<AllocatedData> &owning_vector, LinkedList *linked_list, Vector &input, idx_t &entry_idx,
                      idx_t &count);
static void BuildListVector(ReadDataFromSegment &read_data_from_segment, LinkedList *linked_list, Vector &result,
                            idx_t &initial_total_count);
static void CopyLinkedList(CopyDataFromSegment &copy_data_from_segment, const LinkedList *source_list,
                           LinkedList &target_list, Allocator &allocator, vector<AllocatedData> &owning_vector);

template <class T>
static data_ptr_t AllocatePrimitiveData(Allocator &allocator, vector<AllocatedData> &owning_vector,
                                        const uint16_t &capacity) {

	owning_vector.emplace_back(allocator.Allocate(sizeof(ListSegment) + capacity * (sizeof(bool) + sizeof(T))));
	return owning_vector.back().get();
}

static data_ptr_t AllocateListData(Allocator &allocator, vector<AllocatedData> &owning_vector,
                                   const uint16_t &capacity) {

	owning_vector.emplace_back(
	    allocator.Allocate(sizeof(ListSegment) + capacity * (sizeof(bool) + sizeof(uint64_t)) + sizeof(LinkedList)));
	return owning_vector.back().get();
}

static data_ptr_t AllocateStructData(Allocator &allocator, vector<AllocatedData> &owning_vector,
                                     const uint16_t &capacity, const idx_t &child_count) {

	owning_vector.emplace_back(
	    allocator.Allocate(sizeof(ListSegment) + capacity * sizeof(bool) + child_count * sizeof(ListSegment *)));
	return owning_vector.back().get();
}

template <class T>
static T *GetPrimitiveData(const ListSegment *segment) {
	return (T *)(((char *)segment) + sizeof(ListSegment) + segment->capacity * sizeof(bool));
}

static uint64_t *GetListLengthData(const ListSegment *segment) {
	return (uint64_t *)(((char *)segment) + sizeof(ListSegment) + segment->capacity * sizeof(bool));
}

static LinkedList *GetListChildData(const ListSegment *segment) {
	return (LinkedList *)(((char *)segment) + sizeof(ListSegment) +
	                      segment->capacity * (sizeof(bool) + sizeof(uint64_t)));
}

static ListSegment **GetStructData(const ListSegment *segment) {
	return (ListSegment **)(((char *)segment) + sizeof(ListSegment) + segment->capacity * sizeof(bool));
}

static bool *GetNullMask(const ListSegment *segment) {
	return (bool *)(((char *)segment) + sizeof(ListSegment));
}

static uint16_t GetCapacityForNewSegment(const LinkedList *linked_list) {

	// consecutive segments grow by the power of two
	uint16_t capacity = 4;
	if (linked_list->last_segment) {
		auto next_power_of_two = linked_list->last_segment->capacity * 2;
		capacity = next_power_of_two < 65536 ? next_power_of_two : linked_list->last_segment->capacity;
	}
	return capacity;
}

template <class T>
static ListSegment *CreatePrimitiveSegment(WriteDataToSegment &, Allocator &allocator,
                                           vector<AllocatedData> &owning_vector, const uint16_t &capacity) {

	// allocate data and set the header
	auto segment = (ListSegment *)AllocatePrimitiveData<T>(allocator, owning_vector, capacity);
	segment->capacity = capacity;
	segment->count = 0;
	segment->next = nullptr;
	return segment;
}

static ListSegment *CreateListSegment(WriteDataToSegment &, Allocator &allocator, vector<AllocatedData> &owning_vector,
                                      const uint16_t &capacity) {

	// allocate data and set the header
	auto segment = (ListSegment *)AllocateListData(allocator, owning_vector, capacity);
	segment->capacity = capacity;
	segment->count = 0;
	segment->next = nullptr;

	// create an empty linked list for the child vector
	auto linked_child_list = GetListChildData(segment);
	LinkedList linked_list(0, nullptr, nullptr);
	Store<LinkedList>(linked_list, (data_ptr_t)linked_child_list);

	return segment;
}

static ListSegment *CreateStructSegment(WriteDataToSegment &write_data_to_segment, Allocator &allocator,
                                        vector<AllocatedData> &owning_vector, const uint16_t &capacity) {

	// allocate data and set header
	auto segment = (ListSegment *)AllocateStructData(allocator, owning_vector, capacity,
	                                                 write_data_to_segment.child_functions.size());
	segment->capacity = capacity;
	segment->count = 0;
	segment->next = nullptr;

	// create a child ListSegment with exactly the same capacity for each child vector
	auto child_segments = GetStructData(segment);
	for (idx_t i = 0; i < write_data_to_segment.child_functions.size(); i++) {
		auto child_function = write_data_to_segment.child_functions[i];
		auto child_segment = child_function.create_segment(child_function, allocator, owning_vector, capacity);
		Store<ListSegment *>(child_segment, (data_ptr_t)(child_segments + i));
	}

	return segment;
}

static ListSegment *GetSegment(WriteDataToSegment &write_data_to_segment, Allocator &allocator,
                               vector<AllocatedData> &owning_vector, LinkedList *linked_list) {

	ListSegment *segment = nullptr;

	// determine segment
	if (!linked_list->last_segment) {
		// empty linked list, create the first (and last) segment
		auto capacity = GetCapacityForNewSegment(linked_list);
		segment = write_data_to_segment.create_segment(write_data_to_segment, allocator, owning_vector, capacity);
		linked_list->first_segment = segment;
		linked_list->last_segment = segment;

	} else if (linked_list->last_segment->capacity == linked_list->last_segment->count) {
		// the last segment of the linked list is full, create a new one and append it
		auto capacity = GetCapacityForNewSegment(linked_list);
		segment = write_data_to_segment.create_segment(write_data_to_segment, allocator, owning_vector, capacity);
		linked_list->last_segment->next = segment;
		linked_list->last_segment = segment;

	} else {
		// the last segment of the linked list is not full, append the data to it
		segment = linked_list->last_segment;
	}

	D_ASSERT(segment);
	return segment;
}

template <class T>
static void WriteDataToPrimitiveSegment(WriteDataToSegment &, Allocator &allocator,
                                        vector<AllocatedData> &owning_vector, ListSegment *segment, Vector &input,
                                        idx_t &entry_idx, idx_t &count) {

	// get the vector data and the source index of the entry that we want to write
	auto input_data = FlatVector::GetData(input);

	// write null validity
	auto null_mask = GetNullMask(segment);
	auto is_null = FlatVector::IsNull(input, entry_idx);
	null_mask[segment->count] = is_null;

	// write value
	if (!is_null) {
		auto data = GetPrimitiveData<T>(segment);
		Store<T>(((T *)input_data)[entry_idx], (data_ptr_t)(data + segment->count));
	}
}

static void WriteDataToVarcharSegment(WriteDataToSegment &write_data_to_segment, Allocator &allocator,
                                      vector<AllocatedData> &owning_vector, ListSegment *segment, Vector &input,
                                      idx_t &entry_idx, idx_t &count) {

	// get the vector data and the source index of the entry that we want to write
	auto input_data = FlatVector::GetData(input);

	// write null validity
	auto null_mask = GetNullMask(segment);
	auto is_null = FlatVector::IsNull(input, entry_idx);
	null_mask[segment->count] = is_null;

	// set the length of this string
	auto str_length_data = GetListLengthData(segment);
	uint64_t str_length = 0;

	// get the string
	string_t str_t;
	if (!is_null) {
		str_t = ((string_t *)input_data)[entry_idx];
		str_length = str_t.GetSize();
	}

	// we can reconstruct the offset from the length
	Store<uint64_t>(str_length, (data_ptr_t)(str_length_data + segment->count));

	if (is_null) {
		return;
	}

	// write the characters to the linked list of child segments
	auto child_segments = Load<LinkedList>((data_ptr_t)GetListChildData(segment));
	for (char &c : str_t.GetString()) {
		auto child_segment =
		    GetSegment(write_data_to_segment.child_functions.back(), allocator, owning_vector, &child_segments);
		auto data = GetPrimitiveData<char>(child_segment);
		data[child_segment->count] = c;
		child_segment->count++;
		child_segments.total_capacity++;
	}

	// store the updated linked list
	Store<LinkedList>(child_segments, (data_ptr_t)GetListChildData(segment));
}

static void WriteDataToListSegment(WriteDataToSegment &write_data_to_segment, Allocator &allocator,
                                   vector<AllocatedData> &owning_vector, ListSegment *segment, Vector &input,
                                   idx_t &entry_idx, idx_t &count) {

	// get the vector data and the source index of the entry that we want to write
	auto input_data = FlatVector::GetData(input);

	// write null validity
	auto null_mask = GetNullMask(segment);
	auto is_null = FlatVector::IsNull(input, entry_idx);
	null_mask[segment->count] = is_null;

	// set the length of this list
	auto list_length_data = GetListLengthData(segment);
	uint64_t list_length = 0;

	if (!is_null) {
		// get list entry information
		auto list_entries = (list_entry_t *)input_data;
		const auto &list_entry = list_entries[entry_idx];
		list_length = list_entry.length;

		// get the child vector and its data
		auto lists_size = ListVector::GetListSize(input);
		auto &child_vector = ListVector::GetEntry(input);

		// loop over the child vector entries and recurse on them
		auto child_segments = Load<LinkedList>((data_ptr_t)GetListChildData(segment));
		D_ASSERT(write_data_to_segment.child_functions.size() == 1);
		for (idx_t child_idx = 0; child_idx < list_entry.length; child_idx++) {
			auto source_idx_child = list_entry.offset + child_idx;
			AppendRow(write_data_to_segment.child_functions[0], allocator, owning_vector, &child_segments, child_vector,
			          source_idx_child, lists_size);
		}
		// store the updated linked list
		Store<LinkedList>(child_segments, (data_ptr_t)GetListChildData(segment));
	}

	Store<uint64_t>(list_length, (data_ptr_t)(list_length_data + segment->count));
}

static void WriteDataToStructSegment(WriteDataToSegment &write_data_to_segment, Allocator &allocator,
                                     vector<AllocatedData> &owning_vector, ListSegment *segment, Vector &input,
                                     idx_t &entry_idx, idx_t &count) {

	// write null validity
	auto null_mask = GetNullMask(segment);
	auto is_null = FlatVector::IsNull(input, entry_idx);
	null_mask[segment->count] = is_null;

	// write value
	auto &children = StructVector::GetEntries(input);
	D_ASSERT(children.size() == write_data_to_segment.child_functions.size());
	auto child_list = GetStructData(segment);

	// write the data of each of the children of the struct
	for (idx_t child_count = 0; child_count < children.size(); child_count++) {
		auto child_list_segment = Load<ListSegment *>((data_ptr_t)(child_list + child_count));
		auto &child_function = write_data_to_segment.child_functions[child_count];
		child_function.segment_function(child_function, allocator, owning_vector, child_list_segment,
		                                *children[child_count], entry_idx, count);
		child_list_segment->count++;
	}
}

static void AppendRow(WriteDataToSegment &write_data_to_segment, Allocator &allocator,
                      vector<AllocatedData> &owning_vector, LinkedList *linked_list, Vector &input, idx_t &entry_idx,
                      idx_t &count) {

	D_ASSERT(input.GetVectorType() == VectorType::FLAT_VECTOR);

	auto segment = GetSegment(write_data_to_segment, allocator, owning_vector, linked_list);
	write_data_to_segment.segment_function(write_data_to_segment, allocator, owning_vector, segment, input, entry_idx,
	                                       count);

	linked_list->total_capacity++;
	segment->count++;
}

template <class T>
static void ReadDataFromPrimitiveSegment(ReadDataFromSegment &, const ListSegment *segment, Vector &result,
                                         idx_t &total_count) {

	auto &aggr_vector_validity = FlatVector::Validity(result);

	// set NULLs
	auto null_mask = GetNullMask(segment);
	for (idx_t i = 0; i < segment->count; i++) {
		if (null_mask[i]) {
			aggr_vector_validity.SetInvalid(total_count + i);
		}
	}

	auto aggr_vector_data = FlatVector::GetData(result);

	// load values
	for (idx_t i = 0; i < segment->count; i++) {
		if (aggr_vector_validity.RowIsValid(total_count + i)) {
			auto data = GetPrimitiveData<T>(segment);
			((T *)aggr_vector_data)[total_count + i] = Load<T>((data_ptr_t)(data + i));
		}
	}
}

static void ReadDataFromVarcharSegment(ReadDataFromSegment &, const ListSegment *segment, Vector &result,
                                       idx_t &total_count) {

	auto &aggr_vector_validity = FlatVector::Validity(result);

	// set NULLs
	auto null_mask = GetNullMask(segment);
	for (idx_t i = 0; i < segment->count; i++) {
		if (null_mask[i]) {
			aggr_vector_validity.SetInvalid(total_count + i);
		}
	}

	// append all the child chars to one string
	string str = "";
	auto linked_child_list = Load<LinkedList>((data_ptr_t)GetListChildData(segment));
	while (linked_child_list.first_segment) {
		auto child_segment = linked_child_list.first_segment;
		auto data = GetPrimitiveData<char>(child_segment);
		str.append(data, child_segment->count);
		linked_child_list.first_segment = child_segment->next;
	}
	linked_child_list.last_segment = nullptr;

	// use length and (reconstructed) offset to get the correct substrings
	auto aggr_vector_data = FlatVector::GetData(result);
	auto str_length_data = GetListLengthData(segment);

	// get the substrings and write them to the result vector
	idx_t offset = 0;
	for (idx_t i = 0; i < segment->count; i++) {
		if (!null_mask[i]) {
			auto str_length = Load<uint64_t>((data_ptr_t)(str_length_data + i));
			auto substr = str.substr(offset, str_length);
			auto str_t = StringVector::AddStringOrBlob(result, substr);
			((string_t *)aggr_vector_data)[total_count + i] = str_t;
			offset += str_length;
		}
	}
}

static void ReadDataFromListSegment(ReadDataFromSegment &read_data_from_segment, const ListSegment *segment,
                                    Vector &result, idx_t &total_count) {

	auto &aggr_vector_validity = FlatVector::Validity(result);

	// set NULLs
	auto null_mask = GetNullMask(segment);
	for (idx_t i = 0; i < segment->count; i++) {
		if (null_mask[i]) {
			aggr_vector_validity.SetInvalid(total_count + i);
		}
	}

	auto list_vector_data = FlatVector::GetData<list_entry_t>(result);

	// get the starting offset
	idx_t offset = 0;
	if (total_count != 0) {
		offset = list_vector_data[total_count - 1].offset + list_vector_data[total_count - 1].length;
	}
	idx_t starting_offset = offset;

	// set length and offsets
	auto list_length_data = GetListLengthData(segment);
	for (idx_t i = 0; i < segment->count; i++) {
		auto list_length = Load<uint64_t>((data_ptr_t)(list_length_data + i));
		list_vector_data[total_count + i].length = list_length;
		list_vector_data[total_count + i].offset = offset;
		offset += list_length;
	}

	auto &child_vector = ListVector::GetEntry(result);
	auto linked_child_list = Load<LinkedList>((data_ptr_t)GetListChildData(segment));
	ListVector::Reserve(result, offset);

	// recurse into the linked list of child values
	D_ASSERT(read_data_from_segment.child_functions.size() == 1);
	BuildListVector(read_data_from_segment.child_functions[0], &linked_child_list, child_vector, starting_offset);
}

static void ReadDataFromStructSegment(ReadDataFromSegment &read_data_from_segment, const ListSegment *segment,
                                      Vector &result, idx_t &total_count) {

	auto &aggr_vector_validity = FlatVector::Validity(result);

	// set NULLs
	auto null_mask = GetNullMask(segment);
	for (idx_t i = 0; i < segment->count; i++) {
		if (null_mask[i]) {
			aggr_vector_validity.SetInvalid(total_count + i);
		}
	}

	auto &children = StructVector::GetEntries(result);

	// recurse into the child segments of each child of the struct
	D_ASSERT(children.size() == read_data_from_segment.child_functions.size());
	auto struct_children = GetStructData(segment);
	for (idx_t child_count = 0; child_count < children.size(); child_count++) {
		auto struct_children_segment = Load<ListSegment *>((data_ptr_t)(struct_children + child_count));
		auto &child_function = read_data_from_segment.child_functions[child_count];
		child_function.segment_function(child_function, struct_children_segment, *children[child_count], total_count);
	}
}

static void BuildListVector(ReadDataFromSegment &read_data_from_segment, LinkedList *linked_list, Vector &result,
                            idx_t &initial_total_count) {

	idx_t total_count = initial_total_count;
	while (linked_list->first_segment) {
		auto segment = linked_list->first_segment;
		read_data_from_segment.segment_function(read_data_from_segment, segment, result, total_count);

		total_count += segment->count;
		linked_list->first_segment = segment->next;
	}

	linked_list->last_segment = nullptr;
}

template <class T>
static ListSegment *CopyDataFromPrimitiveSegment(CopyDataFromSegment &, const ListSegment *source, Allocator &allocator,
                                                 vector<AllocatedData> &owning_vector) {

	auto target = (ListSegment *)AllocatePrimitiveData<T>(allocator, owning_vector, source->capacity);
	memcpy(target, source, sizeof(ListSegment) + source->capacity * (sizeof(bool) + sizeof(T)));
	target->next = nullptr;
	return target;
}

static ListSegment *CopyDataFromListSegment(CopyDataFromSegment &copy_data_from_segment, const ListSegment *source,
                                            Allocator &allocator, vector<AllocatedData> &owning_vector) {

	// create an empty linked list for the child vector of target
	auto source_linked_child_list = Load<LinkedList>((data_ptr_t)GetListChildData(source));

	// create the segment
	auto target = (ListSegment *)AllocateListData(allocator, owning_vector, source->capacity);
	memcpy(target, source,
	       sizeof(ListSegment) + source->capacity * (sizeof(bool) + sizeof(uint64_t)) + sizeof(LinkedList));
	target->next = nullptr;

	auto target_linked_list = GetListChildData(target);
	LinkedList linked_list(source_linked_child_list.total_capacity, nullptr, nullptr);
	Store<LinkedList>(linked_list, (data_ptr_t)target_linked_list);

	// recurse to copy the linked child list
	auto target_linked_child_list = Load<LinkedList>((data_ptr_t)GetListChildData(target));
	D_ASSERT(copy_data_from_segment.child_functions.size() == 1);
	CopyLinkedList(copy_data_from_segment.child_functions[0], &source_linked_child_list, target_linked_child_list,
	               allocator, owning_vector);

	// store the updated linked list
	Store<LinkedList>(target_linked_child_list, (data_ptr_t)GetListChildData(target));
	return target;
}

static ListSegment *CopyDataFromStructSegment(CopyDataFromSegment &copy_data_from_segment, const ListSegment *source,
                                              Allocator &allocator, vector<AllocatedData> &owning_vector) {

	auto source_child_count = copy_data_from_segment.child_functions.size();
	auto target = (ListSegment *)AllocateStructData(allocator, owning_vector, source->capacity, source_child_count);
	memcpy(target, source,
	       sizeof(ListSegment) + source->capacity * sizeof(bool) + source_child_count * sizeof(ListSegment *));
	target->next = nullptr;

	// recurse and copy the children
	auto source_child_segments = GetStructData(source);
	auto target_child_segments = GetStructData(target);

	for (idx_t i = 0; i < copy_data_from_segment.child_functions.size(); i++) {
		auto child_function = copy_data_from_segment.child_functions[i];
		auto source_child_segment = Load<ListSegment *>((data_ptr_t)(source_child_segments + i));
		auto target_child_segment =
		    child_function.segment_function(child_function, source_child_segment, allocator, owning_vector);
		Store<ListSegment *>(target_child_segment, (data_ptr_t)(target_child_segments + i));
	}
	return target;
}

static void CopyLinkedList(CopyDataFromSegment &copy_data_from_segment, const LinkedList *source_list,
                           LinkedList &target_list, Allocator &allocator, vector<AllocatedData> &owning_vector) {

	auto source_segment = source_list->first_segment;

	while (source_segment) {
		auto target_segment =
		    copy_data_from_segment.segment_function(copy_data_from_segment, source_segment, allocator, owning_vector);
		source_segment = source_segment->next;

		if (!target_list.first_segment) {
			target_list.first_segment = target_segment;
		}
		if (target_list.last_segment) {
			target_list.last_segment->next = target_segment;
		}
		target_list.last_segment = target_segment;
	}
}

static void InitializeValidities(Vector &vector, idx_t &capacity) {

	auto &validity_mask = FlatVector::Validity(vector);
	validity_mask.Initialize(capacity);

	auto internal_type = vector.GetType().InternalType();
	if (internal_type == PhysicalType::LIST) {
		auto &child_vector = ListVector::GetEntry(vector);
		InitializeValidities(child_vector, capacity);
	} else if (internal_type == PhysicalType::STRUCT) {
		auto &children = StructVector::GetEntries(vector);
		for (auto &child : children) {
			InitializeValidities(*child, capacity);
		}
	}
}

static void RecursiveFlatten(Vector &vector, idx_t &count) {

	if (vector.GetVectorType() != VectorType::FLAT_VECTOR) {
		vector.Flatten(count);
	}

	auto internal_type = vector.GetType().InternalType();
	if (internal_type == PhysicalType::LIST) {
		auto &child_vector = ListVector::GetEntry(vector);
		auto child_vector_count = ListVector::GetListSize(vector);
		RecursiveFlatten(child_vector, child_vector_count);
	} else if (internal_type == PhysicalType::STRUCT) {
		auto &children = StructVector::GetEntries(vector);
		for (auto &child : children) {
			RecursiveFlatten(*child, count);
		}
	}
}

struct ListBindData : public FunctionData {
	explicit ListBindData(const LogicalType &stype_p);
	~ListBindData() override;

	LogicalType stype;
	WriteDataToSegment write_data_to_segment;
	ReadDataFromSegment read_data_from_segment;
	CopyDataFromSegment copy_data_from_segment;

	unique_ptr<FunctionData> Copy() const override {
		return make_unique<ListBindData>(stype);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const ListBindData &)other_p;
		return stype == other.stype;
	}
};

static void GetSegmentDataFunctions(WriteDataToSegment &write_data_to_segment,
                                    ReadDataFromSegment &read_data_from_segment,
                                    CopyDataFromSegment &copy_data_from_segment, const LogicalType &type) {

	auto physical_type = type.InternalType();
	switch (physical_type) {
	case PhysicalType::BIT:
	case PhysicalType::BOOL: {
		write_data_to_segment.create_segment = CreatePrimitiveSegment<bool>;
		write_data_to_segment.segment_function = WriteDataToPrimitiveSegment<bool>;
		read_data_from_segment.segment_function = ReadDataFromPrimitiveSegment<bool>;
		copy_data_from_segment.segment_function = CopyDataFromPrimitiveSegment<bool>;
		break;
	}
	case PhysicalType::INT8: {
		write_data_to_segment.create_segment = CreatePrimitiveSegment<int8_t>;
		write_data_to_segment.segment_function = WriteDataToPrimitiveSegment<int8_t>;
		read_data_from_segment.segment_function = ReadDataFromPrimitiveSegment<int8_t>;
		copy_data_from_segment.segment_function = CopyDataFromPrimitiveSegment<int8_t>;
		break;
	}
	case PhysicalType::INT16: {
		write_data_to_segment.create_segment = CreatePrimitiveSegment<int16_t>;
		write_data_to_segment.segment_function = WriteDataToPrimitiveSegment<int16_t>;
		read_data_from_segment.segment_function = ReadDataFromPrimitiveSegment<int16_t>;
		copy_data_from_segment.segment_function = CopyDataFromPrimitiveSegment<int16_t>;
		break;
	}
	case PhysicalType::INT32: {
		write_data_to_segment.create_segment = CreatePrimitiveSegment<int32_t>;
		write_data_to_segment.segment_function = WriteDataToPrimitiveSegment<int32_t>;
		read_data_from_segment.segment_function = ReadDataFromPrimitiveSegment<int32_t>;
		copy_data_from_segment.segment_function = CopyDataFromPrimitiveSegment<int32_t>;
		break;
	}
	case PhysicalType::INT64: {
		write_data_to_segment.create_segment = CreatePrimitiveSegment<int64_t>;
		write_data_to_segment.segment_function = WriteDataToPrimitiveSegment<int64_t>;
		read_data_from_segment.segment_function = ReadDataFromPrimitiveSegment<int64_t>;
		copy_data_from_segment.segment_function = CopyDataFromPrimitiveSegment<int64_t>;
		break;
	}
	case PhysicalType::UINT8: {
		write_data_to_segment.create_segment = CreatePrimitiveSegment<uint8_t>;
		write_data_to_segment.segment_function = WriteDataToPrimitiveSegment<uint8_t>;
		read_data_from_segment.segment_function = ReadDataFromPrimitiveSegment<uint8_t>;
		copy_data_from_segment.segment_function = CopyDataFromPrimitiveSegment<uint8_t>;
		break;
	}
	case PhysicalType::UINT16: {
		write_data_to_segment.create_segment = CreatePrimitiveSegment<uint16_t>;
		write_data_to_segment.segment_function = WriteDataToPrimitiveSegment<uint16_t>;
		read_data_from_segment.segment_function = ReadDataFromPrimitiveSegment<uint16_t>;
		copy_data_from_segment.segment_function = CopyDataFromPrimitiveSegment<uint16_t>;
		break;
	}
	case PhysicalType::UINT32: {
		write_data_to_segment.create_segment = CreatePrimitiveSegment<uint32_t>;
		write_data_to_segment.segment_function = WriteDataToPrimitiveSegment<uint32_t>;
		read_data_from_segment.segment_function = ReadDataFromPrimitiveSegment<uint32_t>;
		copy_data_from_segment.segment_function = CopyDataFromPrimitiveSegment<uint32_t>;
		break;
	}
	case PhysicalType::UINT64: {
		write_data_to_segment.create_segment = CreatePrimitiveSegment<uint64_t>;
		write_data_to_segment.segment_function = WriteDataToPrimitiveSegment<uint64_t>;
		read_data_from_segment.segment_function = ReadDataFromPrimitiveSegment<uint64_t>;
		copy_data_from_segment.segment_function = CopyDataFromPrimitiveSegment<uint64_t>;
		break;
	}
	case PhysicalType::FLOAT: {
		write_data_to_segment.create_segment = CreatePrimitiveSegment<float>;
		write_data_to_segment.segment_function = WriteDataToPrimitiveSegment<float>;
		read_data_from_segment.segment_function = ReadDataFromPrimitiveSegment<float>;
		copy_data_from_segment.segment_function = CopyDataFromPrimitiveSegment<float>;
		break;
	}
	case PhysicalType::DOUBLE: {
		write_data_to_segment.create_segment = CreatePrimitiveSegment<double>;
		write_data_to_segment.segment_function = WriteDataToPrimitiveSegment<double>;
		read_data_from_segment.segment_function = ReadDataFromPrimitiveSegment<double>;
		copy_data_from_segment.segment_function = CopyDataFromPrimitiveSegment<double>;
		break;
	}
	case PhysicalType::INT128: {
		write_data_to_segment.create_segment = CreatePrimitiveSegment<hugeint_t>;
		write_data_to_segment.segment_function = WriteDataToPrimitiveSegment<hugeint_t>;
		read_data_from_segment.segment_function = ReadDataFromPrimitiveSegment<hugeint_t>;
		copy_data_from_segment.segment_function = CopyDataFromPrimitiveSegment<hugeint_t>;
		break;
	}
	case PhysicalType::INTERVAL: {
		write_data_to_segment.create_segment = CreatePrimitiveSegment<interval_t>;
		write_data_to_segment.segment_function = WriteDataToPrimitiveSegment<interval_t>;
		read_data_from_segment.segment_function = ReadDataFromPrimitiveSegment<interval_t>;
		copy_data_from_segment.segment_function = CopyDataFromPrimitiveSegment<interval_t>;
		break;
	}
	case PhysicalType::VARCHAR: {
		write_data_to_segment.create_segment = CreateListSegment;
		write_data_to_segment.segment_function = WriteDataToVarcharSegment;
		read_data_from_segment.segment_function = ReadDataFromVarcharSegment;
		copy_data_from_segment.segment_function = CopyDataFromListSegment;

		write_data_to_segment.child_functions.emplace_back(WriteDataToSegment());
		write_data_to_segment.child_functions.back().create_segment = CreatePrimitiveSegment<char>;
		copy_data_from_segment.child_functions.emplace_back(CopyDataFromSegment());
		copy_data_from_segment.child_functions.back().segment_function = CopyDataFromPrimitiveSegment<char>;
		break;
	}
	case PhysicalType::LIST: {
		write_data_to_segment.create_segment = CreateListSegment;
		write_data_to_segment.segment_function = WriteDataToListSegment;
		read_data_from_segment.segment_function = ReadDataFromListSegment;
		copy_data_from_segment.segment_function = CopyDataFromListSegment;

		// recurse
		write_data_to_segment.child_functions.emplace_back(WriteDataToSegment());
		read_data_from_segment.child_functions.emplace_back(ReadDataFromSegment());
		copy_data_from_segment.child_functions.emplace_back(CopyDataFromSegment());
		GetSegmentDataFunctions(write_data_to_segment.child_functions.back(),
		                        read_data_from_segment.child_functions.back(),
		                        copy_data_from_segment.child_functions.back(), ListType::GetChildType(type));
		break;
	}
	case PhysicalType::STRUCT: {
		write_data_to_segment.create_segment = CreateStructSegment;
		write_data_to_segment.segment_function = WriteDataToStructSegment;
		read_data_from_segment.segment_function = ReadDataFromStructSegment;
		copy_data_from_segment.segment_function = CopyDataFromStructSegment;

		// recurse
		auto child_types = StructType::GetChildTypes(type);
		for (idx_t i = 0; i < child_types.size(); i++) {
			write_data_to_segment.child_functions.emplace_back(WriteDataToSegment());
			read_data_from_segment.child_functions.emplace_back(ReadDataFromSegment());
			copy_data_from_segment.child_functions.emplace_back(CopyDataFromSegment());
			GetSegmentDataFunctions(write_data_to_segment.child_functions.back(),
			                        read_data_from_segment.child_functions.back(),
			                        copy_data_from_segment.child_functions.back(), child_types[i].second);
		}
		break;
	}
	default:
		throw InternalException("LIST aggregate not yet implemented for " + type.ToString());
	}
}

ListBindData::ListBindData(const LogicalType &stype_p) : stype(stype_p) {

	// always unnest once because the result vector is of type LIST
	auto type = ListType::GetChildType(stype_p);
	GetSegmentDataFunctions(write_data_to_segment, read_data_from_segment, copy_data_from_segment, type);
}

ListBindData::~ListBindData() {
}

struct ListAggState {
	LinkedList *linked_list;
	LogicalType *type;
	vector<AllocatedData> *owning_vector;
};

struct ListFunction {
	template <class STATE>
	static void Initialize(STATE *state) {
		state->linked_list = nullptr;
		state->type = nullptr;
		state->owning_vector = nullptr;
	}

	template <class STATE>
	static void Destroy(STATE *state) {
		D_ASSERT(state);
		if (state->linked_list) {
			delete state->linked_list;
			state->linked_list = nullptr;
		}
		if (state->type) {
			delete state->type;
			state->type = nullptr;
		}
		if (state->owning_vector) {
			state->owning_vector->clear();
			delete state->owning_vector;
			state->owning_vector = nullptr;
		}
	}
	static bool IgnoreNull() {
		return false;
	}
};

static void ListUpdateFunction(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                               Vector &state_vector, idx_t count) {
	D_ASSERT(input_count == 1);

	auto &input = inputs[0];
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);

	auto states = (ListAggState **)sdata.data;
	RecursiveFlatten(input, count);

	auto &list_bind_data = (ListBindData &)*aggr_input_data.bind_data;

	for (idx_t i = 0; i < count; i++) {
		auto state = states[sdata.sel->get_index(i)];
		if (!state->linked_list) {
			state->linked_list = new LinkedList(0, nullptr, nullptr);
			state->type = new LogicalType(input.GetType());
			state->owning_vector = new vector<AllocatedData>;
		}
		D_ASSERT(state->type);
		AppendRow(list_bind_data.write_data_to_segment, aggr_input_data.allocator, *state->owning_vector,
		          state->linked_list, input, i, count);
	}
}

static void ListCombineFunction(Vector &state, Vector &combined, AggregateInputData &aggr_input_data, idx_t count) {
	UnifiedVectorFormat sdata;
	state.ToUnifiedFormat(count, sdata);
	auto states_ptr = (ListAggState **)sdata.data;

	auto &list_bind_data = (ListBindData &)*aggr_input_data.bind_data;

	auto combined_ptr = FlatVector::GetData<ListAggState *>(combined);
	for (idx_t i = 0; i < count; i++) {
		auto state = states_ptr[sdata.sel->get_index(i)];
		if (!state->linked_list) {
			// NULL, no need to append.
			continue;
		}
		D_ASSERT(state->type);
		D_ASSERT(state->owning_vector);

		if (!combined_ptr[i]->linked_list) {
			combined_ptr[i]->linked_list = new LinkedList(0, nullptr, nullptr);
			combined_ptr[i]->owning_vector = new vector<AllocatedData>;
			combined_ptr[i]->type = new LogicalType(*state->type);
		}
		auto owning_vector = combined_ptr[i]->owning_vector;

		// copy the linked list of the state
		auto copied_linked_list = LinkedList(state->linked_list->total_capacity, nullptr, nullptr);
		CopyLinkedList(list_bind_data.copy_data_from_segment, state->linked_list, copied_linked_list,
		               aggr_input_data.allocator, *owning_vector);

		// append the copied linked list to the combined state
		if (combined_ptr[i]->linked_list->last_segment) {
			combined_ptr[i]->linked_list->last_segment->next = copied_linked_list.first_segment;
		} else {
			combined_ptr[i]->linked_list->first_segment = copied_linked_list.first_segment;
		}
		combined_ptr[i]->linked_list->last_segment = copied_linked_list.last_segment;
		combined_ptr[i]->linked_list->total_capacity += copied_linked_list.total_capacity;
	}
}

static void ListFinalize(Vector &state_vector, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                         idx_t offset) {
	UnifiedVectorFormat sdata;
	state_vector.ToUnifiedFormat(count, sdata);
	auto states = (ListAggState **)sdata.data;

	D_ASSERT(result.GetType().id() == LogicalTypeId::LIST);

	auto &mask = FlatVector::Validity(result);
	auto result_data = FlatVector::GetData<list_entry_t>(result);
	size_t total_len = ListVector::GetListSize(result);

	auto &list_bind_data = (ListBindData &)*aggr_input_data.bind_data;

	for (idx_t i = 0; i < count; i++) {

		auto state = states[sdata.sel->get_index(i)];
		const auto rid = i + offset;
		if (!state->linked_list) {
			mask.SetInvalid(rid);
			continue;
		}

		// set the length and offset of this list in the result vector
		auto total_capacity = state->linked_list->total_capacity;
		result_data[rid].length = total_capacity;
		result_data[rid].offset = total_len;
		total_len += total_capacity;

		D_ASSERT(state->type);

		Vector aggr_vector(*state->type, total_capacity);
		// FIXME: this is a workaround because the constructor of a vector does not set the size
		// of the validity mask, and by default it is set to STANDARD_VECTOR_SIZE
		// ListVector::Reserve only increases the validity mask, if (to_reserve > capacity),
		// which will not be the case if the value passed to the constructor of aggr_vector
		// is greater than to_reserve
		InitializeValidities(aggr_vector, total_capacity);

		idx_t total_count = 0;
		BuildListVector(list_bind_data.read_data_from_segment, state->linked_list, aggr_vector, total_count);
		ListVector::Append(result, aggr_vector, total_capacity);

		// now destroy the state (for parallel destruction)
		ListFunction::Destroy<ListAggState>(state);
	}
}

unique_ptr<FunctionData> ListBindFunction(ClientContext &context, AggregateFunction &function,
                                          vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(arguments.size() == 1);
	D_ASSERT(function.arguments.size() == 1);

	if (arguments[0]->return_type.id() == LogicalTypeId::UNKNOWN) {
		function.arguments[0] = LogicalTypeId::UNKNOWN;
		function.return_type = LogicalType::SQLNULL;
		return nullptr;
	}

	function.return_type = LogicalType::LIST(arguments[0]->return_type);
	return make_unique<ListBindData>(function.return_type);
}

void ListFun::RegisterFunction(BuiltinFunctions &set) {
	auto agg =
	    AggregateFunction("list", {LogicalType::ANY}, LogicalTypeId::LIST, AggregateFunction::StateSize<ListAggState>,
	                      AggregateFunction::StateInitialize<ListAggState, ListFunction>, ListUpdateFunction,
	                      ListCombineFunction, ListFinalize, nullptr, ListBindFunction,
	                      AggregateFunction::StateDestroy<ListAggState, ListFunction>, nullptr, nullptr);
	set.AddFunction(agg);
	agg.name = "array_agg";
	set.AddFunction(agg);
}

} // namespace duckdb


namespace duckdb {

void BuiltinFunctions::RegisterNestedAggregates() {
	Register<ListFun>();
	Register<HistogramFun>();
}

} // namespace duckdb






namespace duckdb {
struct RegrState {
	double sum;
	size_t count;
};

struct RegrAvgFunction {
	template <class STATE>
	static void Initialize(STATE *state) {
		state->sum = 0;
		state->count = 0;
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &) {
		target->sum += source.sum;
		target->count += source.count;
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &, STATE *state, T *target, ValidityMask &mask, idx_t idx) {
		if (state->count == 0) {
			mask.SetInvalid(idx);
		} else {
			target[idx] = state->sum / (double)state->count;
		}
	}
	static bool IgnoreNull() {
		return true;
	}
};
struct RegrAvgXFunction : RegrAvgFunction {
	template <class A_TYPE, class B_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &, A_TYPE *x_data, B_TYPE *y_data, ValidityMask &amask,
	                      ValidityMask &bmask, idx_t xidx, idx_t yidx) {
		state->sum += y_data[yidx];
		state->count++;
	}
};

struct RegrAvgYFunction : RegrAvgFunction {
	template <class A_TYPE, class B_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &, A_TYPE *x_data, B_TYPE *y_data, ValidityMask &amask,
	                      ValidityMask &bmask, idx_t xidx, idx_t yidx) {
		state->sum += x_data[xidx];
		state->count++;
	}
};

void RegrAvgxFun::RegisterFunction(BuiltinFunctions &set) {
	AggregateFunctionSet corr("regr_avgx");
	corr.AddFunction(AggregateFunction::BinaryAggregate<RegrState, double, double, double, RegrAvgXFunction>(
	    LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE));
	set.AddFunction(corr);
}

void RegrAvgyFun::RegisterFunction(BuiltinFunctions &set) {
	AggregateFunctionSet corr("regr_avgy");
	corr.AddFunction(AggregateFunction::BinaryAggregate<RegrState, double, double, double, RegrAvgYFunction>(
	    LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE));
	set.AddFunction(corr);
}

} // namespace duckdb







namespace duckdb {

void RegrCountFun::RegisterFunction(BuiltinFunctions &set) {
	auto regr_count = AggregateFunction::BinaryAggregate<size_t, double, double, uint32_t, RegrCountFunction>(
	    LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::UINTEGER);
	regr_count.name = "regr_count";
	regr_count.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	set.AddFunction(regr_count);
}

} // namespace duckdb
//! AVG(y)-REGR_SLOPE(y,x)*AVG(x)





namespace duckdb {

struct RegrInterceptState {
	size_t count;
	double sum_x;
	double sum_y;
	RegrSlopeState slope;
};

struct RegrInterceptOperation {
	template <class STATE>
	static void Initialize(STATE *state) {
		state->count = 0;
		state->sum_x = 0;
		state->sum_y = 0;
		RegrSlopeOperation::Initialize<RegrSlopeState>(&state->slope);
	}

	template <class A_TYPE, class B_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &aggr_input_data, A_TYPE *x_data, B_TYPE *y_data,
	                      ValidityMask &amask, ValidityMask &bmask, idx_t xidx, idx_t yidx) {
		state->count++;
		state->sum_x += y_data[yidx];
		state->sum_y += x_data[xidx];
		RegrSlopeOperation::Operation<A_TYPE, B_TYPE, RegrSlopeState, OP>(&state->slope, aggr_input_data, x_data,
		                                                                  y_data, amask, bmask, xidx, yidx);
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &aggr_input_data) {
		target->count += source.count;
		target->sum_x += source.sum_x;
		target->sum_y += source.sum_y;
		RegrSlopeOperation::Combine<RegrSlopeState, OP>(source.slope, &target->slope, aggr_input_data);
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &aggr_input_data, STATE *state, T *target,
	                     ValidityMask &mask, idx_t idx) {
		if (state->count == 0) {
			mask.SetInvalid(idx);
			return;
		}
		RegrSlopeOperation::Finalize<T, RegrSlopeState>(result, aggr_input_data, &state->slope, target, mask, idx);
		auto x_avg = state->sum_x / state->count;
		auto y_avg = state->sum_y / state->count;
		target[idx] = y_avg - target[idx] * x_avg;
	}

	static bool IgnoreNull() {
		return true;
	}
};

void RegrInterceptFun::RegisterFunction(BuiltinFunctions &set) {
	AggregateFunctionSet fun("regr_intercept");
	fun.AddFunction(
	    AggregateFunction::BinaryAggregate<RegrInterceptState, double, double, double, RegrInterceptOperation>(
	        LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE));
	set.AddFunction(fun);
}

} // namespace duckdb
// Returns the coefficient of determination for non-null pairs in a group.
// It is computed for non-null pairs using the following formula:
// null                 if var_pop(x) = 0, else
// 1                    if var_pop(y) = 0 and var_pop(x) <> 0, else
// power(corr(y,x), 2)





namespace duckdb {
struct RegrR2State {
	CorrState corr;
	StddevState var_pop_x;
	StddevState var_pop_y;
};

struct RegrR2Operation {
	template <class STATE>
	static void Initialize(STATE *state) {
		CorrOperation::Initialize<CorrState>(&state->corr);
		STDDevBaseOperation::Initialize<StddevState>(&state->var_pop_x);
		STDDevBaseOperation::Initialize<StddevState>(&state->var_pop_y);
	}

	template <class A_TYPE, class B_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &aggr_input_data, A_TYPE *x_data, B_TYPE *y_data,
	                      ValidityMask &amask, ValidityMask &bmask, idx_t xidx, idx_t yidx) {
		CorrOperation::Operation<A_TYPE, B_TYPE, CorrState, OP>(&state->corr, aggr_input_data, y_data, x_data, bmask,
		                                                        amask, yidx, xidx);
		STDDevBaseOperation::Operation<A_TYPE, StddevState, OP>(&state->var_pop_x, aggr_input_data, y_data, bmask,
		                                                        yidx);
		STDDevBaseOperation::Operation<A_TYPE, StddevState, OP>(&state->var_pop_y, aggr_input_data, x_data, amask,
		                                                        xidx);
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &aggr_input_data) {
		CorrOperation::Combine<CorrState, OP>(source.corr, &target->corr, aggr_input_data);
		STDDevBaseOperation::Combine<StddevState, OP>(source.var_pop_x, &target->var_pop_x, aggr_input_data);
		STDDevBaseOperation::Combine<StddevState, OP>(source.var_pop_y, &target->var_pop_y, aggr_input_data);
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &aggr_input_data, STATE *state, T *target,
	                     ValidityMask &mask, idx_t idx) {
		auto var_pop_x = state->var_pop_x.count > 1 ? (state->var_pop_x.dsquared / state->var_pop_x.count) : 0;
		if (!Value::DoubleIsFinite(var_pop_x)) {
			throw OutOfRangeException("VARPOP(X) is out of range!");
		}
		if (var_pop_x == 0) {
			mask.SetInvalid(idx);
			return;
		}
		auto var_pop_y = state->var_pop_y.count > 1 ? (state->var_pop_y.dsquared / state->var_pop_y.count) : 0;
		if (!Value::DoubleIsFinite(var_pop_y)) {
			throw OutOfRangeException("VARPOP(Y) is out of range!");
		}
		if (var_pop_y == 0) {
			target[idx] = 1;
			return;
		}
		CorrOperation::Finalize<T, CorrState>(result, aggr_input_data, &state->corr, target, mask, idx);
		target[idx] = pow(target[idx], 2);
	}

	static bool IgnoreNull() {
		return true;
	}
};

void RegrR2Fun::RegisterFunction(BuiltinFunctions &set) {
	AggregateFunctionSet fun("regr_r2");
	fun.AddFunction(AggregateFunction::BinaryAggregate<RegrR2State, double, double, double, RegrR2Operation>(
	    LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE));
	set.AddFunction(fun);
}
} // namespace duckdb
// REGR_SLOPE(y, x)
// Returns the slope of the linear regression line for non-null pairs in a group.
// It is computed for non-null pairs using the following formula:
// COVAR_POP(x,y) / VAR_POP(x)

//! Input : Any numeric type
//! Output : Double





namespace duckdb {

void RegrSlopeFun::RegisterFunction(BuiltinFunctions &set) {
	AggregateFunctionSet fun("regr_slope");
	fun.AddFunction(AggregateFunction::BinaryAggregate<RegrSlopeState, double, double, double, RegrSlopeOperation>(
	    LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE));
	set.AddFunction(fun);
}

} // namespace duckdb
// regr_sxx
// Returns REGR_COUNT(y, x) * VAR_POP(x) for non-null pairs.
// regrsyy
// Returns REGR_COUNT(y, x) * VAR_POP(y) for non-null pairs.





namespace duckdb {

struct RegrSState {
	size_t count;
	StddevState var_pop;
};

struct RegrBaseOperation {
	template <class STATE>
	static void Initialize(STATE *state) {
		RegrCountFunction::Initialize<size_t>(&state->count);
		STDDevBaseOperation::Initialize<StddevState>(&state->var_pop);
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &aggr_input_data) {
		RegrCountFunction::Combine<size_t, OP>(source.count, &target->count, aggr_input_data);
		STDDevBaseOperation::Combine<StddevState, OP>(source.var_pop, &target->var_pop, aggr_input_data);
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &aggr_input_data, STATE *state, T *target,
	                     ValidityMask &mask, idx_t idx) {
		if (state->var_pop.count == 0) {
			mask.SetInvalid(idx);
			return;
		}
		auto var_pop = state->var_pop.count > 1 ? (state->var_pop.dsquared / state->var_pop.count) : 0;
		if (!Value::DoubleIsFinite(var_pop)) {
			throw OutOfRangeException("VARPOP is out of range!");
		}
		RegrCountFunction::Finalize<T, size_t>(result, aggr_input_data, &state->count, target, mask, idx);
		target[idx] *= var_pop;
	}

	static bool IgnoreNull() {
		return true;
	}
};

struct RegrSXXOperation : RegrBaseOperation {
	template <class A_TYPE, class B_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &aggr_input_data, A_TYPE *x_data, B_TYPE *y_data,
	                      ValidityMask &amask, ValidityMask &bmask, idx_t xidx, idx_t yidx) {
		RegrCountFunction::Operation<A_TYPE, B_TYPE, size_t, OP>(&state->count, aggr_input_data, y_data, x_data, bmask,
		                                                         amask, yidx, xidx);
		STDDevBaseOperation::Operation<A_TYPE, StddevState, OP>(&state->var_pop, aggr_input_data, y_data, bmask, yidx);
	}
};

struct RegrSYYOperation : RegrBaseOperation {
	template <class A_TYPE, class B_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &aggr_input_data, A_TYPE *x_data, B_TYPE *y_data,
	                      ValidityMask &amask, ValidityMask &bmask, idx_t xidx, idx_t yidx) {
		RegrCountFunction::Operation<A_TYPE, B_TYPE, size_t, OP>(&state->count, aggr_input_data, y_data, x_data, bmask,
		                                                         amask, yidx, xidx);
		STDDevBaseOperation::Operation<A_TYPE, StddevState, OP>(&state->var_pop, aggr_input_data, x_data, bmask, xidx);
	}
};

void RegrSXXFun::RegisterFunction(BuiltinFunctions &set) {
	AggregateFunctionSet fun("regr_sxx");
	fun.AddFunction(AggregateFunction::BinaryAggregate<RegrSState, double, double, double, RegrSXXOperation>(
	    LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE));
	set.AddFunction(fun);
}

void RegrSYYFun::RegisterFunction(BuiltinFunctions &set) {
	AggregateFunctionSet fun("regr_syy");
	fun.AddFunction(AggregateFunction::BinaryAggregate<RegrSState, double, double, double, RegrSYYOperation>(
	    LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE));
	set.AddFunction(fun);
}

} // namespace duckdb
// Returns REGR_COUNT(expr1, expr2) * COVAR_POP(expr1, expr2) for non-null pairs.






namespace duckdb {

struct RegrSXyState {
	size_t count;
	CovarState cov_pop;
};

struct RegrSXYOperation {
	template <class STATE>
	static void Initialize(STATE *state) {
		RegrCountFunction::Initialize<size_t>(&state->count);
		CovarOperation::Initialize<CovarState>(&state->cov_pop);
	}

	template <class A_TYPE, class B_TYPE, class STATE, class OP>
	static void Operation(STATE *state, AggregateInputData &aggr_input_data, A_TYPE *x_data, B_TYPE *y_data,
	                      ValidityMask &amask, ValidityMask &bmask, idx_t xidx, idx_t yidx) {
		RegrCountFunction::Operation<A_TYPE, B_TYPE, size_t, OP>(&state->count, aggr_input_data, y_data, x_data, bmask,
		                                                         amask, yidx, xidx);
		CovarOperation::Operation<A_TYPE, B_TYPE, CovarState, OP>(&state->cov_pop, aggr_input_data, x_data, y_data,
		                                                          amask, bmask, xidx, yidx);
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &aggr_input_data) {
		CovarOperation::Combine<CovarState, OP>(source.cov_pop, &target->cov_pop, aggr_input_data);
		RegrCountFunction::Combine<size_t, OP>(source.count, &target->count, aggr_input_data);
	}

	template <class T, class STATE>
	static void Finalize(Vector &result, AggregateInputData &aggr_input_data, STATE *state, T *target,
	                     ValidityMask &mask, idx_t idx) {
		CovarPopOperation::Finalize<T, CovarState>(result, aggr_input_data, &state->cov_pop, target, mask, idx);
		auto cov_pop = target[idx];
		RegrCountFunction::Finalize<T, size_t>(result, aggr_input_data, &state->count, target, mask, idx);
		target[idx] *= cov_pop;
	}

	static bool IgnoreNull() {
		return true;
	}
};

void RegrSXYFun::RegisterFunction(BuiltinFunctions &set) {
	AggregateFunctionSet fun("regr_sxy");
	fun.AddFunction(AggregateFunction::BinaryAggregate<RegrSXyState, double, double, double, RegrSXYOperation>(
	    LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE));
	set.AddFunction(fun);
}

} // namespace duckdb






namespace duckdb {

void BuiltinFunctions::RegisterRegressiveAggregates() {
	Register<RegrAvgxFun>();
	Register<RegrAvgyFun>();
	Register<RegrCountFun>();
	Register<RegrSlopeFun>();
	Register<RegrR2Fun>();
	Register<RegrSYYFun>();
	Register<RegrSXXFun>();
	Register<RegrSXYFun>();
	Register<RegrInterceptFun>();
}

} // namespace duckdb






namespace duckdb {

struct SortedAggregateBindData : public FunctionData {
	SortedAggregateBindData(ClientContext &context, const AggregateFunction &function_p,
	                        vector<unique_ptr<Expression>> &children, unique_ptr<FunctionData> bind_info_p,
	                        const BoundOrderModifier &order_bys)
	    : buffer_manager(BufferManager::GetBufferManager(context)), function(function_p),
	      bind_info(std::move(bind_info_p)) {
		arg_types.reserve(children.size());
		for (const auto &child : children) {
			arg_types.emplace_back(child->return_type);
		}
		sort_types.reserve(order_bys.orders.size());
		for (auto &order : order_bys.orders) {
			orders.emplace_back(order.Copy());
			sort_types.emplace_back(order.expression->return_type);
		}
	}

	SortedAggregateBindData(const SortedAggregateBindData &other)
	    : buffer_manager(other.buffer_manager), function(other.function), arg_types(other.arg_types),
	      sort_types(other.sort_types) {
		if (other.bind_info) {
			bind_info = other.bind_info->Copy();
		}
		for (auto &order : other.orders) {
			orders.emplace_back(order.Copy());
		}
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_unique<SortedAggregateBindData>(*this);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const SortedAggregateBindData &)other_p;
		if (bind_info && other.bind_info) {
			if (!bind_info->Equals(*other.bind_info)) {
				return false;
			}
		} else if (bind_info || other.bind_info) {
			return false;
		}
		if (function != other.function) {
			return false;
		}
		if (orders.size() != other.orders.size()) {
			return false;
		}
		for (size_t i = 0; i < orders.size(); ++i) {
			if (!orders[i].Equals(other.orders[i])) {
				return false;
			}
		}
		return true;
	}

	BufferManager &buffer_manager;
	AggregateFunction function;
	vector<LogicalType> arg_types;
	unique_ptr<FunctionData> bind_info;

	vector<BoundOrderByNode> orders;
	vector<LogicalType> sort_types;
};

struct SortedAggregateState {
	static const idx_t BUFFER_CAPACITY = STANDARD_VECTOR_SIZE;

	SortedAggregateState() : nsel(0) {
	}

	static inline void InitializeBuffer(DataChunk &chunk, const vector<LogicalType> &types) {
		if (!chunk.ColumnCount() && !types.empty()) {
			chunk.Initialize(Allocator::DefaultAllocator(), types);
		}
	}

	void Flush(SortedAggregateBindData &order_bind) {
		if (ordering) {
			return;
		}

		ordering = make_unique<ColumnDataCollection>(order_bind.buffer_manager, order_bind.sort_types);
		InitializeBuffer(sort_buffer, order_bind.sort_types);
		ordering->Append(sort_buffer);

		arguments = make_unique<ColumnDataCollection>(order_bind.buffer_manager, order_bind.arg_types);
		InitializeBuffer(arg_buffer, order_bind.arg_types);
		arguments->Append(arg_buffer);
	}

	void Update(SortedAggregateBindData &order_bind, DataChunk &sort_chunk, DataChunk &arg_chunk) {
		// Lazy instantiation of the buffer chunks
		InitializeBuffer(sort_buffer, order_bind.sort_types);
		InitializeBuffer(arg_buffer, order_bind.arg_types);

		if (sort_chunk.size() + sort_buffer.size() > BUFFER_CAPACITY) {
			Flush(order_bind);
		}
		if (ordering) {
			ordering->Append(sort_chunk);
			arguments->Append(arg_chunk);
		} else {
			sort_buffer.Append(sort_chunk, true);
			arg_buffer.Append(arg_chunk, true);
		}
	}

	void UpdateSlice(SortedAggregateBindData &order_bind, DataChunk &sort_inputs, DataChunk &arg_inputs) {
		// Lazy instantiation of the buffer chunks
		InitializeBuffer(sort_buffer, order_bind.sort_types);
		InitializeBuffer(arg_buffer, order_bind.arg_types);

		if (nsel + sort_buffer.size() > BUFFER_CAPACITY) {
			Flush(order_bind);
		}
		if (ordering) {
			sort_buffer.Reset();
			sort_buffer.Slice(sort_inputs, sel, nsel);
			ordering->Append(sort_buffer);

			arg_buffer.Reset();
			arg_buffer.Slice(arg_inputs, sel, nsel);
			arguments->Append(arg_buffer);
		} else {
			sort_buffer.Append(sort_inputs, true, &sel, nsel);
			arg_buffer.Append(arg_inputs, true, &sel, nsel);
		}

		nsel = 0;
	}

	void Combine(SortedAggregateBindData &order_bind, SortedAggregateState &other) {
		if (other.ordering) {
			// Force CDC if the other hash it
			Flush(order_bind);
			ordering->Combine(*other.ordering);
			arguments->Combine(*other.arguments);
		} else if (other.sort_buffer.size()) {
			Update(order_bind, other.sort_buffer, other.arg_buffer);
		}
	}

	void Finalize(LocalSortState &local_sort) {
		if (ordering) {
			ColumnDataScanState sort_state;
			ordering->InitializeScan(sort_state);
			ColumnDataScanState arg_state;
			arguments->InitializeScan(arg_state);
			for (sort_buffer.Reset(); ordering->Scan(sort_state, sort_buffer); sort_buffer.Reset()) {
				arg_buffer.Reset();
				arguments->Scan(arg_state, arg_buffer);
				local_sort.SinkChunk(sort_buffer, arg_buffer);
			}
			ordering->Reset();
			arguments->Reset();
		} else {
			local_sort.SinkChunk(sort_buffer, arg_buffer);
		}
	}

	unique_ptr<ColumnDataCollection> arguments;
	unique_ptr<ColumnDataCollection> ordering;

	DataChunk sort_buffer;
	DataChunk arg_buffer;

	// Selection for scattering
	SelectionVector sel;
	idx_t nsel;
};

struct SortedAggregateFunction {
	template <typename STATE>
	static void Initialize(STATE *state) {
		new (state) STATE();
	}

	template <typename STATE>
	static void Destroy(STATE *state) {
		state->~STATE();
	}

	static void ProjectInputs(Vector inputs[], SortedAggregateBindData *order_bind, idx_t input_count, idx_t count,
	                          DataChunk &arg_chunk, DataChunk &sort_chunk) {
		idx_t col = 0;

		arg_chunk.InitializeEmpty(order_bind->arg_types);
		for (auto &dst : arg_chunk.data) {
			dst.Reference(inputs[col++]);
		}
		arg_chunk.SetCardinality(count);

		sort_chunk.InitializeEmpty(order_bind->sort_types);
		for (auto &dst : sort_chunk.data) {
			dst.Reference(inputs[col++]);
		}
		sort_chunk.SetCardinality(count);
	}

	static void SimpleUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, data_ptr_t state,
	                         idx_t count) {
		const auto order_bind = (SortedAggregateBindData *)aggr_input_data.bind_data;
		DataChunk arg_chunk;
		DataChunk sort_chunk;
		ProjectInputs(inputs, order_bind, input_count, count, arg_chunk, sort_chunk);

		const auto order_state = (SortedAggregateState *)state;
		order_state->Update(*order_bind, sort_chunk, arg_chunk);
	}

	static void ScatterUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &states,
	                          idx_t count) {
		if (!count) {
			return;
		}

		// Append the arguments to the two sub-collections
		const auto order_bind = (SortedAggregateBindData *)aggr_input_data.bind_data;
		DataChunk arg_inputs;
		DataChunk sort_inputs;
		ProjectInputs(inputs, order_bind, input_count, count, arg_inputs, sort_inputs);

		// We have to scatter the chunks one at a time
		// so build a selection vector for each one.
		UnifiedVectorFormat svdata;
		states.ToUnifiedFormat(count, svdata);

		// Build the selection vector for each state.
		auto sdata = (SortedAggregateState **)svdata.data;
		for (idx_t i = 0; i < count; ++i) {
			auto sidx = svdata.sel->get_index(i);
			auto order_state = sdata[sidx];
			if (!order_state->sel.data()) {
				order_state->sel.Initialize();
			}
			order_state->sel.set_index(order_state->nsel++, i);
		}

		// Append nonempty slices to the arguments
		for (idx_t i = 0; i < count; ++i) {
			auto sidx = svdata.sel->get_index(i);
			auto order_state = sdata[sidx];
			if (!order_state->nsel) {
				continue;
			}

			order_state->UpdateSlice(*order_bind, sort_inputs, arg_inputs);
		}
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE *target, AggregateInputData &aggr_input_data) {
		const auto order_bind = (SortedAggregateBindData *)aggr_input_data.bind_data;
		auto &other = const_cast<STATE &>(source);
		target->Combine(*order_bind, other);
	}

	static void Window(Vector inputs[], const ValidityMask &filter_mask, AggregateInputData &aggr_input_data,
	                   idx_t input_count, data_ptr_t state, const FrameBounds &frame, const FrameBounds &prev,
	                   Vector &result, idx_t rid, idx_t bias) {
		throw InternalException("Sorted aggregates should not be generated for window clauses");
	}

	static void Finalize(Vector &states, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
	                     idx_t offset) {
		const auto order_bind = (SortedAggregateBindData *)aggr_input_data.bind_data;
		auto &buffer_manager = order_bind->buffer_manager;
		auto &orders = order_bind->orders;
		RowLayout payload_layout;
		payload_layout.Initialize(order_bind->arg_types);

		//	 Reusable inner state
		vector<data_t> agg_state(order_bind->function.state_size());
		Vector agg_state_vec(Value::POINTER((idx_t)agg_state.data()));

		// State variables
		const auto input_count = order_bind->function.arguments.size();
		auto bind_info = order_bind->bind_info.get();
		AggregateInputData aggr_bind_info(bind_info, Allocator::DefaultAllocator());

		// Inner aggregate APIs
		auto initialize = order_bind->function.initialize;
		auto destructor = order_bind->function.destructor;
		auto simple_update = order_bind->function.simple_update;
		auto update = order_bind->function.update;
		auto finalize = order_bind->function.finalize;

		auto sdata = FlatVector::GetData<SortedAggregateState *>(states);
		for (idx_t i = 0; i < count; ++i) {
			initialize(agg_state.data());
			auto state = sdata[i];

			// Apply the sort before delegating the chunks
			auto global_sort = make_unique<GlobalSortState>(buffer_manager, orders, payload_layout);
			LocalSortState local_sort;
			local_sort.Initialize(*global_sort, global_sort->buffer_manager);
			state->Finalize(local_sort);
			global_sort->AddLocalState(local_sort);

			if (!global_sort->sorted_blocks.empty()) {
				global_sort->PrepareMergePhase();
				while (global_sort->sorted_blocks.size() > 1) {
					global_sort->InitializeMergeRound();
					MergeSorter merge_sorter(*global_sort, global_sort->buffer_manager);
					merge_sorter.PerformInMergeRound();
					global_sort->CompleteMergeRound(false);
				}

				auto &chunk = state->arg_buffer;
				PayloadScanner scanner(*global_sort);
				for (;;) {
					chunk.Reset();
					scanner.Scan(chunk);
					if (chunk.size() == 0) {
						break;
					}
					// These are all simple updates, so use it if available
					if (simple_update) {
						simple_update(chunk.data.data(), aggr_bind_info, input_count, agg_state.data(), chunk.size());
					} else {
						// We are only updating a constant state
						agg_state_vec.SetVectorType(VectorType::CONSTANT_VECTOR);
						update(chunk.data.data(), aggr_bind_info, input_count, agg_state_vec, chunk.size());
					}
				}
			}

			// Finalize a single value at the next offset
			agg_state_vec.SetVectorType(states.GetVectorType());
			finalize(agg_state_vec, aggr_bind_info, result, 1, i + offset);

			if (destructor) {
				destructor(agg_state_vec, 1);
			}
		}
	}

	static void Serialize(FieldWriter &writer, const FunctionData *bind_data, const AggregateFunction &function) {
		throw NotImplementedException("FIXME: serialize sorted aggregate not supported");
	}
	static unique_ptr<FunctionData> Deserialize(ClientContext &context, FieldReader &reader,
	                                            AggregateFunction &function) {
		throw NotImplementedException("FIXME: deserialize sorted aggregate not supported");
	}
};

unique_ptr<FunctionData> FunctionBinder::BindSortedAggregate(AggregateFunction &bound_function,
                                                             vector<unique_ptr<Expression>> &children,
                                                             unique_ptr<FunctionData> bind_info,
                                                             unique_ptr<BoundOrderModifier> order_bys) {

	auto sorted_bind =
	    make_unique<SortedAggregateBindData>(context, bound_function, children, std::move(bind_info), *order_bys);

	// The arguments are the children plus the sort columns.
	for (auto &order : order_bys->orders) {
		children.emplace_back(std::move(order.expression));
	}

	vector<LogicalType> arguments;
	arguments.reserve(children.size());
	for (const auto &child : children) {
		arguments.emplace_back(child->return_type);
	}

	// Replace the aggregate with the wrapper
	AggregateFunction ordered_aggregate(
	    bound_function.name, arguments, bound_function.return_type, AggregateFunction::StateSize<SortedAggregateState>,
	    AggregateFunction::StateInitialize<SortedAggregateState, SortedAggregateFunction>,
	    SortedAggregateFunction::ScatterUpdate,
	    AggregateFunction::StateCombine<SortedAggregateState, SortedAggregateFunction>,
	    SortedAggregateFunction::Finalize, bound_function.null_handling, SortedAggregateFunction::SimpleUpdate, nullptr,
	    AggregateFunction::StateDestroy<SortedAggregateState, SortedAggregateFunction>, nullptr,
	    SortedAggregateFunction::Window, SortedAggregateFunction::Serialize, SortedAggregateFunction::Deserialize);

	bound_function = std::move(ordered_aggregate);

	return std::move(sorted_bind);
}

} // namespace duckdb










namespace duckdb {

BuiltinFunctions::BuiltinFunctions(CatalogTransaction transaction, Catalog &catalog)
    : transaction(transaction), catalog(catalog) {
}

BuiltinFunctions::~BuiltinFunctions() {
}

void BuiltinFunctions::AddCollation(string name, ScalarFunction function, bool combinable,
                                    bool not_required_for_equality) {
	CreateCollationInfo info(std::move(name), std::move(function), combinable, not_required_for_equality);
	info.internal = true;
	catalog.CreateCollation(transaction, &info);
}

void BuiltinFunctions::AddFunction(AggregateFunctionSet set) {
	CreateAggregateFunctionInfo info(std::move(set));
	info.internal = true;
	catalog.CreateFunction(transaction, &info);
}

void BuiltinFunctions::AddFunction(AggregateFunction function) {
	CreateAggregateFunctionInfo info(std::move(function));
	info.internal = true;
	catalog.CreateFunction(transaction, &info);
}

void BuiltinFunctions::AddFunction(PragmaFunction function) {
	CreatePragmaFunctionInfo info(std::move(function));
	info.internal = true;
	catalog.CreatePragmaFunction(transaction, &info);
}

void BuiltinFunctions::AddFunction(const string &name, PragmaFunctionSet functions) {
	CreatePragmaFunctionInfo info(name, std::move(functions));
	info.internal = true;
	catalog.CreatePragmaFunction(transaction, &info);
}

void BuiltinFunctions::AddFunction(ScalarFunction function) {
	CreateScalarFunctionInfo info(std::move(function));
	info.internal = true;
	catalog.CreateFunction(transaction, &info);
}

void BuiltinFunctions::AddFunction(const vector<string> &names, ScalarFunction function) { // NOLINT: false positive
	for (auto &name : names) {
		function.name = name;
		AddFunction(function);
	}
}

void BuiltinFunctions::AddFunction(ScalarFunctionSet set) {
	CreateScalarFunctionInfo info(std::move(set));
	info.internal = true;
	catalog.CreateFunction(transaction, &info);
}

void BuiltinFunctions::AddFunction(TableFunction function) {
	CreateTableFunctionInfo info(std::move(function));
	info.internal = true;
	catalog.CreateTableFunction(transaction, &info);
}

void BuiltinFunctions::AddFunction(TableFunctionSet set) {
	CreateTableFunctionInfo info(std::move(set));
	info.internal = true;
	catalog.CreateTableFunction(transaction, &info);
}

void BuiltinFunctions::AddFunction(CopyFunction function) {
	CreateCopyFunctionInfo info(std::move(function));
	info.internal = true;
	catalog.CreateCopyFunction(transaction, &info);
}

} // namespace duckdb



namespace duckdb {

BoundCastInfo DefaultCasts::BitCastSwitch(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// bit to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<string_t, duckdb::CastFromBit>);
	case LogicalTypeId::BLOB:
		return DefaultCasts::ReinterpretCast;
	default:
		return DefaultCasts::TryVectorNullCast;
	}
}

} // namespace duckdb



namespace duckdb {

BoundCastInfo DefaultCasts::BlobCastSwitch(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// blob to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<string_t, duckdb::CastFromBlob>);
	case LogicalTypeId::AGGREGATE_STATE:
	case LogicalTypeId::BIT:
		return DefaultCasts::ReinterpretCast;
	default:
		return DefaultCasts::TryVectorNullCast;
	}
}

} // namespace duckdb






namespace duckdb {

BindCastInput::BindCastInput(CastFunctionSet &function_set, BindCastInfo *info, ClientContext *context)
    : function_set(function_set), info(info), context(context) {
}

BoundCastInfo BindCastInput::GetCastFunction(const LogicalType &source, const LogicalType &target) {
	GetCastFunctionInput input(context);
	return function_set.GetCastFunction(source, target, input);
}

BindCastFunction::BindCastFunction(bind_cast_function_t function_p, unique_ptr<BindCastInfo> info_p)
    : function(function_p), info(std::move(info_p)) {
}

CastFunctionSet::CastFunctionSet() : map_info(nullptr) {
	bind_functions.emplace_back(DefaultCasts::GetDefaultCastFunction);
}

CastFunctionSet &CastFunctionSet::Get(ClientContext &context) {
	return DBConfig::GetConfig(context).GetCastFunctions();
}

CastFunctionSet &CastFunctionSet::Get(DatabaseInstance &db) {
	return DBConfig::GetConfig(db).GetCastFunctions();
}

BoundCastInfo CastFunctionSet::GetCastFunction(const LogicalType &source, const LogicalType &target,
                                               GetCastFunctionInput &get_input) {
	if (source == target) {
		return DefaultCasts::NopCast;
	}
	// the first function is the default
	// we iterate the set of bind functions backwards
	for (idx_t i = bind_functions.size(); i > 0; i--) {
		auto &bind_function = bind_functions[i - 1];
		BindCastInput input(*this, bind_function.info.get(), get_input.context);
		auto result = bind_function.function(input, source, target);
		if (result.function) {
			// found a cast function! return it
			return result;
		}
	}
	// no cast found: return the default null cast
	return DefaultCasts::TryVectorNullCast;
}

struct MapCastNode {
	MapCastNode(BoundCastInfo info, int64_t implicit_cast_cost)
	    : cast_info(std::move(info)), bind_function(nullptr), implicit_cast_cost(implicit_cast_cost) {
	}
	MapCastNode(bind_cast_function_t func, int64_t implicit_cast_cost)
	    : cast_info(nullptr), bind_function(func), implicit_cast_cost(implicit_cast_cost) {
	}

	BoundCastInfo cast_info;
	bind_cast_function_t bind_function;
	int64_t implicit_cast_cost;
};

struct MapCastInfo : public BindCastInfo {
	type_map_t<type_map_t<MapCastNode>> casts;
};

int64_t CastFunctionSet::ImplicitCastCost(const LogicalType &source, const LogicalType &target) {
	// check if a cast has been registered
	if (map_info) {
		auto source_entry = map_info->casts.find(source);
		if (source_entry != map_info->casts.end()) {
			auto target_entry = source_entry->second.find(target);
			if (target_entry != source_entry->second.end()) {
				return target_entry->second.implicit_cast_cost;
			}
		}
	}
	// if not, fallback to the default implicit cast rules
	return CastRules::ImplicitCast(source, target);
}

BoundCastInfo MapCastFunction(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	D_ASSERT(input.info);
	auto &map_info = (MapCastInfo &)*input.info;
	auto &casts = map_info.casts;

	auto entry = casts.find(source);
	if (entry == casts.end()) {
		// source type not found
		return nullptr;
	}
	auto target_entry = entry->second.find(target);
	if (target_entry == entry->second.end()) {
		// target type not found
		return nullptr;
	}
	if (target_entry->second.bind_function) {
		return target_entry->second.bind_function(input, source, target);
	}
	return target_entry->second.cast_info.Copy();
}

void CastFunctionSet::RegisterCastFunction(const LogicalType &source, const LogicalType &target, BoundCastInfo function,
                                           int64_t implicit_cast_cost) {
	RegisterCastFunction(source, target, MapCastNode(std::move(function), implicit_cast_cost));
}

void CastFunctionSet::RegisterCastFunction(const LogicalType &source, const LogicalType &target,
                                           bind_cast_function_t bind_function, int64_t implicit_cast_cost) {
	RegisterCastFunction(source, target, MapCastNode(bind_function, implicit_cast_cost));
}

void CastFunctionSet::RegisterCastFunction(const LogicalType &source, const LogicalType &target, MapCastNode node) {
	if (!map_info) {
		// create the cast map and the cast map function
		auto info = make_unique<MapCastInfo>();
		map_info = info.get();
		bind_functions.emplace_back(MapCastFunction, std::move(info));
	}
	map_info->casts[source].insert(make_pair(target, std::move(node)));
}

} // namespace duckdb








namespace duckdb {

template <class T>
static bool FromDecimalCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	auto &source_type = source.GetType();
	auto width = DecimalType::GetWidth(source_type);
	auto scale = DecimalType::GetScale(source_type);
	switch (source_type.InternalType()) {
	case PhysicalType::INT16:
		return VectorCastHelpers::TemplatedDecimalCast<int16_t, T, TryCastFromDecimal>(
		    source, result, count, parameters.error_message, width, scale);
	case PhysicalType::INT32:
		return VectorCastHelpers::TemplatedDecimalCast<int32_t, T, TryCastFromDecimal>(
		    source, result, count, parameters.error_message, width, scale);
	case PhysicalType::INT64:
		return VectorCastHelpers::TemplatedDecimalCast<int64_t, T, TryCastFromDecimal>(
		    source, result, count, parameters.error_message, width, scale);
	case PhysicalType::INT128:
		return VectorCastHelpers::TemplatedDecimalCast<hugeint_t, T, TryCastFromDecimal>(
		    source, result, count, parameters.error_message, width, scale);
	default:
		throw InternalException("Unimplemented internal type for decimal");
	}
}

template <class LIMIT_TYPE, class FACTOR_TYPE = LIMIT_TYPE>
struct DecimalScaleInput {
	DecimalScaleInput(Vector &result_p, FACTOR_TYPE factor_p) : result(result_p), factor(factor_p) {
	}
	DecimalScaleInput(Vector &result_p, LIMIT_TYPE limit_p, FACTOR_TYPE factor_p, string *error_message_p,
	                  uint8_t source_width_p, uint8_t source_scale_p)
	    : result(result_p), limit(limit_p), factor(factor_p), error_message(error_message_p),
	      source_width(source_width_p), source_scale(source_scale_p) {
	}

	Vector &result;
	LIMIT_TYPE limit;
	FACTOR_TYPE factor;
	bool all_converted = true;
	string *error_message;
	uint8_t source_width;
	uint8_t source_scale;
};

struct DecimalScaleUpOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, ValidityMask &mask, idx_t idx, void *dataptr) {
		auto data = (DecimalScaleInput<INPUT_TYPE, RESULT_TYPE> *)dataptr;
		return Cast::Operation<INPUT_TYPE, RESULT_TYPE>(input) * data->factor;
	}
};

struct DecimalScaleUpCheckOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, ValidityMask &mask, idx_t idx, void *dataptr) {
		auto data = (DecimalScaleInput<INPUT_TYPE, RESULT_TYPE> *)dataptr;
		if (input >= data->limit || input <= -data->limit) {
			auto error = StringUtil::Format("Casting value \"%s\" to type %s failed: value is out of range!",
			                                Decimal::ToString(input, data->source_width, data->source_scale),
			                                data->result.GetType().ToString());
			return HandleVectorCastError::Operation<RESULT_TYPE>(std::move(error), mask, idx, data->error_message,
			                                                     data->all_converted);
		}
		return Cast::Operation<INPUT_TYPE, RESULT_TYPE>(input) * data->factor;
	}
};

template <class SOURCE, class DEST, class POWERS_SOURCE, class POWERS_DEST>
bool TemplatedDecimalScaleUp(Vector &source, Vector &result, idx_t count, string *error_message) {
	auto source_scale = DecimalType::GetScale(source.GetType());
	auto source_width = DecimalType::GetWidth(source.GetType());
	auto result_scale = DecimalType::GetScale(result.GetType());
	auto result_width = DecimalType::GetWidth(result.GetType());
	D_ASSERT(result_scale >= source_scale);
	idx_t scale_difference = result_scale - source_scale;
	DEST multiply_factor = POWERS_DEST::POWERS_OF_TEN[scale_difference];
	idx_t target_width = result_width - scale_difference;
	if (source_width < target_width) {
		DecimalScaleInput<SOURCE, DEST> input(result, multiply_factor);
		// type will always fit: no need to check limit
		UnaryExecutor::GenericExecute<SOURCE, DEST, DecimalScaleUpOperator>(source, result, count, &input);
		return true;
	} else {
		// type might not fit: check limit
		auto limit = POWERS_SOURCE::POWERS_OF_TEN[target_width];
		DecimalScaleInput<SOURCE, DEST> input(result, limit, multiply_factor, error_message, source_width,
		                                      source_scale);
		UnaryExecutor::GenericExecute<SOURCE, DEST, DecimalScaleUpCheckOperator>(source, result, count, &input,
		                                                                         error_message);
		return input.all_converted;
	}
}

struct DecimalScaleDownOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, ValidityMask &mask, idx_t idx, void *dataptr) {
		auto data = (DecimalScaleInput<INPUT_TYPE> *)dataptr;
		return Cast::Operation<INPUT_TYPE, RESULT_TYPE>(input / data->factor);
	}
};

struct DecimalScaleDownCheckOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, ValidityMask &mask, idx_t idx, void *dataptr) {
		auto data = (DecimalScaleInput<INPUT_TYPE> *)dataptr;
		if (input >= data->limit || input <= -data->limit) {
			auto error = StringUtil::Format("Casting value \"%s\" to type %s failed: value is out of range!",
			                                Decimal::ToString(input, data->source_width, data->source_scale),
			                                data->result.GetType().ToString());
			return HandleVectorCastError::Operation<RESULT_TYPE>(std::move(error), mask, idx, data->error_message,
			                                                     data->all_converted);
		}
		return Cast::Operation<INPUT_TYPE, RESULT_TYPE>(input / data->factor);
	}
};

template <class SOURCE, class DEST, class POWERS_SOURCE>
bool TemplatedDecimalScaleDown(Vector &source, Vector &result, idx_t count, string *error_message) {
	auto source_scale = DecimalType::GetScale(source.GetType());
	auto source_width = DecimalType::GetWidth(source.GetType());
	auto result_scale = DecimalType::GetScale(result.GetType());
	auto result_width = DecimalType::GetWidth(result.GetType());
	D_ASSERT(result_scale < source_scale);
	idx_t scale_difference = source_scale - result_scale;
	idx_t target_width = result_width + scale_difference;
	SOURCE divide_factor = POWERS_SOURCE::POWERS_OF_TEN[scale_difference];
	if (source_width < target_width) {
		DecimalScaleInput<SOURCE> input(result, divide_factor);
		// type will always fit: no need to check limit
		UnaryExecutor::GenericExecute<SOURCE, DEST, DecimalScaleDownOperator>(source, result, count, &input);
		return true;
	} else {
		// type might not fit: check limit
		auto limit = POWERS_SOURCE::POWERS_OF_TEN[target_width];
		DecimalScaleInput<SOURCE> input(result, limit, divide_factor, error_message, source_width, source_scale);
		UnaryExecutor::GenericExecute<SOURCE, DEST, DecimalScaleDownCheckOperator>(source, result, count, &input,
		                                                                           error_message);
		return input.all_converted;
	}
}

template <class SOURCE, class POWERS_SOURCE>
static bool DecimalDecimalCastSwitch(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	auto source_scale = DecimalType::GetScale(source.GetType());
	auto result_scale = DecimalType::GetScale(result.GetType());
	source.GetType().Verify();
	result.GetType().Verify();

	// we need to either multiply or divide by the difference in scales
	if (result_scale >= source_scale) {
		// multiply
		switch (result.GetType().InternalType()) {
		case PhysicalType::INT16:
			return TemplatedDecimalScaleUp<SOURCE, int16_t, POWERS_SOURCE, NumericHelper>(source, result, count,
			                                                                              parameters.error_message);
		case PhysicalType::INT32:
			return TemplatedDecimalScaleUp<SOURCE, int32_t, POWERS_SOURCE, NumericHelper>(source, result, count,
			                                                                              parameters.error_message);
		case PhysicalType::INT64:
			return TemplatedDecimalScaleUp<SOURCE, int64_t, POWERS_SOURCE, NumericHelper>(source, result, count,
			                                                                              parameters.error_message);
		case PhysicalType::INT128:
			return TemplatedDecimalScaleUp<SOURCE, hugeint_t, POWERS_SOURCE, Hugeint>(source, result, count,
			                                                                          parameters.error_message);
		default:
			throw NotImplementedException("Unimplemented internal type for decimal");
		}
	} else {
		// divide
		switch (result.GetType().InternalType()) {
		case PhysicalType::INT16:
			return TemplatedDecimalScaleDown<SOURCE, int16_t, POWERS_SOURCE>(source, result, count,
			                                                                 parameters.error_message);
		case PhysicalType::INT32:
			return TemplatedDecimalScaleDown<SOURCE, int32_t, POWERS_SOURCE>(source, result, count,
			                                                                 parameters.error_message);
		case PhysicalType::INT64:
			return TemplatedDecimalScaleDown<SOURCE, int64_t, POWERS_SOURCE>(source, result, count,
			                                                                 parameters.error_message);
		case PhysicalType::INT128:
			return TemplatedDecimalScaleDown<SOURCE, hugeint_t, POWERS_SOURCE>(source, result, count,
			                                                                   parameters.error_message);
		default:
			throw NotImplementedException("Unimplemented internal type for decimal");
		}
	}
}

struct DecimalCastInput {
	DecimalCastInput(Vector &result_p, uint8_t width_p, uint8_t scale_p)
	    : result(result_p), width(width_p), scale(scale_p) {
	}

	Vector &result;
	uint8_t width;
	uint8_t scale;
};

struct StringCastFromDecimalOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, ValidityMask &mask, idx_t idx, void *dataptr) {
		auto data = (DecimalCastInput *)dataptr;
		return StringCastFromDecimal::Operation<INPUT_TYPE>(input, data->width, data->scale, data->result);
	}
};

template <class SRC>
static bool DecimalToStringCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	auto &source_type = source.GetType();
	auto width = DecimalType::GetWidth(source_type);
	auto scale = DecimalType::GetScale(source_type);
	DecimalCastInput input(result, width, scale);

	UnaryExecutor::GenericExecute<SRC, string_t, StringCastFromDecimalOperator>(source, result, count, (void *)&input);
	return true;
}

BoundCastInfo DefaultCasts::DecimalCastSwitch(BindCastInput &input, const LogicalType &source,
                                              const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::BOOLEAN:
		return FromDecimalCast<bool>;
	case LogicalTypeId::TINYINT:
		return FromDecimalCast<int8_t>;
	case LogicalTypeId::SMALLINT:
		return FromDecimalCast<int16_t>;
	case LogicalTypeId::INTEGER:
		return FromDecimalCast<int32_t>;
	case LogicalTypeId::BIGINT:
		return FromDecimalCast<int64_t>;
	case LogicalTypeId::UTINYINT:
		return FromDecimalCast<uint8_t>;
	case LogicalTypeId::USMALLINT:
		return FromDecimalCast<uint16_t>;
	case LogicalTypeId::UINTEGER:
		return FromDecimalCast<uint32_t>;
	case LogicalTypeId::UBIGINT:
		return FromDecimalCast<uint64_t>;
	case LogicalTypeId::HUGEINT:
		return FromDecimalCast<hugeint_t>;
	case LogicalTypeId::DECIMAL: {
		// decimal to decimal cast
		// first we need to figure out the source and target internal types
		switch (source.InternalType()) {
		case PhysicalType::INT16:
			return DecimalDecimalCastSwitch<int16_t, NumericHelper>;
		case PhysicalType::INT32:
			return DecimalDecimalCastSwitch<int32_t, NumericHelper>;
		case PhysicalType::INT64:
			return DecimalDecimalCastSwitch<int64_t, NumericHelper>;
		case PhysicalType::INT128:
			return DecimalDecimalCastSwitch<hugeint_t, Hugeint>;
		default:
			throw NotImplementedException("Unimplemented internal type for decimal in decimal_decimal cast");
		}
	}
	case LogicalTypeId::FLOAT:
		return FromDecimalCast<float>;
	case LogicalTypeId::DOUBLE:
		return FromDecimalCast<double>;
	case LogicalTypeId::VARCHAR: {
		switch (source.InternalType()) {
		case PhysicalType::INT16:
			return DecimalToStringCast<int16_t>;
		case PhysicalType::INT32:
			return DecimalToStringCast<int32_t>;
		case PhysicalType::INT64:
			return DecimalToStringCast<int64_t>;
		case PhysicalType::INT128:
			return DecimalToStringCast<hugeint_t>;
		default:
			throw InternalException("Unimplemented internal decimal type");
		}
	}
	default:
		return DefaultCasts::TryVectorNullCast;
	}
}

} // namespace duckdb












namespace duckdb {

BindCastInfo::~BindCastInfo() {
}

BoundCastData::~BoundCastData() {
}

BoundCastInfo::BoundCastInfo(cast_function_t function_p, unique_ptr<BoundCastData> cast_data_p,
                             init_cast_local_state_t init_local_state_p)
    : function(function_p), init_local_state(init_local_state_p), cast_data(std::move(cast_data_p)) {
}

BoundCastInfo BoundCastInfo::Copy() const {
	return BoundCastInfo(function, cast_data ? cast_data->Copy() : nullptr, init_local_state);
}

bool DefaultCasts::NopCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	result.Reference(source);
	return true;
}

static string UnimplementedCastMessage(const LogicalType &source_type, const LogicalType &target_type) {
	return StringUtil::Format("Unimplemented type for cast (%s -> %s)", source_type.ToString(), target_type.ToString());
}

// NULL cast only works if all values in source are NULL, otherwise an unimplemented cast exception is thrown
bool DefaultCasts::TryVectorNullCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	bool success = true;
	if (VectorOperations::HasNotNull(source, count)) {
		HandleCastError::AssignError(UnimplementedCastMessage(source.GetType(), result.GetType()),
		                             parameters.error_message);
		success = false;
	}
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	ConstantVector::SetNull(result, true);
	return success;
}

bool DefaultCasts::ReinterpretCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	result.Reinterpret(source);
	return true;
}

static bool AggregateStateToBlobCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	if (result.GetType().id() != LogicalTypeId::BLOB) {
		throw TypeMismatchException(source.GetType(), result.GetType(),
		                            "Cannot cast AGGREGATE_STATE to anything but BLOB");
	}
	result.Reinterpret(source);
	return true;
}

static bool NullTypeCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	// cast a NULL to another type, just copy the properties and change the type
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	ConstantVector::SetNull(result, true);
	return true;
}

BoundCastInfo DefaultCasts::GetDefaultCastFunction(BindCastInput &input, const LogicalType &source,
                                                   const LogicalType &target) {
	D_ASSERT(source != target);

	// first check if were casting to a union
	if (source.id() != LogicalTypeId::UNION && source.id() != LogicalTypeId::SQLNULL &&
	    target.id() == LogicalTypeId::UNION) {
		return ImplicitToUnionCast(input, source, target);
	}

	// else, switch on source type
	switch (source.id()) {
	case LogicalTypeId::BOOLEAN:
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
		return NumericCastSwitch(input, source, target);
	case LogicalTypeId::POINTER:
		return PointerCastSwitch(input, source, target);
	case LogicalTypeId::UUID:
		return UUIDCastSwitch(input, source, target);
	case LogicalTypeId::DECIMAL:
		return DecimalCastSwitch(input, source, target);
	case LogicalTypeId::DATE:
		return DateCastSwitch(input, source, target);
	case LogicalTypeId::TIME:
		return TimeCastSwitch(input, source, target);
	case LogicalTypeId::TIME_TZ:
		return TimeTzCastSwitch(input, source, target);
	case LogicalTypeId::TIMESTAMP:
		return TimestampCastSwitch(input, source, target);
	case LogicalTypeId::TIMESTAMP_TZ:
		return TimestampTzCastSwitch(input, source, target);
	case LogicalTypeId::TIMESTAMP_NS:
		return TimestampNsCastSwitch(input, source, target);
	case LogicalTypeId::TIMESTAMP_MS:
		return TimestampMsCastSwitch(input, source, target);
	case LogicalTypeId::TIMESTAMP_SEC:
		return TimestampSecCastSwitch(input, source, target);
	case LogicalTypeId::INTERVAL:
		return IntervalCastSwitch(input, source, target);
	case LogicalTypeId::VARCHAR:
		return StringCastSwitch(input, source, target);
	case LogicalTypeId::BLOB:
		return BlobCastSwitch(input, source, target);
	case LogicalTypeId::BIT:
		return BitCastSwitch(input, source, target);
	case LogicalTypeId::SQLNULL:
		return NullTypeCast;
	case LogicalTypeId::MAP:
		return MapCastSwitch(input, source, target);
	case LogicalTypeId::STRUCT:
		return StructCastSwitch(input, source, target);
	case LogicalTypeId::LIST:
		return ListCastSwitch(input, source, target);
	case LogicalTypeId::UNION:
		return UnionCastSwitch(input, source, target);
	case LogicalTypeId::ENUM:
		return EnumCastSwitch(input, source, target);
	case LogicalTypeId::AGGREGATE_STATE:
		return AggregateStateToBlobCast;
	default:
		return nullptr;
	}
}

} // namespace duckdb




namespace duckdb {

template <class SRC_TYPE, class RES_TYPE>
bool EnumEnumCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	bool all_converted = true;
	result.SetVectorType(VectorType::FLAT_VECTOR);

	auto &str_vec = EnumType::GetValuesInsertOrder(source.GetType());
	auto str_vec_ptr = FlatVector::GetData<string_t>(str_vec);

	auto res_enum_type = result.GetType();

	UnifiedVectorFormat vdata;
	source.ToUnifiedFormat(count, vdata);

	auto source_data = (SRC_TYPE *)vdata.data;
	auto source_sel = vdata.sel;
	auto source_mask = vdata.validity;

	auto result_data = FlatVector::GetData<RES_TYPE>(result);
	auto &result_mask = FlatVector::Validity(result);

	for (idx_t i = 0; i < count; i++) {
		auto src_idx = source_sel->get_index(i);
		if (!source_mask.RowIsValid(src_idx)) {
			result_mask.SetInvalid(i);
			continue;
		}
		auto key = EnumType::GetPos(res_enum_type, str_vec_ptr[source_data[src_idx]]);
		if (key == -1) {
			// key doesn't exist on result enum
			if (!parameters.error_message) {
				result_data[i] = HandleVectorCastError::Operation<RES_TYPE>(
				    CastExceptionText<SRC_TYPE, RES_TYPE>(source_data[src_idx]), result_mask, i,
				    parameters.error_message, all_converted);
			} else {
				result_mask.SetInvalid(i);
			}
			continue;
		}
		result_data[i] = key;
	}
	return all_converted;
}

template <class SRC_TYPE>
BoundCastInfo EnumEnumCastSwitch(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	switch (target.InternalType()) {
	case PhysicalType::UINT8:
		return EnumEnumCast<SRC_TYPE, uint8_t>;
	case PhysicalType::UINT16:
		return EnumEnumCast<SRC_TYPE, uint16_t>;
	case PhysicalType::UINT32:
		return EnumEnumCast<SRC_TYPE, uint32_t>;
	default:
		throw InternalException("ENUM can only have unsigned integers (except UINT64) as physical types");
	}
}

template <class SRC>
static bool EnumToVarcharCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	auto &enum_dictionary = EnumType::GetValuesInsertOrder(source.GetType());
	auto dictionary_data = FlatVector::GetData<string_t>(enum_dictionary);
	auto result_data = FlatVector::GetData<string_t>(result);
	auto &result_mask = FlatVector::Validity(result);

	UnifiedVectorFormat vdata;
	source.ToUnifiedFormat(count, vdata);

	auto source_data = (SRC *)vdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto source_idx = vdata.sel->get_index(i);
		if (!vdata.validity.RowIsValid(source_idx)) {
			result_mask.SetInvalid(i);
			continue;
		}
		auto enum_idx = source_data[source_idx];
		result_data[i] = dictionary_data[enum_idx];
	}
	if (source.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	} else {
		result.SetVectorType(VectorType::FLAT_VECTOR);
	}
	return true;
}

struct EnumBoundCastData : public BoundCastData {
	EnumBoundCastData(BoundCastInfo to_varchar_cast, BoundCastInfo from_varchar_cast)
	    : to_varchar_cast(std::move(to_varchar_cast)), from_varchar_cast(std::move(from_varchar_cast)) {
	}

	BoundCastInfo to_varchar_cast;
	BoundCastInfo from_varchar_cast;

public:
	unique_ptr<BoundCastData> Copy() const override {
		return make_unique<EnumBoundCastData>(to_varchar_cast.Copy(), from_varchar_cast.Copy());
	}
};

unique_ptr<BoundCastData> BindEnumCast(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	auto to_varchar_cast = input.GetCastFunction(source, LogicalType::VARCHAR);
	auto from_varchar_cast = input.GetCastFunction(LogicalType::VARCHAR, target);
	return make_unique<EnumBoundCastData>(std::move(to_varchar_cast), std::move(from_varchar_cast));
}

static bool EnumToAnyCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	auto &cast_data = (EnumBoundCastData &)*parameters.cast_data;

	Vector varchar_cast(LogicalType::VARCHAR, count);

	// cast to varchar
	CastParameters to_varchar_params(parameters, cast_data.to_varchar_cast.cast_data.get());
	cast_data.to_varchar_cast.function(source, varchar_cast, count, to_varchar_params);

	// cast from varchar to the target
	CastParameters from_varchar_params(parameters, cast_data.from_varchar_cast.cast_data.get());
	cast_data.from_varchar_cast.function(varchar_cast, result, count, from_varchar_params);
	return true;
}

BoundCastInfo DefaultCasts::EnumCastSwitch(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	auto enum_physical_type = source.InternalType();
	switch (target.id()) {
	case LogicalTypeId::ENUM: {
		// This means they are both ENUMs, but of different types.
		switch (enum_physical_type) {
		case PhysicalType::UINT8:
			return EnumEnumCastSwitch<uint8_t>(input, source, target);
		case PhysicalType::UINT16:
			return EnumEnumCastSwitch<uint16_t>(input, source, target);
		case PhysicalType::UINT32:
			return EnumEnumCastSwitch<uint32_t>(input, source, target);
		default:
			throw InternalException("ENUM can only have unsigned integers (except UINT64) as physical types");
		}
	}
	case LogicalTypeId::VARCHAR:
		switch (enum_physical_type) {
		case PhysicalType::UINT8:
			return EnumToVarcharCast<uint8_t>;
		case PhysicalType::UINT16:
			return EnumToVarcharCast<uint16_t>;
		case PhysicalType::UINT32:
			return EnumToVarcharCast<uint32_t>;
		default:
			throw InternalException("ENUM can only have unsigned integers (except UINT64) as physical types");
		}
	default: {
		return BoundCastInfo(EnumToAnyCast, BindEnumCast(input, source, target));
	}
	}
}

} // namespace duckdb



namespace duckdb {

unique_ptr<BoundCastData> ListBoundCastData::BindListToListCast(BindCastInput &input, const LogicalType &source,
                                                                const LogicalType &target) {
	vector<BoundCastInfo> child_cast_info;
	auto &source_child_type = ListType::GetChildType(source);
	auto &result_child_type = ListType::GetChildType(target);
	auto child_cast = input.GetCastFunction(source_child_type, result_child_type);
	return make_unique<ListBoundCastData>(std::move(child_cast));
}

bool ListCast::ListToListCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	auto &cast_data = (ListBoundCastData &)*parameters.cast_data;

	// only handle constant and flat vectors here for now
	if (source.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		result.SetVectorType(source.GetVectorType());
		ConstantVector::SetNull(result, ConstantVector::IsNull(source));

		auto ldata = ConstantVector::GetData<list_entry_t>(source);
		auto tdata = ConstantVector::GetData<list_entry_t>(result);
		*tdata = *ldata;
	} else {
		source.Flatten(count);
		result.SetVectorType(VectorType::FLAT_VECTOR);
		FlatVector::SetValidity(result, FlatVector::Validity(source));

		auto ldata = FlatVector::GetData<list_entry_t>(source);
		auto tdata = FlatVector::GetData<list_entry_t>(result);
		for (idx_t i = 0; i < count; i++) {
			tdata[i] = ldata[i];
		}
	}
	auto &source_cc = ListVector::GetEntry(source);
	auto source_size = ListVector::GetListSize(source);

	ListVector::Reserve(result, source_size);
	auto &append_vector = ListVector::GetEntry(result);

	CastParameters child_parameters(parameters, cast_data.child_cast_info.cast_data.get());
	if (!cast_data.child_cast_info.function(source_cc, append_vector, source_size, child_parameters)) {
		return false;
	}
	ListVector::SetListSize(result, source_size);
	D_ASSERT(ListVector::GetListSize(result) == source_size);
	return true;
}

static bool ListToVarcharCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	auto constant = source.GetVectorType() == VectorType::CONSTANT_VECTOR;
	// first cast the child vector to varchar
	Vector varchar_list(LogicalType::LIST(LogicalType::VARCHAR), count);
	ListCast::ListToListCast(source, varchar_list, count, parameters);

	// now construct the actual varchar vector
	varchar_list.Flatten(count);
	auto &child = ListVector::GetEntry(varchar_list);
	auto list_data = FlatVector::GetData<list_entry_t>(varchar_list);
	auto &validity = FlatVector::Validity(varchar_list);

	child.Flatten(count);
	auto child_data = FlatVector::GetData<string_t>(child);
	auto &child_validity = FlatVector::Validity(child);

	auto result_data = FlatVector::GetData<string_t>(result);
	static constexpr const idx_t SEP_LENGTH = 2;
	static constexpr const idx_t NULL_LENGTH = 4;
	for (idx_t i = 0; i < count; i++) {
		if (!validity.RowIsValid(i)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto list = list_data[i];
		// figure out how long the result needs to be
		idx_t list_length = 2; // "[" and "]"
		for (idx_t list_idx = 0; list_idx < list.length; list_idx++) {
			auto idx = list.offset + list_idx;
			if (list_idx > 0) {
				list_length += SEP_LENGTH; // ", "
			}
			// string length, or "NULL"
			list_length += child_validity.RowIsValid(idx) ? child_data[idx].GetSize() : NULL_LENGTH;
		}
		result_data[i] = StringVector::EmptyString(result, list_length);
		auto dataptr = result_data[i].GetDataWriteable();
		auto offset = 0;
		dataptr[offset++] = '[';
		for (idx_t list_idx = 0; list_idx < list.length; list_idx++) {
			auto idx = list.offset + list_idx;
			if (list_idx > 0) {
				memcpy(dataptr + offset, ", ", SEP_LENGTH);
				offset += SEP_LENGTH;
			}
			if (child_validity.RowIsValid(idx)) {
				auto len = child_data[idx].GetSize();
				memcpy(dataptr + offset, child_data[idx].GetDataUnsafe(), len);
				offset += len;
			} else {
				memcpy(dataptr + offset, "NULL", NULL_LENGTH);
				offset += NULL_LENGTH;
			}
		}
		dataptr[offset] = ']';
		result_data[i].Finalize();
	}

	if (constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
	return true;
}

BoundCastInfo DefaultCasts::ListCastSwitch(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	switch (target.id()) {
	case LogicalTypeId::LIST:
		return BoundCastInfo(ListCast::ListToListCast, ListBoundCastData::BindListToListCast(input, source, target));
	case LogicalTypeId::VARCHAR:
		return BoundCastInfo(ListToVarcharCast, ListBoundCastData::BindListToListCast(
		                                            input, source, LogicalType::LIST(LogicalType::VARCHAR)));
	default:
		return DefaultCasts::TryVectorNullCast;
	}
}

} // namespace duckdb



namespace duckdb {

unique_ptr<BoundCastData> MapBoundCastData::BindMapToMapCast(BindCastInput &input, const LogicalType &source,
                                                             const LogicalType &target) {
	vector<BoundCastInfo> child_cast_info;
	auto source_key = MapType::KeyType(source);
	auto target_key = MapType::KeyType(target);
	auto source_val = MapType::ValueType(source);
	auto target_val = MapType::ValueType(target);
	auto key_cast = input.GetCastFunction(source_key, target_key);
	auto value_cast = input.GetCastFunction(source_val, target_val);
	return make_unique<MapBoundCastData>(std::move(key_cast), std::move(value_cast));
}

static bool MapToVarcharCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	auto constant = source.GetVectorType() == VectorType::CONSTANT_VECTOR;
	auto varchar_type = LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR);
	Vector varchar_map(varchar_type, count);

	// since map's physical type is a list, the ListCast can be utilized
	ListCast::ListToListCast(source, varchar_map, count, parameters);

	varchar_map.Flatten(count);
	auto &validity = FlatVector::Validity(varchar_map);
	auto &key_str = MapVector::GetKeys(varchar_map);
	auto &val_str = MapVector::GetValues(varchar_map);

	key_str.Flatten(ListVector::GetListSize(source));
	val_str.Flatten(ListVector::GetListSize(source));

	auto list_data = ListVector::GetData(varchar_map);
	auto key_data = FlatVector::GetData<string_t>(key_str);
	auto val_data = FlatVector::GetData<string_t>(val_str);
	auto &key_validity = FlatVector::Validity(key_str);
	auto &val_validity = FlatVector::Validity(val_str);
	auto &struct_validity = FlatVector::Validity(ListVector::GetEntry(varchar_map));

	auto result_data = FlatVector::GetData<string_t>(result);
	for (idx_t i = 0; i < count; i++) {
		if (!validity.RowIsValid(i)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		auto list = list_data[i];
		string ret = "{";
		for (idx_t list_idx = 0; list_idx < list.length; list_idx++) {
			if (list_idx > 0) {
				ret += ", ";
			}
			auto idx = list.offset + list_idx;

			if (!struct_validity.RowIsValid(idx)) {
				ret += "NULL";
				continue;
			}
			if (!key_validity.RowIsValid(idx)) {
				// throw InternalException("Error in map: key validity invalid?!");
				ret += "invalid";
				continue;
			}
			ret += key_data[idx].GetString();
			ret += "=";
			ret += val_validity.RowIsValid(idx) ? val_data[idx].GetString() : "NULL";
		}
		ret += "}";
		result_data[i] = StringVector::AddString(result, ret);
	}

	if (constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
	return true;
}

BoundCastInfo DefaultCasts::MapCastSwitch(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	switch (target.id()) {
	case LogicalTypeId::MAP:
		return BoundCastInfo(ListCast::ListToListCast, ListBoundCastData::BindListToListCast(input, source, target));
	case LogicalTypeId::VARCHAR: {
		auto varchar_type = LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR);
		return BoundCastInfo(MapToVarcharCast, ListBoundCastData::BindListToListCast(input, source, varchar_type));
	}
	default:
		return TryVectorNullCast;
	}
}

} // namespace duckdb





namespace duckdb {

template <class SRC>
static BoundCastInfo InternalNumericCastSwitch(const LogicalType &source, const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::BOOLEAN:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<SRC, bool, duckdb::NumericTryCast>);
	case LogicalTypeId::TINYINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<SRC, int8_t, duckdb::NumericTryCast>);
	case LogicalTypeId::SMALLINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<SRC, int16_t, duckdb::NumericTryCast>);
	case LogicalTypeId::INTEGER:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<SRC, int32_t, duckdb::NumericTryCast>);
	case LogicalTypeId::BIGINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<SRC, int64_t, duckdb::NumericTryCast>);
	case LogicalTypeId::UTINYINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<SRC, uint8_t, duckdb::NumericTryCast>);
	case LogicalTypeId::USMALLINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<SRC, uint16_t, duckdb::NumericTryCast>);
	case LogicalTypeId::UINTEGER:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<SRC, uint32_t, duckdb::NumericTryCast>);
	case LogicalTypeId::UBIGINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<SRC, uint64_t, duckdb::NumericTryCast>);
	case LogicalTypeId::HUGEINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<SRC, hugeint_t, duckdb::NumericTryCast>);
	case LogicalTypeId::FLOAT:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<SRC, float, duckdb::NumericTryCast>);
	case LogicalTypeId::DOUBLE:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<SRC, double, duckdb::NumericTryCast>);
	case LogicalTypeId::DECIMAL:
		return BoundCastInfo(&VectorCastHelpers::ToDecimalCast<SRC>);
	case LogicalTypeId::VARCHAR:
		return BoundCastInfo(&VectorCastHelpers::StringCast<SRC, duckdb::StringCast>);
	default:
		return DefaultCasts::TryVectorNullCast;
	}
}

BoundCastInfo DefaultCasts::NumericCastSwitch(BindCastInput &input, const LogicalType &source,
                                              const LogicalType &target) {
	switch (source.id()) {
	case LogicalTypeId::BOOLEAN:
		return InternalNumericCastSwitch<bool>(source, target);
	case LogicalTypeId::TINYINT:
		return InternalNumericCastSwitch<int8_t>(source, target);
	case LogicalTypeId::SMALLINT:
		return InternalNumericCastSwitch<int16_t>(source, target);
	case LogicalTypeId::INTEGER:
		return InternalNumericCastSwitch<int32_t>(source, target);
	case LogicalTypeId::BIGINT:
		return InternalNumericCastSwitch<int64_t>(source, target);
	case LogicalTypeId::UTINYINT:
		return InternalNumericCastSwitch<uint8_t>(source, target);
	case LogicalTypeId::USMALLINT:
		return InternalNumericCastSwitch<uint16_t>(source, target);
	case LogicalTypeId::UINTEGER:
		return InternalNumericCastSwitch<uint32_t>(source, target);
	case LogicalTypeId::UBIGINT:
		return InternalNumericCastSwitch<uint64_t>(source, target);
	case LogicalTypeId::HUGEINT:
		return InternalNumericCastSwitch<hugeint_t>(source, target);
	case LogicalTypeId::FLOAT:
		return InternalNumericCastSwitch<float>(source, target);
	case LogicalTypeId::DOUBLE:
		return InternalNumericCastSwitch<double>(source, target);
	default:
		throw InternalException("NumericCastSwitch called with non-numeric argument");
	}
}

} // namespace duckdb



namespace duckdb {

BoundCastInfo DefaultCasts::PointerCastSwitch(BindCastInput &input, const LogicalType &source,
                                              const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// pointer to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<uintptr_t, duckdb::CastFromPointer>);
	default:
		return nullptr;
	}
}

} // namespace duckdb






namespace duckdb {

template <class T>
bool StringEnumCastLoop(string_t *source_data, ValidityMask &source_mask, const LogicalType &source_type,
                        T *result_data, ValidityMask &result_mask, const LogicalType &result_type, idx_t count,
                        string *error_message, const SelectionVector *sel) {
	bool all_converted = true;
	for (idx_t i = 0; i < count; i++) {
		idx_t source_idx = i;
		if (sel) {
			source_idx = sel->get_index(i);
		}
		if (source_mask.RowIsValid(source_idx)) {
			auto pos = EnumType::GetPos(result_type, source_data[source_idx]);
			if (pos == -1) {
				result_data[i] =
				    HandleVectorCastError::Operation<T>(CastExceptionText<string_t, T>(source_data[source_idx]),
				                                        result_mask, i, error_message, all_converted);
			} else {
				result_data[i] = pos;
			}
		} else {
			result_mask.SetInvalid(i);
		}
	}
	return all_converted;
}

template <class T>
bool StringEnumCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	D_ASSERT(source.GetType().id() == LogicalTypeId::VARCHAR);
	auto enum_name = EnumType::GetTypeName(result.GetType());
	switch (source.GetVectorType()) {
	case VectorType::CONSTANT_VECTOR: {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);

		auto source_data = ConstantVector::GetData<string_t>(source);
		auto source_mask = ConstantVector::Validity(source);
		auto result_data = ConstantVector::GetData<T>(result);
		auto &result_mask = ConstantVector::Validity(result);

		return StringEnumCastLoop(source_data, source_mask, source.GetType(), result_data, result_mask,
		                          result.GetType(), 1, parameters.error_message, nullptr);
	}
	default: {
		UnifiedVectorFormat vdata;
		source.ToUnifiedFormat(count, vdata);

		result.SetVectorType(VectorType::FLAT_VECTOR);

		auto source_data = (string_t *)vdata.data;
		auto source_sel = vdata.sel;
		auto source_mask = vdata.validity;
		auto result_data = FlatVector::GetData<T>(result);
		auto &result_mask = FlatVector::Validity(result);

		return StringEnumCastLoop(source_data, source_mask, source.GetType(), result_data, result_mask,
		                          result.GetType(), count, parameters.error_message, source_sel);
	}
	}
}

static BoundCastInfo VectorStringCastNumericSwitch(BindCastInput &input, const LogicalType &source,
                                                   const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::ENUM: {
		switch (target.InternalType()) {
		case PhysicalType::UINT8:
			return StringEnumCast<uint8_t>;
		case PhysicalType::UINT16:
			return StringEnumCast<uint16_t>;
		case PhysicalType::UINT32:
			return StringEnumCast<uint32_t>;
		default:
			throw InternalException("ENUM can only have unsigned integers (except UINT64) as physical types");
		}
	}
	case LogicalTypeId::BOOLEAN:
		return BoundCastInfo(&VectorCastHelpers::TryCastStrictLoop<string_t, bool, duckdb::TryCast>);
	case LogicalTypeId::TINYINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastStrictLoop<string_t, int8_t, duckdb::TryCast>);
	case LogicalTypeId::SMALLINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastStrictLoop<string_t, int16_t, duckdb::TryCast>);
	case LogicalTypeId::INTEGER:
		return BoundCastInfo(&VectorCastHelpers::TryCastStrictLoop<string_t, int32_t, duckdb::TryCast>);
	case LogicalTypeId::BIGINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastStrictLoop<string_t, int64_t, duckdb::TryCast>);
	case LogicalTypeId::UTINYINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastStrictLoop<string_t, uint8_t, duckdb::TryCast>);
	case LogicalTypeId::USMALLINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastStrictLoop<string_t, uint16_t, duckdb::TryCast>);
	case LogicalTypeId::UINTEGER:
		return BoundCastInfo(&VectorCastHelpers::TryCastStrictLoop<string_t, uint32_t, duckdb::TryCast>);
	case LogicalTypeId::UBIGINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastStrictLoop<string_t, uint64_t, duckdb::TryCast>);
	case LogicalTypeId::HUGEINT:
		return BoundCastInfo(&VectorCastHelpers::TryCastStrictLoop<string_t, hugeint_t, duckdb::TryCast>);
	case LogicalTypeId::FLOAT:
		return BoundCastInfo(&VectorCastHelpers::TryCastStrictLoop<string_t, float, duckdb::TryCast>);
	case LogicalTypeId::DOUBLE:
		return BoundCastInfo(&VectorCastHelpers::TryCastStrictLoop<string_t, double, duckdb::TryCast>);
	case LogicalTypeId::INTERVAL:
		return BoundCastInfo(&VectorCastHelpers::TryCastErrorLoop<string_t, interval_t, duckdb::TryCastErrorMessage>);
	case LogicalTypeId::DECIMAL:
		return BoundCastInfo(&VectorCastHelpers::ToDecimalCast<string_t>);
	default:
		return DefaultCasts::TryVectorNullCast;
	}
}

// string -> list casting
bool VectorStringToList::StringToNestedTypeCastLoop(string_t *source_data, ValidityMask &source_mask, Vector &result,
                                                    ValidityMask &result_mask, idx_t count, CastParameters &parameters,
                                                    const SelectionVector *sel) {
	idx_t total_list_size = 0;
	for (idx_t i = 0; i < count; i++) {
		idx_t idx = i;
		if (sel) {
			idx = sel->get_index(i);
		}
		if (!source_mask.RowIsValid(idx)) {
			continue;
		}
		total_list_size += VectorStringToList::CountPartsList(source_data[idx]);
	}

	Vector varchar_vector(LogicalType::VARCHAR, total_list_size);

	ListVector::Reserve(result, total_list_size);
	ListVector::SetListSize(result, total_list_size);

	auto list_data = ListVector::GetData(result);
	auto child_data = FlatVector::GetData<string_t>(varchar_vector);

	bool all_converted = true;
	idx_t total = 0;
	for (idx_t i = 0; i < count; i++) {
		idx_t idx = i;
		if (sel) {
			idx = sel->get_index(i);
		}
		if (!source_mask.RowIsValid(idx)) {
			result_mask.SetInvalid(i);
			continue;
		}

		list_data[i].offset = total;
		if (!VectorStringToList::SplitStringList(source_data[idx], child_data, total, varchar_vector)) {
			string text = "Type VARCHAR with value '" + source_data[idx].GetString() +
			              "' can't be cast to the destination type LIST";
			HandleVectorCastError::Operation<string_t>(text, result_mask, idx, parameters.error_message, all_converted);
		}
		list_data[i].length = total - list_data[i].offset; // length is the amount of parts coming from this string
	}
	D_ASSERT(total_list_size == total);

	auto &result_child = ListVector::GetEntry(result);
	auto &cast_data = (ListBoundCastData &)*parameters.cast_data;
	CastParameters child_parameters(parameters, cast_data.child_cast_info.cast_data.get());
	return cast_data.child_cast_info.function(varchar_vector, result_child, total_list_size, child_parameters) &&
	       all_converted;
}

static LogicalType InitVarcharStructType(const LogicalType &target) {
	child_list_t<LogicalType> child_types;
	for (auto &child : StructType::GetChildTypes(target)) {
		child_types.push_back(make_pair(child.first, LogicalType::VARCHAR));
	}

	return LogicalType::STRUCT(child_types);
}

// string -> struct casting
bool VectorStringToStruct::StringToNestedTypeCastLoop(string_t *source_data, ValidityMask &source_mask, Vector &result,
                                                      ValidityMask &result_mask, idx_t count,
                                                      CastParameters &parameters, const SelectionVector *sel) {

	auto varchar_struct_type = InitVarcharStructType(result.GetType());
	Vector varchar_vector(varchar_struct_type, count);
	auto &child_vectors = StructVector::GetEntries(varchar_vector);
	auto &result_children = StructVector::GetEntries(result);

	string_map_t<idx_t> child_names;
	vector<ValidityMask *> child_masks;
	for (idx_t child_idx = 0; child_idx < result_children.size(); child_idx++) {
		child_names.insert({StructType::GetChildName(result.GetType(), child_idx), child_idx});
		child_masks.emplace_back(&FlatVector::Validity(*child_vectors[child_idx]));
		child_masks[child_idx]->SetAllInvalid(count);
	}

	bool all_converted = true;
	for (idx_t i = 0; i < count; i++) {
		idx_t idx = i;
		if (sel) {
			idx = sel->get_index(i);
		}
		if (!source_mask.RowIsValid(idx)) {
			result_mask.SetInvalid(i);
			continue;
		}
		if (!VectorStringToStruct::SplitStruct(source_data[idx], child_vectors, i, child_names, child_masks)) {
			string text = "Type VARCHAR with value '" + source_data[idx].GetString() +
			              "' can't be cast to the destination type STRUCT";
			for (auto &child_mask : child_masks) {
				child_mask->SetInvalid(idx); // some values may have already been found and set valid
			}
			HandleVectorCastError::Operation<string_t>(text, result_mask, idx, parameters.error_message, all_converted);
		}
	}

	auto &cast_data = (StructBoundCastData &)*parameters.cast_data;
	D_ASSERT(cast_data.child_cast_info.size() == result_children.size());

	for (idx_t child_idx = 0; child_idx < result_children.size(); child_idx++) {
		auto &varchar_vector = *child_vectors[child_idx];
		auto &result_child_vector = *result_children[child_idx];
		auto &child_cast_info = cast_data.child_cast_info[child_idx];
		CastParameters child_parameters(parameters, child_cast_info.cast_data.get());
		if (!child_cast_info.function(varchar_vector, result_child_vector, count, child_parameters)) {
			all_converted = false;
		}
	}
	return all_converted;
}

// string -> map casting
bool VectorStringToMap::StringToNestedTypeCastLoop(string_t *source_data, ValidityMask &source_mask, Vector &result,
                                                   ValidityMask &result_mask, idx_t count, CastParameters &parameters,
                                                   const SelectionVector *sel) {
	idx_t total_elements = 0;
	for (idx_t i = 0; i < count; i++) {
		idx_t idx = i;
		if (sel) {
			idx = sel->get_index(i);
		}
		if (!source_mask.RowIsValid(idx)) {
			continue;
		}
		total_elements += (VectorStringToMap::CountPartsMap(source_data[idx]) + 1) / 2;
	}

	Vector varchar_key_vector(LogicalType::VARCHAR, total_elements);
	Vector varchar_val_vector(LogicalType::VARCHAR, total_elements);
	auto child_key_data = FlatVector::GetData<string_t>(varchar_key_vector);
	auto child_val_data = FlatVector::GetData<string_t>(varchar_val_vector);

	ListVector::Reserve(result, total_elements);
	ListVector::SetListSize(result, total_elements);
	auto list_data = ListVector::GetData(result);

	bool all_converted = true;
	idx_t total = 0;
	for (idx_t i = 0; i < count; i++) {
		idx_t idx = i;
		if (sel) {
			idx = sel->get_index(i);
		}
		if (!source_mask.RowIsValid(idx)) {
			result_mask.SetInvalid(idx);
			continue;
		}

		list_data[i].offset = total;
		if (!VectorStringToMap::SplitStringMap(source_data[idx], child_key_data, child_val_data, total,
		                                       varchar_key_vector, varchar_val_vector)) {
			string text = "Type VARCHAR with value '" + source_data[idx].GetString() +
			              "' can't be cast to the destination type MAP";
			FlatVector::SetNull(result, idx, true);
			HandleVectorCastError::Operation<string_t>(text, result_mask, idx, parameters.error_message, all_converted);
		}
		list_data[i].length = total - list_data[i].offset;
	}
	D_ASSERT(total_elements == total);

	auto &result_key_child = MapVector::GetKeys(result);
	auto &result_val_child = MapVector::GetValues(result);
	auto &cast_data = (MapBoundCastData &)*parameters.cast_data;

	CastParameters key_params(parameters, cast_data.key_cast.cast_data.get());
	if (!cast_data.key_cast.function(varchar_key_vector, result_key_child, total_elements, key_params)) {
		all_converted = false;
	}
	CastParameters val_params(parameters, cast_data.value_cast.cast_data.get());
	if (!cast_data.value_cast.function(varchar_val_vector, result_val_child, total_elements, val_params)) {
		all_converted = false;
	}

	auto &key_validity = FlatVector::Validity(result_key_child);
	if (!all_converted) {
		for (idx_t row_idx = 0; row_idx < count; row_idx++) {
			if (!result_mask.RowIsValid(row_idx)) {
				continue;
			}
			auto list = list_data[row_idx];
			for (idx_t list_idx = 0; list_idx < list.length; list_idx++) {
				auto idx = list.offset + list_idx;
				if (!key_validity.RowIsValid(idx)) {
					result_mask.SetInvalid(row_idx);
				}
			}
		}
	}
	MapConversionVerify(result, count);
	return all_converted;
}

template <class T>
bool StringToNestedTypeCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	D_ASSERT(source.GetType().id() == LogicalTypeId::VARCHAR);

	switch (source.GetVectorType()) {
	case VectorType::CONSTANT_VECTOR: {
		auto source_data = ConstantVector::GetData<string_t>(source);
		auto &source_mask = ConstantVector::Validity(source);
		auto &result_mask = FlatVector::Validity(result);
		auto ret = T::StringToNestedTypeCastLoop(source_data, source_mask, result, result_mask, 1, parameters, nullptr);
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		return ret;
	}
	default: {
		UnifiedVectorFormat unified_source;

		source.ToUnifiedFormat(count, unified_source);
		auto source_sel = unified_source.sel;
		auto source_data = (string_t *)unified_source.data;
		auto &source_mask = unified_source.validity;
		auto &result_mask = FlatVector::Validity(result);

		return T::StringToNestedTypeCastLoop(source_data, source_mask, result, result_mask, count, parameters,
		                                     source_sel);
	}
	}
}

BoundCastInfo DefaultCasts::StringCastSwitch(BindCastInput &input, const LogicalType &source,
                                             const LogicalType &target) {
	switch (target.id()) {
	case LogicalTypeId::DATE:
		return BoundCastInfo(&VectorCastHelpers::TryCastErrorLoop<string_t, date_t, duckdb::TryCastErrorMessage>);
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_TZ:
		return BoundCastInfo(&VectorCastHelpers::TryCastErrorLoop<string_t, dtime_t, duckdb::TryCastErrorMessage>);
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ:
		return BoundCastInfo(&VectorCastHelpers::TryCastErrorLoop<string_t, timestamp_t, duckdb::TryCastErrorMessage>);
	case LogicalTypeId::TIMESTAMP_NS:
		return BoundCastInfo(
		    &VectorCastHelpers::TryCastStrictLoop<string_t, timestamp_t, duckdb::TryCastToTimestampNS>);
	case LogicalTypeId::TIMESTAMP_SEC:
		return BoundCastInfo(
		    &VectorCastHelpers::TryCastStrictLoop<string_t, timestamp_t, duckdb::TryCastToTimestampSec>);
	case LogicalTypeId::TIMESTAMP_MS:
		return BoundCastInfo(
		    &VectorCastHelpers::TryCastStrictLoop<string_t, timestamp_t, duckdb::TryCastToTimestampMS>);
	case LogicalTypeId::BLOB:
		return BoundCastInfo(&VectorCastHelpers::TryCastStringLoop<string_t, string_t, duckdb::TryCastToBlob>);
	case LogicalTypeId::BIT:
		return BoundCastInfo(&VectorCastHelpers::TryCastStringLoop<string_t, string_t, duckdb::TryCastToBit>);
	case LogicalTypeId::UUID:
		return BoundCastInfo(&VectorCastHelpers::TryCastStringLoop<string_t, hugeint_t, duckdb::TryCastToUUID>);
	case LogicalTypeId::SQLNULL:
		return &DefaultCasts::TryVectorNullCast;
	case LogicalTypeId::VARCHAR:
		return &DefaultCasts::ReinterpretCast;
	case LogicalTypeId::LIST:
		// the second argument allows for a secondary casting function to be passed in the CastParameters
		return BoundCastInfo(
		    &StringToNestedTypeCast<VectorStringToList>,
		    ListBoundCastData::BindListToListCast(input, LogicalType::LIST(LogicalType::VARCHAR), target));
	case LogicalTypeId::STRUCT:
		return BoundCastInfo(&StringToNestedTypeCast<VectorStringToStruct>,
		                     StructBoundCastData::BindStructToStructCast(input, InitVarcharStructType(target), target));
	case LogicalTypeId::MAP:
		return BoundCastInfo(&StringToNestedTypeCast<VectorStringToMap>,
		                     MapBoundCastData::BindMapToMapCast(
		                         input, LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR), target));
	default:
		return VectorStringCastNumericSwitch(input, source, target);
	}
}

} // namespace duckdb



namespace duckdb {

unique_ptr<BoundCastData> StructBoundCastData::BindStructToStructCast(BindCastInput &input, const LogicalType &source,
                                                                      const LogicalType &target) {
	vector<BoundCastInfo> child_cast_info;
	auto &source_child_types = StructType::GetChildTypes(source);
	auto &result_child_types = StructType::GetChildTypes(target);
	if (source_child_types.size() != result_child_types.size()) {
		throw TypeMismatchException(source, target, "Cannot cast STRUCTs of different size");
	}
	for (idx_t i = 0; i < source_child_types.size(); i++) {
		auto child_cast = input.GetCastFunction(source_child_types[i].second, result_child_types[i].second);
		child_cast_info.push_back(std::move(child_cast));
	}
	return make_unique<StructBoundCastData>(std::move(child_cast_info), target);
}

static bool StructToStructCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	auto &cast_data = (StructBoundCastData &)*parameters.cast_data;
	auto &source_child_types = StructType::GetChildTypes(source.GetType());
	auto &source_children = StructVector::GetEntries(source);
	D_ASSERT(source_children.size() == StructType::GetChildTypes(result.GetType()).size());

	auto &result_children = StructVector::GetEntries(result);
	bool all_converted = true;
	for (idx_t c_idx = 0; c_idx < source_child_types.size(); c_idx++) {
		auto &result_child_vector = *result_children[c_idx];
		auto &source_child_vector = *source_children[c_idx];
		CastParameters child_parameters(parameters, cast_data.child_cast_info[c_idx].cast_data.get());
		if (!cast_data.child_cast_info[c_idx].function(source_child_vector, result_child_vector, count,
		                                               child_parameters)) {
			all_converted = false;
		}
	}
	if (source.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		ConstantVector::SetNull(result, ConstantVector::IsNull(source));
	} else {
		source.Flatten(count);
		FlatVector::Validity(result) = FlatVector::Validity(source);
	}
	return all_converted;
}

static bool StructToVarcharCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	auto constant = source.GetVectorType() == VectorType::CONSTANT_VECTOR;
	// first cast all child elements to varchar
	auto &cast_data = (StructBoundCastData &)*parameters.cast_data;
	Vector varchar_struct(cast_data.target, count);
	StructToStructCast(source, varchar_struct, count, parameters);

	// now construct the actual varchar vector
	varchar_struct.Flatten(count);
	auto &child_types = StructType::GetChildTypes(source.GetType());
	auto &children = StructVector::GetEntries(varchar_struct);
	auto &validity = FlatVector::Validity(varchar_struct);
	auto result_data = FlatVector::GetData<string_t>(result);
	static constexpr const idx_t SEP_LENGTH = 2;
	static constexpr const idx_t NAME_SEP_LENGTH = 4;
	static constexpr const idx_t NULL_LENGTH = 4;
	for (idx_t i = 0; i < count; i++) {
		if (!validity.RowIsValid(i)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		idx_t string_length = 2; // {}
		for (idx_t c = 0; c < children.size(); c++) {
			if (c > 0) {
				string_length += SEP_LENGTH;
			}
			children[c]->Flatten(count);
			auto &child_validity = FlatVector::Validity(*children[c]);
			auto data = FlatVector::GetData<string_t>(*children[c]);
			auto &name = child_types[c].first;
			string_length += name.size() + NAME_SEP_LENGTH; // "'{name}': "
			string_length += child_validity.RowIsValid(i) ? data[i].GetSize() : NULL_LENGTH;
		}
		result_data[i] = StringVector::EmptyString(result, string_length);
		auto dataptr = result_data[i].GetDataWriteable();
		idx_t offset = 0;
		dataptr[offset++] = '{';
		for (idx_t c = 0; c < children.size(); c++) {
			if (c > 0) {
				memcpy(dataptr + offset, ", ", SEP_LENGTH);
				offset += SEP_LENGTH;
			}
			auto &child_validity = FlatVector::Validity(*children[c]);
			auto data = FlatVector::GetData<string_t>(*children[c]);
			auto &name = child_types[c].first;
			// "'{name}': "
			dataptr[offset++] = '\'';
			memcpy(dataptr + offset, name.c_str(), name.size());
			offset += name.size();
			dataptr[offset++] = '\'';
			dataptr[offset++] = ':';
			dataptr[offset++] = ' ';
			// value
			if (child_validity.RowIsValid(i)) {
				auto len = data[i].GetSize();
				memcpy(dataptr + offset, data[i].GetDataUnsafe(), len);
				offset += len;
			} else {
				memcpy(dataptr + offset, "NULL", NULL_LENGTH);
				offset += NULL_LENGTH;
			}
		}
		dataptr[offset++] = '}';
		result_data[i].Finalize();
	}

	if (constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
	return true;
}

BoundCastInfo DefaultCasts::StructCastSwitch(BindCastInput &input, const LogicalType &source,
                                             const LogicalType &target) {
	switch (target.id()) {
	case LogicalTypeId::STRUCT:
		return BoundCastInfo(StructToStructCast, StructBoundCastData::BindStructToStructCast(input, source, target));
	case LogicalTypeId::VARCHAR: {
		// bind a cast in which we convert all child entries to VARCHAR entries
		auto &struct_children = StructType::GetChildTypes(source);
		child_list_t<LogicalType> varchar_children;
		for (auto &child_entry : struct_children) {
			varchar_children.push_back(make_pair(child_entry.first, LogicalType::VARCHAR));
		}
		auto varchar_type = LogicalType::STRUCT(std::move(varchar_children));
		return BoundCastInfo(StructToVarcharCast,
		                     StructBoundCastData::BindStructToStructCast(input, source, varchar_type));
	}
	default:
		return TryVectorNullCast;
	}
}

} // namespace duckdb



namespace duckdb {

BoundCastInfo DefaultCasts::DateCastSwitch(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// date to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<date_t, duckdb::StringCast>);
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ:
		// date to timestamp
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<date_t, timestamp_t, duckdb::TryCast>);
	case LogicalTypeId::TIMESTAMP_NS:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<date_t, timestamp_t, duckdb::TryCastToTimestampNS>);
	case LogicalTypeId::TIMESTAMP_SEC:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<date_t, timestamp_t, duckdb::TryCastToTimestampSec>);
	case LogicalTypeId::TIMESTAMP_MS:
		return BoundCastInfo(&VectorCastHelpers::TryCastLoop<date_t, timestamp_t, duckdb::TryCastToTimestampMS>);
	default:
		return TryVectorNullCast;
	}
}

BoundCastInfo DefaultCasts::TimeCastSwitch(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// time to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<dtime_t, duckdb::StringCast>);
	case LogicalTypeId::TIME_TZ:
		// time to time with time zone
		return ReinterpretCast;
	default:
		return TryVectorNullCast;
	}
}

BoundCastInfo DefaultCasts::TimeTzCastSwitch(BindCastInput &input, const LogicalType &source,
                                             const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// time with time zone to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<dtime_t, duckdb::StringCastTZ>);
	case LogicalTypeId::TIME:
		// time with time zone to time
		return ReinterpretCast;
	default:
		return TryVectorNullCast;
	}
}

BoundCastInfo DefaultCasts::TimestampCastSwitch(BindCastInput &input, const LogicalType &source,
                                                const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// timestamp to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<timestamp_t, duckdb::StringCast>);
	case LogicalTypeId::DATE:
		// timestamp to date
		return BoundCastInfo(&VectorCastHelpers::TemplatedCastLoop<timestamp_t, date_t, duckdb::Cast>);
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_TZ:
		// timestamp to time
		return BoundCastInfo(&VectorCastHelpers::TemplatedCastLoop<timestamp_t, dtime_t, duckdb::Cast>);
	case LogicalTypeId::TIMESTAMP_TZ:
		// timestamp (us) to timestamp with time zone
		return BoundCastInfo(&VectorCastHelpers::TemplatedCastLoop<timestamp_t, timestamp_t, duckdb::Cast>);
	case LogicalTypeId::TIMESTAMP_NS:
		// timestamp (us) to timestamp (ns)
		return BoundCastInfo(
		    &VectorCastHelpers::TemplatedCastLoop<timestamp_t, timestamp_t, duckdb::CastTimestampUsToNs>);
	case LogicalTypeId::TIMESTAMP_MS:
		// timestamp (us) to timestamp (ms)
		return BoundCastInfo(
		    &VectorCastHelpers::TemplatedCastLoop<timestamp_t, timestamp_t, duckdb::CastTimestampUsToMs>);
	case LogicalTypeId::TIMESTAMP_SEC:
		// timestamp (us) to timestamp (s)
		return BoundCastInfo(
		    &VectorCastHelpers::TemplatedCastLoop<timestamp_t, timestamp_t, duckdb::CastTimestampUsToSec>);
	default:
		return TryVectorNullCast;
	}
}

BoundCastInfo DefaultCasts::TimestampTzCastSwitch(BindCastInput &input, const LogicalType &source,
                                                  const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// timestamp with time zone to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<timestamp_t, duckdb::StringCastTZ>);
	case LogicalTypeId::TIME_TZ:
		// timestamp with time zone to time with time zone.
		// TODO: set the offset to +00
		return BoundCastInfo(&VectorCastHelpers::TemplatedCastLoop<timestamp_t, dtime_t, duckdb::Cast>);
	case LogicalTypeId::TIMESTAMP:
		// timestamp with time zone to timestamp (us)
		return BoundCastInfo(&VectorCastHelpers::TemplatedCastLoop<timestamp_t, timestamp_t, duckdb::Cast>);
	default:
		return TryVectorNullCast;
	}
}

BoundCastInfo DefaultCasts::TimestampNsCastSwitch(BindCastInput &input, const LogicalType &source,
                                                  const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// timestamp (ns) to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<timestamp_t, duckdb::CastFromTimestampNS>);
	case LogicalTypeId::TIMESTAMP:
		// timestamp (ns) to timestamp (us)
		return BoundCastInfo(
		    &VectorCastHelpers::TemplatedCastLoop<timestamp_t, timestamp_t, duckdb::CastTimestampNsToUs>);
	default:
		return TryVectorNullCast;
	}
}

BoundCastInfo DefaultCasts::TimestampMsCastSwitch(BindCastInput &input, const LogicalType &source,
                                                  const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// timestamp (ms) to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<timestamp_t, duckdb::CastFromTimestampMS>);
	case LogicalTypeId::TIMESTAMP:
		// timestamp (ms) to timestamp (us)
		return BoundCastInfo(
		    &VectorCastHelpers::TemplatedCastLoop<timestamp_t, timestamp_t, duckdb::CastTimestampMsToUs>);
	default:
		return TryVectorNullCast;
	}
}

BoundCastInfo DefaultCasts::TimestampSecCastSwitch(BindCastInput &input, const LogicalType &source,
                                                   const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// timestamp (sec) to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<timestamp_t, duckdb::CastFromTimestampSec>);
	case LogicalTypeId::TIMESTAMP:
		// timestamp (s) to timestamp (us)
		return BoundCastInfo(
		    &VectorCastHelpers::TemplatedCastLoop<timestamp_t, timestamp_t, duckdb::CastTimestampSecToUs>);
	default:
		return TryVectorNullCast;
	}
}
BoundCastInfo DefaultCasts::IntervalCastSwitch(BindCastInput &input, const LogicalType &source,
                                               const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// time to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<interval_t, duckdb::StringCast>);
	default:
		return TryVectorNullCast;
	}
}

} // namespace duckdb




#include <algorithm> // for std::sort

namespace duckdb {

//--------------------------------------------------------------------------------------------------
// ??? -> UNION
//--------------------------------------------------------------------------------------------------
// if the source can be implicitly cast to a member of the target union, the cast is valid

struct ToUnionBoundCastData : public BoundCastData {
	ToUnionBoundCastData(union_tag_t member_idx, string name, LogicalType type, int64_t cost,
	                     BoundCastInfo member_cast_info)
	    : tag(member_idx), name(std::move(name)), type(std::move(type)), cost(cost),
	      member_cast_info(std::move(member_cast_info)) {
	}

	union_tag_t tag;
	string name;
	LogicalType type;
	int64_t cost;
	BoundCastInfo member_cast_info;

public:
	unique_ptr<BoundCastData> Copy() const override {
		return make_unique<ToUnionBoundCastData>(tag, name, type, cost, member_cast_info.Copy());
	}

	static bool SortByCostAscending(const ToUnionBoundCastData &left, const ToUnionBoundCastData &right) {
		return left.cost < right.cost;
	}
};

unique_ptr<BoundCastData> BindToUnionCast(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	D_ASSERT(target.id() == LogicalTypeId::UNION);

	vector<ToUnionBoundCastData> candidates;

	for (idx_t member_idx = 0; member_idx < UnionType::GetMemberCount(target); member_idx++) {
		auto member_type = UnionType::GetMemberType(target, member_idx);
		auto member_name = UnionType::GetMemberName(target, member_idx);
		auto member_cast_cost = input.function_set.ImplicitCastCost(source, member_type);
		if (member_cast_cost != -1) {
			auto member_cast_info = input.GetCastFunction(source, member_type);
			candidates.emplace_back(member_idx, member_name, member_type, member_cast_cost,
			                        std::move(member_cast_info));
		}
	};

	// no possible casts found!
	if (candidates.empty()) {
		auto message = StringUtil::Format(
		    "Type %s can't be cast as %s. %s can't be implicitly cast to any of the union member types: ",
		    source.ToString(), target.ToString(), source.ToString());

		auto member_count = UnionType::GetMemberCount(target);
		for (idx_t member_idx = 0; member_idx < member_count; member_idx++) {
			auto member_type = UnionType::GetMemberType(target, member_idx);
			message += member_type.ToString();
			if (member_idx < member_count - 1) {
				message += ", ";
			}
		}
		throw CastException(message);
	}

	// sort the candidate casts by cost
	std::sort(candidates.begin(), candidates.end(), ToUnionBoundCastData::SortByCostAscending);

	// select the lowest possible cost cast
	auto &selected_cast = candidates[0];
	auto selected_cost = candidates[0].cost;

	// check if the cast is ambiguous (2 or more casts have the same cost)
	if (candidates.size() > 1 && candidates[1].cost == selected_cost) {

		// collect all the ambiguous types
		auto message = StringUtil::Format(
		    "Type %s can't be cast as %s. The cast is ambiguous, multiple possible members in target: ", source,
		    target);
		for (size_t i = 0; i < candidates.size(); i++) {
			if (candidates[i].cost == selected_cost) {
				message += StringUtil::Format("'%s (%s)'", candidates[i].name, candidates[i].type.ToString());
				if (i < candidates.size() - 1) {
					message += ", ";
				}
			}
		}
		message += ". Disambiguate the target type by using the 'union_value(<tag> := <arg>)' function to promote the "
		           "source value to a single member union before casting.";
		throw CastException(message);
	}

	// otherwise, return the selected cast
	return make_unique<ToUnionBoundCastData>(std::move(selected_cast));
}

static bool ToUnionCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	D_ASSERT(result.GetType().id() == LogicalTypeId::UNION);
	auto &cast_data = (ToUnionBoundCastData &)*parameters.cast_data;
	auto &selected_member_vector = UnionVector::GetMember(result, cast_data.tag);

	CastParameters child_parameters(parameters, cast_data.member_cast_info.cast_data.get());
	if (!cast_data.member_cast_info.function(source, selected_member_vector, count, child_parameters)) {
		return false;
	}

	// cast succeeded, create union vector
	UnionVector::SetToMember(result, cast_data.tag, selected_member_vector, count, true);

	result.Verify(count);

	return true;
}

BoundCastInfo DefaultCasts::ImplicitToUnionCast(BindCastInput &input, const LogicalType &source,
                                                const LogicalType &target) {
	return BoundCastInfo(&ToUnionCast, BindToUnionCast(input, source, target));
}

//--------------------------------------------------------------------------------------------------
// UNION -> UNION
//--------------------------------------------------------------------------------------------------
// if the source member tags is a subset of the target member tags, and all the source members can be
// implicitly cast to the corresponding target members, the cast is valid.
//
// VALID: 	UNION(A, B) 	-> 	UNION(A, B, C)
// VALID: 	UNION(A, B) 	-> 	UNION(A, C)		if B can be implicitly cast to C
//
// INVALID: UNION(A, B, C)	->	UNION(A, B)
// INVALID:	UNION(A, B) 	->	UNION(A, C)		if B can't be implicitly cast to C
// INVALID:	UNION(A, B, D) 	->	UNION(A, B, C)

struct UnionToUnionBoundCastData : public BoundCastData {

	// mapping from source member index to target member index
	// these are always the same size as the source member count
	// (since all source members must be present in the target)
	vector<idx_t> tag_map;
	vector<BoundCastInfo> member_casts;

	LogicalType target_type;

	UnionToUnionBoundCastData(vector<idx_t> tag_map, vector<BoundCastInfo> member_casts, LogicalType target_type)
	    : tag_map(std::move(tag_map)), member_casts(std::move(member_casts)), target_type(std::move(target_type)) {
	}

public:
	unique_ptr<BoundCastData> Copy() const override {
		vector<BoundCastInfo> member_casts_copy;
		for (auto &member_cast : member_casts) {
			member_casts_copy.push_back(member_cast.Copy());
		}
		return make_unique<UnionToUnionBoundCastData>(tag_map, std::move(member_casts_copy), target_type);
	}
};

unique_ptr<BoundCastData> BindUnionToUnionCast(BindCastInput &input, const LogicalType &source,
                                               const LogicalType &target) {
	D_ASSERT(source.id() == LogicalTypeId::UNION);
	D_ASSERT(target.id() == LogicalTypeId::UNION);

	auto source_member_count = UnionType::GetMemberCount(source);

	auto tag_map = vector<idx_t>(source_member_count);
	auto member_casts = vector<BoundCastInfo>();

	for (idx_t source_idx = 0; source_idx < source_member_count; source_idx++) {
		auto &source_member_type = UnionType::GetMemberType(source, source_idx);
		auto &source_member_name = UnionType::GetMemberName(source, source_idx);

		bool found = false;
		for (idx_t target_idx = 0; target_idx < UnionType::GetMemberCount(target); target_idx++) {
			auto &target_member_name = UnionType::GetMemberName(target, target_idx);

			// found a matching member
			if (source_member_name == target_member_name) {
				auto &target_member_type = UnionType::GetMemberType(target, target_idx);
				tag_map[source_idx] = target_idx;
				member_casts.push_back(input.GetCastFunction(source_member_type, target_member_type));
				found = true;
				break;
			}
		}
		if (!found) {
			// no matching member tag found in the target set
			auto message =
			    StringUtil::Format("Type %s can't be cast as %s. The member '%s' is not present in target union",
			                       source.ToString(), target.ToString(), source_member_name);
			throw CastException(message);
		}
	}

	return make_unique<UnionToUnionBoundCastData>(tag_map, std::move(member_casts), target);
}

static bool UnionToUnionCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	auto &cast_data = (UnionToUnionBoundCastData &)*parameters.cast_data;

	auto source_member_count = UnionType::GetMemberCount(source.GetType());
	auto target_member_count = UnionType::GetMemberCount(result.GetType());

	auto target_member_is_mapped = vector<bool>(target_member_count);

	// Perform the casts from source to target members
	for (idx_t member_idx = 0; member_idx < source_member_count; member_idx++) {
		auto target_member_idx = cast_data.tag_map[member_idx];

		auto &source_member_vector = UnionVector::GetMember(source, member_idx);
		auto &target_member_vector = UnionVector::GetMember(result, target_member_idx);
		auto &member_cast = cast_data.member_casts[member_idx];

		CastParameters child_parameters(parameters, member_cast.cast_data.get());
		if (!member_cast.function(source_member_vector, target_member_vector, count, child_parameters)) {
			return false;
		}

		target_member_is_mapped[target_member_idx] = true;
	}

	// All member casts succeeded!

	// Set the unmapped target members to constant NULL.
	// If we cast UNION(A, B) -> UNION(A, B, C) we need to invalidate C so that
	// the invariants of the result union hold. (only member columns "selected"
	// by the rowwise corresponding tag in the tag vector should be valid)
	for (idx_t target_member_idx = 0; target_member_idx < target_member_count; target_member_idx++) {
		if (!target_member_is_mapped[target_member_idx]) {
			auto &target_member_vector = UnionVector::GetMember(result, target_member_idx);
			target_member_vector.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(target_member_vector, true);
		}
	}

	// Update the tags in the result vector
	auto &source_tag_vector = UnionVector::GetTags(source);
	auto &result_tag_vector = UnionVector::GetTags(result);

	if (source.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		// Constant vector case optimization
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		if (ConstantVector::IsNull(source)) {
			ConstantVector::SetNull(result, true);
		} else {
			// map the tag
			auto source_tag = ConstantVector::GetData<union_tag_t>(source_tag_vector)[0];
			auto mapped_tag = cast_data.tag_map[source_tag];
			ConstantVector::GetData<union_tag_t>(result_tag_vector)[0] = mapped_tag;
		}
	} else {
		// Otherwise, use the unified vector format to access the source vector.

		// Ensure that all the result members are flat vectors
		// This is not always the case, e.g. when a member is cast using the default TryNullCast function
		// the resulting member vector will be a constant null vector.
		for (idx_t target_member_idx = 0; target_member_idx < target_member_count; target_member_idx++) {
			UnionVector::GetMember(result, target_member_idx).Flatten(count);
		}

		// We assume that a union tag vector validity matches the union vector validity.
		UnifiedVectorFormat source_tag_format;
		source_tag_vector.ToUnifiedFormat(count, source_tag_format);

		for (idx_t row_idx = 0; row_idx < count; row_idx++) {
			auto source_row_idx = source_tag_format.sel->get_index(row_idx);
			if (source_tag_format.validity.RowIsValid(source_row_idx)) {
				// map the tag
				auto source_tag = ((union_tag_t *)source_tag_format.data)[source_row_idx];
				auto target_tag = cast_data.tag_map[source_tag];
				FlatVector::GetData<union_tag_t>(result_tag_vector)[row_idx] = target_tag;
			} else {

				// Issue: The members of the result is not always flatvectors
				// In the case of TryNullCast, the result member is constant.
				FlatVector::SetNull(result, row_idx, true);
			}
		}
	}

	result.Verify(count);

	return true;
}

static bool UnionToVarcharCast(Vector &source, Vector &result, idx_t count, CastParameters &parameters) {
	auto constant = source.GetVectorType() == VectorType::CONSTANT_VECTOR;
	// first cast all union members to varchar
	auto &cast_data = (UnionToUnionBoundCastData &)*parameters.cast_data;
	Vector varchar_union(cast_data.target_type, count);

	UnionToUnionCast(source, varchar_union, count, parameters);

	// now construct the actual varchar vector
	varchar_union.Flatten(count);
	auto &tag_vector = UnionVector::GetTags(source);
	auto tags = FlatVector::GetData<union_tag_t>(tag_vector);

	auto &validity = FlatVector::Validity(varchar_union);
	auto result_data = FlatVector::GetData<string_t>(result);

	for (idx_t i = 0; i < count; i++) {
		if (!validity.RowIsValid(i)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}

		auto &member = UnionVector::GetMember(varchar_union, tags[i]);
		UnifiedVectorFormat member_vdata;
		member.ToUnifiedFormat(count, member_vdata);

		auto mapped_idx = member_vdata.sel->get_index(i);
		auto member_valid = member_vdata.validity.RowIsValid(mapped_idx);
		if (member_valid) {
			auto member_str = ((string_t *)member_vdata.data)[mapped_idx];
			result_data[i] = StringVector::AddString(result, member_str);
		} else {
			result_data[i] = StringVector::AddString(result, "NULL");
		}
	}

	if (constant) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}

	result.Verify(count);
	return true;
}

BoundCastInfo DefaultCasts::UnionCastSwitch(BindCastInput &input, const LogicalType &source,
                                            const LogicalType &target) {
	switch (target.id()) {
	case LogicalTypeId::VARCHAR: {
		// bind a cast in which we convert all members to VARCHAR first
		child_list_t<LogicalType> varchar_members;
		for (idx_t member_idx = 0; member_idx < UnionType::GetMemberCount(source); member_idx++) {
			varchar_members.push_back(make_pair(UnionType::GetMemberName(source, member_idx), LogicalType::VARCHAR));
		}
		auto varchar_type = LogicalType::UNION(std::move(varchar_members));
		return BoundCastInfo(UnionToVarcharCast, BindUnionToUnionCast(input, source, varchar_type));
	} break;
	case LogicalTypeId::UNION:
		return BoundCastInfo(UnionToUnionCast, BindUnionToUnionCast(input, source, target));
	default:
		return TryVectorNullCast;
	}
}

} // namespace duckdb




namespace duckdb {

BoundCastInfo DefaultCasts::UUIDCastSwitch(BindCastInput &input, const LogicalType &source, const LogicalType &target) {
	// now switch on the result type
	switch (target.id()) {
	case LogicalTypeId::VARCHAR:
		// uuid to varchar
		return BoundCastInfo(&VectorCastHelpers::StringCast<hugeint_t, duckdb::CastFromUUID>);
	default:
		return TryVectorNullCast;
	}
}

} // namespace duckdb


namespace duckdb {

// ------- Helper functions for splitting string nested types  -------
static bool IsNull(const char *buf, idx_t start_pos, Vector &child, idx_t row_idx) {
	if (buf[start_pos] == 'N' && buf[start_pos + 1] == 'U' && buf[start_pos + 2] == 'L' && buf[start_pos + 3] == 'L') {
		FlatVector::SetNull(child, row_idx, true);
		return true;
	}
	return false;
}

inline static void SkipWhitespace(const char *buf, idx_t &pos, idx_t len) {
	while (pos < len && StringUtil::CharacterIsSpace(buf[pos])) {
		pos++;
	}
}

static bool SkipToCloseQuotes(idx_t &pos, const char *buf, idx_t &len) {
	char quote = buf[pos];
	pos++;

	while (pos < len) {
		if (buf[pos] == quote) {
			return true;
		}
		pos++;
	}
	return false;
}

static bool SkipToClose(idx_t &idx, const char *buf, idx_t &len, idx_t &lvl, char close_bracket) {
	idx++;

	while (idx < len) {
		if (buf[idx] == '"' || buf[idx] == '\'') {
			if (!SkipToCloseQuotes(idx, buf, len)) {
				return false;
			}
		} else if (buf[idx] == '{') {
			if (!SkipToClose(idx, buf, len, lvl, '}')) {
				return false;
			}
		} else if (buf[idx] == '[') {
			if (!SkipToClose(idx, buf, len, lvl, ']')) {
				return false;
			}
			lvl++;
		} else if (buf[idx] == close_bracket) {
			if (close_bracket == ']') {
				lvl--;
			}
			return true;
		}
		idx++;
	}
	return false;
}

static idx_t StringTrim(const char *buf, idx_t &start_pos, idx_t pos) {
	idx_t trailing_whitespace = 0;
	while (StringUtil::CharacterIsSpace(buf[pos - trailing_whitespace - 1])) {
		trailing_whitespace++;
	}
	if ((buf[start_pos] == '"' && buf[pos - trailing_whitespace - 1] == '"') ||
	    (buf[start_pos] == '\'' && buf[pos - trailing_whitespace - 1] == '\'')) {
		start_pos++;
		trailing_whitespace++;
	}
	return (pos - trailing_whitespace);
}

struct CountPartOperation {
	idx_t count = 0;

	bool HandleKey(const char *buf, idx_t start_pos, idx_t pos) {
		count++;
		return true;
	}
	void HandleValue(const char *buf, idx_t start_pos, idx_t pos) {
		count++;
	}
};

// ------- LIST SPLIT -------
struct SplitStringListOperation {
	SplitStringListOperation(string_t *child_data, idx_t &child_start, Vector &child)
	    : child_data(child_data), child_start(child_start), child(child) {
	}

	string_t *child_data;
	idx_t &child_start;
	Vector &child;

	void HandleValue(const char *buf, idx_t start_pos, idx_t pos) {
		if ((pos - start_pos) == 4 && IsNull(buf, start_pos, child, child_start)) {
			child_start++;
			return;
		}
		child_data[child_start] = StringVector::AddString(child, buf + start_pos, pos - start_pos);
		child_start++;
	}
};

template <class OP>
static bool SplitStringListInternal(const string_t &input, OP &state) {
	const char *buf = input.GetDataUnsafe();
	idx_t len = input.GetSize();
	idx_t lvl = 1;
	idx_t pos = 0;

	SkipWhitespace(buf, pos, len);
	if (pos == len || buf[pos] != '[') {
		return false;
	}

	SkipWhitespace(buf, ++pos, len);
	idx_t start_pos = pos;
	while (pos < len) {
		if (buf[pos] == '[') {
			if (!SkipToClose(pos, buf, len, ++lvl, ']')) {
				return false;
			}
		} else if ((buf[pos] == '"' || buf[pos] == '\'') && pos == start_pos) {
			SkipToCloseQuotes(pos, buf, len);
		} else if (buf[pos] == '{') {
			idx_t struct_lvl = 0;
			SkipToClose(pos, buf, len, struct_lvl, '}');
		} else if (buf[pos] == ',' || buf[pos] == ']') {
			idx_t trailing_whitespace = 0;
			while (StringUtil::CharacterIsSpace(buf[pos - trailing_whitespace - 1])) {
				trailing_whitespace++;
			}
			if (!(buf[pos] == ']' && start_pos == pos)) {
				state.HandleValue(buf, start_pos, pos - trailing_whitespace);
			} // else the list is empty
			if (buf[pos] == ']') {
				lvl--;
				break;
			}
			SkipWhitespace(buf, ++pos, len);
			start_pos = pos;
			continue;
		}
		pos++;
	}
	SkipWhitespace(buf, ++pos, len);
	return (pos == len && lvl == 0);
}

bool VectorStringToList::SplitStringList(const string_t &input, string_t *child_data, idx_t &child_start,
                                         Vector &child) {
	SplitStringListOperation state(child_data, child_start, child);
	return SplitStringListInternal<SplitStringListOperation>(input, state);
}

idx_t VectorStringToList::CountPartsList(const string_t &input) {
	CountPartOperation state;
	SplitStringListInternal<CountPartOperation>(input, state);
	return state.count;
}

// ------- MAP SPLIT -------
struct SplitStringMapOperation {
	SplitStringMapOperation(string_t *child_key_data, string_t *child_val_data, idx_t &child_start, Vector &varchar_key,
	                        Vector &varchar_val)
	    : child_key_data(child_key_data), child_val_data(child_val_data), child_start(child_start),
	      varchar_key(varchar_key), varchar_val(varchar_val) {
	}

	string_t *child_key_data;
	string_t *child_val_data;
	idx_t &child_start;
	Vector &varchar_key;
	Vector &varchar_val;

	bool HandleKey(const char *buf, idx_t start_pos, idx_t pos) {
		if ((pos - start_pos) == 4 && IsNull(buf, start_pos, varchar_key, child_start)) {
			FlatVector::SetNull(varchar_val, child_start, true);
			child_start++;
			return false;
		}
		child_key_data[child_start] = StringVector::AddString(varchar_key, buf + start_pos, pos - start_pos);
		return true;
	}

	void HandleValue(const char *buf, idx_t start_pos, idx_t pos) {
		if ((pos - start_pos) == 4 && IsNull(buf, start_pos, varchar_val, child_start)) {
			child_start++;
			return;
		}
		child_val_data[child_start] = StringVector::AddString(varchar_val, buf + start_pos, pos - start_pos);
		child_start++;
	}
};

template <class OP>
static bool FindKeyOrValueMap(const char *buf, idx_t len, idx_t &pos, OP &state, bool key) {
	auto start_pos = pos;
	idx_t lvl = 0;
	while (pos < len) {
		if (buf[pos] == '"' || buf[pos] == '\'') {
			SkipToCloseQuotes(pos, buf, len);
		} else if (buf[pos] == '{') {
			SkipToClose(pos, buf, len, lvl, '}');
		} else if (buf[pos] == '[') {
			SkipToClose(pos, buf, len, lvl, ']');
		} else if (key && buf[pos] == '=') {
			idx_t end_pos = StringTrim(buf, start_pos, pos);
			return state.HandleKey(buf, start_pos, end_pos); // put string in KEY_child_vector
		} else if (!key && (buf[pos] == ',' || buf[pos] == '}')) {
			idx_t end_pos = StringTrim(buf, start_pos, pos);
			state.HandleValue(buf, start_pos, end_pos); // put string in VALUE_child_vector
			return true;
		}
		pos++;
	}
	return false;
}

template <class OP>
static bool SplitStringMapInternal(const string_t &input, OP &state) {
	const char *buf = input.GetDataUnsafe();
	idx_t len = input.GetSize();
	idx_t pos = 0;

	SkipWhitespace(buf, pos, len);
	if (pos == len || buf[pos] != '{') {
		return false;
	}
	SkipWhitespace(buf, ++pos, len);
	if (pos == len) {
		return false;
	}
	if (buf[pos] == '}') {
		SkipWhitespace(buf, ++pos, len);
		return (pos == len);
	}
	while (pos < len) {
		if (!FindKeyOrValueMap(buf, len, pos, state, true)) {
			return false;
		}
		SkipWhitespace(buf, ++pos, len);
		if (!FindKeyOrValueMap(buf, len, pos, state, false)) {
			return false;
		}
		SkipWhitespace(buf, ++pos, len);
	}
	return true;
}

bool VectorStringToMap::SplitStringMap(const string_t &input, string_t *child_key_data, string_t *child_val_data,
                                       idx_t &child_start, Vector &varchar_key, Vector &varchar_val) {
	SplitStringMapOperation state(child_key_data, child_val_data, child_start, varchar_key, varchar_val);
	return SplitStringMapInternal<SplitStringMapOperation>(input, state);
}

idx_t VectorStringToMap::CountPartsMap(const string_t &input) {
	CountPartOperation state;
	SplitStringMapInternal<CountPartOperation>(input, state);
	return state.count;
}

// ------- STRUCT SPLIT -------
static bool FindKeyStruct(const char *buf, idx_t len, idx_t &pos) {
	while (pos < len) {
		if (buf[pos] == ':') {
			return true;
		}
		pos++;
	}
	return false;
}

static bool FindValueStruct(const char *buf, idx_t len, idx_t &pos, Vector &varchar_child, idx_t &row_idx,
                            ValidityMask *child_mask) {
	auto start_pos = pos;
	idx_t lvl = 0;
	while (pos < len) {
		if (buf[pos] == '"' || buf[pos] == '\'') {
			SkipToCloseQuotes(pos, buf, len);
		} else if (buf[pos] == '{') {
			SkipToClose(pos, buf, len, lvl, '}');
		} else if (buf[pos] == '[') {
			SkipToClose(pos, buf, len, lvl, ']');
		} else if (buf[pos] == ',' || buf[pos] == '}') {
			idx_t end_pos = StringTrim(buf, start_pos, pos);
			if ((end_pos - start_pos) == 4 && IsNull(buf, start_pos, varchar_child, row_idx)) {
				return true;
			}
			FlatVector::GetData<string_t>(varchar_child)[row_idx] =
			    StringVector::AddString(varchar_child, buf + start_pos, end_pos - start_pos);
			child_mask->SetValid(row_idx); // any child not set to valid will remain invalid
			return true;
		}
		pos++;
	}
	return false;
}

bool VectorStringToStruct::SplitStruct(string_t &input, std::vector<std::unique_ptr<Vector>> &varchar_vectors,
                                       idx_t &row_idx, string_map_t<idx_t> &child_names,
                                       std::vector<ValidityMask *> &child_masks) {
	const char *buf = input.GetDataUnsafe();
	idx_t len = input.GetSize();
	idx_t pos = 0;
	idx_t child_idx;

	SkipWhitespace(buf, pos, len);
	if (pos == len || buf[pos] != '{') {
		return false;
	}
	SkipWhitespace(buf, ++pos, len);
	if (buf[pos] == '}') {
		pos++;
	} else {
		while (pos < len) {
			auto key_start = pos;
			if (!FindKeyStruct(buf, len, pos)) {
				return false;
			}
			auto key_end = StringTrim(buf, key_start, pos);
			string_t found_key(buf + key_start, key_end - key_start);

			auto it = child_names.find(found_key);
			if (it == child_names.end()) {
				return false; // false key
			}
			child_idx = it->second;
			SkipWhitespace(buf, ++pos, len);
			if (!FindValueStruct(buf, len, pos, *varchar_vectors[child_idx], row_idx, child_masks[child_idx])) {
				return false;
			}
			SkipWhitespace(buf, ++pos, len);
		}
	}
	SkipWhitespace(buf, pos, len);
	return (pos == len);
}

} // namespace duckdb


namespace duckdb {

//! The target type determines the preferred implicit casts
static int64_t TargetTypeCost(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::INTEGER:
		return 103;
	case LogicalTypeId::BIGINT:
		return 101;
	case LogicalTypeId::DOUBLE:
		return 102;
	case LogicalTypeId::HUGEINT:
		return 120;
	case LogicalTypeId::TIMESTAMP:
		return 120;
	case LogicalTypeId::VARCHAR:
		return 149;
	case LogicalTypeId::DECIMAL:
		return 104;
	case LogicalTypeId::STRUCT:
	case LogicalTypeId::MAP:
	case LogicalTypeId::LIST:
	case LogicalTypeId::UNION:
		return 160;
	default:
		return 110;
	}
}

static int64_t ImplicitCastTinyint(const LogicalType &to) {
	switch (to.id()) {
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
		return TargetTypeCost(to);
	default:
		return -1;
	}
}

static int64_t ImplicitCastSmallint(const LogicalType &to) {
	switch (to.id()) {
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
		return TargetTypeCost(to);
	default:
		return -1;
	}
}

static int64_t ImplicitCastInteger(const LogicalType &to) {
	switch (to.id()) {
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
		return TargetTypeCost(to);
	default:
		return -1;
	}
}

static int64_t ImplicitCastBigint(const LogicalType &to) {
	switch (to.id()) {
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::DECIMAL:
		return TargetTypeCost(to);
	default:
		return -1;
	}
}

static int64_t ImplicitCastUTinyint(const LogicalType &to) {
	switch (to.id()) {
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
		return TargetTypeCost(to);
	default:
		return -1;
	}
}

static int64_t ImplicitCastUSmallint(const LogicalType &to) {
	switch (to.id()) {
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
		return TargetTypeCost(to);
	default:
		return -1;
	}
}

static int64_t ImplicitCastUInteger(const LogicalType &to) {
	switch (to.id()) {

	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
		return TargetTypeCost(to);
	default:
		return -1;
	}
}

static int64_t ImplicitCastUBigint(const LogicalType &to) {
	switch (to.id()) {
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::DECIMAL:
		return TargetTypeCost(to);
	default:
		return -1;
	}
}

static int64_t ImplicitCastFloat(const LogicalType &to) {
	switch (to.id()) {
	case LogicalTypeId::DOUBLE:
		return TargetTypeCost(to);
	default:
		return -1;
	}
}

static int64_t ImplicitCastDouble(const LogicalType &to) {
	switch (to.id()) {
	default:
		return -1;
	}
}

static int64_t ImplicitCastDecimal(const LogicalType &to) {
	switch (to.id()) {
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
		return TargetTypeCost(to);
	default:
		return -1;
	}
}

static int64_t ImplicitCastHugeint(const LogicalType &to) {
	switch (to.id()) {
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
		return TargetTypeCost(to);
	default:
		return -1;
	}
}

static int64_t ImplicitCastDate(const LogicalType &to) {
	switch (to.id()) {
	case LogicalTypeId::TIMESTAMP:
		return TargetTypeCost(to);
	default:
		return -1;
	}
}

int64_t CastRules::ImplicitCast(const LogicalType &from, const LogicalType &to) {
	if (from.id() == LogicalTypeId::SQLNULL) {
		// NULL expression can be cast to anything
		return TargetTypeCost(to);
	}
	if (from.id() == LogicalTypeId::UNKNOWN) {
		// parameter expression can be cast to anything for no cost
		return 0;
	}
	if (to.id() == LogicalTypeId::ANY) {
		// anything can be cast to ANY type for (almost no) cost
		return 1;
	}
	if (from.GetAlias() != to.GetAlias()) {
		// if aliases are different, an implicit cast is not possible
		return -1;
	}
	if (from.id() == to.id()) {
		// arguments match: do nothing
		return 0;
	}
	if (from.id() == LogicalTypeId::BLOB && to.id() == LogicalTypeId::VARCHAR) {
		// Implicit cast not allowed from BLOB to VARCHAR
		return -1;
	}
	if (to.id() == LogicalTypeId::VARCHAR) {
		// everything can be cast to VARCHAR, but this cast has a high cost
		return TargetTypeCost(to);
	}
	if (from.id() == LogicalTypeId::LIST && to.id() == LogicalTypeId::LIST) {
		// Lists can be cast if their child types can be cast
		return ImplicitCast(ListType::GetChildType(from), ListType::GetChildType(to));
	}

	if (from.id() == LogicalTypeId::UNION && to.id() == LogicalTypeId::UNION) {
		// Unions can be cast if the source tags are a subset of the target tags
		// in which case the most expensive cost is used
		int cost = -1;
		for (idx_t from_member_idx = 0; from_member_idx < UnionType::GetMemberCount(from); from_member_idx++) {
			auto &from_member_name = UnionType::GetMemberName(from, from_member_idx);

			bool found = false;
			for (idx_t to_member_idx = 0; to_member_idx < UnionType::GetMemberCount(to); to_member_idx++) {
				auto &to_member_name = UnionType::GetMemberName(to, to_member_idx);

				if (from_member_name == to_member_name) {
					auto &from_member_type = UnionType::GetMemberType(from, from_member_idx);
					auto &to_member_type = UnionType::GetMemberType(to, to_member_idx);

					int child_cost = ImplicitCast(from_member_type, to_member_type);
					if (child_cost > cost) {
						cost = child_cost;
					}
					found = true;
					break;
				}
			}
			if (!found) {
				return -1;
			}
		}
		return cost;
	}

	if (to.id() == LogicalTypeId::UNION) {
		// check that the union type is fully resolved.
		if (to.AuxInfo() == nullptr) {
			return -1;
		}
		// every type can be implicitly be cast to a union if the source type is a member of the union
		for (idx_t i = 0; i < UnionType::GetMemberCount(to); i++) {
			auto member = UnionType::GetMemberType(to, i);
			if (from == member) {
				return 0;
			}
		}
	}

	if ((from.id() == LogicalTypeId::TIMESTAMP_SEC || from.id() == LogicalTypeId::TIMESTAMP_MS ||
	     from.id() == LogicalTypeId::TIMESTAMP_NS) &&
	    to.id() == LogicalTypeId::TIMESTAMP) {
		//! Any timestamp type can be converted to the default (us) type at low cost
		return 101;
	}
	if ((to.id() == LogicalTypeId::TIMESTAMP_SEC || to.id() == LogicalTypeId::TIMESTAMP_MS ||
	     to.id() == LogicalTypeId::TIMESTAMP_NS) &&
	    from.id() == LogicalTypeId::TIMESTAMP) {
		//! Any timestamp type can be converted to the default (us) type at low cost
		return 100;
	}
	switch (from.id()) {
	case LogicalTypeId::TINYINT:
		return ImplicitCastTinyint(to);
	case LogicalTypeId::SMALLINT:
		return ImplicitCastSmallint(to);
	case LogicalTypeId::INTEGER:
		return ImplicitCastInteger(to);
	case LogicalTypeId::BIGINT:
		return ImplicitCastBigint(to);
	case LogicalTypeId::UTINYINT:
		return ImplicitCastUTinyint(to);
	case LogicalTypeId::USMALLINT:
		return ImplicitCastUSmallint(to);
	case LogicalTypeId::UINTEGER:
		return ImplicitCastUInteger(to);
	case LogicalTypeId::UBIGINT:
		return ImplicitCastUBigint(to);
	case LogicalTypeId::HUGEINT:
		return ImplicitCastHugeint(to);
	case LogicalTypeId::FLOAT:
		return ImplicitCastFloat(to);
	case LogicalTypeId::DOUBLE:
		return ImplicitCastDouble(to);
	case LogicalTypeId::DATE:
		return ImplicitCastDate(to);
	case LogicalTypeId::DECIMAL:
		return ImplicitCastDecimal(to);
	default:
		return -1;
	}
}

} // namespace duckdb





namespace duckdb {

typedef CompressionFunction (*get_compression_function_t)(PhysicalType type);
typedef bool (*compression_supports_type_t)(PhysicalType type);

struct DefaultCompressionMethod {
	CompressionType type;
	get_compression_function_t get_function;
	compression_supports_type_t supports_type;
};

static DefaultCompressionMethod internal_compression_methods[] = {
    {CompressionType::COMPRESSION_CONSTANT, ConstantFun::GetFunction, ConstantFun::TypeIsSupported},
    {CompressionType::COMPRESSION_UNCOMPRESSED, UncompressedFun::GetFunction, UncompressedFun::TypeIsSupported},
    {CompressionType::COMPRESSION_RLE, RLEFun::GetFunction, RLEFun::TypeIsSupported},
    {CompressionType::COMPRESSION_BITPACKING, BitpackingFun::GetFunction, BitpackingFun::TypeIsSupported},
    {CompressionType::COMPRESSION_DICTIONARY, DictionaryCompressionFun::GetFunction,
     DictionaryCompressionFun::TypeIsSupported},
    {CompressionType::COMPRESSION_CHIMP, ChimpCompressionFun::GetFunction, ChimpCompressionFun::TypeIsSupported},
    {CompressionType::COMPRESSION_PATAS, PatasCompressionFun::GetFunction, PatasCompressionFun::TypeIsSupported},
    {CompressionType::COMPRESSION_FSST, FSSTFun::GetFunction, FSSTFun::TypeIsSupported},
    {CompressionType::COMPRESSION_AUTO, nullptr, nullptr}};

static CompressionFunction *FindCompressionFunction(CompressionFunctionSet &set, CompressionType type,
                                                    PhysicalType data_type) {
	auto &functions = set.functions;
	auto comp_entry = functions.find(type);
	if (comp_entry != functions.end()) {
		auto &type_functions = comp_entry->second;
		auto type_entry = type_functions.find(data_type);
		if (type_entry != type_functions.end()) {
			return &type_entry->second;
		}
	}
	return nullptr;
}

static CompressionFunction *LoadCompressionFunction(CompressionFunctionSet &set, CompressionType type,
                                                    PhysicalType data_type) {
	for (idx_t index = 0; internal_compression_methods[index].get_function; index++) {
		const auto &method = internal_compression_methods[index];
		if (method.type == type) {
			// found the correct compression type
			if (!method.supports_type(data_type)) {
				// but it does not support this data type: bail out
				return nullptr;
			}
			// the type is supported: create the function and insert it into the set
			auto function = method.get_function(data_type);
			set.functions[type].insert(make_pair(data_type, function));
			return FindCompressionFunction(set, type, data_type);
		}
	}
	throw InternalException("Unsupported compression function type");
}

static void TryLoadCompression(DBConfig &config, vector<CompressionFunction *> &result, CompressionType type,
                               PhysicalType data_type) {
	auto function = config.GetCompressionFunction(type, data_type);
	if (!function) {
		return;
	}
	result.push_back(function);
}

vector<CompressionFunction *> DBConfig::GetCompressionFunctions(PhysicalType data_type) {
	vector<CompressionFunction *> result;
	TryLoadCompression(*this, result, CompressionType::COMPRESSION_UNCOMPRESSED, data_type);
	TryLoadCompression(*this, result, CompressionType::COMPRESSION_RLE, data_type);
	TryLoadCompression(*this, result, CompressionType::COMPRESSION_BITPACKING, data_type);
	TryLoadCompression(*this, result, CompressionType::COMPRESSION_DICTIONARY, data_type);
	TryLoadCompression(*this, result, CompressionType::COMPRESSION_CHIMP, data_type);
	TryLoadCompression(*this, result, CompressionType::COMPRESSION_PATAS, data_type);
	TryLoadCompression(*this, result, CompressionType::COMPRESSION_FSST, data_type);
	return result;
}

CompressionFunction *DBConfig::GetCompressionFunction(CompressionType type, PhysicalType data_type) {
	lock_guard<mutex> l(compression_functions->lock);
	// check if the function is already loaded
	auto function = FindCompressionFunction(*compression_functions, type, data_type);
	if (function) {
		return function;
	}
	// else load the function
	return LoadCompressionFunction(*compression_functions, type, data_type);
}

} // namespace duckdb










namespace duckdb {

FunctionData::~FunctionData() {
}

bool FunctionData::Equals(const FunctionData *left, const FunctionData *right) {
	if (left == right) {
		return true;
	}
	if (!left || !right) {
		return false;
	}
	return left->Equals(*right);
}

TableFunctionData::~TableFunctionData() {
}

unique_ptr<FunctionData> TableFunctionData::Copy() const {
	throw InternalException("Copy not supported for TableFunctionData");
}

bool TableFunctionData::Equals(const FunctionData &other) const {
	return false;
}

Function::Function(string name_p) : name(std::move(name_p)) {
}
Function::~Function() {
}

SimpleFunction::SimpleFunction(string name_p, vector<LogicalType> arguments_p, LogicalType varargs_p)
    : Function(std::move(name_p)), arguments(std::move(arguments_p)), varargs(std::move(varargs_p)) {
}

SimpleFunction::~SimpleFunction() {
}

string SimpleFunction::ToString() {
	return Function::CallToString(name, arguments);
}

bool SimpleFunction::HasVarArgs() const {
	return varargs.id() != LogicalTypeId::INVALID;
}

SimpleNamedParameterFunction::SimpleNamedParameterFunction(string name_p, vector<LogicalType> arguments_p,
                                                           LogicalType varargs_p)
    : SimpleFunction(std::move(name_p), std::move(arguments_p), std::move(varargs_p)) {
}

SimpleNamedParameterFunction::~SimpleNamedParameterFunction() {
}

string SimpleNamedParameterFunction::ToString() {
	return Function::CallToString(name, arguments, named_parameters);
}

bool SimpleNamedParameterFunction::HasNamedParameters() {
	return !named_parameters.empty();
}

BaseScalarFunction::BaseScalarFunction(string name_p, vector<LogicalType> arguments_p, LogicalType return_type_p,
                                       FunctionSideEffects side_effects, LogicalType varargs_p,
                                       FunctionNullHandling null_handling)
    : SimpleFunction(std::move(name_p), std::move(arguments_p), std::move(varargs_p)),
      return_type(std::move(return_type_p)), side_effects(side_effects), null_handling(null_handling) {
}

BaseScalarFunction::~BaseScalarFunction() {
}

string BaseScalarFunction::ToString() {
	return Function::CallToString(name, arguments, return_type);
}

// add your initializer for new functions here
void BuiltinFunctions::Initialize() {
	RegisterTableScanFunctions();
	RegisterSQLiteFunctions();
	RegisterReadFunctions();
	RegisterTableFunctions();
	RegisterArrowFunctions();

	RegisterAlgebraicAggregates();
	RegisterDistributiveAggregates();
	RegisterNestedAggregates();
	RegisterHolisticAggregates();
	RegisterRegressiveAggregates();

	RegisterDateFunctions();
	RegisterEnumFunctions();
	RegisterGenericFunctions();
	RegisterMathFunctions();
	RegisterOperators();
	RegisterSequenceFunctions();
	RegisterStringFunctions();
	RegisterNestedFunctions();
	RegisterTrigonometricsFunctions();

	RegisterPragmaFunctions();

	// initialize collations
	AddCollation("nocase", LowerFun::GetFunction(), true);
	AddCollation("noaccent", StripAccentsFun::GetFunction());
	AddCollation("nfc", NFCNormalizeFun::GetFunction());
}

hash_t BaseScalarFunction::Hash() const {
	hash_t hash = return_type.Hash();
	for (auto &arg : arguments) {
		duckdb::CombineHash(hash, arg.Hash());
	}
	return hash;
}

string Function::CallToString(const string &name, const vector<LogicalType> &arguments) {
	string result = name + "(";
	result += StringUtil::Join(arguments, arguments.size(), ", ",
	                           [](const LogicalType &argument) { return argument.ToString(); });
	return result + ")";
}

string Function::CallToString(const string &name, const vector<LogicalType> &arguments,
                              const LogicalType &return_type) {
	string result = CallToString(name, arguments);
	result += " -> " + return_type.ToString();
	return result;
}

string Function::CallToString(const string &name, const vector<LogicalType> &arguments,
                              const named_parameter_type_map_t &named_parameters) {
	vector<string> input_arguments;
	input_arguments.reserve(arguments.size() + named_parameters.size());
	for (auto &arg : arguments) {
		input_arguments.push_back(arg.ToString());
	}
	for (auto &kv : named_parameters) {
		input_arguments.push_back(StringUtil::Format("%s : %s", kv.first, kv.second.ToString()));
	}
	return StringUtil::Format("%s(%s)", name, StringUtil::Join(input_arguments, ", "));
}

void Function::EraseArgument(SimpleFunction &bound_function, vector<unique_ptr<Expression>> &arguments,
                             idx_t argument_index) {
	if (bound_function.original_arguments.empty()) {
		bound_function.original_arguments = bound_function.arguments;
	}
	D_ASSERT(arguments.size() == bound_function.arguments.size());
	D_ASSERT(argument_index < arguments.size());
	arguments.erase(arguments.begin() + argument_index);
	bound_function.arguments.erase(bound_function.arguments.begin() + argument_index);
}

} // namespace duckdb














namespace duckdb {

FunctionBinder::FunctionBinder(ClientContext &context) : context(context) {
}

int64_t FunctionBinder::BindVarArgsFunctionCost(const SimpleFunction &func, const vector<LogicalType> &arguments) {
	if (arguments.size() < func.arguments.size()) {
		// not enough arguments to fulfill the non-vararg part of the function
		return -1;
	}
	int64_t cost = 0;
	for (idx_t i = 0; i < arguments.size(); i++) {
		LogicalType arg_type = i < func.arguments.size() ? func.arguments[i] : func.varargs;
		if (arguments[i] == arg_type) {
			// arguments match: do nothing
			continue;
		}
		int64_t cast_cost = CastFunctionSet::Get(context).ImplicitCastCost(arguments[i], arg_type);
		if (cast_cost >= 0) {
			// we can implicitly cast, add the cost to the total cost
			cost += cast_cost;
		} else {
			// we can't implicitly cast: throw an error
			return -1;
		}
	}
	return cost;
}

int64_t FunctionBinder::BindFunctionCost(const SimpleFunction &func, const vector<LogicalType> &arguments) {
	if (func.HasVarArgs()) {
		// special case varargs function
		return BindVarArgsFunctionCost(func, arguments);
	}
	if (func.arguments.size() != arguments.size()) {
		// invalid argument count: check the next function
		return -1;
	}
	int64_t cost = 0;
	for (idx_t i = 0; i < arguments.size(); i++) {
		int64_t cast_cost = CastFunctionSet::Get(context).ImplicitCastCost(arguments[i], func.arguments[i]);
		if (cast_cost >= 0) {
			// we can implicitly cast, add the cost to the total cost
			cost += cast_cost;
		} else {
			// we can't implicitly cast: throw an error
			return -1;
		}
	}
	return cost;
}

template <class T>
vector<idx_t> FunctionBinder::BindFunctionsFromArguments(const string &name, FunctionSet<T> &functions,
                                                         const vector<LogicalType> &arguments, string &error) {
	idx_t best_function = DConstants::INVALID_INDEX;
	int64_t lowest_cost = NumericLimits<int64_t>::Maximum();
	vector<idx_t> candidate_functions;
	for (idx_t f_idx = 0; f_idx < functions.functions.size(); f_idx++) {
		auto &func = functions.functions[f_idx];
		// check the arguments of the function
		int64_t cost = BindFunctionCost(func, arguments);
		if (cost < 0) {
			// auto casting was not possible
			continue;
		}
		if (cost == lowest_cost) {
			candidate_functions.push_back(f_idx);
			continue;
		}
		if (cost > lowest_cost) {
			continue;
		}
		candidate_functions.clear();
		lowest_cost = cost;
		best_function = f_idx;
	}
	if (best_function == DConstants::INVALID_INDEX) {
		// no matching function was found, throw an error
		string call_str = Function::CallToString(name, arguments);
		string candidate_str = "";
		for (auto &f : functions.functions) {
			candidate_str += "\t" + f.ToString() + "\n";
		}
		error = StringUtil::Format("No function matches the given name and argument types '%s'. You might need to add "
		                           "explicit type casts.\n\tCandidate functions:\n%s",
		                           call_str, candidate_str);
		return candidate_functions;
	}
	candidate_functions.push_back(best_function);
	return candidate_functions;
}

template <class T>
idx_t FunctionBinder::MultipleCandidateException(const string &name, FunctionSet<T> &functions,
                                                 vector<idx_t> &candidate_functions,
                                                 const vector<LogicalType> &arguments, string &error) {
	D_ASSERT(functions.functions.size() > 1);
	// there are multiple possible function definitions
	// throw an exception explaining which overloads are there
	string call_str = Function::CallToString(name, arguments);
	string candidate_str = "";
	for (auto &conf : candidate_functions) {
		T f = functions.GetFunctionByOffset(conf);
		candidate_str += "\t" + f.ToString() + "\n";
	}
	error = StringUtil::Format("Could not choose a best candidate function for the function call \"%s\". In order to "
	                           "select one, please add explicit type casts.\n\tCandidate functions:\n%s",
	                           call_str, candidate_str);
	return DConstants::INVALID_INDEX;
}

template <class T>
idx_t FunctionBinder::BindFunctionFromArguments(const string &name, FunctionSet<T> &functions,
                                                const vector<LogicalType> &arguments, string &error) {
	auto candidate_functions = BindFunctionsFromArguments<T>(name, functions, arguments, error);
	if (candidate_functions.empty()) {
		// no candidates
		return DConstants::INVALID_INDEX;
	}
	if (candidate_functions.size() > 1) {
		// multiple candidates, check if there are any unknown arguments
		bool has_parameters = false;
		for (auto &arg_type : arguments) {
			if (arg_type.id() == LogicalTypeId::UNKNOWN) {
				//! there are! we could not resolve parameters in this case
				throw ParameterNotResolvedException();
			}
		}
		if (!has_parameters) {
			return MultipleCandidateException(name, functions, candidate_functions, arguments, error);
		}
	}
	return candidate_functions[0];
}

idx_t FunctionBinder::BindFunction(const string &name, ScalarFunctionSet &functions,
                                   const vector<LogicalType> &arguments, string &error) {
	return BindFunctionFromArguments(name, functions, arguments, error);
}

idx_t FunctionBinder::BindFunction(const string &name, AggregateFunctionSet &functions,
                                   const vector<LogicalType> &arguments, string &error) {
	return BindFunctionFromArguments(name, functions, arguments, error);
}

idx_t FunctionBinder::BindFunction(const string &name, TableFunctionSet &functions,
                                   const vector<LogicalType> &arguments, string &error) {
	return BindFunctionFromArguments(name, functions, arguments, error);
}

idx_t FunctionBinder::BindFunction(const string &name, PragmaFunctionSet &functions, PragmaInfo &info, string &error) {
	vector<LogicalType> types;
	for (auto &value : info.parameters) {
		types.push_back(value.type());
	}
	idx_t entry = BindFunctionFromArguments(name, functions, types, error);
	if (entry == DConstants::INVALID_INDEX) {
		throw BinderException(error);
	}
	auto candidate_function = functions.GetFunctionByOffset(entry);
	// cast the input parameters
	for (idx_t i = 0; i < info.parameters.size(); i++) {
		auto target_type =
		    i < candidate_function.arguments.size() ? candidate_function.arguments[i] : candidate_function.varargs;
		info.parameters[i] = info.parameters[i].CastAs(context, target_type);
	}
	return entry;
}

vector<LogicalType> FunctionBinder::GetLogicalTypesFromExpressions(vector<unique_ptr<Expression>> &arguments) {
	vector<LogicalType> types;
	types.reserve(arguments.size());
	for (auto &argument : arguments) {
		types.push_back(argument->return_type);
	}
	return types;
}

idx_t FunctionBinder::BindFunction(const string &name, ScalarFunctionSet &functions,
                                   vector<unique_ptr<Expression>> &arguments, string &error) {
	auto types = GetLogicalTypesFromExpressions(arguments);
	return BindFunction(name, functions, types, error);
}

idx_t FunctionBinder::BindFunction(const string &name, AggregateFunctionSet &functions,
                                   vector<unique_ptr<Expression>> &arguments, string &error) {
	auto types = GetLogicalTypesFromExpressions(arguments);
	return BindFunction(name, functions, types, error);
}

idx_t FunctionBinder::BindFunction(const string &name, TableFunctionSet &functions,
                                   vector<unique_ptr<Expression>> &arguments, string &error) {
	auto types = GetLogicalTypesFromExpressions(arguments);
	return BindFunction(name, functions, types, error);
}

enum class LogicalTypeComparisonResult { IDENTICAL_TYPE, TARGET_IS_ANY, DIFFERENT_TYPES };

LogicalTypeComparisonResult RequiresCast(const LogicalType &source_type, const LogicalType &target_type) {
	if (target_type.id() == LogicalTypeId::ANY) {
		return LogicalTypeComparisonResult::TARGET_IS_ANY;
	}
	if (source_type == target_type) {
		return LogicalTypeComparisonResult::IDENTICAL_TYPE;
	}
	if (source_type.id() == LogicalTypeId::LIST && target_type.id() == LogicalTypeId::LIST) {
		return RequiresCast(ListType::GetChildType(source_type), ListType::GetChildType(target_type));
	}
	return LogicalTypeComparisonResult::DIFFERENT_TYPES;
}

void FunctionBinder::CastToFunctionArguments(SimpleFunction &function, vector<unique_ptr<Expression>> &children) {
	for (idx_t i = 0; i < children.size(); i++) {
		auto target_type = i < function.arguments.size() ? function.arguments[i] : function.varargs;
		target_type.Verify();
		// don't cast lambda children, they get removed anyways
		if (children[i]->return_type.id() == LogicalTypeId::LAMBDA) {
			continue;
		}
		// check if the type of child matches the type of function argument
		// if not we need to add a cast
		auto cast_result = RequiresCast(children[i]->return_type, target_type);
		// except for one special case: if the function accepts ANY argument
		// in that case we don't add a cast
		if (cast_result == LogicalTypeComparisonResult::DIFFERENT_TYPES) {
			children[i] = BoundCastExpression::AddCastToType(context, std::move(children[i]), target_type);
		}
	}
}

unique_ptr<Expression> FunctionBinder::BindScalarFunction(const string &schema, const string &name,
                                                          vector<unique_ptr<Expression>> children, string &error,
                                                          bool is_operator, Binder *binder) {
	// bind the function
	auto function =
	    Catalog::GetSystemCatalog(context).GetEntry(context, CatalogType::SCALAR_FUNCTION_ENTRY, schema, name);
	D_ASSERT(function && function->type == CatalogType::SCALAR_FUNCTION_ENTRY);
	return BindScalarFunction((ScalarFunctionCatalogEntry &)*function, std::move(children), error, is_operator, binder);
}

unique_ptr<Expression> FunctionBinder::BindScalarFunction(ScalarFunctionCatalogEntry &func,
                                                          vector<unique_ptr<Expression>> children, string &error,
                                                          bool is_operator, Binder *binder) {
	// bind the function
	idx_t best_function = BindFunction(func.name, func.functions, children, error);
	if (best_function == DConstants::INVALID_INDEX) {
		return nullptr;
	}

	// found a matching function!
	auto bound_function = func.functions.GetFunctionByOffset(best_function);

	if (bound_function.null_handling == FunctionNullHandling::DEFAULT_NULL_HANDLING) {
		for (auto &child : children) {
			if (child->return_type == LogicalTypeId::SQLNULL) {
				return make_unique<BoundConstantExpression>(Value(LogicalType::SQLNULL));
			}
		}
	}
	return BindScalarFunction(bound_function, std::move(children), is_operator);
}

unique_ptr<BoundFunctionExpression> FunctionBinder::BindScalarFunction(ScalarFunction bound_function,
                                                                       vector<unique_ptr<Expression>> children,
                                                                       bool is_operator) {
	unique_ptr<FunctionData> bind_info;
	if (bound_function.bind) {
		bind_info = bound_function.bind(context, bound_function, children);
	}
	// check if we need to add casts to the children
	CastToFunctionArguments(bound_function, children);

	// now create the function
	auto return_type = bound_function.return_type;
	return make_unique<BoundFunctionExpression>(std::move(return_type), std::move(bound_function), std::move(children),
	                                            std::move(bind_info), is_operator);
}

unique_ptr<BoundAggregateExpression> FunctionBinder::BindAggregateFunction(AggregateFunction bound_function,
                                                                           vector<unique_ptr<Expression>> children,
                                                                           unique_ptr<Expression> filter,
                                                                           AggregateType aggr_type,
                                                                           unique_ptr<BoundOrderModifier> order_bys) {
	unique_ptr<FunctionData> bind_info;
	if (bound_function.bind) {
		bind_info = bound_function.bind(context, bound_function, children);
		// we may have lost some arguments in the bind
		children.resize(MinValue(bound_function.arguments.size(), children.size()));
	}

	// check if we need to add casts to the children
	CastToFunctionArguments(bound_function, children);

	// Special case: for ORDER BY aggregates, we wrap the aggregate function in a SortedAggregateFunction
	// The children are the sort clauses and the binding contains the ordering data.
	if (order_bys && !order_bys->orders.empty()) {
		bind_info = BindSortedAggregate(bound_function, children, std::move(bind_info), std::move(order_bys));
	}

	return make_unique<BoundAggregateExpression>(std::move(bound_function), std::move(children), std::move(filter),
	                                             std::move(bind_info), aggr_type);
}

} // namespace duckdb



namespace duckdb {

ScalarFunctionSet::ScalarFunctionSet(string name) : FunctionSet(std::move(name)) {
}

ScalarFunction ScalarFunctionSet::GetFunctionByArguments(ClientContext &context, const vector<LogicalType> &arguments) {
	string error;
	FunctionBinder binder(context);
	idx_t index = binder.BindFunction(name, *this, arguments, error);
	if (index == DConstants::INVALID_INDEX) {
		throw InternalException("Failed to find function %s(%s)\n%s", name, StringUtil::ToString(arguments, ","),
		                        error);
	}
	return GetFunctionByOffset(index);
}

AggregateFunctionSet::AggregateFunctionSet(string name) : FunctionSet(std::move(name)) {
}

AggregateFunction AggregateFunctionSet::GetFunctionByArguments(ClientContext &context,
                                                               const vector<LogicalType> &arguments) {
	string error;
	FunctionBinder binder(context);
	idx_t index = binder.BindFunction(name, *this, arguments, error);
	if (index == DConstants::INVALID_INDEX) {
		// check if the arguments are a prefix of any of the arguments
		// this is used for functions such as quantile or string_agg that delete part of their arguments during bind
		// FIXME: we should come up with a better solution here
		for (auto &func : functions) {
			if (arguments.size() >= func.arguments.size()) {
				continue;
			}
			bool is_prefix = true;
			for (idx_t k = 0; k < arguments.size(); k++) {
				if (arguments[k] != func.arguments[k]) {
					is_prefix = false;
					break;
				}
			}
			if (is_prefix) {
				return func;
			}
		}
		throw InternalException("Failed to find function %s(%s)\n%s", name, StringUtil::ToString(arguments, ","),
		                        error);
	}
	return GetFunctionByOffset(index);
}

TableFunctionSet::TableFunctionSet(string name) : FunctionSet(std::move(name)) {
}

TableFunction TableFunctionSet::GetFunctionByArguments(ClientContext &context, const vector<LogicalType> &arguments) {
	string error;
	FunctionBinder binder(context);
	idx_t index = binder.BindFunction(name, *this, arguments, error);
	if (index == DConstants::INVALID_INDEX) {
		throw InternalException("Failed to find function %s(%s)\n%s", name, StringUtil::ToString(arguments, ","),
		                        error);
	}
	return GetFunctionByOffset(index);
}

} // namespace duckdb











namespace duckdb {

// MacroFunction::MacroFunction(unique_ptr<ParsedExpression> expression) : expression(std::move(expression)) {}

MacroFunction::MacroFunction(MacroType type) : type(type) {
}

string MacroFunction::ValidateArguments(MacroFunction &macro_def, const string &name, FunctionExpression &function_expr,
                                        vector<unique_ptr<ParsedExpression>> &positionals,
                                        unordered_map<string, unique_ptr<ParsedExpression>> &defaults) {

	// separate positional and default arguments
	for (auto &arg : function_expr.children) {
		if (!arg->alias.empty()) {
			// default argument
			if (!macro_def.default_parameters.count(arg->alias)) {
				return StringUtil::Format("Macro %s does not have default parameter %s!", name, arg->alias);
			} else if (defaults.count(arg->alias)) {
				return StringUtil::Format("Duplicate default parameters %s!", arg->alias);
			}
			defaults[arg->alias] = std::move(arg);
		} else if (!defaults.empty()) {
			return "Positional parameters cannot come after parameters with a default value!";
		} else {
			// positional argument
			positionals.push_back(std::move(arg));
		}
	}

	// validate if the right number of arguments was supplied
	string error;
	auto &parameters = macro_def.parameters;
	if (parameters.size() != positionals.size()) {
		error = StringUtil::Format(
		    "Macro function '%s(%s)' requires ", name,
		    StringUtil::Join(parameters, parameters.size(), ", ", [](const unique_ptr<ParsedExpression> &p) {
			    return ((ColumnRefExpression &)*p).column_names[0];
		    }));
		error += parameters.size() == 1 ? "a single positional argument"
		                                : StringUtil::Format("%i positional arguments", parameters.size());
		error += ", but ";
		error += positionals.size() == 1 ? "a single positional argument was"
		                                 : StringUtil::Format("%i positional arguments were", positionals.size());
		error += " provided.";
		return error;
	}

	// Add the default values for parameters that have defaults, that were not explicitly assigned to
	for (auto it = macro_def.default_parameters.begin(); it != macro_def.default_parameters.end(); it++) {
		auto &parameter_name = it->first;
		auto &parameter_default = it->second;
		if (!defaults.count(parameter_name)) {
			// This parameter was not set yet, set it with the default value
			defaults[parameter_name] = parameter_default->Copy();
		}
	}

	return error;
}

void MacroFunction::CopyProperties(MacroFunction &other) {
	other.type = type;
	for (auto &param : parameters) {
		other.parameters.push_back(param->Copy());
	}
	for (auto &kv : default_parameters) {
		other.default_parameters[kv.first] = kv.second->Copy();
	}
}

string MacroFunction::ToSQL(const string &schema, const string &name) {
	vector<string> param_strings;
	for (auto &param : parameters) {
		param_strings.push_back(param->ToString());
	}
	for (auto &named_param : default_parameters) {
		param_strings.push_back(StringUtil::Format("%s := %s", named_param.first, named_param.second->ToString()));
	}

	return StringUtil::Format("CREATE MACRO %s.%s(%s) AS ", schema, name, StringUtil::Join(param_strings, ", "));
}

} // namespace duckdb













#include <cctype>

namespace duckdb {

static void PragmaEnableProfilingStatement(ClientContext &context, const FunctionParameters &parameters) {
	auto &config = ClientConfig::GetConfig(context);
	config.enable_profiler = true;
	config.emit_profiler_output = true;
}

void RegisterEnableProfiling(BuiltinFunctions &set) {
	PragmaFunctionSet functions("");
	functions.AddFunction(PragmaFunction::PragmaStatement(string(), PragmaEnableProfilingStatement));

	set.AddFunction("enable_profile", functions);
	set.AddFunction("enable_profiling", functions);
}

static void PragmaDisableProfiling(ClientContext &context, const FunctionParameters &parameters) {
	auto &config = ClientConfig::GetConfig(context);
	config.enable_profiler = false;
}

static void PragmaEnableProgressBar(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).enable_progress_bar = true;
}

static void PragmaDisableProgressBar(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).enable_progress_bar = false;
}

static void PragmaEnablePrintProgressBar(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).print_progress_bar = true;
}

static void PragmaDisablePrintProgressBar(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).print_progress_bar = false;
}

static void PragmaEnableVerification(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).query_verification_enabled = true;
	ClientConfig::GetConfig(context).verify_serializer = true;
}

static void PragmaDisableVerification(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).query_verification_enabled = false;
	ClientConfig::GetConfig(context).verify_serializer = false;
}

static void PragmaVerifySerializer(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).verify_serializer = true;
}

static void PragmaDisableVerifySerializer(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).verify_serializer = false;
}

static void PragmaEnableExternalVerification(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).verify_external = true;
}

static void PragmaDisableExternalVerification(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).verify_external = false;
}

static void PragmaEnableForceParallelism(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).verify_parallelism = true;
}

static void PragmaEnableForceIndexJoin(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).force_index_join = true;
}

static void PragmaForceCheckpoint(ClientContext &context, const FunctionParameters &parameters) {
	DBConfig::GetConfig(context).options.force_checkpoint = true;
}

static void PragmaDisableForceParallelism(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).verify_parallelism = false;
}

static void PragmaEnableObjectCache(ClientContext &context, const FunctionParameters &parameters) {
	DBConfig::GetConfig(context).options.object_cache_enable = true;
}

static void PragmaDisableObjectCache(ClientContext &context, const FunctionParameters &parameters) {
	DBConfig::GetConfig(context).options.object_cache_enable = false;
}

static void PragmaEnableCheckpointOnShutdown(ClientContext &context, const FunctionParameters &parameters) {
	DBConfig::GetConfig(context).options.checkpoint_on_shutdown = true;
}

static void PragmaDisableCheckpointOnShutdown(ClientContext &context, const FunctionParameters &parameters) {
	DBConfig::GetConfig(context).options.checkpoint_on_shutdown = false;
}

static void PragmaEnableOptimizer(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).enable_optimizer = true;
}

static void PragmaDisableOptimizer(ClientContext &context, const FunctionParameters &parameters) {
	ClientConfig::GetConfig(context).enable_optimizer = false;
}

void PragmaFunctions::RegisterFunction(BuiltinFunctions &set) {
	RegisterEnableProfiling(set);

	set.AddFunction(PragmaFunction::PragmaStatement("disable_profile", PragmaDisableProfiling));
	set.AddFunction(PragmaFunction::PragmaStatement("disable_profiling", PragmaDisableProfiling));

	set.AddFunction(PragmaFunction::PragmaStatement("enable_verification", PragmaEnableVerification));
	set.AddFunction(PragmaFunction::PragmaStatement("disable_verification", PragmaDisableVerification));

	set.AddFunction(PragmaFunction::PragmaStatement("verify_external", PragmaEnableExternalVerification));
	set.AddFunction(PragmaFunction::PragmaStatement("disable_verify_external", PragmaDisableExternalVerification));

	set.AddFunction(PragmaFunction::PragmaStatement("verify_serializer", PragmaVerifySerializer));
	set.AddFunction(PragmaFunction::PragmaStatement("disable_verify_serializer", PragmaDisableVerifySerializer));

	set.AddFunction(PragmaFunction::PragmaStatement("verify_parallelism", PragmaEnableForceParallelism));
	set.AddFunction(PragmaFunction::PragmaStatement("disable_verify_parallelism", PragmaDisableForceParallelism));

	set.AddFunction(PragmaFunction::PragmaStatement("enable_object_cache", PragmaEnableObjectCache));
	set.AddFunction(PragmaFunction::PragmaStatement("disable_object_cache", PragmaDisableObjectCache));

	set.AddFunction(PragmaFunction::PragmaStatement("enable_optimizer", PragmaEnableOptimizer));
	set.AddFunction(PragmaFunction::PragmaStatement("disable_optimizer", PragmaDisableOptimizer));

	set.AddFunction(PragmaFunction::PragmaStatement("force_index_join", PragmaEnableForceIndexJoin));
	set.AddFunction(PragmaFunction::PragmaStatement("force_checkpoint", PragmaForceCheckpoint));

	set.AddFunction(PragmaFunction::PragmaStatement("enable_progress_bar", PragmaEnableProgressBar));
	set.AddFunction(PragmaFunction::PragmaStatement("disable_progress_bar", PragmaDisableProgressBar));

	set.AddFunction(PragmaFunction::PragmaStatement("enable_print_progress_bar", PragmaEnablePrintProgressBar));
	set.AddFunction(PragmaFunction::PragmaStatement("disable_print_progress_bar", PragmaDisablePrintProgressBar));

	set.AddFunction(PragmaFunction::PragmaStatement("enable_checkpoint_on_shutdown", PragmaEnableCheckpointOnShutdown));
	set.AddFunction(
	    PragmaFunction::PragmaStatement("disable_checkpoint_on_shutdown", PragmaDisableCheckpointOnShutdown));
}

} // namespace duckdb








namespace duckdb {

string PragmaTableInfo(ClientContext &context, const FunctionParameters &parameters) {
	return StringUtil::Format("SELECT * FROM pragma_table_info('%s');", parameters.values[0].ToString());
}

string PragmaShowTables(ClientContext &context, const FunctionParameters &parameters) {
	return "SELECT name FROM sqlite_master ORDER BY name;";
}

string PragmaShowTablesExpanded(ClientContext &context, const FunctionParameters &parameters) {
	return R"(
			SELECT
				t.table_name,
				LIST(c.column_name order by c.column_name) AS column_names,
				LIST(c.data_type order by c.column_name) AS column_types,
				FIRST(t.temporary) AS temporary
			FROM duckdb_tables t
			JOIN duckdb_columns c
			USING (table_oid)
			GROUP BY t.table_name
			ORDER BY t.table_name;
	)";
}

string PragmaShowDatabases(ClientContext &context, const FunctionParameters &parameters) {
	return "SELECT database_name FROM duckdb_databases() WHERE NOT internal ORDER BY database_name;";
}

string PragmaAllProfiling(ClientContext &context, const FunctionParameters &parameters) {
	return "SELECT * FROM pragma_last_profiling_output() JOIN pragma_detailed_profiling_output() ON "
	       "(pragma_last_profiling_output.operator_id);";
}

string PragmaDatabaseList(ClientContext &context, const FunctionParameters &parameters) {
	return "SELECT * FROM pragma_database_list;";
}

string PragmaCollations(ClientContext &context, const FunctionParameters &parameters) {
	return "SELECT * FROM pragma_collations() ORDER BY 1;";
}

string PragmaFunctionsQuery(ClientContext &context, const FunctionParameters &parameters) {
	return "SELECT function_name AS name, upper(function_type) AS type, parameter_types AS parameters, varargs, "
	       "return_type, has_side_effects AS side_effects"
	       " FROM duckdb_functions()"
	       " WHERE function_type IN ('scalar', 'aggregate')"
	       " ORDER BY 1;";
}

string PragmaShow(ClientContext &context, const FunctionParameters &parameters) {
	// PRAGMA table_info but with some aliases
	return StringUtil::Format(
	    "SELECT name AS \"column_name\", type as \"column_type\", CASE WHEN \"notnull\" THEN 'NO' ELSE 'YES' "
	    "END AS \"null\", NULL AS \"key\", dflt_value AS \"default\", NULL AS \"extra\" FROM pragma_table_info('%s');",
	    parameters.values[0].ToString());
}

string PragmaVersion(ClientContext &context, const FunctionParameters &parameters) {
	return "SELECT * FROM pragma_version();";
}

string PragmaImportDatabase(ClientContext &context, const FunctionParameters &parameters) {
	auto &config = DBConfig::GetConfig(context);
	if (!config.options.enable_external_access) {
		throw PermissionException("Import is disabled through configuration");
	}
	auto &fs = FileSystem::GetFileSystem(context);
	auto *opener = FileSystem::GetFileOpener(context);

	string final_query;
	// read the "shema.sql" and "load.sql" files
	vector<string> files = {"schema.sql", "load.sql"};
	for (auto &file : files) {
		auto file_path = fs.JoinPath(parameters.values[0].ToString(), file);
		auto handle = fs.OpenFile(file_path, FileFlags::FILE_FLAGS_READ, FileSystem::DEFAULT_LOCK,
		                          FileSystem::DEFAULT_COMPRESSION, opener);
		auto fsize = fs.GetFileSize(*handle);
		auto buffer = unique_ptr<char[]>(new char[fsize]);
		fs.Read(*handle, buffer.get(), fsize);
		auto query = string(buffer.get(), fsize);
		// Replace the placeholder with the path provided to IMPORT
		if (file == "load.sql") {
			Parser parser;
			parser.ParseQuery(query);
			auto copy_statements = std::move(parser.statements);
			query.clear();
			for (auto &statement_p : copy_statements) {
				D_ASSERT(statement_p->type == StatementType::COPY_STATEMENT);
				auto &statement = (CopyStatement &)*statement_p;
				auto &info = *statement.info;
				auto file_name = fs.ExtractName(info.file_path);
				info.file_path = fs.JoinPath(parameters.values[0].ToString(), file_name);
				query += statement.ToString() + ";";
			}
		}
		final_query += query;
	}
	return final_query;
}

string PragmaDatabaseSize(ClientContext &context, const FunctionParameters &parameters) {
	return "SELECT * FROM pragma_database_size();";
}

string PragmaStorageInfo(ClientContext &context, const FunctionParameters &parameters) {
	return StringUtil::Format("SELECT * FROM pragma_storage_info('%s');", parameters.values[0].ToString());
}

void PragmaQueries::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(PragmaFunction::PragmaCall("table_info", PragmaTableInfo, {LogicalType::VARCHAR}));
	set.AddFunction(PragmaFunction::PragmaCall("storage_info", PragmaStorageInfo, {LogicalType::VARCHAR}));
	set.AddFunction(PragmaFunction::PragmaStatement("show_tables", PragmaShowTables));
	set.AddFunction(PragmaFunction::PragmaStatement("show_tables_expanded", PragmaShowTablesExpanded));
	set.AddFunction(PragmaFunction::PragmaStatement("show_databases", PragmaShowDatabases));
	set.AddFunction(PragmaFunction::PragmaStatement("database_list", PragmaDatabaseList));
	set.AddFunction(PragmaFunction::PragmaStatement("collations", PragmaCollations));
	set.AddFunction(PragmaFunction::PragmaCall("show", PragmaShow, {LogicalType::VARCHAR}));
	set.AddFunction(PragmaFunction::PragmaStatement("version", PragmaVersion));
	set.AddFunction(PragmaFunction::PragmaStatement("database_size", PragmaDatabaseSize));
	set.AddFunction(PragmaFunction::PragmaStatement("functions", PragmaFunctionsQuery));
	set.AddFunction(PragmaFunction::PragmaCall("import_database", PragmaImportDatabase, {LogicalType::VARCHAR}));
	set.AddFunction(PragmaFunction::PragmaStatement("all_profiling_output", PragmaAllProfiling));
}

} // namespace duckdb



namespace duckdb {

PragmaFunction::PragmaFunction(string name, PragmaType pragma_type, pragma_query_t query, pragma_function_t function,
                               vector<LogicalType> arguments, LogicalType varargs)
    : SimpleNamedParameterFunction(std::move(name), std::move(arguments), std::move(varargs)), type(pragma_type),
      query(query), function(function) {
}

PragmaFunction PragmaFunction::PragmaCall(const string &name, pragma_query_t query, vector<LogicalType> arguments,
                                          LogicalType varargs) {
	return PragmaFunction(name, PragmaType::PRAGMA_CALL, query, nullptr, std::move(arguments), std::move(varargs));
}

PragmaFunction PragmaFunction::PragmaCall(const string &name, pragma_function_t function, vector<LogicalType> arguments,
                                          LogicalType varargs) {
	return PragmaFunction(name, PragmaType::PRAGMA_CALL, nullptr, function, std::move(arguments), std::move(varargs));
}

PragmaFunction PragmaFunction::PragmaStatement(const string &name, pragma_query_t query) {
	vector<LogicalType> types;
	return PragmaFunction(name, PragmaType::PRAGMA_STATEMENT, query, nullptr, std::move(types), LogicalType::INVALID);
}

PragmaFunction PragmaFunction::PragmaStatement(const string &name, pragma_function_t function) {
	vector<LogicalType> types;
	return PragmaFunction(name, PragmaType::PRAGMA_STATEMENT, nullptr, function, std::move(types),
	                      LogicalType::INVALID);
}

string PragmaFunction::ToString() {
	switch (type) {
	case PragmaType::PRAGMA_STATEMENT:
		return StringUtil::Format("PRAGMA %s", name);
	case PragmaType::PRAGMA_CALL: {
		return StringUtil::Format("PRAGMA %s", SimpleNamedParameterFunction::ToString());
	}
	default:
		return "UNKNOWN";
	}
}

} // namespace duckdb



namespace duckdb {

struct Base64EncodeOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {
		auto result_str = StringVector::EmptyString(result, Blob::ToBase64Size(input));
		Blob::ToBase64(input, result_str.GetDataWriteable());
		result_str.Finalize();
		return result_str;
	}
};

struct Base64DecodeOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {
		auto result_size = Blob::FromBase64Size(input);
		auto result_blob = StringVector::EmptyString(result, result_size);
		Blob::FromBase64(input, (data_ptr_t)result_blob.GetDataWriteable(), result_size);
		result_blob.Finalize();
		return result_blob;
	}
};

static void Base64EncodeFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	// decode is also a nop cast, but requires verification if the provided string is actually
	UnaryExecutor::ExecuteString<string_t, string_t, Base64EncodeOperator>(args.data[0], result, args.size());
}

static void Base64DecodeFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	// decode is also a nop cast, but requires verification if the provided string is actually
	UnaryExecutor::ExecuteString<string_t, string_t, Base64DecodeOperator>(args.data[0], result, args.size());
}

void Base64Fun::RegisterFunction(BuiltinFunctions &set) {
	// base64 encode
	ScalarFunction to_base64({LogicalType::BLOB}, LogicalType::VARCHAR, Base64EncodeFunction);
	set.AddFunction({"base64", "to_base64"}, to_base64); // to_base64 is a mysql alias

	set.AddFunction(ScalarFunction("from_base64", {LogicalType::VARCHAR}, LogicalType::BLOB, Base64DecodeFunction));
}

} // namespace duckdb



namespace duckdb {

static void EncodeFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	// encode is essentially a nop cast from varchar to blob
	// we only need to reinterpret the data using the blob type
	result.Reinterpret(args.data[0]);
}

struct BlobDecodeOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input) {
		auto input_data = input.GetDataUnsafe();
		auto input_length = input.GetSize();
		if (Utf8Proc::Analyze(input_data, input_length) == UnicodeType::INVALID) {
			throw ConversionException(
			    "Failure in decode: could not convert blob to UTF8 string, the blob contained invalid UTF8 characters");
		}
		return input;
	}
};

static void DecodeFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	// decode is also a nop cast, but requires verification if the provided string is actually
	UnaryExecutor::Execute<string_t, string_t, BlobDecodeOperator>(args.data[0], result, args.size());
	StringVector::AddHeapReference(result, args.data[0]);
}

void EncodeFun::RegisterFunction(BuiltinFunctions &set) {
	// encode goes from varchar -> blob, this never fails
	set.AddFunction(ScalarFunction("encode", {LogicalType::VARCHAR}, LogicalType::BLOB, EncodeFunction));
	// decode goes from blob -> varchar, this fails if the varchar is not valid utf8
	set.AddFunction(ScalarFunction("decode", {LogicalType::BLOB}, LogicalType::VARCHAR, DecodeFunction));
}

} // namespace duckdb








namespace duckdb {

static void AgeFunctionStandard(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.ColumnCount() == 1);
	auto current_timestamp = Timestamp::GetCurrentTimestamp();

	UnaryExecutor::ExecuteWithNulls<timestamp_t, interval_t>(input.data[0], result, input.size(),
	                                                         [&](timestamp_t input, ValidityMask &mask, idx_t idx) {
		                                                         if (Timestamp::IsFinite(input)) {
			                                                         return Interval::GetAge(current_timestamp, input);
		                                                         } else {
			                                                         mask.SetInvalid(idx);
			                                                         return interval_t();
		                                                         }
	                                                         });
}

static void AgeFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.ColumnCount() == 2);

	BinaryExecutor::ExecuteWithNulls<timestamp_t, timestamp_t, interval_t>(
	    input.data[0], input.data[1], result, input.size(),
	    [&](timestamp_t input1, timestamp_t input2, ValidityMask &mask, idx_t idx) {
		    if (Timestamp::IsFinite(input1) && Timestamp::IsFinite(input2)) {
			    return Interval::GetAge(input1, input2);
		    } else {
			    mask.SetInvalid(idx);
			    return interval_t();
		    }
	    });
}

void AgeFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet age("age");
	age.AddFunction(ScalarFunction({LogicalType::TIMESTAMP}, LogicalType::INTERVAL, AgeFunctionStandard));
	age.AddFunction(
	    ScalarFunction({LogicalType::TIMESTAMP, LogicalType::TIMESTAMP}, LogicalType::INTERVAL, AgeFunction));
	set.AddFunction(age);
}

} // namespace duckdb









namespace duckdb {

static timestamp_t GetTransactionTimestamp(ExpressionState &state) {
	return MetaTransaction::Get(state.GetContext()).start_timestamp;
}

static void CurrentTimeFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.ColumnCount() == 0);
	auto val = Value::TIME(Timestamp::GetTime(GetTransactionTimestamp(state)));
	result.Reference(val);
}

static void CurrentDateFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.ColumnCount() == 0);

	auto val = Value::DATE(Timestamp::GetDate(GetTransactionTimestamp(state)));
	result.Reference(val);
}

static void CurrentTimestampFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.ColumnCount() == 0);

	auto val = Value::TIMESTAMPTZ(GetTransactionTimestamp(state));
	result.Reference(val);
}

void CurrentTimeFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunction current_time("get_current_time", {}, LogicalType::TIME, CurrentTimeFunction);
	;
	current_time.side_effects = FunctionSideEffects::HAS_SIDE_EFFECTS;
	set.AddFunction(current_time);
}

void CurrentDateFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunction current_date({}, LogicalType::DATE, CurrentDateFunction);
	;
	current_date.side_effects = FunctionSideEffects::HAS_SIDE_EFFECTS;
	set.AddFunction({"today", "current_date"}, current_date);
}

void CurrentTimestampFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunction current_timestamp({}, LogicalType::TIMESTAMP_TZ, CurrentTimestampFunction);
	current_timestamp.side_effects = FunctionSideEffects::HAS_SIDE_EFFECTS;
	set.AddFunction({"now", "get_current_timestamp", "transaction_timestamp"}, current_timestamp);
}

} // namespace duckdb













namespace duckdb {

// This function is an implementation of the "period-crossing" date difference function from T-SQL
// https://docs.microsoft.com/en-us/sql/t-sql/functions/datediff-transact-sql?view=sql-server-ver15
struct DateDiff {
	template <class TA, class TB, class TR, class OP>
	static inline void BinaryExecute(Vector &left, Vector &right, Vector &result, idx_t count) {
		BinaryExecutor::ExecuteWithNulls<TA, TB, TR>(
		    left, right, result, count, [&](TA startdate, TB enddate, ValidityMask &mask, idx_t idx) {
			    if (Value::IsFinite(startdate) && Value::IsFinite(enddate)) {
				    return OP::template Operation<TA, TB, TR>(startdate, enddate);
			    } else {
				    mask.SetInvalid(idx);
				    return TR();
			    }
		    });
	}

	struct YearOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return Date::ExtractYear(enddate) - Date::ExtractYear(startdate);
		}
	};

	struct MonthOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			int32_t start_year, start_month, start_day;
			Date::Convert(startdate, start_year, start_month, start_day);
			int32_t end_year, end_month, end_day;
			Date::Convert(enddate, end_year, end_month, end_day);

			return (end_year * 12 + end_month - 1) - (start_year * 12 + start_month - 1);
		}
	};

	struct DayOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return TR(Date::EpochDays(enddate)) - TR(Date::EpochDays(startdate));
		}
	};

	struct DecadeOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return Date::ExtractYear(enddate) / 10 - Date::ExtractYear(startdate) / 10;
		}
	};

	struct CenturyOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return Date::ExtractYear(enddate) / 100 - Date::ExtractYear(startdate) / 100;
		}
	};

	struct MilleniumOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return Date::ExtractYear(enddate) / 1000 - Date::ExtractYear(startdate) / 1000;
		}
	};

	struct QuarterOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			int32_t start_year, start_month, start_day;
			Date::Convert(startdate, start_year, start_month, start_day);
			int32_t end_year, end_month, end_day;
			Date::Convert(enddate, end_year, end_month, end_day);

			return (end_year * 12 + end_month - 1) / Interval::MONTHS_PER_QUARTER -
			       (start_year * 12 + start_month - 1) / Interval::MONTHS_PER_QUARTER;
		}
	};

	struct WeekOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return Date::Epoch(Date::GetMondayOfCurrentWeek(enddate)) / Interval::SECS_PER_WEEK -
			       Date::Epoch(Date::GetMondayOfCurrentWeek(startdate)) / Interval::SECS_PER_WEEK;
		}
	};

	struct ISOYearOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return Date::ExtractISOYearNumber(enddate) - Date::ExtractISOYearNumber(startdate);
		}
	};

	struct MicrosecondsOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return Date::EpochMicroseconds(enddate) - Date::EpochMicroseconds(startdate);
		}
	};

	struct MillisecondsOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return Date::EpochMicroseconds(enddate) / Interval::MICROS_PER_MSEC -
			       Date::EpochMicroseconds(startdate) / Interval::MICROS_PER_MSEC;
		}
	};

	struct SecondsOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return Date::Epoch(enddate) - Date::Epoch(startdate);
		}
	};

	struct MinutesOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return Date::Epoch(enddate) / Interval::SECS_PER_MINUTE -
			       Date::Epoch(startdate) / Interval::SECS_PER_MINUTE;
		}
	};

	struct HoursOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return Date::Epoch(enddate) / Interval::SECS_PER_HOUR - Date::Epoch(startdate) / Interval::SECS_PER_HOUR;
		}
	};
};

// TIMESTAMP specialisations
template <>
int64_t DateDiff::YearOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return YearOperator::Operation<date_t, date_t, int64_t>(Timestamp::GetDate(startdate), Timestamp::GetDate(enddate));
}

template <>
int64_t DateDiff::MonthOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return MonthOperator::Operation<date_t, date_t, int64_t>(Timestamp::GetDate(startdate),
	                                                         Timestamp::GetDate(enddate));
}

template <>
int64_t DateDiff::DayOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return DayOperator::Operation<date_t, date_t, int64_t>(Timestamp::GetDate(startdate), Timestamp::GetDate(enddate));
}

template <>
int64_t DateDiff::DecadeOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return DecadeOperator::Operation<date_t, date_t, int64_t>(Timestamp::GetDate(startdate),
	                                                          Timestamp::GetDate(enddate));
}

template <>
int64_t DateDiff::CenturyOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return CenturyOperator::Operation<date_t, date_t, int64_t>(Timestamp::GetDate(startdate),
	                                                           Timestamp::GetDate(enddate));
}

template <>
int64_t DateDiff::MilleniumOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return MilleniumOperator::Operation<date_t, date_t, int64_t>(Timestamp::GetDate(startdate),
	                                                             Timestamp::GetDate(enddate));
}

template <>
int64_t DateDiff::QuarterOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return QuarterOperator::Operation<date_t, date_t, int64_t>(Timestamp::GetDate(startdate),
	                                                           Timestamp::GetDate(enddate));
}

template <>
int64_t DateDiff::WeekOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return WeekOperator::Operation<date_t, date_t, int64_t>(Timestamp::GetDate(startdate), Timestamp::GetDate(enddate));
}

template <>
int64_t DateDiff::ISOYearOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return ISOYearOperator::Operation<date_t, date_t, int64_t>(Timestamp::GetDate(startdate),
	                                                           Timestamp::GetDate(enddate));
}

template <>
int64_t DateDiff::MicrosecondsOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	const auto start = Timestamp::GetEpochMicroSeconds(startdate);
	const auto end = Timestamp::GetEpochMicroSeconds(enddate);
	return SubtractOperatorOverflowCheck::Operation<int64_t, int64_t, int64_t>(end, start);
}

template <>
int64_t DateDiff::MillisecondsOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return Timestamp::GetEpochMs(enddate) - Timestamp::GetEpochMs(startdate);
}

template <>
int64_t DateDiff::SecondsOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return Timestamp::GetEpochSeconds(enddate) - Timestamp::GetEpochSeconds(startdate);
}

template <>
int64_t DateDiff::MinutesOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return Timestamp::GetEpochSeconds(enddate) / Interval::SECS_PER_MINUTE -
	       Timestamp::GetEpochSeconds(startdate) / Interval::SECS_PER_MINUTE;
}

template <>
int64_t DateDiff::HoursOperator::Operation(timestamp_t startdate, timestamp_t enddate) {
	return Timestamp::GetEpochSeconds(enddate) / Interval::SECS_PER_HOUR -
	       Timestamp::GetEpochSeconds(startdate) / Interval::SECS_PER_HOUR;
}

// TIME specialisations
template <>
int64_t DateDiff::YearOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"year\" not recognized");
}

template <>
int64_t DateDiff::MonthOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"month\" not recognized");
}

template <>
int64_t DateDiff::DayOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"day\" not recognized");
}

template <>
int64_t DateDiff::DecadeOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"decade\" not recognized");
}

template <>
int64_t DateDiff::CenturyOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"century\" not recognized");
}

template <>
int64_t DateDiff::MilleniumOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"millennium\" not recognized");
}

template <>
int64_t DateDiff::QuarterOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"quarter\" not recognized");
}

template <>
int64_t DateDiff::WeekOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"week\" not recognized");
}

template <>
int64_t DateDiff::ISOYearOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"isoyear\" not recognized");
}

template <>
int64_t DateDiff::MicrosecondsOperator::Operation(dtime_t startdate, dtime_t enddate) {
	return enddate.micros - startdate.micros;
}

template <>
int64_t DateDiff::MillisecondsOperator::Operation(dtime_t startdate, dtime_t enddate) {
	return enddate.micros / Interval::MICROS_PER_MSEC - startdate.micros / Interval::MICROS_PER_MSEC;
}

template <>
int64_t DateDiff::SecondsOperator::Operation(dtime_t startdate, dtime_t enddate) {
	return enddate.micros / Interval::MICROS_PER_SEC - startdate.micros / Interval::MICROS_PER_SEC;
}

template <>
int64_t DateDiff::MinutesOperator::Operation(dtime_t startdate, dtime_t enddate) {
	return enddate.micros / Interval::MICROS_PER_MINUTE - startdate.micros / Interval::MICROS_PER_MINUTE;
}

template <>
int64_t DateDiff::HoursOperator::Operation(dtime_t startdate, dtime_t enddate) {
	return enddate.micros / Interval::MICROS_PER_HOUR - startdate.micros / Interval::MICROS_PER_HOUR;
}

template <typename TA, typename TB, typename TR>
static int64_t DifferenceDates(DatePartSpecifier type, TA startdate, TB enddate) {
	switch (type) {
	case DatePartSpecifier::YEAR:
		return DateDiff::YearOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::MONTH:
		return DateDiff::MonthOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::DAY:
	case DatePartSpecifier::DOW:
	case DatePartSpecifier::ISODOW:
	case DatePartSpecifier::DOY:
		return DateDiff::DayOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::DECADE:
		return DateDiff::DecadeOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::CENTURY:
		return DateDiff::CenturyOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::MILLENNIUM:
		return DateDiff::MilleniumOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::QUARTER:
		return DateDiff::QuarterOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::WEEK:
	case DatePartSpecifier::YEARWEEK:
		return DateDiff::WeekOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::ISOYEAR:
		return DateDiff::ISOYearOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::MICROSECONDS:
		return DateDiff::MicrosecondsOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::MILLISECONDS:
		return DateDiff::MillisecondsOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::SECOND:
	case DatePartSpecifier::EPOCH:
		return DateDiff::SecondsOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::MINUTE:
		return DateDiff::MinutesOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::HOUR:
		return DateDiff::HoursOperator::template Operation<TA, TB, TR>(startdate, enddate);
	default:
		throw NotImplementedException("Specifier type not implemented for DATEDIFF");
	}
}

struct DateDiffTernaryOperator {
	template <typename TS, typename TA, typename TB, typename TR>
	static inline TR Operation(TS part, TA startdate, TB enddate, ValidityMask &mask, idx_t idx) {
		if (Value::IsFinite(startdate) && Value::IsFinite(enddate)) {
			return DifferenceDates<TA, TB, TR>(GetDatePartSpecifier(part.GetString()), startdate, enddate);
		} else {
			mask.SetInvalid(idx);
			return TR();
		}
	}
};

template <typename TA, typename TB, typename TR>
static void DateDiffBinaryExecutor(DatePartSpecifier type, Vector &left, Vector &right, Vector &result, idx_t count) {
	switch (type) {
	case DatePartSpecifier::YEAR:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::YearOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::MONTH:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::MonthOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::DAY:
	case DatePartSpecifier::DOW:
	case DatePartSpecifier::ISODOW:
	case DatePartSpecifier::DOY:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::DayOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::DECADE:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::DecadeOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::CENTURY:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::CenturyOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::MILLENNIUM:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::MilleniumOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::QUARTER:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::QuarterOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::WEEK:
	case DatePartSpecifier::YEARWEEK:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::WeekOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::ISOYEAR:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::ISOYearOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::MICROSECONDS:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::MicrosecondsOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::MILLISECONDS:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::MillisecondsOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::SECOND:
	case DatePartSpecifier::EPOCH:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::SecondsOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::MINUTE:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::MinutesOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::HOUR:
		DateDiff::BinaryExecute<TA, TB, TR, DateDiff::HoursOperator>(left, right, result, count);
		break;
	default:
		throw NotImplementedException("Specifier type not implemented for DATEDIFF");
	}
}

template <typename T>
static void DateDiffFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 3);
	auto &part_arg = args.data[0];
	auto &start_arg = args.data[1];
	auto &end_arg = args.data[2];

	if (part_arg.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		// Common case of constant part.
		if (ConstantVector::IsNull(part_arg)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
		} else {
			const auto type = GetDatePartSpecifier(ConstantVector::GetData<string_t>(part_arg)->GetString());
			DateDiffBinaryExecutor<T, T, int64_t>(type, start_arg, end_arg, result, args.size());
		}
	} else {
		TernaryExecutor::ExecuteWithNulls<string_t, T, T, int64_t>(
		    part_arg, start_arg, end_arg, result, args.size(),
		    DateDiffTernaryOperator::Operation<string_t, T, T, int64_t>);
	}
}

void DateDiffFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet date_diff("date_diff");
	date_diff.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::DATE, LogicalType::DATE},
	                                     LogicalType::BIGINT, DateDiffFunction<date_t>));
	date_diff.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::TIMESTAMP, LogicalType::TIMESTAMP},
	                                     LogicalType::BIGINT, DateDiffFunction<timestamp_t>));
	date_diff.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::TIME, LogicalType::TIME},
	                                     LogicalType::BIGINT, DateDiffFunction<dtime_t>));
	set.AddFunction(date_diff);

	date_diff.name = "datediff";
	set.AddFunction(date_diff);
}

} // namespace duckdb














namespace duckdb {

bool TryGetDatePartSpecifier(const string &specifier_p, DatePartSpecifier &result) {
	auto specifier = StringUtil::Lower(specifier_p);
	if (specifier == "year" || specifier == "yr" || specifier == "y" || specifier == "years" || specifier == "yrs") {
		result = DatePartSpecifier::YEAR;
	} else if (specifier == "month" || specifier == "mon" || specifier == "months" || specifier == "mons") {
		result = DatePartSpecifier::MONTH;
	} else if (specifier == "day" || specifier == "days" || specifier == "d" || specifier == "dayofmonth") {
		result = DatePartSpecifier::DAY;
	} else if (specifier == "decade" || specifier == "dec" || specifier == "decades" || specifier == "decs") {
		result = DatePartSpecifier::DECADE;
	} else if (specifier == "century" || specifier == "cent" || specifier == "centuries" || specifier == "c") {
		result = DatePartSpecifier::CENTURY;
	} else if (specifier == "millennium" || specifier == "mil" || specifier == "millenniums" ||
	           specifier == "millennia" || specifier == "mils" || specifier == "millenium") {
		result = DatePartSpecifier::MILLENNIUM;
	} else if (specifier == "microseconds" || specifier == "microsecond" || specifier == "us" || specifier == "usec" ||
	           specifier == "usecs" || specifier == "usecond" || specifier == "useconds") {
		result = DatePartSpecifier::MICROSECONDS;
	} else if (specifier == "milliseconds" || specifier == "millisecond" || specifier == "ms" || specifier == "msec" ||
	           specifier == "msecs" || specifier == "msecond" || specifier == "mseconds") {
		result = DatePartSpecifier::MILLISECONDS;
	} else if (specifier == "second" || specifier == "sec" || specifier == "seconds" || specifier == "secs" ||
	           specifier == "s") {
		result = DatePartSpecifier::SECOND;
	} else if (specifier == "minute" || specifier == "min" || specifier == "minutes" || specifier == "mins" ||
	           specifier == "m") {
		result = DatePartSpecifier::MINUTE;
	} else if (specifier == "hour" || specifier == "hr" || specifier == "hours" || specifier == "hrs" ||
	           specifier == "h") {
		result = DatePartSpecifier::HOUR;
	} else if (specifier == "epoch") {
		// seconds since 1970-01-01
		result = DatePartSpecifier::EPOCH;
	} else if (specifier == "dow" || specifier == "dayofweek" || specifier == "weekday") {
		// day of the week (Sunday = 0, Saturday = 6)
		result = DatePartSpecifier::DOW;
	} else if (specifier == "isodow") {
		// isodow (Monday = 1, Sunday = 7)
		result = DatePartSpecifier::ISODOW;
	} else if (specifier == "week" || specifier == "weeks" || specifier == "w" || specifier == "weekofyear") {
		// ISO week number
		result = DatePartSpecifier::WEEK;
	} else if (specifier == "doy" || specifier == "dayofyear") {
		// day of the year (1-365/366)
		result = DatePartSpecifier::DOY;
	} else if (specifier == "quarter" || specifier == "quarters") {
		// quarter of the year (1-4)
		result = DatePartSpecifier::QUARTER;
	} else if (specifier == "yearweek") {
		// Combined isoyear and isoweek YYYYWW
		result = DatePartSpecifier::YEARWEEK;
	} else if (specifier == "isoyear") {
		// ISO year (first week of the year may be in previous year)
		result = DatePartSpecifier::ISOYEAR;
	} else if (specifier == "era") {
		result = DatePartSpecifier::ERA;
	} else if (specifier == "timezone") {
		result = DatePartSpecifier::TIMEZONE;
	} else if (specifier == "timezone_hour") {
		result = DatePartSpecifier::TIMEZONE_HOUR;
	} else if (specifier == "timezone_minute") {
		result = DatePartSpecifier::TIMEZONE_MINUTE;
	} else {
		return false;
	}
	return true;
}

DatePartSpecifier GetDatePartSpecifier(const string &specifier) {
	DatePartSpecifier result;
	if (!TryGetDatePartSpecifier(specifier, result)) {
		throw ConversionException("extract specifier \"%s\" not recognized", specifier);
	}
	return result;
}

DatePartSpecifier GetDateTypePartSpecifier(const string &specifier, LogicalType &type) {
	const auto part = GetDatePartSpecifier(specifier);
	switch (type.id()) {
	case LogicalType::TIMESTAMP:
	case LogicalType::TIMESTAMP_TZ:
		return part;
	case LogicalType::DATE:
		switch (part) {
		case DatePartSpecifier::YEAR:
		case DatePartSpecifier::MONTH:
		case DatePartSpecifier::DAY:
		case DatePartSpecifier::DECADE:
		case DatePartSpecifier::CENTURY:
		case DatePartSpecifier::MILLENNIUM:
		case DatePartSpecifier::DOW:
		case DatePartSpecifier::ISODOW:
		case DatePartSpecifier::ISOYEAR:
		case DatePartSpecifier::WEEK:
		case DatePartSpecifier::QUARTER:
		case DatePartSpecifier::DOY:
		case DatePartSpecifier::YEARWEEK:
		case DatePartSpecifier::ERA:
			return part;
		default:
			break;
		}
		break;
	case LogicalType::TIME:
		switch (part) {
		case DatePartSpecifier::MICROSECONDS:
		case DatePartSpecifier::MILLISECONDS:
		case DatePartSpecifier::SECOND:
		case DatePartSpecifier::MINUTE:
		case DatePartSpecifier::HOUR:
		case DatePartSpecifier::EPOCH:
		case DatePartSpecifier::TIMEZONE:
		case DatePartSpecifier::TIMEZONE_HOUR:
		case DatePartSpecifier::TIMEZONE_MINUTE:
			return part;
		default:
			break;
		}
		break;
	case LogicalType::INTERVAL:
		switch (part) {
		case DatePartSpecifier::YEAR:
		case DatePartSpecifier::MONTH:
		case DatePartSpecifier::DAY:
		case DatePartSpecifier::DECADE:
		case DatePartSpecifier::CENTURY:
		case DatePartSpecifier::QUARTER:
		case DatePartSpecifier::MILLENNIUM:
		case DatePartSpecifier::MICROSECONDS:
		case DatePartSpecifier::MILLISECONDS:
		case DatePartSpecifier::SECOND:
		case DatePartSpecifier::MINUTE:
		case DatePartSpecifier::HOUR:
		case DatePartSpecifier::EPOCH:
			return part;
		default:
			break;
		}
		break;
	default:
		break;
	}

	throw NotImplementedException("\"%s\" units \"%s\" not recognized", LogicalTypeIdToString(type.id()), specifier);
}

template <int64_t MIN, int64_t MAX>
static unique_ptr<BaseStatistics> PropagateSimpleDatePartStatistics(vector<unique_ptr<BaseStatistics>> &child_stats) {
	// we can always propagate simple date part statistics
	// since the min and max can never exceed these bounds
	auto result = make_unique<NumericStatistics>(LogicalType::BIGINT, Value::BIGINT(MIN), Value::BIGINT(MAX),
	                                             StatisticsType::LOCAL_STATS);
	if (!child_stats[0]) {
		// if there are no child stats, we don't know
		result->validity_stats = make_unique<ValidityStatistics>(true);
	} else if (child_stats[0]->validity_stats) {
		result->validity_stats = child_stats[0]->validity_stats->Copy();
	}
	return std::move(result);
}

struct DatePart {
	template <class T, class OP>
	static unique_ptr<BaseStatistics> PropagateDatePartStatistics(vector<unique_ptr<BaseStatistics>> &child_stats) {
		// we can only propagate complex date part stats if the child has stats
		if (!child_stats[0]) {
			return nullptr;
		}
		auto &nstats = (NumericStatistics &)*child_stats[0];
		if (nstats.min.IsNull() || nstats.max.IsNull()) {
			return nullptr;
		}
		// run the operator on both the min and the max, this gives us the [min, max] bound
		auto min = nstats.min.GetValueUnsafe<T>();
		auto max = nstats.max.GetValueUnsafe<T>();
		if (min > max) {
			return nullptr;
		}
		// Infinities prevent us from computing generic ranges
		if (!Value::IsFinite(min) || !Value::IsFinite(max)) {
			return nullptr;
		}
		auto min_part = OP::template Operation<T, int64_t>(min);
		auto max_part = OP::template Operation<T, int64_t>(max);
		auto result = make_unique<NumericStatistics>(LogicalType::BIGINT, Value::BIGINT(min_part),
		                                             Value::BIGINT(max_part), StatisticsType::LOCAL_STATS);
		if (child_stats[0]->validity_stats) {
			result->validity_stats = child_stats[0]->validity_stats->Copy();
		}
		return std::move(result);
	}

	template <typename OP>
	struct PartOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input, ValidityMask &mask, idx_t idx, void *dataptr) {
			if (Value::IsFinite(input)) {
				return OP::template Operation<TA, TR>(input);
			} else {
				mask.SetInvalid(idx);
				return TR();
			}
		}
	};

	template <class TA, class TR, class OP>
	static void UnaryFunction(DataChunk &input, ExpressionState &state, Vector &result) {
		D_ASSERT(input.ColumnCount() >= 1);
		using IOP = PartOperator<OP>;
		UnaryExecutor::GenericExecute<TA, TR, IOP>(input.data[0], result, input.size(), nullptr, true);
	}

	struct YearOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::ExtractYear(input);
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateDatePartStatistics<T, YearOperator>(input.child_stats);
		}
	};

	struct MonthOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::ExtractMonth(input);
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			// min/max of month operator is [1, 12]
			return PropagateSimpleDatePartStatistics<1, 12>(input.child_stats);
		}
	};

	struct DayOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::ExtractDay(input);
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			// min/max of day operator is [1, 31]
			return PropagateSimpleDatePartStatistics<1, 31>(input.child_stats);
		}
	};

	struct DecadeOperator {
		// From the PG docs: "The year field divided by 10"
		template <typename TR>
		static inline TR DecadeFromYear(TR yyyy) {
			return yyyy / 10;
		}

		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return DecadeFromYear(YearOperator::Operation<TA, TR>(input));
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateDatePartStatistics<T, DecadeOperator>(input.child_stats);
		}
	};

	struct CenturyOperator {
		// From the PG docs:
		// "The first century starts at 0001-01-01 00:00:00 AD, although they did not know it at the time.
		// This definition applies to all Gregorian calendar countries.
		// There is no century number 0, you go from -1 century to 1 century.
		// If you disagree with this, please write your complaint to: Pope, Cathedral Saint-Peter of Roma, Vatican."
		// (To be fair, His Holiness had nothing to do with this -
		// it was the lack of zero in the counting systems of the time...)
		template <typename TR>
		static inline TR CenturyFromYear(TR yyyy) {
			if (yyyy > 0) {
				return ((yyyy - 1) / 100) + 1;
			} else {
				return (yyyy / 100) - 1;
			}
		}

		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return CenturyFromYear(YearOperator::Operation<TA, TR>(input));
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateDatePartStatistics<T, CenturyOperator>(input.child_stats);
		}
	};

	struct MillenniumOperator {
		// See the century comment
		template <typename TR>
		static inline TR MillenniumFromYear(TR yyyy) {
			if (yyyy > 0) {
				return ((yyyy - 1) / 1000) + 1;
			} else {
				return (yyyy / 1000) - 1;
			}
		}

		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return MillenniumFromYear<TR>(YearOperator::Operation<TA, TR>(input));
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateDatePartStatistics<T, MillenniumOperator>(input.child_stats);
		}
	};

	struct QuarterOperator {
		template <class TR>
		static inline TR QuarterFromMonth(TR mm) {
			return (mm - 1) / Interval::MONTHS_PER_QUARTER + 1;
		}

		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return QuarterFromMonth(Date::ExtractMonth(input));
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			// min/max of quarter operator is [1, 4]
			return PropagateSimpleDatePartStatistics<1, 4>(input.child_stats);
		}
	};

	struct DayOfWeekOperator {
		template <class TR>
		static inline TR DayOfWeekFromISO(TR isodow) {
			// day of the week (Sunday = 0, Saturday = 6)
			// turn sunday into 0 by doing mod 7
			return isodow % 7;
		}

		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return DayOfWeekFromISO(Date::ExtractISODayOfTheWeek(input));
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateSimpleDatePartStatistics<0, 6>(input.child_stats);
		}
	};

	struct ISODayOfWeekOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			// isodow (Monday = 1, Sunday = 7)
			return Date::ExtractISODayOfTheWeek(input);
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateSimpleDatePartStatistics<1, 7>(input.child_stats);
		}
	};

	struct DayOfYearOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::ExtractDayOfTheYear(input);
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateSimpleDatePartStatistics<1, 366>(input.child_stats);
		}
	};

	struct WeekOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::ExtractISOWeekNumber(input);
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateSimpleDatePartStatistics<1, 54>(input.child_stats);
		}
	};

	struct ISOYearOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::ExtractISOYearNumber(input);
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateDatePartStatistics<T, ISOYearOperator>(input.child_stats);
		}
	};

	struct YearWeekOperator {
		template <class TR>
		static inline TR YearWeekFromParts(TR yyyy, TR ww) {
			return yyyy * 100 + ((yyyy > 0) ? ww : -ww);
		}

		template <class TA, class TR>
		static inline TR Operation(TA input) {
			int32_t yyyy, ww;
			Date::ExtractISOYearWeek(input, yyyy, ww);
			return YearWeekFromParts(yyyy, ww);
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateDatePartStatistics<T, YearWeekOperator>(input.child_stats);
		}
	};

	struct MicrosecondsOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return 0;
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateSimpleDatePartStatistics<0, 60000000>(input.child_stats);
		}
	};

	struct MillisecondsOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return 0;
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateSimpleDatePartStatistics<0, 60000>(input.child_stats);
		}
	};

	struct SecondsOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return 0;
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateSimpleDatePartStatistics<0, 60>(input.child_stats);
		}
	};

	struct MinutesOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return 0;
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateSimpleDatePartStatistics<0, 60>(input.child_stats);
		}
	};

	struct HoursOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return 0;
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateSimpleDatePartStatistics<0, 24>(input.child_stats);
		}
	};

	struct EpochOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::Epoch(input);
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateDatePartStatistics<T, EpochOperator>(input.child_stats);
		}
	};

	struct EraOperator {
		template <class TR>
		static inline TR EraFromYear(TR yyyy) {
			return yyyy > 0 ? 1 : 0;
		}

		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return EraFromYear(Date::ExtractYear(input));
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateSimpleDatePartStatistics<0, 1>(input.child_stats);
		}
	};

	struct TimezoneOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			// Regular timestamps are UTC.
			return 0;
		}

		template <class T>
		static unique_ptr<BaseStatistics> PropagateStatistics(ClientContext &context, FunctionStatisticsInput &input) {
			return PropagateSimpleDatePartStatistics<0, 0>(input.child_stats);
		}
	};

	// These are all zero and have the same restrictions
	using TimezoneHourOperator = TimezoneOperator;
	using TimezoneMinuteOperator = TimezoneOperator;

	struct StructOperator {
		using part_codes_t = vector<DatePartSpecifier>;
		using part_mask_t = uint64_t;

		enum MaskBits : uint8_t {
			YMD = 1 << 0,
			DOW = 1 << 1,
			DOY = 1 << 2,
			EPOCH = 1 << 3,
			TIME = 1 << 4,
			ZONE = 1 << 5,
			ISO = 1 << 6
		};

		static part_mask_t GetMask(const part_codes_t &part_codes) {
			part_mask_t mask = 0;
			for (const auto &part_code : part_codes) {
				switch (part_code) {
				case DatePartSpecifier::YEAR:
				case DatePartSpecifier::MONTH:
				case DatePartSpecifier::DAY:
				case DatePartSpecifier::DECADE:
				case DatePartSpecifier::CENTURY:
				case DatePartSpecifier::MILLENNIUM:
				case DatePartSpecifier::QUARTER:
				case DatePartSpecifier::ERA:
					mask |= YMD;
					break;
				case DatePartSpecifier::YEARWEEK:
				case DatePartSpecifier::WEEK:
				case DatePartSpecifier::ISOYEAR:
					mask |= ISO;
					break;
				case DatePartSpecifier::DOW:
				case DatePartSpecifier::ISODOW:
					mask |= DOW;
					break;
				case DatePartSpecifier::DOY:
					mask |= DOY;
					break;
				case DatePartSpecifier::EPOCH:
					mask |= EPOCH;
					break;
				case DatePartSpecifier::MICROSECONDS:
				case DatePartSpecifier::MILLISECONDS:
				case DatePartSpecifier::SECOND:
				case DatePartSpecifier::MINUTE:
				case DatePartSpecifier::HOUR:
					mask |= TIME;
					break;
				case DatePartSpecifier::TIMEZONE:
				case DatePartSpecifier::TIMEZONE_HOUR:
				case DatePartSpecifier::TIMEZONE_MINUTE:
					mask |= ZONE;
					break;
				}
			}
			return mask;
		}

		template <typename P>
		static inline P HasPartValue(P *part_values, DatePartSpecifier part) {
			return part_values[int(part)];
		}

		template <class TA, class TR>
		static inline void Operation(TR **part_values, const TA &input, const idx_t idx, const part_mask_t mask) {
			TR *part_data;
			// YMD calculations
			int32_t yyyy = 1970;
			int32_t mm = 0;
			int32_t dd = 1;
			if (mask & YMD) {
				Date::Convert(input, yyyy, mm, dd);
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::YEAR))) {
					part_data[idx] = yyyy;
				}
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::MONTH))) {
					part_data[idx] = mm;
				}
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::DAY))) {
					part_data[idx] = dd;
				}
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::DECADE))) {
					part_data[idx] = DecadeOperator::DecadeFromYear(yyyy);
				}
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::CENTURY))) {
					part_data[idx] = CenturyOperator::CenturyFromYear(yyyy);
				}
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::MILLENNIUM))) {
					part_data[idx] = MillenniumOperator::MillenniumFromYear(yyyy);
				}
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::QUARTER))) {
					part_data[idx] = QuarterOperator::QuarterFromMonth(mm);
				}
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::ERA))) {
					part_data[idx] = EraOperator::EraFromYear(yyyy);
				}
			}

			// Week calculations
			if (mask & DOW) {
				auto isodow = Date::ExtractISODayOfTheWeek(input);
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::DOW))) {
					part_data[idx] = DayOfWeekOperator::DayOfWeekFromISO(isodow);
				}
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::ISODOW))) {
					part_data[idx] = isodow;
				}
			}

			// ISO calculations
			if (mask & ISO) {
				int32_t ww = 0;
				int32_t iyyy = 0;
				Date::ExtractISOYearWeek(input, iyyy, ww);
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::WEEK))) {
					part_data[idx] = ww;
				}
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::ISOYEAR))) {
					part_data[idx] = iyyy;
				}
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::YEARWEEK))) {
					part_data[idx] = YearWeekOperator::YearWeekFromParts(iyyy, ww);
				}
			}

			if (mask & EPOCH) {
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::EPOCH))) {
					part_data[idx] = Date::Epoch(input);
				}
			}
			if (mask & DOY) {
				if ((part_data = HasPartValue(part_values, DatePartSpecifier::DOY))) {
					part_data[idx] = Date::ExtractDayOfTheYear(input);
				}
			}
		}
	};
};

template <class T>
static void LastYearFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	int32_t last_year = 0;
	UnaryExecutor::ExecuteWithNulls<T, int64_t>(args.data[0], result, args.size(),
	                                            [&](T input, ValidityMask &mask, idx_t idx) {
		                                            if (Value::IsFinite(input)) {
			                                            return Date::ExtractYear(input, &last_year);
		                                            } else {
			                                            mask.SetInvalid(idx);
			                                            return 0;
		                                            }
	                                            });
}

template <>
int64_t DatePart::YearOperator::Operation(timestamp_t input) {
	return YearOperator::Operation<date_t, int64_t>(Timestamp::GetDate(input));
}

template <>
int64_t DatePart::YearOperator::Operation(interval_t input) {
	return input.months / Interval::MONTHS_PER_YEAR;
}

template <>
int64_t DatePart::YearOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"year\" not recognized");
}

template <>
int64_t DatePart::MonthOperator::Operation(timestamp_t input) {
	return MonthOperator::Operation<date_t, int64_t>(Timestamp::GetDate(input));
}

template <>
int64_t DatePart::MonthOperator::Operation(interval_t input) {
	return input.months % Interval::MONTHS_PER_YEAR;
}

template <>
int64_t DatePart::MonthOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"month\" not recognized");
}

template <>
int64_t DatePart::DayOperator::Operation(timestamp_t input) {
	return DayOperator::Operation<date_t, int64_t>(Timestamp::GetDate(input));
}

template <>
int64_t DatePart::DayOperator::Operation(interval_t input) {
	return input.days;
}

template <>
int64_t DatePart::DayOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"day\" not recognized");
}

template <>
int64_t DatePart::DecadeOperator::Operation(interval_t input) {
	return input.months / Interval::MONTHS_PER_DECADE;
}

template <>
int64_t DatePart::DecadeOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"decade\" not recognized");
}

template <>
int64_t DatePart::CenturyOperator::Operation(interval_t input) {
	return input.months / Interval::MONTHS_PER_CENTURY;
}

template <>
int64_t DatePart::CenturyOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"century\" not recognized");
}

template <>
int64_t DatePart::MillenniumOperator::Operation(interval_t input) {
	return input.months / Interval::MONTHS_PER_MILLENIUM;
}

template <>
int64_t DatePart::MillenniumOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"millennium\" not recognized");
}

template <>
int64_t DatePart::QuarterOperator::Operation(timestamp_t input) {
	return QuarterOperator::Operation<date_t, int64_t>(Timestamp::GetDate(input));
}

template <>
int64_t DatePart::QuarterOperator::Operation(interval_t input) {
	return MonthOperator::Operation<interval_t, int64_t>(input) / Interval::MONTHS_PER_QUARTER + 1;
}

template <>
int64_t DatePart::QuarterOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"quarter\" not recognized");
}

template <>
int64_t DatePart::DayOfWeekOperator::Operation(timestamp_t input) {
	return DayOfWeekOperator::Operation<date_t, int64_t>(Timestamp::GetDate(input));
}

template <>
int64_t DatePart::DayOfWeekOperator::Operation(interval_t input) {
	throw NotImplementedException("interval units \"dow\" not recognized");
}

template <>
int64_t DatePart::DayOfWeekOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"dow\" not recognized");
}

template <>
int64_t DatePart::ISODayOfWeekOperator::Operation(timestamp_t input) {
	return ISODayOfWeekOperator::Operation<date_t, int64_t>(Timestamp::GetDate(input));
}

template <>
int64_t DatePart::ISODayOfWeekOperator::Operation(interval_t input) {
	throw NotImplementedException("interval units \"isodow\" not recognized");
}

template <>
int64_t DatePart::ISODayOfWeekOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"isodow\" not recognized");
}

template <>
int64_t DatePart::DayOfYearOperator::Operation(timestamp_t input) {
	return DayOfYearOperator::Operation<date_t, int64_t>(Timestamp::GetDate(input));
}

template <>
int64_t DatePart::DayOfYearOperator::Operation(interval_t input) {
	throw NotImplementedException("interval units \"doy\" not recognized");
}

template <>
int64_t DatePart::DayOfYearOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"doy\" not recognized");
}

template <>
int64_t DatePart::WeekOperator::Operation(timestamp_t input) {
	return WeekOperator::Operation<date_t, int64_t>(Timestamp::GetDate(input));
}

template <>
int64_t DatePart::WeekOperator::Operation(interval_t input) {
	throw NotImplementedException("interval units \"week\" not recognized");
}

template <>
int64_t DatePart::WeekOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"week\" not recognized");
}

template <>
int64_t DatePart::ISOYearOperator::Operation(timestamp_t input) {
	return ISOYearOperator::Operation<date_t, int64_t>(Timestamp::GetDate(input));
}

template <>
int64_t DatePart::ISOYearOperator::Operation(interval_t input) {
	throw NotImplementedException("interval units \"isoyear\" not recognized");
}

template <>
int64_t DatePart::ISOYearOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"isoyear\" not recognized");
}

template <>
int64_t DatePart::YearWeekOperator::Operation(timestamp_t input) {
	return YearWeekOperator::Operation<date_t, int64_t>(Timestamp::GetDate(input));
}

template <>
int64_t DatePart::YearWeekOperator::Operation(interval_t input) {
	const auto yyyy = YearOperator::Operation<interval_t, int64_t>(input);
	const auto ww = WeekOperator::Operation<interval_t, int64_t>(input);
	return YearWeekOperator::YearWeekFromParts<int64_t>(yyyy, ww);
}

template <>
int64_t DatePart::YearWeekOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"yearweek\" not recognized");
}

template <>
int64_t DatePart::MicrosecondsOperator::Operation(timestamp_t input) {
	auto time = Timestamp::GetTime(input);
	// remove everything but the second & microsecond part
	return time.micros % Interval::MICROS_PER_MINUTE;
}

template <>
int64_t DatePart::MicrosecondsOperator::Operation(interval_t input) {
	// remove everything but the second & microsecond part
	return input.micros % Interval::MICROS_PER_MINUTE;
}

template <>
int64_t DatePart::MicrosecondsOperator::Operation(dtime_t input) {
	// remove everything but the second & microsecond part
	return input.micros % Interval::MICROS_PER_MINUTE;
}

template <>
int64_t DatePart::MillisecondsOperator::Operation(timestamp_t input) {
	return MicrosecondsOperator::Operation<timestamp_t, int64_t>(input) / Interval::MICROS_PER_MSEC;
}

template <>
int64_t DatePart::MillisecondsOperator::Operation(interval_t input) {
	return MicrosecondsOperator::Operation<interval_t, int64_t>(input) / Interval::MICROS_PER_MSEC;
}

template <>
int64_t DatePart::MillisecondsOperator::Operation(dtime_t input) {
	return MicrosecondsOperator::Operation<dtime_t, int64_t>(input) / Interval::MICROS_PER_MSEC;
}

template <>
int64_t DatePart::SecondsOperator::Operation(timestamp_t input) {
	return MicrosecondsOperator::Operation<timestamp_t, int64_t>(input) / Interval::MICROS_PER_SEC;
}

template <>
int64_t DatePart::SecondsOperator::Operation(interval_t input) {
	return MicrosecondsOperator::Operation<interval_t, int64_t>(input) / Interval::MICROS_PER_SEC;
}

template <>
int64_t DatePart::SecondsOperator::Operation(dtime_t input) {
	return MicrosecondsOperator::Operation<dtime_t, int64_t>(input) / Interval::MICROS_PER_SEC;
}

template <>
int64_t DatePart::MinutesOperator::Operation(timestamp_t input) {
	auto time = Timestamp::GetTime(input);
	// remove the hour part, and truncate to minutes
	return (time.micros % Interval::MICROS_PER_HOUR) / Interval::MICROS_PER_MINUTE;
}

template <>
int64_t DatePart::MinutesOperator::Operation(interval_t input) {
	// remove the hour part, and truncate to minutes
	return (input.micros % Interval::MICROS_PER_HOUR) / Interval::MICROS_PER_MINUTE;
}

template <>
int64_t DatePart::MinutesOperator::Operation(dtime_t input) {
	// remove the hour part, and truncate to minutes
	return (input.micros % Interval::MICROS_PER_HOUR) / Interval::MICROS_PER_MINUTE;
}

template <>
int64_t DatePart::HoursOperator::Operation(timestamp_t input) {
	return Timestamp::GetTime(input).micros / Interval::MICROS_PER_HOUR;
}

template <>
int64_t DatePart::HoursOperator::Operation(interval_t input) {
	return input.micros / Interval::MICROS_PER_HOUR;
}

template <>
int64_t DatePart::HoursOperator::Operation(dtime_t input) {
	return input.micros / Interval::MICROS_PER_HOUR;
}

template <>
int64_t DatePart::EpochOperator::Operation(timestamp_t input) {
	return Timestamp::GetEpochSeconds(input);
}

template <>
int64_t DatePart::EpochOperator::Operation(interval_t input) {
	int64_t interval_years = input.months / Interval::MONTHS_PER_YEAR;
	int64_t interval_days;
	interval_days = Interval::DAYS_PER_YEAR * interval_years;
	interval_days += Interval::DAYS_PER_MONTH * (input.months % Interval::MONTHS_PER_YEAR);
	interval_days += input.days;
	int64_t interval_epoch;
	interval_epoch = interval_days * Interval::SECS_PER_DAY;
	// we add 0.25 days per year to sort of account for leap days
	interval_epoch += interval_years * (Interval::SECS_PER_DAY / 4);
	interval_epoch += input.micros / Interval::MICROS_PER_SEC;
	return interval_epoch;
}

template <>
int64_t DatePart::EpochOperator::Operation(dtime_t input) {
	return input.micros / Interval::MICROS_PER_SEC;
}

template <>
unique_ptr<BaseStatistics> DatePart::EpochOperator::PropagateStatistics<dtime_t>(ClientContext &context,
                                                                                 FunctionStatisticsInput &input) {
	// time seconds range over a single day
	return PropagateSimpleDatePartStatistics<0, 86400>(input.child_stats);
}

template <>
int64_t DatePart::EraOperator::Operation(timestamp_t input) {
	return EraOperator::Operation<date_t, int64_t>(Timestamp::GetDate(input));
}

template <>
int64_t DatePart::EraOperator::Operation(interval_t input) {
	throw NotImplementedException("interval units \"era\" not recognized");
}

template <>
int64_t DatePart::EraOperator::Operation(dtime_t input) {
	throw NotImplementedException("\"time\" units \"era\" not recognized");
}

template <>
int64_t DatePart::TimezoneOperator::Operation(date_t input) {
	throw NotImplementedException("\"date\" units \"timezone\" not recognized");
}

template <>
int64_t DatePart::TimezoneOperator::Operation(interval_t input) {
	throw NotImplementedException("\"interval\" units \"timezone\" not recognized");
}

template <>
int64_t DatePart::TimezoneOperator::Operation(dtime_t input) {
	return 0;
}

template <>
void DatePart::StructOperator::Operation(int64_t **part_values, const dtime_t &input, const idx_t idx,
                                         const part_mask_t mask) {
	int64_t *part_data;
	if (mask & TIME) {
		const auto micros = MicrosecondsOperator::Operation<dtime_t, int64_t>(input);
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::MICROSECONDS))) {
			part_data[idx] = micros;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::MILLISECONDS))) {
			part_data[idx] = micros / Interval::MICROS_PER_MSEC;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::SECOND))) {
			part_data[idx] = micros / Interval::MICROS_PER_SEC;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::MINUTE))) {
			part_data[idx] = MinutesOperator::Operation<dtime_t, int64_t>(input);
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::HOUR))) {
			part_data[idx] = HoursOperator::Operation<dtime_t, int64_t>(input);
		}
	}

	if (mask & EPOCH) {
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::EPOCH))) {
			part_data[idx] = EpochOperator::Operation<dtime_t, int64_t>(input);
			;
		}
	}

	if (mask & ZONE) {
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::TIMEZONE))) {
			part_data[idx] = 0;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::TIMEZONE_HOUR))) {
			part_data[idx] = 0;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::TIMEZONE_MINUTE))) {
			part_data[idx] = 0;
		}
	}
}

template <>
void DatePart::StructOperator::Operation(int64_t **part_values, const timestamp_t &input, const idx_t idx,
                                         const part_mask_t mask) {
	date_t d;
	dtime_t t;
	Timestamp::Convert(input, d, t);

	// Both define epoch, and the correct value is the sum.
	// So mask it out and compute it separately.
	Operation(part_values, d, idx, mask & ~EPOCH);
	Operation(part_values, t, idx, mask & ~EPOCH);

	if (mask & EPOCH) {
		auto part_data = HasPartValue(part_values, DatePartSpecifier::EPOCH);
		if (part_data) {
			part_data[idx] = EpochOperator::Operation<timestamp_t, int64_t>(input);
		}
	}
}

template <>
void DatePart::StructOperator::Operation(int64_t **part_values, const interval_t &input, const idx_t idx,
                                         const part_mask_t mask) {
	int64_t *part_data;
	if (mask & YMD) {
		const auto mm = input.months % Interval::MONTHS_PER_YEAR;
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::YEAR))) {
			part_data[idx] = input.months / Interval::MONTHS_PER_YEAR;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::MONTH))) {
			part_data[idx] = mm;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::DAY))) {
			part_data[idx] = input.days;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::DECADE))) {
			part_data[idx] = input.months / Interval::MONTHS_PER_DECADE;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::CENTURY))) {
			part_data[idx] = input.months / Interval::MONTHS_PER_CENTURY;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::MILLENNIUM))) {
			part_data[idx] = input.months / Interval::MONTHS_PER_MILLENIUM;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::QUARTER))) {
			part_data[idx] = mm / Interval::MONTHS_PER_QUARTER + 1;
		}
	}

	if (mask & TIME) {
		const auto micros = MicrosecondsOperator::Operation<interval_t, int64_t>(input);
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::MICROSECONDS))) {
			part_data[idx] = micros;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::MILLISECONDS))) {
			part_data[idx] = micros / Interval::MICROS_PER_MSEC;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::SECOND))) {
			part_data[idx] = micros / Interval::MICROS_PER_SEC;
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::MINUTE))) {
			part_data[idx] = MinutesOperator::Operation<interval_t, int64_t>(input);
		}
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::HOUR))) {
			part_data[idx] = HoursOperator::Operation<interval_t, int64_t>(input);
		}
	}

	if (mask & EPOCH) {
		if ((part_data = HasPartValue(part_values, DatePartSpecifier::EPOCH))) {
			part_data[idx] = EpochOperator::Operation<interval_t, int64_t>(input);
		}
	}
}

template <typename T>
static int64_t ExtractElement(DatePartSpecifier type, T element) {
	switch (type) {
	case DatePartSpecifier::YEAR:
		return DatePart::YearOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::MONTH:
		return DatePart::MonthOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::DAY:
		return DatePart::DayOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::DECADE:
		return DatePart::DecadeOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::CENTURY:
		return DatePart::CenturyOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::MILLENNIUM:
		return DatePart::MillenniumOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::QUARTER:
		return DatePart::QuarterOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::DOW:
		return DatePart::DayOfWeekOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::ISODOW:
		return DatePart::ISODayOfWeekOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::DOY:
		return DatePart::DayOfYearOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::WEEK:
		return DatePart::WeekOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::ISOYEAR:
		return DatePart::ISOYearOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::YEARWEEK:
		return DatePart::YearWeekOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::EPOCH:
		return DatePart::EpochOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::MICROSECONDS:
		return DatePart::MicrosecondsOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::MILLISECONDS:
		return DatePart::MillisecondsOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::SECOND:
		return DatePart::SecondsOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::MINUTE:
		return DatePart::MinutesOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::HOUR:
		return DatePart::HoursOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::ERA:
		return DatePart::EraOperator::template Operation<T, int64_t>(element);
	case DatePartSpecifier::TIMEZONE:
	case DatePartSpecifier::TIMEZONE_HOUR:
	case DatePartSpecifier::TIMEZONE_MINUTE:
		return DatePart::TimezoneOperator::template Operation<T, int64_t>(element);
	default:
		throw NotImplementedException("Specifier type not implemented for DATEPART");
	}
}

template <typename T>
static void DatePartFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 2);
	auto &spec_arg = args.data[0];
	auto &date_arg = args.data[1];

	BinaryExecutor::ExecuteWithNulls<string_t, T, int64_t>(
	    spec_arg, date_arg, result, args.size(), [&](string_t specifier, T date, ValidityMask &mask, idx_t idx) {
		    if (Value::IsFinite(date)) {
			    return ExtractElement<T>(GetDatePartSpecifier(specifier.GetString()), date);
		    } else {
			    mask.SetInvalid(idx);
			    return int64_t(0);
		    }
	    });
}

void AddGenericDatePartOperator(BuiltinFunctions &set, const string &name, scalar_function_t date_func,
                                scalar_function_t ts_func, scalar_function_t interval_func,
                                function_statistics_t date_stats, function_statistics_t ts_stats) {
	ScalarFunctionSet operator_set(name);
	operator_set.AddFunction(
	    ScalarFunction({LogicalType::DATE}, LogicalType::BIGINT, std::move(date_func), nullptr, nullptr, date_stats));
	operator_set.AddFunction(
	    ScalarFunction({LogicalType::TIMESTAMP}, LogicalType::BIGINT, std::move(ts_func), nullptr, nullptr, ts_stats));
	operator_set.AddFunction(ScalarFunction({LogicalType::INTERVAL}, LogicalType::BIGINT, std::move(interval_func)));
	set.AddFunction(operator_set);
}

template <class OP>
static void AddDatePartOperator(BuiltinFunctions &set, string name) {
	AddGenericDatePartOperator(set, name, DatePart::UnaryFunction<date_t, int64_t, OP>,
	                           DatePart::UnaryFunction<timestamp_t, int64_t, OP>,
	                           ScalarFunction::UnaryFunction<interval_t, int64_t, OP>,
	                           OP::template PropagateStatistics<date_t>, OP::template PropagateStatistics<timestamp_t>);
}

void AddGenericTimePartOperator(BuiltinFunctions &set, const string &name, scalar_function_t date_func,
                                scalar_function_t ts_func, scalar_function_t interval_func, scalar_function_t time_func,
                                function_statistics_t date_stats, function_statistics_t ts_stats,
                                function_statistics_t time_stats) {
	ScalarFunctionSet operator_set(name);
	operator_set.AddFunction(
	    ScalarFunction({LogicalType::DATE}, LogicalType::BIGINT, std::move(date_func), nullptr, nullptr, date_stats));
	operator_set.AddFunction(
	    ScalarFunction({LogicalType::TIMESTAMP}, LogicalType::BIGINT, std::move(ts_func), nullptr, nullptr, ts_stats));
	operator_set.AddFunction(ScalarFunction({LogicalType::INTERVAL}, LogicalType::BIGINT, std::move(interval_func)));
	operator_set.AddFunction(
	    ScalarFunction({LogicalType::TIME}, LogicalType::BIGINT, std::move(time_func), nullptr, nullptr, time_stats));
	set.AddFunction(operator_set);
}

template <class OP>
static void AddTimePartOperator(BuiltinFunctions &set, string name) {
	AddGenericTimePartOperator(
	    set, name, DatePart::UnaryFunction<date_t, int64_t, OP>, DatePart::UnaryFunction<timestamp_t, int64_t, OP>,
	    ScalarFunction::UnaryFunction<interval_t, int64_t, OP>, ScalarFunction::UnaryFunction<dtime_t, int64_t, OP>,
	    OP::template PropagateStatistics<date_t>, OP::template PropagateStatistics<timestamp_t>,
	    OP::template PropagateStatistics<dtime_t>);
}

struct LastDayOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		int32_t yyyy, mm, dd;
		Date::Convert(input, yyyy, mm, dd);
		yyyy += (mm / 12);
		mm %= 12;
		++mm;
		return Date::FromDate(yyyy, mm, 1) - 1;
	}
};

template <>
date_t LastDayOperator::Operation(timestamp_t input) {
	return LastDayOperator::Operation<date_t, date_t>(Timestamp::GetDate(input));
}

struct MonthNameOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return Date::MONTH_NAMES[DatePart::MonthOperator::Operation<TA, int64_t>(input) - 1];
	}
};

struct DayNameOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return Date::DAY_NAMES[DatePart::DayOfWeekOperator::Operation<TA, int64_t>(input)];
	}
};

struct StructDatePart {
	using part_codes_t = vector<DatePartSpecifier>;

	struct BindData : public VariableReturnBindData {
		part_codes_t part_codes;

		explicit BindData(const LogicalType &stype, const part_codes_t &part_codes_p)
		    : VariableReturnBindData(stype), part_codes(part_codes_p) {
		}

		unique_ptr<FunctionData> Copy() const override {
			return make_unique<BindData>(stype, part_codes);
		}
	};

	static unique_ptr<FunctionData> Bind(ClientContext &context, ScalarFunction &bound_function,
	                                     vector<unique_ptr<Expression>> &arguments) {
		// collect names and deconflict, construct return type
		if (arguments[0]->HasParameter()) {
			throw ParameterNotResolvedException();
		}
		if (!arguments[0]->IsFoldable()) {
			throw BinderException("%s can only take constant lists of part names", bound_function.name);
		}

		case_insensitive_set_t name_collision_set;
		child_list_t<LogicalType> struct_children;
		part_codes_t part_codes;

		Value parts_list = ExpressionExecutor::EvaluateScalar(context, *arguments[0]);
		if (parts_list.type().id() == LogicalTypeId::LIST) {
			auto &list_children = ListValue::GetChildren(parts_list);
			if (list_children.empty()) {
				throw BinderException("%s requires non-empty lists of part names", bound_function.name);
			}
			for (const auto &part_value : list_children) {
				if (part_value.IsNull()) {
					throw BinderException("NULL struct entry name in %s", bound_function.name);
				}
				const auto part_name = part_value.ToString();
				const auto part_code = GetDateTypePartSpecifier(part_name, arguments[1]->return_type);
				if (name_collision_set.find(part_name) != name_collision_set.end()) {
					throw BinderException("Duplicate struct entry name \"%s\" in %s", part_name, bound_function.name);
				}
				name_collision_set.insert(part_name);
				part_codes.emplace_back(part_code);
				struct_children.emplace_back(make_pair(part_name, LogicalType::BIGINT));
			}
		} else {
			throw BinderException("%s can only take constant lists of part names", bound_function.name);
		}

		Function::EraseArgument(bound_function, arguments, 0);
		bound_function.return_type = LogicalType::STRUCT(std::move(struct_children));
		return make_unique<BindData>(bound_function.return_type, part_codes);
	}

	template <typename INPUT_TYPE>
	static void Function(DataChunk &args, ExpressionState &state, Vector &result) {
		auto &func_expr = (BoundFunctionExpression &)state.expr;
		auto &info = (BindData &)*func_expr.bind_info;
		D_ASSERT(args.ColumnCount() == 1);

		const auto count = args.size();
		Vector &input = args.data[0];
		vector<int64_t *> part_values(int(DatePartSpecifier::TIMEZONE_MINUTE) + 1, nullptr);
		const auto part_mask = DatePart::StructOperator::GetMask(info.part_codes);

		auto &child_entries = StructVector::GetEntries(result);

		// The first computer of a part "owns" it
		// and other requestors just reference the owner
		vector<size_t> owners(int(DatePartSpecifier::TIMEZONE_MINUTE) + 1, child_entries.size());
		for (size_t col = 0; col < child_entries.size(); ++col) {
			const auto part_index = size_t(info.part_codes[col]);
			if (owners[part_index] == child_entries.size()) {
				owners[part_index] = col;
			}
		}

		if (input.GetVectorType() == VectorType::CONSTANT_VECTOR) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);

			if (ConstantVector::IsNull(input)) {
				ConstantVector::SetNull(result, true);
			} else {
				ConstantVector::SetNull(result, false);
				for (size_t col = 0; col < child_entries.size(); ++col) {
					auto &child_entry = child_entries[col];
					ConstantVector::SetNull(*child_entry, false);
					const auto part_index = size_t(info.part_codes[col]);
					if (owners[part_index] == col) {
						part_values[part_index] = ConstantVector::GetData<int64_t>(*child_entry);
					}
				}
				auto tdata = ConstantVector::GetData<INPUT_TYPE>(input);
				if (Value::IsFinite(tdata[0])) {
					DatePart::StructOperator::Operation(part_values.data(), tdata[0], 0, part_mask);
				} else {
					for (auto &child_entry : child_entries) {
						ConstantVector::SetNull(*child_entry, true);
					}
				}
			}
		} else {
			UnifiedVectorFormat rdata;
			input.ToUnifiedFormat(count, rdata);

			const auto &arg_valid = rdata.validity;
			auto tdata = (const INPUT_TYPE *)rdata.data;

			// Start with a valid flat vector
			result.SetVectorType(VectorType::FLAT_VECTOR);
			auto &res_valid = FlatVector::Validity(result);
			if (res_valid.GetData()) {
				res_valid.SetAllValid(count);
			}

			// Start with valid children
			for (size_t col = 0; col < child_entries.size(); ++col) {
				auto &child_entry = child_entries[col];
				child_entry->SetVectorType(VectorType::FLAT_VECTOR);
				auto &child_validity = FlatVector::Validity(*child_entry);
				if (child_validity.GetData()) {
					child_validity.SetAllValid(count);
				}

				// Pre-multiplex
				const auto part_index = size_t(info.part_codes[col]);
				if (owners[part_index] == col) {
					part_values[part_index] = FlatVector::GetData<int64_t>(*child_entry);
				}
			}

			for (idx_t i = 0; i < count; ++i) {
				const auto idx = rdata.sel->get_index(i);
				if (arg_valid.RowIsValid(idx)) {
					if (Value::IsFinite(tdata[idx])) {
						DatePart::StructOperator::Operation(part_values.data(), tdata[idx], i, part_mask);
					} else {
						for (auto &child_entry : child_entries) {
							FlatVector::Validity(*child_entry).SetInvalid(i);
						}
					}
				} else {
					res_valid.SetInvalid(i);
					for (auto &child_entry : child_entries) {
						FlatVector::Validity(*child_entry).SetInvalid(i);
					}
				}
			}
		}

		// Reference any duplicate parts
		for (size_t col = 0; col < child_entries.size(); ++col) {
			const auto part_index = size_t(info.part_codes[col]);
			const auto owner = owners[part_index];
			if (owner != col) {
				child_entries[col]->Reference(*child_entries[owner]);
			}
		}

		result.Verify(count);
	}

	static void SerializeFunction(FieldWriter &writer, const FunctionData *bind_data_p,
	                              const ScalarFunction &function) {
		D_ASSERT(bind_data_p);
		auto &info = (BindData &)*bind_data_p;
		writer.WriteSerializable(info.stype);
		writer.WriteList<DatePartSpecifier>(info.part_codes);
	}

	static unique_ptr<FunctionData> DeserializeFunction(ClientContext &context, FieldReader &reader,
	                                                    ScalarFunction &bound_function) {
		auto stype = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
		auto part_codes = reader.ReadRequiredList<DatePartSpecifier>();
		return make_unique<BindData>(std::move(stype), std::move(part_codes));
	}

	template <typename INPUT_TYPE>
	static ScalarFunction GetFunction(const LogicalType &temporal_type) {
		auto part_type = LogicalType::LIST(LogicalType::VARCHAR);
		auto result_type = LogicalType::STRUCT({});
		ScalarFunction result({part_type, temporal_type}, result_type, Function<INPUT_TYPE>, Bind);
		result.serialize = SerializeFunction;
		result.deserialize = DeserializeFunction;
		return result;
	}
};

void DatePartFun::RegisterFunction(BuiltinFunctions &set) {
	// register the individual operators
	AddGenericDatePartOperator(set, "year", LastYearFunction<date_t>, LastYearFunction<timestamp_t>,
	                           ScalarFunction::UnaryFunction<interval_t, int64_t, DatePart::YearOperator>,
	                           DatePart::YearOperator::PropagateStatistics<date_t>,
	                           DatePart::YearOperator::PropagateStatistics<timestamp_t>);
	AddDatePartOperator<DatePart::MonthOperator>(set, "month");
	AddDatePartOperator<DatePart::DayOperator>(set, "day");
	AddDatePartOperator<DatePart::DecadeOperator>(set, "decade");
	AddDatePartOperator<DatePart::CenturyOperator>(set, "century");
	AddDatePartOperator<DatePart::MillenniumOperator>(set, "millennium");
	AddDatePartOperator<DatePart::QuarterOperator>(set, "quarter");
	AddDatePartOperator<DatePart::DayOfWeekOperator>(set, "dayofweek");
	AddDatePartOperator<DatePart::ISODayOfWeekOperator>(set, "isodow");
	AddDatePartOperator<DatePart::DayOfYearOperator>(set, "dayofyear");
	AddDatePartOperator<DatePart::WeekOperator>(set, "week");
	AddDatePartOperator<DatePart::ISOYearOperator>(set, "isoyear");
	AddDatePartOperator<DatePart::EraOperator>(set, "era");
	AddDatePartOperator<DatePart::TimezoneOperator>(set, "timezone");
	AddDatePartOperator<DatePart::TimezoneHourOperator>(set, "timezone_hour");
	AddDatePartOperator<DatePart::TimezoneMinuteOperator>(set, "timezone_minute");
	AddTimePartOperator<DatePart::EpochOperator>(set, "epoch");
	AddTimePartOperator<DatePart::MicrosecondsOperator>(set, "microsecond");
	AddTimePartOperator<DatePart::MillisecondsOperator>(set, "millisecond");
	AddTimePartOperator<DatePart::SecondsOperator>(set, "second");
	AddTimePartOperator<DatePart::MinutesOperator>(set, "minute");
	AddTimePartOperator<DatePart::HoursOperator>(set, "hour");

	//  register combinations
	AddDatePartOperator<DatePart::YearWeekOperator>(set, "yearweek");

	//  register various aliases
	AddDatePartOperator<DatePart::DayOperator>(set, "dayofmonth");
	AddDatePartOperator<DatePart::DayOfWeekOperator>(set, "weekday");
	AddDatePartOperator<DatePart::WeekOperator>(set, "weekofyear"); //  Note that WeekOperator is ISO-8601, not US

	//  register the last_day function
	ScalarFunctionSet last_day("last_day");
	last_day.AddFunction(ScalarFunction({LogicalType::DATE}, LogicalType::DATE,
	                                    DatePart::UnaryFunction<date_t, date_t, LastDayOperator>));
	last_day.AddFunction(ScalarFunction({LogicalType::TIMESTAMP}, LogicalType::DATE,
	                                    DatePart::UnaryFunction<timestamp_t, date_t, LastDayOperator>));
	set.AddFunction(last_day);

	//  register the monthname function
	ScalarFunctionSet monthname("monthname");
	monthname.AddFunction(ScalarFunction({LogicalType::DATE}, LogicalType::VARCHAR,
	                                     DatePart::UnaryFunction<date_t, string_t, MonthNameOperator>));
	monthname.AddFunction(ScalarFunction({LogicalType::TIMESTAMP}, LogicalType::VARCHAR,
	                                     DatePart::UnaryFunction<timestamp_t, string_t, MonthNameOperator>));
	set.AddFunction(monthname);

	//  register the dayname function
	ScalarFunctionSet dayname("dayname");
	dayname.AddFunction(ScalarFunction({LogicalType::DATE}, LogicalType::VARCHAR,
	                                   DatePart::UnaryFunction<date_t, string_t, DayNameOperator>));
	dayname.AddFunction(ScalarFunction({LogicalType::TIMESTAMP}, LogicalType::VARCHAR,
	                                   DatePart::UnaryFunction<timestamp_t, string_t, DayNameOperator>));
	set.AddFunction(dayname);

	// finally the actual date_part function
	ScalarFunctionSet date_part("date_part");
	date_part.AddFunction(
	    ScalarFunction({LogicalType::VARCHAR, LogicalType::DATE}, LogicalType::BIGINT, DatePartFunction<date_t>));
	date_part.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::TIMESTAMP}, LogicalType::BIGINT,
	                                     DatePartFunction<timestamp_t>));
	date_part.AddFunction(
	    ScalarFunction({LogicalType::VARCHAR, LogicalType::TIME}, LogicalType::BIGINT, DatePartFunction<dtime_t>));
	date_part.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::INTERVAL}, LogicalType::BIGINT,
	                                     DatePartFunction<interval_t>));

	// struct variants
	date_part.AddFunction(StructDatePart::GetFunction<date_t>(LogicalType::DATE));
	date_part.AddFunction(StructDatePart::GetFunction<timestamp_t>(LogicalType::TIMESTAMP));
	date_part.AddFunction(StructDatePart::GetFunction<dtime_t>(LogicalType::TIME));
	date_part.AddFunction(StructDatePart::GetFunction<interval_t>(LogicalType::INTERVAL));

	set.AddFunction(date_part);
	date_part.name = "datepart";
	set.AddFunction(date_part);
}

} // namespace duckdb













namespace duckdb {

struct DateSub {
	static int64_t SubtractMicros(timestamp_t startdate, timestamp_t enddate) {
		const auto start = Timestamp::GetEpochMicroSeconds(startdate);
		const auto end = Timestamp::GetEpochMicroSeconds(enddate);
		return SubtractOperatorOverflowCheck::Operation<int64_t, int64_t, int64_t>(end, start);
	}

	template <class TA, class TB, class TR, class OP>
	static inline void BinaryExecute(Vector &left, Vector &right, Vector &result, idx_t count) {
		BinaryExecutor::ExecuteWithNulls<TA, TB, TR>(
		    left, right, result, count, [&](TA startdate, TB enddate, ValidityMask &mask, idx_t idx) {
			    if (Value::IsFinite(startdate) && Value::IsFinite(enddate)) {
				    return OP::template Operation<TA, TB, TR>(startdate, enddate);
			    } else {
				    mask.SetInvalid(idx);
				    return TR();
			    }
		    });
	}

	struct MonthOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA start_ts, TB end_ts) {

			if (start_ts > end_ts) {
				return -MonthOperator::Operation<TA, TB, TR>(end_ts, start_ts);
			}
			// The number of complete months depends on whether end_ts is on the last day of the month.
			date_t end_date;
			dtime_t end_time;
			Timestamp::Convert(end_ts, end_date, end_time);

			int32_t yyyy, mm, dd;
			Date::Convert(end_date, yyyy, mm, dd);
			const auto end_days = Date::MonthDays(yyyy, mm);
			if (end_days == dd) {
				// Now check whether the start day is after the end day
				date_t start_date;
				dtime_t start_time;
				Timestamp::Convert(start_ts, start_date, start_time);
				Date::Convert(start_date, yyyy, mm, dd);
				if (dd > end_days || (dd == end_days && start_time < end_time)) {
					// Move back to the same time on the last day of the (shorter) end month
					start_date = Date::FromDate(yyyy, mm, end_days);
					start_ts = Timestamp::FromDatetime(start_date, start_time);
				}
			}

			// Our interval difference will now give the correct result.
			// Note that PG gives different interval subtraction results,
			// so if we change this we will have to reimplement.
			return Interval::GetAge(end_ts, start_ts).months;
		}
	};

	struct QuarterOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA start_ts, TB end_ts) {
			return MonthOperator::Operation<TA, TB, TR>(start_ts, end_ts) / Interval::MONTHS_PER_QUARTER;
		}
	};

	struct YearOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA start_ts, TB end_ts) {
			return MonthOperator::Operation<TA, TB, TR>(start_ts, end_ts) / Interval::MONTHS_PER_YEAR;
		}
	};

	struct DecadeOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA start_ts, TB end_ts) {
			return MonthOperator::Operation<TA, TB, TR>(start_ts, end_ts) / Interval::MONTHS_PER_DECADE;
		}
	};

	struct CenturyOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA start_ts, TB end_ts) {
			return MonthOperator::Operation<TA, TB, TR>(start_ts, end_ts) / Interval::MONTHS_PER_CENTURY;
		}
	};

	struct MilleniumOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA start_ts, TB end_ts) {
			return MonthOperator::Operation<TA, TB, TR>(start_ts, end_ts) / Interval::MONTHS_PER_MILLENIUM;
		}
	};

	struct DayOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return SubtractMicros(startdate, enddate) / Interval::MICROS_PER_DAY;
		}
	};

	struct WeekOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return SubtractMicros(startdate, enddate) / Interval::MICROS_PER_WEEK;
		}
	};

	struct MicrosecondsOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return SubtractMicros(startdate, enddate);
		}
	};

	struct MillisecondsOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return SubtractMicros(startdate, enddate) / Interval::MICROS_PER_MSEC;
		}
	};

	struct SecondsOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return SubtractMicros(startdate, enddate) / Interval::MICROS_PER_SEC;
		}
	};

	struct MinutesOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return SubtractMicros(startdate, enddate) / Interval::MICROS_PER_MINUTE;
		}
	};

	struct HoursOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA startdate, TB enddate) {
			return SubtractMicros(startdate, enddate) / Interval::MICROS_PER_HOUR;
		}
	};
};

// DATE specialisations
template <>
int64_t DateSub::YearOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return YearOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                  Timestamp::FromDatetime(enddate, t0));
}

template <>
int64_t DateSub::MonthOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return MonthOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                   Timestamp::FromDatetime(enddate, t0));
}

template <>
int64_t DateSub::DayOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return DayOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                 Timestamp::FromDatetime(enddate, t0));
}

template <>
int64_t DateSub::DecadeOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return DecadeOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                    Timestamp::FromDatetime(enddate, t0));
}

template <>
int64_t DateSub::CenturyOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return CenturyOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                     Timestamp::FromDatetime(enddate, t0));
}

template <>
int64_t DateSub::MilleniumOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return MilleniumOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                       Timestamp::FromDatetime(enddate, t0));
}

template <>
int64_t DateSub::QuarterOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return QuarterOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                     Timestamp::FromDatetime(enddate, t0));
}

template <>
int64_t DateSub::WeekOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return WeekOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                  Timestamp::FromDatetime(enddate, t0));
}

template <>
int64_t DateSub::MicrosecondsOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return MicrosecondsOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                          Timestamp::FromDatetime(enddate, t0));
}

template <>
int64_t DateSub::MillisecondsOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return MillisecondsOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                          Timestamp::FromDatetime(enddate, t0));
}

template <>
int64_t DateSub::SecondsOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return SecondsOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                     Timestamp::FromDatetime(enddate, t0));
}

template <>
int64_t DateSub::MinutesOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return MinutesOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                     Timestamp::FromDatetime(enddate, t0));
}

template <>
int64_t DateSub::HoursOperator::Operation(date_t startdate, date_t enddate) {
	dtime_t t0(0);
	return HoursOperator::Operation<timestamp_t, timestamp_t, int64_t>(Timestamp::FromDatetime(startdate, t0),
	                                                                   Timestamp::FromDatetime(enddate, t0));
}

// TIME specialisations
template <>
int64_t DateSub::YearOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"year\" not recognized");
}

template <>
int64_t DateSub::MonthOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"month\" not recognized");
}

template <>
int64_t DateSub::DayOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"day\" not recognized");
}

template <>
int64_t DateSub::DecadeOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"decade\" not recognized");
}

template <>
int64_t DateSub::CenturyOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"century\" not recognized");
}

template <>
int64_t DateSub::MilleniumOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"millennium\" not recognized");
}

template <>
int64_t DateSub::QuarterOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"quarter\" not recognized");
}

template <>
int64_t DateSub::WeekOperator::Operation(dtime_t startdate, dtime_t enddate) {
	throw NotImplementedException("\"time\" units \"week\" not recognized");
}

template <>
int64_t DateSub::MicrosecondsOperator::Operation(dtime_t startdate, dtime_t enddate) {
	return enddate.micros - startdate.micros;
}

template <>
int64_t DateSub::MillisecondsOperator::Operation(dtime_t startdate, dtime_t enddate) {
	return (enddate.micros - startdate.micros) / Interval::MICROS_PER_MSEC;
}

template <>
int64_t DateSub::SecondsOperator::Operation(dtime_t startdate, dtime_t enddate) {
	return (enddate.micros - startdate.micros) / Interval::MICROS_PER_SEC;
}

template <>
int64_t DateSub::MinutesOperator::Operation(dtime_t startdate, dtime_t enddate) {
	return (enddate.micros - startdate.micros) / Interval::MICROS_PER_MINUTE;
}

template <>
int64_t DateSub::HoursOperator::Operation(dtime_t startdate, dtime_t enddate) {
	return (enddate.micros - startdate.micros) / Interval::MICROS_PER_HOUR;
}

template <typename TA, typename TB, typename TR>
static int64_t SubtractDateParts(DatePartSpecifier type, TA startdate, TB enddate) {
	switch (type) {
	case DatePartSpecifier::YEAR:
	case DatePartSpecifier::ISOYEAR:
		return DateSub::YearOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::MONTH:
		return DateSub::MonthOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::DAY:
	case DatePartSpecifier::DOW:
	case DatePartSpecifier::ISODOW:
	case DatePartSpecifier::DOY:
		return DateSub::DayOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::DECADE:
		return DateSub::DecadeOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::CENTURY:
		return DateSub::CenturyOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::MILLENNIUM:
		return DateSub::MilleniumOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::QUARTER:
		return DateSub::QuarterOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::WEEK:
	case DatePartSpecifier::YEARWEEK:
		return DateSub::WeekOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::MICROSECONDS:
		return DateSub::MicrosecondsOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::MILLISECONDS:
		return DateSub::MillisecondsOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::SECOND:
	case DatePartSpecifier::EPOCH:
		return DateSub::SecondsOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::MINUTE:
		return DateSub::MinutesOperator::template Operation<TA, TB, TR>(startdate, enddate);
	case DatePartSpecifier::HOUR:
		return DateSub::HoursOperator::template Operation<TA, TB, TR>(startdate, enddate);
	default:
		throw NotImplementedException("Specifier type not implemented for DATESUB");
	}
}

struct DateSubTernaryOperator {
	template <typename TS, typename TA, typename TB, typename TR>
	static inline TR Operation(TS part, TA startdate, TB enddate, ValidityMask &mask, idx_t idx) {
		if (Value::IsFinite(startdate) && Value::IsFinite(enddate)) {
			return SubtractDateParts<TA, TB, TR>(GetDatePartSpecifier(part.GetString()), startdate, enddate);
		} else {
			mask.SetInvalid(idx);
			return TR();
		}
	}
};

template <typename TA, typename TB, typename TR>
static void DateSubBinaryExecutor(DatePartSpecifier type, Vector &left, Vector &right, Vector &result, idx_t count) {
	switch (type) {
	case DatePartSpecifier::YEAR:
	case DatePartSpecifier::ISOYEAR:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::YearOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::MONTH:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::MonthOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::DAY:
	case DatePartSpecifier::DOW:
	case DatePartSpecifier::ISODOW:
	case DatePartSpecifier::DOY:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::DayOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::DECADE:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::DecadeOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::CENTURY:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::CenturyOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::MILLENNIUM:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::MilleniumOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::QUARTER:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::QuarterOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::WEEK:
	case DatePartSpecifier::YEARWEEK:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::WeekOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::MICROSECONDS:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::MicrosecondsOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::MILLISECONDS:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::MillisecondsOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::SECOND:
	case DatePartSpecifier::EPOCH:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::SecondsOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::MINUTE:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::MinutesOperator>(left, right, result, count);
		break;
	case DatePartSpecifier::HOUR:
		DateSub::BinaryExecute<TA, TB, TR, DateSub::HoursOperator>(left, right, result, count);
		break;
	default:
		throw NotImplementedException("Specifier type not implemented for DATESUB");
	}
}

template <typename T>
static void DateSubFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 3);
	auto &part_arg = args.data[0];
	auto &start_arg = args.data[1];
	auto &end_arg = args.data[2];

	if (part_arg.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		// Common case of constant part.
		if (ConstantVector::IsNull(part_arg)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
		} else {
			const auto type = GetDatePartSpecifier(ConstantVector::GetData<string_t>(part_arg)->GetString());
			DateSubBinaryExecutor<T, T, int64_t>(type, start_arg, end_arg, result, args.size());
		}
	} else {
		TernaryExecutor::ExecuteWithNulls<string_t, T, T, int64_t>(
		    part_arg, start_arg, end_arg, result, args.size(),
		    DateSubTernaryOperator::Operation<string_t, T, T, int64_t>);
	}
}

void DateSubFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet date_sub("date_sub");
	date_sub.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::DATE, LogicalType::DATE},
	                                    LogicalType::BIGINT, DateSubFunction<date_t>));
	date_sub.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::TIMESTAMP, LogicalType::TIMESTAMP},
	                                    LogicalType::BIGINT, DateSubFunction<timestamp_t>));
	date_sub.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::TIME, LogicalType::TIME},
	                                    LogicalType::BIGINT, DateSubFunction<dtime_t>));
	set.AddFunction(date_sub);

	date_sub.name = "datesub";
	set.AddFunction(date_sub);
}

} // namespace duckdb












namespace duckdb {

struct DateTrunc {
	template <class TA, class TR, class OP>
	static inline TR UnaryFunction(TA input) {
		if (Value::IsFinite(input)) {
			return OP::template Operation<TA, TR>(input);
		} else {
			return Cast::template Operation<TA, TR>(input);
		}
	}

	template <class TA, class TR, class OP>
	static inline void UnaryExecute(Vector &left, Vector &result, idx_t count) {
		UnaryExecutor::Execute<TA, TR>(left, result, count, UnaryFunction<TA, TR, OP>);
	}

	struct MillenniumOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::FromDate((Date::ExtractYear(input) / 1000) * 1000, 1, 1);
		}
	};

	struct CenturyOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::FromDate((Date::ExtractYear(input) / 100) * 100, 1, 1);
		}
	};

	struct DecadeOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::FromDate((Date::ExtractYear(input) / 10) * 10, 1, 1);
		}
	};

	struct YearOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::FromDate(Date::ExtractYear(input), 1, 1);
		}
	};

	struct QuarterOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			int32_t yyyy, mm, dd;
			Date::Convert(input, yyyy, mm, dd);
			mm = 1 + (((mm - 1) / 3) * 3);
			return Date::FromDate(yyyy, mm, 1);
		}
	};

	struct MonthOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::FromDate(Date::ExtractYear(input), Date::ExtractMonth(input), 1);
		}
	};

	struct WeekOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return Date::GetMondayOfCurrentWeek(input);
		}
	};

	struct ISOYearOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			date_t date = Date::GetMondayOfCurrentWeek(input);
			date.days -= (Date::ExtractISOWeekNumber(date) - 1) * Interval::DAYS_PER_WEEK;

			return date;
		}
	};

	struct DayOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return input;
		}
	};

	struct HourOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			int32_t hour, min, sec, micros;
			date_t date;
			dtime_t time;
			Timestamp::Convert(input, date, time);
			Time::Convert(time, hour, min, sec, micros);
			return Timestamp::FromDatetime(date, Time::FromTime(hour, 0, 0, 0));
		}
	};

	struct MinuteOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			int32_t hour, min, sec, micros;
			date_t date;
			dtime_t time;
			Timestamp::Convert(input, date, time);
			Time::Convert(time, hour, min, sec, micros);
			return Timestamp::FromDatetime(date, Time::FromTime(hour, min, 0, 0));
		}
	};

	struct SecondOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			int32_t hour, min, sec, micros;
			date_t date;
			dtime_t time;
			Timestamp::Convert(input, date, time);
			Time::Convert(time, hour, min, sec, micros);
			return Timestamp::FromDatetime(date, Time::FromTime(hour, min, sec, 0));
		}
	};

	struct MillisecondOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			int32_t hour, min, sec, micros;
			date_t date;
			dtime_t time;
			Timestamp::Convert(input, date, time);
			Time::Convert(time, hour, min, sec, micros);
			micros -= micros % Interval::MICROS_PER_MSEC;
			return Timestamp::FromDatetime(date, Time::FromTime(hour, min, sec, micros));
		}
	};

	struct MicrosecondOperator {
		template <class TA, class TR>
		static inline TR Operation(TA input) {
			return input;
		}
	};
};

// DATE specialisations
template <>
date_t DateTrunc::MillenniumOperator::Operation(timestamp_t input) {
	return MillenniumOperator::Operation<date_t, date_t>(Timestamp::GetDate(input));
}

template <>
timestamp_t DateTrunc::MillenniumOperator::Operation(date_t input) {
	return Timestamp::FromDatetime(MillenniumOperator::Operation<date_t, date_t>(input), dtime_t(0));
}

template <>
timestamp_t DateTrunc::MillenniumOperator::Operation(timestamp_t input) {
	return MillenniumOperator::Operation<date_t, timestamp_t>(Timestamp::GetDate(input));
}

template <>
date_t DateTrunc::CenturyOperator::Operation(timestamp_t input) {
	return CenturyOperator::Operation<date_t, date_t>(Timestamp::GetDate(input));
}

template <>
timestamp_t DateTrunc::CenturyOperator::Operation(date_t input) {
	return Timestamp::FromDatetime(CenturyOperator::Operation<date_t, date_t>(input), dtime_t(0));
}

template <>
timestamp_t DateTrunc::CenturyOperator::Operation(timestamp_t input) {
	return CenturyOperator::Operation<date_t, timestamp_t>(Timestamp::GetDate(input));
}

template <>
date_t DateTrunc::DecadeOperator::Operation(timestamp_t input) {
	return DecadeOperator::Operation<date_t, date_t>(Timestamp::GetDate(input));
}

template <>
timestamp_t DateTrunc::DecadeOperator::Operation(date_t input) {
	return Timestamp::FromDatetime(DecadeOperator::Operation<date_t, date_t>(input), dtime_t(0));
}

template <>
timestamp_t DateTrunc::DecadeOperator::Operation(timestamp_t input) {
	return DecadeOperator::Operation<date_t, timestamp_t>(Timestamp::GetDate(input));
}

template <>
date_t DateTrunc::YearOperator::Operation(timestamp_t input) {
	return YearOperator::Operation<date_t, date_t>(Timestamp::GetDate(input));
}

template <>
timestamp_t DateTrunc::YearOperator::Operation(date_t input) {
	return Timestamp::FromDatetime(YearOperator::Operation<date_t, date_t>(input), dtime_t(0));
}

template <>
timestamp_t DateTrunc::YearOperator::Operation(timestamp_t input) {
	return YearOperator::Operation<date_t, timestamp_t>(Timestamp::GetDate(input));
}

template <>
date_t DateTrunc::QuarterOperator::Operation(timestamp_t input) {
	return QuarterOperator::Operation<date_t, date_t>(Timestamp::GetDate(input));
}

template <>
timestamp_t DateTrunc::QuarterOperator::Operation(date_t input) {
	return Timestamp::FromDatetime(QuarterOperator::Operation<date_t, date_t>(input), dtime_t(0));
}

template <>
timestamp_t DateTrunc::QuarterOperator::Operation(timestamp_t input) {
	return QuarterOperator::Operation<date_t, timestamp_t>(Timestamp::GetDate(input));
}

template <>
date_t DateTrunc::MonthOperator::Operation(timestamp_t input) {
	return MonthOperator::Operation<date_t, date_t>(Timestamp::GetDate(input));
}

template <>
timestamp_t DateTrunc::MonthOperator::Operation(date_t input) {
	return Timestamp::FromDatetime(MonthOperator::Operation<date_t, date_t>(input), dtime_t(0));
}

template <>
timestamp_t DateTrunc::MonthOperator::Operation(timestamp_t input) {
	return MonthOperator::Operation<date_t, timestamp_t>(Timestamp::GetDate(input));
}

template <>
date_t DateTrunc::WeekOperator::Operation(timestamp_t input) {
	return WeekOperator::Operation<date_t, date_t>(Timestamp::GetDate(input));
}

template <>
timestamp_t DateTrunc::WeekOperator::Operation(date_t input) {
	return Timestamp::FromDatetime(WeekOperator::Operation<date_t, date_t>(input), dtime_t(0));
}

template <>
timestamp_t DateTrunc::WeekOperator::Operation(timestamp_t input) {
	return WeekOperator::Operation<date_t, timestamp_t>(Timestamp::GetDate(input));
}

template <>
date_t DateTrunc::ISOYearOperator::Operation(timestamp_t input) {
	return ISOYearOperator::Operation<date_t, date_t>(Timestamp::GetDate(input));
}

template <>
timestamp_t DateTrunc::ISOYearOperator::Operation(date_t input) {
	return Timestamp::FromDatetime(ISOYearOperator::Operation<date_t, date_t>(input), dtime_t(0));
}

template <>
timestamp_t DateTrunc::ISOYearOperator::Operation(timestamp_t input) {
	return ISOYearOperator::Operation<date_t, timestamp_t>(Timestamp::GetDate(input));
}

template <>
date_t DateTrunc::DayOperator::Operation(timestamp_t input) {
	return DayOperator::Operation<date_t, date_t>(Timestamp::GetDate(input));
}

template <>
timestamp_t DateTrunc::DayOperator::Operation(date_t input) {
	return Timestamp::FromDatetime(DayOperator::Operation<date_t, date_t>(input), dtime_t(0));
}

template <>
timestamp_t DateTrunc::DayOperator::Operation(timestamp_t input) {
	return DayOperator::Operation<date_t, timestamp_t>(Timestamp::GetDate(input));
}

template <>
date_t DateTrunc::HourOperator::Operation(date_t input) {
	return DayOperator::Operation<date_t, date_t>(input);
}

template <>
timestamp_t DateTrunc::HourOperator::Operation(date_t input) {
	return DayOperator::Operation<date_t, timestamp_t>(input);
}

template <>
date_t DateTrunc::HourOperator::Operation(timestamp_t input) {
	return Timestamp::GetDate(HourOperator::Operation<timestamp_t, timestamp_t>(input));
}

template <>
date_t DateTrunc::MinuteOperator::Operation(date_t input) {
	return DayOperator::Operation<date_t, date_t>(input);
}

template <>
timestamp_t DateTrunc::MinuteOperator::Operation(date_t input) {
	return DayOperator::Operation<date_t, timestamp_t>(input);
}

template <>
date_t DateTrunc::MinuteOperator::Operation(timestamp_t input) {
	return Timestamp::GetDate(HourOperator::Operation<timestamp_t, timestamp_t>(input));
}

template <>
date_t DateTrunc::SecondOperator::Operation(date_t input) {
	return DayOperator::Operation<date_t, date_t>(input);
}

template <>
timestamp_t DateTrunc::SecondOperator::Operation(date_t input) {
	return DayOperator::Operation<date_t, timestamp_t>(input);
}

template <>
date_t DateTrunc::SecondOperator::Operation(timestamp_t input) {
	return Timestamp::GetDate(DayOperator::Operation<timestamp_t, timestamp_t>(input));
}

template <>
date_t DateTrunc::MillisecondOperator::Operation(date_t input) {
	return DayOperator::Operation<date_t, date_t>(input);
}

template <>
timestamp_t DateTrunc::MillisecondOperator::Operation(date_t input) {
	return DayOperator::Operation<date_t, timestamp_t>(input);
}

template <>
date_t DateTrunc::MillisecondOperator::Operation(timestamp_t input) {
	return Timestamp::GetDate(MillisecondOperator::Operation<timestamp_t, timestamp_t>(input));
}

template <>
date_t DateTrunc::MicrosecondOperator::Operation(date_t input) {
	return DayOperator::Operation<date_t, date_t>(input);
}

template <>
timestamp_t DateTrunc::MicrosecondOperator::Operation(date_t input) {
	return DayOperator::Operation<date_t, timestamp_t>(input);
}

template <>
date_t DateTrunc::MicrosecondOperator::Operation(timestamp_t input) {
	return Timestamp::GetDate(MicrosecondOperator::Operation<timestamp_t, timestamp_t>(input));
}

// INTERVAL specialisations
template <>
interval_t DateTrunc::MillenniumOperator::Operation(interval_t input) {
	input.days = 0;
	input.micros = 0;
	input.months = (input.months / Interval::MONTHS_PER_MILLENIUM) * Interval::MONTHS_PER_MILLENIUM;
	return input;
}

template <>
interval_t DateTrunc::CenturyOperator::Operation(interval_t input) {
	input.days = 0;
	input.micros = 0;
	input.months = (input.months / Interval::MONTHS_PER_CENTURY) * Interval::MONTHS_PER_CENTURY;
	return input;
}

template <>
interval_t DateTrunc::DecadeOperator::Operation(interval_t input) {
	input.days = 0;
	input.micros = 0;
	input.months = (input.months / Interval::MONTHS_PER_DECADE) * Interval::MONTHS_PER_DECADE;
	return input;
}

template <>
interval_t DateTrunc::YearOperator::Operation(interval_t input) {
	input.days = 0;
	input.micros = 0;
	input.months = (input.months / Interval::MONTHS_PER_YEAR) * Interval::MONTHS_PER_YEAR;
	return input;
}

template <>
interval_t DateTrunc::QuarterOperator::Operation(interval_t input) {
	input.days = 0;
	input.micros = 0;
	input.months = (input.months / Interval::MONTHS_PER_QUARTER) * Interval::MONTHS_PER_QUARTER;
	return input;
}

template <>
interval_t DateTrunc::MonthOperator::Operation(interval_t input) {
	input.days = 0;
	input.micros = 0;
	return input;
}

template <>
interval_t DateTrunc::WeekOperator::Operation(interval_t input) {
	input.micros = 0;
	input.days = (input.days / Interval::DAYS_PER_WEEK) * Interval::DAYS_PER_WEEK;
	return input;
}

template <>
interval_t DateTrunc::ISOYearOperator::Operation(interval_t input) {
	return YearOperator::Operation<interval_t, interval_t>(input);
}

template <>
interval_t DateTrunc::DayOperator::Operation(interval_t input) {
	input.micros = 0;
	return input;
}

template <>
interval_t DateTrunc::HourOperator::Operation(interval_t input) {
	input.micros = (input.micros / Interval::MICROS_PER_HOUR) * Interval::MICROS_PER_HOUR;
	return input;
}

template <>
interval_t DateTrunc::MinuteOperator::Operation(interval_t input) {
	input.micros = (input.micros / Interval::MICROS_PER_MINUTE) * Interval::MICROS_PER_MINUTE;
	return input;
}

template <>
interval_t DateTrunc::SecondOperator::Operation(interval_t input) {
	input.micros = (input.micros / Interval::MICROS_PER_SEC) * Interval::MICROS_PER_SEC;
	return input;
}

template <>
interval_t DateTrunc::MillisecondOperator::Operation(interval_t input) {
	input.micros = (input.micros / Interval::MICROS_PER_MSEC) * Interval::MICROS_PER_MSEC;
	return input;
}

template <>
interval_t DateTrunc::MicrosecondOperator::Operation(interval_t input) {
	return input;
}

template <class TA, class TR>
static TR TruncateElement(DatePartSpecifier type, TA element) {
	if (!Value::IsFinite(element)) {
		return Cast::template Operation<TA, TR>(element);
	}

	switch (type) {
	case DatePartSpecifier::MILLENNIUM:
		return DateTrunc::MillenniumOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::CENTURY:
		return DateTrunc::CenturyOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::DECADE:
		return DateTrunc::DecadeOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::YEAR:
		return DateTrunc::YearOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::QUARTER:
		return DateTrunc::QuarterOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::MONTH:
		return DateTrunc::MonthOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::WEEK:
	case DatePartSpecifier::YEARWEEK:
		return DateTrunc::WeekOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::ISOYEAR:
		return DateTrunc::ISOYearOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::DAY:
	case DatePartSpecifier::DOW:
	case DatePartSpecifier::ISODOW:
	case DatePartSpecifier::DOY:
		return DateTrunc::DayOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::HOUR:
		return DateTrunc::HourOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::MINUTE:
		return DateTrunc::MinuteOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::SECOND:
	case DatePartSpecifier::EPOCH:
		return DateTrunc::SecondOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::MILLISECONDS:
		return DateTrunc::MillisecondOperator::Operation<TA, TR>(element);
	case DatePartSpecifier::MICROSECONDS:
		return DateTrunc::MicrosecondOperator::Operation<TA, TR>(element);
	default:
		throw NotImplementedException("Specifier type not implemented for DATETRUNC");
	}
}

struct DateTruncBinaryOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA specifier, TB date) {
		return TruncateElement<TB, TR>(GetDatePartSpecifier(specifier.GetString()), date);
	}
};

template <typename TA, typename TR>
static void DateTruncUnaryExecutor(DatePartSpecifier type, Vector &left, Vector &result, idx_t count) {
	switch (type) {
	case DatePartSpecifier::MILLENNIUM:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::MillenniumOperator>(left, result, count);
		break;
	case DatePartSpecifier::CENTURY:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::CenturyOperator>(left, result, count);
		break;
	case DatePartSpecifier::DECADE:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::DecadeOperator>(left, result, count);
		break;
	case DatePartSpecifier::YEAR:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::YearOperator>(left, result, count);
		break;
	case DatePartSpecifier::QUARTER:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::QuarterOperator>(left, result, count);
		break;
	case DatePartSpecifier::MONTH:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::MonthOperator>(left, result, count);
		break;
	case DatePartSpecifier::WEEK:
	case DatePartSpecifier::YEARWEEK:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::WeekOperator>(left, result, count);
		break;
	case DatePartSpecifier::ISOYEAR:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::ISOYearOperator>(left, result, count);
		break;
	case DatePartSpecifier::DAY:
	case DatePartSpecifier::DOW:
	case DatePartSpecifier::ISODOW:
	case DatePartSpecifier::DOY:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::DayOperator>(left, result, count);
		break;
	case DatePartSpecifier::HOUR:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::HourOperator>(left, result, count);
		break;
	case DatePartSpecifier::MINUTE:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::MinuteOperator>(left, result, count);
		break;
	case DatePartSpecifier::SECOND:
	case DatePartSpecifier::EPOCH:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::SecondOperator>(left, result, count);
		break;
	case DatePartSpecifier::MILLISECONDS:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::MillisecondOperator>(left, result, count);
		break;
	case DatePartSpecifier::MICROSECONDS:
		DateTrunc::UnaryExecute<TA, TR, DateTrunc::MicrosecondOperator>(left, result, count);
		break;
	default:
		throw NotImplementedException("Specifier type not implemented for DATETRUNC");
	}
}

template <typename TA, typename TR>
static void DateTruncFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 2);
	auto &part_arg = args.data[0];
	auto &date_arg = args.data[1];

	if (part_arg.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		// Common case of constant part.
		if (ConstantVector::IsNull(part_arg)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
		} else {
			const auto type = GetDatePartSpecifier(ConstantVector::GetData<string_t>(part_arg)->GetString());
			DateTruncUnaryExecutor<TA, TR>(type, date_arg, result, args.size());
		}
	} else {
		BinaryExecutor::ExecuteStandard<string_t, TA, TR, DateTruncBinaryOperator>(part_arg, date_arg, result,
		                                                                           args.size());
	}
}

template <class TA, class TR, class OP>
static unique_ptr<BaseStatistics> DateTruncStatistics(vector<unique_ptr<BaseStatistics>> &child_stats) {
	// we can only propagate date stats if the child has stats
	if (!child_stats[1]) {
		return nullptr;
	}
	auto &nstats = (NumericStatistics &)*child_stats[1];
	if (nstats.min.IsNull() || nstats.max.IsNull()) {
		return nullptr;
	}
	// run the operator on both the min and the max, this gives us the [min, max] bound
	auto min = nstats.min.GetValueUnsafe<TA>();
	auto max = nstats.max.GetValueUnsafe<TA>();
	if (min > max) {
		return nullptr;
	}

	// Infinite values are unmodified
	auto min_part = DateTrunc::UnaryFunction<TA, TR, OP>(min);
	auto max_part = DateTrunc::UnaryFunction<TA, TR, OP>(max);

	auto min_value = Value::CreateValue(min_part);
	auto max_value = Value::CreateValue(max_part);
	auto result = make_unique<NumericStatistics>(min_value.type(), min_value, max_value, StatisticsType::LOCAL_STATS);
	if (child_stats[0]->validity_stats) {
		result->validity_stats = child_stats[1]->validity_stats->Copy();
	}
	return std::move(result);
}

template <class TA, class TR, class OP>
static unique_ptr<BaseStatistics> PropagateDateTruncStatistics(ClientContext &context, FunctionStatisticsInput &input) {
	return DateTruncStatistics<TA, TR, OP>(input.child_stats);
}

template <typename TA, typename TR>
static function_statistics_t DateTruncStats(DatePartSpecifier type) {
	switch (type) {
	case DatePartSpecifier::MILLENNIUM:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::MillenniumOperator>;
	case DatePartSpecifier::CENTURY:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::CenturyOperator>;
	case DatePartSpecifier::DECADE:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::DecadeOperator>;
	case DatePartSpecifier::YEAR:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::YearOperator>;
	case DatePartSpecifier::QUARTER:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::QuarterOperator>;
	case DatePartSpecifier::MONTH:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::MonthOperator>;
	case DatePartSpecifier::WEEK:
	case DatePartSpecifier::YEARWEEK:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::WeekOperator>;
	case DatePartSpecifier::ISOYEAR:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::ISOYearOperator>;
	case DatePartSpecifier::DAY:
	case DatePartSpecifier::DOW:
	case DatePartSpecifier::ISODOW:
	case DatePartSpecifier::DOY:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::DayOperator>;
	case DatePartSpecifier::HOUR:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::HourOperator>;
	case DatePartSpecifier::MINUTE:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::MinuteOperator>;
	case DatePartSpecifier::SECOND:
	case DatePartSpecifier::EPOCH:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::SecondOperator>;
	case DatePartSpecifier::MILLISECONDS:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::MillisecondOperator>;
	case DatePartSpecifier::MICROSECONDS:
		return PropagateDateTruncStatistics<TA, TR, DateTrunc::MicrosecondOperator>;
	default:
		throw NotImplementedException("Specifier type not implemented for DATETRUNC statistics");
	}
}

static unique_ptr<FunctionData> DateTruncBind(ClientContext &context, ScalarFunction &bound_function,
                                              vector<unique_ptr<Expression>> &arguments) {
	if (!arguments[0]->IsFoldable()) {
		return nullptr;
	}

	// Rebind to return a date if we are truncating that far
	Value part_value = ExpressionExecutor::EvaluateScalar(context, *arguments[0]);
	if (part_value.IsNull()) {
		return nullptr;
	}
	const auto part_name = part_value.ToString();
	const auto part_code = GetDatePartSpecifier(part_name);
	switch (part_code) {
	case DatePartSpecifier::MILLENNIUM:
	case DatePartSpecifier::CENTURY:
	case DatePartSpecifier::DECADE:
	case DatePartSpecifier::YEAR:
	case DatePartSpecifier::QUARTER:
	case DatePartSpecifier::MONTH:
	case DatePartSpecifier::WEEK:
	case DatePartSpecifier::YEARWEEK:
	case DatePartSpecifier::ISOYEAR:
	case DatePartSpecifier::DAY:
	case DatePartSpecifier::DOW:
	case DatePartSpecifier::ISODOW:
	case DatePartSpecifier::DOY:
		switch (bound_function.arguments[1].id()) {
		case LogicalType::TIMESTAMP:
			bound_function.function = DateTruncFunction<timestamp_t, date_t>;
			bound_function.statistics = DateTruncStats<timestamp_t, date_t>(part_code);
			break;
		case LogicalType::DATE:
			bound_function.function = DateTruncFunction<date_t, date_t>;
			bound_function.statistics = DateTruncStats<date_t, date_t>(part_code);
			break;
		default:
			throw NotImplementedException("Temporal argument type for DATETRUNC");
		}
		bound_function.return_type = LogicalType::DATE;
		break;
	default:
		switch (bound_function.arguments[1].id()) {
		case LogicalType::TIMESTAMP:
			bound_function.statistics = DateTruncStats<timestamp_t, timestamp_t>(part_code);
			break;
		case LogicalType::DATE:
			bound_function.statistics = DateTruncStats<date_t, timestamp_t>(part_code);
			break;
		default:
			throw NotImplementedException("Temporal argument type for DATETRUNC");
		}
		break;
	}

	return nullptr;
}

void DateTruncFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet date_trunc("date_trunc");
	date_trunc.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::TIMESTAMP}, LogicalType::TIMESTAMP,
	                                      DateTruncFunction<timestamp_t, timestamp_t>, DateTruncBind));
	date_trunc.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::DATE}, LogicalType::TIMESTAMP,
	                                      DateTruncFunction<date_t, timestamp_t>, DateTruncBind));
	date_trunc.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::INTERVAL}, LogicalType::INTERVAL,
	                                      DateTruncFunction<interval_t, interval_t>));
	set.AddFunction(date_trunc);
	date_trunc.name = "datetrunc";
	set.AddFunction(date_trunc);
}

} // namespace duckdb







namespace duckdb {

struct EpochSecOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input) {
		return Timestamp::FromEpochSeconds(input);
	}
};

static void EpochSecFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.ColumnCount() == 1);

	UnaryExecutor::Execute<int64_t, timestamp_t, EpochSecOperator>(input.data[0], result, input.size());
}

struct EpochMillisOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input) {
		return Timestamp::FromEpochMs(input);
	}
};

static void EpochMillisFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.ColumnCount() == 1);

	UnaryExecutor::Execute<int64_t, timestamp_t, EpochMillisOperator>(input.data[0], result, input.size());
}

void EpochFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet epoch("epoch_ms");
	epoch.AddFunction(ScalarFunction({LogicalType::BIGINT}, LogicalType::TIMESTAMP, EpochMillisFunction));
	set.AddFunction(epoch);
	// to_timestamp is an alias from Postgres that converts the time in seconds to a timestamp
	ScalarFunctionSet to_timestamp("to_timestamp");
	to_timestamp.AddFunction(ScalarFunction({LogicalType::BIGINT}, LogicalType::TIMESTAMP, EpochSecFunction));
	set.AddFunction(to_timestamp);
}

} // namespace duckdb







#include <cmath>

namespace duckdb {

struct MakeDateOperator {
	template <typename YYYY, typename MM, typename DD, typename RESULT_TYPE>
	static RESULT_TYPE Operation(YYYY yyyy, MM mm, DD dd) {
		return Date::FromDate(yyyy, mm, dd);
	}
};

template <typename T>
static void ExecuteMakeDate(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.ColumnCount() == 3);
	auto &yyyy = input.data[0];
	auto &mm = input.data[1];
	auto &dd = input.data[2];

	TernaryExecutor::Execute<T, T, T, date_t>(yyyy, mm, dd, result, input.size(),
	                                          MakeDateOperator::Operation<T, T, T, date_t>);
}

template <typename T>
static void ExecuteStructMakeDate(DataChunk &input, ExpressionState &state, Vector &result) {
	// this should be guaranteed by the binder
	D_ASSERT(input.ColumnCount() == 1);
	auto &vec = input.data[0];

	auto &children = StructVector::GetEntries(vec);
	D_ASSERT(children.size() == 3);
	auto &yyyy = *children[0];
	auto &mm = *children[1];
	auto &dd = *children[2];

	TernaryExecutor::Execute<T, T, T, date_t>(yyyy, mm, dd, result, input.size(), Date::FromDate);
}

struct MakeTimeOperator {
	template <typename HH, typename MM, typename SS, typename RESULT_TYPE>
	static RESULT_TYPE Operation(HH hh, MM mm, SS ss) {
		int64_t secs = ss;
		int64_t micros = std::round((ss - secs) * Interval::MICROS_PER_SEC);
		return Time::FromTime(hh, mm, secs, micros);
	}
};

template <typename T>
static void ExecuteMakeTime(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.ColumnCount() == 3);
	auto &yyyy = input.data[0];
	auto &mm = input.data[1];
	auto &dd = input.data[2];

	TernaryExecutor::Execute<T, T, double, dtime_t>(yyyy, mm, dd, result, input.size(),
	                                                MakeTimeOperator::Operation<T, T, double, dtime_t>);
}

struct MakeTimestampOperator {
	template <typename YYYY, typename MM, typename DD, typename HR, typename MN, typename SS, typename RESULT_TYPE>
	static RESULT_TYPE Operation(YYYY yyyy, MM mm, DD dd, HR hr, MN mn, SS ss) {
		const auto d = MakeDateOperator::Operation<YYYY, MM, DD, date_t>(yyyy, mm, dd);
		const auto t = MakeTimeOperator::Operation<HR, MN, SS, dtime_t>(hr, mn, ss);
		return Timestamp::FromDatetime(d, t);
	}
};

template <typename T>
static void ExecuteMakeTimestamp(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.ColumnCount() == 6);

	auto func = MakeTimestampOperator::Operation<T, T, T, T, T, double, timestamp_t>;
	SenaryExecutor::Execute<T, T, T, T, T, double, timestamp_t>(input, result, func);
}

void MakeDateFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet make_date("make_date");
	make_date.AddFunction(ScalarFunction({LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT},
	                                     LogicalType::DATE, ExecuteMakeDate<int64_t>));

	child_list_t<LogicalType> make_date_children {
	    {"year", LogicalType::BIGINT}, {"month", LogicalType::BIGINT}, {"day", LogicalType::BIGINT}};
	make_date.AddFunction(
	    ScalarFunction({LogicalType::STRUCT(make_date_children)}, LogicalType::DATE, ExecuteStructMakeDate<int64_t>));
	set.AddFunction(make_date);

	ScalarFunctionSet make_time("make_time");
	make_time.AddFunction(ScalarFunction({LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::DOUBLE},
	                                     LogicalType::TIME, ExecuteMakeTime<int64_t>));
	set.AddFunction(make_time);

	ScalarFunctionSet make_timestamp("make_timestamp");
	make_timestamp.AddFunction(ScalarFunction({LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT,
	                                           LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::DOUBLE},
	                                          LogicalType::TIMESTAMP, ExecuteMakeTimestamp<int64_t>));
	set.AddFunction(make_timestamp);
}

} // namespace duckdb














#include <cctype>

namespace duckdb {

idx_t StrfTimepecifierSize(StrTimeSpecifier specifier) {
	switch (specifier) {
	case StrTimeSpecifier::ABBREVIATED_WEEKDAY_NAME:
	case StrTimeSpecifier::ABBREVIATED_MONTH_NAME:
		return 3;
	case StrTimeSpecifier::WEEKDAY_DECIMAL:
		return 1;
	case StrTimeSpecifier::DAY_OF_MONTH_PADDED:
	case StrTimeSpecifier::MONTH_DECIMAL_PADDED:
	case StrTimeSpecifier::YEAR_WITHOUT_CENTURY_PADDED:
	case StrTimeSpecifier::HOUR_24_PADDED:
	case StrTimeSpecifier::HOUR_12_PADDED:
	case StrTimeSpecifier::MINUTE_PADDED:
	case StrTimeSpecifier::SECOND_PADDED:
	case StrTimeSpecifier::AM_PM:
	case StrTimeSpecifier::WEEK_NUMBER_PADDED_SUN_FIRST:
	case StrTimeSpecifier::WEEK_NUMBER_PADDED_MON_FIRST:
		return 2;
	case StrTimeSpecifier::MICROSECOND_PADDED:
		return 6;
	case StrTimeSpecifier::MILLISECOND_PADDED:
		return 3;
	case StrTimeSpecifier::DAY_OF_YEAR_PADDED:
		return 3;
	default:
		return 0;
	}
}

void StrTimeFormat::AddLiteral(string literal) {
	constant_size += literal.size();
	literals.push_back(std::move(literal));
}

void StrTimeFormat::AddFormatSpecifier(string preceding_literal, StrTimeSpecifier specifier) {
	AddLiteral(std::move(preceding_literal));
	specifiers.push_back(specifier);
}

void StrfTimeFormat::AddFormatSpecifier(string preceding_literal, StrTimeSpecifier specifier) {
	is_date_specifier.push_back(IsDateSpecifier(specifier));
	idx_t specifier_size = StrfTimepecifierSize(specifier);
	if (specifier_size == 0) {
		// variable length specifier
		var_length_specifiers.push_back(specifier);
	} else {
		// constant size specifier
		constant_size += specifier_size;
	}
	StrTimeFormat::AddFormatSpecifier(std::move(preceding_literal), specifier);
}

idx_t StrfTimeFormat::GetSpecifierLength(StrTimeSpecifier specifier, date_t date, dtime_t time, int32_t utc_offset,
                                         const char *tz_name) {
	switch (specifier) {
	case StrTimeSpecifier::FULL_WEEKDAY_NAME:
		return Date::DAY_NAMES[Date::ExtractISODayOfTheWeek(date) % 7].GetSize();
	case StrTimeSpecifier::FULL_MONTH_NAME:
		return Date::MONTH_NAMES[Date::ExtractMonth(date) - 1].GetSize();
	case StrTimeSpecifier::YEAR_DECIMAL: {
		auto year = Date::ExtractYear(date);
		// Be consistent with WriteStandardSpecifier
		if (0 <= year && year <= 9999) {
			return 4;
		} else {
			return NumericHelper::SignedLength<int32_t, uint32_t>(year);
		}
	}
	case StrTimeSpecifier::MONTH_DECIMAL: {
		idx_t len = 1;
		auto month = Date::ExtractMonth(date);
		len += month >= 10;
		return len;
	}
	case StrTimeSpecifier::UTC_OFFSET:
		// ±HH or ±HH:MM
		return (utc_offset % 60) ? 6 : 3;
	case StrTimeSpecifier::TZ_NAME:
		if (tz_name) {
			return strlen(tz_name);
		}
		// empty for now
		return 0;
	case StrTimeSpecifier::HOUR_24_DECIMAL:
	case StrTimeSpecifier::HOUR_12_DECIMAL:
	case StrTimeSpecifier::MINUTE_DECIMAL:
	case StrTimeSpecifier::SECOND_DECIMAL: {
		// time specifiers
		idx_t len = 1;
		int32_t hour, min, sec, msec;
		Time::Convert(time, hour, min, sec, msec);
		switch (specifier) {
		case StrTimeSpecifier::HOUR_24_DECIMAL:
			len += hour >= 10;
			break;
		case StrTimeSpecifier::HOUR_12_DECIMAL:
			hour = hour % 12;
			if (hour == 0) {
				hour = 12;
			}
			len += hour >= 10;
			break;
		case StrTimeSpecifier::MINUTE_DECIMAL:
			len += min >= 10;
			break;
		case StrTimeSpecifier::SECOND_DECIMAL:
			len += sec >= 10;
			break;
		default:
			throw InternalException("Time specifier mismatch");
		}
		return len;
	}
	case StrTimeSpecifier::DAY_OF_MONTH:
		return NumericHelper::UnsignedLength<uint32_t>(Date::ExtractDay(date));
	case StrTimeSpecifier::DAY_OF_YEAR_DECIMAL:
		return NumericHelper::UnsignedLength<uint32_t>(Date::ExtractDayOfTheYear(date));
	case StrTimeSpecifier::YEAR_WITHOUT_CENTURY:
		return NumericHelper::UnsignedLength<uint32_t>(AbsValue(Date::ExtractYear(date)) % 100);
	default:
		throw InternalException("Unimplemented specifier for GetSpecifierLength");
	}
}

//! Returns the total length of the date formatted by this format specifier
idx_t StrfTimeFormat::GetLength(date_t date, dtime_t time, int32_t utc_offset, const char *tz_name) {
	idx_t size = constant_size;
	if (!var_length_specifiers.empty()) {
		for (auto &specifier : var_length_specifiers) {
			size += GetSpecifierLength(specifier, date, time, utc_offset, tz_name);
		}
	}
	return size;
}

char *StrfTimeFormat::WriteString(char *target, const string_t &str) {
	idx_t size = str.GetSize();
	memcpy(target, str.GetDataUnsafe(), size);
	return target + size;
}

// write a value in the range of 0..99 unpadded (e.g. "1", "2", ... "98", "99")
char *StrfTimeFormat::Write2(char *target, uint8_t value) {
	D_ASSERT(value < 100);
	if (value >= 10) {
		return WritePadded2(target, value);
	} else {
		*target = char(uint8_t('0') + value);
		return target + 1;
	}
}

// write a value in the range of 0..99 padded to 2 digits
char *StrfTimeFormat::WritePadded2(char *target, uint32_t value) {
	D_ASSERT(value < 100);
	auto index = static_cast<unsigned>(value * 2);
	*target++ = duckdb_fmt::internal::data::digits[index];
	*target++ = duckdb_fmt::internal::data::digits[index + 1];
	return target;
}

// write a value in the range of 0..999 padded
char *StrfTimeFormat::WritePadded3(char *target, uint32_t value) {
	D_ASSERT(value < 1000);
	if (value >= 100) {
		WritePadded2(target + 1, value % 100);
		*target = char(uint8_t('0') + value / 100);
		return target + 3;
	} else {
		*target = '0';
		target++;
		return WritePadded2(target, value);
	}
}

// write a value in the range of 0..999999 padded to 6 digits
char *StrfTimeFormat::WritePadded(char *target, uint32_t value, size_t padding) {
	D_ASSERT(padding % 2 == 0);
	for (size_t i = 0; i < padding / 2; i++) {
		int decimals = value % 100;
		WritePadded2(target + padding - 2 * (i + 1), decimals);
		value /= 100;
	}
	return target + padding;
}

bool StrfTimeFormat::IsDateSpecifier(StrTimeSpecifier specifier) {
	switch (specifier) {
	case StrTimeSpecifier::ABBREVIATED_WEEKDAY_NAME:
	case StrTimeSpecifier::FULL_WEEKDAY_NAME:
	case StrTimeSpecifier::DAY_OF_YEAR_PADDED:
	case StrTimeSpecifier::DAY_OF_YEAR_DECIMAL:
	case StrTimeSpecifier::WEEK_NUMBER_PADDED_MON_FIRST:
	case StrTimeSpecifier::WEEK_NUMBER_PADDED_SUN_FIRST:
	case StrTimeSpecifier::WEEKDAY_DECIMAL:
		return true;
	default:
		return false;
	}
}

char *StrfTimeFormat::WriteDateSpecifier(StrTimeSpecifier specifier, date_t date, char *target) {
	switch (specifier) {
	case StrTimeSpecifier::ABBREVIATED_WEEKDAY_NAME: {
		auto dow = Date::ExtractISODayOfTheWeek(date);
		target = WriteString(target, Date::DAY_NAMES_ABBREVIATED[dow % 7]);
		break;
	}
	case StrTimeSpecifier::FULL_WEEKDAY_NAME: {
		auto dow = Date::ExtractISODayOfTheWeek(date);
		target = WriteString(target, Date::DAY_NAMES[dow % 7]);
		break;
	}
	case StrTimeSpecifier::WEEKDAY_DECIMAL: {
		auto dow = Date::ExtractISODayOfTheWeek(date);
		*target = char('0' + uint8_t(dow % 7));
		target++;
		break;
	}
	case StrTimeSpecifier::DAY_OF_YEAR_PADDED: {
		int32_t doy = Date::ExtractDayOfTheYear(date);
		target = WritePadded3(target, doy);
		break;
	}
	case StrTimeSpecifier::WEEK_NUMBER_PADDED_MON_FIRST:
		target = WritePadded2(target, Date::ExtractWeekNumberRegular(date, true));
		break;
	case StrTimeSpecifier::WEEK_NUMBER_PADDED_SUN_FIRST:
		target = WritePadded2(target, Date::ExtractWeekNumberRegular(date, false));
		break;
	case StrTimeSpecifier::DAY_OF_YEAR_DECIMAL: {
		uint32_t doy = Date::ExtractDayOfTheYear(date);
		target += NumericHelper::UnsignedLength<uint32_t>(doy);
		NumericHelper::FormatUnsigned(doy, target);
		break;
	}
	default:
		throw InternalException("Unimplemented date specifier for strftime");
	}
	return target;
}

char *StrfTimeFormat::WriteStandardSpecifier(StrTimeSpecifier specifier, int32_t data[], const char *tz_name,
                                             size_t tz_len, char *target) {
	// data contains [0] year, [1] month, [2] day, [3] hour, [4] minute, [5] second, [6] msec, [7] utc
	switch (specifier) {
	case StrTimeSpecifier::DAY_OF_MONTH_PADDED:
		target = WritePadded2(target, data[2]);
		break;
	case StrTimeSpecifier::ABBREVIATED_MONTH_NAME: {
		auto &month_name = Date::MONTH_NAMES_ABBREVIATED[data[1] - 1];
		return WriteString(target, month_name);
	}
	case StrTimeSpecifier::FULL_MONTH_NAME: {
		auto &month_name = Date::MONTH_NAMES[data[1] - 1];
		return WriteString(target, month_name);
	}
	case StrTimeSpecifier::MONTH_DECIMAL_PADDED:
		target = WritePadded2(target, data[1]);
		break;
	case StrTimeSpecifier::YEAR_WITHOUT_CENTURY_PADDED:
		target = WritePadded2(target, AbsValue(data[0]) % 100);
		break;
	case StrTimeSpecifier::YEAR_DECIMAL:
		if (data[0] >= 0 && data[0] <= 9999) {
			target = WritePadded(target, data[0], 4);
		} else {
			int32_t year = data[0];
			if (data[0] < 0) {
				*target = '-';
				year = -year;
				target++;
			}
			auto len = NumericHelper::UnsignedLength<uint32_t>(year);
			NumericHelper::FormatUnsigned(year, target + len);
			target += len;
		}
		break;
	case StrTimeSpecifier::HOUR_24_PADDED: {
		target = WritePadded2(target, data[3]);
		break;
	}
	case StrTimeSpecifier::HOUR_12_PADDED: {
		int hour = data[3] % 12;
		if (hour == 0) {
			hour = 12;
		}
		target = WritePadded2(target, hour);
		break;
	}
	case StrTimeSpecifier::AM_PM:
		*target++ = data[3] >= 12 ? 'P' : 'A';
		*target++ = 'M';
		break;
	case StrTimeSpecifier::MINUTE_PADDED: {
		target = WritePadded2(target, data[4]);
		break;
	}
	case StrTimeSpecifier::SECOND_PADDED:
		target = WritePadded2(target, data[5]);
		break;
	case StrTimeSpecifier::MICROSECOND_PADDED:
		target = WritePadded(target, data[6], 6);
		break;
	case StrTimeSpecifier::MILLISECOND_PADDED:
		target = WritePadded3(target, data[6] / 1000);
		break;
	case StrTimeSpecifier::UTC_OFFSET: {
		*target++ = (data[7] < 0) ? '-' : '+';

		auto offset = abs(data[7]);
		auto offset_hours = offset / Interval::MINS_PER_HOUR;
		auto offset_minutes = offset % Interval::MINS_PER_HOUR;
		target = WritePadded2(target, offset_hours);
		if (offset_minutes) {
			*target++ = ':';
			target = WritePadded2(target, offset_minutes);
		}
		break;
	}
	case StrTimeSpecifier::TZ_NAME:
		if (tz_name) {
			memcpy(target, tz_name, tz_len);
			target += strlen(tz_name);
		}
		break;
	case StrTimeSpecifier::DAY_OF_MONTH: {
		target = Write2(target, data[2] % 100);
		break;
	}
	case StrTimeSpecifier::MONTH_DECIMAL: {
		target = Write2(target, data[1]);
		break;
	}
	case StrTimeSpecifier::YEAR_WITHOUT_CENTURY: {
		target = Write2(target, AbsValue(data[0]) % 100);
		break;
	}
	case StrTimeSpecifier::HOUR_24_DECIMAL: {
		target = Write2(target, data[3]);
		break;
	}
	case StrTimeSpecifier::HOUR_12_DECIMAL: {
		int hour = data[3] % 12;
		if (hour == 0) {
			hour = 12;
		}
		target = Write2(target, hour);
		break;
	}
	case StrTimeSpecifier::MINUTE_DECIMAL: {
		target = Write2(target, data[4]);
		break;
	}
	case StrTimeSpecifier::SECOND_DECIMAL: {
		target = Write2(target, data[5]);
		break;
	}
	default:
		throw InternalException("Unimplemented specifier for WriteStandardSpecifier in strftime");
	}
	return target;
}

void StrfTimeFormat::FormatString(date_t date, int32_t data[8], const char *tz_name, char *target) {
	D_ASSERT(specifiers.size() + 1 == literals.size());
	idx_t i;
	for (i = 0; i < specifiers.size(); i++) {
		// first copy the current literal
		memcpy(target, literals[i].c_str(), literals[i].size());
		target += literals[i].size();
		// now copy the specifier
		if (is_date_specifier[i]) {
			target = WriteDateSpecifier(specifiers[i], date, target);
		} else {
			auto tz_len = tz_name ? strlen(tz_name) : 0;
			target = WriteStandardSpecifier(specifiers[i], data, tz_name, tz_len, target);
		}
	}
	// copy the final literal into the target
	memcpy(target, literals[i].c_str(), literals[i].size());
}

void StrfTimeFormat::FormatString(date_t date, dtime_t time, char *target) {
	int32_t data[8]; // year, month, day, hour, min, sec, µs, offset
	Date::Convert(date, data[0], data[1], data[2]);
	Time::Convert(time, data[3], data[4], data[5], data[6]);
	data[7] = 0;

	FormatString(date, data, nullptr, target);
}

string StrfTimeFormat::Format(timestamp_t timestamp, const string &format_str) {
	StrfTimeFormat format;
	format.ParseFormatSpecifier(format_str, format);

	auto date = Timestamp::GetDate(timestamp);
	auto time = Timestamp::GetTime(timestamp);

	auto len = format.GetLength(date, time, 0, nullptr);
	auto result = unique_ptr<char[]>(new char[len]);
	format.FormatString(date, time, result.get());
	return string(result.get(), len);
}

string StrTimeFormat::ParseFormatSpecifier(const string &format_string, StrTimeFormat &format) {
	if (format_string.empty()) {
		return "Empty format string";
	}
	format.specifiers.clear();
	format.literals.clear();
	format.numeric_width.clear();
	format.constant_size = 0;
	idx_t pos = 0;
	string current_literal;
	for (idx_t i = 0; i < format_string.size(); i++) {
		if (format_string[i] == '%') {
			if (i + 1 == format_string.size()) {
				return "Trailing format character %";
			}
			if (i > pos) {
				// push the previous string to the current literal
				current_literal += format_string.substr(pos, i - pos);
			}
			char format_char = format_string[++i];
			if (format_char == '%') {
				// special case: %%
				// set the pos for the next literal and continue
				pos = i;
				continue;
			}
			StrTimeSpecifier specifier;
			if (format_char == '-' && i + 1 < format_string.size()) {
				format_char = format_string[++i];
				switch (format_char) {
				case 'd':
					specifier = StrTimeSpecifier::DAY_OF_MONTH;
					break;
				case 'm':
					specifier = StrTimeSpecifier::MONTH_DECIMAL;
					break;
				case 'y':
					specifier = StrTimeSpecifier::YEAR_WITHOUT_CENTURY;
					break;
				case 'H':
					specifier = StrTimeSpecifier::HOUR_24_DECIMAL;
					break;
				case 'I':
					specifier = StrTimeSpecifier::HOUR_12_DECIMAL;
					break;
				case 'M':
					specifier = StrTimeSpecifier::MINUTE_DECIMAL;
					break;
				case 'S':
					specifier = StrTimeSpecifier::SECOND_DECIMAL;
					break;
				case 'j':
					specifier = StrTimeSpecifier::DAY_OF_YEAR_DECIMAL;
					break;
				default:
					return "Unrecognized format for strftime/strptime: %-" + string(1, format_char);
				}
			} else {
				switch (format_char) {
				case 'a':
					specifier = StrTimeSpecifier::ABBREVIATED_WEEKDAY_NAME;
					break;
				case 'A':
					specifier = StrTimeSpecifier::FULL_WEEKDAY_NAME;
					break;
				case 'w':
					specifier = StrTimeSpecifier::WEEKDAY_DECIMAL;
					break;
				case 'd':
					specifier = StrTimeSpecifier::DAY_OF_MONTH_PADDED;
					break;
				case 'h':
				case 'b':
					specifier = StrTimeSpecifier::ABBREVIATED_MONTH_NAME;
					break;
				case 'B':
					specifier = StrTimeSpecifier::FULL_MONTH_NAME;
					break;
				case 'm':
					specifier = StrTimeSpecifier::MONTH_DECIMAL_PADDED;
					break;
				case 'y':
					specifier = StrTimeSpecifier::YEAR_WITHOUT_CENTURY_PADDED;
					break;
				case 'Y':
					specifier = StrTimeSpecifier::YEAR_DECIMAL;
					break;
				case 'H':
					specifier = StrTimeSpecifier::HOUR_24_PADDED;
					break;
				case 'I':
					specifier = StrTimeSpecifier::HOUR_12_PADDED;
					break;
				case 'p':
					specifier = StrTimeSpecifier::AM_PM;
					break;
				case 'M':
					specifier = StrTimeSpecifier::MINUTE_PADDED;
					break;
				case 'S':
					specifier = StrTimeSpecifier::SECOND_PADDED;
					break;
				case 'f':
					specifier = StrTimeSpecifier::MICROSECOND_PADDED;
					break;
				case 'g':
					specifier = StrTimeSpecifier::MILLISECOND_PADDED;
					break;
				case 'z':
					specifier = StrTimeSpecifier::UTC_OFFSET;
					break;
				case 'Z':
					specifier = StrTimeSpecifier::TZ_NAME;
					break;
				case 'j':
					specifier = StrTimeSpecifier::DAY_OF_YEAR_PADDED;
					break;
				case 'U':
					specifier = StrTimeSpecifier::WEEK_NUMBER_PADDED_SUN_FIRST;
					break;
				case 'W':
					specifier = StrTimeSpecifier::WEEK_NUMBER_PADDED_MON_FIRST;
					break;
				case 'c':
				case 'x':
				case 'X':
				case 'T': {
					string subformat;
					if (format_char == 'c') {
						// %c: Locale’s appropriate date and time representation.
						// we push the ISO timestamp representation here
						subformat = "%Y-%m-%d %H:%M:%S";
					} else if (format_char == 'x') {
						// %x - Locale’s appropriate date representation.
						// we push the ISO date format here
						subformat = "%Y-%m-%d";
					} else if (format_char == 'X' || format_char == 'T') {
						// %X - Locale’s appropriate time representation.
						// we push the ISO time format here
						subformat = "%H:%M:%S";
					}
					// parse the subformat in a separate format specifier
					StrfTimeFormat locale_format;
					string error = StrTimeFormat::ParseFormatSpecifier(subformat, locale_format);
					D_ASSERT(error.empty());
					// add the previous literal to the first literal of the subformat
					locale_format.literals[0] = std::move(current_literal) + locale_format.literals[0];
					current_literal = "";
					// now push the subformat into the current format specifier
					for (idx_t i = 0; i < locale_format.specifiers.size(); i++) {
						format.AddFormatSpecifier(std::move(locale_format.literals[i]), locale_format.specifiers[i]);
					}
					pos = i + 1;
					continue;
				}
				default:
					return "Unrecognized format for strftime/strptime: %" + string(1, format_char);
				}
			}
			format.AddFormatSpecifier(std::move(current_literal), specifier);
			current_literal = "";
			pos = i + 1;
		}
	}
	// add the final literal
	if (pos < format_string.size()) {
		current_literal += format_string.substr(pos, format_string.size() - pos);
	}
	format.AddLiteral(std::move(current_literal));
	return string();
}

struct StrfTimeBindData : public FunctionData {
	explicit StrfTimeBindData(StrfTimeFormat format_p, string format_string_p, bool is_null)
	    : format(std::move(format_p)), format_string(std::move(format_string_p)), is_null(is_null) {
	}

	StrfTimeFormat format;
	string format_string;
	bool is_null;

	unique_ptr<FunctionData> Copy() const override {
		return make_unique<StrfTimeBindData>(format, format_string, is_null);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const StrfTimeBindData &)other_p;
		return format_string == other.format_string;
	}
};

template <bool REVERSED>
static unique_ptr<FunctionData> StrfTimeBindFunction(ClientContext &context, ScalarFunction &bound_function,
                                                     vector<unique_ptr<Expression>> &arguments) {
	auto format_idx = REVERSED ? 0 : 1;
	auto &format_arg = arguments[format_idx];
	if (format_arg->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (!format_arg->IsFoldable()) {
		throw InvalidInputException("strftime format must be a constant");
	}
	Value options_str = ExpressionExecutor::EvaluateScalar(context, *format_arg);
	auto format_string = options_str.GetValue<string>();
	StrfTimeFormat format;
	bool is_null = options_str.IsNull();
	if (!is_null) {
		string error = StrTimeFormat::ParseFormatSpecifier(format_string, format);
		if (!error.empty()) {
			throw InvalidInputException("Failed to parse format specifier %s: %s", format_string, error);
		}
	}
	return make_unique<StrfTimeBindData>(format, format_string, is_null);
}

void StrfTimeFormat::ConvertDateVector(Vector &input, Vector &result, idx_t count) {
	D_ASSERT(input.GetType().id() == LogicalTypeId::DATE);
	D_ASSERT(result.GetType().id() == LogicalTypeId::VARCHAR);
	UnaryExecutor::ExecuteWithNulls<date_t, string_t>(input, result, count,
	                                                  [&](date_t input, ValidityMask &mask, idx_t idx) {
		                                                  if (Date::IsFinite(input)) {
			                                                  dtime_t time(0);
			                                                  idx_t len = GetLength(input, time, 0, nullptr);
			                                                  string_t target = StringVector::EmptyString(result, len);
			                                                  FormatString(input, time, target.GetDataWriteable());
			                                                  target.Finalize();
			                                                  return target;
		                                                  } else {
			                                                  mask.SetInvalid(idx);
			                                                  return string_t();
		                                                  }
	                                                  });
}

template <bool REVERSED>
static void StrfTimeFunctionDate(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = (BoundFunctionExpression &)state.expr;
	auto &info = (StrfTimeBindData &)*func_expr.bind_info;

	if (info.is_null) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		ConstantVector::SetNull(result, true);
		return;
	}
	info.format.ConvertDateVector(args.data[REVERSED ? 1 : 0], result, args.size());
}

void StrfTimeFormat::ConvertTimestampVector(Vector &input, Vector &result, idx_t count) {
	D_ASSERT(input.GetType().id() == LogicalTypeId::TIMESTAMP);
	D_ASSERT(result.GetType().id() == LogicalTypeId::VARCHAR);
	UnaryExecutor::ExecuteWithNulls<timestamp_t, string_t>(
	    input, result, count, [&](timestamp_t input, ValidityMask &mask, idx_t idx) {
		    if (Timestamp::IsFinite(input)) {
			    date_t date;
			    dtime_t time;
			    Timestamp::Convert(input, date, time);
			    idx_t len = GetLength(date, time, 0, nullptr);
			    string_t target = StringVector::EmptyString(result, len);
			    FormatString(date, time, target.GetDataWriteable());
			    target.Finalize();
			    return target;
		    } else {
			    mask.SetInvalid(idx);
			    return string_t();
		    }
	    });
}

template <bool REVERSED>
static void StrfTimeFunctionTimestamp(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = (BoundFunctionExpression &)state.expr;
	auto &info = (StrfTimeBindData &)*func_expr.bind_info;

	if (info.is_null) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		ConstantVector::SetNull(result, true);
		return;
	}
	info.format.ConvertTimestampVector(args.data[REVERSED ? 1 : 0], result, args.size());
}

void StrfTimeFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet strftime("strftime");

	strftime.AddFunction(ScalarFunction({LogicalType::DATE, LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                                    StrfTimeFunctionDate<false>, StrfTimeBindFunction<false>));

	strftime.AddFunction(ScalarFunction({LogicalType::TIMESTAMP, LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                                    StrfTimeFunctionTimestamp<false>, StrfTimeBindFunction<false>));

	strftime.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::DATE}, LogicalType::VARCHAR,
	                                    StrfTimeFunctionDate<true>, StrfTimeBindFunction<true>));

	strftime.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::TIMESTAMP}, LogicalType::VARCHAR,
	                                    StrfTimeFunctionTimestamp<true>, StrfTimeBindFunction<true>));

	set.AddFunction(strftime);
}

void StrpTimeFormat::AddFormatSpecifier(string preceding_literal, StrTimeSpecifier specifier) {
	numeric_width.push_back(NumericSpecifierWidth(specifier));
	StrTimeFormat::AddFormatSpecifier(std::move(preceding_literal), specifier);
}

int StrpTimeFormat::NumericSpecifierWidth(StrTimeSpecifier specifier) {
	switch (specifier) {
	case StrTimeSpecifier::WEEKDAY_DECIMAL:
		return 1;
	case StrTimeSpecifier::DAY_OF_MONTH_PADDED:
	case StrTimeSpecifier::DAY_OF_MONTH:
	case StrTimeSpecifier::MONTH_DECIMAL_PADDED:
	case StrTimeSpecifier::MONTH_DECIMAL:
	case StrTimeSpecifier::YEAR_WITHOUT_CENTURY_PADDED:
	case StrTimeSpecifier::YEAR_WITHOUT_CENTURY:
	case StrTimeSpecifier::HOUR_24_PADDED:
	case StrTimeSpecifier::HOUR_24_DECIMAL:
	case StrTimeSpecifier::HOUR_12_PADDED:
	case StrTimeSpecifier::HOUR_12_DECIMAL:
	case StrTimeSpecifier::MINUTE_PADDED:
	case StrTimeSpecifier::MINUTE_DECIMAL:
	case StrTimeSpecifier::SECOND_PADDED:
	case StrTimeSpecifier::SECOND_DECIMAL:
	case StrTimeSpecifier::WEEK_NUMBER_PADDED_SUN_FIRST:
	case StrTimeSpecifier::WEEK_NUMBER_PADDED_MON_FIRST:
		return 2;
	case StrTimeSpecifier::MILLISECOND_PADDED:
	case StrTimeSpecifier::DAY_OF_YEAR_PADDED:
	case StrTimeSpecifier::DAY_OF_YEAR_DECIMAL:
		return 3;
	case StrTimeSpecifier::YEAR_DECIMAL:
		return 4;
	case StrTimeSpecifier::MICROSECOND_PADDED:
		return 6;
	default:
		return -1;
	}
}

enum class TimeSpecifierAMOrPM : uint8_t { TIME_SPECIFIER_NONE = 0, TIME_SPECIFIER_AM = 1, TIME_SPECIFIER_PM = 2 };

int32_t StrpTimeFormat::TryParseCollection(const char *data, idx_t &pos, idx_t size, const string_t collection[],
                                           idx_t collection_count) {
	for (idx_t c = 0; c < collection_count; c++) {
		auto &entry = collection[c];
		auto entry_data = entry.GetDataUnsafe();
		auto entry_size = entry.GetSize();
		// check if this entry matches
		if (pos + entry_size > size) {
			// too big: can't match
			continue;
		}
		// compare the characters
		idx_t i;
		for (i = 0; i < entry_size; i++) {
			if (std::tolower(entry_data[i]) != std::tolower(data[pos + i])) {
				break;
			}
		}
		if (i == entry_size) {
			// full match
			pos += entry_size;
			return c;
		}
	}
	return -1;
}

//! Parses a timestamp using the given specifier
bool StrpTimeFormat::Parse(string_t str, ParseResult &result) {
	auto &result_data = result.data;
	auto &error_message = result.error_message;
	auto &error_position = result.error_position;

	// initialize the result
	result_data[0] = 1900;
	result_data[1] = 1;
	result_data[2] = 1;
	result_data[3] = 0;
	result_data[4] = 0;
	result_data[5] = 0;
	result_data[6] = 0;
	result_data[7] = 0;

	auto data = str.GetDataUnsafe();
	idx_t size = str.GetSize();
	// skip leading spaces
	while (StringUtil::CharacterIsSpace(*data)) {
		data++;
		size--;
	}
	idx_t pos = 0;
	TimeSpecifierAMOrPM ampm = TimeSpecifierAMOrPM::TIME_SPECIFIER_NONE;

	// Year offset state (Year+W/j)
	auto offset_specifier = StrTimeSpecifier::WEEKDAY_DECIMAL;
	uint64_t weekno = 0;
	uint64_t weekday = 0;
	uint64_t yearday = 0;

	for (idx_t i = 0;; i++) {
		D_ASSERT(i < literals.size());
		// first compare the literal
		const auto &literal = literals[i];
		for (size_t l = 0; l < literal.size();) {
			// Match runs of spaces to runs of spaces.
			if (StringUtil::CharacterIsSpace(literal[l])) {
				if (!StringUtil::CharacterIsSpace(data[pos])) {
					error_message = "Space does not match, expected " + literals[i];
					error_position = pos;
					return false;
				}
				for (++pos; pos < size && StringUtil::CharacterIsSpace(data[pos]); ++pos) {
					continue;
				}
				for (++l; l < literal.size() && StringUtil::CharacterIsSpace(literal[l]); ++l) {
					continue;
				}
				continue;
			}
			// literal does not match
			if (data[pos++] != literal[l++]) {
				error_message = "Literal does not match, expected " + literal;
				error_position = pos;
				return false;
			}
		}
		if (i == specifiers.size()) {
			break;
		}
		// now parse the specifier
		if (numeric_width[i] > 0) {
			// numeric specifier: parse a number
			uint64_t number = 0;
			size_t start_pos = pos;
			size_t end_pos = start_pos + numeric_width[i];
			while (pos < size && pos < end_pos && StringUtil::CharacterIsDigit(data[pos])) {
				number = number * 10 + data[pos] - '0';
				pos++;
			}
			if (pos == start_pos) {
				// expected a number here
				error_message = "Expected a number";
				error_position = start_pos;
				return false;
			}
			switch (specifiers[i]) {
			case StrTimeSpecifier::DAY_OF_MONTH_PADDED:
			case StrTimeSpecifier::DAY_OF_MONTH:
				if (number < 1 || number > 31) {
					error_message = "Day out of range, expected a value between 1 and 31";
					error_position = start_pos;
					return false;
				}
				// day of the month
				result_data[2] = number;
				offset_specifier = specifiers[i];
				break;
			case StrTimeSpecifier::MONTH_DECIMAL_PADDED:
			case StrTimeSpecifier::MONTH_DECIMAL:
				if (number < 1 || number > 12) {
					error_message = "Month out of range, expected a value between 1 and 12";
					error_position = start_pos;
					return false;
				}
				// month number
				result_data[1] = number;
				offset_specifier = specifiers[i];
				break;
			case StrTimeSpecifier::YEAR_WITHOUT_CENTURY_PADDED:
			case StrTimeSpecifier::YEAR_WITHOUT_CENTURY:
				// year without century..
				// Python uses 69 as a crossover point (i.e. >= 69 is 19.., < 69 is 20..)
				if (number >= 100) {
					// %y only supports numbers between [0..99]
					error_message = "Year without century out of range, expected a value between 0 and 99";
					error_position = start_pos;
					return false;
				}
				if (number >= 69) {
					result_data[0] = int32_t(1900 + number);
				} else {
					result_data[0] = int32_t(2000 + number);
				}
				break;
			case StrTimeSpecifier::YEAR_DECIMAL:
				// year as full number
				result_data[0] = number;
				break;
			case StrTimeSpecifier::HOUR_24_PADDED:
			case StrTimeSpecifier::HOUR_24_DECIMAL:
				if (number >= 24) {
					error_message = "Hour out of range, expected a value between 0 and 23";
					error_position = start_pos;
					return false;
				}
				// hour as full number
				result_data[3] = number;
				break;
			case StrTimeSpecifier::HOUR_12_PADDED:
			case StrTimeSpecifier::HOUR_12_DECIMAL:
				if (number < 1 || number > 12) {
					error_message = "Hour12 out of range, expected a value between 1 and 12";
					error_position = start_pos;
					return false;
				}
				// 12-hour number: start off by just storing the number
				result_data[3] = number;
				break;
			case StrTimeSpecifier::MINUTE_PADDED:
			case StrTimeSpecifier::MINUTE_DECIMAL:
				if (number >= 60) {
					error_message = "Minutes out of range, expected a value between 0 and 59";
					error_position = start_pos;
					return false;
				}
				// minutes
				result_data[4] = number;
				break;
			case StrTimeSpecifier::SECOND_PADDED:
			case StrTimeSpecifier::SECOND_DECIMAL:
				if (number >= 60) {
					error_message = "Seconds out of range, expected a value between 0 and 59";
					error_position = start_pos;
					return false;
				}
				// seconds
				result_data[5] = number;
				break;
			case StrTimeSpecifier::MICROSECOND_PADDED:
				D_ASSERT(number < 1000000ULL); // enforced by the length of the number
				// milliseconds
				result_data[6] = number;
				break;
			case StrTimeSpecifier::MILLISECOND_PADDED:
				D_ASSERT(number < 1000ULL); // enforced by the length of the number
				// milliseconds
				result_data[6] = number * 1000;
				break;
			case StrTimeSpecifier::WEEK_NUMBER_PADDED_SUN_FIRST:
			case StrTimeSpecifier::WEEK_NUMBER_PADDED_MON_FIRST:
				// m/d overrides WU/w but does not conflict
				switch (offset_specifier) {
				case StrTimeSpecifier::DAY_OF_MONTH_PADDED:
				case StrTimeSpecifier::DAY_OF_MONTH:
				case StrTimeSpecifier::MONTH_DECIMAL_PADDED:
				case StrTimeSpecifier::MONTH_DECIMAL:
					// Just validate, don't use
					break;
				case StrTimeSpecifier::WEEKDAY_DECIMAL:
					// First offset specifier
					offset_specifier = specifiers[i];
					break;
				default:
					error_message = "Multiple year offsets specified";
					error_position = start_pos;
					return false;
				}
				if (number > 53) {
					error_message = "Week out of range, expected a value between 0 and 53";
					error_position = start_pos;
					return false;
				}
				weekno = number;
				break;
			case StrTimeSpecifier::WEEKDAY_DECIMAL:
				if (number > 6) {
					error_message = "Weekday out of range, expected a value between 0 and 6";
					error_position = start_pos;
					return false;
				}
				weekday = number;
				break;
			case StrTimeSpecifier::DAY_OF_YEAR_PADDED:
			case StrTimeSpecifier::DAY_OF_YEAR_DECIMAL:
				// m/d overrides j but does not conflict
				switch (offset_specifier) {
				case StrTimeSpecifier::DAY_OF_MONTH_PADDED:
				case StrTimeSpecifier::DAY_OF_MONTH:
				case StrTimeSpecifier::MONTH_DECIMAL_PADDED:
				case StrTimeSpecifier::MONTH_DECIMAL:
					// Just validate, don't use
					break;
				case StrTimeSpecifier::WEEKDAY_DECIMAL:
					// First offset specifier
					offset_specifier = specifiers[i];
					break;
				default:
					error_message = "Multiple year offsets specified";
					error_position = start_pos;
					return false;
				}
				if (number < 1 || number > 366) {
					error_message = "Year day out of range, expected a value between 1 and 366";
					error_position = start_pos;
					return false;
				}
				yearday = number;
				break;
			default:
				throw NotImplementedException("Unsupported specifier for strptime");
			}
		} else {
			switch (specifiers[i]) {
			case StrTimeSpecifier::AM_PM: {
				// parse the next 2 characters
				if (pos + 2 > size) {
					// no characters left to parse
					error_message = "Expected AM/PM";
					error_position = pos;
					return false;
				}
				char pa_char = char(std::tolower(data[pos]));
				char m_char = char(std::tolower(data[pos + 1]));
				if (m_char != 'm') {
					error_message = "Expected AM/PM";
					error_position = pos;
					return false;
				}
				if (pa_char == 'p') {
					ampm = TimeSpecifierAMOrPM::TIME_SPECIFIER_PM;
				} else if (pa_char == 'a') {
					ampm = TimeSpecifierAMOrPM::TIME_SPECIFIER_AM;
				} else {
					error_message = "Expected AM/PM";
					error_position = pos;
					return false;
				}
				pos += 2;
				break;
			}
			// we parse weekday names, but we don't use them as information
			case StrTimeSpecifier::ABBREVIATED_WEEKDAY_NAME:
				if (TryParseCollection(data, pos, size, Date::DAY_NAMES_ABBREVIATED, 7) < 0) {
					error_message = "Expected an abbreviated day name (Mon, Tue, Wed, Thu, Fri, Sat, Sun)";
					error_position = pos;
					return false;
				}
				break;
			case StrTimeSpecifier::FULL_WEEKDAY_NAME:
				if (TryParseCollection(data, pos, size, Date::DAY_NAMES, 7) < 0) {
					error_message = "Expected a full day name (Monday, Tuesday, etc...)";
					error_position = pos;
					return false;
				}
				break;
			case StrTimeSpecifier::ABBREVIATED_MONTH_NAME: {
				int32_t month = TryParseCollection(data, pos, size, Date::MONTH_NAMES_ABBREVIATED, 12);
				if (month < 0) {
					error_message = "Expected an abbreviated month name (Jan, Feb, Mar, etc..)";
					error_position = pos;
					return false;
				}
				result_data[1] = month + 1;
				break;
			}
			case StrTimeSpecifier::FULL_MONTH_NAME: {
				int32_t month = TryParseCollection(data, pos, size, Date::MONTH_NAMES, 12);
				if (month < 0) {
					error_message = "Expected a full month name (January, February, etc...)";
					error_position = pos;
					return false;
				}
				result_data[1] = month + 1;
				break;
			}
			case StrTimeSpecifier::UTC_OFFSET: {
				int hour_offset, minute_offset;
				if (!Timestamp::TryParseUTCOffset(data, pos, size, hour_offset, minute_offset)) {
					error_message = "Expected +HH[MM] or -HH[MM]";
					error_position = pos;
					return false;
				}
				result_data[7] = hour_offset * Interval::MINS_PER_HOUR + minute_offset;
				break;
			}
			case StrTimeSpecifier::TZ_NAME: {
				// skip leading spaces
				while (pos < size && StringUtil::CharacterIsSpace(data[pos])) {
					pos++;
				}
				const auto tz_begin = data + pos;
				// stop when we encounter a non-tz character
				while (pos < size && Timestamp::CharacterIsTimeZone(data[pos])) {
					pos++;
				}
				const auto tz_end = data + pos;
				// Can't fully validate without a list - caller's responsibility.
				// But tz must not be empty.
				if (tz_end == tz_begin) {
					error_message = "Empty Time Zone name";
					error_position = tz_begin - data;
					return false;
				}
				result.tz.assign(tz_begin, tz_end);
				break;
			}
			default:
				throw NotImplementedException("Unsupported specifier for strptime");
			}
		}
	}
	// skip trailing spaces
	while (pos < size && StringUtil::CharacterIsSpace(data[pos])) {
		pos++;
	}
	if (pos != size) {
		error_message = "Full specifier did not match: trailing characters";
		error_position = pos;
		return false;
	}
	if (ampm != TimeSpecifierAMOrPM::TIME_SPECIFIER_NONE) {
		if (result_data[3] > 12) {
			error_message =
			    "Invalid hour: " + to_string(result_data[3]) + " AM/PM, expected an hour within the range [0..12]";
			return false;
		}
		// adjust the hours based on the AM or PM specifier
		if (ampm == TimeSpecifierAMOrPM::TIME_SPECIFIER_AM) {
			// AM: 12AM=0, 1AM=1, 2AM=2, ..., 11AM=11
			if (result_data[3] == 12) {
				result_data[3] = 0;
			}
		} else {
			// PM: 12PM=12, 1PM=13, 2PM=14, ..., 11PM=23
			if (result_data[3] != 12) {
				result_data[3] += 12;
			}
		}
	}
	switch (offset_specifier) {
	case StrTimeSpecifier::WEEK_NUMBER_PADDED_SUN_FIRST:
	case StrTimeSpecifier::WEEK_NUMBER_PADDED_MON_FIRST: {
		// Adjust weekday to be 0-based for the week type
		weekday = (weekday + 7 - int(offset_specifier == StrTimeSpecifier::WEEK_NUMBER_PADDED_MON_FIRST)) % 7;
		// Get the start of week 1, move back 7 days and then weekno * 7 + weekday gives the date
		const auto jan1 = Date::FromDate(result_data[0], 1, 1);
		auto yeardate = Date::GetMondayOfCurrentWeek(jan1);
		yeardate -= int(offset_specifier == StrTimeSpecifier::WEEK_NUMBER_PADDED_SUN_FIRST);
		// Is there a week 0?
		yeardate -= 7 * int(yeardate >= jan1);
		yeardate += weekno * 7 + weekday;
		Date::Convert(yeardate, result_data[0], result_data[1], result_data[2]);
		break;
	}
	case StrTimeSpecifier::DAY_OF_YEAR_PADDED:
	case StrTimeSpecifier::DAY_OF_YEAR_DECIMAL: {
		auto yeardate = Date::FromDate(result_data[0], 1, 1);
		yeardate += yearday - 1;
		Date::Convert(yeardate, result_data[0], result_data[1], result_data[2]);
		break;
	}
	case StrTimeSpecifier::DAY_OF_MONTH_PADDED:
	case StrTimeSpecifier::DAY_OF_MONTH:
	case StrTimeSpecifier::MONTH_DECIMAL_PADDED:
	case StrTimeSpecifier::MONTH_DECIMAL:
		// m/d overrides UWw/j
		break;
	default:
		D_ASSERT(offset_specifier == StrTimeSpecifier::WEEKDAY_DECIMAL);
		break;
	}

	return true;
}

struct StrpTimeBindData : public FunctionData {
	explicit StrpTimeBindData(StrpTimeFormat format_p, string format_string_p)
	    : format(std::move(format_p)), format_string(std::move(format_string_p)) {
	}

	StrpTimeFormat format;
	string format_string;

	unique_ptr<FunctionData> Copy() const override {
		return make_unique<StrpTimeBindData>(format, format_string);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const StrpTimeBindData &)other_p;
		return format_string == other.format_string;
	}
};

static unique_ptr<FunctionData> StrpTimeBindFunction(ClientContext &context, ScalarFunction &bound_function,
                                                     vector<unique_ptr<Expression>> &arguments) {
	if (arguments[1]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (!arguments[1]->IsFoldable()) {
		throw InvalidInputException("strptime format must be a constant");
	}
	Value options_str = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
	string format_string = options_str.ToString();
	StrpTimeFormat format;
	if (!options_str.IsNull()) {
		if (options_str.type().id() != LogicalTypeId::VARCHAR) {
			throw InvalidInputException("strptime format must be a string");
		}
		format.format_specifier = format_string;
		string error = StrTimeFormat::ParseFormatSpecifier(format_string, format);
		if (!error.empty()) {
			throw InvalidInputException("Failed to parse format specifier %s: %s", format_string, error);
		}
		if (format.HasFormatSpecifier(StrTimeSpecifier::UTC_OFFSET)) {
			bound_function.return_type = LogicalType::TIMESTAMP_TZ;
		}
	}
	return make_unique<StrpTimeBindData>(format, format_string);
}

StrpTimeFormat::ParseResult StrpTimeFormat::Parse(const string &format_string, const string &text) {
	StrpTimeFormat format;
	format.format_specifier = format_string;
	string error = StrTimeFormat::ParseFormatSpecifier(format_string, format);
	if (!error.empty()) {
		throw InvalidInputException("Failed to parse format specifier %s: %s", format_string, error);
	}
	StrpTimeFormat::ParseResult result;
	if (!format.Parse(text, result)) {
		throw InvalidInputException("Failed to parse string \"%s\" with format specifier \"%s\"", text, format_string);
	}
	return result;
}

string StrpTimeFormat::FormatStrpTimeError(const string &input, idx_t position) {
	if (position == DConstants::INVALID_INDEX) {
		return string();
	}
	return input + "\n" + string(position, ' ') + "^";
}

date_t StrpTimeFormat::ParseResult::ToDate() {
	return Date::FromDate(data[0], data[1], data[2]);
}

timestamp_t StrpTimeFormat::ParseResult::ToTimestamp() {
	date_t date = Date::FromDate(data[0], data[1], data[2]);
	const auto hour_offset = data[7] / Interval::MINS_PER_HOUR;
	const auto mins_offset = data[7] % Interval::MINS_PER_HOUR;
	dtime_t time = Time::FromTime(data[3] - hour_offset, data[4] - mins_offset, data[5], data[6]);
	return Timestamp::FromDatetime(date, time);
}

string StrpTimeFormat::ParseResult::FormatError(string_t input, const string &format_specifier) {
	return StringUtil::Format("Could not parse string \"%s\" according to format specifier \"%s\"\n%s\nError: %s",
	                          input.GetString(), format_specifier,
	                          FormatStrpTimeError(input.GetString(), error_position), error_message);
}

bool StrpTimeFormat::TryParseDate(string_t input, date_t &result, string &error_message) {
	ParseResult parse_result;
	if (!Parse(input, parse_result)) {
		error_message = parse_result.FormatError(input, format_specifier);
		return false;
	}
	result = parse_result.ToDate();
	return true;
}

bool StrpTimeFormat::TryParseTimestamp(string_t input, timestamp_t &result, string &error_message) {
	ParseResult parse_result;
	if (!Parse(input, parse_result)) {
		error_message = parse_result.FormatError(input, format_specifier);
		return false;
	}
	result = parse_result.ToTimestamp();
	return true;
}

date_t StrpTimeFormat::ParseDate(string_t input) {
	ParseResult result;
	if (!Parse(input, result)) {
		throw InvalidInputException(result.FormatError(input, format_specifier));
	}
	return result.ToDate();
}

timestamp_t StrpTimeFormat::ParseTimestamp(string_t input) {
	ParseResult result;
	if (!Parse(input, result)) {
		throw InvalidInputException(result.FormatError(input, format_specifier));
	}
	return result.ToTimestamp();
}

static void StrpTimeFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = (BoundFunctionExpression &)state.expr;
	auto &info = (StrpTimeBindData &)*func_expr.bind_info;

	if (ConstantVector::IsNull(args.data[1])) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		ConstantVector::SetNull(result, true);
		return;
	}
	UnaryExecutor::Execute<string_t, timestamp_t>(args.data[0], result, args.size(),
	                                              [&](string_t input) { return info.format.ParseTimestamp(input); });
}

void StrpTimeFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet strptime("strptime");

	auto fun = ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::TIMESTAMP, StrpTimeFunction,
	                          StrpTimeBindFunction);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	strptime.AddFunction(fun);

	set.AddFunction(strptime);
}

} // namespace duckdb













namespace duckdb {

struct TimeBucket {

	// Use 2000-01-03 00:00:00 (Monday) as origin when bucket_width is days, hours, ... for TimescaleDB compatibility
	// There are 10959 days between 1970-01-01 and 2000-01-03
	constexpr static const int64_t DEFAULT_ORIGIN_MICROS = 10959 * Interval::MICROS_PER_DAY;
	// Use 2000-01-01 as origin when bucket_width is months, years, ... for TimescaleDB compatibility
	// There are 360 months between 1970-01-01 and 2000-01-01
	constexpr static const int32_t DEFAULT_ORIGIN_MONTHS = 360;

	enum struct BucketWidthType { CONVERTIBLE_TO_MICROS, CONVERTIBLE_TO_MONTHS, UNCLASSIFIED };

	static inline BucketWidthType ClassifyBucketWidth(const interval_t bucket_width) {
		if (bucket_width.months == 0 && Interval::GetMicro(bucket_width) > 0) {
			return BucketWidthType::CONVERTIBLE_TO_MICROS;
		} else if (bucket_width.months > 0 && bucket_width.days == 0 && bucket_width.micros == 0) {
			return BucketWidthType::CONVERTIBLE_TO_MONTHS;
		} else {
			return BucketWidthType::UNCLASSIFIED;
		}
	}

	static inline BucketWidthType ClassifyBucketWidthErrorThrow(const interval_t bucket_width) {
		if (bucket_width.months == 0) {
			int64_t bucket_width_micros = Interval::GetMicro(bucket_width);
			if (bucket_width_micros <= 0) {
				throw NotImplementedException("Period must be greater than 0");
			}
			return BucketWidthType::CONVERTIBLE_TO_MICROS;
		} else if (bucket_width.months != 0 && bucket_width.days == 0 && bucket_width.micros == 0) {
			if (bucket_width.months < 0) {
				throw NotImplementedException("Period must be greater than 0");
			}
			return BucketWidthType::CONVERTIBLE_TO_MONTHS;
		} else {
			throw NotImplementedException("Month intervals cannot have day or time component");
		}
	}

	template <typename T>
	static inline int32_t EpochMonths(T ts) {
		date_t ts_date = Cast::template Operation<T, date_t>(ts);
		return (Date::ExtractYear(ts_date) - 1970) * 12 + Date::ExtractMonth(ts_date) - 1;
	}

	static inline timestamp_t WidthConvertibleToMicrosCommon(int64_t bucket_width_micros, int64_t ts_micros,
	                                                         int64_t origin_micros) {
		origin_micros %= bucket_width_micros;
		ts_micros = SubtractOperatorOverflowCheck::Operation<int64_t, int64_t, int64_t>(ts_micros, origin_micros);

		int64_t result_micros = (ts_micros / bucket_width_micros) * bucket_width_micros;
		if (ts_micros < 0 && ts_micros % bucket_width_micros != 0) {
			result_micros =
			    SubtractOperatorOverflowCheck::Operation<int64_t, int64_t, int64_t>(result_micros, bucket_width_micros);
		}
		result_micros += origin_micros;

		return Timestamp::FromEpochMicroSeconds(result_micros);
	}

	static inline date_t WidthConvertibleToMonthsCommon(int32_t bucket_width_months, int32_t ts_months,
	                                                    int32_t origin_months) {
		origin_months %= bucket_width_months;
		ts_months = SubtractOperatorOverflowCheck::Operation<int32_t, int32_t, int32_t>(ts_months, origin_months);

		int32_t result_months = (ts_months / bucket_width_months) * bucket_width_months;
		if (ts_months < 0 && ts_months % bucket_width_months != 0) {
			result_months =
			    SubtractOperatorOverflowCheck::Operation<int32_t, int32_t, int32_t>(result_months, bucket_width_months);
		}
		result_months += origin_months;

		int32_t year =
		    (result_months < 0 && result_months % 12 != 0) ? 1970 + result_months / 12 - 1 : 1970 + result_months / 12;
		int32_t month =
		    (result_months < 0 && result_months % 12 != 0) ? result_months % 12 + 13 : result_months % 12 + 1;

		return Date::FromDate(year, month, 1);
	}

	struct WidthConvertibleToMicrosBinaryOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA bucket_width, TB ts) {
			if (!Value::IsFinite(ts)) {
				return Cast::template Operation<TB, TR>(ts);
			}
			int64_t bucket_width_micros = Interval::GetMicro(bucket_width);
			int64_t ts_micros = Timestamp::GetEpochMicroSeconds(Cast::template Operation<TB, timestamp_t>(ts));
			return Cast::template Operation<timestamp_t, TR>(
			    WidthConvertibleToMicrosCommon(bucket_width_micros, ts_micros, DEFAULT_ORIGIN_MICROS));
		}
	};

	struct WidthConvertibleToMonthsBinaryOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA bucket_width, TB ts) {
			if (!Value::IsFinite(ts)) {
				return Cast::template Operation<TB, TR>(ts);
			}
			int32_t ts_months = EpochMonths(ts);
			return Cast::template Operation<date_t, TR>(
			    WidthConvertibleToMonthsCommon(bucket_width.months, ts_months, DEFAULT_ORIGIN_MONTHS));
		}
	};

	struct BinaryOperator {
		template <class TA, class TB, class TR>
		static inline TR Operation(TA bucket_width, TB ts) {
			BucketWidthType bucket_width_type = ClassifyBucketWidthErrorThrow(bucket_width);
			switch (bucket_width_type) {
			case BucketWidthType::CONVERTIBLE_TO_MICROS:
				return WidthConvertibleToMicrosBinaryOperator::Operation<TA, TB, TR>(bucket_width, ts);
			case BucketWidthType::CONVERTIBLE_TO_MONTHS:
				return WidthConvertibleToMonthsBinaryOperator::Operation<TA, TB, TR>(bucket_width, ts);
			default:
				throw NotImplementedException("Bucket type not implemented for TIME_BUCKET");
			}
		}
	};

	struct OffsetWidthConvertibleToMicrosTernaryOperator {
		template <class TA, class TB, class TC, class TR>
		static inline TR Operation(TA bucket_width, TB ts, TC offset) {
			if (!Value::IsFinite(ts)) {
				return Cast::template Operation<TB, TR>(ts);
			}
			int64_t bucket_width_micros = Interval::GetMicro(bucket_width);
			int64_t ts_micros = Timestamp::GetEpochMicroSeconds(
			    Interval::Add(Cast::template Operation<TB, timestamp_t>(ts), Interval::Invert(offset)));
			return Cast::template Operation<timestamp_t, TR>(Interval::Add(
			    WidthConvertibleToMicrosCommon(bucket_width_micros, ts_micros, DEFAULT_ORIGIN_MICROS), offset));
		}
	};

	struct OffsetWidthConvertibleToMonthsTernaryOperator {
		template <class TA, class TB, class TC, class TR>
		static inline TR Operation(TA bucket_width, TB ts, TC offset) {
			if (!Value::IsFinite(ts)) {
				return Cast::template Operation<TB, TR>(ts);
			}
			int32_t ts_months = EpochMonths(Interval::Add(ts, Interval::Invert(offset)));
			return Interval::Add(Cast::template Operation<date_t, TR>(WidthConvertibleToMonthsCommon(
			                         bucket_width.months, ts_months, DEFAULT_ORIGIN_MONTHS)),
			                     offset);
		}
	};

	struct OffsetTernaryOperator {
		template <class TA, class TB, class TC, class TR>
		static inline TR Operation(TA bucket_width, TB ts, TC offset) {
			BucketWidthType bucket_width_type = ClassifyBucketWidthErrorThrow(bucket_width);
			switch (bucket_width_type) {
			case BucketWidthType::CONVERTIBLE_TO_MICROS:
				return OffsetWidthConvertibleToMicrosTernaryOperator::Operation<TA, TB, TC, TR>(bucket_width, ts,
				                                                                                offset);
			case BucketWidthType::CONVERTIBLE_TO_MONTHS:
				return OffsetWidthConvertibleToMonthsTernaryOperator::Operation<TA, TB, TC, TR>(bucket_width, ts,
				                                                                                offset);
			default:
				throw NotImplementedException("Bucket type not implemented for TIME_BUCKET");
			}
		}
	};

	struct OriginWidthConvertibleToMicrosTernaryOperator {
		template <class TA, class TB, class TC, class TR>
		static inline TR Operation(TA bucket_width, TB ts, TC origin) {
			if (!Value::IsFinite(ts)) {
				return Cast::template Operation<TB, TR>(ts);
			}
			int64_t bucket_width_micros = Interval::GetMicro(bucket_width);
			int64_t ts_micros = Timestamp::GetEpochMicroSeconds(Cast::template Operation<TB, timestamp_t>(ts));
			int64_t origin_micros = Timestamp::GetEpochMicroSeconds(Cast::template Operation<TB, timestamp_t>(origin));
			return Cast::template Operation<timestamp_t, TR>(
			    WidthConvertibleToMicrosCommon(bucket_width_micros, ts_micros, origin_micros));
		}
	};

	struct OriginWidthConvertibleToMonthsTernaryOperator {
		template <class TA, class TB, class TC, class TR>
		static inline TR Operation(TA bucket_width, TB ts, TC origin) {
			if (!Value::IsFinite(ts)) {
				return Cast::template Operation<TB, TR>(ts);
			}
			int32_t ts_months = EpochMonths(ts);
			int32_t origin_months = EpochMonths(origin);
			return Cast::template Operation<date_t, TR>(
			    WidthConvertibleToMonthsCommon(bucket_width.months, ts_months, origin_months));
		}
	};

	struct OriginTernaryOperator {
		template <class TA, class TB, class TC, class TR>
		static inline TR Operation(TA bucket_width, TB ts, TC origin, ValidityMask &mask, idx_t idx) {
			if (!Value::IsFinite(origin)) {
				mask.SetInvalid(idx);
				return TR();
			}
			BucketWidthType bucket_width_type = ClassifyBucketWidthErrorThrow(bucket_width);
			switch (bucket_width_type) {
			case BucketWidthType::CONVERTIBLE_TO_MICROS:
				return OriginWidthConvertibleToMicrosTernaryOperator::Operation<TA, TB, TC, TR>(bucket_width, ts,
				                                                                                origin);
			case BucketWidthType::CONVERTIBLE_TO_MONTHS:
				return OriginWidthConvertibleToMonthsTernaryOperator::Operation<TA, TB, TC, TR>(bucket_width, ts,
				                                                                                origin);
			default:
				throw NotImplementedException("Bucket type not implemented for TIME_BUCKET");
			}
		}
	};
};

template <typename T>
static void TimeBucketFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 2);

	auto &bucket_width_arg = args.data[0];
	auto &ts_arg = args.data[1];

	if (bucket_width_arg.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		if (ConstantVector::IsNull(bucket_width_arg)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
		} else {
			interval_t bucket_width = *ConstantVector::GetData<interval_t>(bucket_width_arg);
			TimeBucket::BucketWidthType bucket_width_type = TimeBucket::ClassifyBucketWidth(bucket_width);
			switch (bucket_width_type) {
			case TimeBucket::BucketWidthType::CONVERTIBLE_TO_MICROS:
				BinaryExecutor::Execute<interval_t, T, T>(
				    bucket_width_arg, ts_arg, result, args.size(),
				    TimeBucket::WidthConvertibleToMicrosBinaryOperator::Operation<interval_t, T, T>);
				break;
			case TimeBucket::BucketWidthType::CONVERTIBLE_TO_MONTHS:
				BinaryExecutor::Execute<interval_t, T, T>(
				    bucket_width_arg, ts_arg, result, args.size(),
				    TimeBucket::WidthConvertibleToMonthsBinaryOperator::Operation<interval_t, T, T>);
				break;
			case TimeBucket::BucketWidthType::UNCLASSIFIED:
				BinaryExecutor::Execute<interval_t, T, T>(bucket_width_arg, ts_arg, result, args.size(),
				                                          TimeBucket::BinaryOperator::Operation<interval_t, T, T>);
				break;
			default:
				throw NotImplementedException("Bucket type not implemented for TIME_BUCKET");
			}
		}
	} else {
		BinaryExecutor::Execute<interval_t, T, T>(bucket_width_arg, ts_arg, result, args.size(),
		                                          TimeBucket::BinaryOperator::Operation<interval_t, T, T>);
	}
}

template <typename T>
static void TimeBucketOffsetFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 3);

	auto &bucket_width_arg = args.data[0];
	auto &ts_arg = args.data[1];
	auto &offset_arg = args.data[2];

	if (bucket_width_arg.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		if (ConstantVector::IsNull(bucket_width_arg)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
		} else {
			interval_t bucket_width = *ConstantVector::GetData<interval_t>(bucket_width_arg);
			TimeBucket::BucketWidthType bucket_width_type = TimeBucket::ClassifyBucketWidth(bucket_width);
			switch (bucket_width_type) {
			case TimeBucket::BucketWidthType::CONVERTIBLE_TO_MICROS:
				TernaryExecutor::Execute<interval_t, T, interval_t, T>(
				    bucket_width_arg, ts_arg, offset_arg, result, args.size(),
				    TimeBucket::OffsetWidthConvertibleToMicrosTernaryOperator::Operation<interval_t, T, interval_t, T>);
				break;
			case TimeBucket::BucketWidthType::CONVERTIBLE_TO_MONTHS:
				TernaryExecutor::Execute<interval_t, T, interval_t, T>(
				    bucket_width_arg, ts_arg, offset_arg, result, args.size(),
				    TimeBucket::OffsetWidthConvertibleToMonthsTernaryOperator::Operation<interval_t, T, interval_t, T>);
				break;
			case TimeBucket::BucketWidthType::UNCLASSIFIED:
				TernaryExecutor::Execute<interval_t, T, interval_t, T>(
				    bucket_width_arg, ts_arg, offset_arg, result, args.size(),
				    TimeBucket::OffsetTernaryOperator::Operation<interval_t, T, interval_t, T>);
				break;
			default:
				throw NotImplementedException("Bucket type not implemented for TIME_BUCKET");
			}
		}
	} else {
		TernaryExecutor::Execute<interval_t, T, interval_t, T>(
		    bucket_width_arg, ts_arg, offset_arg, result, args.size(),
		    TimeBucket::OffsetTernaryOperator::Operation<interval_t, T, interval_t, T>);
	}
}

template <typename T>
static void TimeBucketOriginFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 3);

	auto &bucket_width_arg = args.data[0];
	auto &ts_arg = args.data[1];
	auto &origin_arg = args.data[2];

	if (bucket_width_arg.GetVectorType() == VectorType::CONSTANT_VECTOR &&
	    origin_arg.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		if (ConstantVector::IsNull(bucket_width_arg) || ConstantVector::IsNull(origin_arg) ||
		    !Value::IsFinite(*ConstantVector::GetData<T>(origin_arg))) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
		} else {
			interval_t bucket_width = *ConstantVector::GetData<interval_t>(bucket_width_arg);
			TimeBucket::BucketWidthType bucket_width_type = TimeBucket::ClassifyBucketWidth(bucket_width);
			switch (bucket_width_type) {
			case TimeBucket::BucketWidthType::CONVERTIBLE_TO_MICROS:
				TernaryExecutor::Execute<interval_t, T, T, T>(
				    bucket_width_arg, ts_arg, origin_arg, result, args.size(),
				    TimeBucket::OriginWidthConvertibleToMicrosTernaryOperator::Operation<interval_t, T, T, T>);
				break;
			case TimeBucket::BucketWidthType::CONVERTIBLE_TO_MONTHS:
				TernaryExecutor::Execute<interval_t, T, T, T>(
				    bucket_width_arg, ts_arg, origin_arg, result, args.size(),
				    TimeBucket::OriginWidthConvertibleToMonthsTernaryOperator::Operation<interval_t, T, T, T>);
				break;
			case TimeBucket::BucketWidthType::UNCLASSIFIED:
				TernaryExecutor::ExecuteWithNulls<interval_t, T, T, T>(
				    bucket_width_arg, ts_arg, origin_arg, result, args.size(),
				    TimeBucket::OriginTernaryOperator::Operation<interval_t, T, T, T>);
				break;
			default:
				throw NotImplementedException("Bucket type not implemented for TIME_BUCKET");
			}
		}
	} else {
		TernaryExecutor::ExecuteWithNulls<interval_t, T, T, T>(
		    bucket_width_arg, ts_arg, origin_arg, result, args.size(),
		    TimeBucket::OriginTernaryOperator::Operation<interval_t, T, T, T>);
	}
}

void TimeBucketFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet time_bucket("time_bucket");
	time_bucket.AddFunction(
	    ScalarFunction({LogicalType::INTERVAL, LogicalType::DATE}, LogicalType::DATE, TimeBucketFunction<date_t>));
	time_bucket.AddFunction(ScalarFunction({LogicalType::INTERVAL, LogicalType::TIMESTAMP}, LogicalType::TIMESTAMP,
	                                       TimeBucketFunction<timestamp_t>));
	time_bucket.AddFunction(ScalarFunction({LogicalType::INTERVAL, LogicalType::DATE, LogicalType::INTERVAL},
	                                       LogicalType::DATE, TimeBucketOffsetFunction<date_t>));
	time_bucket.AddFunction(ScalarFunction({LogicalType::INTERVAL, LogicalType::TIMESTAMP, LogicalType::INTERVAL},
	                                       LogicalType::TIMESTAMP, TimeBucketOffsetFunction<timestamp_t>));
	time_bucket.AddFunction(ScalarFunction({LogicalType::INTERVAL, LogicalType::DATE, LogicalType::DATE},
	                                       LogicalType::DATE, TimeBucketOriginFunction<date_t>));
	time_bucket.AddFunction(ScalarFunction({LogicalType::INTERVAL, LogicalType::TIMESTAMP, LogicalType::TIMESTAMP},
	                                       LogicalType::TIMESTAMP, TimeBucketOriginFunction<timestamp_t>));

	set.AddFunction(time_bucket);
}

} // namespace duckdb




namespace duckdb {

struct ToYearsOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		interval_t result;
		result.days = 0;
		result.micros = 0;
		if (!TryMultiplyOperator::Operation<int32_t, int32_t, int32_t>(input, Interval::MONTHS_PER_YEAR,
		                                                               result.months)) {
			throw OutOfRangeException("Interval value %d years out of range", input);
		}
		return result;
	}
};

struct ToMonthsOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		interval_t result;
		result.months = input;
		result.days = 0;
		result.micros = 0;
		return result;
	}
};

struct ToDaysOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		interval_t result;
		result.months = 0;
		result.days = input;
		result.micros = 0;
		return result;
	}
};

struct ToHoursOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		interval_t result;
		result.months = 0;
		result.days = 0;
		if (!TryMultiplyOperator::Operation<int64_t, int64_t, int64_t>(input, Interval::MICROS_PER_HOUR,
		                                                               result.micros)) {
			throw OutOfRangeException("Interval value %d hours out of range", input);
		}
		return result;
	}
};

struct ToMinutesOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		interval_t result;
		result.months = 0;
		result.days = 0;
		if (!TryMultiplyOperator::Operation<int64_t, int64_t, int64_t>(input, Interval::MICROS_PER_MINUTE,
		                                                               result.micros)) {
			throw OutOfRangeException("Interval value %d minutes out of range", input);
		}
		return result;
	}
};

struct ToSecondsOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		interval_t result;
		result.months = 0;
		result.days = 0;
		if (!TryMultiplyOperator::Operation<int64_t, int64_t, int64_t>(input, Interval::MICROS_PER_SEC,
		                                                               result.micros)) {
			throw OutOfRangeException("Interval value %d seconds out of range", input);
		}
		return result;
	}
};

struct ToMilliSecondsOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		interval_t result;
		result.months = 0;
		result.days = 0;
		if (!TryMultiplyOperator::Operation<int64_t, int64_t, int64_t>(input, Interval::MICROS_PER_MSEC,
		                                                               result.micros)) {
			throw OutOfRangeException("Interval value %d milliseconds out of range", input);
		}
		return result;
	}
};

struct ToMicroSecondsOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		interval_t result;
		result.months = 0;
		result.days = 0;
		result.micros = input;
		return result;
	}
};

void ToIntervalFun::RegisterFunction(BuiltinFunctions &set) {
	// register the individual operators
	set.AddFunction(ScalarFunction("to_years", {LogicalType::INTEGER}, LogicalType::INTERVAL,
	                               ScalarFunction::UnaryFunction<int32_t, interval_t, ToYearsOperator>));
	set.AddFunction(ScalarFunction("to_months", {LogicalType::INTEGER}, LogicalType::INTERVAL,
	                               ScalarFunction::UnaryFunction<int32_t, interval_t, ToMonthsOperator>));
	set.AddFunction(ScalarFunction("to_days", {LogicalType::INTEGER}, LogicalType::INTERVAL,
	                               ScalarFunction::UnaryFunction<int32_t, interval_t, ToDaysOperator>));
	set.AddFunction(ScalarFunction("to_hours", {LogicalType::BIGINT}, LogicalType::INTERVAL,
	                               ScalarFunction::UnaryFunction<int64_t, interval_t, ToHoursOperator>));
	set.AddFunction(ScalarFunction("to_minutes", {LogicalType::BIGINT}, LogicalType::INTERVAL,
	                               ScalarFunction::UnaryFunction<int64_t, interval_t, ToMinutesOperator>));
	set.AddFunction(ScalarFunction("to_seconds", {LogicalType::BIGINT}, LogicalType::INTERVAL,
	                               ScalarFunction::UnaryFunction<int64_t, interval_t, ToSecondsOperator>));
	set.AddFunction(ScalarFunction("to_milliseconds", {LogicalType::BIGINT}, LogicalType::INTERVAL,
	                               ScalarFunction::UnaryFunction<int64_t, interval_t, ToMilliSecondsOperator>));
	set.AddFunction(ScalarFunction("to_microseconds", {LogicalType::BIGINT}, LogicalType::INTERVAL,
	                               ScalarFunction::UnaryFunction<int64_t, interval_t, ToMicroSecondsOperator>));
}

} // namespace duckdb


namespace duckdb {

void BuiltinFunctions::RegisterDateFunctions() {
	Register<AgeFun>();
	Register<DateDiffFun>();
	Register<DatePartFun>();
	Register<DateSubFun>();
	Register<DateTruncFun>();
	Register<CurrentTimeFun>();
	Register<CurrentDateFun>();
	Register<CurrentTimestampFun>();
	Register<EpochFun>();
	Register<MakeDateFun>();
	Register<StrfTimeFun>();
	Register<StrpTimeFun>();
	Register<TimeBucketFun>();
	Register<ToIntervalFun>();
}

} // namespace duckdb


namespace duckdb {

static void EnumFirstFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.GetTypes().size() == 1);
	auto &enum_vector = EnumType::GetValuesInsertOrder(input.GetTypes()[0]);
	auto val = Value(enum_vector.GetValue(0));
	result.Reference(val);
}

static void EnumLastFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.GetTypes().size() == 1);
	auto enum_size = EnumType::GetSize(input.GetTypes()[0]);
	auto &enum_vector = EnumType::GetValuesInsertOrder(input.GetTypes()[0]);
	auto val = Value(enum_vector.GetValue(enum_size - 1));
	result.Reference(val);
}

static void EnumRangeFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.GetTypes().size() == 1);
	auto enum_size = EnumType::GetSize(input.GetTypes()[0]);
	auto &enum_vector = EnumType::GetValuesInsertOrder(input.GetTypes()[0]);
	vector<Value> enum_values;
	for (idx_t i = 0; i < enum_size; i++) {
		enum_values.emplace_back(enum_vector.GetValue(i));
	}
	auto val = Value::LIST(enum_values);
	result.Reference(val);
}

static void EnumRangeBoundaryFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.GetTypes().size() == 2);
	idx_t start, end;
	auto first_param = input.GetValue(0, 0);
	auto second_param = input.GetValue(1, 0);

	auto &enum_vector = first_param.IsNull() ? EnumType::GetValuesInsertOrder(input.GetTypes()[1])
	                                         : EnumType::GetValuesInsertOrder(input.GetTypes()[0]);

	if (first_param.IsNull()) {
		start = 0;
	} else {
		start = first_param.GetValue<uint32_t>();
	}
	if (second_param.IsNull()) {
		end = EnumType::GetSize(input.GetTypes()[0]);
	} else {
		end = second_param.GetValue<uint32_t>() + 1;
	}
	vector<Value> enum_values;
	for (idx_t i = start; i < end; i++) {
		enum_values.emplace_back(enum_vector.GetValue(i));
	}
	Value val;
	if (enum_values.empty()) {
		val = Value::EMPTYLIST(LogicalType::VARCHAR);
	} else {
		val = Value::LIST(enum_values);
	}
	result.Reference(val);
}

static void EnumCodeFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.GetTypes().size() == 1);
	result.Reinterpret(input.data[0]);
}

static void CheckEnumParameter(const Expression &expr) {
	if (expr.HasParameter()) {
		throw ParameterNotResolvedException();
	}
}

unique_ptr<FunctionData> BindEnumFunction(ClientContext &context, ScalarFunction &bound_function,
                                          vector<unique_ptr<Expression>> &arguments) {
	CheckEnumParameter(*arguments[0]);
	if (arguments[0]->return_type.id() != LogicalTypeId::ENUM) {
		throw BinderException("This function needs an ENUM as an argument");
	}
	return nullptr;
}

unique_ptr<FunctionData> BindEnumCodeFunction(ClientContext &context, ScalarFunction &bound_function,
                                              vector<unique_ptr<Expression>> &arguments) {
	CheckEnumParameter(*arguments[0]);
	if (arguments[0]->return_type.id() != LogicalTypeId::ENUM) {
		throw BinderException("This function needs an ENUM as an argument");
	}

	auto phy_type = EnumType::GetPhysicalType(arguments[0]->return_type);
	switch (phy_type) {
	case PhysicalType::UINT8:
		bound_function.return_type = LogicalType(LogicalTypeId::UTINYINT);
		break;
	case PhysicalType::UINT16:
		bound_function.return_type = LogicalType(LogicalTypeId::USMALLINT);
		break;
	case PhysicalType::UINT32:
		bound_function.return_type = LogicalType(LogicalTypeId::UINTEGER);
		break;
	case PhysicalType::UINT64:
		bound_function.return_type = LogicalType(LogicalTypeId::UBIGINT);
		break;
	default:
		throw InternalException("Unsupported Enum Internal Type");
	}

	return nullptr;
}

unique_ptr<FunctionData> BindEnumRangeBoundaryFunction(ClientContext &context, ScalarFunction &bound_function,
                                                       vector<unique_ptr<Expression>> &arguments) {
	CheckEnumParameter(*arguments[0]);
	CheckEnumParameter(*arguments[1]);
	if (arguments[0]->return_type.id() != LogicalTypeId::ENUM && arguments[0]->return_type != LogicalType::SQLNULL) {
		throw BinderException("This function needs an ENUM as an argument");
	}
	if (arguments[1]->return_type.id() != LogicalTypeId::ENUM && arguments[1]->return_type != LogicalType::SQLNULL) {
		throw BinderException("This function needs an ENUM as an argument");
	}
	if (arguments[0]->return_type == LogicalType::SQLNULL && arguments[1]->return_type == LogicalType::SQLNULL) {
		throw BinderException("This function needs an ENUM as an argument");
	}
	if (arguments[0]->return_type.id() == LogicalTypeId::ENUM &&
	    arguments[1]->return_type.id() == LogicalTypeId::ENUM &&
	    arguments[0]->return_type != arguments[1]->return_type) {
		throw BinderException("The parameters need to link to ONLY one enum OR be NULL ");
	}
	return nullptr;
}

void EnumFirst::RegisterFunction(BuiltinFunctions &set) {
	auto fun =
	    ScalarFunction("enum_first", {LogicalType::ANY}, LogicalType::VARCHAR, EnumFirstFunction, BindEnumFunction);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	set.AddFunction(fun);
}

void EnumLast::RegisterFunction(BuiltinFunctions &set) {
	auto fun =
	    ScalarFunction("enum_last", {LogicalType::ANY}, LogicalType::VARCHAR, EnumLastFunction, BindEnumFunction);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	set.AddFunction(fun);
}

void EnumCode::RegisterFunction(BuiltinFunctions &set) {
	auto fun =
	    ScalarFunction("enum_code", {LogicalType::ANY}, LogicalType::ANY, EnumCodeFunction, BindEnumCodeFunction);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	set.AddFunction(fun);
}

void EnumRange::RegisterFunction(BuiltinFunctions &set) {
	auto fun = ScalarFunction("enum_range", {LogicalType::ANY}, LogicalType::LIST(LogicalType::VARCHAR),
	                          EnumRangeFunction, BindEnumFunction);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	set.AddFunction(fun);
}

void EnumRangeBoundary::RegisterFunction(BuiltinFunctions &set) {
	auto fun = ScalarFunction("enum_range_boundary", {LogicalType::ANY, LogicalType::ANY},
	                          LogicalType::LIST(LogicalType::VARCHAR), EnumRangeBoundaryFunction,
	                          BindEnumRangeBoundaryFunction);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	set.AddFunction(fun);
}

} // namespace duckdb


namespace duckdb {

void BuiltinFunctions::RegisterEnumFunctions() {
	Register<EnumFirst>();
	Register<EnumLast>();
	Register<EnumCode>();
	Register<EnumRange>();
	Register<EnumRangeBoundary>();
}

} // namespace duckdb



namespace duckdb {

static void AliasFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = (BoundFunctionExpression &)state.expr;
	Value v(state.expr.alias.empty() ? func_expr.children[0]->GetName() : state.expr.alias);
	result.Reference(v);
}

void AliasFun::RegisterFunction(BuiltinFunctions &set) {
	auto fun = ScalarFunction("alias", {LogicalType::ANY}, LogicalType::VARCHAR, AliasFunction);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	set.AddFunction(fun);
}

} // namespace duckdb





namespace duckdb {

struct ConstantOrNullBindData : public FunctionData {
	explicit ConstantOrNullBindData(Value val) : value(std::move(val)) {
	}

	Value value;

public:
	unique_ptr<FunctionData> Copy() const override {
		return make_unique<ConstantOrNullBindData>(value);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const ConstantOrNullBindData &)other_p;
		return value == other.value;
	}
};

static void ConstantOrNullFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = (BoundFunctionExpression &)state.expr;
	auto &info = (ConstantOrNullBindData &)*func_expr.bind_info;
	result.Reference(info.value);
	for (idx_t idx = 1; idx < args.ColumnCount(); idx++) {
		switch (args.data[idx].GetVectorType()) {
		case VectorType::FLAT_VECTOR: {
			auto &input_mask = FlatVector::Validity(args.data[idx]);
			if (!input_mask.AllValid()) {
				// there are null values: need to merge them into the result
				result.Flatten(args.size());
				auto &result_mask = FlatVector::Validity(result);
				result_mask.Combine(input_mask, args.size());
			}
			break;
		}
		case VectorType::CONSTANT_VECTOR: {
			if (ConstantVector::IsNull(args.data[idx])) {
				// input is constant null, return constant null
				result.Reference(info.value);
				ConstantVector::SetNull(result, true);
				return;
			}
			break;
		}
		default: {
			UnifiedVectorFormat vdata;
			args.data[idx].ToUnifiedFormat(args.size(), vdata);
			if (!vdata.validity.AllValid()) {
				result.Flatten(args.size());
				auto &result_mask = FlatVector::Validity(result);
				for (idx_t i = 0; i < args.size(); i++) {
					if (!vdata.validity.RowIsValid(vdata.sel->get_index(i))) {
						result_mask.SetInvalid(i);
					}
				}
			}
			break;
		}
		}
	}
}

ScalarFunction ConstantOrNull::GetFunction(const LogicalType &return_type) {
	return ScalarFunction("constant_or_null", {return_type, LogicalType::ANY}, return_type, ConstantOrNullFunction);
}

unique_ptr<FunctionData> ConstantOrNull::Bind(Value value) {
	return make_unique<ConstantOrNullBindData>(std::move(value));
}

bool ConstantOrNull::IsConstantOrNull(BoundFunctionExpression &expr, const Value &val) {
	if (expr.function.name != "constant_or_null") {
		return false;
	}
	D_ASSERT(expr.bind_info);
	auto &bind_data = (ConstantOrNullBindData &)*expr.bind_info;
	D_ASSERT(bind_data.value.type() == val.type());
	return bind_data.value == val;
}

unique_ptr<FunctionData> ConstantOrNullBind(ClientContext &context, ScalarFunction &bound_function,
                                            vector<unique_ptr<Expression>> &arguments) {
	if (arguments[0]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (!arguments[0]->IsFoldable()) {
		throw BinderException("ConstantOrNull requires a constant input");
	}
	D_ASSERT(arguments.size() >= 2);
	auto value = ExpressionExecutor::EvaluateScalar(context, *arguments[0]);
	bound_function.return_type = arguments[0]->return_type;
	return make_unique<ConstantOrNullBindData>(std::move(value));
}

void ConstantOrNull::RegisterFunction(BuiltinFunctions &set) {
	auto fun = ConstantOrNull::GetFunction(LogicalType::ANY);
	fun.bind = ConstantOrNullBind;
	fun.varargs = LogicalType::ANY;
	set.AddFunction(fun);
}

} // namespace duckdb







namespace duckdb {

struct CurrentSettingBindData : public FunctionData {
	explicit CurrentSettingBindData(Value value_p) : value(std::move(value_p)) {
	}

	Value value;

public:
	unique_ptr<FunctionData> Copy() const override {
		return make_unique<CurrentSettingBindData>(value);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const CurrentSettingBindData &)other_p;
		return Value::NotDistinctFrom(value, other.value);
	}
};

static void CurrentSettingFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = (BoundFunctionExpression &)state.expr;
	auto &info = (CurrentSettingBindData &)*func_expr.bind_info;
	result.Reference(info.value);
}

unique_ptr<FunctionData> CurrentSettingBind(ClientContext &context, ScalarFunction &bound_function,
                                            vector<unique_ptr<Expression>> &arguments) {

	auto &key_child = arguments[0];
	if (key_child->return_type.id() == LogicalTypeId::UNKNOWN) {
		throw ParameterNotResolvedException();
	}
	if (key_child->return_type.id() != LogicalTypeId::VARCHAR ||
	    key_child->return_type.id() != LogicalTypeId::VARCHAR || !key_child->IsFoldable()) {
		throw ParserException("Key name for current_setting needs to be a constant string");
	}
	Value key_val = ExpressionExecutor::EvaluateScalar(context, *key_child.get());
	D_ASSERT(key_val.type().id() == LogicalTypeId::VARCHAR);
	auto &key_str = StringValue::Get(key_val);
	if (key_val.IsNull() || key_str.empty()) {
		throw ParserException("Key name for current_setting needs to be neither NULL nor empty");
	}

	auto key = StringUtil::Lower(key_str);
	Value val;
	if (!context.TryGetCurrentSetting(key, val)) {
		throw InvalidInputException("unrecognized configuration parameter \"%s\"", key_str);
	}

	bound_function.return_type = val.type();
	return make_unique<CurrentSettingBindData>(val);
}

void CurrentSettingFun::RegisterFunction(BuiltinFunctions &set) {
	auto fun = ScalarFunction("current_setting", {LogicalType::VARCHAR}, LogicalType::ANY, CurrentSettingFunction,
	                          CurrentSettingBind);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	set.AddFunction(fun);
}

} // namespace duckdb


namespace duckdb {

static void HashFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	args.Hash(result);
}

void HashFun::RegisterFunction(BuiltinFunctions &set) {
	auto hash_fun = ScalarFunction("hash", {LogicalType::ANY}, LogicalType::HASH, HashFunction);
	hash_fun.varargs = LogicalType::ANY;
	hash_fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	set.AddFunction(hash_fun);
}

} // namespace duckdb



namespace duckdb {

template <class OP>
struct LeastOperator {
	template <class T>
	static T Operation(T left, T right) {
		return OP::Operation(left, right) ? left : right;
	}
};

template <class T, class OP, bool IS_STRING = false>
static void LeastGreatestFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	if (args.ColumnCount() == 1) {
		// single input: nop
		result.Reference(args.data[0]);
		return;
	}
	auto result_type = VectorType::CONSTANT_VECTOR;
	for (idx_t col_idx = 0; col_idx < args.ColumnCount(); col_idx++) {
		if (args.data[col_idx].GetVectorType() != VectorType::CONSTANT_VECTOR) {
			// non-constant input: result is not a constant vector
			result_type = VectorType::FLAT_VECTOR;
		}
		if (IS_STRING) {
			// for string vectors we add a reference to the heap of the children
			StringVector::AddHeapReference(result, args.data[col_idx]);
		}
	}

	auto result_data = FlatVector::GetData<T>(result);
	auto &result_mask = FlatVector::Validity(result);
	// copy over the first column
	bool result_has_value[STANDARD_VECTOR_SIZE];
	{
		UnifiedVectorFormat vdata;
		args.data[0].ToUnifiedFormat(args.size(), vdata);
		auto input_data = (T *)vdata.data;
		for (idx_t i = 0; i < args.size(); i++) {
			auto vindex = vdata.sel->get_index(i);
			if (vdata.validity.RowIsValid(vindex)) {
				result_data[i] = input_data[vindex];
				result_has_value[i] = true;
			} else {
				result_has_value[i] = false;
			}
		}
	}
	// now handle the remainder of the columns
	for (idx_t col_idx = 1; col_idx < args.ColumnCount(); col_idx++) {
		if (args.data[col_idx].GetVectorType() == VectorType::CONSTANT_VECTOR &&
		    ConstantVector::IsNull(args.data[col_idx])) {
			// ignore null vector
			continue;
		}

		UnifiedVectorFormat vdata;
		args.data[col_idx].ToUnifiedFormat(args.size(), vdata);

		auto input_data = (T *)vdata.data;
		if (!vdata.validity.AllValid()) {
			// potential new null entries: have to check the null mask
			for (idx_t i = 0; i < args.size(); i++) {
				auto vindex = vdata.sel->get_index(i);
				if (vdata.validity.RowIsValid(vindex)) {
					// not a null entry: perform the operation and add to new set
					auto ivalue = input_data[vindex];
					if (!result_has_value[i] || OP::template Operation<T>(ivalue, result_data[i])) {
						result_has_value[i] = true;
						result_data[i] = ivalue;
					}
				}
			}
		} else {
			// no new null entries: only need to perform the operation
			for (idx_t i = 0; i < args.size(); i++) {
				auto vindex = vdata.sel->get_index(i);

				auto ivalue = input_data[vindex];
				if (!result_has_value[i] || OP::template Operation<T>(ivalue, result_data[i])) {
					result_has_value[i] = true;
					result_data[i] = ivalue;
				}
			}
		}
	}
	for (idx_t i = 0; i < args.size(); i++) {
		if (!result_has_value[i]) {
			result_mask.SetInvalid(i);
		}
	}
	result.SetVectorType(result_type);
}

template <typename T, class OP>
ScalarFunction GetLeastGreatestFunction(const LogicalType &type) {
	return ScalarFunction({type}, type, LeastGreatestFunction<T, OP>, nullptr, nullptr, nullptr, nullptr, type,
	                      FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING);
}

template <class OP>
static void RegisterLeastGreatest(BuiltinFunctions &set, const string &fun_name) {
	ScalarFunctionSet fun_set(fun_name);
	fun_set.AddFunction(ScalarFunction({LogicalType::BIGINT}, LogicalType::BIGINT, LeastGreatestFunction<int64_t, OP>,
	                                   nullptr, nullptr, nullptr, nullptr, LogicalType::BIGINT,
	                                   FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING));
	fun_set.AddFunction(ScalarFunction(
	    {LogicalType::HUGEINT}, LogicalType::HUGEINT, LeastGreatestFunction<hugeint_t, OP>, nullptr, nullptr, nullptr,
	    nullptr, LogicalType::HUGEINT, FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING));
	fun_set.AddFunction(ScalarFunction({LogicalType::DOUBLE}, LogicalType::DOUBLE, LeastGreatestFunction<double, OP>,
	                                   nullptr, nullptr, nullptr, nullptr, LogicalType::DOUBLE,
	                                   FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING));
	fun_set.AddFunction(ScalarFunction({LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                                   LeastGreatestFunction<string_t, OP, true>, nullptr, nullptr, nullptr, nullptr,
	                                   LogicalType::VARCHAR, FunctionSideEffects::NO_SIDE_EFFECTS,
	                                   FunctionNullHandling::SPECIAL_HANDLING));

	fun_set.AddFunction(GetLeastGreatestFunction<timestamp_t, OP>(LogicalType::TIMESTAMP));
	fun_set.AddFunction(GetLeastGreatestFunction<time_t, OP>(LogicalType::TIME));
	fun_set.AddFunction(GetLeastGreatestFunction<date_t, OP>(LogicalType::DATE));

	fun_set.AddFunction(GetLeastGreatestFunction<timestamp_t, OP>(LogicalType::TIMESTAMP_TZ));
	fun_set.AddFunction(GetLeastGreatestFunction<time_t, OP>(LogicalType::TIME_TZ));

	set.AddFunction(fun_set);
}

void LeastFun::RegisterFunction(BuiltinFunctions &set) {
	RegisterLeastGreatest<duckdb::LessThan>(set, "least");
}

void GreatestFun::RegisterFunction(BuiltinFunctions &set) {
	RegisterLeastGreatest<duckdb::GreaterThan>(set, "greatest");
}

} // namespace duckdb



namespace duckdb {

struct StatsBindData : public FunctionData {
	explicit StatsBindData(string stats_p = string()) : stats(std::move(stats_p)) {
	}

	string stats;

public:
	unique_ptr<FunctionData> Copy() const override {
		return make_unique<StatsBindData>(stats);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const StatsBindData &)other_p;
		return stats == other.stats;
	}
};

static void StatsFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = (BoundFunctionExpression &)state.expr;
	auto &info = (StatsBindData &)*func_expr.bind_info;
	if (info.stats.empty()) {
		info.stats = "No statistics";
	}
	Value v(info.stats);
	result.Reference(v);
}

unique_ptr<FunctionData> StatsBind(ClientContext &context, ScalarFunction &bound_function,
                                   vector<unique_ptr<Expression>> &arguments) {
	return make_unique<StatsBindData>();
}

static unique_ptr<BaseStatistics> StatsPropagateStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &bind_data = input.bind_data;
	if (child_stats[0]) {
		auto &info = (StatsBindData &)*bind_data;
		info.stats = child_stats[0]->ToString();
	}
	return nullptr;
}

void StatsFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunction stats("stats", {LogicalType::ANY}, LogicalType::VARCHAR, StatsFunction, StatsBind, nullptr,
	                     StatsPropagateStats);
	stats.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	stats.side_effects = FunctionSideEffects::HAS_SIDE_EFFECTS;
	set.AddFunction(stats);
}

} // namespace duckdb


namespace duckdb {

static void TypeOfFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	Value v(args.data[0].GetType().ToString());
	result.Reference(v);
}

void TypeOfFun::RegisterFunction(BuiltinFunctions &set) {
	auto fun = ScalarFunction("typeof", {LogicalType::ANY}, LogicalType::VARCHAR, TypeOfFunction);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	set.AddFunction(fun);
}

} // namespace duckdb


namespace duckdb {

void BuiltinFunctions::RegisterGenericFunctions() {
	Register<AliasFun>();
	Register<HashFun>();
	Register<LeastFun>();
	Register<GreatestFun>();
	Register<StatsFun>();
	Register<TypeOfFun>();
	Register<ConstantOrNull>();
	Register<CurrentSettingFun>();
	Register<SystemFun>();
}

} // namespace duckdb









namespace duckdb {

template <typename INPUT_TYPE, typename INDEX_TYPE>
INDEX_TYPE ValueOffset(const INPUT_TYPE &value) {
	return 0;
}

template <>
int64_t ValueOffset(const list_entry_t &value) {
	return value.offset;
}

template <typename INPUT_TYPE, typename INDEX_TYPE>
INDEX_TYPE ValueLength(const INPUT_TYPE &value) {
	return 0;
}

template <>
int64_t ValueLength(const list_entry_t &value) {
	return value.length;
}

template <>
int32_t ValueLength(const string_t &value) {
	return LengthFun::Length<string_t, int32_t>(value);
}

template <typename INPUT_TYPE, typename INDEX_TYPE>
bool ClampIndex(INDEX_TYPE &index, const INPUT_TYPE &value) {
	const auto length = ValueLength<INPUT_TYPE, INDEX_TYPE>(value);
	if (index < 0) {
		if (-index > length) {
			return false;
		}
		index = length + index;
	} else if (index > length) {
		index = length;
	}
	return true;
}

template <typename INPUT_TYPE, typename INDEX_TYPE>
static bool ClampSlice(const INPUT_TYPE &value, INDEX_TYPE &begin, INDEX_TYPE &end, bool begin_valid, bool end_valid) {
	// Clamp offsets
	begin = begin_valid ? begin : 0;
	end = end_valid ? end : ValueLength<INPUT_TYPE, INDEX_TYPE>(value);
	if (!ClampIndex(begin, value) || !ClampIndex(end, value)) {
		return false;
	}
	end = MaxValue<INDEX_TYPE>(begin, end);

	return true;
}

template <typename INPUT_TYPE, typename INDEX_TYPE>
INPUT_TYPE SliceValue(Vector &result, INPUT_TYPE input, INDEX_TYPE begin, INDEX_TYPE end) {
	return input;
}

template <>
list_entry_t SliceValue(Vector &result, list_entry_t input, int64_t begin, int64_t end) {
	input.offset += begin;
	input.length = end - begin;
	return input;
}

template <>
string_t SliceValue(Vector &result, string_t input, int32_t begin, int32_t end) {
	// one-based - zero has strange semantics
	return SubstringFun::SubstringUnicode(result, input, begin + 1, end - begin);
}

template <typename INPUT_TYPE, typename INDEX_TYPE>
static void ExecuteSlice(Vector &result, Vector &s, Vector &b, Vector &e, const idx_t count) {
	if (result.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		auto rdata = ConstantVector::GetData<INPUT_TYPE>(result);
		auto sdata = ConstantVector::GetData<INPUT_TYPE>(s);
		auto bdata = ConstantVector::GetData<INDEX_TYPE>(b);
		auto edata = ConstantVector::GetData<INDEX_TYPE>(e);

		auto sliced = sdata[0];
		auto begin = (bdata[0] > 0) ? bdata[0] - 1 : bdata[0];
		auto end = edata[0];

		auto svalid = !ConstantVector::IsNull(s);
		auto bvalid = !ConstantVector::IsNull(b);
		auto evalid = !ConstantVector::IsNull(e);

		// Try to slice
		if (!svalid || !ClampSlice(sliced, begin, end, bvalid, evalid)) {
			ConstantVector::SetNull(result, true);
		} else {
			rdata[0] = SliceValue<INPUT_TYPE, INDEX_TYPE>(result, sliced, begin, end);
		}
	} else {
		UnifiedVectorFormat sdata, bdata, edata;

		s.ToUnifiedFormat(count, sdata);
		b.ToUnifiedFormat(count, bdata);
		e.ToUnifiedFormat(count, edata);

		auto rdata = FlatVector::GetData<INPUT_TYPE>(result);
		auto &rmask = FlatVector::Validity(result);

		for (idx_t i = 0; i < count; ++i) {
			auto sidx = sdata.sel->get_index(i);
			auto bidx = bdata.sel->get_index(i);
			auto eidx = edata.sel->get_index(i);

			auto sliced = ((INPUT_TYPE *)sdata.data)[sidx];
			auto begin = ((INDEX_TYPE *)bdata.data)[bidx];
			auto end = ((INDEX_TYPE *)edata.data)[eidx];

			begin = (begin > 0) ? begin - 1 : begin;

			auto svalid = sdata.validity.RowIsValid(sidx);
			auto bvalid = bdata.validity.RowIsValid(bidx);
			auto evalid = edata.validity.RowIsValid(eidx);

			// Try to slice
			if (!svalid || !ClampSlice(sliced, begin, end, bvalid, evalid)) {
				rmask.SetInvalid(i);
			} else {
				rdata[i] = SliceValue<INPUT_TYPE, INDEX_TYPE>(result, sliced, begin, end);
			}
		}
	}

	result.Verify(count);
}

static void ArraySliceFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 3);
	D_ASSERT(args.data.size() == 3);
	auto count = args.size();

	Vector &s = args.data[0];
	Vector &b = args.data[1];
	Vector &e = args.data[2];

	result.SetVectorType(args.AllConstant() ? VectorType::CONSTANT_VECTOR : VectorType::FLAT_VECTOR);
	switch (result.GetType().id()) {
	case LogicalTypeId::LIST:
		// Share the value dictionary as we are just going to slice it
		if (s.GetVectorType() != VectorType::FLAT_VECTOR && s.GetVectorType() != VectorType::CONSTANT_VECTOR) {
			s.Flatten(count);
		}
		ListVector::ReferenceEntry(result, s);
		ExecuteSlice<list_entry_t, int64_t>(result, s, b, e, count);
		break;
	case LogicalTypeId::VARCHAR:
		ExecuteSlice<string_t, int32_t>(result, s, b, e, count);
		break;
	default:
		throw NotImplementedException("Specifier type not implemented");
	}
}

static unique_ptr<FunctionData> ArraySliceBind(ClientContext &context, ScalarFunction &bound_function,
                                               vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(bound_function.arguments.size() == 3);
	switch (arguments[0]->return_type.id()) {
	case LogicalTypeId::LIST:
		// The result is the same type
		bound_function.return_type = arguments[0]->return_type;
		break;
	case LogicalTypeId::VARCHAR:
		// string slice returns a string, but can only accept 32 bit integers
		bound_function.return_type = arguments[0]->return_type;
		bound_function.arguments[1] = LogicalType::INTEGER;
		bound_function.arguments[2] = LogicalType::INTEGER;
		break;
	case LogicalTypeId::SQLNULL:
	case LogicalTypeId::UNKNOWN:
		bound_function.arguments[0] = LogicalTypeId::UNKNOWN;
		bound_function.return_type = LogicalType::SQLNULL;
		break;
	default:
		throw BinderException("ARRAY_SLICE can only operate on LISTs and VARCHARs");
	}

	return make_unique<VariableReturnBindData>(bound_function.return_type);
}

void ArraySliceFun::RegisterFunction(BuiltinFunctions &set) {
	// the arguments and return types are actually set in the binder function
	ScalarFunction fun({LogicalType::ANY, LogicalType::BIGINT, LogicalType::BIGINT}, LogicalType::ANY,
	                   ArraySliceFunction, ArraySliceBind);
	fun.varargs = LogicalType::ANY;
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	set.AddFunction({"array_slice", "list_slice"}, fun);
}

} // namespace duckdb





namespace duckdb {

struct ContainsFunctor {
	static inline bool Initialize() {
		return false;
	}
	static inline bool UpdateResultEntries(idx_t child_idx) {
		return true;
	}
};

struct PositionFunctor {
	static inline int32_t Initialize() {
		return 0;
	}
	static inline int32_t UpdateResultEntries(idx_t child_idx) {
		return child_idx + 1;
	}
};

template <class CHILD_TYPE, class RETURN_TYPE, class OP>
static void TemplatedContainsOrPosition(DataChunk &args, ExpressionState &state, Vector &result,
                                        bool is_nested = false) {
	D_ASSERT(args.ColumnCount() == 2);
	auto count = args.size();
	Vector &list = args.data[0];
	Vector &value_vector = args.data[1];

	// Create a result vector of type RETURN_TYPE
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_entries = FlatVector::GetData<RETURN_TYPE>(result);
	auto &result_validity = FlatVector::Validity(result);

	if (list.GetType().id() == LogicalTypeId::SQLNULL) {
		result_validity.SetInvalid(0);
		return;
	}

	auto list_size = ListVector::GetListSize(list);
	auto &child_vector = ListVector::GetEntry(list);

	UnifiedVectorFormat child_data;
	child_vector.ToUnifiedFormat(list_size, child_data);

	UnifiedVectorFormat list_data;
	list.ToUnifiedFormat(count, list_data);
	auto list_entries = (list_entry_t *)list_data.data;

	UnifiedVectorFormat value_data;
	value_vector.ToUnifiedFormat(count, value_data);

	// not required for a comparison of nested types
	auto child_value = (CHILD_TYPE *)child_data.data;
	auto values = (CHILD_TYPE *)value_data.data;

	for (idx_t i = 0; i < count; i++) {
		auto list_index = list_data.sel->get_index(i);
		auto value_index = value_data.sel->get_index(i);

		if (!list_data.validity.RowIsValid(list_index) || !value_data.validity.RowIsValid(value_index)) {
			result_validity.SetInvalid(i);
			continue;
		}

		const auto &list_entry = list_entries[list_index];

		result_entries[i] = OP::Initialize();
		for (idx_t child_idx = 0; child_idx < list_entry.length; child_idx++) {

			auto child_value_idx = child_data.sel->get_index(list_entry.offset + child_idx);
			if (!child_data.validity.RowIsValid(child_value_idx)) {
				continue;
			}

			if (!is_nested) {
				if (Equals::Operation(child_value[child_value_idx], values[value_index])) {
					result_entries[i] = OP::UpdateResultEntries(child_idx);
					break; // Found value in list, no need to look further
				}
			} else {
				// FIXME: using Value is less efficient than modifying the vector comparison code
				// to more efficiently compare nested types
				auto lvalue = child_vector.GetValue(child_value_idx);
				auto rvalue = value_vector.GetValue(value_index);
				if (Value::NotDistinctFrom(lvalue, rvalue)) {
					result_entries[i] = OP::UpdateResultEntries(child_idx);
					break; // Found value in list, no need to look further
				}
			}
		}
	}

	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

template <class T, class OP>
static void ListContainsOrPosition(DataChunk &args, ExpressionState &state, Vector &result) {
	switch (args.data[1].GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		TemplatedContainsOrPosition<int8_t, T, OP>(args, state, result);
		break;
	case PhysicalType::INT16:
		TemplatedContainsOrPosition<int16_t, T, OP>(args, state, result);
		break;
	case PhysicalType::INT32:
		TemplatedContainsOrPosition<int32_t, T, OP>(args, state, result);
		break;
	case PhysicalType::INT64:
		TemplatedContainsOrPosition<int64_t, T, OP>(args, state, result);
		break;
	case PhysicalType::INT128:
		TemplatedContainsOrPosition<hugeint_t, T, OP>(args, state, result);
		break;
	case PhysicalType::UINT8:
		TemplatedContainsOrPosition<uint8_t, T, OP>(args, state, result);
		break;
	case PhysicalType::UINT16:
		TemplatedContainsOrPosition<uint16_t, T, OP>(args, state, result);
		break;
	case PhysicalType::UINT32:
		TemplatedContainsOrPosition<uint32_t, T, OP>(args, state, result);
		break;
	case PhysicalType::UINT64:
		TemplatedContainsOrPosition<uint64_t, T, OP>(args, state, result);
		break;
	case PhysicalType::FLOAT:
		TemplatedContainsOrPosition<float, T, OP>(args, state, result);
		break;
	case PhysicalType::DOUBLE:
		TemplatedContainsOrPosition<double, T, OP>(args, state, result);
		break;
	case PhysicalType::VARCHAR:
		TemplatedContainsOrPosition<string_t, T, OP>(args, state, result);
		break;
	case PhysicalType::STRUCT:
	case PhysicalType::LIST:
		TemplatedContainsOrPosition<int8_t, T, OP>(args, state, result, true);
		break;
	default:
		throw NotImplementedException("This function has not been implemented for this type");
	}
}

static void ListContainsFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	return ListContainsOrPosition<bool, ContainsFunctor>(args, state, result);
}

static void ListPositionFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	return ListContainsOrPosition<int32_t, PositionFunctor>(args, state, result);
}

template <LogicalTypeId RETURN_TYPE>
static unique_ptr<FunctionData> ListContainsOrPositionBind(ClientContext &context, ScalarFunction &bound_function,
                                                           vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(bound_function.arguments.size() == 2);

	const auto &list = arguments[0]->return_type; // change to list
	const auto &value = arguments[1]->return_type;
	if (list.id() == LogicalTypeId::UNKNOWN) {
		bound_function.return_type = RETURN_TYPE;
		if (value.id() != LogicalTypeId::UNKNOWN) {
			// only list is a parameter, cast it to a list of value type
			bound_function.arguments[0] = LogicalType::LIST(value);
			bound_function.arguments[1] = value;
		}
	} else if (value.id() == LogicalTypeId::UNKNOWN) {
		// only value is a parameter: we expect the child type of list
		auto const &child_type = ListType::GetChildType(list);
		bound_function.arguments[0] = list;
		bound_function.arguments[1] = child_type;
		bound_function.return_type = RETURN_TYPE;
	} else {
		auto const &child_type = ListType::GetChildType(list);
		auto max_child_type = LogicalType::MaxLogicalType(child_type, value);
		auto list_type = LogicalType::LIST(max_child_type);

		bound_function.arguments[0] = list_type;
		bound_function.arguments[1] = value == max_child_type ? value : max_child_type;

		// list_contains and list_position only differ in their return type
		bound_function.return_type = RETURN_TYPE;
	}
	return make_unique<VariableReturnBindData>(bound_function.return_type);
}

static unique_ptr<FunctionData> ListContainsBind(ClientContext &context, ScalarFunction &bound_function,
                                                 vector<unique_ptr<Expression>> &arguments) {
	return ListContainsOrPositionBind<LogicalType::BOOLEAN>(context, bound_function, arguments);
}

static unique_ptr<FunctionData> ListPositionBind(ClientContext &context, ScalarFunction &bound_function,
                                                 vector<unique_ptr<Expression>> &arguments) {
	return ListContainsOrPositionBind<LogicalType::INTEGER>(context, bound_function, arguments);
}

ScalarFunction ListContainsFun::GetFunction() {
	return ScalarFunction({LogicalType::LIST(LogicalType::ANY), LogicalType::ANY}, // argument list
	                      LogicalType::BOOLEAN,                                    // return type
	                      ListContainsFunction, ListContainsBind, nullptr);
}

ScalarFunction ListPositionFun::GetFunction() {
	return ScalarFunction({LogicalType::LIST(LogicalType::ANY), LogicalType::ANY}, // argument list
	                      LogicalType::INTEGER,                                    // return type
	                      ListPositionFunction, ListPositionBind, nullptr);
}

void ListContainsFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction({"list_contains", "array_contains", "list_has", "array_has"}, GetFunction());
}

void ListPositionFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction({"list_position", "list_indexof", "array_position", "array_indexof"}, GetFunction());
}
} // namespace duckdb





namespace duckdb {

void ListFlattenFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 1);

	Vector &input = args.data[0];
	if (input.GetType().id() == LogicalTypeId::SQLNULL) {
		result.Reference(input);
		return;
	}

	idx_t count = args.size();

	UnifiedVectorFormat list_data;
	input.ToUnifiedFormat(count, list_data);
	auto list_entries = (list_entry_t *)list_data.data;

	auto &child_vector = ListVector::GetEntry(input);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_entries = FlatVector::GetData<list_entry_t>(result);
	auto &result_validity = FlatVector::Validity(result);

	if (child_vector.GetType().id() == LogicalTypeId::SQLNULL) {
		auto result_entries = FlatVector::GetData<list_entry_t>(result);
		for (idx_t i = 0; i < count; i++) {
			auto list_index = list_data.sel->get_index(i);
			if (!list_data.validity.RowIsValid(list_index)) {
				result_validity.SetInvalid(i);
				continue;
			}
			result_entries[i].offset = 0;
			result_entries[i].length = 0;
		}
		if (args.AllConstant()) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
		}
		return;
	}

	auto child_size = ListVector::GetListSize(input);
	UnifiedVectorFormat child_data;
	child_vector.ToUnifiedFormat(child_size, child_data);
	auto child_entries = (list_entry_t *)child_data.data;
	auto &data_vector = ListVector::GetEntry(child_vector);

	idx_t offset = 0;
	for (idx_t i = 0; i < count; i++) {
		auto list_index = list_data.sel->get_index(i);
		if (!list_data.validity.RowIsValid(list_index)) {
			result_validity.SetInvalid(i);
			continue;
		}
		auto list_entry = list_entries[list_index];

		idx_t source_offset = 0;
		// Find first valid child list entry to get offset
		for (idx_t j = 0; j < list_entry.length; j++) {
			auto child_list_index = child_data.sel->get_index(list_entry.offset + j);
			if (child_data.validity.RowIsValid(child_list_index)) {
				source_offset = child_entries[child_list_index].offset;
				break;
			}
		}

		idx_t length = 0;
		// Find last valid child list entry to get length
		for (idx_t j = list_entry.length - 1; j != (idx_t)-1; j--) {
			auto child_list_index = child_data.sel->get_index(list_entry.offset + j);
			if (child_data.validity.RowIsValid(child_list_index)) {
				auto child_entry = child_entries[child_list_index];
				length = child_entry.offset + child_entry.length - source_offset;
				break;
			}
		}
		ListVector::Append(result, data_vector, source_offset + length, source_offset);

		result_entries[i].offset = offset;
		result_entries[i].length = length;
		offset += length;
	}

	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static unique_ptr<FunctionData> ListFlattenBind(ClientContext &context, ScalarFunction &bound_function,
                                                vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(bound_function.arguments.size() == 1);

	auto &input_type = arguments[0]->return_type;
	bound_function.arguments[0] = input_type;
	if (input_type.id() == LogicalTypeId::UNKNOWN) {
		bound_function.arguments[0] = LogicalType(LogicalTypeId::UNKNOWN);
		bound_function.return_type = LogicalType(LogicalTypeId::SQLNULL);
		return nullptr;
	}
	D_ASSERT(input_type.id() == LogicalTypeId::LIST);

	auto child_type = ListType::GetChildType(input_type);
	if (child_type.id() == LogicalType::SQLNULL) {
		bound_function.return_type = input_type;
		return make_unique<VariableReturnBindData>(bound_function.return_type);
	}
	if (child_type.id() == LogicalTypeId::UNKNOWN) {
		bound_function.arguments[0] = LogicalType(LogicalTypeId::UNKNOWN);
		bound_function.return_type = LogicalType(LogicalTypeId::SQLNULL);
		return nullptr;
	}
	D_ASSERT(child_type.id() == LogicalTypeId::LIST);

	bound_function.return_type = child_type;
	return make_unique<VariableReturnBindData>(bound_function.return_type);
}

static unique_ptr<BaseStatistics> ListFlattenStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	if (!child_stats[0]) {
		return nullptr;
	}
	auto &list_stats = (ListStatistics &)*child_stats[0];
	if (!list_stats.child_stats || list_stats.child_stats->type == LogicalTypeId::SQLNULL) {
		return nullptr;
	}

	auto child_copy = list_stats.child_stats->Copy();
	child_copy->validity_stats = make_unique<ValidityStatistics>(true);
	return child_copy;
}

void ListFlattenFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunction fun({LogicalType::LIST(LogicalType::LIST(LogicalType::ANY))}, LogicalType::LIST(LogicalType::ANY),
	                   ListFlattenFunction, ListFlattenBind, nullptr, ListFlattenStats);
	set.AddFunction({"flatten"}, fun);
}

} // namespace duckdb











namespace duckdb {

// FIXME: use a local state for each thread to increase performance?
// FIXME: benchmark the use of simple_update against using update (if applicable)

struct ListAggregatesBindData : public FunctionData {
	ListAggregatesBindData(const LogicalType &stype_p, unique_ptr<Expression> aggr_expr_p);
	~ListAggregatesBindData() override;

	LogicalType stype;
	unique_ptr<Expression> aggr_expr;

	unique_ptr<FunctionData> Copy() const override {
		return make_unique<ListAggregatesBindData>(stype, aggr_expr->Copy());
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const ListAggregatesBindData &)other_p;
		return stype == other.stype && aggr_expr->Equals(other.aggr_expr.get());
	}
	static void Serialize(FieldWriter &writer, const FunctionData *bind_data_p, const ScalarFunction &function) {
		throw NotImplementedException("FIXME: list aggr serialize");
	}
	static unique_ptr<FunctionData> Deserialize(ClientContext &context, FieldReader &reader,
	                                            ScalarFunction &bound_function) {
		throw NotImplementedException("FIXME: list aggr deserialize");
	}
};

ListAggregatesBindData::ListAggregatesBindData(const LogicalType &stype_p, unique_ptr<Expression> aggr_expr_p)
    : stype(stype_p), aggr_expr(std::move(aggr_expr_p)) {
}

ListAggregatesBindData::~ListAggregatesBindData() {
}

struct StateVector {
	StateVector(idx_t count_p, unique_ptr<Expression> aggr_expr_p)
	    : count(count_p), aggr_expr(std::move(aggr_expr_p)), state_vector(Vector(LogicalType::POINTER, count_p)) {
	}

	~StateVector() {
		// destroy objects within the aggregate states
		auto &aggr = (BoundAggregateExpression &)*aggr_expr;
		if (aggr.function.destructor) {
			aggr.function.destructor(state_vector, count);
		}
	}

	idx_t count;
	unique_ptr<Expression> aggr_expr;
	Vector state_vector;
};

struct FinalizeValueFunctor {
	template <class T>
	static Value FinalizeValue(T first) {
		return Value::CreateValue(first);
	}
};

struct FinalizeStringValueFunctor {
	template <class T>
	static Value FinalizeValue(T first) {
		string_t value = first;
		return Value::CreateValue(value);
	}
};

struct AggregateFunctor {
	template <class OP, class T, class MAP_TYPE = unordered_map<T, idx_t>>
	static void ListExecuteFunction(Vector &result, Vector &state_vector, idx_t count) {
	}
};

struct DistinctFunctor {
	template <class OP, class T, class MAP_TYPE = unordered_map<T, idx_t>>
	static void ListExecuteFunction(Vector &result, Vector &state_vector, idx_t count) {

		UnifiedVectorFormat sdata;
		state_vector.ToUnifiedFormat(count, sdata);
		auto states = (HistogramAggState<T, MAP_TYPE> **)sdata.data;

		auto result_data = FlatVector::GetData<list_entry_t>(result);

		idx_t offset = 0;
		for (idx_t i = 0; i < count; i++) {

			auto state = states[sdata.sel->get_index(i)];
			result_data[i].offset = offset;

			if (!state->hist) {
				result_data[i].length = 0;
				continue;
			}

			result_data[i].length = state->hist->size();
			offset += state->hist->size();

			for (auto &entry : *state->hist) {
				Value bucket_value = OP::template FinalizeValue<T>(entry.first);
				ListVector::PushBack(result, bucket_value);
			}
		}
		result.Verify(count);
	}
};

struct UniqueFunctor {
	template <class OP, class T, class MAP_TYPE = unordered_map<T, idx_t>>
	static void ListExecuteFunction(Vector &result, Vector &state_vector, idx_t count) {

		UnifiedVectorFormat sdata;
		state_vector.ToUnifiedFormat(count, sdata);
		auto states = (HistogramAggState<T, MAP_TYPE> **)sdata.data;

		auto result_data = FlatVector::GetData<uint64_t>(result);

		for (idx_t i = 0; i < count; i++) {

			auto state = states[sdata.sel->get_index(i)];

			if (!state->hist) {
				result_data[i] = 0;
				continue;
			}

			result_data[i] = state->hist->size();
		}
		result.Verify(count);
	}
};

template <class FUNCTION_FUNCTOR, bool IS_AGGR = false>
static void ListAggregatesFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();
	Vector &lists = args.data[0];

	// set the result vector
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &result_validity = FlatVector::Validity(result);

	if (lists.GetType().id() == LogicalTypeId::SQLNULL) {
		result_validity.SetInvalid(0);
		return;
	}

	// get the aggregate function
	auto &func_expr = (BoundFunctionExpression &)state.expr;
	auto &info = (ListAggregatesBindData &)*func_expr.bind_info;
	auto &aggr = (BoundAggregateExpression &)*info.aggr_expr;
	AggregateInputData aggr_input_data(aggr.bind_info.get(), Allocator::DefaultAllocator());

	D_ASSERT(aggr.function.update);

	auto lists_size = ListVector::GetListSize(lists);
	auto &child_vector = ListVector::GetEntry(lists);
	child_vector.Flatten(lists_size);

	UnifiedVectorFormat child_data;
	child_vector.ToUnifiedFormat(lists_size, child_data);

	UnifiedVectorFormat lists_data;
	lists.ToUnifiedFormat(count, lists_data);
	auto list_entries = (list_entry_t *)lists_data.data;

	// state_buffer holds the state for each list of this chunk
	idx_t size = aggr.function.state_size();
	auto state_buffer = unique_ptr<data_t[]>(new data_t[size * count]);

	// state vector for initialize and finalize
	StateVector state_vector(count, info.aggr_expr->Copy());
	auto states = FlatVector::GetData<data_ptr_t>(state_vector.state_vector);

	// state vector of STANDARD_VECTOR_SIZE holds the pointers to the states
	Vector state_vector_update = Vector(LogicalType::POINTER);
	auto states_update = FlatVector::GetData<data_ptr_t>(state_vector_update);

	// selection vector pointing to the data
	SelectionVector sel_vector(STANDARD_VECTOR_SIZE);
	idx_t states_idx = 0;

	for (idx_t i = 0; i < count; i++) {

		// initialize the state for this list
		auto state_ptr = state_buffer.get() + size * i;
		states[i] = state_ptr;
		aggr.function.initialize(states[i]);

		auto lists_index = lists_data.sel->get_index(i);
		const auto &list_entry = list_entries[lists_index];

		// nothing to do for this list
		if (!lists_data.validity.RowIsValid(lists_index)) {
			result_validity.SetInvalid(i);
			continue;
		}

		// skip empty list
		if (list_entry.length == 0) {
			continue;
		}

		for (idx_t child_idx = 0; child_idx < list_entry.length; child_idx++) {
			// states vector is full, update
			if (states_idx == STANDARD_VECTOR_SIZE) {
				// update the aggregate state(s)
				Vector slice(child_vector, sel_vector, states_idx);
				aggr.function.update(&slice, aggr_input_data, 1, state_vector_update, states_idx);

				// reset values
				states_idx = 0;
			}

			auto source_idx = child_data.sel->get_index(list_entry.offset + child_idx);
			sel_vector.set_index(states_idx, source_idx);
			states_update[states_idx] = state_ptr;
			states_idx++;
		}
	}

	// update the remaining elements of the last list(s)
	if (states_idx != 0) {
		Vector slice(child_vector, sel_vector, states_idx);
		aggr.function.update(&slice, aggr_input_data, 1, state_vector_update, states_idx);
	}

	if (IS_AGGR) {
		// finalize all the aggregate states
		aggr.function.finalize(state_vector.state_vector, aggr_input_data, result, count, 0);

	} else {
		// finalize manually to use the map
		D_ASSERT(aggr.function.arguments.size() == 1);
		auto key_type = aggr.function.arguments[0];

		switch (key_type.InternalType()) {
		case PhysicalType::BOOL:
			FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, bool>(
			    result, state_vector.state_vector, count);
			break;
		case PhysicalType::UINT8:
			FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, uint8_t>(
			    result, state_vector.state_vector, count);
			break;
		case PhysicalType::UINT16:
			FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, uint16_t>(
			    result, state_vector.state_vector, count);
			break;
		case PhysicalType::UINT32:
			FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, uint32_t>(
			    result, state_vector.state_vector, count);
			break;
		case PhysicalType::UINT64:
			FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, uint64_t>(
			    result, state_vector.state_vector, count);
			break;
		case PhysicalType::INT8:
			FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, int8_t>(
			    result, state_vector.state_vector, count);
			break;
		case PhysicalType::INT16:
			FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, int16_t>(
			    result, state_vector.state_vector, count);
			break;
		case PhysicalType::INT32:
			if (key_type.id() == LogicalTypeId::DATE) {
				FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, date_t>(
				    result, state_vector.state_vector, count);
			} else {
				FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, int32_t>(
				    result, state_vector.state_vector, count);
			}
			break;
		case PhysicalType::INT64:
			switch (key_type.id()) {
			case LogicalTypeId::TIME:
				FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, dtime_t>(
				    result, state_vector.state_vector, count);
				break;
			case LogicalTypeId::TIME_TZ:
				FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, dtime_tz_t>(
				    result, state_vector.state_vector, count);
				break;
			case LogicalTypeId::TIMESTAMP:
				FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, timestamp_t>(
				    result, state_vector.state_vector, count);
				break;
			case LogicalTypeId::TIMESTAMP_MS:
				FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, timestamp_ms_t>(
				    result, state_vector.state_vector, count);
				break;
			case LogicalTypeId::TIMESTAMP_NS:
				FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, timestamp_ns_t>(
				    result, state_vector.state_vector, count);
				break;
			case LogicalTypeId::TIMESTAMP_SEC:
				FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, timestamp_sec_t>(
				    result, state_vector.state_vector, count);
				break;
			case LogicalTypeId::TIMESTAMP_TZ:
				FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, timestamp_tz_t>(
				    result, state_vector.state_vector, count);
				break;
			default:
				FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, int64_t>(
				    result, state_vector.state_vector, count);
				break;
			}
			break;
		case PhysicalType::FLOAT:
			FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, float>(
			    result, state_vector.state_vector, count);
			break;
		case PhysicalType::DOUBLE:
			FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeValueFunctor, double>(
			    result, state_vector.state_vector, count);
			break;
		case PhysicalType::VARCHAR:
			FUNCTION_FUNCTOR::template ListExecuteFunction<FinalizeStringValueFunctor, string>(
			    result, state_vector.state_vector, count);
			break;
		default:
			throw InternalException("Unimplemented histogram aggregate");
		}
	}

	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static void ListAggregateFunction(DataChunk &args, ExpressionState &state, Vector &result) {

	D_ASSERT(args.ColumnCount() == 2);
	ListAggregatesFunction<AggregateFunctor, true>(args, state, result);
}

static void ListDistinctFunction(DataChunk &args, ExpressionState &state, Vector &result) {

	D_ASSERT(args.ColumnCount() == 1);
	ListAggregatesFunction<DistinctFunctor>(args, state, result);
}

static void ListUniqueFunction(DataChunk &args, ExpressionState &state, Vector &result) {

	D_ASSERT(args.ColumnCount() == 1);
	ListAggregatesFunction<UniqueFunctor>(args, state, result);
}

template <bool IS_AGGR = false>
static unique_ptr<FunctionData>
ListAggregatesBindFunction(ClientContext &context, ScalarFunction &bound_function, const LogicalType &list_child_type,
                           AggregateFunction &aggr_function, vector<unique_ptr<Expression>> &arguments) {

	// create the child expression and its type
	vector<unique_ptr<Expression>> children;
	auto expr = make_unique<BoundConstantExpression>(Value(list_child_type));
	children.push_back(std::move(expr));
	// push any extra arguments into the list aggregate bind
	if (arguments.size() > 2) {
		for (idx_t i = 2; i < arguments.size(); i++) {
			children.push_back(std::move(arguments[i]));
		}
		arguments.resize(2);
	}

	FunctionBinder function_binder(context);
	auto bound_aggr_function = function_binder.BindAggregateFunction(aggr_function, std::move(children));
	bound_function.arguments[0] = LogicalType::LIST(bound_aggr_function->function.arguments[0]);

	if (IS_AGGR) {
		bound_function.return_type = bound_aggr_function->function.return_type;
	}
	// check if the aggregate function consumed all the extra input arguments
	if (bound_aggr_function->children.size() > 1) {
		throw InvalidInputException(
		    "Aggregate function %s is not supported for list_aggr: extra arguments were not removed during bind",
		    bound_aggr_function->ToString());
	}

	return make_unique<ListAggregatesBindData>(bound_function.return_type, std::move(bound_aggr_function));
}

template <bool IS_AGGR = false>
static unique_ptr<FunctionData> ListAggregatesBind(ClientContext &context, ScalarFunction &bound_function,
                                                   vector<unique_ptr<Expression>> &arguments) {
	if (arguments[0]->return_type.id() == LogicalTypeId::SQLNULL) {
		bound_function.arguments[0] = LogicalType::SQLNULL;
		bound_function.return_type = LogicalType::SQLNULL;
		return make_unique<VariableReturnBindData>(bound_function.return_type);
	}

	bool is_parameter = arguments[0]->return_type.id() == LogicalTypeId::UNKNOWN;
	auto list_child_type = is_parameter ? LogicalTypeId::UNKNOWN : ListType::GetChildType(arguments[0]->return_type);

	string function_name = "histogram";
	if (IS_AGGR) { // get the name of the aggregate function
		if (!arguments[1]->IsFoldable()) {
			throw InvalidInputException("Aggregate function name must be a constant");
		}
		// get the function name
		Value function_value = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
		function_name = function_value.ToString();
	}

	// look up the aggregate function in the catalog
	QueryErrorContext error_context(nullptr, 0);
	auto func =
	    (AggregateFunctionCatalogEntry *)Catalog::GetSystemCatalog(context).GetEntry<AggregateFunctionCatalogEntry>(
	        context, DEFAULT_SCHEMA, function_name, false, error_context);
	D_ASSERT(func->type == CatalogType::AGGREGATE_FUNCTION_ENTRY);

	if (is_parameter) {
		bound_function.arguments[0] = LogicalTypeId::UNKNOWN;
		bound_function.return_type = LogicalType::SQLNULL;
		return nullptr;
	}

	// find a matching aggregate function
	string error;
	vector<LogicalType> types;
	types.push_back(list_child_type);
	// push any extra arguments into the type list
	for (idx_t i = 2; i < arguments.size(); i++) {
		types.push_back(arguments[i]->return_type);
	}

	FunctionBinder function_binder(context);
	auto best_function_idx = function_binder.BindFunction(func->name, func->functions, types, error);
	if (best_function_idx == DConstants::INVALID_INDEX) {
		throw BinderException("No matching aggregate function\n%s", error);
	}

	// found a matching function, bind it as an aggregate
	auto best_function = func->functions.GetFunctionByOffset(best_function_idx);
	if (IS_AGGR) {
		return ListAggregatesBindFunction<IS_AGGR>(context, bound_function, list_child_type, best_function, arguments);
	}

	// create the unordered map histogram function
	D_ASSERT(best_function.arguments.size() == 1);
	auto key_type = best_function.arguments[0];
	auto aggr_function = HistogramFun::GetHistogramUnorderedMap(key_type);
	return ListAggregatesBindFunction<IS_AGGR>(context, bound_function, list_child_type, aggr_function, arguments);
}

static unique_ptr<FunctionData> ListAggregateBind(ClientContext &context, ScalarFunction &bound_function,
                                                  vector<unique_ptr<Expression>> &arguments) {

	// the list column and the name of the aggregate function
	D_ASSERT(bound_function.arguments.size() >= 2);
	D_ASSERT(arguments.size() >= 2);

	return ListAggregatesBind<true>(context, bound_function, arguments);
}

static unique_ptr<FunctionData> ListDistinctBind(ClientContext &context, ScalarFunction &bound_function,
                                                 vector<unique_ptr<Expression>> &arguments) {

	D_ASSERT(bound_function.arguments.size() == 1);
	D_ASSERT(arguments.size() == 1);
	bound_function.return_type = arguments[0]->return_type;

	return ListAggregatesBind<>(context, bound_function, arguments);
}

static unique_ptr<FunctionData> ListUniqueBind(ClientContext &context, ScalarFunction &bound_function,
                                               vector<unique_ptr<Expression>> &arguments) {

	D_ASSERT(bound_function.arguments.size() == 1);
	D_ASSERT(arguments.size() == 1);
	bound_function.return_type = LogicalType::UBIGINT;

	return ListAggregatesBind<>(context, bound_function, arguments);
}

ScalarFunction ListAggregateFun::GetFunction() {
	auto result = ScalarFunction({LogicalType::LIST(LogicalType::ANY), LogicalType::VARCHAR}, LogicalType::ANY,
	                             ListAggregateFunction, ListAggregateBind);
	result.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	result.varargs = LogicalType::ANY;
	result.serialize = ListAggregatesBindData::Serialize;
	result.deserialize = ListAggregatesBindData::Deserialize;
	return result;
}

ScalarFunction ListDistinctFun::GetFunction() {
	return ScalarFunction({LogicalType::LIST(LogicalType::ANY)}, LogicalType::LIST(LogicalType::ANY),
	                      ListDistinctFunction, ListDistinctBind);
}

ScalarFunction ListUniqueFun::GetFunction() {
	return ScalarFunction({LogicalType::LIST(LogicalType::ANY)}, LogicalType::UBIGINT, ListUniqueFunction,
	                      ListUniqueBind);
}

void ListAggregateFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction({"list_aggregate", "array_aggregate", "list_aggr", "array_aggr"}, GetFunction());
}

void ListDistinctFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction({"list_distinct", "array_distinct"}, GetFunction());
}

void ListUniqueFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction({"list_unique", "array_unique"}, GetFunction());
}

} // namespace duckdb








namespace duckdb {

static void ListConcatFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 2);
	auto count = args.size();

	Vector &lhs = args.data[0];
	Vector &rhs = args.data[1];
	if (lhs.GetType().id() == LogicalTypeId::SQLNULL) {
		result.Reference(rhs);
		return;
	}
	if (rhs.GetType().id() == LogicalTypeId::SQLNULL) {
		result.Reference(lhs);
		return;
	}

	UnifiedVectorFormat lhs_data;
	UnifiedVectorFormat rhs_data;
	lhs.ToUnifiedFormat(count, lhs_data);
	rhs.ToUnifiedFormat(count, rhs_data);
	auto lhs_entries = (list_entry_t *)lhs_data.data;
	auto rhs_entries = (list_entry_t *)rhs_data.data;

	auto lhs_list_size = ListVector::GetListSize(lhs);
	auto rhs_list_size = ListVector::GetListSize(rhs);
	auto &lhs_child = ListVector::GetEntry(lhs);
	auto &rhs_child = ListVector::GetEntry(rhs);
	UnifiedVectorFormat lhs_child_data;
	UnifiedVectorFormat rhs_child_data;
	lhs_child.ToUnifiedFormat(lhs_list_size, lhs_child_data);
	rhs_child.ToUnifiedFormat(rhs_list_size, rhs_child_data);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_entries = FlatVector::GetData<list_entry_t>(result);
	auto &result_validity = FlatVector::Validity(result);

	idx_t offset = 0;
	for (idx_t i = 0; i < count; i++) {
		auto lhs_list_index = lhs_data.sel->get_index(i);
		auto rhs_list_index = rhs_data.sel->get_index(i);
		if (!lhs_data.validity.RowIsValid(lhs_list_index) && !rhs_data.validity.RowIsValid(rhs_list_index)) {
			result_validity.SetInvalid(i);
			continue;
		}
		result_entries[i].offset = offset;
		result_entries[i].length = 0;
		if (lhs_data.validity.RowIsValid(lhs_list_index)) {
			const auto &lhs_entry = lhs_entries[lhs_list_index];
			result_entries[i].length += lhs_entry.length;
			ListVector::Append(result, lhs_child, *lhs_child_data.sel, lhs_entry.offset + lhs_entry.length,
			                   lhs_entry.offset);
		}
		if (rhs_data.validity.RowIsValid(rhs_list_index)) {
			const auto &rhs_entry = rhs_entries[rhs_list_index];
			result_entries[i].length += rhs_entry.length;
			ListVector::Append(result, rhs_child, *rhs_child_data.sel, rhs_entry.offset + rhs_entry.length,
			                   rhs_entry.offset);
		}
		offset += result_entries[i].length;
	}
	D_ASSERT(ListVector::GetListSize(result) == offset);

	if (lhs.GetVectorType() == VectorType::CONSTANT_VECTOR && rhs.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static unique_ptr<FunctionData> ListConcatBind(ClientContext &context, ScalarFunction &bound_function,
                                               vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(bound_function.arguments.size() == 2);

	auto &lhs = arguments[0]->return_type;
	auto &rhs = arguments[1]->return_type;
	if (lhs.id() == LogicalTypeId::UNKNOWN || rhs.id() == LogicalTypeId::UNKNOWN) {
		throw ParameterNotResolvedException();
	} else if (lhs.id() == LogicalTypeId::SQLNULL || rhs.id() == LogicalTypeId::SQLNULL) {
		// we mimic postgres behaviour: list_concat(NULL, my_list) = my_list
		bound_function.arguments[0] = lhs;
		bound_function.arguments[1] = rhs;
		bound_function.return_type = rhs.id() == LogicalTypeId::SQLNULL ? lhs : rhs;
	} else {
		D_ASSERT(lhs.id() == LogicalTypeId::LIST);
		D_ASSERT(rhs.id() == LogicalTypeId::LIST);

		// Resolve list type
		LogicalType child_type = LogicalType::SQLNULL;
		for (const auto &argument : arguments) {
			child_type = LogicalType::MaxLogicalType(child_type, ListType::GetChildType(argument->return_type));
		}
		auto list_type = LogicalType::LIST(std::move(child_type));

		bound_function.arguments[0] = list_type;
		bound_function.arguments[1] = list_type;
		bound_function.return_type = list_type;
	}
	return make_unique<VariableReturnBindData>(bound_function.return_type);
}

static unique_ptr<BaseStatistics> ListConcatStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	D_ASSERT(child_stats.size() == 2);
	if (!child_stats[0] || !child_stats[1]) {
		return nullptr;
	}

	auto &left_stats = (ListStatistics &)*child_stats[0];
	auto &right_stats = (ListStatistics &)*child_stats[1];

	auto stats = left_stats.Copy();
	stats->Merge(right_stats);

	return stats;
}

ScalarFunction ListConcatFun::GetFunction() {
	// the arguments and return types are actually set in the binder function
	auto fun = ScalarFunction({LogicalType::LIST(LogicalType::ANY), LogicalType::LIST(LogicalType::ANY)},
	                          LogicalType::LIST(LogicalType::ANY), ListConcatFunction, ListConcatBind, nullptr,
	                          ListConcatStats);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	return fun;
}

void ListConcatFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction({"list_concat", "list_cat", "array_concat", "array_cat"}, GetFunction());
}

} // namespace duckdb












namespace duckdb {

template <class T, bool HEAP_REF = false, bool VALIDITY_ONLY = false>
void ListExtractTemplate(idx_t count, UnifiedVectorFormat &list_data, UnifiedVectorFormat &offsets_data,
                         Vector &child_vector, idx_t list_size, Vector &result) {
	UnifiedVectorFormat child_format;
	child_vector.ToUnifiedFormat(list_size, child_format);

	T *result_data;

	result.SetVectorType(VectorType::FLAT_VECTOR);
	if (!VALIDITY_ONLY) {
		result_data = FlatVector::GetData<T>(result);
	}
	auto &result_mask = FlatVector::Validity(result);

	// heap-ref once
	if (HEAP_REF) {
		StringVector::AddHeapReference(result, child_vector);
	}

	// this is lifted from ExecuteGenericLoop because we can't push the list child data into this otherwise
	// should have gone with GetValue perhaps
	auto child_data = (T *)child_format.data;
	for (idx_t i = 0; i < count; i++) {
		auto list_index = list_data.sel->get_index(i);
		auto offsets_index = offsets_data.sel->get_index(i);
		if (!list_data.validity.RowIsValid(list_index)) {
			result_mask.SetInvalid(i);
			continue;
		}
		if (!offsets_data.validity.RowIsValid(offsets_index)) {
			result_mask.SetInvalid(i);
			continue;
		}
		auto list_entry = ((list_entry_t *)list_data.data)[list_index];
		auto offsets_entry = ((int64_t *)offsets_data.data)[offsets_index];

		// 1-based indexing
		if (offsets_entry == 0) {
			result_mask.SetInvalid(i);
			continue;
		}
		offsets_entry = (offsets_entry > 0) ? offsets_entry - 1 : offsets_entry;

		idx_t child_offset;
		if (offsets_entry < 0) {
			if ((idx_t)-offsets_entry > list_entry.length) {
				result_mask.SetInvalid(i);
				continue;
			}
			child_offset = list_entry.offset + list_entry.length + offsets_entry;
		} else {
			if ((idx_t)offsets_entry >= list_entry.length) {
				result_mask.SetInvalid(i);
				continue;
			}
			child_offset = list_entry.offset + offsets_entry;
		}
		auto child_index = child_format.sel->get_index(child_offset);
		if (child_format.validity.RowIsValid(child_index)) {
			if (!VALIDITY_ONLY) {
				result_data[i] = child_data[child_index];
			}
		} else {
			result_mask.SetInvalid(i);
		}
	}
	if (count == 1) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}
static void ExecuteListExtractInternal(const idx_t count, UnifiedVectorFormat &list, UnifiedVectorFormat &offsets,
                                       Vector &child_vector, idx_t list_size, Vector &result) {
	D_ASSERT(child_vector.GetType() == result.GetType());
	switch (result.GetType().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		ListExtractTemplate<int8_t>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::INT16:
		ListExtractTemplate<int16_t>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::INT32:
		ListExtractTemplate<int32_t>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::INT64:
		ListExtractTemplate<int64_t>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::INT128:
		ListExtractTemplate<hugeint_t>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::UINT8:
		ListExtractTemplate<uint8_t>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::UINT16:
		ListExtractTemplate<uint16_t>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::UINT32:
		ListExtractTemplate<uint32_t>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::UINT64:
		ListExtractTemplate<uint64_t>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::FLOAT:
		ListExtractTemplate<float>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::DOUBLE:
		ListExtractTemplate<double>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::VARCHAR:
		ListExtractTemplate<string_t, true>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::INTERVAL:
		ListExtractTemplate<interval_t>(count, list, offsets, child_vector, list_size, result);
		break;
	case PhysicalType::STRUCT: {
		auto &entries = StructVector::GetEntries(child_vector);
		auto &result_entries = StructVector::GetEntries(result);
		D_ASSERT(entries.size() == result_entries.size());
		// extract the child entries of the struct
		for (idx_t i = 0; i < entries.size(); i++) {
			ExecuteListExtractInternal(count, list, offsets, *entries[i], list_size, *result_entries[i]);
		}
		// extract the validity mask
		ListExtractTemplate<bool, false, true>(count, list, offsets, child_vector, list_size, result);
		break;
	}
	case PhysicalType::LIST: {
		// nested list: we have to reference the child
		auto &child_child_list = ListVector::GetEntry(child_vector);

		ListVector::GetEntry(result).Reference(child_child_list);
		ListVector::SetListSize(result, ListVector::GetListSize(child_vector));
		ListExtractTemplate<list_entry_t>(count, list, offsets, child_vector, list_size, result);
		break;
	}
	default:
		throw NotImplementedException("Unimplemented type for LIST_EXTRACT");
	}
}

static void ExecuteListExtract(Vector &result, Vector &list, Vector &offsets, const idx_t count) {
	D_ASSERT(list.GetType().id() == LogicalTypeId::LIST);
	UnifiedVectorFormat list_data;
	UnifiedVectorFormat offsets_data;

	list.ToUnifiedFormat(count, list_data);
	offsets.ToUnifiedFormat(count, offsets_data);
	ExecuteListExtractInternal(count, list_data, offsets_data, ListVector::GetEntry(list),
	                           ListVector::GetListSize(list), result);
	result.Verify(count);
}

static void ExecuteStringExtract(Vector &result, Vector &input_vector, Vector &subscript_vector, const idx_t count) {
	BinaryExecutor::Execute<string_t, int64_t, string_t>(
	    input_vector, subscript_vector, result, count, [&](string_t input_string, int64_t subscript) {
		    return SubstringFun::SubstringUnicode(result, input_string, subscript, 1);
	    });
}

static void ListExtractFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 2);
	auto count = args.size();

	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	for (idx_t i = 0; i < args.ColumnCount(); i++) {
		if (args.data[i].GetVectorType() != VectorType::CONSTANT_VECTOR) {
			result.SetVectorType(VectorType::FLAT_VECTOR);
		}
	}

	Vector &base = args.data[0];
	Vector &subscript = args.data[1];

	switch (base.GetType().id()) {
	case LogicalTypeId::LIST:
		ExecuteListExtract(result, base, subscript, count);
		break;
	case LogicalTypeId::VARCHAR:
		ExecuteStringExtract(result, base, subscript, count);
		break;
	case LogicalTypeId::SQLNULL:
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		ConstantVector::SetNull(result, true);
		break;
	default:
		throw NotImplementedException("Specifier type not implemented");
	}
}

static unique_ptr<FunctionData> ListExtractBind(ClientContext &context, ScalarFunction &bound_function,
                                                vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(bound_function.arguments.size() == 2);
	D_ASSERT(LogicalTypeId::LIST == arguments[0]->return_type.id());
	// list extract returns the child type of the list as return type
	bound_function.return_type = ListType::GetChildType(arguments[0]->return_type);
	return make_unique<VariableReturnBindData>(bound_function.return_type);
}

static unique_ptr<BaseStatistics> ListExtractStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	if (!child_stats[0]) {
		return nullptr;
	}
	auto &list_stats = (ListStatistics &)*child_stats[0];
	if (!list_stats.child_stats) {
		return nullptr;
	}
	auto child_copy = list_stats.child_stats->Copy();
	// list_extract always pushes a NULL, since if the offset is out of range for a list it inserts a null
	child_copy->validity_stats = make_unique<ValidityStatistics>(true);
	return child_copy;
}

void ListExtractFun::RegisterFunction(BuiltinFunctions &set) {
	// the arguments and return types are actually set in the binder function
	ScalarFunction lfun({LogicalType::LIST(LogicalType::ANY), LogicalType::BIGINT}, LogicalType::ANY,
	                    ListExtractFunction, ListExtractBind, nullptr, ListExtractStats);

	ScalarFunction sfun({LogicalType::VARCHAR, LogicalType::BIGINT}, LogicalType::VARCHAR, ListExtractFunction);

	ScalarFunctionSet list_extract("list_extract");
	list_extract.AddFunction(lfun);
	list_extract.AddFunction(sfun);
	set.AddFunction(list_extract);

	ScalarFunctionSet list_element("list_element");
	list_element.AddFunction(lfun);
	list_element.AddFunction(sfun);
	set.AddFunction(list_element);

	ScalarFunctionSet array_extract("array_extract");
	array_extract.AddFunction(lfun);
	array_extract.AddFunction(sfun);
	array_extract.AddFunction(StructExtractFun::GetFunction());
	set.AddFunction(array_extract);
}

} // namespace duckdb










namespace duckdb {

struct ListLambdaBindData : public FunctionData {
	ListLambdaBindData(const LogicalType &stype_p, unique_ptr<Expression> lambda_expr);
	~ListLambdaBindData() override;

	LogicalType stype;
	unique_ptr<Expression> lambda_expr;

public:
	bool Equals(const FunctionData &other_p) const override;
	unique_ptr<FunctionData> Copy() const override;
	static void Serialize(FieldWriter &writer, const FunctionData *bind_data_p, const ScalarFunction &function) {
		throw NotImplementedException("FIXME: list lambda serialize");
	}
	static unique_ptr<FunctionData> Deserialize(ClientContext &context, FieldReader &reader,
	                                            ScalarFunction &bound_function) {
		throw NotImplementedException("FIXME: list lambda deserialize");
	}
};

ListLambdaBindData::ListLambdaBindData(const LogicalType &stype_p, unique_ptr<Expression> lambda_expr_p)
    : stype(stype_p), lambda_expr(std::move(lambda_expr_p)) {
}

unique_ptr<FunctionData> ListLambdaBindData::Copy() const {
	return make_unique<ListLambdaBindData>(stype, lambda_expr->Copy());
}

bool ListLambdaBindData::Equals(const FunctionData &other_p) const {
	auto &other = (ListLambdaBindData &)other_p;
	return lambda_expr->Equals(other.lambda_expr.get()) && stype == other.stype;
}

ListLambdaBindData::~ListLambdaBindData() {
}

static void AppendTransformedToResult(Vector &lambda_vector, idx_t &elem_cnt, Vector &result) {

	// append the lambda_vector to the result list
	UnifiedVectorFormat lambda_child_data;
	lambda_vector.ToUnifiedFormat(elem_cnt, lambda_child_data);
	ListVector::Append(result, lambda_vector, *lambda_child_data.sel, elem_cnt, 0);
}

static void AppendFilteredToResult(Vector &lambda_vector, list_entry_t *result_entries, idx_t &elem_cnt, Vector &result,
                                   idx_t &curr_list_len, idx_t &curr_list_offset, idx_t &appended_lists_cnt,
                                   vector<idx_t> &lists_len, idx_t &curr_original_list_len, DataChunk &input_chunk) {

	idx_t true_count = 0;
	SelectionVector true_sel(elem_cnt);
	auto lambda_values = FlatVector::GetData<bool>(lambda_vector);
	auto &lambda_validity = FlatVector::Validity(lambda_vector);

	// compute the new lengths and offsets, and create a selection vector
	for (idx_t i = 0; i < elem_cnt; i++) {

		while (appended_lists_cnt < lists_len.size() && lists_len[appended_lists_cnt] == 0) {
			result_entries[appended_lists_cnt].offset = curr_list_offset;
			result_entries[appended_lists_cnt].length = 0;
			appended_lists_cnt++;
		}

		// found a true value
		if (lambda_validity.RowIsValid(i)) {
			if (lambda_values[i] > 0) {
				true_sel.set_index(true_count++, i);
				curr_list_len++;
			}
		}
		curr_original_list_len++;

		if (lists_len[appended_lists_cnt] == curr_original_list_len) {
			result_entries[appended_lists_cnt].offset = curr_list_offset;
			result_entries[appended_lists_cnt].length = curr_list_len;
			curr_list_offset += curr_list_len;
			appended_lists_cnt++;
			curr_list_len = 0;
			curr_original_list_len = 0;
		}
	}

	while (appended_lists_cnt < lists_len.size() && lists_len[appended_lists_cnt] == 0) {
		result_entries[appended_lists_cnt].offset = curr_list_offset;
		result_entries[appended_lists_cnt].length = 0;
		appended_lists_cnt++;
	}

	// slice to get the new lists and append them to the result
	Vector new_lists(input_chunk.data[0], true_sel, true_count);
	new_lists.Flatten(true_count);
	UnifiedVectorFormat new_lists_child_data;
	new_lists.ToUnifiedFormat(true_count, new_lists_child_data);
	ListVector::Append(result, new_lists, *new_lists_child_data.sel, true_count, 0);
}

static void ExecuteExpression(vector<LogicalType> &types, vector<LogicalType> &result_types, idx_t &elem_cnt,
                              SelectionVector &sel, vector<SelectionVector> &sel_vectors, DataChunk &input_chunk,
                              DataChunk &lambda_chunk, Vector &child_vector, DataChunk &args,
                              ExpressionExecutor &expr_executor) {

	input_chunk.SetCardinality(elem_cnt);
	lambda_chunk.SetCardinality(elem_cnt);

	// set the list child vector
	Vector slice(child_vector, sel, elem_cnt);
	Vector second_slice(child_vector, sel, elem_cnt);
	slice.Flatten(elem_cnt);
	second_slice.Flatten(elem_cnt);

	input_chunk.data[0].Reference(slice);
	input_chunk.data[1].Reference(second_slice);

	// set the other vectors
	vector<Vector> slices;
	for (idx_t col_idx = 0; col_idx < args.ColumnCount() - 1; col_idx++) {
		slices.emplace_back(Vector(args.data[col_idx + 1], sel_vectors[col_idx], elem_cnt));
		slices[col_idx].Flatten(elem_cnt);
		input_chunk.data[col_idx + 2].Reference(slices[col_idx]);
	}

	// execute the lambda expression
	expr_executor.Execute(input_chunk, lambda_chunk);
}

template <bool IS_TRANSFORM = true>
static void ListLambdaFunction(DataChunk &args, ExpressionState &state, Vector &result) {

	// always at least the list argument
	D_ASSERT(args.ColumnCount() >= 1);

	auto count = args.size();
	Vector &lists = args.data[0];

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_entries = FlatVector::GetData<list_entry_t>(result);
	auto &result_validity = FlatVector::Validity(result);

	if (lists.GetType().id() == LogicalTypeId::SQLNULL) {
		result_validity.SetInvalid(0);
		return;
	}

	// e.g. window functions in sub queries return dictionary vectors, which segfault on expression execution
	// if not flattened first
	for (idx_t i = 1; i < args.ColumnCount(); i++) {
		if (args.data[i].GetVectorType() != VectorType::FLAT_VECTOR &&
		    args.data[i].GetVectorType() != VectorType::CONSTANT_VECTOR) {
			args.data[i].Flatten(count);
		}
	}

	// get the lists data
	UnifiedVectorFormat lists_data;
	lists.ToUnifiedFormat(count, lists_data);
	auto list_entries = (list_entry_t *)lists_data.data;

	// get the lambda expression
	auto &func_expr = (BoundFunctionExpression &)state.expr;
	auto &info = (ListLambdaBindData &)*func_expr.bind_info;
	auto &lambda_expr = info.lambda_expr;

	// get the child vector and child data
	auto lists_size = ListVector::GetListSize(lists);
	auto &child_vector = ListVector::GetEntry(lists);
	child_vector.Flatten(lists_size);
	UnifiedVectorFormat child_data;
	child_vector.ToUnifiedFormat(lists_size, child_data);

	// to slice the child vector
	SelectionVector sel(STANDARD_VECTOR_SIZE);

	// this vector never contains more than one element
	vector<LogicalType> result_types;
	result_types.push_back(lambda_expr->return_type);

	// non-lambda parameter columns
	vector<UnifiedVectorFormat> columns;
	vector<idx_t> indexes;
	vector<SelectionVector> sel_vectors;

	vector<LogicalType> types;
	types.push_back(child_vector.GetType());
	types.push_back(child_vector.GetType());

	// skip the list column
	for (idx_t i = 1; i < args.ColumnCount(); i++) {
		columns.emplace_back(UnifiedVectorFormat());
		args.data[i].ToUnifiedFormat(count, columns[i - 1]);
		indexes.push_back(0);
		sel_vectors.emplace_back(SelectionVector(STANDARD_VECTOR_SIZE));
		types.push_back(args.data[i].GetType());
	}

	// get the expression executor
	ExpressionExecutor expr_executor(state.GetContext(), *lambda_expr);

	// these are only for the list_filter
	vector<idx_t> lists_len;
	idx_t curr_list_len = 0;
	idx_t curr_list_offset = 0;
	idx_t appended_lists_cnt = 0;
	idx_t curr_original_list_len = 0;

	if (!IS_TRANSFORM) {
		lists_len.reserve(count);
	}

	DataChunk input_chunk;
	DataChunk lambda_chunk;
	input_chunk.InitializeEmpty(types);
	lambda_chunk.Initialize(Allocator::DefaultAllocator(), result_types);

	// loop over the child entries and create chunks to be executed by the expression executor
	idx_t elem_cnt = 0;
	idx_t offset = 0;
	for (idx_t row_idx = 0; row_idx < count; row_idx++) {

		auto lists_index = lists_data.sel->get_index(row_idx);
		const auto &list_entry = list_entries[lists_index];

		// set the result to NULL for this row
		if (!lists_data.validity.RowIsValid(lists_index)) {
			result_validity.SetInvalid(row_idx);
			if (!IS_TRANSFORM) {
				lists_len.push_back(0);
			}
			continue;
		}

		// set the length and offset of the resulting lists of list_transform
		if (IS_TRANSFORM) {
			result_entries[row_idx].offset = offset;
			result_entries[row_idx].length = list_entry.length;
			offset += list_entry.length;
		} else {
			lists_len.push_back(list_entry.length);
		}

		// empty list, nothing to execute
		if (list_entry.length == 0) {
			continue;
		}

		// get the data indexes
		for (idx_t col_idx = 0; col_idx < args.ColumnCount() - 1; col_idx++) {
			indexes[col_idx] = columns[col_idx].sel->get_index(row_idx);
		}

		// iterate list elements and create transformed expression columns
		for (idx_t child_idx = 0; child_idx < list_entry.length; child_idx++) {
			// reached STANDARD_VECTOR_SIZE elements
			if (elem_cnt == STANDARD_VECTOR_SIZE) {
				lambda_chunk.Reset();
				ExecuteExpression(types, result_types, elem_cnt, sel, sel_vectors, input_chunk, lambda_chunk,
				                  child_vector, args, expr_executor);

				auto &lambda_vector = lambda_chunk.data[0];

				if (IS_TRANSFORM) {
					AppendTransformedToResult(lambda_vector, elem_cnt, result);
				} else {
					AppendFilteredToResult(lambda_vector, result_entries, elem_cnt, result, curr_list_len,
					                       curr_list_offset, appended_lists_cnt, lists_len, curr_original_list_len,
					                       input_chunk);
				}
				elem_cnt = 0;
			}

			// to slice the child vector
			auto source_idx = child_data.sel->get_index(list_entry.offset + child_idx);
			sel.set_index(elem_cnt, source_idx);

			// for each column, set the index of the selection vector to slice properly
			for (idx_t col_idx = 0; col_idx < args.ColumnCount() - 1; col_idx++) {
				sel_vectors[col_idx].set_index(elem_cnt, indexes[col_idx]);
			}
			elem_cnt++;
		}
	}

	lambda_chunk.Reset();
	ExecuteExpression(types, result_types, elem_cnt, sel, sel_vectors, input_chunk, lambda_chunk, child_vector, args,
	                  expr_executor);
	auto &lambda_vector = lambda_chunk.data[0];

	if (IS_TRANSFORM) {
		AppendTransformedToResult(lambda_vector, elem_cnt, result);
	} else {
		AppendFilteredToResult(lambda_vector, result_entries, elem_cnt, result, curr_list_len, curr_list_offset,
		                       appended_lists_cnt, lists_len, curr_original_list_len, input_chunk);
	}

	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static void ListTransformFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	ListLambdaFunction<>(args, state, result);
}

static void ListFilterFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	ListLambdaFunction<false>(args, state, result);
}

template <int64_t LAMBDA_PARAM_CNT>
static unique_ptr<FunctionData> ListLambdaBind(ClientContext &context, ScalarFunction &bound_function,
                                               vector<unique_ptr<Expression>> &arguments) {

	auto &bound_lambda_expr = (BoundLambdaExpression &)*arguments[1];
	if (bound_lambda_expr.parameter_count != LAMBDA_PARAM_CNT) {
		throw BinderException("Incorrect number of parameters in lambda function! " + bound_function.name +
		                      " expects " + to_string(LAMBDA_PARAM_CNT) + " parameter(s).");
	}

	if (arguments[0]->return_type.id() == LogicalTypeId::SQLNULL) {
		bound_function.arguments.pop_back();
		bound_function.arguments[0] = LogicalType::SQLNULL;
		bound_function.return_type = LogicalType::SQLNULL;
		return make_unique<VariableReturnBindData>(bound_function.return_type);
	}

	if (arguments[0]->return_type.id() == LogicalTypeId::UNKNOWN) {
		throw ParameterNotResolvedException();
	}

	D_ASSERT(arguments[0]->return_type.id() == LogicalTypeId::LIST);

	// get the lambda expression and put it in the bind info
	auto lambda_expr = std::move(bound_lambda_expr.lambda_expr);
	return make_unique<ListLambdaBindData>(bound_function.return_type, std::move(lambda_expr));
}

static unique_ptr<FunctionData> ListTransformBind(ClientContext &context, ScalarFunction &bound_function,
                                                  vector<unique_ptr<Expression>> &arguments) {

	// at least the list column and the lambda function
	D_ASSERT(arguments.size() == 2);
	if (arguments[1]->expression_class != ExpressionClass::BOUND_LAMBDA) {
		throw BinderException("Invalid lambda expression!");
	}

	auto &bound_lambda_expr = (BoundLambdaExpression &)*arguments[1];
	bound_function.return_type = LogicalType::LIST(bound_lambda_expr.lambda_expr->return_type);
	return ListLambdaBind<1>(context, bound_function, arguments);
}

static unique_ptr<FunctionData> ListFilterBind(ClientContext &context, ScalarFunction &bound_function,
                                               vector<unique_ptr<Expression>> &arguments) {

	// at least the list column and the lambda function
	D_ASSERT(arguments.size() == 2);
	if (arguments[1]->expression_class != ExpressionClass::BOUND_LAMBDA) {
		throw BinderException("Invalid lambda expression!");
	}

	// try to cast to boolean, if the return type of the lambda filter expression is not already boolean
	auto &bound_lambda_expr = (BoundLambdaExpression &)*arguments[1];
	if (bound_lambda_expr.lambda_expr->return_type != LogicalType::BOOLEAN) {
		auto cast_lambda_expr =
		    BoundCastExpression::AddCastToType(context, std::move(bound_lambda_expr.lambda_expr), LogicalType::BOOLEAN);
		bound_lambda_expr.lambda_expr = std::move(cast_lambda_expr);
	}

	bound_function.return_type = arguments[0]->return_type;
	return ListLambdaBind<1>(context, bound_function, arguments);
}

void ListTransformFun::RegisterFunction(BuiltinFunctions &set) {

	ScalarFunction fun("list_transform", {LogicalType::LIST(LogicalType::ANY), LogicalType::LAMBDA},
	                   LogicalType::LIST(LogicalType::ANY), ListTransformFunction, ListTransformBind, nullptr, nullptr);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	fun.serialize = ListLambdaBindData::Serialize;
	fun.deserialize = ListLambdaBindData::Deserialize;
	set.AddFunction(fun);

	fun.name = "array_transform";
	set.AddFunction(fun);
	fun.name = "list_apply";
	set.AddFunction(fun);
	fun.name = "array_apply";
	set.AddFunction(fun);
}

void ListFilterFun::RegisterFunction(BuiltinFunctions &set) {

	ScalarFunction fun("list_filter", {LogicalType::LIST(LogicalType::ANY), LogicalType::LAMBDA},
	                   LogicalType::LIST(LogicalType::ANY), ListFilterFunction, ListFilterBind, nullptr, nullptr);
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	fun.serialize = ListLambdaBindData::Serialize;
	fun.deserialize = ListLambdaBindData::Deserialize;
	set.AddFunction(fun);

	fun.name = "array_filter";
	set.AddFunction(fun);
}

} // namespace duckdb









namespace duckdb {

struct ListSortBindData : public FunctionData {
	ListSortBindData(OrderType order_type_p, OrderByNullType null_order_p, const LogicalType &return_type_p,
	                 const LogicalType &child_type_p, ClientContext &context_p);
	~ListSortBindData() override;

	OrderType order_type;
	OrderByNullType null_order;
	LogicalType return_type;
	LogicalType child_type;

	vector<LogicalType> types;
	vector<LogicalType> payload_types;

	ClientContext &context;
	RowLayout payload_layout;
	vector<BoundOrderByNode> orders;

public:
	bool Equals(const FunctionData &other_p) const override;
	unique_ptr<FunctionData> Copy() const override;
};

ListSortBindData::ListSortBindData(OrderType order_type_p, OrderByNullType null_order_p,
                                   const LogicalType &return_type_p, const LogicalType &child_type_p,
                                   ClientContext &context_p)
    : order_type(order_type_p), null_order(null_order_p), return_type(return_type_p), child_type(child_type_p),
      context(context_p) {

	// get the vector types
	types.emplace_back(LogicalType::USMALLINT);
	types.emplace_back(child_type);
	D_ASSERT(types.size() == 2);

	// get the payload types
	payload_types.emplace_back(LogicalType::UINTEGER);
	D_ASSERT(payload_types.size() == 1);

	// initialize the payload layout
	payload_layout.Initialize(payload_types);

	// get the BoundOrderByNode
	auto idx_col_expr = make_unique_base<Expression, BoundReferenceExpression>(LogicalType::USMALLINT, 0);
	auto lists_col_expr = make_unique_base<Expression, BoundReferenceExpression>(child_type, 1);
	orders.emplace_back(OrderType::ASCENDING, OrderByNullType::ORDER_DEFAULT, std::move(idx_col_expr));
	orders.emplace_back(order_type, null_order, std::move(lists_col_expr));
}

unique_ptr<FunctionData> ListSortBindData::Copy() const {
	return make_unique<ListSortBindData>(order_type, null_order, return_type, child_type, context);
}

bool ListSortBindData::Equals(const FunctionData &other_p) const {
	auto &other = (ListSortBindData &)other_p;
	return order_type == other.order_type && null_order == other.null_order;
}

ListSortBindData::~ListSortBindData() {
}

// create the key_chunk and the payload_chunk and sink them into the local_sort_state
void SinkDataChunk(Vector *child_vector, SelectionVector &sel, idx_t offset_lists_indices, vector<LogicalType> &types,
                   vector<LogicalType> &payload_types, Vector &payload_vector, LocalSortState &local_sort_state,
                   bool &data_to_sort, Vector &lists_indices) {

	// slice the child vector
	Vector slice(*child_vector, sel, offset_lists_indices);

	// initialize and fill key_chunk
	DataChunk key_chunk;
	key_chunk.InitializeEmpty(types);
	key_chunk.data[0].Reference(lists_indices);
	key_chunk.data[1].Reference(slice);
	key_chunk.SetCardinality(offset_lists_indices);

	// initialize and fill key_chunk and payload_chunk
	DataChunk payload_chunk;
	payload_chunk.InitializeEmpty(payload_types);
	payload_chunk.data[0].Reference(payload_vector);
	payload_chunk.SetCardinality(offset_lists_indices);

	key_chunk.Verify();
	payload_chunk.Verify();

	// sink
	key_chunk.Flatten();
	local_sort_state.SinkChunk(key_chunk, payload_chunk);
	data_to_sort = true;
}

static void ListSortFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() >= 1 && args.ColumnCount() <= 3);
	auto count = args.size();
	Vector &input_lists = args.data[0];

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto &result_validity = FlatVector::Validity(result);

	if (input_lists.GetType().id() == LogicalTypeId::SQLNULL) {
		result_validity.SetInvalid(0);
		return;
	}

	auto &func_expr = (BoundFunctionExpression &)state.expr;
	auto &info = (ListSortBindData &)*func_expr.bind_info;

	// initialize the global and local sorting state
	auto &buffer_manager = BufferManager::GetBufferManager(info.context);
	GlobalSortState global_sort_state(buffer_manager, info.orders, info.payload_layout);
	LocalSortState local_sort_state;
	local_sort_state.Initialize(global_sort_state, buffer_manager);

	// this ensures that we do not change the order of the entries in the input chunk
	VectorOperations::Copy(input_lists, result, count, 0, 0);

	// get the child vector
	auto lists_size = ListVector::GetListSize(result);
	auto &child_vector = ListVector::GetEntry(result);
	UnifiedVectorFormat child_data;
	child_vector.ToUnifiedFormat(lists_size, child_data);

	// get the lists data
	UnifiedVectorFormat lists_data;
	result.ToUnifiedFormat(count, lists_data);
	auto list_entries = (list_entry_t *)lists_data.data;

	// create the lists_indices vector, this contains an element for each list's entry,
	// the element corresponds to the list's index, e.g. for [1, 2, 4], [5, 4]
	// lists_indices contains [0, 0, 0, 1, 1]
	Vector lists_indices(LogicalType::USMALLINT);
	auto lists_indices_data = FlatVector::GetData<uint16_t>(lists_indices);

	// create the payload_vector, this is just a vector containing incrementing integers
	// this will later be used as the 'new' selection vector of the child_vector, after
	// rearranging the payload according to the sorting order
	Vector payload_vector(LogicalType::UINTEGER);
	auto payload_vector_data = FlatVector::GetData<uint32_t>(payload_vector);

	// selection vector pointing to the data of the child vector,
	// used for slicing the child_vector correctly
	SelectionVector sel(STANDARD_VECTOR_SIZE);

	idx_t offset_lists_indices = 0;
	uint32_t incr_payload_count = 0;
	bool data_to_sort = false;

	for (idx_t i = 0; i < count; i++) {
		auto lists_index = lists_data.sel->get_index(i);
		const auto &list_entry = list_entries[lists_index];

		// nothing to do for this list
		if (!lists_data.validity.RowIsValid(lists_index)) {
			result_validity.SetInvalid(i);
			continue;
		}

		// empty list, no sorting required
		if (list_entry.length == 0) {
			continue;
		}

		for (idx_t child_idx = 0; child_idx < list_entry.length; child_idx++) {
			// lists_indices vector is full, sink
			if (offset_lists_indices == STANDARD_VECTOR_SIZE) {
				SinkDataChunk(&child_vector, sel, offset_lists_indices, info.types, info.payload_types, payload_vector,
				              local_sort_state, data_to_sort, lists_indices);
				offset_lists_indices = 0;
			}

			auto source_idx = list_entry.offset + child_idx;
			sel.set_index(offset_lists_indices, source_idx);
			lists_indices_data[offset_lists_indices] = (uint32_t)i;
			payload_vector_data[offset_lists_indices] = source_idx;
			offset_lists_indices++;
			incr_payload_count++;
		}
	}

	if (offset_lists_indices != 0) {
		SinkDataChunk(&child_vector, sel, offset_lists_indices, info.types, info.payload_types, payload_vector,
		              local_sort_state, data_to_sort, lists_indices);
	}

	if (data_to_sort) {
		// add local state to global state, which sorts the data
		global_sort_state.AddLocalState(local_sort_state);
		global_sort_state.PrepareMergePhase();

		// selection vector that is to be filled with the 'sorted' payload
		SelectionVector sel_sorted(incr_payload_count);
		idx_t sel_sorted_idx = 0;

		// scan the sorted row data
		PayloadScanner scanner(*global_sort_state.sorted_blocks[0]->payload_data, global_sort_state);
		for (;;) {
			DataChunk result_chunk;
			result_chunk.Initialize(Allocator::DefaultAllocator(), info.payload_types);
			result_chunk.SetCardinality(0);
			scanner.Scan(result_chunk);
			if (result_chunk.size() == 0) {
				break;
			}

			// construct the selection vector with the new order from the result vectors
			Vector result_vector(result_chunk.data[0]);
			auto result_data = FlatVector::GetData<uint32_t>(result_vector);
			auto row_count = result_chunk.size();

			for (idx_t i = 0; i < row_count; i++) {
				sel_sorted.set_index(sel_sorted_idx, result_data[i]);
				D_ASSERT(result_data[i] < lists_size);
				sel_sorted_idx++;
			}
		}

		D_ASSERT(sel_sorted_idx == incr_payload_count);
		child_vector.Slice(sel_sorted, sel_sorted_idx);
		child_vector.Flatten(sel_sorted_idx);
	}

	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static unique_ptr<FunctionData> ListSortBind(ClientContext &context, ScalarFunction &bound_function,
                                             vector<unique_ptr<Expression>> &arguments, OrderType &order,
                                             OrderByNullType &null_order) {
	bound_function.arguments[0] = arguments[0]->return_type;
	bound_function.return_type = arguments[0]->return_type;
	auto child_type = ListType::GetChildType(arguments[0]->return_type);

	return make_unique<ListSortBindData>(order, null_order, bound_function.return_type, child_type, context);
}

OrderByNullType GetNullOrder(ClientContext &context, vector<unique_ptr<Expression>> &arguments, idx_t idx) {

	if (!arguments[idx]->IsFoldable()) {
		throw InvalidInputException("Null sorting order must be a constant");
	}
	Value null_order_value = ExpressionExecutor::EvaluateScalar(context, *arguments[idx]);
	auto null_order_name = StringUtil::Upper(null_order_value.ToString());
	if (null_order_name != "NULLS FIRST" && null_order_name != "NULLS LAST") {
		throw InvalidInputException("Null sorting order must be either NULLS FIRST or NULLS LAST");
	}

	if (null_order_name == "NULLS LAST") {
		return OrderByNullType::NULLS_LAST;
	}
	return OrderByNullType::NULLS_FIRST;
}

static unique_ptr<FunctionData> ListNormalSortBind(ClientContext &context, ScalarFunction &bound_function,
                                                   vector<unique_ptr<Expression>> &arguments) {

	D_ASSERT(bound_function.arguments.size() >= 1 && bound_function.arguments.size() <= 3);
	D_ASSERT(arguments.size() >= 1 && arguments.size() <= 3);

	// set default values
	auto &config = DBConfig::GetConfig(context);
	auto order = config.options.default_order_type;
	auto null_order = config.options.default_null_order;

	// get the sorting order
	if (arguments.size() >= 2) {

		if (!arguments[1]->IsFoldable()) {
			throw InvalidInputException("Sorting order must be a constant");
		}
		Value order_value = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
		auto order_name = StringUtil::Upper(order_value.ToString());
		if (order_name != "DESC" && order_name != "ASC") {
			throw InvalidInputException("Sorting order must be either ASC or DESC");
		}
		if (order_name == "DESC") {
			order = OrderType::DESCENDING;
		} else {
			order = OrderType::ASCENDING;
		}
	}

	// get the null sorting order
	if (arguments.size() == 3) {
		null_order = GetNullOrder(context, arguments, 2);
	}

	return ListSortBind(context, bound_function, arguments, order, null_order);
}

static unique_ptr<FunctionData> ListReverseSortBind(ClientContext &context, ScalarFunction &bound_function,
                                                    vector<unique_ptr<Expression>> &arguments) {

	D_ASSERT(bound_function.arguments.size() == 1 || bound_function.arguments.size() == 2);
	D_ASSERT(arguments.size() == 1 || arguments.size() == 2);

	// set (reverse) default values
	auto &config = DBConfig::GetConfig(context);
	auto order =
	    (config.options.default_order_type == OrderType::ASCENDING) ? OrderType::DESCENDING : OrderType::ASCENDING;
	auto null_order = config.options.default_null_order;

	// get the null sorting order
	if (arguments.size() == 2) {
		null_order = GetNullOrder(context, arguments, 1);
	}

	return ListSortBind(context, bound_function, arguments, order, null_order);
}

void ListSortFun::RegisterFunction(BuiltinFunctions &set) {

	// normal sort

	// one parameter: list
	ScalarFunction sort({LogicalType::LIST(LogicalType::ANY)}, LogicalType::LIST(LogicalType::ANY), ListSortFunction,
	                    ListNormalSortBind);

	// two parameters: list, order
	ScalarFunction sort_order({LogicalType::LIST(LogicalType::ANY), LogicalType::VARCHAR},
	                          LogicalType::LIST(LogicalType::ANY), ListSortFunction, ListNormalSortBind);

	// three parameters: list, order, null order
	ScalarFunction sort_orders({LogicalType::LIST(LogicalType::ANY), LogicalType::VARCHAR, LogicalType::VARCHAR},
	                           LogicalType::LIST(LogicalType::ANY), ListSortFunction, ListNormalSortBind);

	ScalarFunctionSet list_sort("list_sort");
	list_sort.AddFunction(sort);
	list_sort.AddFunction(sort_order);
	list_sort.AddFunction(sort_orders);
	set.AddFunction(list_sort);

	ScalarFunctionSet array_sort("array_sort");
	array_sort.AddFunction(sort);
	array_sort.AddFunction(sort_order);
	array_sort.AddFunction(sort_orders);
	set.AddFunction(array_sort);

	// reverse sort

	// one parameter: list
	ScalarFunction sort_reverse({LogicalType::LIST(LogicalType::ANY)}, LogicalType::LIST(LogicalType::ANY),
	                            ListSortFunction, ListReverseSortBind);

	// two parameters: list, null order
	ScalarFunction sort_reverse_null_order({LogicalType::LIST(LogicalType::ANY), LogicalType::VARCHAR},
	                                       LogicalType::LIST(LogicalType::ANY), ListSortFunction, ListReverseSortBind);

	ScalarFunctionSet list_reverse_sort("list_reverse_sort");
	list_reverse_sort.AddFunction(sort_reverse);
	list_reverse_sort.AddFunction(sort_reverse_null_order);
	set.AddFunction(list_reverse_sort);

	ScalarFunctionSet array_reverse_sort("array_reverse_sort");
	array_reverse_sort.AddFunction(sort_reverse);
	array_reverse_sort.AddFunction(sort_reverse_null_order);
	set.AddFunction(array_reverse_sort);
}

} // namespace duckdb









namespace duckdb {

static void ListValueFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(result.GetType().id() == LogicalTypeId::LIST);
	auto &child_type = ListType::GetChildType(result.GetType());

	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	for (idx_t i = 0; i < args.ColumnCount(); i++) {
		if (args.data[i].GetVectorType() != VectorType::CONSTANT_VECTOR) {
			result.SetVectorType(VectorType::FLAT_VECTOR);
		}
	}

	auto result_data = FlatVector::GetData<list_entry_t>(result);
	for (idx_t i = 0; i < args.size(); i++) {
		result_data[i].offset = ListVector::GetListSize(result);
		for (idx_t col_idx = 0; col_idx < args.ColumnCount(); col_idx++) {
			auto val = args.GetValue(col_idx, i).DefaultCastAs(child_type);
			ListVector::PushBack(result, val);
		}
		result_data[i].length = args.ColumnCount();
	}
	result.Verify(args.size());
}

static unique_ptr<FunctionData> ListValueBind(ClientContext &context, ScalarFunction &bound_function,
                                              vector<unique_ptr<Expression>> &arguments) {
	// collect names and deconflict, construct return type
	LogicalType child_type = arguments.empty() ? LogicalType::SQLNULL : arguments[0]->return_type;
	for (idx_t i = 1; i < arguments.size(); i++) {
		child_type = LogicalType::MaxLogicalType(child_type, arguments[i]->return_type);
	}

	// this is more for completeness reasons
	bound_function.varargs = child_type;
	bound_function.return_type = LogicalType::LIST(std::move(child_type));
	return make_unique<VariableReturnBindData>(bound_function.return_type);
}

unique_ptr<BaseStatistics> ListValueStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &expr = input.expr;
	auto list_stats = make_unique<ListStatistics>(expr.return_type);
	for (idx_t i = 0; i < child_stats.size(); i++) {
		if (child_stats[i]) {
			list_stats->child_stats->Merge(*child_stats[i]);
		} else {
			list_stats->child_stats.reset();
			return std::move(list_stats);
		}
	}
	return std::move(list_stats);
}

void ListValueFun::RegisterFunction(BuiltinFunctions &set) {
	// the arguments and return types are actually set in the binder function
	ScalarFunction fun("list_value", {}, LogicalTypeId::LIST, ListValueFunction, ListValueBind, nullptr,
	                   ListValueStats);
	fun.varargs = LogicalType::ANY;
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	set.AddFunction(fun);
	fun.name = "list_pack";
	set.AddFunction(fun);
}

} // namespace duckdb






namespace duckdb {

struct NumericRangeInfo {
	using TYPE = int64_t;
	using INCREMENT_TYPE = int64_t;

	static int64_t DefaultStart() {
		return 0;
	}
	static int64_t DefaultIncrement() {
		return 1;
	}

	static uint64_t ListLength(int64_t start_value, int64_t end_value, int64_t increment_value, bool inclusive_bound) {
		if (increment_value == 0) {
			return 0;
		}
		if (start_value > end_value && increment_value > 0) {
			return 0;
		}
		if (start_value < end_value && increment_value < 0) {
			return 0;
		}
		hugeint_t total_diff = AbsValue(hugeint_t(end_value) - hugeint_t(start_value));
		hugeint_t increment = AbsValue(hugeint_t(increment_value));
		hugeint_t total_values = total_diff / increment;
		if (total_diff % increment == 0) {
			if (inclusive_bound) {
				total_values += 1;
			}
		} else {
			total_values += 1;
		}
		if (total_values > NumericLimits<uint32_t>::Maximum()) {
			throw InvalidInputException("Lists larger than 2^32 elements are not supported");
		}
		return Hugeint::Cast<uint64_t>(total_values);
	}

	static void Increment(int64_t &input, int64_t increment) {
		input += increment;
	}
};
struct TimestampRangeInfo {
	using TYPE = timestamp_t;
	using INCREMENT_TYPE = interval_t;

	static timestamp_t DefaultStart() {
		throw InternalException("Default start not implemented for timestamp range");
	}
	static interval_t DefaultIncrement() {
		throw InternalException("Default increment not implemented for timestamp range");
	}
	static uint64_t ListLength(timestamp_t start_value, timestamp_t end_value, interval_t increment_value,
	                           bool inclusive_bound) {
		bool is_positive = increment_value.months > 0 || increment_value.days > 0 || increment_value.micros > 0;
		bool is_negative = increment_value.months < 0 || increment_value.days < 0 || increment_value.micros < 0;
		if (!is_negative && !is_positive) {
			// interval is 0: no result
			return 0;
		}
		// We don't allow infinite bounds because they generate errors or infinite loops
		if (!Timestamp::IsFinite(start_value) || !Timestamp::IsFinite(end_value)) {
			throw InvalidInputException("Interval infinite bounds not supported");
		}

		if (is_negative && is_positive) {
			// we don't allow a mix of
			throw InvalidInputException("Interval with mix of negative/positive entries not supported");
		}
		if (start_value > end_value && is_positive) {
			return 0;
		}
		if (start_value < end_value && is_negative) {
			return 0;
		}
		int64_t total_values = 0;
		if (is_negative) {
			// negative interval, start_value is going down
			while (inclusive_bound ? start_value >= end_value : start_value > end_value) {
				start_value = Interval::Add(start_value, increment_value);
				total_values++;
				if (total_values > NumericLimits<uint32_t>::Maximum()) {
					throw InvalidInputException("Lists larger than 2^32 elements are not supported");
				}
			}
		} else {
			// positive interval, start_value is going up
			while (inclusive_bound ? start_value <= end_value : start_value < end_value) {
				start_value = Interval::Add(start_value, increment_value);
				total_values++;
				if (total_values > NumericLimits<uint32_t>::Maximum()) {
					throw InvalidInputException("Lists larger than 2^32 elements are not supported");
				}
			}
		}
		return total_values;
	}

	static void Increment(timestamp_t &input, interval_t increment) {
		input = Interval::Add(input, increment);
	}
};

template <class OP, bool INCLUSIVE_BOUND>
class RangeInfoStruct {
public:
	explicit RangeInfoStruct(DataChunk &args_p) : args(args_p) {
		switch (args.ColumnCount()) {
		case 1:
			args.data[0].ToUnifiedFormat(args.size(), vdata[0]);
			break;
		case 2:
			args.data[0].ToUnifiedFormat(args.size(), vdata[0]);
			args.data[1].ToUnifiedFormat(args.size(), vdata[1]);
			break;
		case 3:
			args.data[0].ToUnifiedFormat(args.size(), vdata[0]);
			args.data[1].ToUnifiedFormat(args.size(), vdata[1]);
			args.data[2].ToUnifiedFormat(args.size(), vdata[2]);
			break;
		default:
			throw InternalException("Unsupported number of parameters for range");
		}
	}

	bool RowIsValid(idx_t row_idx) {
		for (idx_t i = 0; i < args.ColumnCount(); i++) {
			auto idx = vdata[i].sel->get_index(row_idx);
			if (!vdata[i].validity.RowIsValid(idx)) {
				return false;
			}
		}
		return true;
	}

	typename OP::TYPE StartListValue(idx_t row_idx) {
		if (args.ColumnCount() == 1) {
			return OP::DefaultStart();
		} else {
			auto data = (typename OP::TYPE *)vdata[0].data;
			auto idx = vdata[0].sel->get_index(row_idx);
			return data[idx];
		}
	}

	typename OP::TYPE EndListValue(idx_t row_idx) {
		idx_t vdata_idx = args.ColumnCount() == 1 ? 0 : 1;
		auto data = (typename OP::TYPE *)vdata[vdata_idx].data;
		auto idx = vdata[vdata_idx].sel->get_index(row_idx);
		return data[idx];
	}

	typename OP::INCREMENT_TYPE ListIncrementValue(idx_t row_idx) {
		if (args.ColumnCount() < 3) {
			return OP::DefaultIncrement();
		} else {
			auto data = (typename OP::INCREMENT_TYPE *)vdata[2].data;
			auto idx = vdata[2].sel->get_index(row_idx);
			return data[idx];
		}
	}

	void GetListValues(idx_t row_idx, typename OP::TYPE &start_value, typename OP::TYPE &end_value,
	                   typename OP::INCREMENT_TYPE &increment_value) {
		start_value = StartListValue(row_idx);
		end_value = EndListValue(row_idx);
		increment_value = ListIncrementValue(row_idx);
	}

	uint64_t ListLength(idx_t row_idx) {
		typename OP::TYPE start_value;
		typename OP::TYPE end_value;
		typename OP::INCREMENT_TYPE increment_value;
		GetListValues(row_idx, start_value, end_value, increment_value);
		return OP::ListLength(start_value, end_value, increment_value, INCLUSIVE_BOUND);
	}

private:
	DataChunk &args;
	UnifiedVectorFormat vdata[3];
};

template <class OP, bool INCLUSIVE_BOUND>
static void ListRangeFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(result.GetType().id() == LogicalTypeId::LIST);

	RangeInfoStruct<OP, INCLUSIVE_BOUND> info(args);
	idx_t args_size = 1;
	auto result_type = VectorType::CONSTANT_VECTOR;
	for (idx_t i = 0; i < args.ColumnCount(); i++) {
		if (args.data[i].GetVectorType() != VectorType::CONSTANT_VECTOR) {
			args_size = args.size();
			result_type = VectorType::FLAT_VECTOR;
			break;
		}
	}
	auto list_data = FlatVector::GetData<list_entry_t>(result);
	auto &result_validity = FlatVector::Validity(result);
	int64_t total_size = 0;
	for (idx_t i = 0; i < args_size; i++) {
		if (!info.RowIsValid(i)) {
			result_validity.SetInvalid(i);
			list_data[i].offset = total_size;
			list_data[i].length = 0;
		} else {
			list_data[i].offset = total_size;
			list_data[i].length = info.ListLength(i);
			total_size += list_data[i].length;
		}
	}

	// now construct the child vector of the list
	ListVector::Reserve(result, total_size);
	auto range_data = FlatVector::GetData<typename OP::TYPE>(ListVector::GetEntry(result));
	idx_t total_idx = 0;
	for (idx_t i = 0; i < args_size; i++) {
		typename OP::TYPE start_value = info.StartListValue(i);
		typename OP::INCREMENT_TYPE increment = info.ListIncrementValue(i);

		typename OP::TYPE range_value = start_value;
		for (idx_t range_idx = 0; range_idx < list_data[i].length; range_idx++) {
			if (range_idx > 0) {
				OP::Increment(range_value, increment);
			}
			range_data[total_idx++] = range_value;
		}
	}

	ListVector::SetListSize(result, total_size);
	result.SetVectorType(result_type);

	result.Verify(args.size());
}

void ListRangeFun::RegisterFunction(BuiltinFunctions &set) {
	// the arguments and return types are actually set in the binder function
	ScalarFunctionSet range_set("range");
	range_set.AddFunction(ScalarFunction({LogicalType::BIGINT}, LogicalType::LIST(LogicalType::BIGINT),
	                                     ListRangeFunction<NumericRangeInfo, false>));
	range_set.AddFunction(ScalarFunction({LogicalType::BIGINT, LogicalType::BIGINT},
	                                     LogicalType::LIST(LogicalType::BIGINT),
	                                     ListRangeFunction<NumericRangeInfo, false>));
	range_set.AddFunction(ScalarFunction({LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT},
	                                     LogicalType::LIST(LogicalType::BIGINT),
	                                     ListRangeFunction<NumericRangeInfo, false>));
	range_set.AddFunction(ScalarFunction({LogicalType::TIMESTAMP, LogicalType::TIMESTAMP, LogicalType::INTERVAL},
	                                     LogicalType::LIST(LogicalType::TIMESTAMP),
	                                     ListRangeFunction<TimestampRangeInfo, false>));
	set.AddFunction(range_set);

	ScalarFunctionSet generate_series("generate_series");
	generate_series.AddFunction(ScalarFunction({LogicalType::BIGINT}, LogicalType::LIST(LogicalType::BIGINT),
	                                           ListRangeFunction<NumericRangeInfo, true>));
	generate_series.AddFunction(ScalarFunction({LogicalType::BIGINT, LogicalType::BIGINT},
	                                           LogicalType::LIST(LogicalType::BIGINT),
	                                           ListRangeFunction<NumericRangeInfo, true>));
	generate_series.AddFunction(ScalarFunction({LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT},
	                                           LogicalType::LIST(LogicalType::BIGINT),
	                                           ListRangeFunction<NumericRangeInfo, true>));
	generate_series.AddFunction(ScalarFunction({LogicalType::TIMESTAMP, LogicalType::TIMESTAMP, LogicalType::INTERVAL},
	                                           LogicalType::LIST(LogicalType::TIMESTAMP),
	                                           ListRangeFunction<TimestampRangeInfo, true>));
	set.AddFunction(generate_series);
}

} // namespace duckdb






namespace duckdb {

static void CardinalityFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &map = args.data[0];
	UnifiedVectorFormat map_data;
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<uint64_t>(result);
	auto &result_validity = FlatVector::Validity(result);

	map.ToUnifiedFormat(args.size(), map_data);
	for (idx_t row = 0; row < args.size(); row++) {
		auto list_entry = ((list_entry_t *)map_data.data)[map_data.sel->get_index(row)];
		result_data[row] = list_entry.length;
		result_validity.Set(row, map_data.validity.RowIsValid(map_data.sel->get_index(row)));
	}

	if (args.size() == 1) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static unique_ptr<FunctionData> CardinalityBind(ClientContext &context, ScalarFunction &bound_function,
                                                vector<unique_ptr<Expression>> &arguments) {
	if (arguments.size() != 1) {
		throw BinderException("Cardinality must have exactly one arguments");
	}

	if (arguments[0]->return_type.id() != LogicalTypeId::MAP) {
		throw BinderException("Cardinality can only operate on MAPs");
	}

	bound_function.return_type = LogicalType::UBIGINT;
	return make_unique<VariableReturnBindData>(bound_function.return_type);
}

void CardinalityFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunction fun("cardinality", {LogicalType::ANY}, LogicalType::UBIGINT, CardinalityFunction, CardinalityBind);
	fun.varargs = LogicalType::ANY;
	fun.null_handling = FunctionNullHandling::DEFAULT_NULL_HANDLING;
	set.AddFunction(fun);
}

} // namespace duckdb








namespace duckdb {

MapInvalidReason CheckMapValidity(Vector &map, idx_t count, const SelectionVector &sel) {
	D_ASSERT(map.GetType().id() == LogicalTypeId::MAP);
	UnifiedVectorFormat map_vdata;

	map.ToUnifiedFormat(count, map_vdata);
	auto &map_validity = map_vdata.validity;

	auto list_data = ListVector::GetData(map);
	auto &keys = MapVector::GetKeys(map);
	UnifiedVectorFormat key_vdata;
	keys.ToUnifiedFormat(count, key_vdata);
	auto &key_validity = key_vdata.validity;

	for (idx_t row = 0; row < count; row++) {
		auto mapped_row = sel.get_index(row);
		auto row_idx = map_vdata.sel->get_index(mapped_row);
		// map is allowed to be NULL
		if (!map_validity.RowIsValid(row_idx)) {
			continue;
		}
		row_idx = key_vdata.sel->get_index(row);
		value_set_t unique_keys;
		for (idx_t i = 0; i < list_data[row_idx].length; i++) {
			auto index = list_data[row_idx].offset + i;
			index = key_vdata.sel->get_index(index);
			if (!key_validity.RowIsValid(index)) {
				return MapInvalidReason::NULL_KEY;
			}
			auto value = keys.GetValue(index);
			auto result = unique_keys.insert(value);
			if (!result.second) {
				return MapInvalidReason::DUPLICATE_KEY;
			}
		}
	}
	return MapInvalidReason::VALID;
}

void MapConversionVerify(Vector &vector, idx_t count) {
	auto valid_check = CheckMapValidity(vector, count);
	switch (valid_check) {
	case MapInvalidReason::VALID:
		break;
	case MapInvalidReason::DUPLICATE_KEY: {
		throw InvalidInputException("Map keys have to be unique");
	}
	case MapInvalidReason::NULL_KEY: {
		throw InvalidInputException("Map keys can not be NULL");
	}
	case MapInvalidReason::NULL_KEY_LIST: {
		throw InvalidInputException("The list of map keys is not allowed to be NULL");
	}
	default: {
		throw InternalException("MapInvalidReason not implemented");
	}
	}
}

static void MapFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(result.GetType().id() == LogicalTypeId::MAP);

	//! Otherwise if its not a constant vector, this breaks the optimizer
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	for (idx_t i = 0; i < args.ColumnCount(); i++) {
		if (args.data[i].GetVectorType() != VectorType::CONSTANT_VECTOR) {
			result.SetVectorType(VectorType::FLAT_VECTOR);
		}
	}

	auto &key_vector = MapVector::GetKeys(result);
	auto &value_vector = MapVector::GetValues(result);
	auto list_data = ListVector::GetData(result);

	if (args.data.empty()) {
		ListVector::SetListSize(result, 0);
		list_data->offset = 0;
		list_data->length = 0;
		result.Verify(args.size());
		return;
	}

	auto args_data = ListVector::GetData(args.data[0]);
	auto key_count = ListVector::GetListSize(args.data[0]);
	auto value_count = ListVector::GetListSize(args.data[1]);
	if (key_count != value_count) {
		throw InvalidInputException(
		    "Error in MAP creation: key list has a different size from value list (%lld keys, %lld values)", key_count,
		    value_count);
	}
	ListVector::Reserve(result, key_count);
	ListVector::SetListSize(result, key_count);

	for (idx_t i = 0; i < args.size(); i++) {
		list_data[i] = args_data[i];
	}

	key_vector.Reference(ListVector::GetEntry(args.data[0]));
	value_vector.Reference(ListVector::GetEntry(args.data[1]));
	MapConversionVerify(result, args.size());
	result.Verify(args.size());
}

static unique_ptr<FunctionData> MapBind(ClientContext &context, ScalarFunction &bound_function,
                                        vector<unique_ptr<Expression>> &arguments) {
	child_list_t<LogicalType> child_types;

	if (arguments.size() != 2 && !arguments.empty()) {
		throw Exception("We need exactly two lists for a map");
	}
	if (arguments.size() == 2) {
		if (arguments[0]->return_type.id() != LogicalTypeId::LIST) {
			throw Exception("First argument is not a list");
		}
		if (arguments[1]->return_type.id() != LogicalTypeId::LIST) {
			throw Exception("Second argument is not a list");
		}
		child_types.push_back(make_pair("key", arguments[0]->return_type));
		child_types.push_back(make_pair("value", arguments[1]->return_type));
	}

	if (arguments.empty()) {
		auto empty = LogicalType::LIST(LogicalTypeId::SQLNULL);
		child_types.push_back(make_pair("key", empty));
		child_types.push_back(make_pair("value", empty));
	}

	bound_function.return_type =
	    LogicalType::MAP(ListType::GetChildType(child_types[0].second), ListType::GetChildType(child_types[1].second));

	return make_unique<VariableReturnBindData>(bound_function.return_type);
}

void MapFun::RegisterFunction(BuiltinFunctions &set) {
	//! the arguments and return types are actually set in the binder function
	ScalarFunction fun("map", {}, LogicalTypeId::MAP, MapFunction, MapBind);
	fun.varargs = LogicalType::ANY;
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	set.AddFunction(fun);
}

} // namespace duckdb






namespace duckdb {
void FillResult(Value &values, Vector &result, idx_t row) {
	//! First Initialize List Vector
	idx_t current_offset = ListVector::GetListSize(result);
	//! Push Values to List Vector
	auto &list_values = ListValue::GetChildren(values);
	for (idx_t i = 0; i < list_values.size(); i++) {
		ListVector::PushBack(result, list_values[i]);
	}

	//! now set the pointer
	auto &entry = ListVector::GetData(result)[row];
	entry.length = list_values.size();
	entry.offset = current_offset;
}

static void MapExtractFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.data.size() == 2);
	D_ASSERT(args.data[0].GetType().id() == LogicalTypeId::MAP);
	result.SetVectorType(VectorType::FLAT_VECTOR);

	if (args.data[1].GetType().id() == LogicalTypeId::SQLNULL) {
		//! We don't need to look through the map if the 'key' to look for is NULL
		ListVector::SetListSize(result, 0);
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		auto list_data = ConstantVector::GetData<list_entry_t>(result);
		list_data->offset = 0;
		list_data->length = 0;
		result.Verify(args.size());
		return;
	}

	auto &map = args.data[0];
	auto &key = args.data[1];

	UnifiedVectorFormat map_data;

	auto &map_keys = MapVector::GetKeys(map);
	auto &map_values = MapVector::GetValues(map);

	map.ToUnifiedFormat(args.size(), map_data);

	for (idx_t row = 0; row < args.size(); row++) {
		idx_t row_index = map_data.sel->get_index(row);
		auto key_value = key.GetValue(row);

		list_entry_t entry = ListVector::GetData(map)[row_index];
		auto offsets = MapVector::Search(map_keys, args.size(), key_value, entry);
		auto values = FlatVector::GetValuesFromOffsets(map_values, offsets);
		FillResult(values, result, row);
	}

	if (args.size() == 1) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}

	result.Verify(args.size());
}

static unique_ptr<FunctionData> MapExtractBind(ClientContext &context, ScalarFunction &bound_function,
                                               vector<unique_ptr<Expression>> &arguments) {
	if (arguments.size() != 2) {
		throw BinderException("MAP_EXTRACT must have exactly two arguments");
	}
	if (arguments[0]->return_type.id() != LogicalTypeId::MAP) {
		throw BinderException("MAP_EXTRACT can only operate on MAPs");
	}
	auto &value_type = MapType::ValueType(arguments[0]->return_type);

	//! Here we have to construct the List Type that will be returned
	bound_function.return_type = LogicalType::LIST(value_type);
	auto key_type = MapType::KeyType(arguments[0]->return_type);
	if (key_type.id() != LogicalTypeId::SQLNULL && arguments[1]->return_type.id() != LogicalTypeId::SQLNULL) {
		bound_function.arguments[1] = MapType::KeyType(arguments[0]->return_type);
	}
	return make_unique<VariableReturnBindData>(value_type);
}

void MapExtractFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunction fun("map_extract", {LogicalType::ANY, LogicalType::ANY}, LogicalType::ANY, MapExtractFunction,
	                   MapExtractBind);
	fun.varargs = LogicalType::ANY;
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	set.AddFunction(fun);
	fun.name = "element_at";
	set.AddFunction(fun);
}

} // namespace duckdb






namespace duckdb {

static void MapFromEntriesFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto count = args.size();

	result.Reinterpret(args.data[0]);

	MapConversionVerify(result, count);
	result.Verify(count);

	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static unique_ptr<FunctionData> MapFromEntriesBind(ClientContext &context, ScalarFunction &bound_function,
                                                   vector<unique_ptr<Expression>> &arguments) {
	if (arguments.size() != 1) {
		throw InvalidInputException("The input argument must be a list of structs.");
	}
	auto &list = arguments[0]->return_type;

	if (list.id() == LogicalTypeId::UNKNOWN) {
		bound_function.arguments.emplace_back(LogicalTypeId::UNKNOWN);
		bound_function.return_type = LogicalType(LogicalTypeId::SQLNULL);
		return nullptr;
	}

	if (list.id() != LogicalTypeId::LIST) {
		throw InvalidInputException("The provided argument is not a list of structs");
	}
	auto &elem_type = ListType::GetChildType(list);
	if (elem_type.id() != LogicalTypeId::STRUCT) {
		throw InvalidInputException("The elements of the list must be structs");
	}
	auto &children = StructType::GetChildTypes(elem_type);
	if (children.size() != 2) {
		throw InvalidInputException("The provided struct type should only contain 2 fields, a key and a value");
	}

	bound_function.return_type = LogicalType::MAP(elem_type);
	return make_unique<VariableReturnBindData>(bound_function.return_type);
}

void MapFromEntriesFun::RegisterFunction(BuiltinFunctions &set) {
	//! the arguments and return types are actually set in the binder function
	ScalarFunction fun("map_from_entries", {}, LogicalTypeId::MAP, MapFromEntriesFunction, MapFromEntriesBind);
	fun.null_handling = FunctionNullHandling::DEFAULT_NULL_HANDLING;
	fun.varargs = LogicalType::ANY;
	set.AddFunction(fun);
}

} // namespace duckdb












#include <cmath>
#include <errno.h>

namespace duckdb {

template <class TR, class OP>
static scalar_function_t GetScalarIntegerUnaryFunctionFixedReturn(const LogicalType &type) {
	scalar_function_t function;
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
		function = &ScalarFunction::UnaryFunction<int8_t, TR, OP>;
		break;
	case LogicalTypeId::SMALLINT:
		function = &ScalarFunction::UnaryFunction<int16_t, TR, OP>;
		break;
	case LogicalTypeId::INTEGER:
		function = &ScalarFunction::UnaryFunction<int32_t, TR, OP>;
		break;
	case LogicalTypeId::BIGINT:
		function = &ScalarFunction::UnaryFunction<int64_t, TR, OP>;
		break;
	case LogicalTypeId::HUGEINT:
		function = &ScalarFunction::UnaryFunction<hugeint_t, TR, OP>;
		break;
	default:
		throw NotImplementedException("Unimplemented type for GetScalarIntegerUnaryFunctionFixedReturn");
	}
	return function;
}

//===--------------------------------------------------------------------===//
// nextafter
//===--------------------------------------------------------------------===//
struct NextAfterOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA base, TB exponent) {
		throw NotImplementedException("Unimplemented type for NextAfter Function");
	}

	template <class TA, class TB, class TR>
	static inline double Operation(double input, double approximate_to) {
		return nextafter(input, approximate_to);
	}
	template <class TA, class TB, class TR>
	static inline float Operation(float input, float approximate_to) {
		return nextafterf(input, approximate_to);
	}
};

void NextAfterFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet next_after_fun("nextafter");
	next_after_fun.AddFunction(
	    ScalarFunction("nextafter", {LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                   ScalarFunction::BinaryFunction<double, double, double, NextAfterOperator>));
	next_after_fun.AddFunction(ScalarFunction("nextafter", {LogicalType::FLOAT, LogicalType::FLOAT}, LogicalType::FLOAT,
	                                          ScalarFunction::BinaryFunction<float, float, float, NextAfterOperator>));
	set.AddFunction(next_after_fun);
}

//===--------------------------------------------------------------------===//
// abs
//===--------------------------------------------------------------------===//
static unique_ptr<BaseStatistics> PropagateAbsStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &expr = input.expr;
	D_ASSERT(child_stats.size() == 1);
	// can only propagate stats if the children have stats
	if (!child_stats[0]) {
		return nullptr;
	}
	auto &lstats = (NumericStatistics &)*child_stats[0];
	Value new_min, new_max;
	bool potential_overflow = true;
	if (!lstats.min.IsNull() && !lstats.max.IsNull()) {
		switch (expr.return_type.InternalType()) {
		case PhysicalType::INT8:
			potential_overflow = lstats.min.GetValue<int8_t>() == NumericLimits<int8_t>::Minimum();
			break;
		case PhysicalType::INT16:
			potential_overflow = lstats.min.GetValue<int16_t>() == NumericLimits<int16_t>::Minimum();
			break;
		case PhysicalType::INT32:
			potential_overflow = lstats.min.GetValue<int32_t>() == NumericLimits<int32_t>::Minimum();
			break;
		case PhysicalType::INT64:
			potential_overflow = lstats.min.GetValue<int64_t>() == NumericLimits<int64_t>::Minimum();
			break;
		default:
			return nullptr;
		}
	}
	if (potential_overflow) {
		new_min = Value(expr.return_type);
		new_max = Value(expr.return_type);
	} else {
		// no potential overflow

		// compute stats
		auto current_min = lstats.min.GetValue<int64_t>();
		auto current_max = lstats.max.GetValue<int64_t>();

		int64_t min_val, max_val;

		if (current_min < 0 && current_max < 0) {
			// if both min and max are below zero, then min=abs(cur_max) and max=abs(cur_min)
			min_val = AbsValue(current_max);
			max_val = AbsValue(current_min);
		} else if (current_min < 0) {
			D_ASSERT(current_max >= 0);
			// if min is below zero and max is above 0, then min=0 and max=max(cur_max, abs(cur_min))
			min_val = 0;
			max_val = MaxValue(AbsValue(current_min), current_max);
		} else {
			// if both current_min and current_max are > 0, then the abs is a no-op and can be removed entirely
			*input.expr_ptr = std::move(input.expr.children[0]);
			return std::move(child_stats[0]);
		}
		new_min = Value::Numeric(expr.return_type, min_val);
		new_max = Value::Numeric(expr.return_type, max_val);
		expr.function.function = ScalarFunction::GetScalarUnaryFunction<AbsOperator>(expr.return_type);
	}
	auto stats = make_unique<NumericStatistics>(expr.return_type, std::move(new_min), std::move(new_max),
	                                            StatisticsType::LOCAL_STATS);
	stats->validity_stats = lstats.validity_stats->Copy();
	return std::move(stats);
}

template <class OP>
unique_ptr<FunctionData> DecimalUnaryOpBind(ClientContext &context, ScalarFunction &bound_function,
                                            vector<unique_ptr<Expression>> &arguments) {
	auto decimal_type = arguments[0]->return_type;
	switch (decimal_type.InternalType()) {
	case PhysicalType::INT16:
		bound_function.function = ScalarFunction::GetScalarUnaryFunction<OP>(LogicalTypeId::SMALLINT);
		break;
	case PhysicalType::INT32:
		bound_function.function = ScalarFunction::GetScalarUnaryFunction<OP>(LogicalTypeId::INTEGER);
		break;
	case PhysicalType::INT64:
		bound_function.function = ScalarFunction::GetScalarUnaryFunction<OP>(LogicalTypeId::BIGINT);
		break;
	default:
		bound_function.function = ScalarFunction::GetScalarUnaryFunction<OP>(LogicalTypeId::HUGEINT);
		break;
	}
	bound_function.arguments[0] = decimal_type;
	bound_function.return_type = decimal_type;
	return nullptr;
}

void AbsFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet abs("abs");
	for (auto &type : LogicalType::Numeric()) {
		switch (type.id()) {
		case LogicalTypeId::DECIMAL:
			abs.AddFunction(ScalarFunction({type}, type, nullptr, DecimalUnaryOpBind<AbsOperator>));
			break;
		case LogicalTypeId::TINYINT:
		case LogicalTypeId::SMALLINT:
		case LogicalTypeId::INTEGER:
		case LogicalTypeId::BIGINT: {
			ScalarFunction func({type}, type, ScalarFunction::GetScalarUnaryFunction<TryAbsOperator>(type));
			func.statistics = PropagateAbsStats;
			abs.AddFunction(func);
			break;
		}
		case LogicalTypeId::UTINYINT:
		case LogicalTypeId::USMALLINT:
		case LogicalTypeId::UINTEGER:
		case LogicalTypeId::UBIGINT:
			abs.AddFunction(ScalarFunction({type}, type, ScalarFunction::NopFunction));
			break;
		default:
			abs.AddFunction(ScalarFunction({type}, type, ScalarFunction::GetScalarUnaryFunction<AbsOperator>(type)));
			break;
		}
	}
	set.AddFunction(abs);
	abs.name = "@";
	set.AddFunction(abs);
}

//===--------------------------------------------------------------------===//
// bit_count
//===--------------------------------------------------------------------===//
struct BitCntOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		using TU = typename std::make_unsigned<TA>::type;
		TR count = 0;
		for (auto value = TU(input); value; ++count) {
			value &= (value - 1);
		}
		return count;
	}
};

struct HugeIntBitCntOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		using TU = typename std::make_unsigned<int64_t>::type;
		TR count = 0;

		for (auto value = TU(input.upper); value; ++count) {
			value &= (value - 1);
		}
		for (auto value = TU(input.lower); value; ++count) {
			value &= (value - 1);
		}
		return count;
	}
};

struct BitStringBitCntOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		TR count = Bit::BitCount(input);
		return count;
	}
};

void BitCountFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet functions("bit_count");
	functions.AddFunction(ScalarFunction({LogicalType::TINYINT}, LogicalType::TINYINT,
	                                     ScalarFunction::UnaryFunction<int8_t, int8_t, BitCntOperator>));
	functions.AddFunction(ScalarFunction({LogicalType::SMALLINT}, LogicalType::TINYINT,
	                                     ScalarFunction::UnaryFunction<int16_t, int8_t, BitCntOperator>));
	functions.AddFunction(ScalarFunction({LogicalType::INTEGER}, LogicalType::TINYINT,
	                                     ScalarFunction::UnaryFunction<int32_t, int8_t, BitCntOperator>));
	functions.AddFunction(ScalarFunction({LogicalType::BIGINT}, LogicalType::TINYINT,
	                                     ScalarFunction::UnaryFunction<int64_t, int8_t, BitCntOperator>));
	functions.AddFunction(ScalarFunction({LogicalType::HUGEINT}, LogicalType::TINYINT,
	                                     ScalarFunction::UnaryFunction<hugeint_t, int8_t, HugeIntBitCntOperator>));
	functions.AddFunction(ScalarFunction({LogicalType::BIT}, LogicalType::BIGINT,
	                                     ScalarFunction::UnaryFunction<string_t, idx_t, BitStringBitCntOperator>));
	set.AddFunction(functions);
}

//===--------------------------------------------------------------------===//
// sign
//===--------------------------------------------------------------------===//
struct SignOperator {
	template <class TA, class TR>
	static TR Operation(TA input) {
		if (input == TA(0)) {
			return 0;
		} else if (input > TA(0)) {
			return 1;
		} else {
			return -1;
		}
	}
};

template <>
int8_t SignOperator::Operation(float input) {
	if (input == 0 || Value::IsNan(input)) {
		return 0;
	} else if (input > 0) {
		return 1;
	} else {
		return -1;
	}
}

template <>
int8_t SignOperator::Operation(double input) {
	if (input == 0 || Value::IsNan(input)) {
		return 0;
	} else if (input > 0) {
		return 1;
	} else {
		return -1;
	}
}

void SignFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet sign("sign");
	for (auto &type : LogicalType::Numeric()) {
		if (type.id() == LogicalTypeId::DECIMAL) {
			continue;
		} else {
			sign.AddFunction(
			    ScalarFunction({type}, LogicalType::TINYINT,
			                   ScalarFunction::GetScalarUnaryFunctionFixedReturn<int8_t, SignOperator>(type)));
		}
	}
	set.AddFunction(sign);
}

//===--------------------------------------------------------------------===//
// ceil
//===--------------------------------------------------------------------===//
struct CeilOperator {
	template <class TA, class TR>
	static inline TR Operation(TA left) {
		return std::ceil(left);
	}
};

template <class T, class POWERS_OF_TEN, class OP>
static void GenericRoundFunctionDecimal(DataChunk &input, ExpressionState &state, Vector &result) {
	auto &func_expr = (BoundFunctionExpression &)state.expr;
	OP::template Operation<T, POWERS_OF_TEN>(input, DecimalType::GetScale(func_expr.children[0]->return_type), result);
}

template <class OP>
unique_ptr<FunctionData> BindGenericRoundFunctionDecimal(ClientContext &context, ScalarFunction &bound_function,
                                                         vector<unique_ptr<Expression>> &arguments) {
	// ceil essentially removes the scale
	auto &decimal_type = arguments[0]->return_type;
	auto scale = DecimalType::GetScale(decimal_type);
	auto width = DecimalType::GetWidth(decimal_type);
	if (scale == 0) {
		bound_function.function = ScalarFunction::NopFunction;
	} else {
		switch (decimal_type.InternalType()) {
		case PhysicalType::INT16:
			bound_function.function = GenericRoundFunctionDecimal<int16_t, NumericHelper, OP>;
			break;
		case PhysicalType::INT32:
			bound_function.function = GenericRoundFunctionDecimal<int32_t, NumericHelper, OP>;
			break;
		case PhysicalType::INT64:
			bound_function.function = GenericRoundFunctionDecimal<int64_t, NumericHelper, OP>;
			break;
		default:
			bound_function.function = GenericRoundFunctionDecimal<hugeint_t, Hugeint, OP>;
			break;
		}
	}
	bound_function.arguments[0] = decimal_type;
	bound_function.return_type = LogicalType::DECIMAL(width, 0);
	return nullptr;
}

struct CeilDecimalOperator {
	template <class T, class POWERS_OF_TEN_CLASS>
	static void Operation(DataChunk &input, uint8_t scale, Vector &result) {
		T power_of_ten = POWERS_OF_TEN_CLASS::POWERS_OF_TEN[scale];
		UnaryExecutor::Execute<T, T>(input.data[0], result, input.size(), [&](T input) {
			if (input < 0) {
				// below 0 we floor the number (e.g. -10.5 -> -10)
				return input / power_of_ten;
			} else {
				// above 0 we ceil the number
				return ((input - 1) / power_of_ten) + 1;
			}
		});
	}
};

void CeilFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet ceil("ceil");
	for (auto &type : LogicalType::Numeric()) {
		scalar_function_t func = nullptr;
		bind_scalar_function_t bind_func = nullptr;
		if (type.IsIntegral()) {
			// no ceil for integral numbers
			continue;
		}
		switch (type.id()) {
		case LogicalTypeId::FLOAT:
			func = ScalarFunction::UnaryFunction<float, float, CeilOperator>;
			break;
		case LogicalTypeId::DOUBLE:
			func = ScalarFunction::UnaryFunction<double, double, CeilOperator>;
			break;
		case LogicalTypeId::DECIMAL:
			bind_func = BindGenericRoundFunctionDecimal<CeilDecimalOperator>;
			break;
		default:
			throw InternalException("Unimplemented numeric type for function \"ceil\"");
		}
		ceil.AddFunction(ScalarFunction({type}, type, func, bind_func));
	}

	set.AddFunction(ceil);
	ceil.name = "ceiling";
	set.AddFunction(ceil);
}

//===--------------------------------------------------------------------===//
// floor
//===--------------------------------------------------------------------===//
struct FloorOperator {
	template <class TA, class TR>
	static inline TR Operation(TA left) {
		return std::floor(left);
	}
};

struct FloorDecimalOperator {
	template <class T, class POWERS_OF_TEN_CLASS>
	static void Operation(DataChunk &input, uint8_t scale, Vector &result) {
		T power_of_ten = POWERS_OF_TEN_CLASS::POWERS_OF_TEN[scale];
		UnaryExecutor::Execute<T, T>(input.data[0], result, input.size(), [&](T input) {
			if (input < 0) {
				// below 0 we ceil the number (e.g. -10.5 -> -11)
				return ((input + 1) / power_of_ten) - 1;
			} else {
				// above 0 we floor the number
				return input / power_of_ten;
			}
		});
	}
};

void FloorFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet floor("floor");
	for (auto &type : LogicalType::Numeric()) {
		scalar_function_t func = nullptr;
		bind_scalar_function_t bind_func = nullptr;
		if (type.IsIntegral()) {
			// no floor for integral numbers
			continue;
		}
		switch (type.id()) {
		case LogicalTypeId::FLOAT:
			func = ScalarFunction::UnaryFunction<float, float, FloorOperator>;
			break;
		case LogicalTypeId::DOUBLE:
			func = ScalarFunction::UnaryFunction<double, double, FloorOperator>;
			break;
		case LogicalTypeId::DECIMAL:
			bind_func = BindGenericRoundFunctionDecimal<FloorDecimalOperator>;
			break;
		default:
			throw InternalException("Unimplemented numeric type for function \"floor\"");
		}
		floor.AddFunction(ScalarFunction({type}, type, func, bind_func));
	}
	set.AddFunction(floor);
}

//===--------------------------------------------------------------------===//
// round
//===--------------------------------------------------------------------===//
struct RoundOperatorPrecision {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA input, TB precision) {
		double rounded_value;
		if (precision < 0) {
			double modifier = std::pow(10, -precision);
			rounded_value = (std::round(input / modifier)) * modifier;
			if (std::isinf(rounded_value) || std::isnan(rounded_value)) {
				return 0;
			}
		} else {
			double modifier = std::pow(10, precision);
			rounded_value = (std::round(input * modifier)) / modifier;
			if (std::isinf(rounded_value) || std::isnan(rounded_value)) {
				return input;
			}
		}
		return rounded_value;
	}
};

struct RoundOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		double rounded_value = round(input);
		if (std::isinf(rounded_value) || std::isnan(rounded_value)) {
			return input;
		}
		return rounded_value;
	}
};

struct RoundDecimalOperator {
	template <class T, class POWERS_OF_TEN_CLASS>
	static void Operation(DataChunk &input, uint8_t scale, Vector &result) {
		T power_of_ten = POWERS_OF_TEN_CLASS::POWERS_OF_TEN[scale];
		T addition = power_of_ten / 2;
		// regular round rounds towards the nearest number
		// in case of a tie we round away from zero
		// i.e. -10.5 -> -11, 10.5 -> 11
		// we implement this by adding (positive) or subtracting (negative) 0.5
		// and then flooring the number
		// e.g. 10.5 + 0.5 = 11, floor(11) = 11
		//      10.4 + 0.5 = 10.9, floor(10.9) = 10
		UnaryExecutor::Execute<T, T>(input.data[0], result, input.size(), [&](T input) {
			if (input < 0) {
				input -= addition;
			} else {
				input += addition;
			}
			return input / power_of_ten;
		});
	}
};

struct RoundPrecisionFunctionData : public FunctionData {
	explicit RoundPrecisionFunctionData(int32_t target_scale) : target_scale(target_scale) {
	}

	int32_t target_scale;

	unique_ptr<FunctionData> Copy() const override {
		return make_unique<RoundPrecisionFunctionData>(target_scale);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const RoundPrecisionFunctionData &)other_p;
		return target_scale == other.target_scale;
	}
};

template <class T, class POWERS_OF_TEN_CLASS>
static void DecimalRoundNegativePrecisionFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	auto &func_expr = (BoundFunctionExpression &)state.expr;
	auto &info = (RoundPrecisionFunctionData &)*func_expr.bind_info;
	auto source_scale = DecimalType::GetScale(func_expr.children[0]->return_type);
	auto width = DecimalType::GetWidth(func_expr.children[0]->return_type);
	if (-info.target_scale >= width) {
		// scale too big for width
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		result.SetValue(0, Value::INTEGER(0));
		return;
	}
	T divide_power_of_ten = POWERS_OF_TEN_CLASS::POWERS_OF_TEN[-info.target_scale + source_scale];
	T multiply_power_of_ten = POWERS_OF_TEN_CLASS::POWERS_OF_TEN[-info.target_scale];
	T addition = divide_power_of_ten / 2;

	UnaryExecutor::Execute<T, T>(input.data[0], result, input.size(), [&](T input) {
		if (input < 0) {
			input -= addition;
		} else {
			input += addition;
		}
		return input / divide_power_of_ten * multiply_power_of_ten;
	});
}

template <class T, class POWERS_OF_TEN_CLASS>
static void DecimalRoundPositivePrecisionFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	auto &func_expr = (BoundFunctionExpression &)state.expr;
	auto &info = (RoundPrecisionFunctionData &)*func_expr.bind_info;
	auto source_scale = DecimalType::GetScale(func_expr.children[0]->return_type);
	T power_of_ten = POWERS_OF_TEN_CLASS::POWERS_OF_TEN[source_scale - info.target_scale];
	T addition = power_of_ten / 2;
	UnaryExecutor::Execute<T, T>(input.data[0], result, input.size(), [&](T input) {
		if (input < 0) {
			input -= addition;
		} else {
			input += addition;
		}
		return input / power_of_ten;
	});
}

unique_ptr<FunctionData> BindDecimalRoundPrecision(ClientContext &context, ScalarFunction &bound_function,
                                                   vector<unique_ptr<Expression>> &arguments) {
	auto &decimal_type = arguments[0]->return_type;
	if (arguments[1]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (!arguments[1]->IsFoldable()) {
		throw NotImplementedException("ROUND(DECIMAL, INTEGER) with non-constant precision is not supported");
	}
	Value val = ExpressionExecutor::EvaluateScalar(context, *arguments[1]).DefaultCastAs(LogicalType::INTEGER);
	if (val.IsNull()) {
		throw NotImplementedException("ROUND(DECIMAL, INTEGER) with non-constant precision is not supported");
	}
	// our new precision becomes the round value
	// e.g. ROUND(DECIMAL(18,3), 1) -> DECIMAL(18,1)
	// but ONLY if the round value is positive
	// if it is negative the scale becomes zero
	// i.e. ROUND(DECIMAL(18,3), -1) -> DECIMAL(18,0)
	int32_t round_value = IntegerValue::Get(val);
	uint8_t target_scale;
	auto width = DecimalType::GetWidth(decimal_type);
	auto scale = DecimalType::GetScale(decimal_type);
	if (round_value < 0) {
		target_scale = 0;
		switch (decimal_type.InternalType()) {
		case PhysicalType::INT16:
			bound_function.function = DecimalRoundNegativePrecisionFunction<int16_t, NumericHelper>;
			break;
		case PhysicalType::INT32:
			bound_function.function = DecimalRoundNegativePrecisionFunction<int32_t, NumericHelper>;
			break;
		case PhysicalType::INT64:
			bound_function.function = DecimalRoundNegativePrecisionFunction<int64_t, NumericHelper>;
			break;
		default:
			bound_function.function = DecimalRoundNegativePrecisionFunction<hugeint_t, Hugeint>;
			break;
		}
	} else {
		if (round_value >= (int32_t)scale) {
			// if round_value is bigger than or equal to scale we do nothing
			bound_function.function = ScalarFunction::NopFunction;
			target_scale = scale;
		} else {
			target_scale = round_value;
			switch (decimal_type.InternalType()) {
			case PhysicalType::INT16:
				bound_function.function = DecimalRoundPositivePrecisionFunction<int16_t, NumericHelper>;
				break;
			case PhysicalType::INT32:
				bound_function.function = DecimalRoundPositivePrecisionFunction<int32_t, NumericHelper>;
				break;
			case PhysicalType::INT64:
				bound_function.function = DecimalRoundPositivePrecisionFunction<int64_t, NumericHelper>;
				break;
			default:
				bound_function.function = DecimalRoundPositivePrecisionFunction<hugeint_t, Hugeint>;
				break;
			}
		}
	}
	bound_function.arguments[0] = decimal_type;
	bound_function.return_type = LogicalType::DECIMAL(width, target_scale);
	return make_unique<RoundPrecisionFunctionData>(round_value);
}

void RoundFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet round("round");
	for (auto &type : LogicalType::Numeric()) {
		scalar_function_t round_prec_func = nullptr;
		scalar_function_t round_func = nullptr;
		bind_scalar_function_t bind_func = nullptr;
		bind_scalar_function_t bind_prec_func = nullptr;
		if (type.IsIntegral()) {
			// no round for integral numbers
			continue;
		}
		switch (type.id()) {
		case LogicalTypeId::FLOAT:
			round_func = ScalarFunction::UnaryFunction<float, float, RoundOperator>;
			round_prec_func = ScalarFunction::BinaryFunction<float, int32_t, float, RoundOperatorPrecision>;
			break;
		case LogicalTypeId::DOUBLE:
			round_func = ScalarFunction::UnaryFunction<double, double, RoundOperator>;
			round_prec_func = ScalarFunction::BinaryFunction<double, int32_t, double, RoundOperatorPrecision>;
			break;
		case LogicalTypeId::DECIMAL:
			bind_func = BindGenericRoundFunctionDecimal<RoundDecimalOperator>;
			bind_prec_func = BindDecimalRoundPrecision;
			break;
		default:
			throw InternalException("Unimplemented numeric type for function \"floor\"");
		}
		round.AddFunction(ScalarFunction({type}, type, round_func, bind_func));
		round.AddFunction(ScalarFunction({type, LogicalType::INTEGER}, type, round_prec_func, bind_prec_func));
	}
	set.AddFunction(round);
}

//===--------------------------------------------------------------------===//
// exp
//===--------------------------------------------------------------------===//
struct ExpOperator {
	template <class TA, class TR>
	static inline TR Operation(TA left) {
		return std::exp(left);
	}
};

void ExpFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("exp", {LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                               ScalarFunction::UnaryFunction<double, double, ExpOperator>));
}

//===--------------------------------------------------------------------===//
// pow
//===--------------------------------------------------------------------===//
struct PowOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA base, TB exponent) {
		return std::pow(base, exponent);
	}
};

void PowFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunction power_function("pow", {LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                              ScalarFunction::BinaryFunction<double, double, double, PowOperator>);
	set.AddFunction(power_function);
	power_function.name = "power";
	set.AddFunction(power_function);
	power_function.name = "**";
	set.AddFunction(power_function);
	power_function.name = "^";
	set.AddFunction(power_function);
}

//===--------------------------------------------------------------------===//
// sqrt
//===--------------------------------------------------------------------===//
struct SqrtOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		if (input < 0) {
			throw OutOfRangeException("cannot take square root of a negative number");
		}
		return std::sqrt(input);
	}
};

void SqrtFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("sqrt", {LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                               ScalarFunction::UnaryFunction<double, double, SqrtOperator>));
}

//===--------------------------------------------------------------------===//
// cbrt
//===--------------------------------------------------------------------===//
struct CbRtOperator {
	template <class TA, class TR>
	static inline TR Operation(TA left) {
		return std::cbrt(left);
	}
};

void CbrtFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("cbrt", {LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                               ScalarFunction::UnaryFunction<double, double, CbRtOperator>));
}

//===--------------------------------------------------------------------===//
// ln
//===--------------------------------------------------------------------===//

struct LnOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		if (input < 0) {
			throw OutOfRangeException("cannot take logarithm of a negative number");
		}
		if (input == 0) {
			throw OutOfRangeException("cannot take logarithm of zero");
		}
		return std::log(input);
	}
};

void LnFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("ln", {LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                               ScalarFunction::UnaryFunction<double, double, LnOperator>));
}

//===--------------------------------------------------------------------===//
// log
//===--------------------------------------------------------------------===//
struct Log10Operator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		if (input < 0) {
			throw OutOfRangeException("cannot take logarithm of a negative number");
		}
		if (input == 0) {
			throw OutOfRangeException("cannot take logarithm of zero");
		}
		return std::log10(input);
	}
};

void Log10Fun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction({"log10", "log"}, ScalarFunction({LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                                                 ScalarFunction::UnaryFunction<double, double, Log10Operator>));
}

//===--------------------------------------------------------------------===//
// log2
//===--------------------------------------------------------------------===//
struct Log2Operator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		if (input < 0) {
			throw OutOfRangeException("cannot take logarithm of a negative number");
		}
		if (input == 0) {
			throw OutOfRangeException("cannot take logarithm of zero");
		}
		return std::log2(input);
	}
};

void Log2Fun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("log2", {LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                               ScalarFunction::UnaryFunction<double, double, Log2Operator>));
}

//===--------------------------------------------------------------------===//
// pi
//===--------------------------------------------------------------------===//
static void PiFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 0);
	Value pi_value = Value::DOUBLE(PI);
	result.Reference(pi_value);
}

void PiFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("pi", {}, LogicalType::DOUBLE, PiFunction));
}

//===--------------------------------------------------------------------===//
// degrees
//===--------------------------------------------------------------------===//
struct DegreesOperator {
	template <class TA, class TR>
	static inline TR Operation(TA left) {
		return left * (180 / PI);
	}
};

void DegreesFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("degrees", {LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                               ScalarFunction::UnaryFunction<double, double, DegreesOperator>));
}

//===--------------------------------------------------------------------===//
// radians
//===--------------------------------------------------------------------===//
struct RadiansOperator {
	template <class TA, class TR>
	static inline TR Operation(TA left) {
		return left * (PI / 180);
	}
};

void RadiansFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("radians", {LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                               ScalarFunction::UnaryFunction<double, double, RadiansOperator>));
}

//===--------------------------------------------------------------------===//
// isnan
//===--------------------------------------------------------------------===//
struct IsNanOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return Value::IsNan(input);
	}
};

void IsNanFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet funcs("isnan");
	funcs.AddFunction(ScalarFunction({LogicalType::FLOAT}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<float, bool, IsNanOperator>));
	funcs.AddFunction(ScalarFunction({LogicalType::DOUBLE}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<double, bool, IsNanOperator>));
	set.AddFunction(funcs);
}

//===--------------------------------------------------------------------===//
// signbit
//===--------------------------------------------------------------------===//
struct SignBitOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return std::signbit(input);
	}
};

void SignBitFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet funcs("signbit");
	funcs.AddFunction(ScalarFunction({LogicalType::FLOAT}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<float, bool, SignBitOperator>));
	funcs.AddFunction(ScalarFunction({LogicalType::DOUBLE}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<double, bool, SignBitOperator>));
	set.AddFunction(funcs);
}

//===--------------------------------------------------------------------===//
// isinf
//===--------------------------------------------------------------------===//
struct IsInfiniteOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return !Value::IsNan(input) && !Value::IsFinite(input);
	}
};

template <>
bool IsInfiniteOperator::Operation(date_t input) {
	return !Value::IsFinite(input);
}

template <>
bool IsInfiniteOperator::Operation(timestamp_t input) {
	return !Value::IsFinite(input);
}

void IsInfiniteFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet funcs("isinf");
	funcs.AddFunction(ScalarFunction({LogicalType::FLOAT}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<float, bool, IsInfiniteOperator>));
	funcs.AddFunction(ScalarFunction({LogicalType::DOUBLE}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<double, bool, IsInfiniteOperator>));
	funcs.AddFunction(ScalarFunction({LogicalType::DATE}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<date_t, bool, IsInfiniteOperator>));
	funcs.AddFunction(ScalarFunction({LogicalType::TIMESTAMP}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<timestamp_t, bool, IsInfiniteOperator>));
	funcs.AddFunction(ScalarFunction({LogicalType::TIMESTAMP_TZ}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<timestamp_t, bool, IsInfiniteOperator>));
	set.AddFunction(funcs);
}

//===--------------------------------------------------------------------===//
// isfinite
//===--------------------------------------------------------------------===//
struct IsFiniteOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return Value::IsFinite(input);
	}
};

void IsFiniteFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet funcs("isfinite");
	funcs.AddFunction(ScalarFunction({LogicalType::FLOAT}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<float, bool, IsFiniteOperator>));
	funcs.AddFunction(ScalarFunction({LogicalType::DOUBLE}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<double, bool, IsFiniteOperator>));
	funcs.AddFunction(ScalarFunction({LogicalType::DATE}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<date_t, bool, IsFiniteOperator>));
	funcs.AddFunction(ScalarFunction({LogicalType::TIMESTAMP}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<timestamp_t, bool, IsFiniteOperator>));
	funcs.AddFunction(ScalarFunction({LogicalType::TIMESTAMP_TZ}, LogicalType::BOOLEAN,
	                                 ScalarFunction::UnaryFunction<timestamp_t, bool, IsFiniteOperator>));
	set.AddFunction(funcs);
}

//===--------------------------------------------------------------------===//
// sin
//===--------------------------------------------------------------------===//
template <class OP>
struct NoInfiniteDoubleWrapper {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input) {
		if (DUCKDB_UNLIKELY(!Value::IsFinite(input))) {
			if (Value::IsNan(input)) {
				return input;
			}
			throw OutOfRangeException("input value %lf is out of range for numeric function", input);
		}
		return OP::template Operation<INPUT_TYPE, RESULT_TYPE>(input);
	}
};

struct SinOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return std::sin(input);
	}
};

void SinFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(
	    ScalarFunction("sin", {LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                   ScalarFunction::UnaryFunction<double, double, NoInfiniteDoubleWrapper<SinOperator>>));
}

//===--------------------------------------------------------------------===//
// cos
//===--------------------------------------------------------------------===//
struct CosOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return (double)std::cos(input);
	}
};

void CosFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(
	    ScalarFunction("cos", {LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                   ScalarFunction::UnaryFunction<double, double, NoInfiniteDoubleWrapper<CosOperator>>));
}

//===--------------------------------------------------------------------===//
// tan
//===--------------------------------------------------------------------===//
struct TanOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return (double)std::tan(input);
	}
};

void TanFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(
	    ScalarFunction("tan", {LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                   ScalarFunction::UnaryFunction<double, double, NoInfiniteDoubleWrapper<TanOperator>>));
}

//===--------------------------------------------------------------------===//
// asin
//===--------------------------------------------------------------------===//
struct ASinOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		if (input < -1 || input > 1) {
			throw Exception("ASIN is undefined outside [-1,1]");
		}
		return (double)std::asin(input);
	}
};

void AsinFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(
	    ScalarFunction("asin", {LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                   ScalarFunction::UnaryFunction<double, double, NoInfiniteDoubleWrapper<ASinOperator>>));
}

//===--------------------------------------------------------------------===//
// atan
//===--------------------------------------------------------------------===//
struct ATanOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return (double)std::atan(input);
	}
};

void AtanFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("atan", {LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                               ScalarFunction::UnaryFunction<double, double, ATanOperator>));
}

//===--------------------------------------------------------------------===//
// atan2
//===--------------------------------------------------------------------===//
struct ATan2 {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA left, TB right) {
		return (double)std::atan2(left, right);
	}
};

void Atan2Fun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("atan2", {LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                               ScalarFunction::BinaryFunction<double, double, double, ATan2>));
}

//===--------------------------------------------------------------------===//
// acos
//===--------------------------------------------------------------------===//
struct ACos {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return (double)std::acos(input);
	}
};

void AcosFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("acos", {LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                               ScalarFunction::UnaryFunction<double, double, NoInfiniteDoubleWrapper<ACos>>));
}

//===--------------------------------------------------------------------===//
// cot
//===--------------------------------------------------------------------===//
struct CotOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return 1.0 / (double)std::tan(input);
	}
};

void CotFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(
	    ScalarFunction("cot", {LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                   ScalarFunction::UnaryFunction<double, double, NoInfiniteDoubleWrapper<CotOperator>>));
}

//===--------------------------------------------------------------------===//
// gamma
//===--------------------------------------------------------------------===//
struct GammaOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		if (input == 0) {
			throw OutOfRangeException("cannot take gamma of zero");
		}
		return std::tgamma(input);
	}
};

void GammaFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("gamma", {LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                               ScalarFunction::UnaryFunction<double, double, GammaOperator>));
}

//===--------------------------------------------------------------------===//
// gamma
//===--------------------------------------------------------------------===//
#if defined(OS_DARWIN)
extern "C"
{
    double lgamma_r(double x, int * signgamp);
}
#endif

/// Use wrapper and use lgamma_r version because std::lgamma is not threadsafe.
double lgamma_wrapper(double arg)
{
    int signp;
    return lgamma_r(arg, &signp);
}

struct LogGammaOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		if (input == 0) {
			throw OutOfRangeException("cannot take log gamma of zero");
		}
		return lgamma_wrapper(input);
	}
};

void LogGammaFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("lgamma", {LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                               ScalarFunction::UnaryFunction<double, double, LogGammaOperator>));
}

//===--------------------------------------------------------------------===//
// factorial(), !
//===--------------------------------------------------------------------===//

struct FactorialOperator {
	template <class TA, class TR>
	static inline TR Operation(TA left) {
		TR ret = 1;
		for (TA i = 2; i <= left; i++) {
			ret *= i;
		}
		return ret;
	}
};

void FactorialFun::RegisterFunction(BuiltinFunctions &set) {
	auto fun = ScalarFunction({LogicalType::INTEGER}, LogicalType::HUGEINT,
	                          ScalarFunction::UnaryFunction<int32_t, hugeint_t, FactorialOperator>);

	set.AddFunction({"factorial", "!__postfix"}, fun);
}

//===--------------------------------------------------------------------===//
// even
//===--------------------------------------------------------------------===//
struct EvenOperator {
	template <class TA, class TR>
	static inline TR Operation(TA left) {
		double value;
		if (left >= 0) {
			value = std::ceil(left);
		} else {
			value = std::ceil(-left);
			value = -value;
		}
		if (std::floor(value / 2) * 2 != value) {
			if (left >= 0) {
				return value += 1;
			}
			return value -= 1;
		}
		return value;
	}
};

void EvenFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("even", {LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                               ScalarFunction::UnaryFunction<double, double, EvenOperator>));
}

} // namespace duckdb









namespace duckdb {

struct RandomLocalState : public FunctionLocalState {
	explicit RandomLocalState(uint32_t seed) : random_engine(seed) {
	}

	RandomEngine random_engine;
};

static void RandomFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 0);
	auto &lstate = (RandomLocalState &)*ExecuteFunctionState::GetFunctionState(state);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<double>(result);
	for (idx_t i = 0; i < args.size(); i++) {
		result_data[i] = lstate.random_engine.NextRandom();
	}
}

static unique_ptr<FunctionLocalState> RandomInitLocalState(ExpressionState &state, const BoundFunctionExpression &expr,
                                                           FunctionData *bind_data) {
	auto &random_engine = RandomEngine::Get(state.GetContext());
	lock_guard<mutex> guard(random_engine.lock);
	return make_unique<RandomLocalState>(random_engine.NextRandomInteger());
}

void RandomFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunction random("random", {}, LogicalType::DOUBLE, RandomFunction, nullptr, nullptr, nullptr,
	                      RandomInitLocalState);
	random.side_effects = FunctionSideEffects::HAS_SIDE_EFFECTS;
	set.AddFunction(random);
}

static void GenerateUUIDFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 0);
	auto &lstate = (RandomLocalState &)*ExecuteFunctionState::GetFunctionState(state);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<hugeint_t>(result);

	for (idx_t i = 0; i < args.size(); i++) {
		result_data[i] = UUID::GenerateRandomUUID(lstate.random_engine);
	}
}

void UUIDFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunction uuid_function({}, LogicalType::UUID, GenerateUUIDFunction, nullptr, nullptr, nullptr,
	                             RandomInitLocalState);
	// generate a random uuid
	uuid_function.side_effects = FunctionSideEffects::HAS_SIDE_EFFECTS;
	set.AddFunction({"uuid", "gen_random_uuid"}, uuid_function);
}

} // namespace duckdb









namespace duckdb {

struct SetseedBindData : public FunctionData {
	//! The client context for the function call
	ClientContext &context;

	explicit SetseedBindData(ClientContext &context) : context(context) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_unique<SetseedBindData>(context);
	}

	bool Equals(const FunctionData &other_p) const override {
		return true;
	}
};

static void SetSeedFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = (BoundFunctionExpression &)state.expr;
	auto &info = (SetseedBindData &)*func_expr.bind_info;
	auto &input = args.data[0];
	input.Flatten(args.size());

	auto input_seeds = FlatVector::GetData<double>(input);
	uint32_t half_max = NumericLimits<uint32_t>::Maximum() / 2;

	auto &random_engine = RandomEngine::Get(info.context);
	for (idx_t i = 0; i < args.size(); i++) {
		if (input_seeds[i] < -1.0 || input_seeds[i] > 1.0 || Value::IsNan(input_seeds[i])) {
			throw Exception("SETSEED accepts seed values between -1.0 and 1.0, inclusive");
		}
		uint32_t norm_seed = (input_seeds[i] + 1.0) * half_max;
		random_engine.SetSeed(norm_seed);
	}

	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	ConstantVector::SetNull(result, true);
}

unique_ptr<FunctionData> SetSeedBind(ClientContext &context, ScalarFunction &bound_function,
                                     vector<unique_ptr<Expression>> &arguments) {
	return make_unique<SetseedBindData>(context);
}

void SetseedFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunction setseed("setseed", {LogicalType::DOUBLE}, LogicalType::SQLNULL, SetSeedFunction, SetSeedBind);
	setseed.side_effects = FunctionSideEffects::HAS_SIDE_EFFECTS;
	set.AddFunction(setseed);
}

} // namespace duckdb



namespace duckdb {

void BuiltinFunctions::RegisterMathFunctions() {
	Register<AbsFun>();
	Register<SignFun>();

	Register<CeilFun>();
	Register<FloorFun>();
	Register<RoundFun>();

	Register<DegreesFun>();
	Register<RadiansFun>();

	Register<CbrtFun>();
	Register<ExpFun>();
	Register<Log2Fun>();
	Register<Log10Fun>();
	Register<LnFun>();
	Register<PowFun>();
	Register<RandomFun>();
	Register<SetseedFun>();
	Register<SqrtFun>();

	Register<PiFun>();

	Register<BitCountFun>();

	Register<GammaFun>();
	Register<LogGammaFun>();

	Register<FactorialFun>();

	Register<NextAfterFun>();

	Register<EvenFun>();

	Register<IsNanFun>();
	Register<SignBitFun>();
	Register<IsInfiniteFun>();
	Register<IsFiniteFun>();
}

} // namespace duckdb


namespace duckdb {

void BuiltinFunctions::RegisterNestedFunctions() {
	Register<ArraySliceFun>();
	Register<StructPackFun>();
	Register<StructExtractFun>();
	Register<StructInsertFun>();
	Register<ListTransformFun>();
	Register<ListFilterFun>();
	Register<ListConcatFun>();
	Register<ListContainsFun>();
	Register<ListPositionFun>();
	Register<ListAggregateFun>();
	Register<ListDistinctFun>();
	Register<ListUniqueFun>();
	Register<ListValueFun>();
	Register<ListExtractFun>();
	Register<ListSortFun>();
	Register<ListRangeFun>();
	Register<ListFlattenFun>();
	Register<MapFun>();
	Register<MapFromEntriesFun>();
	Register<MapExtractFun>();
	Register<UnionValueFun>();
	Register<UnionExtractFun>();
	Register<UnionTagFun>();
	Register<CardinalityFun>();
}

} // namespace duckdb











#include <limits>

namespace duckdb {

//===--------------------------------------------------------------------===//
// + [add]
//===--------------------------------------------------------------------===//
template <>
float AddOperator::Operation(float left, float right) {
	auto result = left + right;
	if (!Value::FloatIsFinite(result)) {
		throw OutOfRangeException("Overflow in addition of float!");
	}
	return result;
}

template <>
double AddOperator::Operation(double left, double right) {
	auto result = left + right;
	if (!Value::DoubleIsFinite(result)) {
		throw OutOfRangeException("Overflow in addition of double!");
	}
	return result;
}

template <>
interval_t AddOperator::Operation(interval_t left, interval_t right) {
	left.months = AddOperatorOverflowCheck::Operation<int32_t, int32_t, int32_t>(left.months, right.months);
	left.days = AddOperatorOverflowCheck::Operation<int32_t, int32_t, int32_t>(left.days, right.days);
	left.micros = AddOperatorOverflowCheck::Operation<int64_t, int64_t, int64_t>(left.micros, right.micros);
	return left;
}

template <>
date_t AddOperator::Operation(date_t left, int32_t right) {
	if (!Value::IsFinite(left)) {
		return left;
	}
	int32_t days;
	if (!TryAddOperator::Operation(left.days, right, days)) {
		throw OutOfRangeException("Date out of range");
	}
	date_t result(days);
	if (!Value::IsFinite(result)) {
		throw OutOfRangeException("Date out of range");
	}
	return result;
}

template <>
date_t AddOperator::Operation(int32_t left, date_t right) {
	return AddOperator::Operation<date_t, int32_t, date_t>(right, left);
}

template <>
timestamp_t AddOperator::Operation(date_t left, dtime_t right) {
	if (left == date_t::infinity()) {
		return timestamp_t::infinity();
	} else if (left == date_t::ninfinity()) {
		return timestamp_t::ninfinity();
	}
	timestamp_t result;
	if (!Timestamp::TryFromDatetime(left, right, result)) {
		throw OutOfRangeException("Timestamp out of range");
	}
	return result;
}

template <>
timestamp_t AddOperator::Operation(dtime_t left, date_t right) {
	return AddOperator::Operation<date_t, dtime_t, timestamp_t>(right, left);
}

template <>
date_t AddOperator::Operation(date_t left, interval_t right) {
	return Interval::Add(left, right);
}

template <>
date_t AddOperator::Operation(interval_t left, date_t right) {
	return AddOperator::Operation<date_t, interval_t, date_t>(right, left);
}

template <>
timestamp_t AddOperator::Operation(timestamp_t left, interval_t right) {
	return Interval::Add(left, right);
}

template <>
timestamp_t AddOperator::Operation(interval_t left, timestamp_t right) {
	return AddOperator::Operation<timestamp_t, interval_t, timestamp_t>(right, left);
}

//===--------------------------------------------------------------------===//
// + [add] with overflow check
//===--------------------------------------------------------------------===//
struct OverflowCheckedAddition {
	template <class SRCTYPE, class UTYPE>
	static inline bool Operation(SRCTYPE left, SRCTYPE right, SRCTYPE &result) {
		UTYPE uresult = AddOperator::Operation<UTYPE, UTYPE, UTYPE>(UTYPE(left), UTYPE(right));
		if (uresult < NumericLimits<SRCTYPE>::Minimum() || uresult > NumericLimits<SRCTYPE>::Maximum()) {
			return false;
		}
		result = SRCTYPE(uresult);
		return true;
	}
};

template <>
bool TryAddOperator::Operation(uint8_t left, uint8_t right, uint8_t &result) {
	return OverflowCheckedAddition::Operation<uint8_t, uint16_t>(left, right, result);
}
template <>
bool TryAddOperator::Operation(uint16_t left, uint16_t right, uint16_t &result) {
	return OverflowCheckedAddition::Operation<uint16_t, uint32_t>(left, right, result);
}
template <>
bool TryAddOperator::Operation(uint32_t left, uint32_t right, uint32_t &result) {
	return OverflowCheckedAddition::Operation<uint32_t, uint64_t>(left, right, result);
}

template <>
bool TryAddOperator::Operation(uint64_t left, uint64_t right, uint64_t &result) {
	if (NumericLimits<uint64_t>::Maximum() - left < right) {
		return false;
	}
	return OverflowCheckedAddition::Operation<uint64_t, uint64_t>(left, right, result);
}

template <>
bool TryAddOperator::Operation(int8_t left, int8_t right, int8_t &result) {
	return OverflowCheckedAddition::Operation<int8_t, int16_t>(left, right, result);
}

template <>
bool TryAddOperator::Operation(int16_t left, int16_t right, int16_t &result) {
	return OverflowCheckedAddition::Operation<int16_t, int32_t>(left, right, result);
}

template <>
bool TryAddOperator::Operation(int32_t left, int32_t right, int32_t &result) {
	return OverflowCheckedAddition::Operation<int32_t, int64_t>(left, right, result);
}

template <>
bool TryAddOperator::Operation(int64_t left, int64_t right, int64_t &result) {
#if (__GNUC__ >= 5) || defined(__clang__)
	if (__builtin_add_overflow(left, right, &result)) {
		return false;
	}
#else
	// https://blog.regehr.org/archives/1139
	result = int64_t((uint64_t)left + (uint64_t)right);
	if ((left < 0 && right < 0 && result >= 0) || (left >= 0 && right >= 0 && result < 0)) {
		return false;
	}
#endif
	return true;
}

//===--------------------------------------------------------------------===//
// add decimal with overflow check
//===--------------------------------------------------------------------===//
template <class T, T min, T max>
bool TryDecimalAddTemplated(T left, T right, T &result) {
	if (right < 0) {
		if (min - right > left) {
			return false;
		}
	} else {
		if (max - right < left) {
			return false;
		}
	}
	result = left + right;
	return true;
}

template <>
bool TryDecimalAdd::Operation(int16_t left, int16_t right, int16_t &result) {
	return TryDecimalAddTemplated<int16_t, -9999, 9999>(left, right, result);
}

template <>
bool TryDecimalAdd::Operation(int32_t left, int32_t right, int32_t &result) {
	return TryDecimalAddTemplated<int32_t, -999999999, 999999999>(left, right, result);
}

template <>
bool TryDecimalAdd::Operation(int64_t left, int64_t right, int64_t &result) {
	return TryDecimalAddTemplated<int64_t, -999999999999999999, 999999999999999999>(left, right, result);
}

template <>
bool TryDecimalAdd::Operation(hugeint_t left, hugeint_t right, hugeint_t &result) {
	result = left + right;
	if (result <= -Hugeint::POWERS_OF_TEN[38] || result >= Hugeint::POWERS_OF_TEN[38]) {
		return false;
	}
	return true;
}

template <>
hugeint_t DecimalAddOverflowCheck::Operation(hugeint_t left, hugeint_t right) {
	hugeint_t result;
	if (!TryDecimalAdd::Operation(left, right, result)) {
		throw OutOfRangeException("Overflow in addition of DECIMAL(38) (%s + %s);", left.ToString(), right.ToString());
	}
	return result;
}

//===--------------------------------------------------------------------===//
// add time operator
//===--------------------------------------------------------------------===//
template <>
dtime_t AddTimeOperator::Operation(dtime_t left, interval_t right) {
	date_t date(0);
	return Interval::Add(left, right, date);
}

template <>
dtime_t AddTimeOperator::Operation(interval_t left, dtime_t right) {
	return AddTimeOperator::Operation<dtime_t, interval_t, dtime_t>(right, left);
}

} // namespace duckdb
