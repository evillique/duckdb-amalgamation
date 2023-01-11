// See https://raw.githubusercontent.com/duckdb/duckdb/master/LICENSE for licensing information

#include "duckdb.hpp"
#include "duckdb-internal.hpp"
#ifndef DUCKDB_AMALGAMATION
#error header mismatch
#endif

















#include <limits>

namespace duckdb {

template <class OP>
static scalar_function_t GetScalarIntegerFunction(PhysicalType type) {
	scalar_function_t function;
	switch (type) {
	case PhysicalType::INT8:
		function = &ScalarFunction::BinaryFunction<int8_t, int8_t, int8_t, OP>;
		break;
	case PhysicalType::INT16:
		function = &ScalarFunction::BinaryFunction<int16_t, int16_t, int16_t, OP>;
		break;
	case PhysicalType::INT32:
		function = &ScalarFunction::BinaryFunction<int32_t, int32_t, int32_t, OP>;
		break;
	case PhysicalType::INT64:
		function = &ScalarFunction::BinaryFunction<int64_t, int64_t, int64_t, OP>;
		break;
	case PhysicalType::UINT8:
		function = &ScalarFunction::BinaryFunction<uint8_t, uint8_t, uint8_t, OP>;
		break;
	case PhysicalType::UINT16:
		function = &ScalarFunction::BinaryFunction<uint16_t, uint16_t, uint16_t, OP>;
		break;
	case PhysicalType::UINT32:
		function = &ScalarFunction::BinaryFunction<uint32_t, uint32_t, uint32_t, OP>;
		break;
	case PhysicalType::UINT64:
		function = &ScalarFunction::BinaryFunction<uint64_t, uint64_t, uint64_t, OP>;
		break;
	default:
		throw NotImplementedException("Unimplemented type for GetScalarBinaryFunction");
	}
	return function;
}

template <class OP>
static scalar_function_t GetScalarBinaryFunction(PhysicalType type) {
	scalar_function_t function;
	switch (type) {
	case PhysicalType::INT128:
		function = &ScalarFunction::BinaryFunction<hugeint_t, hugeint_t, hugeint_t, OP>;
		break;
	case PhysicalType::FLOAT:
		function = &ScalarFunction::BinaryFunction<float, float, float, OP>;
		break;
	case PhysicalType::DOUBLE:
		function = &ScalarFunction::BinaryFunction<double, double, double, OP>;
		break;
	default:
		function = GetScalarIntegerFunction<OP>(type);
		break;
	}
	return function;
}

//===--------------------------------------------------------------------===//
// + [add]
//===--------------------------------------------------------------------===//
struct AddPropagateStatistics {
	template <class T, class OP>
	static bool Operation(LogicalType type, NumericStatistics &lstats, NumericStatistics &rstats, Value &new_min,
	                      Value &new_max) {
		T min, max;
		// new min is min+min
		if (!OP::Operation(lstats.min.GetValueUnsafe<T>(), rstats.min.GetValueUnsafe<T>(), min)) {
			return true;
		}
		// new max is max+max
		if (!OP::Operation(lstats.max.GetValueUnsafe<T>(), rstats.max.GetValueUnsafe<T>(), max)) {
			return true;
		}
		new_min = Value::Numeric(type, min);
		new_max = Value::Numeric(type, max);
		return false;
	}
};

struct SubtractPropagateStatistics {
	template <class T, class OP>
	static bool Operation(LogicalType type, NumericStatistics &lstats, NumericStatistics &rstats, Value &new_min,
	                      Value &new_max) {
		T min, max;
		if (!OP::Operation(lstats.min.GetValueUnsafe<T>(), rstats.max.GetValueUnsafe<T>(), min)) {
			return true;
		}
		if (!OP::Operation(lstats.max.GetValueUnsafe<T>(), rstats.min.GetValueUnsafe<T>(), max)) {
			return true;
		}
		new_min = Value::Numeric(type, min);
		new_max = Value::Numeric(type, max);
		return false;
	}
};

struct DecimalArithmeticBindData : public FunctionData {
	DecimalArithmeticBindData() : check_overflow(true) {
	}

	unique_ptr<FunctionData> Copy() const override {
		auto res = make_unique<DecimalArithmeticBindData>();
		res->check_overflow = check_overflow;
		return std::move(res);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto other = (DecimalArithmeticBindData &)other_p;
		return other.check_overflow == check_overflow;
	}

	bool check_overflow;
};

template <class OP, class PROPAGATE, class BASEOP>
static unique_ptr<BaseStatistics> PropagateNumericStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &expr = input.expr;
	D_ASSERT(child_stats.size() == 2);
	// can only propagate stats if the children have stats
	if (!child_stats[0] || !child_stats[1]) {
		return nullptr;
	}
	auto &lstats = (NumericStatistics &)*child_stats[0];
	auto &rstats = (NumericStatistics &)*child_stats[1];
	Value new_min, new_max;
	bool potential_overflow = true;
	if (!lstats.min.IsNull() && !lstats.max.IsNull() && !rstats.min.IsNull() && !rstats.max.IsNull()) {
		switch (expr.return_type.InternalType()) {
		case PhysicalType::INT8:
			potential_overflow =
			    PROPAGATE::template Operation<int8_t, OP>(expr.return_type, lstats, rstats, new_min, new_max);
			break;
		case PhysicalType::INT16:
			potential_overflow =
			    PROPAGATE::template Operation<int16_t, OP>(expr.return_type, lstats, rstats, new_min, new_max);
			break;
		case PhysicalType::INT32:
			potential_overflow =
			    PROPAGATE::template Operation<int32_t, OP>(expr.return_type, lstats, rstats, new_min, new_max);
			break;
		case PhysicalType::INT64:
			potential_overflow =
			    PROPAGATE::template Operation<int64_t, OP>(expr.return_type, lstats, rstats, new_min, new_max);
			break;
		default:
			return nullptr;
		}
	}
	if (potential_overflow) {
		new_min = Value(expr.return_type);
		new_max = Value(expr.return_type);
	} else {
		// no potential overflow: replace with non-overflowing operator
		if (input.bind_data) {
			auto bind_data = (DecimalArithmeticBindData *)input.bind_data;
			bind_data->check_overflow = false;
		}
		expr.function.function = GetScalarIntegerFunction<BASEOP>(expr.return_type.InternalType());
	}
	auto stats = make_unique<NumericStatistics>(expr.return_type, std::move(new_min), std::move(new_max),
	                                            StatisticsType::LOCAL_STATS);
	stats->validity_stats = ValidityStatistics::Combine(lstats.validity_stats, rstats.validity_stats);
	return std::move(stats);
}

template <class OP, class OPOVERFLOWCHECK, bool IS_SUBTRACT = false>
unique_ptr<FunctionData> BindDecimalAddSubtract(ClientContext &context, ScalarFunction &bound_function,
                                                vector<unique_ptr<Expression>> &arguments) {

	auto bind_data = make_unique<DecimalArithmeticBindData>();

	// get the max width and scale of the input arguments
	uint8_t max_width = 0, max_scale = 0, max_width_over_scale = 0;
	for (idx_t i = 0; i < arguments.size(); i++) {
		if (arguments[i]->return_type.id() == LogicalTypeId::UNKNOWN) {
			continue;
		}
		uint8_t width, scale;
		auto can_convert = arguments[i]->return_type.GetDecimalProperties(width, scale);
		if (!can_convert) {
			throw InternalException("Could not convert type %s to a decimal.", arguments[i]->return_type.ToString());
		}
		max_width = MaxValue<uint8_t>(width, max_width);
		max_scale = MaxValue<uint8_t>(scale, max_scale);
		max_width_over_scale = MaxValue<uint8_t>(width - scale, max_width_over_scale);
	}
	D_ASSERT(max_width > 0);
	// for addition/subtraction, we add 1 to the width to ensure we don't overflow
	auto required_width = MaxValue<uint8_t>(max_scale + max_width_over_scale, max_width) + 1;
	if (required_width > Decimal::MAX_WIDTH_INT64 && max_width <= Decimal::MAX_WIDTH_INT64) {
		// we don't automatically promote past the hugeint boundary to avoid the large hugeint performance penalty
		bind_data->check_overflow = true;
		required_width = Decimal::MAX_WIDTH_INT64;
	}
	if (required_width > Decimal::MAX_WIDTH_DECIMAL) {
		// target width does not fit in decimal at all: truncate the scale and perform overflow detection
		bind_data->check_overflow = true;
		required_width = Decimal::MAX_WIDTH_DECIMAL;
	}
	// arithmetic between two decimal arguments: check the types of the input arguments
	LogicalType result_type = LogicalType::DECIMAL(required_width, max_scale);
	// we cast all input types to the specified type
	for (idx_t i = 0; i < arguments.size(); i++) {
		// first check if the cast is necessary
		// if the argument has a matching scale and internal type as the output type, no casting is necessary
		auto &argument_type = arguments[i]->return_type;
		uint8_t width, scale;
		argument_type.GetDecimalProperties(width, scale);
		if (scale == DecimalType::GetScale(result_type) && argument_type.InternalType() == result_type.InternalType()) {
			bound_function.arguments[i] = argument_type;
		} else {
			bound_function.arguments[i] = result_type;
		}
	}
	bound_function.return_type = result_type;
	// now select the physical function to execute
	if (bind_data->check_overflow) {
		bound_function.function = GetScalarBinaryFunction<OPOVERFLOWCHECK>(result_type.InternalType());
	} else {
		bound_function.function = GetScalarBinaryFunction<OP>(result_type.InternalType());
	}
	if (result_type.InternalType() != PhysicalType::INT128) {
		if (IS_SUBTRACT) {
			bound_function.statistics =
			    PropagateNumericStats<TryDecimalSubtract, SubtractPropagateStatistics, SubtractOperator>;
		} else {
			bound_function.statistics = PropagateNumericStats<TryDecimalAdd, AddPropagateStatistics, AddOperator>;
		}
	}
	return std::move(bind_data);
}

static void SerializeDecimalArithmetic(FieldWriter &writer, const FunctionData *bind_data_p,
                                       const ScalarFunction &function) {
	D_ASSERT(bind_data_p);
	auto bind_data = (DecimalArithmeticBindData *)bind_data_p;
	writer.WriteField(bind_data->check_overflow);
	writer.WriteSerializable(function.return_type);
	writer.WriteRegularSerializableList(function.arguments);
}

// TODO this is partially duplicated from the bind
template <class OP, class OPOVERFLOWCHECK, bool IS_SUBTRACT = false>
unique_ptr<FunctionData> DeserializeDecimalArithmetic(ClientContext &context, FieldReader &reader,
                                                      ScalarFunction &bound_function) {
	// re-change the function pointers
	auto check_overflow = reader.ReadRequired<bool>();
	auto return_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto arguments = reader.template ReadRequiredSerializableList<LogicalType, LogicalType>();

	if (check_overflow) {
		bound_function.function = GetScalarBinaryFunction<OPOVERFLOWCHECK>(return_type.InternalType());
	} else {
		bound_function.function = GetScalarBinaryFunction<OP>(return_type.InternalType());
	}
	bound_function.statistics = nullptr; // TODO we likely dont want to do stats prop again
	bound_function.return_type = return_type;
	bound_function.arguments = arguments;

	auto bind_data = make_unique<DecimalArithmeticBindData>();
	bind_data->check_overflow = check_overflow;
	return std::move(bind_data);
}

unique_ptr<FunctionData> NopDecimalBind(ClientContext &context, ScalarFunction &bound_function,
                                        vector<unique_ptr<Expression>> &arguments) {
	bound_function.return_type = arguments[0]->return_type;
	bound_function.arguments[0] = arguments[0]->return_type;
	return nullptr;
}

ScalarFunction AddFun::GetFunction(const LogicalType &type) {
	D_ASSERT(type.IsNumeric());
	if (type.id() == LogicalTypeId::DECIMAL) {
		return ScalarFunction("+", {type}, type, ScalarFunction::NopFunction, NopDecimalBind);
	} else {
		return ScalarFunction("+", {type}, type, ScalarFunction::NopFunction);
	}
}

ScalarFunction AddFun::GetFunction(const LogicalType &left_type, const LogicalType &right_type) {
	if (left_type.IsNumeric() && left_type.id() == right_type.id()) {
		if (left_type.id() == LogicalTypeId::DECIMAL) {
			auto function = ScalarFunction("+", {left_type, right_type}, left_type, nullptr,
			                               BindDecimalAddSubtract<AddOperator, DecimalAddOverflowCheck>);
			function.serialize = SerializeDecimalArithmetic;
			function.deserialize = DeserializeDecimalArithmetic<AddOperator, DecimalAddOverflowCheck>;
			return function;
		} else if (left_type.IsIntegral() && left_type.id() != LogicalTypeId::HUGEINT) {
			return ScalarFunction("+", {left_type, right_type}, left_type,
			                      GetScalarIntegerFunction<AddOperatorOverflowCheck>(left_type.InternalType()), nullptr,
			                      nullptr, PropagateNumericStats<TryAddOperator, AddPropagateStatistics, AddOperator>);
		} else {
			return ScalarFunction("+", {left_type, right_type}, left_type,
			                      GetScalarBinaryFunction<AddOperator>(left_type.InternalType()));
		}
	}

	switch (left_type.id()) {
	case LogicalTypeId::DATE:
		if (right_type.id() == LogicalTypeId::INTEGER) {
			return ScalarFunction("+", {left_type, right_type}, LogicalType::DATE,
			                      ScalarFunction::BinaryFunction<date_t, int32_t, date_t, AddOperator>);
		} else if (right_type.id() == LogicalTypeId::INTERVAL) {
			return ScalarFunction("+", {left_type, right_type}, LogicalType::DATE,
			                      ScalarFunction::BinaryFunction<date_t, interval_t, date_t, AddOperator>);
		} else if (right_type.id() == LogicalTypeId::TIME) {
			return ScalarFunction("+", {left_type, right_type}, LogicalType::TIMESTAMP,
			                      ScalarFunction::BinaryFunction<date_t, dtime_t, timestamp_t, AddOperator>);
		}
		break;
	case LogicalTypeId::INTEGER:
		if (right_type.id() == LogicalTypeId::DATE) {
			return ScalarFunction("+", {left_type, right_type}, right_type,
			                      ScalarFunction::BinaryFunction<int32_t, date_t, date_t, AddOperator>);
		}
		break;
	case LogicalTypeId::INTERVAL:
		if (right_type.id() == LogicalTypeId::INTERVAL) {
			return ScalarFunction("+", {left_type, right_type}, LogicalType::INTERVAL,
			                      ScalarFunction::BinaryFunction<interval_t, interval_t, interval_t, AddOperator>);
		} else if (right_type.id() == LogicalTypeId::DATE) {
			return ScalarFunction("+", {left_type, right_type}, LogicalType::DATE,
			                      ScalarFunction::BinaryFunction<interval_t, date_t, date_t, AddOperator>);
		} else if (right_type.id() == LogicalTypeId::TIME) {
			return ScalarFunction("+", {left_type, right_type}, LogicalType::TIME,
			                      ScalarFunction::BinaryFunction<interval_t, dtime_t, dtime_t, AddTimeOperator>);
		} else if (right_type.id() == LogicalTypeId::TIMESTAMP) {
			return ScalarFunction("+", {left_type, right_type}, LogicalType::TIMESTAMP,
			                      ScalarFunction::BinaryFunction<interval_t, timestamp_t, timestamp_t, AddOperator>);
		}
		break;
	case LogicalTypeId::TIME:
		if (right_type.id() == LogicalTypeId::INTERVAL) {
			return ScalarFunction("+", {left_type, right_type}, LogicalType::TIME,
			                      ScalarFunction::BinaryFunction<dtime_t, interval_t, dtime_t, AddTimeOperator>);
		} else if (right_type.id() == LogicalTypeId::DATE) {
			return ScalarFunction("+", {left_type, right_type}, LogicalType::TIMESTAMP,
			                      ScalarFunction::BinaryFunction<dtime_t, date_t, timestamp_t, AddOperator>);
		}
		break;
	case LogicalTypeId::TIMESTAMP:
		if (right_type.id() == LogicalTypeId::INTERVAL) {
			return ScalarFunction("+", {left_type, right_type}, LogicalType::TIMESTAMP,
			                      ScalarFunction::BinaryFunction<timestamp_t, interval_t, timestamp_t, AddOperator>);
		}
		break;
	default:
		break;
	}
	// LCOV_EXCL_START
	throw NotImplementedException("AddFun for types %s, %s", LogicalTypeIdToString(left_type.id()),
	                              LogicalTypeIdToString(right_type.id()));
	// LCOV_EXCL_STOP
}

void AddFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet functions("+");
	for (auto &type : LogicalType::Numeric()) {
		// unary add function is a nop, but only exists for numeric types
		functions.AddFunction(GetFunction(type));
		// binary add function adds two numbers together
		functions.AddFunction(GetFunction(type, type));
	}
	// we can add integers to dates
	functions.AddFunction(GetFunction(LogicalType::DATE, LogicalType::INTEGER));
	functions.AddFunction(GetFunction(LogicalType::INTEGER, LogicalType::DATE));
	// we can add intervals together
	functions.AddFunction(GetFunction(LogicalType::INTERVAL, LogicalType::INTERVAL));
	// we can add intervals to dates/times/timestamps
	functions.AddFunction(GetFunction(LogicalType::DATE, LogicalType::INTERVAL));
	functions.AddFunction(GetFunction(LogicalType::INTERVAL, LogicalType::DATE));

	functions.AddFunction(GetFunction(LogicalType::TIME, LogicalType::INTERVAL));
	functions.AddFunction(GetFunction(LogicalType::INTERVAL, LogicalType::TIME));

	functions.AddFunction(GetFunction(LogicalType::TIMESTAMP, LogicalType::INTERVAL));
	functions.AddFunction(GetFunction(LogicalType::INTERVAL, LogicalType::TIMESTAMP));

	// we can add times to dates
	functions.AddFunction(GetFunction(LogicalType::TIME, LogicalType::DATE));
	functions.AddFunction(GetFunction(LogicalType::DATE, LogicalType::TIME));

	// we can add lists together
	functions.AddFunction(ListConcatFun::GetFunction());

	set.AddFunction(functions);

	functions.name = "add";
	set.AddFunction(functions);
}

//===--------------------------------------------------------------------===//
// - [subtract]
//===--------------------------------------------------------------------===//
struct NegateOperator {
	template <class T>
	static bool CanNegate(T input) {
		using Limits = std::numeric_limits<T>;
		return !(Limits::is_integer && Limits::is_signed && Limits::lowest() == input);
	}

	template <class TA, class TR>
	static inline TR Operation(TA input) {
		auto cast = (TR)input;
		if (!CanNegate<TR>(cast)) {
			throw OutOfRangeException("Overflow in negation of integer!");
		}
		return -cast;
	}
};

template <>
bool NegateOperator::CanNegate(float input) {
	return Value::FloatIsFinite(input);
}

template <>
bool NegateOperator::CanNegate(double input) {
	return Value::DoubleIsFinite(input);
}

template <>
interval_t NegateOperator::Operation(interval_t input) {
	interval_t result;
	result.months = NegateOperator::Operation<int32_t, int32_t>(input.months);
	result.days = NegateOperator::Operation<int32_t, int32_t>(input.days);
	result.micros = NegateOperator::Operation<int64_t, int64_t>(input.micros);
	return result;
}

struct DecimalNegateBindData : public FunctionData {
	DecimalNegateBindData() : bound_type(LogicalTypeId::INVALID) {
	}

	unique_ptr<FunctionData> Copy() const override {
		auto res = make_unique<DecimalNegateBindData>();
		res->bound_type = bound_type;
		return std::move(res);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto other = (DecimalNegateBindData &)other_p;
		return other.bound_type == bound_type;
	}

	LogicalTypeId bound_type;
};

unique_ptr<FunctionData> DecimalNegateBind(ClientContext &context, ScalarFunction &bound_function,
                                           vector<unique_ptr<Expression>> &arguments) {

	auto bind_data = make_unique<DecimalNegateBindData>();

	auto &decimal_type = arguments[0]->return_type;
	auto width = DecimalType::GetWidth(decimal_type);
	if (width <= Decimal::MAX_WIDTH_INT16) {
		bound_function.function = ScalarFunction::GetScalarUnaryFunction<NegateOperator>(LogicalTypeId::SMALLINT);
	} else if (width <= Decimal::MAX_WIDTH_INT32) {
		bound_function.function = ScalarFunction::GetScalarUnaryFunction<NegateOperator>(LogicalTypeId::INTEGER);
	} else if (width <= Decimal::MAX_WIDTH_INT64) {
		bound_function.function = ScalarFunction::GetScalarUnaryFunction<NegateOperator>(LogicalTypeId::BIGINT);
	} else {
		D_ASSERT(width <= Decimal::MAX_WIDTH_INT128);
		bound_function.function = ScalarFunction::GetScalarUnaryFunction<NegateOperator>(LogicalTypeId::HUGEINT);
	}
	decimal_type.Verify();
	bound_function.arguments[0] = decimal_type;
	bound_function.return_type = decimal_type;
	return nullptr;
}

struct NegatePropagateStatistics {
	template <class T>
	static bool Operation(LogicalType type, NumericStatistics &istats, Value &new_min, Value &new_max) {
		auto max_value = istats.max.GetValueUnsafe<T>();
		auto min_value = istats.min.GetValueUnsafe<T>();
		if (!NegateOperator::CanNegate<T>(min_value) || !NegateOperator::CanNegate<T>(max_value)) {
			return true;
		}
		// new min is -max
		new_min = Value::Numeric(type, NegateOperator::Operation<T, T>(max_value));
		// new max is -min
		new_max = Value::Numeric(type, NegateOperator::Operation<T, T>(min_value));
		return false;
	}
};

static unique_ptr<BaseStatistics> NegateBindStatistics(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &expr = input.expr;
	D_ASSERT(child_stats.size() == 1);
	// can only propagate stats if the children have stats
	if (!child_stats[0]) {
		return nullptr;
	}
	auto &istats = (NumericStatistics &)*child_stats[0];
	Value new_min, new_max;
	bool potential_overflow = true;
	if (!istats.min.IsNull() && !istats.max.IsNull()) {
		switch (expr.return_type.InternalType()) {
		case PhysicalType::INT8:
			potential_overflow =
			    NegatePropagateStatistics::Operation<int8_t>(expr.return_type, istats, new_min, new_max);
			break;
		case PhysicalType::INT16:
			potential_overflow =
			    NegatePropagateStatistics::Operation<int16_t>(expr.return_type, istats, new_min, new_max);
			break;
		case PhysicalType::INT32:
			potential_overflow =
			    NegatePropagateStatistics::Operation<int32_t>(expr.return_type, istats, new_min, new_max);
			break;
		case PhysicalType::INT64:
			potential_overflow =
			    NegatePropagateStatistics::Operation<int64_t>(expr.return_type, istats, new_min, new_max);
			break;
		default:
			return nullptr;
		}
	}
	if (potential_overflow) {
		new_min = Value(expr.return_type);
		new_max = Value(expr.return_type);
	}
	auto stats = make_unique<NumericStatistics>(expr.return_type, std::move(new_min), std::move(new_max),
	                                            StatisticsType::LOCAL_STATS);
	if (istats.validity_stats) {
		stats->validity_stats = istats.validity_stats->Copy();
	}
	return std::move(stats);
}

ScalarFunction SubtractFun::GetFunction(const LogicalType &type) {
	if (type.id() == LogicalTypeId::INTERVAL) {
		return ScalarFunction("-", {type}, type, ScalarFunction::UnaryFunction<interval_t, interval_t, NegateOperator>);
	} else if (type.id() == LogicalTypeId::DECIMAL) {
		return ScalarFunction("-", {type}, type, nullptr, DecimalNegateBind, nullptr, NegateBindStatistics);
	} else {
		D_ASSERT(type.IsNumeric());
		return ScalarFunction("-", {type}, type, ScalarFunction::GetScalarUnaryFunction<NegateOperator>(type), nullptr,
		                      nullptr, NegateBindStatistics);
	}
}

ScalarFunction SubtractFun::GetFunction(const LogicalType &left_type, const LogicalType &right_type) {
	if (left_type.IsNumeric() && left_type.id() == right_type.id()) {
		if (left_type.id() == LogicalTypeId::DECIMAL) {
			auto function =
			    ScalarFunction("-", {left_type, right_type}, left_type, nullptr,
			                   BindDecimalAddSubtract<SubtractOperator, DecimalSubtractOverflowCheck, true>);
			function.serialize = SerializeDecimalArithmetic;
			function.deserialize = DeserializeDecimalArithmetic<SubtractOperator, DecimalSubtractOverflowCheck>;
			return function;
		} else if (left_type.IsIntegral() && left_type.id() != LogicalTypeId::HUGEINT) {
			return ScalarFunction(
			    "-", {left_type, right_type}, left_type,
			    GetScalarIntegerFunction<SubtractOperatorOverflowCheck>(left_type.InternalType()), nullptr, nullptr,
			    PropagateNumericStats<TrySubtractOperator, SubtractPropagateStatistics, SubtractOperator>);

		} else {
			return ScalarFunction("-", {left_type, right_type}, left_type,
			                      GetScalarBinaryFunction<SubtractOperator>(left_type.InternalType()));
		}
	}

	switch (left_type.id()) {
	case LogicalTypeId::DATE:
		if (right_type.id() == LogicalTypeId::DATE) {
			return ScalarFunction("-", {left_type, right_type}, LogicalType::BIGINT,
			                      ScalarFunction::BinaryFunction<date_t, date_t, int64_t, SubtractOperator>);

		} else if (right_type.id() == LogicalTypeId::INTEGER) {
			return ScalarFunction("-", {left_type, right_type}, LogicalType::DATE,
			                      ScalarFunction::BinaryFunction<date_t, int32_t, date_t, SubtractOperator>);
		} else if (right_type.id() == LogicalTypeId::INTERVAL) {
			return ScalarFunction("-", {left_type, right_type}, LogicalType::DATE,
			                      ScalarFunction::BinaryFunction<date_t, interval_t, date_t, SubtractOperator>);
		}
		break;
	case LogicalTypeId::TIMESTAMP:
		if (right_type.id() == LogicalTypeId::TIMESTAMP) {
			return ScalarFunction(
			    "-", {left_type, right_type}, LogicalType::INTERVAL,
			    ScalarFunction::BinaryFunction<timestamp_t, timestamp_t, interval_t, SubtractOperator>);
		} else if (right_type.id() == LogicalTypeId::INTERVAL) {
			return ScalarFunction(
			    "-", {left_type, right_type}, LogicalType::TIMESTAMP,
			    ScalarFunction::BinaryFunction<timestamp_t, interval_t, timestamp_t, SubtractOperator>);
		}
		break;
	case LogicalTypeId::INTERVAL:
		if (right_type.id() == LogicalTypeId::INTERVAL) {
			return ScalarFunction("-", {left_type, right_type}, LogicalType::INTERVAL,
			                      ScalarFunction::BinaryFunction<interval_t, interval_t, interval_t, SubtractOperator>);
		}
		break;
	case LogicalTypeId::TIME:
		if (right_type.id() == LogicalTypeId::INTERVAL) {
			return ScalarFunction("-", {left_type, right_type}, LogicalType::TIME,
			                      ScalarFunction::BinaryFunction<dtime_t, interval_t, dtime_t, SubtractTimeOperator>);
		}
		break;
	default:
		break;
	}
	// LCOV_EXCL_START
	throw NotImplementedException("SubtractFun for types %s, %s", LogicalTypeIdToString(left_type.id()),
	                              LogicalTypeIdToString(right_type.id()));
	// LCOV_EXCL_STOP
}

void SubtractFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet functions("-");
	for (auto &type : LogicalType::Numeric()) {
		// unary subtract function, negates the input (i.e. multiplies by -1)
		functions.AddFunction(GetFunction(type));
		// binary subtract function "a - b", subtracts b from a
		functions.AddFunction(GetFunction(type, type));
	}
	// we can subtract dates from each other
	functions.AddFunction(GetFunction(LogicalType::DATE, LogicalType::DATE));
	// we can subtract integers from dates
	functions.AddFunction(GetFunction(LogicalType::DATE, LogicalType::INTEGER));
	// we can subtract timestamps from each other
	functions.AddFunction(GetFunction(LogicalType::TIMESTAMP, LogicalType::TIMESTAMP));
	// we can subtract intervals from each other
	functions.AddFunction(GetFunction(LogicalType::INTERVAL, LogicalType::INTERVAL));
	// we can subtract intervals from dates/times/timestamps, but not the other way around
	functions.AddFunction(GetFunction(LogicalType::DATE, LogicalType::INTERVAL));
	functions.AddFunction(GetFunction(LogicalType::TIME, LogicalType::INTERVAL));
	functions.AddFunction(GetFunction(LogicalType::TIMESTAMP, LogicalType::INTERVAL));
	// we can negate intervals
	functions.AddFunction(GetFunction(LogicalType::INTERVAL));
	set.AddFunction(functions);

	functions.name = "subtract";
	set.AddFunction(functions);
}

//===--------------------------------------------------------------------===//
// * [multiply]
//===--------------------------------------------------------------------===//
struct MultiplyPropagateStatistics {
	template <class T, class OP>
	static bool Operation(LogicalType type, NumericStatistics &lstats, NumericStatistics &rstats, Value &new_min,
	                      Value &new_max) {
		// statistics propagation on the multiplication is slightly less straightforward because of negative numbers
		// the new min/max depend on the signs of the input types
		// if both are positive the result is [lmin * rmin][lmax * rmax]
		// if lmin/lmax are negative the result is [lmin * rmax][lmax * rmin]
		// etc
		// rather than doing all this switcheroo we just multiply all combinations of lmin/lmax with rmin/rmax
		// and check what the minimum/maximum value is
		T lvals[] {lstats.min.GetValueUnsafe<T>(), lstats.max.GetValueUnsafe<T>()};
		T rvals[] {rstats.min.GetValueUnsafe<T>(), rstats.max.GetValueUnsafe<T>()};
		T min = NumericLimits<T>::Maximum();
		T max = NumericLimits<T>::Minimum();
		// multiplications
		for (idx_t l = 0; l < 2; l++) {
			for (idx_t r = 0; r < 2; r++) {
				T result;
				if (!OP::Operation(lvals[l], rvals[r], result)) {
					// potential overflow
					return true;
				}
				if (result < min) {
					min = result;
				}
				if (result > max) {
					max = result;
				}
			}
		}
		new_min = Value::Numeric(type, min);
		new_max = Value::Numeric(type, max);
		return false;
	}
};

unique_ptr<FunctionData> BindDecimalMultiply(ClientContext &context, ScalarFunction &bound_function,
                                             vector<unique_ptr<Expression>> &arguments) {

	auto bind_data = make_unique<DecimalArithmeticBindData>();

	uint8_t result_width = 0, result_scale = 0;
	uint8_t max_width = 0;
	for (idx_t i = 0; i < arguments.size(); i++) {
		if (arguments[i]->return_type.id() == LogicalTypeId::UNKNOWN) {
			continue;
		}
		uint8_t width, scale;
		auto can_convert = arguments[i]->return_type.GetDecimalProperties(width, scale);
		if (!can_convert) {
			throw InternalException("Could not convert type %s to a decimal?", arguments[i]->return_type.ToString());
		}
		if (width > max_width) {
			max_width = width;
		}
		result_width += width;
		result_scale += scale;
	}
	D_ASSERT(max_width > 0);
	if (result_scale > Decimal::MAX_WIDTH_DECIMAL) {
		throw OutOfRangeException(
		    "Needed scale %d to accurately represent the multiplication result, but this is out of range of the "
		    "DECIMAL type. Max scale is %d; could not perform an accurate multiplication. Either add a cast to DOUBLE, "
		    "or add an explicit cast to a decimal with a lower scale.",
		    result_scale, Decimal::MAX_WIDTH_DECIMAL);
	}
	if (result_width > Decimal::MAX_WIDTH_INT64 && max_width <= Decimal::MAX_WIDTH_INT64 &&
	    result_scale < Decimal::MAX_WIDTH_INT64) {
		bind_data->check_overflow = true;
		result_width = Decimal::MAX_WIDTH_INT64;
	}
	if (result_width > Decimal::MAX_WIDTH_DECIMAL) {
		bind_data->check_overflow = true;
		result_width = Decimal::MAX_WIDTH_DECIMAL;
	}
	LogicalType result_type = LogicalType::DECIMAL(result_width, result_scale);
	// since our scale is the summation of our input scales, we do not need to cast to the result scale
	// however, we might need to cast to the correct internal type
	for (idx_t i = 0; i < arguments.size(); i++) {
		auto &argument_type = arguments[i]->return_type;
		if (argument_type.InternalType() == result_type.InternalType()) {
			bound_function.arguments[i] = argument_type;
		} else {
			uint8_t width, scale;
			if (!argument_type.GetDecimalProperties(width, scale)) {
				scale = 0;
			}

			bound_function.arguments[i] = LogicalType::DECIMAL(result_width, scale);
		}
	}
	result_type.Verify();
	bound_function.return_type = result_type;
	// now select the physical function to execute
	if (bind_data->check_overflow) {
		bound_function.function = GetScalarBinaryFunction<DecimalMultiplyOverflowCheck>(result_type.InternalType());
	} else {
		bound_function.function = GetScalarBinaryFunction<MultiplyOperator>(result_type.InternalType());
	}
	if (result_type.InternalType() != PhysicalType::INT128) {
		bound_function.statistics =
		    PropagateNumericStats<TryDecimalMultiply, MultiplyPropagateStatistics, MultiplyOperator>;
	}
	return std::move(bind_data);
}

void MultiplyFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet functions("*");
	for (auto &type : LogicalType::Numeric()) {
		if (type.id() == LogicalTypeId::DECIMAL) {
			ScalarFunction function({type, type}, type, nullptr, BindDecimalMultiply);
			function.serialize = SerializeDecimalArithmetic;
			function.deserialize = DeserializeDecimalArithmetic<MultiplyOperator, DecimalMultiplyOverflowCheck>;
			functions.AddFunction(function);
		} else if (TypeIsIntegral(type.InternalType()) && type.id() != LogicalTypeId::HUGEINT) {
			functions.AddFunction(ScalarFunction(
			    {type, type}, type, GetScalarIntegerFunction<MultiplyOperatorOverflowCheck>(type.InternalType()),
			    nullptr, nullptr,
			    PropagateNumericStats<TryMultiplyOperator, MultiplyPropagateStatistics, MultiplyOperator>));
		} else {
			functions.AddFunction(
			    ScalarFunction({type, type}, type, GetScalarBinaryFunction<MultiplyOperator>(type.InternalType())));
		}
	}
	functions.AddFunction(
	    ScalarFunction({LogicalType::INTERVAL, LogicalType::BIGINT}, LogicalType::INTERVAL,
	                   ScalarFunction::BinaryFunction<interval_t, int64_t, interval_t, MultiplyOperator>));
	functions.AddFunction(
	    ScalarFunction({LogicalType::BIGINT, LogicalType::INTERVAL}, LogicalType::INTERVAL,
	                   ScalarFunction::BinaryFunction<int64_t, interval_t, interval_t, MultiplyOperator>));
	set.AddFunction(functions);

	functions.name = "multiply";
	set.AddFunction(functions);
}

//===--------------------------------------------------------------------===//
// / [divide]
//===--------------------------------------------------------------------===//
template <>
float DivideOperator::Operation(float left, float right) {
	auto result = left / right;
	if (!Value::FloatIsFinite(result)) {
		throw OutOfRangeException("Overflow in division of float!");
	}
	return result;
}

template <>
double DivideOperator::Operation(double left, double right) {
	auto result = left / right;
	if (!Value::DoubleIsFinite(result)) {
		throw OutOfRangeException("Overflow in division of double!");
	}
	return result;
}

template <>
hugeint_t DivideOperator::Operation(hugeint_t left, hugeint_t right) {
	if (right.lower == 0 && right.upper == 0) {
		throw InternalException("Hugeint division by zero!");
	}
	return left / right;
}

template <>
interval_t DivideOperator::Operation(interval_t left, int64_t right) {
	left.days /= right;
	left.months /= right;
	left.micros /= right;
	return left;
}

struct BinaryNumericDivideWrapper {
	template <class FUNC, class OP, class LEFT_TYPE, class RIGHT_TYPE, class RESULT_TYPE>
	static inline RESULT_TYPE Operation(FUNC fun, LEFT_TYPE left, RIGHT_TYPE right, ValidityMask &mask, idx_t idx) {
		if (left == NumericLimits<LEFT_TYPE>::Minimum() && right == -1) {
			throw OutOfRangeException("Overflow in division of %d / %d", left, right);
		} else if (right == 0) {
			mask.SetInvalid(idx);
			return left;
		} else {
			return OP::template Operation<LEFT_TYPE, RIGHT_TYPE, RESULT_TYPE>(left, right);
		}
	}

	static bool AddsNulls() {
		return true;
	}
};

struct BinaryZeroIsNullWrapper {
	template <class FUNC, class OP, class LEFT_TYPE, class RIGHT_TYPE, class RESULT_TYPE>
	static inline RESULT_TYPE Operation(FUNC fun, LEFT_TYPE left, RIGHT_TYPE right, ValidityMask &mask, idx_t idx) {
		if (right == 0) {
			mask.SetInvalid(idx);
			return left;
		} else {
			return OP::template Operation<LEFT_TYPE, RIGHT_TYPE, RESULT_TYPE>(left, right);
		}
	}

	static bool AddsNulls() {
		return true;
	}
};

struct BinaryZeroIsNullHugeintWrapper {
	template <class FUNC, class OP, class LEFT_TYPE, class RIGHT_TYPE, class RESULT_TYPE>
	static inline RESULT_TYPE Operation(FUNC fun, LEFT_TYPE left, RIGHT_TYPE right, ValidityMask &mask, idx_t idx) {
		if (right.upper == 0 && right.lower == 0) {
			mask.SetInvalid(idx);
			return left;
		} else {
			return OP::template Operation<LEFT_TYPE, RIGHT_TYPE, RESULT_TYPE>(left, right);
		}
	}

	static bool AddsNulls() {
		return true;
	}
};

template <class TA, class TB, class TC, class OP, class ZWRAPPER = BinaryZeroIsNullWrapper>
static void BinaryScalarFunctionIgnoreZero(DataChunk &input, ExpressionState &state, Vector &result) {
	BinaryExecutor::Execute<TA, TB, TC, OP, ZWRAPPER>(input.data[0], input.data[1], result, input.size());
}

template <class OP>
static scalar_function_t GetBinaryFunctionIgnoreZero(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
		return BinaryScalarFunctionIgnoreZero<int8_t, int8_t, int8_t, OP, BinaryNumericDivideWrapper>;
	case LogicalTypeId::SMALLINT:
		return BinaryScalarFunctionIgnoreZero<int16_t, int16_t, int16_t, OP, BinaryNumericDivideWrapper>;
	case LogicalTypeId::INTEGER:
		return BinaryScalarFunctionIgnoreZero<int32_t, int32_t, int32_t, OP, BinaryNumericDivideWrapper>;
	case LogicalTypeId::BIGINT:
		return BinaryScalarFunctionIgnoreZero<int64_t, int64_t, int64_t, OP, BinaryNumericDivideWrapper>;
	case LogicalTypeId::UTINYINT:
		return BinaryScalarFunctionIgnoreZero<uint8_t, uint8_t, uint8_t, OP>;
	case LogicalTypeId::USMALLINT:
		return BinaryScalarFunctionIgnoreZero<uint16_t, uint16_t, uint16_t, OP>;
	case LogicalTypeId::UINTEGER:
		return BinaryScalarFunctionIgnoreZero<uint32_t, uint32_t, uint32_t, OP>;
	case LogicalTypeId::UBIGINT:
		return BinaryScalarFunctionIgnoreZero<uint64_t, uint64_t, uint64_t, OP>;
	case LogicalTypeId::HUGEINT:
		return BinaryScalarFunctionIgnoreZero<hugeint_t, hugeint_t, hugeint_t, OP, BinaryZeroIsNullHugeintWrapper>;
	case LogicalTypeId::FLOAT:
		return BinaryScalarFunctionIgnoreZero<float, float, float, OP>;
	case LogicalTypeId::DOUBLE:
		return BinaryScalarFunctionIgnoreZero<double, double, double, OP>;
	default:
		throw NotImplementedException("Unimplemented type for GetScalarUnaryFunction");
	}
}

void DivideFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet functions("/");
	for (auto &type : LogicalType::Numeric()) {
		if (type.id() == LogicalTypeId::DECIMAL) {
			continue;
		} else {
			functions.AddFunction(
			    ScalarFunction({type, type}, type, GetBinaryFunctionIgnoreZero<DivideOperator>(type)));
		}
	}
	functions.AddFunction(
	    ScalarFunction({LogicalType::INTERVAL, LogicalType::BIGINT}, LogicalType::INTERVAL,
	                   BinaryScalarFunctionIgnoreZero<interval_t, int64_t, interval_t, DivideOperator>));

	set.AddFunction(functions);

	functions.name = "divide";
	set.AddFunction(functions);
}

//===--------------------------------------------------------------------===//
// % [modulo]
//===--------------------------------------------------------------------===//
template <>
float ModuloOperator::Operation(float left, float right) {
	D_ASSERT(right != 0);
	auto result = std::fmod(left, right);
	if (!Value::FloatIsFinite(result)) {
		throw OutOfRangeException("Overflow in modulo of float!");
	}
	return result;
}

template <>
double ModuloOperator::Operation(double left, double right) {
	D_ASSERT(right != 0);
	auto result = std::fmod(left, right);
	if (!Value::DoubleIsFinite(result)) {
		throw OutOfRangeException("Overflow in modulo of double!");
	}
	return result;
}

template <>
hugeint_t ModuloOperator::Operation(hugeint_t left, hugeint_t right) {
	if (right.lower == 0 && right.upper == 0) {
		throw InternalException("Hugeint division by zero!");
	}
	return left % right;
}

void ModFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet functions("%");
	for (auto &type : LogicalType::Numeric()) {
		if (type.id() == LogicalTypeId::DECIMAL) {
			continue;
		} else {
			functions.AddFunction(
			    ScalarFunction({type, type}, type, GetBinaryFunctionIgnoreZero<ModuloOperator>(type)));
		}
	}
	set.AddFunction(functions);
	functions.name = "mod";
	set.AddFunction(functions);
}

} // namespace duckdb






namespace duckdb {

template <class OP>
static scalar_function_t GetScalarIntegerUnaryFunction(const LogicalType &type) {
	scalar_function_t function;
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
		function = &ScalarFunction::UnaryFunction<int8_t, int8_t, OP>;
		break;
	case LogicalTypeId::SMALLINT:
		function = &ScalarFunction::UnaryFunction<int16_t, int16_t, OP>;
		break;
	case LogicalTypeId::INTEGER:
		function = &ScalarFunction::UnaryFunction<int32_t, int32_t, OP>;
		break;
	case LogicalTypeId::BIGINT:
		function = &ScalarFunction::UnaryFunction<int64_t, int64_t, OP>;
		break;
	case LogicalTypeId::UTINYINT:
		function = &ScalarFunction::UnaryFunction<uint8_t, uint8_t, OP>;
		break;
	case LogicalTypeId::USMALLINT:
		function = &ScalarFunction::UnaryFunction<uint16_t, uint16_t, OP>;
		break;
	case LogicalTypeId::UINTEGER:
		function = &ScalarFunction::UnaryFunction<uint32_t, uint32_t, OP>;
		break;
	case LogicalTypeId::UBIGINT:
		function = &ScalarFunction::UnaryFunction<uint64_t, uint64_t, OP>;
		break;
	case LogicalTypeId::HUGEINT:
		function = &ScalarFunction::UnaryFunction<hugeint_t, hugeint_t, OP>;
		break;
	default:
		throw NotImplementedException("Unimplemented type for GetScalarIntegerUnaryFunction");
	}
	return function;
}

template <class OP>
static scalar_function_t GetScalarIntegerBinaryFunction(const LogicalType &type) {
	scalar_function_t function;
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
		function = &ScalarFunction::BinaryFunction<int8_t, int8_t, int8_t, OP>;
		break;
	case LogicalTypeId::SMALLINT:
		function = &ScalarFunction::BinaryFunction<int16_t, int16_t, int16_t, OP>;
		break;
	case LogicalTypeId::INTEGER:
		function = &ScalarFunction::BinaryFunction<int32_t, int32_t, int32_t, OP>;
		break;
	case LogicalTypeId::BIGINT:
		function = &ScalarFunction::BinaryFunction<int64_t, int64_t, int64_t, OP>;
		break;
	case LogicalTypeId::UTINYINT:
		function = &ScalarFunction::BinaryFunction<uint8_t, uint8_t, uint8_t, OP>;
		break;
	case LogicalTypeId::USMALLINT:
		function = &ScalarFunction::BinaryFunction<uint16_t, uint16_t, uint16_t, OP>;
		break;
	case LogicalTypeId::UINTEGER:
		function = &ScalarFunction::BinaryFunction<uint32_t, uint32_t, uint32_t, OP>;
		break;
	case LogicalTypeId::UBIGINT:
		function = &ScalarFunction::BinaryFunction<uint64_t, uint64_t, uint64_t, OP>;
		break;
	case LogicalTypeId::HUGEINT:
		function = &ScalarFunction::BinaryFunction<hugeint_t, hugeint_t, hugeint_t, OP>;
		break;
	default:
		throw NotImplementedException("Unimplemented type for GetScalarIntegerBinaryFunction");
	}
	return function;
}

//===--------------------------------------------------------------------===//
// & [bitwise_and]
//===--------------------------------------------------------------------===//
struct BitwiseANDOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA left, TB right) {
		return left & right;
	}
};

static void BitwiseANDOperation(DataChunk &args, ExpressionState &state, Vector &result) {
	BinaryExecutor::Execute<string_t, string_t, string_t>(
	    args.data[0], args.data[1], result, args.size(), [&](string_t rhs, string_t lhs) {
		    string_t target = StringVector::EmptyString(result, rhs.GetSize());

		    Bit::BitwiseAnd(rhs, lhs, target);
		    return target;
	    });
}

void BitwiseAndFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet functions("&");
	for (auto &type : LogicalType::Integral()) {
		functions.AddFunction(
		    ScalarFunction({type, type}, type, GetScalarIntegerBinaryFunction<BitwiseANDOperator>(type)));
	}
	functions.AddFunction(ScalarFunction({LogicalType::BIT, LogicalType::BIT}, LogicalType::BIT, BitwiseANDOperation));

	set.AddFunction(functions);
}

//===--------------------------------------------------------------------===//
// | [bitwise_or]
//===--------------------------------------------------------------------===//
struct BitwiseOROperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA left, TB right) {
		return left | right;
	}
};

static void BitwiseOROperation(DataChunk &args, ExpressionState &state, Vector &result) {
	BinaryExecutor::Execute<string_t, string_t, string_t>(
	    args.data[0], args.data[1], result, args.size(), [&](string_t rhs, string_t lhs) {
		    string_t target = StringVector::EmptyString(result, rhs.GetSize());

		    Bit::BitwiseOr(rhs, lhs, target);
		    return target;
	    });
}

void BitwiseOrFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet functions("|");
	for (auto &type : LogicalType::Integral()) {
		functions.AddFunction(
		    ScalarFunction({type, type}, type, GetScalarIntegerBinaryFunction<BitwiseOROperator>(type)));
	}
	functions.AddFunction(ScalarFunction({LogicalType::BIT, LogicalType::BIT}, LogicalType::BIT, BitwiseOROperation));
	set.AddFunction(functions);
}

//===--------------------------------------------------------------------===//
// # [bitwise_xor]
//===--------------------------------------------------------------------===//
struct BitwiseXOROperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA left, TB right) {
		return left ^ right;
	}
};

static void BitwiseXOROperation(DataChunk &args, ExpressionState &state, Vector &result) {
	BinaryExecutor::Execute<string_t, string_t, string_t>(
	    args.data[0], args.data[1], result, args.size(), [&](string_t rhs, string_t lhs) {
		    string_t target = StringVector::EmptyString(result, rhs.GetSize());

		    Bit::BitwiseXor(rhs, lhs, target);
		    return target;
	    });
}

void BitwiseXorFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet functions("xor");
	for (auto &type : LogicalType::Integral()) {
		functions.AddFunction(
		    ScalarFunction({type, type}, type, GetScalarIntegerBinaryFunction<BitwiseXOROperator>(type)));
	}
	functions.AddFunction(ScalarFunction({LogicalType::BIT, LogicalType::BIT}, LogicalType::BIT, BitwiseXOROperation));
	set.AddFunction(functions);
}

//===--------------------------------------------------------------------===//
// ~ [bitwise_not]
//===--------------------------------------------------------------------===//
struct BitwiseNotOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return ~input;
	}
};

static void BitwiseNOTOperation(DataChunk &args, ExpressionState &state, Vector &result) {
	UnaryExecutor::Execute<string_t, string_t>(args.data[0], result, args.size(), [&](string_t input) {
		string_t target = StringVector::EmptyString(result, input.GetSize());

		Bit::BitwiseNot(input, target);
		return target;
	});
}

void BitwiseNotFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet functions("~");
	for (auto &type : LogicalType::Integral()) {
		functions.AddFunction(ScalarFunction({type}, type, GetScalarIntegerUnaryFunction<BitwiseNotOperator>(type)));
	}
	functions.AddFunction(ScalarFunction({LogicalType::BIT}, LogicalType::BIT, BitwiseNOTOperation));
	set.AddFunction(functions);
}

//===--------------------------------------------------------------------===//
// << [bitwise_left_shift]
//===--------------------------------------------------------------------===//

struct BitwiseShiftLeftOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA input, TB shift) {
		TA max_shift = TA(sizeof(TA) * 8);
		if (input < 0) {
			throw OutOfRangeException("Cannot left-shift negative number %s", NumericHelper::ToString(input));
		}
		if (shift < 0) {
			throw OutOfRangeException("Cannot left-shift by negative number %s", NumericHelper::ToString(shift));
		}
		if (shift >= max_shift) {
			if (input == 0) {
				return 0;
			}
			throw OutOfRangeException("Left-shift value %s is out of range", NumericHelper::ToString(shift));
		}
		if (shift == 0) {
			return input;
		}
		TA max_value = (TA(1) << (max_shift - shift - 1));
		if (input >= max_value) {
			throw OutOfRangeException("Overflow in left shift (%s << %s)", NumericHelper::ToString(input),
			                          NumericHelper::ToString(shift));
		}
		return input << shift;
	}
};

static void BitwiseShiftLeftOperation(DataChunk &args, ExpressionState &state, Vector &result) {
	BinaryExecutor::Execute<string_t, int32_t, string_t>(
	    args.data[0], args.data[1], result, args.size(), [&](string_t input, int32_t shift) {
		    int32_t max_shift = Bit::BitLength(input);
		    if (shift == 0) {
			    return input;
		    }
		    if (shift < 0) {
			    throw OutOfRangeException("Cannot left-shift by negative number %s", NumericHelper::ToString(shift));
		    }
		    string_t target = StringVector::EmptyString(result, input.GetSize());

		    if (shift >= max_shift) {
			    Bit::SetEmptyBitString(target, input);
			    return target;
		    }
		    Bit::LeftShift(input, shift, target);
		    return target;
	    });
}

void LeftShiftFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet functions("<<");
	for (auto &type : LogicalType::Integral()) {
		functions.AddFunction(
		    ScalarFunction({type, type}, type, GetScalarIntegerBinaryFunction<BitwiseShiftLeftOperator>(type)));
	}
	functions.AddFunction(
	    ScalarFunction({LogicalType::BIT, LogicalType::INTEGER}, LogicalType::BIT, BitwiseShiftLeftOperation));
	set.AddFunction(functions);
}

//===--------------------------------------------------------------------===//
// >> [bitwise_right_shift]
//===--------------------------------------------------------------------===//
template <class T>
bool RightShiftInRange(T shift) {
	return shift >= 0 && shift < T(sizeof(T) * 8);
}

struct BitwiseShiftRightOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA input, TB shift) {
		return RightShiftInRange(shift) ? input >> shift : 0;
	}
};

static void BitwiseShiftRightOperation(DataChunk &args, ExpressionState &state, Vector &result) {
	BinaryExecutor::Execute<string_t, int32_t, string_t>(
	    args.data[0], args.data[1], result, args.size(), [&](string_t input, int32_t shift) {
		    int32_t max_shift = Bit::BitLength(input);
		    if (shift == 0) {
			    return input;
		    }
		    string_t target = StringVector::EmptyString(result, input.GetSize());
		    if (shift < 0 || shift >= max_shift) {
			    Bit::SetEmptyBitString(target, input);
			    return target;
		    }
		    Bit::RightShift(input, shift, target);
		    return target;
	    });
}

void RightShiftFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet functions(">>");
	for (auto &type : LogicalType::Integral()) {
		functions.AddFunction(
		    ScalarFunction({type, type}, type, GetScalarIntegerBinaryFunction<BitwiseShiftRightOperator>(type)));
	}
	functions.AddFunction(
	    ScalarFunction({LogicalType::BIT, LogicalType::INTEGER}, LogicalType::BIT, BitwiseShiftRightOperation));
	set.AddFunction(functions);
}

//===--------------------------------------------------------------------===//
// get_bit
//===--------------------------------------------------------------------===//
struct GetBitOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA input, TB n) {
		if (n < 0 || (idx_t)n > Bit::BitLength(input) - 1) {
			throw OutOfRangeException("bit index %s out of valid range (0..%s)", NumericHelper::ToString(n),
			                          NumericHelper::ToString(Bit::BitLength(input) - 1));
		}
		return Bit::GetBit(input, n);
	}
};

void GetBitFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("get_bit", {LogicalType::BIT, LogicalType::INTEGER}, LogicalType::INTEGER,
	                               ScalarFunction::BinaryFunction<string_t, int32_t, int32_t, GetBitOperator>));
}

//===--------------------------------------------------------------------===//
// set_bit
//===--------------------------------------------------------------------===//
static void SetBitOperation(DataChunk &args, ExpressionState &state, Vector &result) {
	TernaryExecutor::Execute<string_t, int32_t, int32_t, string_t>(
	    args.data[0], args.data[1], args.data[2], result, args.size(),
	    [&](string_t input, int32_t n, int32_t new_value) {
		    if (new_value != 0 && new_value != 1) {
			    throw InvalidInputException("The new bit must be 1 or 0");
		    }
		    if (n < 0 || (idx_t)n > Bit::BitLength(input) - 1) {
			    throw OutOfRangeException("bit index %s out of valid range (0..%s)", NumericHelper::ToString(n),
			                              NumericHelper::ToString(Bit::BitLength(input) - 1));
		    }
		    string_t target = StringVector::EmptyString(result, input.GetSize());
		    Bit::SetBit(input, n, new_value, target);
		    return target;
	    });
}

void SetBitFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("set_bit", {LogicalType::BIT, LogicalType::INTEGER, LogicalType::INTEGER},
	                               LogicalType::BIT, SetBitOperation));
}

//===--------------------------------------------------------------------===//
// bit_position
//===--------------------------------------------------------------------===//
struct BitPositionOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA substring, TB input) {
		if (substring.GetSize() > input.GetSize()) {
			return 0;
		}
		return Bit::BitPosition(substring, input);
	}
};

void BitPositionFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("bit_position", {LogicalType::BIT, LogicalType::BIT}, LogicalType::INTEGER,
	                               ScalarFunction::BinaryFunction<string_t, string_t, int32_t, BitPositionOperator>));
}

} // namespace duckdb







#include <limits>
#include <algorithm>

namespace duckdb {

//===--------------------------------------------------------------------===//
// * [multiply]
//===--------------------------------------------------------------------===//
template <>
float MultiplyOperator::Operation(float left, float right) {
	auto result = left * right;
	if (!Value::FloatIsFinite(result)) {
		throw OutOfRangeException("Overflow in multiplication of float!");
	}
	return result;
}

template <>
double MultiplyOperator::Operation(double left, double right) {
	auto result = left * right;
	if (!Value::DoubleIsFinite(result)) {
		throw OutOfRangeException("Overflow in multiplication of double!");
	}
	return result;
}

template <>
interval_t MultiplyOperator::Operation(interval_t left, int64_t right) {
	left.months = MultiplyOperatorOverflowCheck::Operation<int32_t, int32_t, int32_t>(left.months, right);
	left.days = MultiplyOperatorOverflowCheck::Operation<int32_t, int32_t, int32_t>(left.days, right);
	left.micros = MultiplyOperatorOverflowCheck::Operation<int64_t, int64_t, int64_t>(left.micros, right);
	return left;
}

template <>
interval_t MultiplyOperator::Operation(int64_t left, interval_t right) {
	return MultiplyOperator::Operation<interval_t, int64_t, interval_t>(right, left);
}

//===--------------------------------------------------------------------===//
// * [multiply] with overflow check
//===--------------------------------------------------------------------===//
struct OverflowCheckedMultiply {
	template <class SRCTYPE, class UTYPE>
	static inline bool Operation(SRCTYPE left, SRCTYPE right, SRCTYPE &result) {
		UTYPE uresult = MultiplyOperator::Operation<UTYPE, UTYPE, UTYPE>(UTYPE(left), UTYPE(right));
		if (uresult < NumericLimits<SRCTYPE>::Minimum() || uresult > NumericLimits<SRCTYPE>::Maximum()) {
			return false;
		}
		result = SRCTYPE(uresult);
		return true;
	}
};

template <>
bool TryMultiplyOperator::Operation(uint8_t left, uint8_t right, uint8_t &result) {
	return OverflowCheckedMultiply::Operation<uint8_t, uint16_t>(left, right, result);
}
template <>
bool TryMultiplyOperator::Operation(uint16_t left, uint16_t right, uint16_t &result) {
	return OverflowCheckedMultiply::Operation<uint16_t, uint32_t>(left, right, result);
}
template <>
bool TryMultiplyOperator::Operation(uint32_t left, uint32_t right, uint32_t &result) {
	return OverflowCheckedMultiply::Operation<uint32_t, uint64_t>(left, right, result);
}
template <>
bool TryMultiplyOperator::Operation(uint64_t left, uint64_t right, uint64_t &result) {
	if (left > right) {
		std::swap(left, right);
	}
	if (left > NumericLimits<uint32_t>::Maximum()) {
		return false;
	}
	uint32_t c = right >> 32;
	uint32_t d = NumericLimits<uint32_t>::Maximum() & right;
	uint64_t r = left * c;
	uint64_t s = left * d;
	if (r > NumericLimits<uint32_t>::Maximum()) {
		return false;
	}
	r <<= 32;
	if (NumericLimits<uint64_t>::Maximum() - s < r) {
		return false;
	}
	return OverflowCheckedMultiply::Operation<uint64_t, uint64_t>(left, right, result);
}

template <>
bool TryMultiplyOperator::Operation(int8_t left, int8_t right, int8_t &result) {
	return OverflowCheckedMultiply::Operation<int8_t, int16_t>(left, right, result);
}

template <>
bool TryMultiplyOperator::Operation(int16_t left, int16_t right, int16_t &result) {
	return OverflowCheckedMultiply::Operation<int16_t, int32_t>(left, right, result);
}

template <>
bool TryMultiplyOperator::Operation(int32_t left, int32_t right, int32_t &result) {
	return OverflowCheckedMultiply::Operation<int32_t, int64_t>(left, right, result);
}

template <>
bool TryMultiplyOperator::Operation(int64_t left, int64_t right, int64_t &result) {
#if (__GNUC__ >= 5) || defined(__clang__)
	if (__builtin_mul_overflow(left, right, &result)) {
		return false;
	}
#else
	if (left == std::numeric_limits<int64_t>::min()) {
		if (right == 0) {
			result = 0;
			return true;
		}
		if (right == 1) {
			result = left;
			return true;
		}
		return false;
	}
	if (right == std::numeric_limits<int64_t>::min()) {
		if (left == 0) {
			result = 0;
			return true;
		}
		if (left == 1) {
			result = right;
			return true;
		}
		return false;
	}
	uint64_t left_non_negative = uint64_t(std::abs(left));
	uint64_t right_non_negative = uint64_t(std::abs(right));
	// split values into 2 32-bit parts
	uint64_t left_high_bits = left_non_negative >> 32;
	uint64_t left_low_bits = left_non_negative & 0xffffffff;
	uint64_t right_high_bits = right_non_negative >> 32;
	uint64_t right_low_bits = right_non_negative & 0xffffffff;

	// check the high bits of both
	// the high bits define the overflow
	if (left_high_bits == 0) {
		if (right_high_bits != 0) {
			// only the right has high bits set
			// multiply the high bits of right with the low bits of left
			// multiply the low bits, and carry any overflow to the high bits
			// then check for any overflow
			auto low_low = left_low_bits * right_low_bits;
			auto low_high = left_low_bits * right_high_bits;
			auto high_bits = low_high + (low_low >> 32);
			if (high_bits & 0xffffff80000000) {
				// there is! abort
				return false;
			}
		}
	} else if (right_high_bits == 0) {
		// only the left has high bits set
		// multiply the high bits of left with the low bits of right
		// multiply the low bits, and carry any overflow to the high bits
		// then check for any overflow
		auto low_low = left_low_bits * right_low_bits;
		auto high_low = left_high_bits * right_low_bits;
		auto high_bits = high_low + (low_low >> 32);
		if (high_bits & 0xffffff80000000) {
			// there is! abort
			return false;
		}
	} else {
		// both left and right have high bits set: guaranteed overflow
		// abort!
		return false;
	}
	// now we know that there is no overflow, we can just perform the multiplication
	result = left * right;
#endif
	return true;
}

//===--------------------------------------------------------------------===//
// multiply  decimal with overflow check
//===--------------------------------------------------------------------===//
template <class T, T min, T max>
bool TryDecimalMultiplyTemplated(T left, T right, T &result) {
	if (!TryMultiplyOperator::Operation(left, right, result) || result < min || result > max) {
		return false;
	}
	return true;
}

template <>
bool TryDecimalMultiply::Operation(int16_t left, int16_t right, int16_t &result) {
	return TryDecimalMultiplyTemplated<int16_t, -9999, 9999>(left, right, result);
}

template <>
bool TryDecimalMultiply::Operation(int32_t left, int32_t right, int32_t &result) {
	return TryDecimalMultiplyTemplated<int32_t, -999999999, 999999999>(left, right, result);
}

template <>
bool TryDecimalMultiply::Operation(int64_t left, int64_t right, int64_t &result) {
	return TryDecimalMultiplyTemplated<int64_t, -999999999999999999, 999999999999999999>(left, right, result);
}

template <>
bool TryDecimalMultiply::Operation(hugeint_t left, hugeint_t right, hugeint_t &result) {
	result = left * right;
	if (result <= -Hugeint::POWERS_OF_TEN[38] || result >= Hugeint::POWERS_OF_TEN[38]) {
		return false;
	}
	return true;
}

template <>
hugeint_t DecimalMultiplyOverflowCheck::Operation(hugeint_t left, hugeint_t right) {
	hugeint_t result;
	if (!TryDecimalMultiply::Operation(left, right, result)) {
		throw OutOfRangeException("Overflow in multiplication of DECIMAL(38) (%s * %s). You might want to add an "
		                          "explicit cast to a decimal with a smaller scale.",
		                          left.ToString(), right.ToString());
	}
	return result;
}

} // namespace duckdb









namespace duckdb {

//===--------------------------------------------------------------------===//
// - [subtract]
//===--------------------------------------------------------------------===//
template <>
float SubtractOperator::Operation(float left, float right) {
	auto result = left - right;
	if (!Value::FloatIsFinite(result)) {
		throw OutOfRangeException("Overflow in subtraction of float!");
	}
	return result;
}

template <>
double SubtractOperator::Operation(double left, double right) {
	auto result = left - right;
	if (!Value::DoubleIsFinite(result)) {
		throw OutOfRangeException("Overflow in subtraction of double!");
	}
	return result;
}

template <>
int64_t SubtractOperator::Operation(date_t left, date_t right) {
	return int64_t(left.days) - int64_t(right.days);
}

template <>
date_t SubtractOperator::Operation(date_t left, int32_t right) {
	if (!Date::IsFinite(left)) {
		return left;
	}
	int32_t days;
	if (!TrySubtractOperator::Operation(left.days, right, days)) {
		throw OutOfRangeException("Date out of range");
	}

	date_t result(days);
	if (!Date::IsFinite(result)) {
		throw OutOfRangeException("Date out of range");
	}
	return result;
}

template <>
interval_t SubtractOperator::Operation(interval_t left, interval_t right) {
	interval_t result;
	result.months = left.months - right.months;
	result.days = left.days - right.days;
	result.micros = left.micros - right.micros;
	return result;
}

template <>
date_t SubtractOperator::Operation(date_t left, interval_t right) {
	return AddOperator::Operation<date_t, interval_t, date_t>(left, Interval::Invert(right));
}

template <>
timestamp_t SubtractOperator::Operation(timestamp_t left, interval_t right) {
	return AddOperator::Operation<timestamp_t, interval_t, timestamp_t>(left, Interval::Invert(right));
}

template <>
interval_t SubtractOperator::Operation(timestamp_t left, timestamp_t right) {
	return Interval::GetDifference(left, right);
}

//===--------------------------------------------------------------------===//
// - [subtract] with overflow check
//===--------------------------------------------------------------------===//
struct OverflowCheckedSubtract {
	template <class SRCTYPE, class UTYPE>
	static inline bool Operation(SRCTYPE left, SRCTYPE right, SRCTYPE &result) {
		UTYPE uresult = SubtractOperator::Operation<UTYPE, UTYPE, UTYPE>(UTYPE(left), UTYPE(right));
		if (uresult < NumericLimits<SRCTYPE>::Minimum() || uresult > NumericLimits<SRCTYPE>::Maximum()) {
			return false;
		}
		result = SRCTYPE(uresult);
		return true;
	}
};

template <>
bool TrySubtractOperator::Operation(uint8_t left, uint8_t right, uint8_t &result) {
	if (right > left) {
		return false;
	}
	return OverflowCheckedSubtract::Operation<uint8_t, uint16_t>(left, right, result);
}

template <>
bool TrySubtractOperator::Operation(uint16_t left, uint16_t right, uint16_t &result) {
	if (right > left) {
		return false;
	}
	return OverflowCheckedSubtract::Operation<uint16_t, uint32_t>(left, right, result);
}

template <>
bool TrySubtractOperator::Operation(uint32_t left, uint32_t right, uint32_t &result) {
	if (right > left) {
		return false;
	}
	return OverflowCheckedSubtract::Operation<uint32_t, uint64_t>(left, right, result);
}

template <>
bool TrySubtractOperator::Operation(uint64_t left, uint64_t right, uint64_t &result) {
	if (right > left) {
		return false;
	}
	return OverflowCheckedSubtract::Operation<uint64_t, uint64_t>(left, right, result);
}

template <>
bool TrySubtractOperator::Operation(int8_t left, int8_t right, int8_t &result) {
	return OverflowCheckedSubtract::Operation<int8_t, int16_t>(left, right, result);
}

template <>
bool TrySubtractOperator::Operation(int16_t left, int16_t right, int16_t &result) {
	return OverflowCheckedSubtract::Operation<int16_t, int32_t>(left, right, result);
}

template <>
bool TrySubtractOperator::Operation(int32_t left, int32_t right, int32_t &result) {
	return OverflowCheckedSubtract::Operation<int32_t, int64_t>(left, right, result);
}

template <>
bool TrySubtractOperator::Operation(int64_t left, int64_t right, int64_t &result) {
#if (__GNUC__ >= 5) || defined(__clang__)
	if (__builtin_sub_overflow(left, right, &result)) {
		return false;
	}
#else
	if (right < 0) {
		if (NumericLimits<int64_t>::Maximum() + right < left) {
			return false;
		}
	} else {
		if (NumericLimits<int64_t>::Minimum() + right > left) {
			return false;
		}
	}
	result = left - right;
#endif
	return true;
}

template <>
bool TrySubtractOperator::Operation(hugeint_t left, hugeint_t right, hugeint_t &result) {
	result = left;
	return Hugeint::SubtractInPlace(result, right);
}

//===--------------------------------------------------------------------===//
// subtract decimal with overflow check
//===--------------------------------------------------------------------===//
template <class T, T min, T max>
bool TryDecimalSubtractTemplated(T left, T right, T &result) {
	if (right < 0) {
		if (max + right < left) {
			return false;
		}
	} else {
		if (min + right > left) {
			return false;
		}
	}
	result = left - right;
	return true;
}

template <>
bool TryDecimalSubtract::Operation(int16_t left, int16_t right, int16_t &result) {
	return TryDecimalSubtractTemplated<int16_t, -9999, 9999>(left, right, result);
}

template <>
bool TryDecimalSubtract::Operation(int32_t left, int32_t right, int32_t &result) {
	return TryDecimalSubtractTemplated<int32_t, -999999999, 999999999>(left, right, result);
}

template <>
bool TryDecimalSubtract::Operation(int64_t left, int64_t right, int64_t &result) {
	return TryDecimalSubtractTemplated<int64_t, -999999999999999999, 999999999999999999>(left, right, result);
}

template <>
bool TryDecimalSubtract::Operation(hugeint_t left, hugeint_t right, hugeint_t &result) {
	result = left - right;
	if (result <= -Hugeint::POWERS_OF_TEN[38] || result >= Hugeint::POWERS_OF_TEN[38]) {
		return false;
	}
	return true;
}

template <>
hugeint_t DecimalSubtractOverflowCheck::Operation(hugeint_t left, hugeint_t right) {
	hugeint_t result;
	if (!TryDecimalSubtract::Operation(left, right, result)) {
		throw OutOfRangeException("Overflow in subtract of DECIMAL(38) (%s - %s);", left.ToString(), right.ToString());
	}
	return result;
}

//===--------------------------------------------------------------------===//
// subtract time operator
//===--------------------------------------------------------------------===//
template <>
dtime_t SubtractTimeOperator::Operation(dtime_t left, interval_t right) {
	right.micros = -right.micros;
	return AddTimeOperator::Operation<dtime_t, interval_t, dtime_t>(left, right);
}

} // namespace duckdb



namespace duckdb {

void BuiltinFunctions::RegisterOperators() {
	Register<AddFun>();
	Register<SubtractFun>();
	Register<MultiplyFun>();
	Register<DivideFun>();
	Register<ModFun>();
	Register<LeftShiftFun>();
	Register<RightShiftFun>();
	Register<BitwiseAndFun>();
	Register<BitwiseOrFun>();
	Register<BitwiseXorFun>();
	Register<BitwiseNotFun>();
}

} // namespace duckdb


namespace duckdb {

void BuiltinFunctions::RegisterPragmaFunctions() {
	Register<PragmaQueries>();
	Register<PragmaFunctions>();
}

} // namespace duckdb














namespace duckdb {

struct NextvalBindData : public FunctionData {
	explicit NextvalBindData(SequenceCatalogEntry *sequence) : sequence(sequence) {
	}

	//! The sequence to use for the nextval computation; only if the sequence is a constant
	SequenceCatalogEntry *sequence;

	unique_ptr<FunctionData> Copy() const override {
		return make_unique<NextvalBindData>(sequence);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (NextvalBindData &)other_p;
		return sequence == other.sequence;
	}
};

struct CurrentSequenceValueOperator {
	static int64_t Operation(DuckTransaction &transaction, SequenceCatalogEntry *seq) {
		lock_guard<mutex> seqlock(seq->lock);
		int64_t result;
		if (seq->usage_count == 0u) {
			throw SequenceException("currval: sequence is not yet defined in this session");
		}
		result = seq->last_value;
		return result;
	}
};

struct NextSequenceValueOperator {
	static int64_t Operation(DuckTransaction &transaction, SequenceCatalogEntry *seq) {
		lock_guard<mutex> seqlock(seq->lock);
		int64_t result;
		result = seq->counter;
		bool overflow = !TryAddOperator::Operation(seq->counter, seq->increment, seq->counter);
		if (seq->cycle) {
			if (overflow) {
				seq->counter = seq->increment < 0 ? seq->max_value : seq->min_value;
			} else if (seq->counter < seq->min_value) {
				seq->counter = seq->max_value;
			} else if (seq->counter > seq->max_value) {
				seq->counter = seq->min_value;
			}
		} else {
			if (result < seq->min_value || (overflow && seq->increment < 0)) {
				throw SequenceException("nextval: reached minimum value of sequence \"%s\" (%lld)", seq->name,
				                        seq->min_value);
			}
			if (result > seq->max_value || overflow) {
				throw SequenceException("nextval: reached maximum value of sequence \"%s\" (%lld)", seq->name,
				                        seq->max_value);
			}
		}
		seq->last_value = result;
		seq->usage_count++;
		if (!seq->temporary) {
			transaction.sequence_usage[seq] = SequenceValue(seq->usage_count, seq->counter);
		}
		return result;
	}
};

SequenceCatalogEntry *BindSequence(ClientContext &context, const string &name) {
	auto qname = QualifiedName::Parse(name);
	// fetch the sequence from the catalog
	Binder::BindSchemaOrCatalog(context, qname.catalog, qname.schema);
	return Catalog::GetEntry<SequenceCatalogEntry>(context, qname.catalog, qname.schema, qname.name);
}

template <class OP>
static void NextValFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = (BoundFunctionExpression &)state.expr;
	auto &info = (NextvalBindData &)*func_expr.bind_info;
	auto &input = args.data[0];

	auto &context = state.GetContext();
	if (info.sequence) {
		auto &transaction = DuckTransaction::Get(context, *info.sequence->catalog);
		// sequence to use is hard coded
		// increment the sequence
		result.SetVectorType(VectorType::FLAT_VECTOR);
		auto result_data = FlatVector::GetData<int64_t>(result);
		for (idx_t i = 0; i < args.size(); i++) {
			// get the next value from the sequence
			result_data[i] = OP::Operation(transaction, info.sequence);
		}
	} else {
		// sequence to use comes from the input
		UnaryExecutor::Execute<string_t, int64_t>(input, result, args.size(), [&](string_t value) {
			// fetch the sequence from the catalog
			auto sequence = BindSequence(context, value.GetString());
			// finally get the next value from the sequence
			auto &transaction = DuckTransaction::Get(context, *sequence->catalog);
			return OP::Operation(transaction, sequence);
		});
	}
}

static unique_ptr<FunctionData> NextValBind(ClientContext &context, ScalarFunction &bound_function,
                                            vector<unique_ptr<Expression>> &arguments) {
	SequenceCatalogEntry *sequence = nullptr;
	if (arguments[0]->IsFoldable()) {
		// parameter to nextval function is a foldable constant
		// evaluate the constant and perform the catalog lookup already
		auto seqname = ExpressionExecutor::EvaluateScalar(context, *arguments[0]);
		if (!seqname.IsNull()) {
			sequence = BindSequence(context, seqname.ToString());
		}
	}
	return make_unique<NextvalBindData>(sequence);
}

static void NextValDependency(BoundFunctionExpression &expr, DependencyList &dependencies) {
	auto &info = (NextvalBindData &)*expr.bind_info;
	if (info.sequence) {
		dependencies.AddDependency(info.sequence);
	}
}

void NextvalFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunction next_val("nextval", {LogicalType::VARCHAR}, LogicalType::BIGINT,
	                        NextValFunction<NextSequenceValueOperator>, NextValBind, NextValDependency);
	next_val.side_effects = FunctionSideEffects::HAS_SIDE_EFFECTS;
	set.AddFunction(next_val);
}

void CurrvalFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunction curr_val("currval", {LogicalType::VARCHAR}, LogicalType::BIGINT,
	                        NextValFunction<CurrentSequenceValueOperator>, NextValBind, NextValDependency);
	curr_val.side_effects = FunctionSideEffects::HAS_SIDE_EFFECTS;
	set.AddFunction(curr_val);
}

} // namespace duckdb


namespace duckdb {

void BuiltinFunctions::RegisterSequenceFunctions() {
	Register<NextvalFun>();
	Register<CurrvalFun>();
}

} // namespace duckdb




namespace duckdb {

struct AsciiOperator {
	template <class TA, class TR>
	static inline TR Operation(const TA &input) {
		auto str = input.GetDataUnsafe();
		if (Utf8Proc::Analyze(str, input.GetSize()) == UnicodeType::ASCII) {
			return str[0];
		}
		int utf8_bytes = 4;
		return Utf8Proc::UTF8ToCodepoint(str, utf8_bytes);
	}
};

void ASCII::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunction ascii("ascii", {LogicalType::VARCHAR}, LogicalType::INTEGER,
	                     ScalarFunction::UnaryFunction<string_t, int32_t, AsciiOperator>);
	set.AddFunction(ascii);
}

} // namespace duckdb









namespace duckdb {

static string_t BarScalarFunction(double x, double min, double max, double max_width, string &result) {
	static const char *FULL_BLOCK = UnicodeBar::FullBlock();
	static const char *const *PARTIAL_BLOCKS = UnicodeBar::PartialBlocks();
	static const idx_t PARTIAL_BLOCKS_COUNT = UnicodeBar::PartialBlocksCount();

	if (!Value::IsFinite(max_width)) {
		throw ValueOutOfRangeException("Max bar width must not be NaN or infinity");
	}
	if (max_width < 1) {
		throw ValueOutOfRangeException("Max bar width must be >= 1");
	}
	if (max_width > 1000) {
		throw ValueOutOfRangeException("Max bar width must be <= 1000");
	}

	double width;

	if (Value::IsNan(x) || Value::IsNan(min) || Value::IsNan(max) || x <= min) {
		width = 0;
	} else if (x >= max) {
		width = max_width;
	} else {
		width = max_width * (x - min) / (max - min);
	}

	if (!Value::IsFinite(width)) {
		throw ValueOutOfRangeException("Bar width must not be NaN or infinity");
	}

	result.clear();

	int32_t width_as_int = static_cast<int32_t>(width * PARTIAL_BLOCKS_COUNT);
	idx_t full_blocks_count = (width_as_int / PARTIAL_BLOCKS_COUNT);
	for (idx_t i = 0; i < full_blocks_count; i++) {
		result += FULL_BLOCK;
	}

	idx_t remaining = width_as_int % PARTIAL_BLOCKS_COUNT;

	if (remaining) {
		result += PARTIAL_BLOCKS[remaining];
	}

	return string_t(result);
}

static void BarFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 3 || args.ColumnCount() == 4);
	auto &x_arg = args.data[0];
	auto &min_arg = args.data[1];
	auto &max_arg = args.data[2];
	string buffer;

	if (args.ColumnCount() == 3) {
		GenericExecutor::ExecuteTernary<PrimitiveType<double>, PrimitiveType<double>, PrimitiveType<double>,
		                                PrimitiveType<string_t>>(
		    x_arg, min_arg, max_arg, result, args.size(),
		    [&](PrimitiveType<double> x, PrimitiveType<double> min, PrimitiveType<double> max) {
			    return StringVector::AddString(result, BarScalarFunction(x.val, min.val, max.val, 80, buffer));
		    });
	} else {
		auto &width_arg = args.data[3];
		GenericExecutor::ExecuteQuaternary<PrimitiveType<double>, PrimitiveType<double>, PrimitiveType<double>,
		                                   PrimitiveType<double>, PrimitiveType<string_t>>(
		    x_arg, min_arg, max_arg, width_arg, result, args.size(),
		    [&](PrimitiveType<double> x, PrimitiveType<double> min, PrimitiveType<double> max,
		        PrimitiveType<double> width) {
			    return StringVector::AddString(result, BarScalarFunction(x.val, min.val, max.val, width.val, buffer));
		    });
	}
}

void BarFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet bar("bar");
	bar.AddFunction(ScalarFunction({LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE},
	                               LogicalType::VARCHAR, BarFunction));
	bar.AddFunction(ScalarFunction({LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE},
	                               LogicalType::VARCHAR, BarFunction));
	set.AddFunction(bar);
}

} // namespace duckdb









#include <string.h>

namespace duckdb {

uint8_t UpperFun::ascii_to_upper_map[] = {
    0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,
    22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
    44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
    66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,
    88,  89,  90,  91,  92,  93,  94,  95,  96,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
    78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  123, 124, 125, 126, 127, 128, 129, 130, 131,
    132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
    154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
    198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
    220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241,
    242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254};
uint8_t LowerFun::ascii_to_lower_map[] = {
    0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,
    22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
    44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  97,
    98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    120, 121, 122, 91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
    132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
    154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
    198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
    220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241,
    242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254};

template <bool IS_UPPER>
static string_t ASCIICaseConvert(Vector &result, const char *input_data, idx_t input_length) {
	idx_t output_length = input_length;
	auto result_str = StringVector::EmptyString(result, output_length);
	auto result_data = result_str.GetDataWriteable();
	for (idx_t i = 0; i < input_length; i++) {
		result_data[i] = IS_UPPER ? UpperFun::ascii_to_upper_map[uint8_t(input_data[i])]
		                          : LowerFun::ascii_to_lower_map[uint8_t(input_data[i])];
	}
	result_str.Finalize();
	return result_str;
}

template <bool IS_UPPER>
static idx_t GetResultLength(const char *input_data, idx_t input_length) {
	idx_t output_length = 0;
	for (idx_t i = 0; i < input_length;) {
		if (input_data[i] & 0x80) {
			// unicode
			int sz = 0;
			int codepoint = utf8proc_codepoint(input_data + i, sz);
			int converted_codepoint = IS_UPPER ? utf8proc_toupper(codepoint) : utf8proc_tolower(codepoint);
			int new_sz = utf8proc_codepoint_length(converted_codepoint);
			D_ASSERT(new_sz >= 0);
			output_length += new_sz;
			i += sz;
		} else {
			// ascii
			output_length++;
			i++;
		}
	}
	return output_length;
}

template <bool IS_UPPER>
static void CaseConvert(const char *input_data, idx_t input_length, char *result_data) {
	for (idx_t i = 0; i < input_length;) {
		if (input_data[i] & 0x80) {
			// non-ascii character
			int sz = 0, new_sz = 0;
			int codepoint = utf8proc_codepoint(input_data + i, sz);
			int converted_codepoint = IS_UPPER ? utf8proc_toupper(codepoint) : utf8proc_tolower(codepoint);
			auto success = utf8proc_codepoint_to_utf8(converted_codepoint, new_sz, result_data);
			D_ASSERT(success);
			(void)success;
			result_data += new_sz;
			i += sz;
		} else {
			// ascii
			*result_data = IS_UPPER ? UpperFun::ascii_to_upper_map[uint8_t(input_data[i])]
			                        : LowerFun::ascii_to_lower_map[uint8_t(input_data[i])];
			result_data++;
			i++;
		}
	}
}

idx_t LowerFun::LowerLength(const char *input_data, idx_t input_length) {
	return GetResultLength<false>(input_data, input_length);
}

void LowerFun::LowerCase(const char *input_data, idx_t input_length, char *result_data) {
	CaseConvert<false>(input_data, input_length, result_data);
}

template <bool IS_UPPER>
static string_t UnicodeCaseConvert(Vector &result, const char *input_data, idx_t input_length) {
	// first figure out the output length
	idx_t output_length = GetResultLength<IS_UPPER>(input_data, input_length);
	auto result_str = StringVector::EmptyString(result, output_length);
	auto result_data = result_str.GetDataWriteable();

	CaseConvert<IS_UPPER>(input_data, input_length, result_data);
	result_str.Finalize();
	return result_str;
}

template <bool IS_UPPER>
struct CaseConvertOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {
		auto input_data = input.GetDataUnsafe();
		auto input_length = input.GetSize();
		return UnicodeCaseConvert<IS_UPPER>(result, input_data, input_length);
	}
};

template <bool IS_UPPER>
static void CaseConvertFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	UnaryExecutor::ExecuteString<string_t, string_t, CaseConvertOperator<IS_UPPER>>(args.data[0], result, args.size());
}

template <bool IS_UPPER>
struct CaseConvertOperatorASCII {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {
		auto input_data = input.GetDataUnsafe();
		auto input_length = input.GetSize();
		return ASCIICaseConvert<IS_UPPER>(result, input_data, input_length);
	}
};

template <bool IS_UPPER>
static void CaseConvertFunctionASCII(DataChunk &args, ExpressionState &state, Vector &result) {
	UnaryExecutor::ExecuteString<string_t, string_t, CaseConvertOperatorASCII<IS_UPPER>>(args.data[0], result,
	                                                                                     args.size());
}

template <bool IS_UPPER>
static unique_ptr<BaseStatistics> CaseConvertPropagateStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &expr = input.expr;
	D_ASSERT(child_stats.size() == 1);
	// can only propagate stats if the children have stats
	if (!child_stats[0]) {
		return nullptr;
	}
	auto &sstats = (StringStatistics &)*child_stats[0];
	if (!sstats.has_unicode) {
		expr.function.function = CaseConvertFunctionASCII<IS_UPPER>;
	}
	return nullptr;
}

ScalarFunction LowerFun::GetFunction() {
	return ScalarFunction("lower", {LogicalType::VARCHAR}, LogicalType::VARCHAR, CaseConvertFunction<false>, nullptr,
	                      nullptr, CaseConvertPropagateStats<false>);
}

void LowerFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction({"lower", "lcase"}, LowerFun::GetFunction());
}

void UpperFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction({"upper", "ucase"},
	                ScalarFunction({LogicalType::VARCHAR}, LogicalType::VARCHAR, CaseConvertFunction<true>, nullptr,
	                               nullptr, CaseConvertPropagateStats<true>));
}

} // namespace duckdb




namespace duckdb {

struct ChrOperator {
	static void GetCodepoint(int32_t input, char c[], int &utf8_bytes) {
		if (input < 0 || !Utf8Proc::CodepointToUtf8(input, utf8_bytes, &c[0])) {
			throw InvalidInputException("Invalid UTF8 Codepoint %d", input);
		}
	}

	template <class TA, class TR>
	static inline TR Operation(const TA &input) {
		char c[5] = {'\0', '\0', '\0', '\0', '\0'};
		int utf8_bytes;
		GetCodepoint(input, c, utf8_bytes);
		return string_t(&c[0], utf8_bytes);
	}
};

#ifdef DUCKDB_DEBUG_NO_INLINE
// the chr function depends on the data always being inlined (which is always possible, since it outputs max 4 bytes)
// to enable chr when string inlining is disabled we create a special function here
static void ChrFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &code_vec = args.data[0];

	char c[5] = {'\0', '\0', '\0', '\0', '\0'};
	int utf8_bytes;
	UnaryExecutor::Execute<int32_t, string_t>(code_vec, result, args.size(), [&](int32_t input) {
		ChrOperator::GetCodepoint(input, c, utf8_bytes);
		return StringVector::AddString(result, &c[0], utf8_bytes);
	});
}
#endif

void CHR::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunction chr("chr", {LogicalType::INTEGER}, LogicalType::VARCHAR,
#ifdef DUCKDB_DEBUG_NO_INLINE
	                   ChrFunction
#else
	                   ScalarFunction::UnaryFunction<int32_t, string_t, ChrOperator>
#endif
	);
	set.AddFunction(chr);
}

} // namespace duckdb







#include <string.h>

namespace duckdb {

static void ConcatFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	// iterate over the vectors to count how large the final string will be
	idx_t constant_lengths = 0;
	vector<idx_t> result_lengths(args.size(), 0);
	for (idx_t col_idx = 0; col_idx < args.ColumnCount(); col_idx++) {
		auto &input = args.data[col_idx];
		D_ASSERT(input.GetType().id() == LogicalTypeId::VARCHAR);
		if (input.GetVectorType() == VectorType::CONSTANT_VECTOR) {
			if (ConstantVector::IsNull(input)) {
				// constant null, skip
				continue;
			}
			auto input_data = ConstantVector::GetData<string_t>(input);
			constant_lengths += input_data->GetSize();
		} else {
			// non-constant vector: set the result type to a flat vector
			result.SetVectorType(VectorType::FLAT_VECTOR);
			// now get the lengths of each of the input elements
			UnifiedVectorFormat vdata;
			input.ToUnifiedFormat(args.size(), vdata);

			auto input_data = (string_t *)vdata.data;
			// now add the length of each vector to the result length
			for (idx_t i = 0; i < args.size(); i++) {
				auto idx = vdata.sel->get_index(i);
				if (!vdata.validity.RowIsValid(idx)) {
					continue;
				}
				result_lengths[i] += input_data[idx].GetSize();
			}
		}
	}

	// first we allocate the empty strings for each of the values
	auto result_data = FlatVector::GetData<string_t>(result);
	for (idx_t i = 0; i < args.size(); i++) {
		// allocate an empty string of the required size
		idx_t str_length = constant_lengths + result_lengths[i];
		result_data[i] = StringVector::EmptyString(result, str_length);
		// we reuse the result_lengths vector to store the currently appended size
		result_lengths[i] = 0;
	}

	// now that the empty space for the strings has been allocated, perform the concatenation
	for (idx_t col_idx = 0; col_idx < args.ColumnCount(); col_idx++) {
		auto &input = args.data[col_idx];

		// loop over the vector and concat to all results
		if (input.GetVectorType() == VectorType::CONSTANT_VECTOR) {
			// constant vector
			if (ConstantVector::IsNull(input)) {
				// constant null, skip
				continue;
			}
			// append the constant vector to each of the strings
			auto input_data = ConstantVector::GetData<string_t>(input);
			auto input_ptr = input_data->GetDataUnsafe();
			auto input_len = input_data->GetSize();
			for (idx_t i = 0; i < args.size(); i++) {
				memcpy(result_data[i].GetDataWriteable() + result_lengths[i], input_ptr, input_len);
				result_lengths[i] += input_len;
			}
		} else {
			// standard vector
			UnifiedVectorFormat idata;
			input.ToUnifiedFormat(args.size(), idata);

			auto input_data = (string_t *)idata.data;
			for (idx_t i = 0; i < args.size(); i++) {
				auto idx = idata.sel->get_index(i);
				if (!idata.validity.RowIsValid(idx)) {
					continue;
				}
				auto input_ptr = input_data[idx].GetDataUnsafe();
				auto input_len = input_data[idx].GetSize();
				memcpy(result_data[i].GetDataWriteable() + result_lengths[i], input_ptr, input_len);
				result_lengths[i] += input_len;
			}
		}
	}
	for (idx_t i = 0; i < args.size(); i++) {
		result_data[i].Finalize();
	}
}

static void ConcatOperator(DataChunk &args, ExpressionState &state, Vector &result) {
	BinaryExecutor::Execute<string_t, string_t, string_t>(
	    args.data[0], args.data[1], result, args.size(), [&](string_t a, string_t b) {
		    auto a_data = a.GetDataUnsafe();
		    auto b_data = b.GetDataUnsafe();
		    auto a_length = a.GetSize();
		    auto b_length = b.GetSize();

		    auto target_length = a_length + b_length;
		    auto target = StringVector::EmptyString(result, target_length);
		    auto target_data = target.GetDataWriteable();

		    memcpy(target_data, a_data, a_length);
		    memcpy(target_data + a_length, b_data, b_length);
		    target.Finalize();
		    return target;
	    });
}

static void TemplatedConcatWS(DataChunk &args, string_t *sep_data, const SelectionVector &sep_sel,
                              const SelectionVector &rsel, idx_t count, Vector &result) {
	vector<idx_t> result_lengths(args.size(), 0);
	vector<bool> has_results(args.size(), false);
	auto orrified_data = unique_ptr<UnifiedVectorFormat[]>(new UnifiedVectorFormat[args.ColumnCount() - 1]);
	for (idx_t col_idx = 1; col_idx < args.ColumnCount(); col_idx++) {
		args.data[col_idx].ToUnifiedFormat(args.size(), orrified_data[col_idx - 1]);
	}

	// first figure out the lengths
	for (idx_t col_idx = 1; col_idx < args.ColumnCount(); col_idx++) {
		auto &idata = orrified_data[col_idx - 1];

		auto input_data = (string_t *)idata.data;
		for (idx_t i = 0; i < count; i++) {
			auto ridx = rsel.get_index(i);
			auto sep_idx = sep_sel.get_index(ridx);
			auto idx = idata.sel->get_index(ridx);
			if (!idata.validity.RowIsValid(idx)) {
				continue;
			}
			if (has_results[ridx]) {
				result_lengths[ridx] += sep_data[sep_idx].GetSize();
			}
			result_lengths[ridx] += input_data[idx].GetSize();
			has_results[ridx] = true;
		}
	}

	// first we allocate the empty strings for each of the values
	auto result_data = FlatVector::GetData<string_t>(result);
	for (idx_t i = 0; i < count; i++) {
		auto ridx = rsel.get_index(i);
		// allocate an empty string of the required size
		result_data[ridx] = StringVector::EmptyString(result, result_lengths[ridx]);
		// we reuse the result_lengths vector to store the currently appended size
		result_lengths[ridx] = 0;
		has_results[ridx] = false;
	}

	// now that the empty space for the strings has been allocated, perform the concatenation
	for (idx_t col_idx = 1; col_idx < args.ColumnCount(); col_idx++) {
		auto &idata = orrified_data[col_idx - 1];
		auto input_data = (string_t *)idata.data;
		for (idx_t i = 0; i < count; i++) {
			auto ridx = rsel.get_index(i);
			auto sep_idx = sep_sel.get_index(ridx);
			auto idx = idata.sel->get_index(ridx);
			if (!idata.validity.RowIsValid(idx)) {
				continue;
			}
			if (has_results[ridx]) {
				auto sep_size = sep_data[sep_idx].GetSize();
				auto sep_ptr = sep_data[sep_idx].GetDataUnsafe();
				memcpy(result_data[ridx].GetDataWriteable() + result_lengths[ridx], sep_ptr, sep_size);
				result_lengths[ridx] += sep_size;
			}
			auto input_ptr = input_data[idx].GetDataUnsafe();
			auto input_len = input_data[idx].GetSize();
			memcpy(result_data[ridx].GetDataWriteable() + result_lengths[ridx], input_ptr, input_len);
			result_lengths[ridx] += input_len;
			has_results[ridx] = true;
		}
	}
	for (idx_t i = 0; i < count; i++) {
		auto ridx = rsel.get_index(i);
		result_data[ridx].Finalize();
	}
}

static void ConcatWSFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &separator = args.data[0];
	UnifiedVectorFormat vdata;
	separator.ToUnifiedFormat(args.size(), vdata);

	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	for (idx_t col_idx = 0; col_idx < args.ColumnCount(); col_idx++) {
		if (args.data[col_idx].GetVectorType() != VectorType::CONSTANT_VECTOR) {
			result.SetVectorType(VectorType::FLAT_VECTOR);
			break;
		}
	}
	switch (separator.GetVectorType()) {
	case VectorType::CONSTANT_VECTOR: {
		if (ConstantVector::IsNull(separator)) {
			// constant NULL as separator: return constant NULL vector
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
			return;
		}
		// no null values
		auto sel = FlatVector::IncrementalSelectionVector();
		TemplatedConcatWS(args, (string_t *)vdata.data, *vdata.sel, *sel, args.size(), result);
		return;
	}
	default: {
		// default case: loop over nullmask and create a non-null selection vector
		idx_t not_null_count = 0;
		SelectionVector not_null_vector(STANDARD_VECTOR_SIZE);
		auto &result_mask = FlatVector::Validity(result);
		for (idx_t i = 0; i < args.size(); i++) {
			if (!vdata.validity.RowIsValid(vdata.sel->get_index(i))) {
				result_mask.SetInvalid(i);
			} else {
				not_null_vector.set_index(not_null_count++, i);
			}
		}
		TemplatedConcatWS(args, (string_t *)vdata.data, *vdata.sel, not_null_vector, not_null_count, result);
		return;
	}
	}
}

void ConcatFun::RegisterFunction(BuiltinFunctions &set) {
	// the concat operator and concat function have different behavior regarding NULLs
	// this is strange but seems consistent with postgresql and mysql
	// (sqlite does not support the concat function, only the concat operator)

	// the concat operator behaves as one would expect: any NULL value present results in a NULL
	// i.e. NULL || 'hello' = NULL
	// the concat function, however, treats NULL values as an empty string
	// i.e. concat(NULL, 'hello') = 'hello'
	// concat_ws functions similarly to the concat function, except the result is NULL if the separator is NULL
	// if the separator is not NULL, however, NULL values are counted as empty string
	// there is one separate rule: there are no separators added between NULL values
	// so the NULL value and empty string are different!
	// e.g.:
	// concat_ws(',', NULL, NULL) = ""
	// concat_ws(',', '', '') = ","
	ScalarFunction concat = ScalarFunction("concat", {LogicalType::VARCHAR}, LogicalType::VARCHAR, ConcatFunction);
	concat.varargs = LogicalType::VARCHAR;
	concat.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	set.AddFunction(concat);

	ScalarFunctionSet concat_op("||");
	concat_op.AddFunction(
	    ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::VARCHAR, ConcatOperator));
	concat_op.AddFunction(ScalarFunction({LogicalType::BLOB, LogicalType::BLOB}, LogicalType::BLOB, ConcatOperator));
	concat_op.AddFunction(ListConcatFun::GetFunction());
	for (auto &fun : concat_op.functions) {
		fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	}
	set.AddFunction(concat_op);

	ScalarFunction concat_ws = ScalarFunction("concat_ws", {LogicalType::VARCHAR, LogicalType::VARCHAR},
	                                          LogicalType::VARCHAR, ConcatWSFunction);
	concat_ws.varargs = LogicalType::VARCHAR;
	concat_ws.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	set.AddFunction(concat_ws);
}

} // namespace duckdb






namespace duckdb {

template <class UNSIGNED, int NEEDLE_SIZE>
static idx_t ContainsUnaligned(const unsigned char *haystack, idx_t haystack_size, const unsigned char *needle,
                               idx_t base_offset) {
	if (NEEDLE_SIZE > haystack_size) {
		// needle is bigger than haystack: haystack cannot contain needle
		return DConstants::INVALID_INDEX;
	}
	// contains for a small unaligned needle (3/5/6/7 bytes)
	// we perform unsigned integer comparisons to check for equality of the entire needle in a single comparison
	// this implementation is inspired by the memmem implementation of freebsd

	// first we set up the needle and the first NEEDLE_SIZE characters of the haystack as UNSIGNED integers
	UNSIGNED needle_entry = 0;
	UNSIGNED haystack_entry = 0;
	const UNSIGNED start = (sizeof(UNSIGNED) * 8) - 8;
	const UNSIGNED shift = (sizeof(UNSIGNED) - NEEDLE_SIZE) * 8;
	for (int i = 0; i < NEEDLE_SIZE; i++) {
		needle_entry |= UNSIGNED(needle[i]) << UNSIGNED(start - i * 8);
		haystack_entry |= UNSIGNED(haystack[i]) << UNSIGNED(start - i * 8);
	}
	// now we perform the actual search
	for (idx_t offset = NEEDLE_SIZE; offset < haystack_size; offset++) {
		// for this position we first compare the haystack with the needle
		if (haystack_entry == needle_entry) {
			return base_offset + offset - NEEDLE_SIZE;
		}
		// now we adjust the haystack entry by
		// (1) removing the left-most character (shift by 8)
		// (2) adding the next character (bitwise or, with potential shift)
		// this shift is only necessary if the needle size is not aligned with the unsigned integer size
		// (e.g. needle size 3, unsigned integer size 4, we need to shift by 1)
		haystack_entry = (haystack_entry << 8) | ((UNSIGNED(haystack[offset])) << shift);
	}
	if (haystack_entry == needle_entry) {
		return base_offset + haystack_size - NEEDLE_SIZE;
	}
	return DConstants::INVALID_INDEX;
}

template <class UNSIGNED>
static idx_t ContainsAligned(const unsigned char *haystack, idx_t haystack_size, const unsigned char *needle,
                             idx_t base_offset) {
	if (sizeof(UNSIGNED) > haystack_size) {
		// needle is bigger than haystack: haystack cannot contain needle
		return DConstants::INVALID_INDEX;
	}
	// contains for a small needle aligned with unsigned integer (2/4/8)
	// similar to ContainsUnaligned, but simpler because we only need to do a reinterpret cast
	auto needle_entry = Load<UNSIGNED>(needle);
	for (idx_t offset = 0; offset <= haystack_size - sizeof(UNSIGNED); offset++) {
		// for this position we first compare the haystack with the needle
		auto haystack_entry = Load<UNSIGNED>(haystack + offset);
		if (needle_entry == haystack_entry) {
			return base_offset + offset;
		}
	}
	return DConstants::INVALID_INDEX;
}

idx_t ContainsGeneric(const unsigned char *haystack, idx_t haystack_size, const unsigned char *needle,
                      idx_t needle_size, idx_t base_offset) {
	if (needle_size > haystack_size) {
		// needle is bigger than haystack: haystack cannot contain needle
		return DConstants::INVALID_INDEX;
	}
	// this implementation is inspired by Raphael Javaux's faststrstr (https://github.com/RaphaelJ/fast_strstr)
	// generic contains; note that we can't use strstr because we don't have null-terminated strings anymore
	// we keep track of a shifting window sum of all characters with window size equal to needle_size
	// this shifting sum is used to avoid calling into memcmp;
	// we only need to call into memcmp when the window sum is equal to the needle sum
	// when that happens, the characters are potentially the same and we call into memcmp to check if they are
	uint32_t sums_diff = 0;
	for (idx_t i = 0; i < needle_size; i++) {
		sums_diff += haystack[i];
		sums_diff -= needle[i];
	}
	idx_t offset = 0;
	while (true) {
		if (sums_diff == 0 && haystack[offset] == needle[0]) {
			if (memcmp(haystack + offset, needle, needle_size) == 0) {
				return base_offset + offset;
			}
		}
		if (offset >= haystack_size - needle_size) {
			return DConstants::INVALID_INDEX;
		}
		sums_diff -= haystack[offset];
		sums_diff += haystack[offset + needle_size];
		offset++;
	}
}

idx_t ContainsFun::Find(const unsigned char *haystack, idx_t haystack_size, const unsigned char *needle,
                        idx_t needle_size) {
	D_ASSERT(needle_size > 0);
	// start off by performing a memchr to find the first character of the
	auto location = memchr(haystack, needle[0], haystack_size);
	if (location == nullptr) {
		return DConstants::INVALID_INDEX;
	}
	idx_t base_offset = (const unsigned char *)location - haystack;
	haystack_size -= base_offset;
	haystack = (const unsigned char *)location;
	// switch algorithm depending on needle size
	switch (needle_size) {
	case 1:
		return base_offset;
	case 2:
		return ContainsAligned<uint16_t>(haystack, haystack_size, needle, base_offset);
	case 3:
		return ContainsUnaligned<uint32_t, 3>(haystack, haystack_size, needle, base_offset);
	case 4:
		return ContainsAligned<uint32_t>(haystack, haystack_size, needle, base_offset);
	case 5:
		return ContainsUnaligned<uint64_t, 5>(haystack, haystack_size, needle, base_offset);
	case 6:
		return ContainsUnaligned<uint64_t, 6>(haystack, haystack_size, needle, base_offset);
	case 7:
		return ContainsUnaligned<uint64_t, 7>(haystack, haystack_size, needle, base_offset);
	case 8:
		return ContainsAligned<uint64_t>(haystack, haystack_size, needle, base_offset);
	default:
		return ContainsGeneric(haystack, haystack_size, needle, needle_size, base_offset);
	}
}

idx_t ContainsFun::Find(const string_t &haystack_s, const string_t &needle_s) {
	auto haystack = (const unsigned char *)haystack_s.GetDataUnsafe();
	auto haystack_size = haystack_s.GetSize();
	auto needle = (const unsigned char *)needle_s.GetDataUnsafe();
	auto needle_size = needle_s.GetSize();
	if (needle_size == 0) {
		// empty needle: always true
		return 0;
	}
	return ContainsFun::Find(haystack, haystack_size, needle, needle_size);
}

struct ContainsOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA left, TB right) {
		return ContainsFun::Find(left, right) != DConstants::INVALID_INDEX;
	}
};

ScalarFunction ContainsFun::GetFunction() {
	return ScalarFunction("contains",                                   // name of the function
	                      {LogicalType::VARCHAR, LogicalType::VARCHAR}, // argument list
	                      LogicalType::BOOLEAN,                         // return type
	                      ScalarFunction::BinaryFunction<string_t, string_t, bool, ContainsOperator>);
}

void ContainsFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(GetFunction());
}

} // namespace duckdb








namespace duckdb {

struct InstrOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA haystack, TB needle) {
		int64_t string_position = 0;

		auto location = ContainsFun::Find(haystack, needle);
		if (location != DConstants::INVALID_INDEX) {
			auto len = (utf8proc_ssize_t)location;
			auto str = reinterpret_cast<const utf8proc_uint8_t *>(haystack.GetDataUnsafe());
			D_ASSERT(len <= (utf8proc_ssize_t)haystack.GetSize());
			for (++string_position; len > 0; ++string_position) {
				utf8proc_int32_t codepoint;
				auto bytes = utf8proc_iterate(str, len, &codepoint);
				str += bytes;
				len -= bytes;
			}
		}
		return string_position;
	}
};

struct InstrAsciiOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA haystack, TB needle) {
		auto location = ContainsFun::Find(haystack, needle);
		return location == DConstants::INVALID_INDEX ? 0 : location + 1;
	}
};

static unique_ptr<BaseStatistics> InStrPropagateStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &expr = input.expr;
	D_ASSERT(child_stats.size() == 2);
	// can only propagate stats if the children have stats
	if (!child_stats[0]) {
		return nullptr;
	}
	// for strpos, we only care if the FIRST string has unicode or not
	auto &sstats = (StringStatistics &)*child_stats[0];
	if (!sstats.has_unicode) {
		expr.function.function = ScalarFunction::BinaryFunction<string_t, string_t, int64_t, InstrAsciiOperator>;
	}
	return nullptr;
}

void InstrFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunction instr("instr",                                      // name of the function
	                     {LogicalType::VARCHAR, LogicalType::VARCHAR}, // argument list
	                     LogicalType::BIGINT,                          // return type
	                     ScalarFunction::BinaryFunction<string_t, string_t, int64_t, InstrOperator>, nullptr, nullptr,
	                     InStrPropagateStats);
	set.AddFunction(instr);
	instr.name = "strpos";
	set.AddFunction(instr);
	instr.name = "position";
	set.AddFunction(instr);
}

} // namespace duckdb




#include <ctype.h>

namespace duckdb {

static inline map<char, idx_t> GetSet(const string_t &str) {
	auto map_of_chars = map<char, idx_t> {};
	idx_t str_len = str.GetSize();
	auto s = str.GetDataUnsafe();

	for (idx_t pos = 0; pos < str_len; pos++) {
		map_of_chars.insert(std::make_pair(s[pos], 1));
	}
	return map_of_chars;
}

static double JaccardSimilarity(const string_t &str, const string_t &txt) {
	if (str.GetSize() < 1 || txt.GetSize() < 1) {
		throw InvalidInputException("Jaccard Function: An argument too short!");
	}
	map<char, idx_t> m_str, m_txt;

	m_str = GetSet(str);
	m_txt = GetSet(txt);

	if (m_str.size() > m_txt.size()) {
		m_str.swap(m_txt);
	}

	for (auto const &achar : m_str) {
		++m_txt[achar.first];
	}
	// m_txt.size is now size of union.

	idx_t size_intersect = 0;
	for (const auto &apair : m_txt) {
		if (apair.second > 1) {
			size_intersect++;
		}
	}

	return (double)size_intersect / (double)m_txt.size();
}

static double JaccardScalarFunction(Vector &result, const string_t str, string_t tgt) {
	return (double)JaccardSimilarity(str, tgt);
}

static void JaccardFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &str_vec = args.data[0];
	auto &tgt_vec = args.data[1];

	BinaryExecutor::Execute<string_t, string_t, double>(
	    str_vec, tgt_vec, result, args.size(),
	    [&](string_t str, string_t tgt) { return JaccardScalarFunction(result, str, tgt); });
}

void JaccardFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet jaccard("jaccard");
	jaccard.AddFunction(ScalarFunction("jaccard", {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::DOUBLE,
	                                   JaccardFunction)); // Pointer to function implementation
	set.AddFunction(jaccard);
}

} // namespace duckdb




namespace duckdb {

static inline double JaroScalarFunction(const string_t &s1, const string_t &s2) {
	auto s1_begin = s1.GetDataUnsafe();
	auto s2_begin = s2.GetDataUnsafe();
	return duckdb_jaro_winkler::jaro_similarity(s1_begin, s1_begin + s1.GetSize(), s2_begin, s2_begin + s2.GetSize());
}

static inline double JaroWinklerScalarFunction(const string_t &s1, const string_t &s2) {
	auto s1_begin = s1.GetDataUnsafe();
	auto s2_begin = s2.GetDataUnsafe();
	return duckdb_jaro_winkler::jaro_winkler_similarity(s1_begin, s1_begin + s1.GetSize(), s2_begin,
	                                                    s2_begin + s2.GetSize());
}

template <class CACHED_SIMILARITY>
static void CachedFunction(Vector &constant, Vector &other, Vector &result, idx_t count) {
	auto val = constant.GetValue(0);
	if (val.IsNull()) {
		auto &result_validity = FlatVector::Validity(result);
		result_validity.SetAllInvalid(count);
		return;
	}

	auto str_val = StringValue::Get(val);
	auto cached = CACHED_SIMILARITY(str_val);
	UnaryExecutor::Execute<string_t, double>(other, result, count, [&](const string_t &other_str) {
		auto other_str_begin = other_str.GetDataUnsafe();
		return cached.similarity(other_str_begin, other_str_begin + other_str.GetSize());
	});
}

template <class CACHED_SIMILARITY, class SIMILARITY_FUNCTION = std::function<double(string_t, string_t)>>
static void TemplatedJaroWinklerFunction(DataChunk &args, Vector &result, SIMILARITY_FUNCTION fun) {
	bool arg0_constant = args.data[0].GetVectorType() == VectorType::CONSTANT_VECTOR;
	bool arg1_constant = args.data[1].GetVectorType() == VectorType::CONSTANT_VECTOR;
	if (!(arg0_constant ^ arg1_constant)) {
		// We can't optimize by caching one of the two strings
		BinaryExecutor::Execute<string_t, string_t, double>(args.data[0], args.data[1], result, args.size(), fun);
		return;
	}

	if (arg0_constant) {
		CachedFunction<CACHED_SIMILARITY>(args.data[0], args.data[1], result, args.size());
	} else {
		CachedFunction<CACHED_SIMILARITY>(args.data[1], args.data[0], result, args.size());
	}
}

static void JaroFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	TemplatedJaroWinklerFunction<duckdb_jaro_winkler::CachedJaroSimilarity<char>>(args, result, JaroScalarFunction);
}

static void JaroWinklerFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	TemplatedJaroWinklerFunction<duckdb_jaro_winkler::CachedJaroWinklerSimilarity<char>>(args, result,
	                                                                                     JaroWinklerScalarFunction);
}

void JaroWinklerFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("jaro_similarity", {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::DOUBLE,
	                               JaroFunction));
	set.AddFunction(ScalarFunction("jaro_winkler_similarity", {LogicalType::VARCHAR, LogicalType::VARCHAR},
	                               LogicalType::DOUBLE, JaroWinklerFunction));
}

} // namespace duckdb




#include <ctype.h>
#include <algorithm>

namespace duckdb {

struct LeftRightUnicode {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return LengthFun::Length<TA, TR>(input);
	}

	static string_t Substring(Vector &result, string_t input, int64_t offset, int64_t length) {
		return SubstringFun::SubstringUnicode(result, input, offset, length);
	}
};

struct LeftRightGrapheme {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return LengthFun::GraphemeCount<TA, TR>(input);
	}

	static string_t Substring(Vector &result, string_t input, int64_t offset, int64_t length) {
		return SubstringFun::SubstringGrapheme(result, input, offset, length);
	}
};

template <class OP>
static string_t LeftScalarFunction(Vector &result, const string_t str, int64_t pos) {
	if (pos >= 0) {
		return OP::Substring(result, str, 1, pos);
	}

	int64_t num_characters = OP::template Operation<string_t, int64_t>(str);
	pos = MaxValue<int64_t>(0, num_characters + pos);
	return OP::Substring(result, str, 1, pos);
}

template <class OP>
static void LeftFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &str_vec = args.data[0];
	auto &pos_vec = args.data[1];

	BinaryExecutor::Execute<string_t, int64_t, string_t>(
	    str_vec, pos_vec, result, args.size(),
	    [&](string_t str, int64_t pos) { return LeftScalarFunction<OP>(result, str, pos); });
}

void LeftFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("left", {LogicalType::VARCHAR, LogicalType::BIGINT}, LogicalType::VARCHAR,
	                               LeftFunction<LeftRightUnicode>));
	set.AddFunction(ScalarFunction("left_grapheme", {LogicalType::VARCHAR, LogicalType::BIGINT}, LogicalType::VARCHAR,
	                               LeftFunction<LeftRightGrapheme>));
}

template <class OP>
static string_t RightScalarFunction(Vector &result, const string_t str, int64_t pos) {
	int64_t num_characters = OP::template Operation<string_t, int64_t>(str);
	if (pos >= 0) {
		int64_t len = MinValue<int64_t>(num_characters, pos);
		int64_t start = num_characters - len + 1;
		return OP::Substring(result, str, start, len);
	}

	int64_t len = 0;
	if (pos != std::numeric_limits<int64_t>::min()) {
		len = num_characters - MinValue<int64_t>(num_characters, -pos);
	}
	int64_t start = num_characters - len + 1;
	return OP::Substring(result, str, start, len);
}

template <class OP>
static void RightFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &str_vec = args.data[0];
	auto &pos_vec = args.data[1];
	BinaryExecutor::Execute<string_t, int64_t, string_t>(
	    str_vec, pos_vec, result, args.size(),
	    [&](string_t str, int64_t pos) { return RightScalarFunction<OP>(result, str, pos); });
}

void RightFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("right", {LogicalType::VARCHAR, LogicalType::BIGINT}, LogicalType::VARCHAR,
	                               RightFunction<LeftRightUnicode>));
	set.AddFunction(ScalarFunction("right_grapheme", {LogicalType::VARCHAR, LogicalType::BIGINT}, LogicalType::VARCHAR,
	                               RightFunction<LeftRightGrapheme>));
}

} // namespace duckdb










namespace duckdb {

// length returns the number of unicode codepoints
struct StringLengthOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return LengthFun::Length<TA, TR>(input);
	}
};

struct GraphemeCountOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return LengthFun::GraphemeCount<TA, TR>(input);
	}
};

struct ArrayLengthOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return input.length;
	}
};

struct ArrayLengthBinaryOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA input, TB dimension) {
		if (dimension != 1) {
			throw NotImplementedException("array_length for dimensions other than 1 not implemented");
		}
		return input.length;
	}
};

// strlen returns the size in bytes
struct StrLenOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return input.GetSize();
	}
};

struct OctetLenOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return Bit::OctetLength(input);
	}
};

// bitlen returns the size in bits
struct BitLenOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return 8 * input.GetSize();
	}
};

// bitstringlen returns the amount of bits in a bitstring
struct BitStringLenOperator {
	template <class TA, class TR>
	static inline TR Operation(TA input) {
		return Bit::BitLength(input);
	}
};

static unique_ptr<BaseStatistics> LengthPropagateStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &expr = input.expr;
	D_ASSERT(child_stats.size() == 1);
	// can only propagate stats if the children have stats
	if (!child_stats[0]) {
		return nullptr;
	}
	auto &sstats = (StringStatistics &)*child_stats[0];
	if (!sstats.has_unicode) {
		expr.function.function = ScalarFunction::UnaryFunction<string_t, int64_t, StrLenOperator>;
	}
	return nullptr;
}

static unique_ptr<FunctionData> ListLengthBind(ClientContext &context, ScalarFunction &bound_function,
                                               vector<unique_ptr<Expression>> &arguments) {
	if (arguments[0]->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	bound_function.arguments[0] = arguments[0]->return_type;
	return nullptr;
}

void LengthFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunction array_length_unary =
	    ScalarFunction({LogicalType::LIST(LogicalType::ANY)}, LogicalType::BIGINT,
	                   ScalarFunction::UnaryFunction<list_entry_t, int64_t, ArrayLengthOperator>, ListLengthBind);
	ScalarFunctionSet length("length");
	length.AddFunction(ScalarFunction({LogicalType::VARCHAR}, LogicalType::BIGINT,
	                                  ScalarFunction::UnaryFunction<string_t, int64_t, StringLengthOperator>, nullptr,
	                                  nullptr, LengthPropagateStats));
	length.AddFunction(ScalarFunction({LogicalType::BIT}, LogicalType::BIGINT,
	                                  ScalarFunction::UnaryFunction<string_t, int64_t, BitStringLenOperator>));
	length.AddFunction(array_length_unary);
	set.AddFunction(length);
	length.name = "len";
	set.AddFunction(length);

	ScalarFunctionSet length_grapheme("length_grapheme");
	length_grapheme.AddFunction(ScalarFunction({LogicalType::VARCHAR}, LogicalType::BIGINT,
	                                           ScalarFunction::UnaryFunction<string_t, int64_t, GraphemeCountOperator>,
	                                           nullptr, nullptr, LengthPropagateStats));
	set.AddFunction(length_grapheme);

	ScalarFunctionSet array_length("array_length");
	array_length.AddFunction(array_length_unary);
	array_length.AddFunction(ScalarFunction(
	    {LogicalType::LIST(LogicalType::ANY), LogicalType::BIGINT}, LogicalType::BIGINT,
	    ScalarFunction::BinaryFunction<list_entry_t, int64_t, int64_t, ArrayLengthBinaryOperator>, ListLengthBind));
	set.AddFunction(array_length);

	set.AddFunction(ScalarFunction("strlen", {LogicalType::VARCHAR}, LogicalType::BIGINT,
	                               ScalarFunction::UnaryFunction<string_t, int64_t, StrLenOperator>));
	ScalarFunctionSet bit_length("bit_length");
	bit_length.AddFunction(ScalarFunction({LogicalType::VARCHAR}, LogicalType::BIGINT,
	                                      ScalarFunction::UnaryFunction<string_t, int64_t, BitLenOperator>));
	bit_length.AddFunction(ScalarFunction({LogicalType::BIT}, LogicalType::BIGINT,
	                                      ScalarFunction::UnaryFunction<string_t, int64_t, BitStringLenOperator>));
	set.AddFunction(bit_length);
	// length for BLOB type
	ScalarFunctionSet octet_length("octet_length");
	octet_length.AddFunction(ScalarFunction({LogicalType::BLOB}, LogicalType::BIGINT,
	                                        ScalarFunction::UnaryFunction<string_t, int64_t, StrLenOperator>));
	octet_length.AddFunction(ScalarFunction({LogicalType::BIT}, LogicalType::BIGINT,
	                                        ScalarFunction::UnaryFunction<string_t, int64_t, OctetLenOperator>));
	set.AddFunction(octet_length);
}

struct UnicodeOperator {
	template <class TA, class TR>
	static inline TR Operation(const TA &input) {
		auto str = reinterpret_cast<const utf8proc_uint8_t *>(input.GetDataUnsafe());
		auto len = input.GetSize();
		utf8proc_int32_t codepoint;
		(void)utf8proc_iterate(str, len, &codepoint);
		return codepoint;
	}
};

void UnicodeFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunction unicode("unicode", {LogicalType::VARCHAR}, LogicalType::INTEGER,
	                       ScalarFunction::UnaryFunction<string_t, int32_t, UnicodeOperator>);
	set.AddFunction(unicode);
	unicode.name = "ord";
	set.AddFunction(unicode);
}

} // namespace duckdb




#include <ctype.h>
#include <algorithm>

namespace duckdb {

// See: https://www.kdnuggets.com/2020/10/optimizing-levenshtein-distance-measuring-text-similarity.html
// And: Iterative 2-row algorithm: https://en.wikipedia.org/wiki/Levenshtein_distance
// Note: A first implementation using the array algorithm version resulted in an error raised by duckdb
// (too muach memory usage)

static idx_t LevenshteinDistance(const string_t &txt, const string_t &tgt) {
	auto txt_len = txt.GetSize();
	auto tgt_len = tgt.GetSize();

	// If one string is empty, the distance equals the length of the other string
	if (txt_len == 0) {
		return tgt_len;
	} else if (tgt_len == 0) {
		return txt_len;
	}

	auto txt_str = txt.GetDataUnsafe();
	auto tgt_str = tgt.GetDataUnsafe();

	// Create two working vectors
	std::vector<idx_t> distances0(tgt_len + 1, 0);
	std::vector<idx_t> distances1(tgt_len + 1, 0);

	idx_t cost_substitution = 0;
	idx_t cost_insertion = 0;
	idx_t cost_deletion = 0;

	// initialize distances0 vector
	// edit distance for an empty txt string is just the number of characters to delete from tgt
	for (idx_t pos_tgt = 0; pos_tgt <= tgt_len; pos_tgt++) {
		distances0[pos_tgt] = pos_tgt;
	}

	for (idx_t pos_txt = 0; pos_txt < txt_len; pos_txt++) {
		// calculate distances1 (current raw distances) from the previous row

		distances1[0] = pos_txt + 1;

		for (idx_t pos_tgt = 0; pos_tgt < tgt_len; pos_tgt++) {
			cost_deletion = distances0[pos_tgt + 1] + 1;
			cost_insertion = distances1[pos_tgt] + 1;
			cost_substitution = distances0[pos_tgt];

			if (txt_str[pos_txt] != tgt_str[pos_tgt]) {
				cost_substitution += 1;
			}

			distances1[pos_tgt + 1] = MinValue(cost_deletion, MinValue(cost_substitution, cost_insertion));
		}
		// copy distances1 (current row) to distances0 (previous row) for next iteration
		// since data in distances1 is always invalidated, a swap without copy is more efficient
		distances0 = distances1;
	}

	return distances0[tgt_len];
}

static int64_t LevenshteinScalarFunction(Vector &result, const string_t str, string_t tgt) {
	return (int64_t)LevenshteinDistance(str, tgt);
}

static void LevenshteinFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &str_vec = args.data[0];
	auto &tgt_vec = args.data[1];

	BinaryExecutor::Execute<string_t, string_t, int64_t>(
	    str_vec, tgt_vec, result, args.size(),
	    [&](string_t str, string_t tgt) { return LevenshteinScalarFunction(result, str, tgt); });
}

void LevenshteinFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet levenshtein("levenshtein");
	levenshtein.AddFunction(ScalarFunction("levenshtein", {LogicalType::VARCHAR, LogicalType::VARCHAR},
	                                       LogicalType::BIGINT,
	                                       LevenshteinFunction)); // Pointer to function implementation
	set.AddFunction(levenshtein);

	ScalarFunctionSet editdist3("editdist3");
	editdist3.AddFunction(ScalarFunction("levenshtein", {LogicalType::VARCHAR, LogicalType::VARCHAR},
	                                     LogicalType::BIGINT, LevenshteinFunction));
	set.AddFunction(editdist3);
}

} // namespace duckdb







namespace duckdb {

struct StandardCharacterReader {
	static char Operation(const char *data, idx_t pos) {
		return data[pos];
	}
};

struct ASCIILCaseReader {
	static char Operation(const char *data, idx_t pos) {
		return (char)LowerFun::ascii_to_lower_map[(uint8_t)data[pos]];
	}
};

template <char PERCENTAGE, char UNDERSCORE, bool HAS_ESCAPE, class READER = StandardCharacterReader>
bool TemplatedLikeOperator(const char *sdata, idx_t slen, const char *pdata, idx_t plen, char escape) {
	idx_t pidx = 0;
	idx_t sidx = 0;
	for (; pidx < plen && sidx < slen; pidx++) {
		char pchar = READER::Operation(pdata, pidx);
		char schar = READER::Operation(sdata, sidx);
		if (HAS_ESCAPE && pchar == escape) {
			pidx++;
			if (pidx == plen) {
				throw SyntaxException("Like pattern must not end with escape character!");
			}
			if (pdata[pidx] != schar) {
				return false;
			}
			sidx++;
		} else if (pchar == UNDERSCORE) {
			sidx++;
		} else if (pchar == PERCENTAGE) {
			pidx++;
			while (pidx < plen && pdata[pidx] == PERCENTAGE) {
				pidx++;
			}
			if (pidx == plen) {
				return true; /* tail is acceptable */
			}
			for (; sidx < slen; sidx++) {
				if (TemplatedLikeOperator<PERCENTAGE, UNDERSCORE, HAS_ESCAPE, READER>(
				        sdata + sidx, slen - sidx, pdata + pidx, plen - pidx, escape)) {
					return true;
				}
			}
			return false;
		} else if (pchar == schar) {
			sidx++;
		} else {
			return false;
		}
	}
	while (pidx < plen && pdata[pidx] == PERCENTAGE) {
		pidx++;
	}
	return pidx == plen && sidx == slen;
}

struct LikeSegment {
	explicit LikeSegment(string pattern) : pattern(std::move(pattern)) {
	}

	string pattern;
};

struct LikeMatcher : public FunctionData {
	LikeMatcher(string like_pattern_p, vector<LikeSegment> segments, bool has_start_percentage, bool has_end_percentage)
	    : like_pattern(std::move(like_pattern_p)), segments(std::move(segments)),
	      has_start_percentage(has_start_percentage), has_end_percentage(has_end_percentage) {
	}

	bool Match(string_t &str) {
		auto str_data = (const unsigned char *)str.GetDataUnsafe();
		auto str_len = str.GetSize();
		idx_t segment_idx = 0;
		idx_t end_idx = segments.size() - 1;
		if (!has_start_percentage) {
			// no start sample_size: match the first part of the string directly
			auto &segment = segments[0];
			if (str_len < segment.pattern.size()) {
				return false;
			}
			if (memcmp(str_data, segment.pattern.c_str(), segment.pattern.size()) != 0) {
				return false;
			}
			str_data += segment.pattern.size();
			str_len -= segment.pattern.size();
			segment_idx++;
			if (segments.size() == 1) {
				// only one segment, and it matches
				// we have a match if there is an end sample_size, OR if the memcmp was an exact match (remaining str is
				// empty)
				return has_end_percentage || str_len == 0;
			}
		}
		// main match loop: for every segment in the middle, use Contains to find the needle in the haystack
		for (; segment_idx < end_idx; segment_idx++) {
			auto &segment = segments[segment_idx];
			// find the pattern of the current segment
			idx_t next_offset = ContainsFun::Find(str_data, str_len, (const unsigned char *)segment.pattern.c_str(),
			                                      segment.pattern.size());
			if (next_offset == DConstants::INVALID_INDEX) {
				// could not find this pattern in the string: no match
				return false;
			}
			idx_t offset = next_offset + segment.pattern.size();
			str_data += offset;
			str_len -= offset;
		}
		if (!has_end_percentage) {
			end_idx--;
			// no end sample_size: match the final segment now
			auto &segment = segments.back();
			if (str_len < segment.pattern.size()) {
				return false;
			}
			if (memcmp(str_data + str_len - segment.pattern.size(), segment.pattern.c_str(), segment.pattern.size()) !=
			    0) {
				return false;
			}
			return true;
		} else {
			auto &segment = segments.back();
			// find the pattern of the current segment
			idx_t next_offset = ContainsFun::Find(str_data, str_len, (const unsigned char *)segment.pattern.c_str(),
			                                      segment.pattern.size());
			return next_offset != DConstants::INVALID_INDEX;
		}
	}

	static unique_ptr<LikeMatcher> CreateLikeMatcher(string like_pattern, char escape = '\0') {
		vector<LikeSegment> segments;
		idx_t last_non_pattern = 0;
		bool has_start_percentage = false;
		bool has_end_percentage = false;
		for (idx_t i = 0; i < like_pattern.size(); i++) {
			auto ch = like_pattern[i];
			if (ch == escape || ch == '%' || ch == '_') {
				// special character, push a constant pattern
				if (i > last_non_pattern) {
					segments.emplace_back(like_pattern.substr(last_non_pattern, i - last_non_pattern));
				}
				last_non_pattern = i + 1;
				if (ch == escape || ch == '_') {
					// escape or underscore: could not create efficient like matcher
					// FIXME: we could handle escaped percentages here
					return nullptr;
				} else {
					// sample_size
					if (i == 0) {
						has_start_percentage = true;
					}
					if (i + 1 == like_pattern.size()) {
						has_end_percentage = true;
					}
				}
			}
		}
		if (last_non_pattern < like_pattern.size()) {
			segments.emplace_back(like_pattern.substr(last_non_pattern, like_pattern.size() - last_non_pattern));
		}
		if (segments.empty()) {
			return nullptr;
		}
		return make_unique<LikeMatcher>(std::move(like_pattern), std::move(segments), has_start_percentage,
		                                has_end_percentage);
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_unique<LikeMatcher>(like_pattern, segments, has_start_percentage, has_end_percentage);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const LikeMatcher &)other_p;
		return like_pattern == other.like_pattern;
	}

private:
	string like_pattern;
	vector<LikeSegment> segments;
	bool has_start_percentage;
	bool has_end_percentage;
};

static unique_ptr<FunctionData> LikeBindFunction(ClientContext &context, ScalarFunction &bound_function,
                                                 vector<unique_ptr<Expression>> &arguments) {
	// pattern is the second argument. If its constant, we can already prepare the pattern and store it for later.
	D_ASSERT(arguments.size() == 2 || arguments.size() == 3);
	if (arguments[1]->IsFoldable()) {
		Value pattern_str = ExpressionExecutor::EvaluateScalar(context, *arguments[1]);
		if (pattern_str.IsNull()) {
			return nullptr;
		}
		return LikeMatcher::CreateLikeMatcher(pattern_str.ToString());
	}
	return nullptr;
}

bool LikeOperatorFunction(const char *s, idx_t slen, const char *pattern, idx_t plen, char escape) {
	return TemplatedLikeOperator<'%', '_', true>(s, slen, pattern, plen, escape);
}

bool LikeOperatorFunction(const char *s, idx_t slen, const char *pattern, idx_t plen) {
	return TemplatedLikeOperator<'%', '_', false>(s, slen, pattern, plen, '\0');
}

bool LikeOperatorFunction(string_t &s, string_t &pat) {
	return LikeOperatorFunction(s.GetDataUnsafe(), s.GetSize(), pat.GetDataUnsafe(), pat.GetSize());
}

bool LikeOperatorFunction(string_t &s, string_t &pat, char escape) {
	return LikeOperatorFunction(s.GetDataUnsafe(), s.GetSize(), pat.GetDataUnsafe(), pat.GetSize(), escape);
}

bool LikeFun::Glob(const char *string, idx_t slen, const char *pattern, idx_t plen) {
	idx_t sidx = 0;
	idx_t pidx = 0;
main_loop : {
	// main matching loop
	while (sidx < slen && pidx < plen) {
		char s = string[sidx];
		char p = pattern[pidx];
		switch (p) {
		case '*': {
			// asterisk: match any set of characters
			// skip any subsequent asterisks
			pidx++;
			while (pidx < plen && pattern[pidx] == '*') {
				pidx++;
			}
			// if the asterisk is the last character, the pattern always matches
			if (pidx == plen) {
				return true;
			}
			// recursively match the remainder of the pattern
			for (; sidx < slen; sidx++) {
				if (LikeFun::Glob(string + sidx, slen - sidx, pattern + pidx, plen - pidx)) {
					return true;
				}
			}
			return false;
		}
		case '?':
			// wildcard: matches anything but null
			break;
		case '[':
			pidx++;
			goto parse_bracket;
		case '\\':
			// escape character, next character needs to match literally
			pidx++;
			// check that we still have a character remaining
			if (pidx == plen) {
				return false;
			}
			p = pattern[pidx];
			if (s != p) {
				return false;
			}
			break;
		default:
			// not a control character: characters need to match literally
			if (s != p) {
				return false;
			}
			break;
		}
		sidx++;
		pidx++;
	}
	while (pidx < plen && pattern[pidx] == '*') {
		pidx++;
	}
	// we are finished only if we have consumed the full pattern
	return pidx == plen && sidx == slen;
}
parse_bracket : {
	// inside a bracket
	if (pidx == plen) {
		return false;
	}
	// check the first character
	// if it is an exclamation mark we need to invert our logic
	char p = pattern[pidx];
	char s = string[sidx];
	bool invert = false;
	if (p == '!') {
		invert = true;
		pidx++;
	}
	bool found_match = invert;
	idx_t start_pos = pidx;
	bool found_closing_bracket = false;
	// now check the remainder of the pattern
	while (pidx < plen) {
		p = pattern[pidx];
		// if the first character is a closing bracket, we match it literally
		// otherwise it indicates an end of bracket
		if (p == ']' && pidx > start_pos) {
			// end of bracket found: we are done
			found_closing_bracket = true;
			pidx++;
			break;
		}
		// we either match a range (a-b) or a single character (a)
		// check if the next character is a dash
		if (pidx + 1 == plen) {
			// no next character!
			break;
		}
		bool matches;
		if (pattern[pidx + 1] == '-') {
			// range! find the next character in the range
			if (pidx + 2 == plen) {
				break;
			}
			char next_char = pattern[pidx + 2];
			// check if the current character is within the range
			matches = s >= p && s <= next_char;
			// shift the pattern forward past the range
			pidx += 3;
		} else {
			// no range! perform a direct match
			matches = p == s;
			// shift the pattern forward past the character
			pidx++;
		}
		if (found_match == invert && matches) {
			// found a match! set the found_matches flag
			// we keep on pattern matching after this until we reach the end bracket
			// however, we don't need to update the found_match flag anymore
			found_match = !invert;
		}
	}
	if (!found_closing_bracket) {
		// no end of bracket: invalid pattern
		return false;
	}
	if (!found_match) {
		// did not match the bracket: return false;
		return false;
	}
	// finished the bracket matching: move forward
	sidx++;
	goto main_loop;
}
}

static char GetEscapeChar(string_t escape) {
	// Only one escape character should be allowed
	if (escape.GetSize() > 1) {
		throw SyntaxException("Invalid escape string. Escape string must be empty or one character.");
	}
	return escape.GetSize() == 0 ? '\0' : *escape.GetDataUnsafe();
}

struct LikeEscapeOperator {
	template <class TA, class TB, class TC>
	static inline bool Operation(TA str, TB pattern, TC escape) {
		char escape_char = GetEscapeChar(escape);
		return LikeOperatorFunction(str.GetDataUnsafe(), str.GetSize(), pattern.GetDataUnsafe(), pattern.GetSize(),
		                            escape_char);
	}
};

struct NotLikeEscapeOperator {
	template <class TA, class TB, class TC>
	static inline bool Operation(TA str, TB pattern, TC escape) {
		return !LikeEscapeOperator::Operation(str, pattern, escape);
	}
};

struct LikeOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA str, TB pattern) {
		return LikeOperatorFunction(str, pattern);
	}
};

bool ILikeOperatorFunction(string_t &str, string_t &pattern, char escape = '\0') {
	auto str_data = str.GetDataUnsafe();
	auto str_size = str.GetSize();
	auto pat_data = pattern.GetDataUnsafe();
	auto pat_size = pattern.GetSize();

	// lowercase both the str and the pattern
	idx_t str_llength = LowerFun::LowerLength(str_data, str_size);
	auto str_ldata = unique_ptr<char[]>(new char[str_llength]);
	LowerFun::LowerCase(str_data, str_size, str_ldata.get());

	idx_t pat_llength = LowerFun::LowerLength(pat_data, pat_size);
	auto pat_ldata = unique_ptr<char[]>(new char[pat_llength]);
	LowerFun::LowerCase(pat_data, pat_size, pat_ldata.get());
	string_t str_lcase(str_ldata.get(), str_llength);
	string_t pat_lcase(pat_ldata.get(), pat_llength);
	return LikeOperatorFunction(str_lcase, pat_lcase, escape);
}

struct ILikeEscapeOperator {
	template <class TA, class TB, class TC>
	static inline bool Operation(TA str, TB pattern, TC escape) {
		char escape_char = GetEscapeChar(escape);
		return ILikeOperatorFunction(str, pattern, escape_char);
	}
};

struct NotILikeEscapeOperator {
	template <class TA, class TB, class TC>
	static inline bool Operation(TA str, TB pattern, TC escape) {
		return !ILikeEscapeOperator::Operation(str, pattern, escape);
	}
};

struct ILikeOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA str, TB pattern) {
		return ILikeOperatorFunction(str, pattern);
	}
};

struct NotLikeOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA str, TB pattern) {
		return !LikeOperatorFunction(str, pattern);
	}
};

struct NotILikeOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA str, TB pattern) {
		return !ILikeOperator::Operation<TA, TB, TR>(str, pattern);
	}
};

struct ILikeOperatorASCII {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA str, TB pattern) {
		return TemplatedLikeOperator<'%', '_', false, ASCIILCaseReader>(
		    str.GetDataUnsafe(), str.GetSize(), pattern.GetDataUnsafe(), pattern.GetSize(), '\0');
	}
};

struct NotILikeOperatorASCII {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA str, TB pattern) {
		return !ILikeOperatorASCII::Operation<TA, TB, TR>(str, pattern);
	}
};

struct GlobOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA str, TB pattern) {
		return LikeFun::Glob(str.GetDataUnsafe(), str.GetSize(), pattern.GetDataUnsafe(), pattern.GetSize());
	}
};

// This can be moved to the scalar_function class
template <typename FUNC>
static void LikeEscapeFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &str = args.data[0];
	auto &pattern = args.data[1];
	auto &escape = args.data[2];

	TernaryExecutor::Execute<string_t, string_t, string_t, bool>(
	    str, pattern, escape, result, args.size(), FUNC::template Operation<string_t, string_t, string_t>);
}

template <class ASCII_OP>
static unique_ptr<BaseStatistics> ILikePropagateStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &expr = input.expr;
	D_ASSERT(child_stats.size() >= 1);
	// can only propagate stats if the children have stats
	if (!child_stats[0]) {
		return nullptr;
	}
	auto &sstats = (StringStatistics &)*child_stats[0];
	if (!sstats.has_unicode) {
		expr.function.function = ScalarFunction::BinaryFunction<string_t, string_t, bool, ASCII_OP>;
	}
	return nullptr;
}

template <class OP, bool INVERT>
static void RegularLikeFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	auto &func_expr = (BoundFunctionExpression &)state.expr;
	if (func_expr.bind_info) {
		auto &matcher = (LikeMatcher &)*func_expr.bind_info;
		// use fast like matcher
		UnaryExecutor::Execute<string_t, bool>(input.data[0], result, input.size(), [&](string_t input) {
			return INVERT ? !matcher.Match(input) : matcher.Match(input);
		});
	} else {
		// use generic like matcher
		BinaryExecutor::ExecuteStandard<string_t, string_t, bool, OP>(input.data[0], input.data[1], result,
		                                                              input.size());
	}
}
void LikeFun::RegisterFunction(BuiltinFunctions &set) {
	// like
	set.AddFunction(ScalarFunction("~~", {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BOOLEAN,
	                               RegularLikeFunction<LikeOperator, false>, LikeBindFunction));
	// not like
	set.AddFunction(ScalarFunction("!~~", {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BOOLEAN,
	                               RegularLikeFunction<NotLikeOperator, true>, LikeBindFunction));
	// glob
	set.AddFunction(ScalarFunction("~~~", {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BOOLEAN,
	                               ScalarFunction::BinaryFunction<string_t, string_t, bool, GlobOperator>));
	// ilike
	set.AddFunction(ScalarFunction("~~*", {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BOOLEAN,
	                               ScalarFunction::BinaryFunction<string_t, string_t, bool, ILikeOperator>, nullptr,
	                               nullptr, ILikePropagateStats<ILikeOperatorASCII>));
	// not ilike
	set.AddFunction(ScalarFunction("!~~*", {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BOOLEAN,
	                               ScalarFunction::BinaryFunction<string_t, string_t, bool, NotILikeOperator>, nullptr,
	                               nullptr, ILikePropagateStats<NotILikeOperatorASCII>));
}

void LikeEscapeFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction({"like_escape"}, ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
	                                                LogicalType::BOOLEAN, LikeEscapeFunction<LikeEscapeOperator>));
	set.AddFunction({"not_like_escape"},
	                ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
	                               LogicalType::BOOLEAN, LikeEscapeFunction<NotLikeEscapeOperator>));

	set.AddFunction({"ilike_escape"}, ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
	                                                 LogicalType::BOOLEAN, LikeEscapeFunction<ILikeEscapeOperator>));
	set.AddFunction({"not_ilike_escape"},
	                ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
	                               LogicalType::BOOLEAN, LikeEscapeFunction<NotILikeEscapeOperator>));
}
} // namespace duckdb






namespace duckdb {

struct MD5Operator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {
		auto hash = StringVector::EmptyString(result, MD5Context::MD5_HASH_LENGTH_TEXT);
		MD5Context context;
		context.Add(input);
		context.FinishHex(hash.GetDataWriteable());
		hash.Finalize();
		return hash;
	}
};

struct MD5Number128Operator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input) {
		data_t digest[MD5Context::MD5_HASH_LENGTH_BINARY];

		MD5Context context;
		context.Add(input);
		context.Finish(digest);
		return *reinterpret_cast<hugeint_t *>(digest);
	}
};

template <bool lower>
struct MD5Number64Operator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input) {
		data_t digest[MD5Context::MD5_HASH_LENGTH_BINARY];

		MD5Context context;
		context.Add(input);
		context.Finish(digest);
		return *reinterpret_cast<uint64_t *>(&digest[lower ? 8 : 0]);
	}
};

static void MD5Function(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &input = args.data[0];

	UnaryExecutor::ExecuteString<string_t, string_t, MD5Operator>(input, result, args.size());
}

static void MD5NumberFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &input = args.data[0];

	UnaryExecutor::Execute<string_t, hugeint_t, MD5Number128Operator>(input, result, args.size());
}

static void MD5NumberUpperFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &input = args.data[0];

	UnaryExecutor::Execute<string_t, uint64_t, MD5Number64Operator<false>>(input, result, args.size());
}

static void MD5NumberLowerFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &input = args.data[0];

	UnaryExecutor::Execute<string_t, uint64_t, MD5Number64Operator<true>>(input, result, args.size());
}

void MD5Fun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("md5",                  // name of the function
	                               {LogicalType::VARCHAR}, // argument list
	                               LogicalType::VARCHAR,   // return type
	                               MD5Function));          // pointer to function implementation

	set.AddFunction(ScalarFunction("md5_number",           // name of the function
	                               {LogicalType::VARCHAR}, // argument list
	                               LogicalType::HUGEINT,   // return type
	                               MD5NumberFunction));    // pointer to function implementation

	set.AddFunction(ScalarFunction("md5_number_upper",       // name of the function
	                               {LogicalType::VARCHAR},   // argument list
	                               LogicalType::UBIGINT,     // return type
	                               MD5NumberUpperFunction)); // pointer to function implementation

	set.AddFunction(ScalarFunction("md5_number_lower",       // name of the function
	                               {LogicalType::VARCHAR},   // argument list
	                               LogicalType::UBIGINT,     // return type
	                               MD5NumberLowerFunction)); // pointer to function implementation
}

} // namespace duckdb



#include <ctype.h>
#include <algorithm>

namespace duckdb {

static int64_t MismatchesScalarFunction(Vector &result, const string_t str, string_t tgt) {
	idx_t str_len = str.GetSize();
	idx_t tgt_len = tgt.GetSize();

	if (str_len != tgt_len) {
		throw InvalidInputException("Mismatch Function: Strings must be of equal length!");
	}
	if (str_len < 1) {
		throw InvalidInputException("Mismatch Function: Strings must be of length > 0!");
	}

	idx_t mismatches = 0;
	auto str_str = str.GetDataUnsafe();
	auto tgt_str = tgt.GetDataUnsafe();

	for (idx_t idx = 0; idx < str_len; ++idx) {
		if (str_str[idx] != tgt_str[idx]) {
			mismatches++;
		}
	}
	return (int64_t)mismatches;
}

static void MismatchesFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &str_vec = args.data[0];
	auto &tgt_vec = args.data[1];

	BinaryExecutor::Execute<string_t, string_t, int64_t>(
	    str_vec, tgt_vec, result, args.size(),
	    [&](string_t str, string_t tgt) { return MismatchesScalarFunction(result, str, tgt); });
}

void MismatchesFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet mismatches("mismatches");
	mismatches.AddFunction(ScalarFunction("mismatches", {LogicalType::VARCHAR, LogicalType::VARCHAR},
	                                      LogicalType::BIGINT,
	                                      MismatchesFunction)); // Pointer to function implementation
	set.AddFunction(mismatches);

	ScalarFunctionSet hamming("hamming");
	hamming.AddFunction(ScalarFunction("mismatches", {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BIGINT,
	                                   MismatchesFunction)); // Pointer to function implementation
	set.AddFunction(hamming);
}

} // namespace duckdb




namespace duckdb {

struct NFCNormalizeOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {
		auto input_data = input.GetDataUnsafe();
		auto input_length = input.GetSize();
		if (StripAccentsFun::IsAscii(input_data, input_length)) {
			return input;
		}
		auto normalized_str = Utf8Proc::Normalize(input_data, input_length);
		D_ASSERT(normalized_str);
		auto result_str = StringVector::AddString(result, normalized_str);
		free(normalized_str);
		return result_str;
	}
};

static void NFCNormalizeFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 1);

	UnaryExecutor::ExecuteString<string_t, string_t, NFCNormalizeOperator>(args.data[0], result, args.size());
	StringVector::AddHeapReference(result, args.data[0]);
}

ScalarFunction NFCNormalizeFun::GetFunction() {
	return ScalarFunction("nfc_normalize", {LogicalType::VARCHAR}, LogicalType::VARCHAR, NFCNormalizeFunction);
}

void NFCNormalizeFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(NFCNormalizeFun::GetFunction());
}

} // namespace duckdb










namespace duckdb {

static pair<idx_t, idx_t> PadCountChars(const idx_t len, const char *data, const idx_t size) {
	//  Count how much of str will fit in the output
	auto str = reinterpret_cast<const utf8proc_uint8_t *>(data);
	idx_t nbytes = 0;
	idx_t nchars = 0;
	for (; nchars < len && nbytes < size; ++nchars) {
		utf8proc_int32_t codepoint;
		auto bytes = utf8proc_iterate(str + nbytes, size - nbytes, &codepoint);
		D_ASSERT(bytes > 0);
		nbytes += bytes;
	}

	return pair<idx_t, idx_t>(nbytes, nchars);
}

static bool InsertPadding(const idx_t len, const string_t &pad, vector<char> &result) {
	//  Copy the padding until the output is long enough
	auto data = pad.GetDataUnsafe();
	auto size = pad.GetSize();

	//  Check whether we need data that we don't have
	if (len > 0 && size == 0) {
		return false;
	}

	//  Insert characters until we have all we need.
	auto str = reinterpret_cast<const utf8proc_uint8_t *>(data);
	idx_t nbytes = 0;
	for (idx_t nchars = 0; nchars < len; ++nchars) {
		//  If we are at the end of the pad, flush all of it and loop back
		if (nbytes >= size) {
			result.insert(result.end(), data, data + size);
			nbytes = 0;
		}

		//  Write the next character
		utf8proc_int32_t codepoint;
		auto bytes = utf8proc_iterate(str + nbytes, size - nbytes, &codepoint);
		D_ASSERT(bytes > 0);
		nbytes += bytes;
	}

	//  Flush the remaining pad
	result.insert(result.end(), data, data + nbytes);

	return true;
}

static string_t LeftPadFunction(const string_t &str, const int32_t len, const string_t &pad, vector<char> &result) {
	//  Reuse the buffer
	result.clear();

	// Get information about the base string
	auto data_str = str.GetDataUnsafe();
	auto size_str = str.GetSize();

	//  Count how much of str will fit in the output
	auto written = PadCountChars(len, data_str, size_str);

	//  Left pad by the number of characters still needed
	if (!InsertPadding(len - written.second, pad, result)) {
		throw Exception("Insufficient padding in LPAD.");
	}

	//  Append as much of the original string as fits
	result.insert(result.end(), data_str, data_str + written.first);

	return string_t(result.data(), result.size());
}

struct LeftPadOperator {
	static inline string_t Operation(const string_t &str, const int32_t len, const string_t &pad,
	                                 vector<char> &result) {
		return LeftPadFunction(str, len, pad, result);
	}
};

static string_t RightPadFunction(const string_t &str, const int32_t len, const string_t &pad, vector<char> &result) {
	//  Reuse the buffer
	result.clear();

	// Get information about the base string
	auto data_str = str.GetDataUnsafe();
	auto size_str = str.GetSize();

	// Count how much of str will fit in the output
	auto written = PadCountChars(len, data_str, size_str);

	//  Append as much of the original string as fits
	result.insert(result.end(), data_str, data_str + written.first);

	//  Right pad by the number of characters still needed
	if (!InsertPadding(len - written.second, pad, result)) {
		throw Exception("Insufficient padding in RPAD.");
	};

	return string_t(result.data(), result.size());
}

struct RightPadOperator {
	static inline string_t Operation(const string_t &str, const int32_t len, const string_t &pad,
	                                 vector<char> &result) {
		return RightPadFunction(str, len, pad, result);
	}
};

template <class OP>
static void PadFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &str_vector = args.data[0];
	auto &len_vector = args.data[1];
	auto &pad_vector = args.data[2];

	vector<char> buffer;
	TernaryExecutor::Execute<string_t, int32_t, string_t, string_t>(
	    str_vector, len_vector, pad_vector, result, args.size(), [&](string_t str, int32_t len, string_t pad) {
		    len = MaxValue<int32_t>(len, 0);
		    return StringVector::AddString(result, OP::Operation(str, len, pad, buffer));
	    });
}

void LpadFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("lpad",                                      // name of the function
	                               {LogicalType::VARCHAR, LogicalType::INTEGER, // argument list
	                                LogicalType::VARCHAR},
	                               LogicalType::VARCHAR,           // return type
	                               PadFunction<LeftPadOperator>)); // pointer to function implementation
}

void RpadFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("rpad",                                      // name of the function
	                               {LogicalType::VARCHAR, LogicalType::INTEGER, // argument list
	                                LogicalType::VARCHAR},
	                               LogicalType::VARCHAR,            // return type
	                               PadFunction<RightPadOperator>)); // pointer to function implementation
}

} // namespace duckdb





namespace duckdb {

static bool PrefixFunction(const string_t &str, const string_t &pattern);

struct PrefixOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA left, TB right) {
		return PrefixFunction(left, right);
	}
};
static bool PrefixFunction(const string_t &str, const string_t &pattern) {
	auto str_length = str.GetSize();
	auto patt_length = pattern.GetSize();
	if (patt_length > str_length) {
		return false;
	}
	if (patt_length <= string_t::PREFIX_LENGTH) {
		// short prefix
		if (patt_length == 0) {
			// length = 0, return true
			return true;
		}

		// prefix early out
		const char *str_pref = str.GetPrefix();
		const char *patt_pref = pattern.GetPrefix();
		for (idx_t i = 0; i < patt_length; ++i) {
			if (str_pref[i] != patt_pref[i]) {
				return false;
			}
		}
		return true;
	} else {
		// prefix early out
		const char *str_pref = str.GetPrefix();
		const char *patt_pref = pattern.GetPrefix();
		for (idx_t i = 0; i < string_t::PREFIX_LENGTH; ++i) {
			if (str_pref[i] != patt_pref[i]) {
				// early out
				return false;
			}
		}
		// compare the rest of the prefix
		const char *str_data = str.GetDataUnsafe();
		const char *patt_data = pattern.GetDataUnsafe();
		D_ASSERT(patt_length <= str_length);
		for (idx_t i = string_t::PREFIX_LENGTH; i < patt_length; ++i) {
			if (str_data[i] != patt_data[i]) {
				return false;
			}
		}
		return true;
	}
}

ScalarFunction PrefixFun::GetFunction() {
	return ScalarFunction("prefix",                                     // name of the function
	                      {LogicalType::VARCHAR, LogicalType::VARCHAR}, // argument list
	                      LogicalType::BOOLEAN,                         // return type
	                      ScalarFunction::BinaryFunction<string_t, string_t, bool, PrefixOperator>);
}

void PrefixFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(GetFunction());
}

} // namespace duckdb






namespace duckdb {

struct FMTPrintf {
	template <class CTX>
	static string OP(const char *format_str, std::vector<duckdb_fmt::basic_format_arg<CTX>> &format_args) {
		return duckdb_fmt::vsprintf(
		    format_str, duckdb_fmt::basic_format_args<CTX>(format_args.data(), static_cast<int>(format_args.size())));
	}
};

struct FMTFormat {
	template <class CTX>
	static string OP(const char *format_str, std::vector<duckdb_fmt::basic_format_arg<CTX>> &format_args) {
		return duckdb_fmt::vformat(
		    format_str, duckdb_fmt::basic_format_args<CTX>(format_args.data(), static_cast<int>(format_args.size())));
	}
};

unique_ptr<FunctionData> BindPrintfFunction(ClientContext &context, ScalarFunction &bound_function,
                                            vector<unique_ptr<Expression>> &arguments) {
	for (idx_t i = 1; i < arguments.size(); i++) {
		switch (arguments[i]->return_type.id()) {
		case LogicalTypeId::BOOLEAN:
		case LogicalTypeId::TINYINT:
		case LogicalTypeId::SMALLINT:
		case LogicalTypeId::INTEGER:
		case LogicalTypeId::BIGINT:
		case LogicalTypeId::FLOAT:
		case LogicalTypeId::DOUBLE:
		case LogicalTypeId::VARCHAR:
			// these types are natively supported
			bound_function.arguments.push_back(arguments[i]->return_type);
			break;
		case LogicalTypeId::DECIMAL:
			// decimal type: add cast to double
			bound_function.arguments.emplace_back(LogicalType::DOUBLE);
			break;
		case LogicalTypeId::UNKNOWN:
			// parameter: accept any input and rebind later
			bound_function.arguments.emplace_back(LogicalType::ANY);
			break;
		default:
			// all other types: add cast to string
			bound_function.arguments.emplace_back(LogicalType::VARCHAR);
			break;
		}
	}
	return nullptr;
}

template <class FORMAT_FUN, class CTX>
static void PrintfFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &format_string = args.data[0];
	auto &result_validity = FlatVector::Validity(result);
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	result_validity.Initialize(args.size());
	for (idx_t i = 0; i < args.ColumnCount(); i++) {
		switch (args.data[i].GetVectorType()) {
		case VectorType::CONSTANT_VECTOR:
			if (ConstantVector::IsNull(args.data[i])) {
				// constant null! result is always NULL regardless of other input
				result.SetVectorType(VectorType::CONSTANT_VECTOR);
				ConstantVector::SetNull(result, true);
				return;
			}
			break;
		default:
			// FLAT VECTOR, we can directly OR the nullmask
			args.data[i].Flatten(args.size());
			result.SetVectorType(VectorType::FLAT_VECTOR);
			result_validity.Combine(FlatVector::Validity(args.data[i]), args.size());
			break;
		}
	}
	idx_t count = result.GetVectorType() == VectorType::CONSTANT_VECTOR ? 1 : args.size();

	auto format_data = FlatVector::GetData<string_t>(format_string);
	auto result_data = FlatVector::GetData<string_t>(result);
	for (idx_t idx = 0; idx < count; idx++) {
		if (result.GetVectorType() == VectorType::FLAT_VECTOR && FlatVector::IsNull(result, idx)) {
			// this entry is NULL: skip it
			continue;
		}

		// first fetch the format string
		auto fmt_idx = format_string.GetVectorType() == VectorType::CONSTANT_VECTOR ? 0 : idx;
		auto format_string = format_data[fmt_idx].GetString();

		// now gather all the format arguments
		std::vector<duckdb_fmt::basic_format_arg<CTX>> format_args;
		std::vector<unique_ptr<data_t[]>> string_args;

		for (idx_t col_idx = 1; col_idx < args.ColumnCount(); col_idx++) {
			auto &col = args.data[col_idx];
			idx_t arg_idx = col.GetVectorType() == VectorType::CONSTANT_VECTOR ? 0 : idx;
			switch (col.GetType().id()) {
			case LogicalTypeId::BOOLEAN: {
				auto arg_data = FlatVector::GetData<bool>(col);
				format_args.emplace_back(duckdb_fmt::internal::make_arg<CTX>(arg_data[arg_idx]));
				break;
			}
			case LogicalTypeId::TINYINT: {
				auto arg_data = FlatVector::GetData<int8_t>(col);
				format_args.emplace_back(duckdb_fmt::internal::make_arg<CTX>(arg_data[arg_idx]));
				break;
			}
			case LogicalTypeId::SMALLINT: {
				auto arg_data = FlatVector::GetData<int16_t>(col);
				format_args.emplace_back(duckdb_fmt::internal::make_arg<CTX>(arg_data[arg_idx]));
				break;
			}
			case LogicalTypeId::INTEGER: {
				auto arg_data = FlatVector::GetData<int32_t>(col);
				format_args.emplace_back(duckdb_fmt::internal::make_arg<CTX>(arg_data[arg_idx]));
				break;
			}
			case LogicalTypeId::BIGINT: {
				auto arg_data = FlatVector::GetData<int64_t>(col);
				format_args.emplace_back(duckdb_fmt::internal::make_arg<CTX>(arg_data[arg_idx]));
				break;
			}
			case LogicalTypeId::FLOAT: {
				auto arg_data = FlatVector::GetData<float>(col);
				format_args.emplace_back(duckdb_fmt::internal::make_arg<CTX>(arg_data[arg_idx]));
				break;
			}
			case LogicalTypeId::DOUBLE: {
				auto arg_data = FlatVector::GetData<double>(col);
				format_args.emplace_back(duckdb_fmt::internal::make_arg<CTX>(arg_data[arg_idx]));
				break;
			}
			case LogicalTypeId::VARCHAR: {
				auto arg_data = FlatVector::GetData<string_t>(col);
				auto string_view =
				    duckdb_fmt::basic_string_view<char>(arg_data[arg_idx].GetDataUnsafe(), arg_data[arg_idx].GetSize());
				format_args.emplace_back(duckdb_fmt::internal::make_arg<CTX>(string_view));
				break;
			}
			default:
				throw InternalException("Unexpected type for printf format");
			}
		}
		// finally actually perform the format
		string dynamic_result = FORMAT_FUN::template OP<CTX>(format_string.c_str(), format_args);
		result_data[idx] = StringVector::AddString(result, dynamic_result);
	}
}

void PrintfFun::RegisterFunction(BuiltinFunctions &set) {
	// duckdb_fmt::printf_context, duckdb_fmt::vsprintf
	ScalarFunction printf_fun =
	    ScalarFunction("printf", {LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                   PrintfFunction<FMTPrintf, duckdb_fmt::printf_context>, BindPrintfFunction);
	printf_fun.varargs = LogicalType::ANY;
	set.AddFunction(printf_fun);

	// duckdb_fmt::format_context, duckdb_fmt::vformat
	ScalarFunction format_fun =
	    ScalarFunction("format", {LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                   PrintfFunction<FMTFormat, duckdb_fmt::format_context>, BindPrintfFunction);
	format_fun.varargs = LogicalType::ANY;
	set.AddFunction(format_fun);
}

} // namespace duckdb












namespace duckdb {

static bool RegexOptionsEquals(const duckdb_re2::RE2::Options &opt_a, const duckdb_re2::RE2::Options &opt_b) {
	return opt_a.case_sensitive() == opt_b.case_sensitive();
}

RegexpBaseBindData::RegexpBaseBindData() : constant_pattern(false) {
}
RegexpBaseBindData::RegexpBaseBindData(duckdb_re2::RE2::Options options, string constant_string_p,
                                       bool constant_pattern)
    : options(options), constant_string(std::move(constant_string_p)), constant_pattern(constant_pattern) {
}

RegexpBaseBindData::~RegexpBaseBindData() {
}

bool RegexpBaseBindData::Equals(const FunctionData &other_p) const {
	auto &other = (RegexpBaseBindData &)other_p;
	return constant_pattern == other.constant_pattern && constant_string == other.constant_string &&
	       RegexOptionsEquals(options, other.options);
}

static inline duckdb_re2::StringPiece CreateStringPiece(string_t &input) {
	return duckdb_re2::StringPiece(input.GetDataUnsafe(), input.GetSize());
}

unique_ptr<FunctionLocalState> RegexInitLocalState(ExpressionState &state, const BoundFunctionExpression &expr,
                                                   FunctionData *bind_data) {
	auto &info = (RegexpBaseBindData &)*bind_data;
	if (info.constant_pattern) {
		return make_unique<RegexLocalState>(info);
	}
	return nullptr;
}

static void ParseRegexOptions(const string &options, duckdb_re2::RE2::Options &result, bool *global_replace = nullptr) {
	for (idx_t i = 0; i < options.size(); i++) {
		switch (options[i]) {
		case 'c':
			// case-sensitive matching
			result.set_case_sensitive(true);
			break;
		case 'i':
			// case-insensitive matching
			result.set_case_sensitive(false);
			break;
		case 'l':
			// literal matching
			result.set_literal(true);
			break;
		case 'm':
		case 'n':
		case 'p':
			// newline-sensitive matching
			result.set_dot_nl(false);
			break;
		case 's':
			// non-newline-sensitive matching
			result.set_dot_nl(true);
			break;
		case 'g':
			// global replace, only available for regexp_replace
			if (global_replace) {
				*global_replace = true;
			} else {
				throw InvalidInputException("Option 'g' (global replace) is only valid for regexp_replace");
			}
			break;
		case ' ':
		case '\t':
		case '\n':
			// ignore whitespace
			break;
		default:
			throw InvalidInputException("Unrecognized Regex option %c", options[i]);
		}
	}
}

void ParseRegexOptions(ClientContext &context, Expression &expr, RE2::Options &target, bool *global_replace = nullptr) {
	if (expr.HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (!expr.IsFoldable()) {
		throw InvalidInputException("Regex options field must be a constant");
	}
	Value options_str = ExpressionExecutor::EvaluateScalar(context, expr);
	if (!options_str.IsNull() && options_str.type().id() == LogicalTypeId::VARCHAR) {
		ParseRegexOptions(StringValue::Get(options_str), target, global_replace);
	}
}

static bool TryParseConstantPattern(ClientContext &context, Expression &expr, string &constant_string) {
	if (!expr.IsFoldable()) {
		return false;
	}
	Value pattern_str = ExpressionExecutor::EvaluateScalar(context, expr);
	if (!pattern_str.IsNull() && pattern_str.type().id() == LogicalTypeId::VARCHAR) {
		constant_string = StringValue::Get(pattern_str);
		return true;
	}
	return false;
}

//===--------------------------------------------------------------------===//
// Regexp Matches
//===--------------------------------------------------------------------===//
RegexpMatchesBindData::RegexpMatchesBindData(duckdb_re2::RE2::Options options, string constant_string_p,
                                             bool constant_pattern)
    : RegexpBaseBindData(options, std::move(constant_string_p), constant_pattern) {
	if (constant_pattern) {
		auto pattern = make_unique<RE2>(constant_string, options);
		if (!pattern->ok()) {
			throw Exception(pattern->error());
		}

		range_success = pattern->PossibleMatchRange(&range_min, &range_max, 1000);
	} else {
		range_success = false;
	}
}

RegexpMatchesBindData::RegexpMatchesBindData(duckdb_re2::RE2::Options options, string constant_string_p,
                                             bool constant_pattern, string range_min_p, string range_max_p,
                                             bool range_success)
    : RegexpBaseBindData(options, std::move(constant_string_p), constant_pattern), range_min(std::move(range_min_p)),
      range_max(std::move(range_max_p)), range_success(range_success) {
}

unique_ptr<FunctionData> RegexpMatchesBindData::Copy() const {
	return make_unique<RegexpMatchesBindData>(options, constant_string, constant_pattern, range_min, range_max,
	                                          range_success);
}

unique_ptr<FunctionData> RegexpMatchesBind(ClientContext &context, ScalarFunction &bound_function,
                                           vector<unique_ptr<Expression>> &arguments) {
	// pattern is the second argument. If its constant, we can already prepare the pattern and store it for later.
	D_ASSERT(arguments.size() == 2 || arguments.size() == 3);
	RE2::Options options;
	options.set_log_errors(false);
	if (arguments.size() == 3) {
		ParseRegexOptions(context, *arguments[2], options);
	}

	string constant_string;
	bool constant_pattern;
	constant_pattern = TryParseConstantPattern(context, *arguments[1], constant_string);
	return make_unique<RegexpMatchesBindData>(options, std::move(constant_string), constant_pattern);
}

struct RegexPartialMatch {
	static inline bool Operation(const duckdb_re2::StringPiece &input, duckdb_re2::RE2 &re) {
		return duckdb_re2::RE2::PartialMatch(input, re);
	}
};

struct RegexFullMatch {
	static inline bool Operation(const duckdb_re2::StringPiece &input, duckdb_re2::RE2 &re) {
		return duckdb_re2::RE2::FullMatch(input, re);
	}
};

template <class OP>
static void RegexpMatchesFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &strings = args.data[0];
	auto &patterns = args.data[1];

	auto &func_expr = (BoundFunctionExpression &)state.expr;
	auto &info = (RegexpMatchesBindData &)*func_expr.bind_info;

	if (info.constant_pattern) {
		auto &lstate = (RegexLocalState &)*ExecuteFunctionState::GetFunctionState(state);
		UnaryExecutor::Execute<string_t, bool>(strings, result, args.size(), [&](string_t input) {
			return OP::Operation(CreateStringPiece(input), lstate.constant_pattern);
		});
	} else {
		BinaryExecutor::Execute<string_t, string_t, bool>(strings, patterns, result, args.size(),
		                                                  [&](string_t input, string_t pattern) {
			                                                  RE2 re(CreateStringPiece(pattern), info.options);
			                                                  if (!re.ok()) {
				                                                  throw Exception(re.error());
			                                                  }
			                                                  return OP::Operation(CreateStringPiece(input), re);
		                                                  });
	}
}

//===--------------------------------------------------------------------===//
// Regexp Replace
//===--------------------------------------------------------------------===//
RegexpReplaceBindData::RegexpReplaceBindData() : global_replace(false) {
}

RegexpReplaceBindData::RegexpReplaceBindData(duckdb_re2::RE2::Options options, string constant_string_p,
                                             bool constant_pattern, bool global_replace)
    : RegexpBaseBindData(options, std::move(constant_string_p), constant_pattern), global_replace(global_replace) {
}

unique_ptr<FunctionData> RegexpReplaceBindData::Copy() const {
	auto copy = make_unique<RegexpReplaceBindData>(options, constant_string, constant_pattern, global_replace);
	return std::move(copy);
}

bool RegexpReplaceBindData::Equals(const FunctionData &other_p) const {
	auto &other = (const RegexpReplaceBindData &)other_p;
	return RegexpBaseBindData::Equals(other) && global_replace == other.global_replace;
}

static unique_ptr<FunctionData> RegexReplaceBind(ClientContext &context, ScalarFunction &bound_function,
                                                 vector<unique_ptr<Expression>> &arguments) {
	auto data = make_unique<RegexpReplaceBindData>();

	data->constant_pattern = TryParseConstantPattern(context, *arguments[1], data->constant_string);
	if (arguments.size() == 4) {
		ParseRegexOptions(context, *arguments[3], data->options, &data->global_replace);
	}
	data->options.set_log_errors(false);
	return std::move(data);
}

static void RegexReplaceFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = (BoundFunctionExpression &)state.expr;
	auto &info = (RegexpReplaceBindData &)*func_expr.bind_info;

	auto &strings = args.data[0];
	auto &patterns = args.data[1];
	auto &replaces = args.data[2];

	if (info.constant_pattern) {
		auto &lstate = (RegexLocalState &)*ExecuteFunctionState::GetFunctionState(state);
		BinaryExecutor::Execute<string_t, string_t, string_t>(
		    strings, replaces, result, args.size(), [&](string_t input, string_t replace) {
			    std::string sstring = input.GetString();
			    if (info.global_replace) {
				    RE2::GlobalReplace(&sstring, lstate.constant_pattern, CreateStringPiece(replace));
			    } else {
				    RE2::Replace(&sstring, lstate.constant_pattern, CreateStringPiece(replace));
			    }
			    return StringVector::AddString(result, sstring);
		    });
	} else {
		TernaryExecutor::Execute<string_t, string_t, string_t, string_t>(
		    strings, patterns, replaces, result, args.size(), [&](string_t input, string_t pattern, string_t replace) {
			    RE2 re(CreateStringPiece(pattern), info.options);
			    std::string sstring = input.GetString();
			    if (info.global_replace) {
				    RE2::GlobalReplace(&sstring, re, CreateStringPiece(replace));
			    } else {
				    RE2::Replace(&sstring, re, CreateStringPiece(replace));
			    }
			    return StringVector::AddString(result, sstring);
		    });
	}
}

//===--------------------------------------------------------------------===//
// Regexp Extract
//===--------------------------------------------------------------------===//
RegexpExtractBindData::RegexpExtractBindData() {
}

RegexpExtractBindData::RegexpExtractBindData(duckdb_re2::RE2::Options options, string constant_string_p,
                                             bool constant_pattern, string group_string_p)
    : RegexpBaseBindData(options, std::move(constant_string_p), constant_pattern),
      group_string(std::move(group_string_p)), rewrite(group_string) {
}

unique_ptr<FunctionData> RegexpExtractBindData::Copy() const {
	return make_unique<RegexpExtractBindData>(options, constant_string, constant_pattern, group_string);
}

bool RegexpExtractBindData::Equals(const FunctionData &other_p) const {
	auto &other = (const RegexpExtractBindData &)other_p;
	return RegexpBaseBindData::Equals(other) && group_string == other.group_string;
}

static unique_ptr<FunctionData> RegexExtractBind(ClientContext &context, ScalarFunction &bound_function,
                                                 vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(arguments.size() >= 2);

	duckdb_re2::RE2::Options options;

	string constant_string;
	bool constant_pattern = TryParseConstantPattern(context, *arguments[1], constant_string);

	string group_string = "";
	if (arguments.size() >= 3) {
		if (arguments[2]->HasParameter()) {
			throw ParameterNotResolvedException();
		}
		if (!arguments[2]->IsFoldable()) {
			throw InvalidInputException("Group index field field must be a constant!");
		}
		Value group = ExpressionExecutor::EvaluateScalar(context, *arguments[2]);
		if (!group.IsNull()) {
			auto group_idx = group.GetValue<int32_t>();
			if (group_idx < 0 || group_idx > 9) {
				throw InvalidInputException("Group index must be between 0 and 9!");
			}
			group_string = "\\" + to_string(group_idx);
		}
	} else {
		group_string = "\\0";
	}
	if (arguments.size() >= 4) {
		ParseRegexOptions(context, *arguments[3], options);
	}
	return make_unique<RegexpExtractBindData>(options, std::move(constant_string), constant_pattern,
	                                          std::move(group_string));
}

inline static string_t Extract(const string_t &input, Vector &result, const RE2 &re,
                               const duckdb_re2::StringPiece &rewrite) {
	std::string extracted;
	RE2::Extract(input.GetString(), re, rewrite, &extracted);
	return StringVector::AddString(result, extracted.c_str(), extracted.size());
}

static void RegexExtractFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = (BoundFunctionExpression &)state.expr;
	const auto &info = (RegexpExtractBindData &)*func_expr.bind_info;

	auto &strings = args.data[0];
	auto &patterns = args.data[1];
	if (info.constant_pattern) {
		auto &lstate = (RegexLocalState &)*ExecuteFunctionState::GetFunctionState(state);
		UnaryExecutor::Execute<string_t, string_t>(strings, result, args.size(), [&](string_t input) {
			return Extract(input, result, lstate.constant_pattern, info.rewrite);
		});
	} else {
		BinaryExecutor::Execute<string_t, string_t, string_t>(strings, patterns, result, args.size(),
		                                                      [&](string_t input, string_t pattern) {
			                                                      RE2 re(CreateStringPiece(pattern), info.options);
			                                                      return Extract(input, result, re, info.rewrite);
		                                                      });
	}
}

void RegexpFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet regexp_full_match("regexp_full_match");
	regexp_full_match.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BOOLEAN, RegexpMatchesFunction<RegexFullMatch>,
	    RegexpMatchesBind, nullptr, nullptr, RegexInitLocalState, LogicalType::INVALID,
	    FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING));
	regexp_full_match.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BOOLEAN,
	    RegexpMatchesFunction<RegexFullMatch>, RegexpMatchesBind, nullptr, nullptr, RegexInitLocalState,
	    LogicalType::INVALID, FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING));

	ScalarFunctionSet regexp_partial_match("regexp_matches");
	regexp_partial_match.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BOOLEAN, RegexpMatchesFunction<RegexPartialMatch>,
	    RegexpMatchesBind, nullptr, nullptr, RegexInitLocalState, LogicalType::INVALID,
	    FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING));
	regexp_partial_match.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BOOLEAN,
	    RegexpMatchesFunction<RegexPartialMatch>, RegexpMatchesBind, nullptr, nullptr, RegexInitLocalState,
	    LogicalType::INVALID, FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING));

	ScalarFunctionSet regexp_replace("regexp_replace");
	regexp_replace.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
	                                          LogicalType::VARCHAR, RegexReplaceFunction, RegexReplaceBind, nullptr,
	                                          nullptr, RegexInitLocalState));
	regexp_replace.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::VARCHAR,
	    RegexReplaceFunction, RegexReplaceBind, nullptr, nullptr, RegexInitLocalState));

	ScalarFunctionSet regexp_extract("regexp_extract");
	regexp_extract.AddFunction(
	    ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::VARCHAR, RegexExtractFunction,
	                   RegexExtractBind, nullptr, nullptr, RegexInitLocalState, LogicalType::INVALID,
	                   FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING));
	regexp_extract.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER}, LogicalType::VARCHAR, RegexExtractFunction,
	    RegexExtractBind, nullptr, nullptr, RegexInitLocalState, LogicalType::INVALID,
	    FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING));
	regexp_extract.AddFunction(ScalarFunction(
	    {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::INTEGER, LogicalType::VARCHAR}, LogicalType::VARCHAR,
	    RegexExtractFunction, RegexExtractBind, nullptr, nullptr, RegexInitLocalState, LogicalType::INVALID,
	    FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING));

	set.AddFunction(regexp_full_match);
	set.AddFunction(regexp_partial_match);
	set.AddFunction(regexp_replace);
	set.AddFunction(regexp_extract);
}

} // namespace duckdb





#include <string.h>
#include <ctype.h>

namespace duckdb {

static string_t RepeatScalarFunction(const string_t &str, const int64_t cnt, vector<char> &result) {
	// Get information about the repeated string
	auto input_str = str.GetDataUnsafe();
	auto size_str = str.GetSize();

	//  Reuse the buffer
	result.clear();
	for (auto remaining = cnt; remaining-- > 0;) {
		result.insert(result.end(), input_str, input_str + size_str);
	}

	return string_t(result.data(), result.size());
}

static void RepeatFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &str_vector = args.data[0];
	auto &cnt_vector = args.data[1];

	vector<char> buffer;
	BinaryExecutor::Execute<string_t, int64_t, string_t>(
	    str_vector, cnt_vector, result, args.size(), [&](string_t str, int64_t cnt) {
		    return StringVector::AddString(result, RepeatScalarFunction(str, cnt, buffer));
	    });
}

void RepeatFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("repeat",                                    // name of the function
	                               {LogicalType::VARCHAR, LogicalType::BIGINT}, // argument list
	                               LogicalType::VARCHAR,                        // return type
	                               RepeatFunction));                            // pointer to function implementation
}

} // namespace duckdb






#include <string.h>
#include <ctype.h>
#include <unordered_map>

namespace duckdb {

static idx_t NextNeedle(const char *input_haystack, idx_t size_haystack, const char *input_needle,
                        const idx_t size_needle) {
	// Needle needs something to proceed
	if (size_needle > 0) {
		// Haystack should be bigger or equal size to the needle
		for (idx_t string_position = 0; (size_haystack - string_position) >= size_needle; ++string_position) {
			// Compare Needle to the Haystack
			if ((memcmp(input_haystack + string_position, input_needle, size_needle) == 0)) {
				return string_position;
			}
		}
	}
	// Did not find the needle
	return size_haystack;
}

static string_t ReplaceScalarFunction(const string_t &haystack, const string_t &needle, const string_t &thread,
                                      vector<char> &result) {
	// Get information about the needle, the haystack and the "thread"
	auto input_haystack = haystack.GetDataUnsafe();
	auto size_haystack = haystack.GetSize();

	auto input_needle = needle.GetDataUnsafe();
	auto size_needle = needle.GetSize();

	auto input_thread = thread.GetDataUnsafe();
	auto size_thread = thread.GetSize();

	//  Reuse the buffer
	result.clear();

	for (;;) {
		//  Append the non-matching characters
		auto string_position = NextNeedle(input_haystack, size_haystack, input_needle, size_needle);
		result.insert(result.end(), input_haystack, input_haystack + string_position);
		input_haystack += string_position;
		size_haystack -= string_position;

		//  Stop when we have read the entire haystack
		if (size_haystack == 0) {
			break;
		}

		//  Replace the matching characters
		result.insert(result.end(), input_thread, input_thread + size_thread);
		input_haystack += size_needle;
		size_haystack -= size_needle;
	}

	return string_t(result.data(), result.size());
}

static void ReplaceFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &haystack_vector = args.data[0];
	auto &needle_vector = args.data[1];
	auto &thread_vector = args.data[2];

	vector<char> buffer;
	TernaryExecutor::Execute<string_t, string_t, string_t, string_t>(
	    haystack_vector, needle_vector, thread_vector, result, args.size(),
	    [&](string_t input_string, string_t needle_string, string_t thread_string) {
		    return StringVector::AddString(result,
		                                   ReplaceScalarFunction(input_string, needle_string, thread_string, buffer));
	    });
}

void ReplaceFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("replace",             // name of the function
	                               {LogicalType::VARCHAR, // argument list
	                                LogicalType::VARCHAR, LogicalType::VARCHAR},
	                               LogicalType::VARCHAR, // return type
	                               ReplaceFunction));    // pointer to function implementation
}

} // namespace duckdb







#include <string.h>

namespace duckdb {

//! Fast ASCII string reverse, returns false if the input data is not ascii
static bool StrReverseASCII(const char *input, idx_t n, char *output) {
	for (idx_t i = 0; i < n; i++) {
		if (input[i] & 0x80) {
			// non-ascii character
			return false;
		}
		output[n - i - 1] = input[i];
	}
	return true;
}

//! Unicode string reverse using grapheme breakers
static void StrReverseUnicode(const char *input, idx_t n, char *output) {
	utf8proc_grapheme_callback(input, n, [&](size_t start, size_t end) {
		memcpy(output + n - end, input + start, end - start);
		return true;
	});
}

struct ReverseOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {
		auto input_data = input.GetDataUnsafe();
		auto input_length = input.GetSize();

		auto target = StringVector::EmptyString(result, input_length);
		auto target_data = target.GetDataWriteable();
		if (!StrReverseASCII(input_data, input_length, target_data)) {
			StrReverseUnicode(input_data, input_length, target_data);
		}
		target.Finalize();
		return target;
	}
};

static void ReverseFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	UnaryExecutor::ExecuteString<string_t, string_t, ReverseOperator>(args.data[0], result, args.size());
}

void ReverseFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("reverse", {LogicalType::VARCHAR}, LogicalType::VARCHAR, ReverseFunction));
}

} // namespace duckdb






namespace duckdb {

static bool StartsWith(const unsigned char *haystack, idx_t haystack_size, const unsigned char *needle,
                       idx_t needle_size) {
	D_ASSERT(needle_size > 0);
	if (needle_size > haystack_size) {
		// needle is bigger than haystack: haystack cannot start with needle
		return false;
	}
	return memcmp(haystack, needle, needle_size) == 0;
}

static bool StartsWith(const string_t &haystack_s, const string_t &needle_s) {
	auto haystack = (const unsigned char *)haystack_s.GetDataUnsafe();
	auto haystack_size = haystack_s.GetSize();
	auto needle = (const unsigned char *)needle_s.GetDataUnsafe();
	auto needle_size = needle_s.GetSize();
	if (needle_size == 0) {
		// empty needle: always true
		return true;
	}
	return StartsWith(haystack, haystack_size, needle, needle_size);
}

struct StartsWithOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA left, TB right) {
		return StartsWith(left, right);
	}
};

void StartsWithFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunction starts_with =
	    ScalarFunction("starts_with", {LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::BOOLEAN,
	                   ScalarFunction::BinaryFunction<string_t, string_t, bool, StartsWithOperator>);
	set.AddFunction(starts_with);
	starts_with.name = "^@";
	set.AddFunction(starts_with);
}

} // namespace duckdb








namespace duckdb {

struct StringSplitInput {
	StringSplitInput(Vector &result_list, Vector &result_child, idx_t offset)
	    : result_list(result_list), result_child(result_child), offset(offset) {
	}

	Vector &result_list;
	Vector &result_child;
	idx_t offset;

	void AddSplit(const char *split_data, idx_t split_size, idx_t list_idx) {
		auto list_entry = offset + list_idx;
		if (list_entry >= ListVector::GetListCapacity(result_list)) {
			ListVector::SetListSize(result_list, offset + list_idx);
			ListVector::Reserve(result_list, ListVector::GetListCapacity(result_list) * 2);
		}
		FlatVector::GetData<string_t>(result_child)[list_entry] =
		    StringVector::AddString(result_child, split_data, split_size);
	}
};

struct RegularStringSplit {
	static idx_t Find(const char *input_data, idx_t input_size, const char *delim_data, idx_t delim_size,
	                  idx_t &match_size, void *data) {
		match_size = delim_size;
		if (delim_size == 0) {
			return 0;
		}
		return ContainsFun::Find((const unsigned char *)input_data, input_size, (const unsigned char *)delim_data,
		                         delim_size);
	}
};

struct ConstantRegexpStringSplit {
	static idx_t Find(const char *input_data, idx_t input_size, const char *delim_data, idx_t delim_size,
	                  idx_t &match_size, void *data) {
		D_ASSERT(data);
		auto regex = (duckdb_re2::RE2 *)data;
		duckdb_re2::StringPiece match;
		if (!regex->Match(duckdb_re2::StringPiece(input_data, input_size), 0, input_size, RE2::UNANCHORED, &match, 1)) {
			return DConstants::INVALID_INDEX;
		}
		match_size = match.size();
		return match.data() - input_data;
	}
};

struct RegexpStringSplit {
	static idx_t Find(const char *input_data, idx_t input_size, const char *delim_data, idx_t delim_size,
	                  idx_t &match_size, void *data) {
		duckdb_re2::RE2 regex(duckdb_re2::StringPiece(delim_data, delim_size));
		if (!regex.ok()) {
			throw InvalidInputException(regex.error());
		}
		return ConstantRegexpStringSplit::Find(input_data, input_size, delim_data, delim_size, match_size, &regex);
	}
};

struct StringSplitter {
	template <class OP>
	static idx_t Split(string_t input, string_t delim, StringSplitInput &state, void *data) {
		auto input_data = input.GetDataUnsafe();
		auto input_size = input.GetSize();
		auto delim_data = delim.GetDataUnsafe();
		auto delim_size = delim.GetSize();
		idx_t list_idx = 0;
		while (input_size > 0) {
			idx_t match_size = 0;
			auto pos = OP::Find(input_data, input_size, delim_data, delim_size, match_size, data);
			if (pos > input_size) {
				break;
			}
			if (match_size == 0 && pos == 0) {
				// special case: 0 length match and pos is 0
				// move to the next character
				for (pos++; pos < input_size; pos++) {
					if (LengthFun::IsCharacter(input_data[pos])) {
						break;
					}
				}
				if (pos == input_size) {
					break;
				}
			}
			D_ASSERT(input_size >= pos + match_size);
			state.AddSplit(input_data, pos, list_idx);

			list_idx++;
			input_data += (pos + match_size);
			input_size -= (pos + match_size);
		}
		state.AddSplit(input_data, input_size, list_idx);
		list_idx++;
		return list_idx;
	}
};

template <class OP>
static void StringSplitExecutor(DataChunk &args, ExpressionState &state, Vector &result, void *data = nullptr) {
	UnifiedVectorFormat input_data;
	args.data[0].ToUnifiedFormat(args.size(), input_data);
	auto inputs = (string_t *)input_data.data;

	UnifiedVectorFormat delim_data;
	args.data[1].ToUnifiedFormat(args.size(), delim_data);
	auto delims = (string_t *)delim_data.data;

	D_ASSERT(result.GetType().id() == LogicalTypeId::LIST);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	ListVector::SetListSize(result, 0);

	auto list_struct_data = FlatVector::GetData<list_entry_t>(result);

	// count all the splits and set up the list entries
	auto &child_entry = ListVector::GetEntry(result);
	auto &result_mask = FlatVector::Validity(result);
	idx_t total_splits = 0;
	for (idx_t i = 0; i < args.size(); i++) {
		auto input_idx = input_data.sel->get_index(i);
		auto delim_idx = delim_data.sel->get_index(i);
		if (!input_data.validity.RowIsValid(input_idx)) {
			result_mask.SetInvalid(i);
			continue;
		}
		StringSplitInput split_input(result, child_entry, total_splits);
		if (!delim_data.validity.RowIsValid(delim_idx)) {
			// delim is NULL: copy the complete entry
			split_input.AddSplit(inputs[input_idx].GetDataUnsafe(), inputs[input_idx].GetSize(), 0);
			list_struct_data[i].length = 1;
			list_struct_data[i].offset = total_splits;
			total_splits++;
			continue;
		}
		auto list_length = StringSplitter::Split<OP>(inputs[input_idx], delims[delim_idx], split_input, data);
		list_struct_data[i].length = list_length;
		list_struct_data[i].offset = total_splits;
		total_splits += list_length;
	}
	ListVector::SetListSize(result, total_splits);
	D_ASSERT(ListVector::GetListSize(result) == total_splits);

	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static void StringSplitFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	StringSplitExecutor<RegularStringSplit>(args, state, result, nullptr);
}

static void StringSplitRegexFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = (BoundFunctionExpression &)state.expr;
	auto &info = (RegexpMatchesBindData &)*func_expr.bind_info;
	if (info.constant_pattern) {
		// fast path: pre-compiled regex
		auto &lstate = (RegexLocalState &)*ExecuteFunctionState::GetFunctionState(state);
		StringSplitExecutor<ConstantRegexpStringSplit>(args, state, result, &lstate.constant_pattern);
	} else {
		// slow path: have to re-compile regex for every row
		StringSplitExecutor<RegexpStringSplit>(args, state, result);
	}
}

void StringSplitFun::RegisterFunction(BuiltinFunctions &set) {
	auto varchar_list_type = LogicalType::LIST(LogicalType::VARCHAR);

	auto regular_fun =
	    ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, varchar_list_type, StringSplitFunction);
	regular_fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	set.AddFunction({"string_split", "str_split", "string_to_array", "split"}, regular_fun);

	ScalarFunctionSet regexp_split("string_split_regex");
	ScalarFunction regex_fun({LogicalType::VARCHAR, LogicalType::VARCHAR}, varchar_list_type, StringSplitRegexFunction,
	                         RegexpMatchesBind, nullptr, nullptr, RegexInitLocalState, LogicalType::INVALID,
	                         FunctionSideEffects::NO_SIDE_EFFECTS, FunctionNullHandling::SPECIAL_HANDLING);
	regexp_split.AddFunction(regex_fun);
	// regexp options
	regex_fun.arguments.emplace_back(LogicalType::VARCHAR);
	regexp_split.AddFunction(regex_fun);
	for (auto &name : {"string_split_regex", "str_split_regex", "regexp_split_to_array"}) {
		regexp_split.name = name;
		set.AddFunction(regexp_split);
	}
}

} // namespace duckdb




namespace duckdb {

bool StripAccentsFun::IsAscii(const char *input, idx_t n) {
	for (idx_t i = 0; i < n; i++) {
		if (input[i] & 0x80) {
			// non-ascii character
			return false;
		}
	}
	return true;
}

struct StripAccentsOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {
		if (StripAccentsFun::IsAscii(input.GetDataUnsafe(), input.GetSize())) {
			return input;
		}

		// non-ascii, perform collation
		auto stripped = utf8proc_remove_accents((const utf8proc_uint8_t *)input.GetDataUnsafe(), input.GetSize());
		auto result_str = StringVector::AddString(result, (const char *)stripped);
		free(stripped);
		return result_str;
	}
};

static void StripAccentsFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(args.ColumnCount() == 1);

	UnaryExecutor::ExecuteString<string_t, string_t, StripAccentsOperator>(args.data[0], result, args.size());
	StringVector::AddHeapReference(result, args.data[0]);
}

ScalarFunction StripAccentsFun::GetFunction() {
	return ScalarFunction("strip_accents", {LogicalType::VARCHAR}, LogicalType::VARCHAR, StripAccentsFunction);
}

void StripAccentsFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(StripAccentsFun::GetFunction());
}

} // namespace duckdb











namespace duckdb {

static const int64_t SUPPORTED_UPPER_BOUND = NumericLimits<uint32_t>::Maximum();
static const int64_t SUPPORTED_LOWER_BOUND = -SUPPORTED_UPPER_BOUND - 1;

static inline void AssertInSupportedRange(idx_t input_size, int64_t offset, int64_t length) {

	if (input_size > (uint64_t)SUPPORTED_UPPER_BOUND) {
		throw OutOfRangeException("Substring input size is too large (> %d)", SUPPORTED_UPPER_BOUND);
	}
	if (offset < SUPPORTED_LOWER_BOUND) {
		throw OutOfRangeException("Substring offset outside of supported range (< %d)", SUPPORTED_LOWER_BOUND);
	}
	if (offset > SUPPORTED_UPPER_BOUND) {
		throw OutOfRangeException("Substring offset outside of supported range (> %d)", SUPPORTED_UPPER_BOUND);
	}
	if (length < SUPPORTED_LOWER_BOUND) {
		throw OutOfRangeException("Substring length outside of supported range (< %d)", SUPPORTED_LOWER_BOUND);
	}
	if (length > SUPPORTED_UPPER_BOUND) {
		throw OutOfRangeException("Substring length outside of supported range (> %d)", SUPPORTED_UPPER_BOUND);
	}
}

string_t SubstringEmptyString(Vector &result) {
	auto result_string = StringVector::EmptyString(result, 0);
	result_string.Finalize();
	return result_string;
}

string_t SubstringSlice(Vector &result, const char *input_data, int64_t offset, int64_t length) {
	auto result_string = StringVector::EmptyString(result, length);
	auto result_data = result_string.GetDataWriteable();
	memcpy(result_data, input_data + offset, length);
	result_string.Finalize();
	return result_string;
}

// compute start and end characters from the given input size and offset/length
bool SubstringStartEnd(int64_t input_size, int64_t offset, int64_t length, int64_t &start, int64_t &end) {
	if (length == 0) {
		return false;
	}
	if (offset > 0) {
		// positive offset: scan from start
		start = MinValue<int64_t>(input_size, offset - 1);
	} else if (offset < 0) {
		// negative offset: scan from end (i.e. start = end + offset)
		start = MaxValue<int64_t>(input_size + offset, 0);
	} else {
		// offset = 0: special case, we start 1 character BEHIND the first character
		start = 0;
		length--;
		if (length <= 0) {
			return false;
		}
	}
	if (length > 0) {
		// positive length: go forward (i.e. end = start + offset)
		end = MinValue<int64_t>(input_size, start + length);
	} else {
		// negative length: go backwards (i.e. end = start, start = start + length)
		end = start;
		start = MaxValue<int64_t>(0, start + length);
	}
	if (start == end) {
		return false;
	}
	D_ASSERT(start < end);
	return true;
}

string_t SubstringASCII(Vector &result, string_t input, int64_t offset, int64_t length) {
	auto input_data = input.GetDataUnsafe();
	auto input_size = input.GetSize();

	AssertInSupportedRange(input_size, offset, length);

	int64_t start, end;
	if (!SubstringStartEnd(input_size, offset, length, start, end)) {
		return SubstringEmptyString(result);
	}
	return SubstringSlice(result, input_data, start, end - start);
}

string_t SubstringFun::SubstringUnicode(Vector &result, string_t input, int64_t offset, int64_t length) {
	auto input_data = input.GetDataUnsafe();
	auto input_size = input.GetSize();

	AssertInSupportedRange(input_size, offset, length);

	if (length == 0) {
		return SubstringEmptyString(result);
	}
	// first figure out which direction we need to scan
	idx_t start_pos;
	idx_t end_pos;
	if (offset < 0) {
		start_pos = 0;
		end_pos = DConstants::INVALID_INDEX;

		// negative offset: scan backwards
		int64_t start, end;

		// we express start and end as unicode codepoints from the back
		offset--;
		if (length < 0) {
			// negative length
			start = -offset - length;
			end = -offset;
		} else {
			// positive length
			start = -offset;
			end = -offset - length;
		}
		if (end <= 0) {
			end_pos = input_size;
		}
		int64_t current_character = 0;
		for (idx_t i = input_size; i > 0; i--) {
			if (LengthFun::IsCharacter(input_data[i - 1])) {
				current_character++;
				if (current_character == start) {
					start_pos = i;
					break;
				} else if (current_character == end) {
					end_pos = i;
				}
			}
		}
		while (!LengthFun::IsCharacter(input_data[start_pos])) {
			start_pos++;
		}
		while (end_pos < input_size && !LengthFun::IsCharacter(input_data[end_pos])) {
			end_pos++;
		}

		if (end_pos == DConstants::INVALID_INDEX) {
			return SubstringEmptyString(result);
		}
	} else {
		start_pos = DConstants::INVALID_INDEX;
		end_pos = input_size;

		// positive offset: scan forwards
		int64_t start, end;

		// we express start and end as unicode codepoints from the front
		offset--;
		if (length < 0) {
			// negative length
			start = MaxValue<int64_t>(0, offset + length);
			end = offset;
		} else {
			// positive length
			start = MaxValue<int64_t>(0, offset);
			end = offset + length;
		}

		int64_t current_character = 0;
		for (idx_t i = 0; i < input_size; i++) {
			if (LengthFun::IsCharacter(input_data[i])) {
				if (current_character == start) {
					start_pos = i;
				} else if (current_character == end) {
					end_pos = i;
					break;
				}
				current_character++;
			}
		}
		if (start_pos == DConstants::INVALID_INDEX || end == 0 || end <= start) {
			return SubstringEmptyString(result);
		}
	}
	D_ASSERT(end_pos >= start_pos);
	// after we have found these, we can slice the substring
	return SubstringSlice(result, input_data, start_pos, end_pos - start_pos);
}

string_t SubstringFun::SubstringGrapheme(Vector &result, string_t input, int64_t offset, int64_t length) {
	auto input_data = input.GetDataUnsafe();
	auto input_size = input.GetSize();

	AssertInSupportedRange(input_size, offset, length);

	// we don't know yet if the substring is ascii, but we assume it is (for now)
	// first get the start and end as if this was an ascii string
	int64_t start, end;
	if (!SubstringStartEnd(input_size, offset, length, start, end)) {
		return SubstringEmptyString(result);
	}

	// now check if all the characters between 0 and end are ascii characters
	// note that we scan one further to check for a potential combining diacritics (e.g. i + diacritic is ï)
	bool is_ascii = true;
	idx_t ascii_end = MinValue<idx_t>(end + 1, input_size);
	for (idx_t i = 0; i < ascii_end; i++) {
		if (input_data[i] & 0x80) {
			// found a non-ascii character: eek
			is_ascii = false;
			break;
		}
	}
	if (is_ascii) {
		// all characters are ascii, we can just slice the substring
		return SubstringSlice(result, input_data, start, end - start);
	}
	// if the characters are not ascii, we need to scan grapheme clusters
	// first figure out which direction we need to scan
	// offset = 0 case is taken care of in SubstringStartEnd
	if (offset < 0) {
		// negative offset, this case is more difficult
		// we first need to count the number of characters in the string
		idx_t num_characters = 0;
		utf8proc_grapheme_callback(input_data, input_size, [&](size_t start, size_t end) {
			num_characters++;
			return true;
		});
		// now call substring start and end again, but with the number of unicode characters this time
		SubstringStartEnd(num_characters, offset, length, start, end);
	}

	// now scan the graphemes of the string to find the positions of the start and end characters
	int64_t current_character = 0;
	idx_t start_pos = DConstants::INVALID_INDEX, end_pos = input_size;
	utf8proc_grapheme_callback(input_data, input_size, [&](size_t gstart, size_t gend) {
		if (current_character == start) {
			start_pos = gstart;
		} else if (current_character == end) {
			end_pos = gstart;
			return false;
		}
		current_character++;
		return true;
	});
	if (start_pos == DConstants::INVALID_INDEX) {
		return SubstringEmptyString(result);
	}
	// after we have found these, we can slice the substring
	return SubstringSlice(result, input_data, start_pos, end_pos - start_pos);
}

struct SubstringUnicodeOp {
	static string_t Substring(Vector &result, string_t input, int64_t offset, int64_t length) {
		return SubstringFun::SubstringUnicode(result, input, offset, length);
	}
};

struct SubstringGraphemeOp {
	static string_t Substring(Vector &result, string_t input, int64_t offset, int64_t length) {
		return SubstringFun::SubstringGrapheme(result, input, offset, length);
	}
};

template <class OP>
static void SubstringFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &input_vector = args.data[0];
	auto &offset_vector = args.data[1];
	if (args.ColumnCount() == 3) {
		auto &length_vector = args.data[2];

		TernaryExecutor::Execute<string_t, int64_t, int64_t, string_t>(
		    input_vector, offset_vector, length_vector, result, args.size(),
		    [&](string_t input_string, int64_t offset, int64_t length) {
			    return OP::Substring(result, input_string, offset, length);
		    });
	} else {
		BinaryExecutor::Execute<string_t, int64_t, string_t>(
		    input_vector, offset_vector, result, args.size(), [&](string_t input_string, int64_t offset) {
			    return OP::Substring(result, input_string, offset, NumericLimits<uint32_t>::Maximum());
		    });
	}
}

static void SubstringFunctionASCII(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &input_vector = args.data[0];
	auto &offset_vector = args.data[1];
	if (args.ColumnCount() == 3) {
		auto &length_vector = args.data[2];

		TernaryExecutor::Execute<string_t, int64_t, int64_t, string_t>(
		    input_vector, offset_vector, length_vector, result, args.size(),
		    [&](string_t input_string, int64_t offset, int64_t length) {
			    return SubstringASCII(result, input_string, offset, length);
		    });
	} else {
		BinaryExecutor::Execute<string_t, int64_t, string_t>(
		    input_vector, offset_vector, result, args.size(), [&](string_t input_string, int64_t offset) {
			    return SubstringASCII(result, input_string, offset, NumericLimits<uint32_t>::Maximum());
		    });
	}
}

static unique_ptr<BaseStatistics> SubstringPropagateStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &expr = input.expr;
	// can only propagate stats if the children have stats
	if (!child_stats[0]) {
		return nullptr;
	}
	// we only care about the stats of the first child (i.e. the string)
	auto &sstats = (StringStatistics &)*child_stats[0];
	if (!sstats.has_unicode) {
		expr.function.function = SubstringFunctionASCII;
	}
	return nullptr;
}

void SubstringFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet substr("substring");
	substr.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT},
	                                  LogicalType::VARCHAR, SubstringFunction<SubstringUnicodeOp>, nullptr, nullptr,
	                                  SubstringPropagateStats));
	substr.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::BIGINT}, LogicalType::VARCHAR,
	                                  SubstringFunction<SubstringUnicodeOp>, nullptr, nullptr,
	                                  SubstringPropagateStats));
	set.AddFunction(substr);
	substr.name = "substr";
	set.AddFunction(substr);

	ScalarFunctionSet substr_grapheme("substring_grapheme");
	substr_grapheme.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::BIGINT, LogicalType::BIGINT},
	                                           LogicalType::VARCHAR, SubstringFunction<SubstringGraphemeOp>, nullptr,
	                                           nullptr, SubstringPropagateStats));
	substr_grapheme.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::BIGINT}, LogicalType::VARCHAR,
	                                           SubstringFunction<SubstringGraphemeOp>, nullptr, nullptr,
	                                           SubstringPropagateStats));
	set.AddFunction(substr_grapheme);
}

} // namespace duckdb





namespace duckdb {

static bool SuffixFunction(const string_t &str, const string_t &suffix);

struct SuffixOperator {
	template <class TA, class TB, class TR>
	static inline TR Operation(TA left, TB right) {
		return SuffixFunction(left, right);
	}
};

static bool SuffixFunction(const string_t &str, const string_t &suffix) {
	auto suffix_size = suffix.GetSize();
	auto str_size = str.GetSize();
	if (suffix_size > str_size) {
		return false;
	}

	auto suffix_data = suffix.GetDataUnsafe();
	auto str_data = str.GetDataUnsafe();
	int32_t suf_idx = suffix_size - 1;
	idx_t str_idx = str_size - 1;
	for (; suf_idx >= 0; --suf_idx, --str_idx) {
		if (suffix_data[suf_idx] != str_data[str_idx]) {
			return false;
		}
	}
	return true;
}

ScalarFunction SuffixFun::GetFunction() {
	return ScalarFunction("suffix",                                     // name of the function
	                      {LogicalType::VARCHAR, LogicalType::VARCHAR}, // argument list
	                      LogicalType::BOOLEAN,                         // return type
	                      ScalarFunction::BinaryFunction<string_t, string_t, bool, SuffixOperator>);
}

void SuffixFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(GetFunction());
}

} // namespace duckdb








#include <string.h>
#include <ctype.h>
#include <unordered_map>
#include <unordered_set>

namespace duckdb {

static string_t TranslateScalarFunction(const string_t &haystack, const string_t &needle, const string_t &thread,
                                        vector<char> &result) {
	// Get information about the haystack, the needle and the "thread"
	auto input_haystack = haystack.GetDataUnsafe();
	auto size_haystack = haystack.GetSize();

	auto input_needle = needle.GetDataUnsafe();
	auto size_needle = needle.GetSize();

	auto input_thread = thread.GetDataUnsafe();
	auto size_thread = thread.GetSize();

	// Reuse the buffer
	result.clear();
	result.reserve(size_haystack);

	idx_t i = 0, j = 0;
	int sz = 0, c_sz = 0;

	// Character to be replaced
	unordered_map<int32_t, int32_t> to_replace;
	while (i < size_needle && j < size_thread) {
		auto codepoint_needle = Utf8Proc::UTF8ToCodepoint(input_needle, sz);
		input_needle += sz;
		i += sz;
		auto codepoint_thread = Utf8Proc::UTF8ToCodepoint(input_thread, sz);
		input_thread += sz;
		j += sz;
		// Ignore unicode character that is existed in to_replace
		if (to_replace.count(codepoint_needle) == 0) {
			to_replace[codepoint_needle] = codepoint_thread;
		}
	}

	// Character to be deleted
	unordered_set<int32_t> to_delete;
	while (i < size_needle) {
		auto codepoint_needle = Utf8Proc::UTF8ToCodepoint(input_needle, sz);
		input_needle += sz;
		i += sz;
		// Add unicode character that will be deleted
		if (to_replace.count(codepoint_needle) == 0) {
			to_delete.insert(codepoint_needle);
		}
	}

	char c[5] = {'\0', '\0', '\0', '\0', '\0'};
	for (i = 0; i < size_haystack; i += sz) {
		auto codepoint_haystack = Utf8Proc::UTF8ToCodepoint(input_haystack, sz);
		if (to_replace.count(codepoint_haystack) != 0) {
			Utf8Proc::CodepointToUtf8(to_replace[codepoint_haystack], c_sz, c);
			result.insert(result.end(), c, c + c_sz);
		} else if (to_delete.count(codepoint_haystack) == 0) {
			result.insert(result.end(), input_haystack, input_haystack + sz);
		}
		input_haystack += sz;
	}

	return string_t(result.data(), result.size());
}

static void TranslateFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &haystack_vector = args.data[0];
	auto &needle_vector = args.data[1];
	auto &thread_vector = args.data[2];

	vector<char> buffer;
	TernaryExecutor::Execute<string_t, string_t, string_t, string_t>(
	    haystack_vector, needle_vector, thread_vector, result, args.size(),
	    [&](string_t input_string, string_t needle_string, string_t thread_string) {
		    return StringVector::AddString(result,
		                                   TranslateScalarFunction(input_string, needle_string, thread_string, buffer));
	    });
}

void TranslateFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(ScalarFunction("translate",           // name of the function
	                               {LogicalType::VARCHAR, // argument list
	                                LogicalType::VARCHAR, LogicalType::VARCHAR},
	                               LogicalType::VARCHAR, // return type
	                               TranslateFunction));  // pointer to function implementation
}

} // namespace duckdb







#include <string.h>

namespace duckdb {

template <bool LTRIM, bool RTRIM>
struct TrimOperator {
	template <class INPUT_TYPE, class RESULT_TYPE>
	static RESULT_TYPE Operation(INPUT_TYPE input, Vector &result) {
		auto data = input.GetDataUnsafe();
		auto size = input.GetSize();

		utf8proc_int32_t codepoint;
		auto str = reinterpret_cast<const utf8proc_uint8_t *>(data);

		// Find the first character that is not left trimmed
		idx_t begin = 0;
		if (LTRIM) {
			while (begin < size) {
				auto bytes = utf8proc_iterate(str + begin, size - begin, &codepoint);
				D_ASSERT(bytes > 0);
				if (utf8proc_category(codepoint) != UTF8PROC_CATEGORY_ZS) {
					break;
				}
				begin += bytes;
			}
		}

		// Find the last character that is not right trimmed
		idx_t end;
		if (RTRIM) {
			end = begin;
			for (auto next = begin; next < size;) {
				auto bytes = utf8proc_iterate(str + next, size - next, &codepoint);
				D_ASSERT(bytes > 0);
				next += bytes;
				if (utf8proc_category(codepoint) != UTF8PROC_CATEGORY_ZS) {
					end = next;
				}
			}
		} else {
			end = size;
		}

		// Copy the trimmed string
		auto target = StringVector::EmptyString(result, end - begin);
		auto output = target.GetDataWriteable();
		memcpy(output, data + begin, end - begin);

		target.Finalize();
		return target;
	}
};

template <bool LTRIM, bool RTRIM>
static void UnaryTrimFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	UnaryExecutor::ExecuteString<string_t, string_t, TrimOperator<LTRIM, RTRIM>>(args.data[0], result, args.size());
}

static void GetIgnoredCodepoints(string_t ignored, unordered_set<utf8proc_int32_t> &ignored_codepoints) {
	auto dataptr = (utf8proc_uint8_t *)ignored.GetDataUnsafe();
	auto size = ignored.GetSize();
	idx_t pos = 0;
	while (pos < size) {
		utf8proc_int32_t codepoint;
		pos += utf8proc_iterate(dataptr + pos, size - pos, &codepoint);
		ignored_codepoints.insert(codepoint);
	}
}

template <bool LTRIM, bool RTRIM>
static void BinaryTrimFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	BinaryExecutor::Execute<string_t, string_t, string_t>(
	    input.data[0], input.data[1], result, input.size(), [&](string_t input, string_t ignored) {
		    auto data = input.GetDataUnsafe();
		    auto size = input.GetSize();

		    unordered_set<utf8proc_int32_t> ignored_codepoints;
		    GetIgnoredCodepoints(ignored, ignored_codepoints);

		    utf8proc_int32_t codepoint;
		    auto str = reinterpret_cast<const utf8proc_uint8_t *>(data);

		    // Find the first character that is not left trimmed
		    idx_t begin = 0;
		    if (LTRIM) {
			    while (begin < size) {
				    auto bytes = utf8proc_iterate(str + begin, size - begin, &codepoint);
				    if (ignored_codepoints.find(codepoint) == ignored_codepoints.end()) {
					    break;
				    }
				    begin += bytes;
			    }
		    }

		    // Find the last character that is not right trimmed
		    idx_t end;
		    if (RTRIM) {
			    end = begin;
			    for (auto next = begin; next < size;) {
				    auto bytes = utf8proc_iterate(str + next, size - next, &codepoint);
				    D_ASSERT(bytes > 0);
				    next += bytes;
				    if (ignored_codepoints.find(codepoint) == ignored_codepoints.end()) {
					    end = next;
				    }
			    }
		    } else {
			    end = size;
		    }

		    // Copy the trimmed string
		    auto target = StringVector::EmptyString(result, end - begin);
		    auto output = target.GetDataWriteable();
		    memcpy(output, data + begin, end - begin);

		    target.Finalize();
		    return target;
	    });
}

void TrimFun::RegisterFunction(BuiltinFunctions &set) {
	ScalarFunctionSet ltrim("ltrim");
	ScalarFunctionSet rtrim("rtrim");
	ScalarFunctionSet trim("trim");

	ltrim.AddFunction(ScalarFunction({LogicalType::VARCHAR}, LogicalType::VARCHAR, UnaryTrimFunction<true, false>));
	rtrim.AddFunction(ScalarFunction({LogicalType::VARCHAR}, LogicalType::VARCHAR, UnaryTrimFunction<false, true>));
	trim.AddFunction(ScalarFunction({LogicalType::VARCHAR}, LogicalType::VARCHAR, UnaryTrimFunction<true, true>));

	ltrim.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                                 BinaryTrimFunction<true, false>));
	rtrim.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                                 BinaryTrimFunction<false, true>));
	trim.AddFunction(ScalarFunction({LogicalType::VARCHAR, LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                                BinaryTrimFunction<true, true>));

	set.AddFunction(ltrim);
	set.AddFunction(rtrim);
	set.AddFunction(trim);
}

} // namespace duckdb





namespace duckdb {

void BuiltinFunctions::RegisterStringFunctions() {
	Register<ReverseFun>();
	Register<LowerFun>();
	Register<UpperFun>();
	Register<StripAccentsFun>();
	Register<ConcatFun>();
	Register<StartsWithFun>();
	Register<ContainsFun>();
	Register<LengthFun>();
	Register<LikeFun>();
	Register<LikeEscapeFun>();
	Register<LpadFun>();
	Register<LeftFun>();
	Register<MD5Fun>();
	Register<RightFun>();
	Register<PrintfFun>();
	Register<RegexpFun>();
	Register<SubstringFun>();
	Register<InstrFun>();
	Register<PrefixFun>();
	Register<RepeatFun>();
	Register<ReplaceFun>();
	Register<RpadFun>();
	Register<TranslateFun>();
	Register<SuffixFun>();
	Register<TrimFun>();
	Register<UnicodeFun>();
	Register<NFCNormalizeFun>();
	Register<StringSplitFun>();
	Register<BarFun>();
	Register<ASCII>();
	Register<CHR>();
	Register<MismatchesFun>();
	Register<LevenshteinFun>();
	Register<JaccardFun>();
	Register<JaroWinklerFun>();

	// blob functions
	Register<Base64Fun>();
	Register<EncodeFun>();

	// bit functions
	Register<GetBitFun>();
	Register<SetBitFun>();
	Register<BitPositionFun>();

	// uuid functions
	Register<UUIDFun>();
}

} // namespace duckdb







namespace duckdb {

struct StructExtractBindData : public FunctionData {
	StructExtractBindData(string key, idx_t index, LogicalType type)
	    : key(std::move(key)), index(index), type(std::move(type)) {
	}

	string key;
	idx_t index;
	LogicalType type;

public:
	unique_ptr<FunctionData> Copy() const override {
		return make_unique<StructExtractBindData>(key, index, type);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const StructExtractBindData &)other_p;
		return key == other.key && index == other.index && type == other.type;
	}
};

static void StructExtractFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = (BoundFunctionExpression &)state.expr;
	auto &info = (StructExtractBindData &)*func_expr.bind_info;

	// this should be guaranteed by the binder
	auto &vec = args.data[0];

	vec.Verify(args.size());
	auto &children = StructVector::GetEntries(vec);
	D_ASSERT(info.index < children.size());
	auto &struct_child = children[info.index];
	result.Reference(*struct_child);
	result.Verify(args.size());
}

static unique_ptr<FunctionData> StructExtractBind(ClientContext &context, ScalarFunction &bound_function,
                                                  vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(bound_function.arguments.size() == 2);
	if (arguments[0]->return_type.id() == LogicalTypeId::UNKNOWN) {
		throw ParameterNotResolvedException();
	}
	D_ASSERT(LogicalTypeId::STRUCT == arguments[0]->return_type.id());
	auto &struct_children = StructType::GetChildTypes(arguments[0]->return_type);
	if (struct_children.empty()) {
		throw InternalException("Can't extract something from an empty struct");
	}
	bound_function.arguments[0] = arguments[0]->return_type;

	auto &key_child = arguments[1];
	if (key_child->HasParameter()) {
		throw ParameterNotResolvedException();
	}

	if (key_child->return_type.id() != LogicalTypeId::VARCHAR || !key_child->IsFoldable()) {
		throw BinderException("Key name for struct_extract needs to be a constant string");
	}
	Value key_val = ExpressionExecutor::EvaluateScalar(context, *key_child.get());
	D_ASSERT(key_val.type().id() == LogicalTypeId::VARCHAR);
	auto &key_str = StringValue::Get(key_val);
	if (key_val.IsNull() || key_str.empty()) {
		throw BinderException("Key name for struct_extract needs to be neither NULL nor empty");
	}
	string key = StringUtil::Lower(key_str);

	LogicalType return_type;
	idx_t key_index = 0;
	bool found_key = false;

	for (size_t i = 0; i < struct_children.size(); i++) {
		auto &child = struct_children[i];
		if (StringUtil::Lower(child.first) == key) {
			found_key = true;
			key_index = i;
			return_type = child.second;
			break;
		}
	}

	if (!found_key) {
		vector<string> candidates;
		candidates.reserve(struct_children.size());
		for (auto &struct_child : struct_children) {
			candidates.push_back(struct_child.first);
		}
		auto closest_settings = StringUtil::TopNLevenshtein(candidates, key);
		auto message = StringUtil::CandidatesMessage(closest_settings, "Candidate Entries");
		throw BinderException("Could not find key \"%s\" in struct\n%s", key, message);
	}

	bound_function.return_type = return_type;
	return make_unique<StructExtractBindData>(key, key_index, return_type);
}

static unique_ptr<BaseStatistics> PropagateStructExtractStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &bind_data = input.bind_data;
	if (!child_stats[0]) {
		return nullptr;
	}
	auto &struct_stats = (StructStatistics &)*child_stats[0];
	auto &info = (StructExtractBindData &)*bind_data;
	if (info.index >= struct_stats.child_stats.size() || !struct_stats.child_stats[info.index]) {
		return nullptr;
	}
	return struct_stats.child_stats[info.index]->Copy();
}

ScalarFunction StructExtractFun::GetFunction() {
	return ScalarFunction("struct_extract", {LogicalTypeId::STRUCT, LogicalType::VARCHAR}, LogicalType::ANY,
	                      StructExtractFunction, StructExtractBind, nullptr, PropagateStructExtractStats);
}

void StructExtractFun::RegisterFunction(BuiltinFunctions &set) {
	// the arguments and return types are actually set in the binder function
	auto fun = GetFunction();
	set.AddFunction(fun);
}

} // namespace duckdb








namespace duckdb {

static void StructInsertFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &starting_vec = args.data[0];

	starting_vec.Verify(args.size());

	auto &starting_child_entries = StructVector::GetEntries(starting_vec);
	auto &result_child_entries = StructVector::GetEntries(result);

	// Assign the starting vector entries to the result vector
	for (size_t i = 0; i < starting_child_entries.size(); i++) {
		auto &starting_child = starting_child_entries[i];
		result_child_entries[i]->Reference(*starting_child);
	}

	// Assign the new entries to the result vector
	for (size_t i = 1; i < args.ColumnCount(); i++) {
		result_child_entries[starting_child_entries.size() + i - 1]->Reference(args.data[i]);
	}

	result.Verify(args.size());

	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static unique_ptr<FunctionData> StructInsertBind(ClientContext &context, ScalarFunction &bound_function,
                                                 vector<unique_ptr<Expression>> &arguments) {
	case_insensitive_set_t name_collision_set;

	if (arguments.empty()) {
		throw Exception("Missing required arguments for struct_insert function.");
	}

	if (LogicalTypeId::STRUCT != arguments[0]->return_type.id()) {
		throw Exception("The first argument to struct_insert must be a STRUCT");
	}

	if (arguments.size() < 2) {
		throw Exception("Can't insert nothing into a struct");
	}

	child_list_t<LogicalType> new_struct_children;

	auto &existing_struct_children = StructType::GetChildTypes(arguments[0]->return_type);

	for (size_t i = 0; i < existing_struct_children.size(); i++) {
		auto &child = existing_struct_children[i];
		name_collision_set.insert(child.first);
		new_struct_children.push_back(make_pair(child.first, child.second));
	}

	// Loop through the additional arguments (name/value pairs)
	for (idx_t i = 1; i < arguments.size(); i++) {
		auto &child = arguments[i];
		if (child->alias.empty() && bound_function.name == "struct_insert") {
			throw BinderException("Need named argument for struct insert, e.g. STRUCT_PACK(a := b)");
		}
		if (name_collision_set.find(child->alias) != name_collision_set.end()) {
			throw BinderException("Duplicate struct entry name \"%s\"", child->alias);
		}
		name_collision_set.insert(child->alias);
		new_struct_children.push_back(make_pair(child->alias, arguments[i]->return_type));
	}

	// this is more for completeness reasons
	bound_function.return_type = LogicalType::STRUCT(std::move(new_struct_children));
	return make_unique<VariableReturnBindData>(bound_function.return_type);
}

unique_ptr<BaseStatistics> StructInsertStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &expr = input.expr;
	if (child_stats.empty() || !child_stats[0]) {
		return nullptr;
	}
	auto &existing_struct_stats = (StructStatistics &)*child_stats[0];
	auto new_struct_stats = make_unique<StructStatistics>(expr.return_type);

	for (idx_t i = 0; i < existing_struct_stats.child_stats.size(); i++) {
		new_struct_stats->child_stats[i] =
		    existing_struct_stats.child_stats[i] ? existing_struct_stats.child_stats[i]->Copy() : nullptr;
	}

	auto offset = new_struct_stats->child_stats.size() - child_stats.size();
	for (idx_t i = 1; i < child_stats.size(); i++) {
		new_struct_stats->child_stats[offset + i] = child_stats[i] ? child_stats[i]->Copy() : nullptr;
	}
	return std::move(new_struct_stats);
}

void StructInsertFun::RegisterFunction(BuiltinFunctions &set) {
	// the arguments and return types are actually set in the binder function
	ScalarFunction fun("struct_insert", {}, LogicalTypeId::STRUCT, StructInsertFunction, StructInsertBind, nullptr,
	                   StructInsertStats);
	fun.varargs = LogicalType::ANY;
	fun.serialize = VariableReturnBindData::Serialize;
	fun.deserialize = VariableReturnBindData::Deserialize;
	set.AddFunction(fun);
}

} // namespace duckdb








namespace duckdb {

static void StructPackFunction(DataChunk &args, ExpressionState &state, Vector &result) {
#ifdef DEBUG
	auto &func_expr = (BoundFunctionExpression &)state.expr;
	auto &info = (VariableReturnBindData &)*func_expr.bind_info;
	// this should never happen if the binder below is sane
	D_ASSERT(args.ColumnCount() == StructType::GetChildTypes(info.stype).size());
#endif
	bool all_const = true;
	auto &child_entries = StructVector::GetEntries(result);
	for (size_t i = 0; i < args.ColumnCount(); i++) {
		if (args.data[i].GetVectorType() != VectorType::CONSTANT_VECTOR) {
			all_const = false;
		}
		// same holds for this
		child_entries[i]->Reference(args.data[i]);
	}
	result.SetVectorType(all_const ? VectorType::CONSTANT_VECTOR : VectorType::FLAT_VECTOR);

	result.Verify(args.size());
}

static unique_ptr<FunctionData> StructPackBind(ClientContext &context, ScalarFunction &bound_function,
                                               vector<unique_ptr<Expression>> &arguments) {
	case_insensitive_set_t name_collision_set;

	// collect names and deconflict, construct return type
	if (arguments.empty()) {
		throw Exception("Can't pack nothing into a struct");
	}
	child_list_t<LogicalType> struct_children;
	for (idx_t i = 0; i < arguments.size(); i++) {
		auto &child = arguments[i];
		if (child->alias.empty() && bound_function.name == "struct_pack") {
			throw BinderException("Need named argument for struct pack, e.g. STRUCT_PACK(a := b)");
		}
		if (child->alias.empty() && bound_function.name == "row") {
			child->alias = "v" + std::to_string(i + 1);
		}
		if (name_collision_set.find(child->alias) != name_collision_set.end()) {
			throw BinderException("Duplicate struct entry name \"%s\"", child->alias);
		}
		name_collision_set.insert(child->alias);
		struct_children.push_back(make_pair(child->alias, arguments[i]->return_type));
	}

	// this is more for completeness reasons
	bound_function.return_type = LogicalType::STRUCT(std::move(struct_children));
	return make_unique<VariableReturnBindData>(bound_function.return_type);
}

unique_ptr<BaseStatistics> StructPackStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &expr = input.expr;
	auto struct_stats = make_unique<StructStatistics>(expr.return_type);
	D_ASSERT(child_stats.size() == struct_stats->child_stats.size());
	for (idx_t i = 0; i < struct_stats->child_stats.size(); i++) {
		struct_stats->child_stats[i] = child_stats[i] ? child_stats[i]->Copy() : nullptr;
	}
	return std::move(struct_stats);
}

void StructPackFun::RegisterFunction(BuiltinFunctions &set) {
	// the arguments and return types are actually set in the binder function
	ScalarFunction fun("struct_pack", {}, LogicalTypeId::STRUCT, StructPackFunction, StructPackBind, nullptr,
	                   StructPackStats);
	fun.varargs = LogicalType::ANY;
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	fun.serialize = VariableReturnBindData::Serialize;
	fun.deserialize = VariableReturnBindData::Deserialize;
	set.AddFunction(fun);
	fun.name = "row";
	set.AddFunction(fun);
}

} // namespace duckdb









namespace duckdb {

// aggregate state export
struct ExportAggregateBindData : public FunctionData {
	AggregateFunction aggr;
	idx_t state_size;

	explicit ExportAggregateBindData(AggregateFunction aggr_p, idx_t state_size_p)
	    : aggr(std::move(aggr_p)), state_size(state_size_p) {
	}

	unique_ptr<FunctionData> Copy() const override {
		return make_unique<ExportAggregateBindData>(aggr, state_size);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const ExportAggregateBindData &)other_p;
		return aggr == other.aggr && state_size == other.state_size;
	}

	static ExportAggregateBindData &GetFrom(ExpressionState &state) {
		auto &func_expr = (BoundFunctionExpression &)state.expr;
		return (ExportAggregateBindData &)*func_expr.bind_info;
	}
};

struct CombineState : public FunctionLocalState {
	idx_t state_size;

	unique_ptr<data_t[]> state_buffer0, state_buffer1;
	Vector state_vector0, state_vector1;

	explicit CombineState(idx_t state_size_p)
	    : state_size(state_size_p), state_buffer0(unique_ptr<data_t[]>(new data_t[state_size_p])),
	      state_buffer1(unique_ptr<data_t[]>(new data_t[state_size_p])),
	      state_vector0(Value::POINTER((uintptr_t)state_buffer0.get())),
	      state_vector1(Value::POINTER((uintptr_t)state_buffer1.get())) {
	}
};

static unique_ptr<FunctionLocalState> InitCombineState(ExpressionState &state, const BoundFunctionExpression &expr,
                                                       FunctionData *bind_data_p) {
	auto &bind_data = *(ExportAggregateBindData *)bind_data_p;
	return make_unique<CombineState>(bind_data.state_size);
}

struct FinalizeState : public FunctionLocalState {
	idx_t state_size;
	unique_ptr<data_t[]> state_buffer;
	Vector addresses;

	explicit FinalizeState(idx_t state_size_p)
	    : state_size(state_size_p),
	      state_buffer(unique_ptr<data_t[]>(new data_t[STANDARD_VECTOR_SIZE * AlignValue(state_size_p)])),
	      addresses(LogicalType::POINTER) {
	}
};

static unique_ptr<FunctionLocalState> InitFinalizeState(ExpressionState &state, const BoundFunctionExpression &expr,
                                                        FunctionData *bind_data_p) {
	auto &bind_data = *(ExportAggregateBindData *)bind_data_p;
	return make_unique<FinalizeState>(bind_data.state_size);
}

static void AggregateStateFinalize(DataChunk &input, ExpressionState &state_p, Vector &result) {
	auto &bind_data = ExportAggregateBindData::GetFrom(state_p);
	auto &local_state = (FinalizeState &)*((ExecuteFunctionState &)state_p).local_state;

	D_ASSERT(bind_data.state_size == bind_data.aggr.state_size());
	D_ASSERT(input.data.size() == 1);
	D_ASSERT(input.data[0].GetType().id() == LogicalTypeId::AGGREGATE_STATE);
	auto aligned_state_size = AlignValue(bind_data.state_size);

	auto state_vec_ptr = FlatVector::GetData<data_ptr_t>(local_state.addresses);

	UnifiedVectorFormat state_data;
	input.data[0].ToUnifiedFormat(input.size(), state_data);
	for (idx_t i = 0; i < input.size(); i++) {
		auto state_idx = state_data.sel->get_index(i);
		auto state_entry = &((string_t *)state_data.data)[state_idx];
		auto target_ptr = (const char *)local_state.state_buffer.get() + aligned_state_size * i;

		if (state_data.validity.RowIsValid(state_idx)) {
			D_ASSERT(state_entry->GetSize() == bind_data.state_size);
			memcpy((void *)target_ptr, state_entry->GetDataUnsafe(), bind_data.state_size);
		} else {
			// create a dummy state because finalize does not understand NULLs in its input
			// we put the NULL back in explicitly below
			bind_data.aggr.initialize((data_ptr_t)target_ptr);
		}
		state_vec_ptr[i] = (data_ptr_t)target_ptr;
	}

	AggregateInputData aggr_input_data(nullptr, Allocator::DefaultAllocator());
	bind_data.aggr.finalize(local_state.addresses, aggr_input_data, result, input.size(), 0);

	for (idx_t i = 0; i < input.size(); i++) {
		auto state_idx = state_data.sel->get_index(i);
		if (!state_data.validity.RowIsValid(state_idx)) {
			FlatVector::SetNull(result, i, true);
		}
	}
}

static void AggregateStateCombine(DataChunk &input, ExpressionState &state_p, Vector &result) {
	auto &bind_data = ExportAggregateBindData::GetFrom(state_p);
	auto &local_state = (CombineState &)*((ExecuteFunctionState &)state_p).local_state;

	D_ASSERT(bind_data.state_size == bind_data.aggr.state_size());

	D_ASSERT(input.data.size() == 2);
	D_ASSERT(input.data[0].GetType().id() == LogicalTypeId::AGGREGATE_STATE);
	D_ASSERT(input.data[0].GetType() == result.GetType());

	if (input.data[0].GetType().InternalType() != input.data[1].GetType().InternalType()) {
		throw IOException("Aggregate state combine type mismatch, expect %s, got %s",
		                  input.data[0].GetType().ToString(), input.data[1].GetType().ToString());
	}

	UnifiedVectorFormat state0_data, state1_data;
	input.data[0].ToUnifiedFormat(input.size(), state0_data);
	input.data[1].ToUnifiedFormat(input.size(), state1_data);

	auto result_ptr = FlatVector::GetData<string_t>(result);

	for (idx_t i = 0; i < input.size(); i++) {
		auto state0_idx = state0_data.sel->get_index(i);
		auto state1_idx = state1_data.sel->get_index(i);

		auto &state0 = ((string_t *)state0_data.data)[state0_idx];
		auto &state1 = ((string_t *)state1_data.data)[state1_idx];

		// if both are NULL, we return NULL. If either of them is not, the result is that one
		if (!state0_data.validity.RowIsValid(state0_idx) && !state1_data.validity.RowIsValid(state1_idx)) {
			FlatVector::SetNull(result, i, true);
			continue;
		}
		if (state0_data.validity.RowIsValid(state0_idx) && !state1_data.validity.RowIsValid(state1_idx)) {
			result_ptr[i] =
			    StringVector::AddStringOrBlob(result, (const char *)state0.GetDataUnsafe(), bind_data.state_size);
			continue;
		}
		if (!state0_data.validity.RowIsValid(state0_idx) && state1_data.validity.RowIsValid(state1_idx)) {
			result_ptr[i] =
			    StringVector::AddStringOrBlob(result, (const char *)state1.GetDataUnsafe(), bind_data.state_size);
			continue;
		}

		// we actually have to combine
		if (state0.GetSize() != bind_data.state_size || state1.GetSize() != bind_data.state_size) {
			throw IOException("Aggregate state size mismatch, expect %llu, got %llu and %llu", bind_data.state_size,
			                  state0.GetSize(), state1.GetSize());
		}

		memcpy(local_state.state_buffer0.get(), state0.GetDataUnsafe(), bind_data.state_size);
		memcpy(local_state.state_buffer1.get(), state1.GetDataUnsafe(), bind_data.state_size);

		AggregateInputData aggr_input_data(nullptr, Allocator::DefaultAllocator());
		bind_data.aggr.combine(local_state.state_vector0, local_state.state_vector1, aggr_input_data, 1);

		result_ptr[i] =
		    StringVector::AddStringOrBlob(result, (const char *)local_state.state_buffer1.get(), bind_data.state_size);
	}
}

static unique_ptr<FunctionData> BindAggregateState(ClientContext &context, ScalarFunction &bound_function,
                                                   vector<unique_ptr<Expression>> &arguments) {

	// grab the aggregate type and bind the aggregate again

	// the aggregate name and types are in the logical type of the aggregate state, make sure its sane
	auto &arg_return_type = arguments[0]->return_type;
	for (auto &arg_type : bound_function.arguments) {
		arg_type = arg_return_type;
	}

	if (arg_return_type.id() != LogicalTypeId::AGGREGATE_STATE) {
		throw BinderException("Can only FINALIZE aggregate state, not %s", arg_return_type.ToString());
	}
	// combine
	if (arguments.size() == 2 && arguments[0]->return_type != arguments[1]->return_type &&
	    arguments[1]->return_type.id() != LogicalTypeId::BLOB) {
		throw BinderException("Cannot COMBINE aggregate states from different functions, %s <> %s",
		                      arguments[0]->return_type.ToString(), arguments[1]->return_type.ToString());
	}

	// following error states are only reachable when someone messes up creating the state_type
	// which is impossible from SQL

	auto state_type = AggregateStateType::GetStateType(arg_return_type);

	// now we can look up the function in the catalog again and bind it
	auto func = Catalog::GetSystemCatalog(context).GetEntry(context, CatalogType::SCALAR_FUNCTION_ENTRY, DEFAULT_SCHEMA,
	                                                        state_type.function_name);
	if (func->type != CatalogType::AGGREGATE_FUNCTION_ENTRY) {
		throw InternalException("Could not find aggregate %s", state_type.function_name);
	}
	auto aggr = (AggregateFunctionCatalogEntry *)func;

	string error;

	FunctionBinder function_binder(context);
	idx_t best_function =
	    function_binder.BindFunction(aggr->name, aggr->functions, state_type.bound_argument_types, error);
	if (best_function == DConstants::INVALID_INDEX) {
		throw InternalException("Could not re-bind exported aggregate %s: %s", state_type.function_name, error);
	}
	auto bound_aggr = aggr->functions.GetFunctionByOffset(best_function);
	if (bound_aggr.bind) {
		// FIXME: this is really hacky
		// but the aggregate state export needs a rework around how it handles more complex aggregates anyway
		vector<unique_ptr<Expression>> args;
		args.reserve(state_type.bound_argument_types.size());
		for (auto &arg_type : state_type.bound_argument_types) {
			args.push_back(make_unique<BoundConstantExpression>(Value(arg_type)));
		}
		auto bind_info = bound_aggr.bind(context, bound_aggr, args);
		if (bind_info) {
			throw BinderException("Aggregate function with bind info not supported yet in aggregate state export");
		}
	}

	if (bound_aggr.return_type != state_type.return_type || bound_aggr.arguments != state_type.bound_argument_types) {
		throw InternalException("Type mismatch for exported aggregate %s", state_type.function_name);
	}

	if (bound_function.name == "finalize") {
		bound_function.return_type = bound_aggr.return_type;
	} else {
		D_ASSERT(bound_function.name == "combine");
		bound_function.return_type = arg_return_type;
	}

	return make_unique<ExportAggregateBindData>(bound_aggr, bound_aggr.state_size());
}

static void ExportAggregateFinalize(Vector &state, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                                    idx_t offset) {
	D_ASSERT(offset == 0);
	auto bind_data = (ExportAggregateFunctionBindData *)aggr_input_data.bind_data;
	auto state_size = bind_data->aggregate->function.state_size();
	auto blob_ptr = FlatVector::GetData<string_t>(result);
	auto addresses_ptr = FlatVector::GetData<data_ptr_t>(state);
	for (idx_t row_idx = 0; row_idx < count; row_idx++) {
		auto data_ptr = addresses_ptr[row_idx];
		blob_ptr[row_idx] = StringVector::AddStringOrBlob(result, (const char *)data_ptr, state_size);
	}
}

ExportAggregateFunctionBindData::ExportAggregateFunctionBindData(unique_ptr<Expression> aggregate_p) {
	D_ASSERT(aggregate_p->type == ExpressionType::BOUND_AGGREGATE);
	aggregate = unique_ptr<BoundAggregateExpression>((BoundAggregateExpression *)aggregate_p.release());
}

unique_ptr<FunctionData> ExportAggregateFunctionBindData::Copy() const {
	return make_unique<ExportAggregateFunctionBindData>(aggregate->Copy());
}

bool ExportAggregateFunctionBindData::Equals(const FunctionData &other_p) const {
	auto &other = (const ExportAggregateFunctionBindData &)other_p;
	return aggregate->Equals(other.aggregate.get());
}

static void ExportStateAggregateSerialize(FieldWriter &writer, const FunctionData *bind_data_p,
                                          const AggregateFunction &function) {
	throw NotImplementedException("FIXME: export state serialize");
}
static unique_ptr<FunctionData> ExportStateAggregateDeserialize(ClientContext &context, FieldReader &reader,
                                                                AggregateFunction &bound_function) {
	throw NotImplementedException("FIXME: export state deserialize");
}

static void ExportStateScalarSerialize(FieldWriter &writer, const FunctionData *bind_data_p,
                                       const ScalarFunction &function) {
	throw NotImplementedException("FIXME: export state serialize");
}
static unique_ptr<FunctionData> ExportStateScalarDeserialize(ClientContext &context, FieldReader &reader,
                                                             ScalarFunction &bound_function) {
	throw NotImplementedException("FIXME: export state deserialize");
}

unique_ptr<BoundAggregateExpression>
ExportAggregateFunction::Bind(unique_ptr<BoundAggregateExpression> child_aggregate) {
	auto &bound_function = child_aggregate->function;
	if (!bound_function.combine) {
		throw BinderException("Cannot use EXPORT_STATE for non-combinable function %s", bound_function.name);
	}
	if (bound_function.bind) {
		throw BinderException("Cannot use EXPORT_STATE on aggregate functions with custom binders");
	}
	if (bound_function.destructor) {
		throw BinderException("Cannot use EXPORT_STATE on aggregate functions with custom destructors");
	}
	// this should be required
	D_ASSERT(bound_function.state_size);
	D_ASSERT(bound_function.finalize);

	D_ASSERT(child_aggregate->function.return_type.id() != LogicalTypeId::INVALID);
#ifdef DEBUG
	for (auto &arg_type : child_aggregate->function.arguments) {
		D_ASSERT(arg_type.id() != LogicalTypeId::INVALID);
	}
#endif
	auto export_bind_data = make_unique<ExportAggregateFunctionBindData>(child_aggregate->Copy());
	aggregate_state_t state_type(child_aggregate->function.name, child_aggregate->function.return_type,
	                             child_aggregate->function.arguments);
	auto return_type = LogicalType::AGGREGATE_STATE(std::move(state_type));

	auto export_function =
	    AggregateFunction("aggregate_state_export_" + bound_function.name, bound_function.arguments, return_type,
	                      bound_function.state_size, bound_function.initialize, bound_function.update,
	                      bound_function.combine, ExportAggregateFinalize, bound_function.simple_update,
	                      /* can't bind this again */ nullptr, /* no dynamic state yet */ nullptr,
	                      /* can't propagate statistics */ nullptr, nullptr);
	export_function.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	export_function.serialize = ExportStateAggregateSerialize;
	export_function.deserialize = ExportStateAggregateDeserialize;

	return make_unique<BoundAggregateExpression>(export_function, std::move(child_aggregate->children),
	                                             std::move(child_aggregate->filter), std::move(export_bind_data),
	                                             child_aggregate->aggr_type);
}

ScalarFunction ExportAggregateFunction::GetFinalize() {
	auto result = ScalarFunction("finalize", {LogicalTypeId::AGGREGATE_STATE}, LogicalTypeId::INVALID,
	                             AggregateStateFinalize, BindAggregateState, nullptr, nullptr, InitFinalizeState);
	result.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	result.serialize = ExportStateScalarSerialize;
	result.deserialize = ExportStateScalarDeserialize;
	return result;
}

ScalarFunction ExportAggregateFunction::GetCombine() {
	auto result =
	    ScalarFunction("combine", {LogicalTypeId::AGGREGATE_STATE, LogicalTypeId::ANY}, LogicalTypeId::AGGREGATE_STATE,
	                   AggregateStateCombine, BindAggregateState, nullptr, nullptr, InitCombineState);
	result.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	result.serialize = ExportStateScalarSerialize;
	result.deserialize = ExportStateScalarDeserialize;
	return result;
}

} // namespace duckdb










namespace duckdb {

// current_query
static void CurrentQueryFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	Value val(state.GetContext().GetCurrentQuery());
	result.Reference(val);
}

// current_schema
static void CurrentSchemaFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	Value val(ClientData::Get(state.GetContext()).catalog_search_path->GetDefault().schema);
	result.Reference(val);
}

// current_database
static void CurrentDatabaseFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	Value val(DatabaseManager::GetDefaultDatabase(state.GetContext()));
	result.Reference(val);
}

// current_schemas
static void CurrentSchemasFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	if (!input.AllConstant()) {
		throw NotImplementedException("current_schemas requires a constant input");
	}
	if (ConstantVector::IsNull(input.data[0])) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		ConstantVector::SetNull(result, true);
		return;
	}
	auto implicit_schemas = *ConstantVector::GetData<bool>(input.data[0]);
	vector<Value> schema_list;
	auto &catalog_search_path = ClientData::Get(state.GetContext()).catalog_search_path;
	auto &search_path = implicit_schemas ? catalog_search_path->Get() : catalog_search_path->GetSetPaths();
	std::transform(search_path.begin(), search_path.end(), std::back_inserter(schema_list),
	               [](const CatalogSearchEntry &s) -> Value { return Value(s.schema); });

	auto val = Value::LIST(LogicalType::VARCHAR, schema_list);
	result.Reference(val);
}

// txid_current
static void TransactionIdCurrent(DataChunk &input, ExpressionState &state, Vector &result) {
	auto &context = state.GetContext();
	auto &catalog = Catalog::GetCatalog(context, DatabaseManager::GetDefaultDatabase(context));
	auto &transaction = DuckTransaction::Get(context, catalog);
	auto val = Value::BIGINT(transaction.start_time);
	result.Reference(val);
}

// version
static void VersionFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	auto val = Value(DuckDB::LibraryVersion());
	result.Reference(val);
}

void SystemFun::RegisterFunction(BuiltinFunctions &set) {
	auto varchar_list_type = LogicalType::LIST(LogicalType::VARCHAR);

	ScalarFunction current_query("current_query", {}, LogicalType::VARCHAR, CurrentQueryFunction);
	current_query.side_effects = FunctionSideEffects::HAS_SIDE_EFFECTS;
	set.AddFunction(current_query);
	set.AddFunction(ScalarFunction("current_schema", {}, LogicalType::VARCHAR, CurrentSchemaFunction));
	set.AddFunction(ScalarFunction("current_database", {}, LogicalType::VARCHAR, CurrentDatabaseFunction));
	set.AddFunction(
	    ScalarFunction("current_schemas", {LogicalType::BOOLEAN}, varchar_list_type, CurrentSchemasFunction));
	set.AddFunction(ScalarFunction("txid_current", {}, LogicalType::BIGINT, TransactionIdCurrent));
	set.AddFunction(ScalarFunction("version", {}, LogicalType::VARCHAR, VersionFunction));
	set.AddFunction(ExportAggregateFunction::GetCombine());
	set.AddFunction(ExportAggregateFunction::GetFinalize());
}

} // namespace duckdb




namespace duckdb {

void BuiltinFunctions::RegisterTrigonometricsFunctions() {
	Register<SinFun>();
	Register<CosFun>();
	Register<TanFun>();
	Register<AsinFun>();
	Register<AcosFun>();
	Register<AtanFun>();
	Register<CotFun>();
	Register<Atan2Fun>();
}

} // namespace duckdb






namespace duckdb {

struct UnionExtractBindData : public FunctionData {
	UnionExtractBindData(string key, idx_t index, LogicalType type)
	    : key(std::move(key)), index(index), type(std::move(type)) {
	}

	string key;
	idx_t index;
	LogicalType type;

public:
	unique_ptr<FunctionData> Copy() const override {
		return make_unique<UnionExtractBindData>(key, index, type);
	}
	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const UnionExtractBindData &)other_p;
		return key == other.key && index == other.index && type == other.type;
	}
};

static void UnionExtractFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = (BoundFunctionExpression &)state.expr;
	auto &info = (UnionExtractBindData &)*func_expr.bind_info;

	// this should be guaranteed by the binder
	auto &vec = args.data[0];
	vec.Verify(args.size());

	D_ASSERT(info.index < UnionType::GetMemberCount(vec.GetType()));
	auto &member = UnionVector::GetMember(vec, info.index);
	result.Reference(member);
	result.Verify(args.size());
}

static unique_ptr<FunctionData> UnionExtractBind(ClientContext &context, ScalarFunction &bound_function,
                                                 vector<unique_ptr<Expression>> &arguments) {
	D_ASSERT(bound_function.arguments.size() == 2);
	if (arguments[0]->return_type.id() == LogicalTypeId::UNKNOWN) {
		throw ParameterNotResolvedException();
	}
	D_ASSERT(LogicalTypeId::UNION == arguments[0]->return_type.id());
	idx_t union_member_count = UnionType::GetMemberCount(arguments[0]->return_type);
	if (union_member_count == 0) {
		throw InternalException("Can't extract something from an empty union");
	}
	bound_function.arguments[0] = arguments[0]->return_type;

	auto &key_child = arguments[1];
	if (key_child->HasParameter()) {
		throw ParameterNotResolvedException();
	}

	if (key_child->return_type.id() != LogicalTypeId::VARCHAR || !key_child->IsFoldable()) {
		throw BinderException("Key name for union_extract needs to be a constant string");
	}
	Value key_val = ExpressionExecutor::EvaluateScalar(context, *key_child.get());
	D_ASSERT(key_val.type().id() == LogicalTypeId::VARCHAR);
	auto &key_str = StringValue::Get(key_val);
	if (key_val.IsNull() || key_str.empty()) {
		throw BinderException("Key name for union_extract needs to be neither NULL nor empty");
	}
	string key = StringUtil::Lower(key_str);

	LogicalType return_type;
	idx_t key_index = 0;
	bool found_key = false;

	for (size_t i = 0; i < union_member_count; i++) {
		auto &member_name = UnionType::GetMemberName(arguments[0]->return_type, i);
		if (StringUtil::Lower(member_name) == key) {
			found_key = true;
			key_index = i;
			return_type = UnionType::GetMemberType(arguments[0]->return_type, i);
			break;
		}
	}

	if (!found_key) {
		vector<string> candidates;
		candidates.reserve(union_member_count);
		for (idx_t i = 0; i < union_member_count; i++) {
			candidates.push_back(UnionType::GetMemberName(arguments[0]->return_type, i));
		}
		auto closest_settings = StringUtil::TopNLevenshtein(candidates, key);
		auto message = StringUtil::CandidatesMessage(closest_settings, "Candidate Entries");
		throw BinderException("Could not find key \"%s\" in union\n%s", key, message);
	}

	bound_function.return_type = return_type;
	return make_unique<UnionExtractBindData>(key, key_index, return_type);
}

void UnionExtractFun::RegisterFunction(BuiltinFunctions &set) {
	// the arguments and return types are actually set in the binder function
	auto fun = ScalarFunction("union_extract", {LogicalTypeId::UNION, LogicalType::VARCHAR}, LogicalType::ANY,
	                          UnionExtractFunction, UnionExtractBind, nullptr, nullptr);

	ScalarFunctionSet extract("union_extract");
	extract.AddFunction(fun);
	set.AddFunction(extract);
}

} // namespace duckdb






namespace duckdb {

static unique_ptr<FunctionData> UnionTagBind(ClientContext &context, ScalarFunction &bound_function,
                                             vector<unique_ptr<Expression>> &arguments) {

	if (arguments.empty()) {
		throw BinderException("Missing required arguments for union_tag function.");
	}

	if (LogicalTypeId::UNKNOWN == arguments[0]->return_type.id()) {
		throw ParameterNotResolvedException();
	}

	if (LogicalTypeId::UNION != arguments[0]->return_type.id()) {
		throw BinderException("First argument to union_tag function must be a union type.");
	}

	if (arguments.size() > 1) {
		throw BinderException("Too many arguments, union_tag takes at most one argument.");
	}

	auto member_count = UnionType::GetMemberCount(arguments[0]->return_type);
	if (member_count == 0) {
		// this should never happen, empty unions are not allowed
		throw InternalException("Can't get tags from an empty union");
	}

	bound_function.arguments[0] = arguments[0]->return_type;

	auto varchar_vector = Vector(LogicalType::VARCHAR, member_count);
	for (idx_t i = 0; i < member_count; i++) {
		auto str = string_t(UnionType::GetMemberName(arguments[0]->return_type, i));
		FlatVector::GetData<string_t>(varchar_vector)[i] =
		    str.IsInlined() ? str : StringVector::AddString(varchar_vector, str);
	}
	auto enum_type = LogicalType::ENUM("", varchar_vector, member_count);
	bound_function.return_type = enum_type;

	return nullptr;
}

static void UnionTagFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	D_ASSERT(result.GetType().id() == LogicalTypeId::ENUM);
	result.Reinterpret(UnionVector::GetTags(args.data[0]));
}

void UnionTagFun::RegisterFunction(BuiltinFunctions &set) {
	auto fun = ScalarFunction("union_tag", {LogicalTypeId::UNION}, LogicalTypeId::ANY, UnionTagFunction, UnionTagBind,
	                          nullptr, nullptr); // TODO: Statistics?

	ScalarFunctionSet union_tag("union_tag");
	union_tag.AddFunction(fun);
	set.AddFunction(union_tag);
}

} // namespace duckdb






namespace duckdb {

UnionInvalidReason CheckUnionValidity(Vector &vector, idx_t count, const SelectionVector &sel) {
	D_ASSERT(vector.GetType().id() == LogicalTypeId::UNION);
	auto member_count = UnionType::GetMemberCount(vector.GetType());
	if (member_count == 0) {
		return UnionInvalidReason::NO_MEMBERS;
	}

	UnifiedVectorFormat union_vdata;
	vector.ToUnifiedFormat(count, union_vdata);

	UnifiedVectorFormat tags_vdata;
	auto &tag_vector = UnionVector::GetTags(vector);
	tag_vector.ToUnifiedFormat(count, tags_vdata);

	// check that only one member is valid at a time
	for (idx_t row_idx = 0; row_idx < count; row_idx++) {
		auto union_mapped_row_idx = sel.get_index(row_idx);
		if (!union_vdata.validity.RowIsValid(union_mapped_row_idx)) {
			continue;
		}

		auto tag_mapped_row_idx = tags_vdata.sel->get_index(row_idx);
		if (!tags_vdata.validity.RowIsValid(tag_mapped_row_idx)) {
			continue;
		}

		auto tag = ((union_tag_t *)tags_vdata.data)[tag_mapped_row_idx];
		if (tag >= member_count) {
			return UnionInvalidReason::TAG_OUT_OF_RANGE;
		}

		bool found_valid = false;
		for (idx_t member_idx = 0; member_idx < member_count; member_idx++) {

			UnifiedVectorFormat member_vdata;
			auto &member = UnionVector::GetMember(vector, member_idx);
			member.ToUnifiedFormat(count, member_vdata);

			auto mapped_row_idx = member_vdata.sel->get_index(row_idx);
			if (member_vdata.validity.RowIsValid(mapped_row_idx)) {
				if (found_valid) {
					return UnionInvalidReason::VALIDITY_OVERLAP;
				}
				found_valid = true;
			}
		}
	}

	return UnionInvalidReason::VALID;
}

struct UnionValueBindData : public FunctionData {
	UnionValueBindData() {
	}

public:
	unique_ptr<FunctionData> Copy() const override {
		return make_unique<UnionValueBindData>();
	}
	bool Equals(const FunctionData &other_p) const override {
		return true;
	}
};

static void UnionValueFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	// Assign the new entries to the result vector
	UnionVector::GetMember(result, 0).Reference(args.data[0]);

	// Set the result tag vector to a constant value
	auto &tag_vector = UnionVector::GetTags(result);
	tag_vector.SetVectorType(VectorType::CONSTANT_VECTOR);
	ConstantVector::GetData<union_tag_t>(tag_vector)[0] = 0;

	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}

	result.Verify(args.size());
}

static unique_ptr<FunctionData> UnionValueBind(ClientContext &context, ScalarFunction &bound_function,
                                               vector<unique_ptr<Expression>> &arguments) {

	if (arguments.size() != 1) {
		throw BinderException("union_value takes exactly one argument");
	}
	auto &child = arguments[0];

	if (child->alias.empty()) {
		throw BinderException("Need named argument for union tag, e.g. UNION_VALUE(a := b)");
	}

	child_list_t<LogicalType> union_members;

	union_members.push_back(make_pair(child->alias, child->return_type));

	bound_function.return_type = LogicalType::UNION(std::move(union_members));
	return make_unique<VariableReturnBindData>(bound_function.return_type);
}

void UnionValueFun::RegisterFunction(BuiltinFunctions &set) {

	auto fun =
	    ScalarFunction("union_value", {}, LogicalTypeId::UNION, UnionValueFunction, UnionValueBind, nullptr, nullptr);
	fun.varargs = LogicalType::ANY;
	fun.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	fun.serialize = VariableReturnBindData::Serialize;
	fun.deserialize = VariableReturnBindData::Deserialize;

	ScalarFunctionSet union_value("union_value");
	union_value.AddFunction(fun);
	set.AddFunction(union_value);
}

} // namespace duckdb


namespace duckdb {

FunctionLocalState::~FunctionLocalState() {
}

ScalarFunction::ScalarFunction(string name, vector<LogicalType> arguments, LogicalType return_type,
                               scalar_function_t function, bind_scalar_function_t bind,
                               dependency_function_t dependency, function_statistics_t statistics,
                               init_local_state_t init_local_state, LogicalType varargs,
                               FunctionSideEffects side_effects, FunctionNullHandling null_handling)
    : BaseScalarFunction(std::move(name), std::move(arguments), std::move(return_type), side_effects,
                         std::move(varargs), null_handling),
      function(std::move(function)), bind(bind), init_local_state(init_local_state), dependency(dependency),
      statistics(statistics), serialize(nullptr), deserialize(nullptr) {
}

ScalarFunction::ScalarFunction(vector<LogicalType> arguments, LogicalType return_type, scalar_function_t function,
                               bind_scalar_function_t bind, dependency_function_t dependency,
                               function_statistics_t statistics, init_local_state_t init_local_state,
                               LogicalType varargs, FunctionSideEffects side_effects,
                               FunctionNullHandling null_handling)
    : ScalarFunction(string(), std::move(arguments), std::move(return_type), std::move(function), bind, dependency,
                     statistics, init_local_state, std::move(varargs), side_effects, null_handling) {
}

bool ScalarFunction::operator==(const ScalarFunction &rhs) const {
	return CompareScalarFunctionT(rhs.function) && bind == rhs.bind && dependency == rhs.dependency &&
	       statistics == rhs.statistics;
}
bool ScalarFunction::operator!=(const ScalarFunction &rhs) const {
	return !(*this == rhs);
}

bool ScalarFunction::Equal(const ScalarFunction &rhs) const {
	// number of types
	if (this->arguments.size() != rhs.arguments.size()) {
		return false;
	}
	// argument types
	for (idx_t i = 0; i < this->arguments.size(); ++i) {
		if (this->arguments[i] != rhs.arguments[i]) {
			return false;
		}
	}
	// return type
	if (this->return_type != rhs.return_type) {
		return false;
	}
	// varargs
	if (this->varargs != rhs.varargs) {
		return false;
	}

	return true; // they are equal
}

bool ScalarFunction::CompareScalarFunctionT(const scalar_function_t &other) const {
	typedef void(scalar_function_ptr_t)(DataChunk &, ExpressionState &, Vector &);

	auto func_ptr = (scalar_function_ptr_t **)function.template target<scalar_function_ptr_t *>();
	auto other_ptr = (scalar_function_ptr_t **)other.template target<scalar_function_ptr_t *>();

	// Case the functions were created from lambdas the target will return a nullptr
	if (!func_ptr && !other_ptr) {
		return true;
	}
	if (func_ptr == nullptr || other_ptr == nullptr) {
		// scalar_function_t (std::functions) from lambdas cannot be compared
		return false;
	}
	return ((size_t)*func_ptr == (size_t)*other_ptr);
}

void ScalarFunction::NopFunction(DataChunk &input, ExpressionState &state, Vector &result) {
	D_ASSERT(input.ColumnCount() >= 1);
	result.Reference(input.data[0]);
}

} // namespace duckdb
//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/scalar_macro_function.hpp
//
//
//===----------------------------------------------------------------------===//







namespace duckdb {

ScalarMacroFunction::ScalarMacroFunction(unique_ptr<ParsedExpression> expression)
    : MacroFunction(MacroType::SCALAR_MACRO), expression(std::move(expression)) {
}

ScalarMacroFunction::ScalarMacroFunction(void) : MacroFunction(MacroType::SCALAR_MACRO) {
}

unique_ptr<MacroFunction> ScalarMacroFunction::Copy() {
	auto result = make_unique<ScalarMacroFunction>();
	result->expression = expression->Copy();
	CopyProperties(*result);

	return std::move(result);
}

void RemoveQualificationRecursive(unique_ptr<ParsedExpression> &expr) {
	if (expr->GetExpressionType() == ExpressionType::COLUMN_REF) {
		auto &col_ref = (ColumnRefExpression &)*expr;
		auto &col_names = col_ref.column_names;
		if (col_names.size() == 2 && col_names[0].find(DummyBinding::DUMMY_NAME) != string::npos) {
			col_names.erase(col_names.begin());
		}
	} else {
		ParsedExpressionIterator::EnumerateChildren(
		    *expr, [](unique_ptr<ParsedExpression> &child) { RemoveQualificationRecursive(child); });
	}
}

string ScalarMacroFunction::ToSQL(const string &schema, const string &name) {
	// In case of nested macro's we need to fix it a bit
	auto expression_copy = expression->Copy();
	RemoveQualificationRecursive(expression_copy);
	return MacroFunction::ToSQL(schema, name) + StringUtil::Format("(%s);", expression_copy->ToString());
}

} // namespace duckdb













namespace duckdb {

LogicalType ArrowTableFunction::GetArrowLogicalType(
    ArrowSchema &schema, std::unordered_map<idx_t, unique_ptr<ArrowConvertData>> &arrow_convert_data, idx_t col_idx) {
	auto format = string(schema.format);
	if (arrow_convert_data.find(col_idx) == arrow_convert_data.end()) {
		arrow_convert_data[col_idx] = make_unique<ArrowConvertData>();
	}
	if (format == "n") {
		return LogicalType::SQLNULL;
	} else if (format == "b") {
		return LogicalType::BOOLEAN;
	} else if (format == "c") {
		return LogicalType::TINYINT;
	} else if (format == "s") {
		return LogicalType::SMALLINT;
	} else if (format == "i") {
		return LogicalType::INTEGER;
	} else if (format == "l") {
		return LogicalType::BIGINT;
	} else if (format == "C") {
		return LogicalType::UTINYINT;
	} else if (format == "S") {
		return LogicalType::USMALLINT;
	} else if (format == "I") {
		return LogicalType::UINTEGER;
	} else if (format == "L") {
		return LogicalType::UBIGINT;
	} else if (format == "f") {
		return LogicalType::FLOAT;
	} else if (format == "g") {
		return LogicalType::DOUBLE;
	} else if (format[0] == 'd') { //! this can be either decimal128 or decimal 256 (e.g., d:38,0)
		std::string parameters = format.substr(format.find(':'));
		uint8_t width = std::stoi(parameters.substr(1, parameters.find(',')));
		uint8_t scale = std::stoi(parameters.substr(parameters.find(',') + 1));
		if (width > 38) {
			throw NotImplementedException("Unsupported Internal Arrow Type for Decimal %s", format);
		}
		return LogicalType::DECIMAL(width, scale);
	} else if (format == "u") {
		arrow_convert_data[col_idx]->variable_sz_type.emplace_back(ArrowVariableSizeType::NORMAL, 0);
		return LogicalType::VARCHAR;
	} else if (format == "U") {
		arrow_convert_data[col_idx]->variable_sz_type.emplace_back(ArrowVariableSizeType::SUPER_SIZE, 0);
		return LogicalType::VARCHAR;
	} else if (format == "tsn:") {
		return LogicalTypeId::TIMESTAMP_NS;
	} else if (format == "tsu:") {
		return LogicalTypeId::TIMESTAMP;
	} else if (format == "tsm:") {
		return LogicalTypeId::TIMESTAMP_MS;
	} else if (format == "tss:") {
		return LogicalTypeId::TIMESTAMP_SEC;
	} else if (format == "tdD") {
		arrow_convert_data[col_idx]->date_time_precision.emplace_back(ArrowDateTimeType::DAYS);
		return LogicalType::DATE;
	} else if (format == "tdm") {
		arrow_convert_data[col_idx]->date_time_precision.emplace_back(ArrowDateTimeType::MILLISECONDS);
		return LogicalType::DATE;
	} else if (format == "tts") {
		arrow_convert_data[col_idx]->date_time_precision.emplace_back(ArrowDateTimeType::SECONDS);
		return LogicalType::TIME;
	} else if (format == "ttm") {
		arrow_convert_data[col_idx]->date_time_precision.emplace_back(ArrowDateTimeType::MILLISECONDS);
		return LogicalType::TIME;
	} else if (format == "ttu") {
		arrow_convert_data[col_idx]->date_time_precision.emplace_back(ArrowDateTimeType::MICROSECONDS);
		return LogicalType::TIME;
	} else if (format == "ttn") {
		arrow_convert_data[col_idx]->date_time_precision.emplace_back(ArrowDateTimeType::NANOSECONDS);
		return LogicalType::TIME;
	} else if (format == "tDs") {
		arrow_convert_data[col_idx]->date_time_precision.emplace_back(ArrowDateTimeType::SECONDS);
		return LogicalType::INTERVAL;
	} else if (format == "tDm") {
		arrow_convert_data[col_idx]->date_time_precision.emplace_back(ArrowDateTimeType::MILLISECONDS);
		return LogicalType::INTERVAL;
	} else if (format == "tDu") {
		arrow_convert_data[col_idx]->date_time_precision.emplace_back(ArrowDateTimeType::MICROSECONDS);
		return LogicalType::INTERVAL;
	} else if (format == "tDn") {
		arrow_convert_data[col_idx]->date_time_precision.emplace_back(ArrowDateTimeType::NANOSECONDS);
		return LogicalType::INTERVAL;
	} else if (format == "tiD") {
		arrow_convert_data[col_idx]->date_time_precision.emplace_back(ArrowDateTimeType::DAYS);
		return LogicalType::INTERVAL;
	} else if (format == "tiM") {
		arrow_convert_data[col_idx]->date_time_precision.emplace_back(ArrowDateTimeType::MONTHS);
		return LogicalType::INTERVAL;
	} else if (format == "+l") {
		arrow_convert_data[col_idx]->variable_sz_type.emplace_back(ArrowVariableSizeType::NORMAL, 0);
		auto child_type = GetArrowLogicalType(*schema.children[0], arrow_convert_data, col_idx);
		return LogicalType::LIST(child_type);
	} else if (format == "+L") {
		arrow_convert_data[col_idx]->variable_sz_type.emplace_back(ArrowVariableSizeType::SUPER_SIZE, 0);
		auto child_type = GetArrowLogicalType(*schema.children[0], arrow_convert_data, col_idx);
		return LogicalType::LIST(child_type);
	} else if (format[0] == '+' && format[1] == 'w') {
		std::string parameters = format.substr(format.find(':') + 1);
		idx_t fixed_size = std::stoi(parameters);
		arrow_convert_data[col_idx]->variable_sz_type.emplace_back(ArrowVariableSizeType::FIXED_SIZE, fixed_size);
		auto child_type = GetArrowLogicalType(*schema.children[0], arrow_convert_data, col_idx);
		return LogicalType::LIST(std::move(child_type));
	} else if (format == "+s") {
		child_list_t<LogicalType> child_types;
		for (idx_t type_idx = 0; type_idx < (idx_t)schema.n_children; type_idx++) {
			auto child_type = GetArrowLogicalType(*schema.children[type_idx], arrow_convert_data, col_idx);
			child_types.push_back({schema.children[type_idx]->name, child_type});
		}
		return LogicalType::STRUCT(std::move(child_types));

	} else if (format == "+m") {
		arrow_convert_data[col_idx]->variable_sz_type.emplace_back(ArrowVariableSizeType::NORMAL, 0);

		auto &arrow_struct_type = *schema.children[0];
		D_ASSERT(arrow_struct_type.n_children == 2);
		auto key_type = GetArrowLogicalType(*arrow_struct_type.children[0], arrow_convert_data, col_idx);
		auto value_type = GetArrowLogicalType(*arrow_struct_type.children[1], arrow_convert_data, col_idx);
		return LogicalType::MAP(key_type, value_type);
	} else if (format == "z") {
		arrow_convert_data[col_idx]->variable_sz_type.emplace_back(ArrowVariableSizeType::NORMAL, 0);
		return LogicalType::BLOB;
	} else if (format == "Z") {
		arrow_convert_data[col_idx]->variable_sz_type.emplace_back(ArrowVariableSizeType::SUPER_SIZE, 0);
		return LogicalType::BLOB;
	} else if (format[0] == 'w') {
		std::string parameters = format.substr(format.find(':') + 1);
		idx_t fixed_size = std::stoi(parameters);
		arrow_convert_data[col_idx]->variable_sz_type.emplace_back(ArrowVariableSizeType::FIXED_SIZE, fixed_size);
		return LogicalType::BLOB;
	} else if (format[0] == 't' && format[1] == 's') {
		// Timestamp with Timezone
		if (format[2] == 'n') {
			arrow_convert_data[col_idx]->date_time_precision.emplace_back(ArrowDateTimeType::NANOSECONDS);
		} else if (format[2] == 'u') {
			arrow_convert_data[col_idx]->date_time_precision.emplace_back(ArrowDateTimeType::MICROSECONDS);
		} else if (format[2] == 'm') {
			arrow_convert_data[col_idx]->date_time_precision.emplace_back(ArrowDateTimeType::MILLISECONDS);
		} else if (format[2] == 's') {
			arrow_convert_data[col_idx]->date_time_precision.emplace_back(ArrowDateTimeType::SECONDS);
		} else {
			throw NotImplementedException(" Timestamptz precision of not accepted");
		}
		// TODO right now we just get the UTC value. We probably want to support this properly in the future
		return LogicalType::TIMESTAMP_TZ;
	} else {
		throw NotImplementedException("Unsupported Internal Arrow Type %s", format);
	}
}

void ArrowTableFunction::RenameArrowColumns(vector<string> &names) {
	unordered_map<string, idx_t> name_map;
	for (auto &column_name : names) {
		// put it all lower_case
		auto low_column_name = StringUtil::Lower(column_name);
		if (name_map.find(low_column_name) == name_map.end()) {
			// Name does not exist yet
			name_map[low_column_name]++;
		} else {
			// Name already exists, we add _x where x is the repetition number
			string new_column_name = column_name + "_" + std::to_string(name_map[low_column_name]);
			auto new_column_name_low = StringUtil::Lower(new_column_name);
			while (name_map.find(new_column_name_low) != name_map.end()) {
				// This name is already here due to a previous definition
				name_map[low_column_name]++;
				new_column_name = column_name + "_" + std::to_string(name_map[low_column_name]);
				new_column_name_low = StringUtil::Lower(new_column_name);
			}
			column_name = new_column_name;
			name_map[new_column_name_low]++;
		}
	}
}

unique_ptr<FunctionData> ArrowTableFunction::ArrowScanBind(ClientContext &context, TableFunctionBindInput &input,
                                                           vector<LogicalType> &return_types, vector<string> &names) {
	auto stream_factory_ptr = input.inputs[0].GetPointer();
	auto stream_factory_produce = (stream_factory_produce_t)input.inputs[1].GetPointer();
	auto stream_factory_get_schema = (stream_factory_get_schema_t)input.inputs[2].GetPointer();

	auto res = make_unique<ArrowScanFunctionData>(stream_factory_produce, stream_factory_ptr);

	auto &data = *res;
	stream_factory_get_schema(stream_factory_ptr, data.schema_root);
	for (idx_t col_idx = 0; col_idx < (idx_t)data.schema_root.arrow_schema.n_children; col_idx++) {
		auto &schema = *data.schema_root.arrow_schema.children[col_idx];
		if (!schema.release) {
			throw InvalidInputException("arrow_scan: released schema passed");
		}
		if (schema.dictionary) {
			res->arrow_convert_data[col_idx] =
			    make_unique<ArrowConvertData>(GetArrowLogicalType(schema, res->arrow_convert_data, col_idx));
			return_types.emplace_back(GetArrowLogicalType(*schema.dictionary, res->arrow_convert_data, col_idx));
		} else {
			return_types.emplace_back(GetArrowLogicalType(schema, res->arrow_convert_data, col_idx));
		}
		auto format = string(schema.format);
		auto name = string(schema.name);
		if (name.empty()) {
			name = string("v") + to_string(col_idx);
		}
		names.push_back(name);
	}
	RenameArrowColumns(names);
	res->all_types = return_types;
	return std::move(res);
}

unique_ptr<ArrowArrayStreamWrapper> ProduceArrowScan(const ArrowScanFunctionData &function,
                                                     const vector<column_t> &column_ids, TableFilterSet *filters) {
	//! Generate Projection Pushdown Vector
	ArrowStreamParameters parameters;
	D_ASSERT(!column_ids.empty());
	for (idx_t idx = 0; idx < column_ids.size(); idx++) {
		auto col_idx = column_ids[idx];
		if (col_idx != COLUMN_IDENTIFIER_ROW_ID) {
			auto &schema = *function.schema_root.arrow_schema.children[col_idx];
			parameters.projected_columns.projection_map[idx] = schema.name;
			parameters.projected_columns.columns.emplace_back(schema.name);
		}
	}
	parameters.filters = filters;
	return function.scanner_producer(function.stream_factory_ptr, parameters);
}

idx_t ArrowTableFunction::ArrowScanMaxThreads(ClientContext &context, const FunctionData *bind_data_p) {
	return context.db->NumberOfThreads();
}

bool ArrowTableFunction::ArrowScanParallelStateNext(ClientContext &context, const FunctionData *bind_data_p,
                                                    ArrowScanLocalState &state, ArrowScanGlobalState &parallel_state) {
	lock_guard<mutex> parallel_lock(parallel_state.main_mutex);
	if (parallel_state.done) {
		return false;
	}
	state.chunk_offset = 0;
	state.batch_index = ++parallel_state.batch_index;

	auto current_chunk = parallel_state.stream->GetNextChunk();
	while (current_chunk->arrow_array.length == 0 && current_chunk->arrow_array.release) {
		current_chunk = parallel_state.stream->GetNextChunk();
	}
	state.chunk = std::move(current_chunk);
	//! have we run out of chunks? we are done
	if (!state.chunk->arrow_array.release) {
		parallel_state.done = true;
		return false;
	}
	return true;
}

unique_ptr<GlobalTableFunctionState> ArrowTableFunction::ArrowScanInitGlobal(ClientContext &context,
                                                                             TableFunctionInitInput &input) {
	auto &bind_data = (const ArrowScanFunctionData &)*input.bind_data;
	auto result = make_unique<ArrowScanGlobalState>();
	result->stream = ProduceArrowScan(bind_data, input.column_ids, input.filters);
	result->max_threads = ArrowScanMaxThreads(context, input.bind_data);
	if (input.CanRemoveFilterColumns()) {
		result->projection_ids = input.projection_ids;
		for (const auto &col_idx : input.column_ids) {
			if (col_idx == COLUMN_IDENTIFIER_ROW_ID) {
				result->scanned_types.emplace_back(LogicalType::ROW_TYPE);
			} else {
				result->scanned_types.push_back(bind_data.all_types[col_idx]);
			}
		}
	}
	return std::move(result);
}

unique_ptr<LocalTableFunctionState> ArrowTableFunction::ArrowScanInitLocal(ExecutionContext &context,
                                                                           TableFunctionInitInput &input,
                                                                           GlobalTableFunctionState *global_state_p) {
	auto &global_state = (ArrowScanGlobalState &)*global_state_p;
	auto current_chunk = make_unique<ArrowArrayWrapper>();
	auto result = make_unique<ArrowScanLocalState>(std::move(current_chunk));
	result->column_ids = input.column_ids;
	result->filters = input.filters;
	if (input.CanRemoveFilterColumns()) {
		auto &asgs = (ArrowScanGlobalState &)*global_state_p;
		result->all_columns.Initialize(context.client, asgs.scanned_types);
	}
	if (!ArrowScanParallelStateNext(context.client, input.bind_data, *result, global_state)) {
		return nullptr;
	}
	return std::move(result);
}

void ArrowTableFunction::ArrowScanFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	if (!data_p.local_state) {
		return;
	}
	auto &data = (ArrowScanFunctionData &)*data_p.bind_data;
	auto &state = (ArrowScanLocalState &)*data_p.local_state;
	auto &global_state = (ArrowScanGlobalState &)*data_p.global_state;

	//! Out of tuples in this chunk
	if (state.chunk_offset >= (idx_t)state.chunk->arrow_array.length) {
		if (!ArrowScanParallelStateNext(context, data_p.bind_data, state, global_state)) {
			return;
		}
	}
	int64_t output_size = MinValue<int64_t>(STANDARD_VECTOR_SIZE, state.chunk->arrow_array.length - state.chunk_offset);
	data.lines_read += output_size;
	if (global_state.CanRemoveFilterColumns()) {
		state.all_columns.Reset();
		state.all_columns.SetCardinality(output_size);
		ArrowToDuckDB(state, data.arrow_convert_data, state.all_columns, data.lines_read - output_size);
		output.ReferenceColumns(state.all_columns, global_state.projection_ids);
	} else {
		output.SetCardinality(output_size);
		ArrowToDuckDB(state, data.arrow_convert_data, output, data.lines_read - output_size);
	}

	output.Verify();
	state.chunk_offset += output.size();
}

unique_ptr<NodeStatistics> ArrowTableFunction::ArrowScanCardinality(ClientContext &context, const FunctionData *data) {
	return make_unique<NodeStatistics>();
}

idx_t ArrowTableFunction::ArrowGetBatchIndex(ClientContext &context, const FunctionData *bind_data_p,
                                             LocalTableFunctionState *local_state,
                                             GlobalTableFunctionState *global_state) {
	auto &state = (ArrowScanLocalState &)*local_state;
	return state.batch_index;
}

void ArrowTableFunction::RegisterFunction(BuiltinFunctions &set) {
	TableFunction arrow("arrow_scan", {LogicalType::POINTER, LogicalType::POINTER, LogicalType::POINTER},
	                    ArrowScanFunction, ArrowScanBind, ArrowScanInitGlobal, ArrowScanInitLocal);
	arrow.cardinality = ArrowScanCardinality;
	arrow.get_batch_index = ArrowGetBatchIndex;
	arrow.projection_pushdown = true;
	arrow.filter_pushdown = true;
	arrow.filter_prune = true;
	set.AddFunction(arrow);

	TableFunction arrow_dumb("arrow_scan_dumb", {LogicalType::POINTER, LogicalType::POINTER, LogicalType::POINTER},
	                         ArrowScanFunction, ArrowScanBind, ArrowScanInitGlobal, ArrowScanInitLocal);
	arrow_dumb.cardinality = ArrowScanCardinality;
	arrow_dumb.get_batch_index = ArrowGetBatchIndex;
	arrow_dumb.projection_pushdown = false;
	arrow_dumb.filter_pushdown = false;
	arrow_dumb.filter_prune = false;
	set.AddFunction(arrow_dumb);
}

void BuiltinFunctions::RegisterArrowFunctions() {
	ArrowTableFunction::RegisterFunction(*this);
}
} // namespace duckdb







namespace duckdb {

void ShiftRight(unsigned char *ar, int size, int shift) {
	int carry = 0;
	while (shift--) {
		for (int i = size - 1; i >= 0; --i) {
			int next = (ar[i] & 1) ? 0x80 : 0;
			ar[i] = carry | (ar[i] >> 1);
			carry = next;
		}
	}
}

void GetValidityMask(ValidityMask &mask, ArrowArray &array, ArrowScanLocalState &scan_state, idx_t size,
                     int64_t nested_offset = -1, bool add_null = false) {
	if (array.null_count != 0 && array.buffers[0]) {
		auto bit_offset = scan_state.chunk_offset + array.offset;
		if (nested_offset != -1) {
			bit_offset = nested_offset;
		}
		mask.EnsureWritable();
#if STANDARD_VECTOR_SIZE > 64
		auto n_bitmask_bytes = (size + 8 - 1) / 8;
		if (bit_offset % 8 == 0) {
			//! just memcpy nullmask
			memcpy((void *)mask.GetData(), (uint8_t *)array.buffers[0] + bit_offset / 8, n_bitmask_bytes);
		} else {
			//! need to re-align nullmask
			std::vector<uint8_t> temp_nullmask(n_bitmask_bytes + 1);
			memcpy(temp_nullmask.data(), (uint8_t *)array.buffers[0] + bit_offset / 8, n_bitmask_bytes + 1);
			ShiftRight(temp_nullmask.data(), n_bitmask_bytes + 1,
			           bit_offset % 8); //! why this has to be a right shift is a mystery to me
			memcpy((void *)mask.GetData(), (data_ptr_t)temp_nullmask.data(), n_bitmask_bytes);
		}
#else
		auto byte_offset = bit_offset / 8;
		auto source_data = (uint8_t *)array.buffers[0];
		bit_offset %= 8;
		for (idx_t i = 0; i < size; i++) {
			mask.Set(i, source_data[byte_offset] & (1 << bit_offset));
			bit_offset++;
			if (bit_offset == 8) {
				bit_offset = 0;
				byte_offset++;
			}
		}
#endif
	}
	if (add_null) {
		//! We are setting a validity mask of the data part of dictionary vector
		//! For some reason, Nulls are allowed to be indexes, hence we need to set the last element here to be null
		//! We might have to resize the mask
		mask.Resize(size, size + 1);
		mask.SetInvalid(size);
	}
}

void SetValidityMask(Vector &vector, ArrowArray &array, ArrowScanLocalState &scan_state, idx_t size,
                     int64_t nested_offset, bool add_null = false) {
	D_ASSERT(vector.GetVectorType() == VectorType::FLAT_VECTOR);
	auto &mask = FlatVector::Validity(vector);
	GetValidityMask(mask, array, scan_state, size, nested_offset, add_null);
}

void ColumnArrowToDuckDB(Vector &vector, ArrowArray &array, ArrowScanLocalState &scan_state, idx_t size,
                         std::unordered_map<idx_t, unique_ptr<ArrowConvertData>> &arrow_convert_data, idx_t col_idx,
                         std::pair<idx_t, idx_t> &arrow_convert_idx, int64_t nested_offset = -1,
                         ValidityMask *parent_mask = nullptr);

void ArrowToDuckDBList(Vector &vector, ArrowArray &array, ArrowScanLocalState &scan_state, idx_t size,
                       std::unordered_map<idx_t, unique_ptr<ArrowConvertData>> &arrow_convert_data, idx_t col_idx,
                       std::pair<idx_t, idx_t> &arrow_convert_idx, int64_t nested_offset, ValidityMask *parent_mask) {
	auto original_type = arrow_convert_data[col_idx]->variable_sz_type[arrow_convert_idx.first++];
	idx_t list_size = 0;
	SetValidityMask(vector, array, scan_state, size, nested_offset);
	idx_t start_offset = 0;
	idx_t cur_offset = 0;
	if (original_type.first == ArrowVariableSizeType::FIXED_SIZE) {
		//! Have to check validity mask before setting this up
		idx_t offset = (scan_state.chunk_offset + array.offset) * original_type.second;
		if (nested_offset != -1) {
			offset = original_type.second * nested_offset;
		}
		start_offset = offset;
		auto list_data = FlatVector::GetData<list_entry_t>(vector);
		for (idx_t i = 0; i < size; i++) {
			auto &le = list_data[i];
			le.offset = cur_offset;
			le.length = original_type.second;
			cur_offset += original_type.second;
		}
		list_size = start_offset + cur_offset;
	} else if (original_type.first == ArrowVariableSizeType::NORMAL) {
		auto offsets = (uint32_t *)array.buffers[1] + array.offset + scan_state.chunk_offset;
		if (nested_offset != -1) {
			offsets = (uint32_t *)array.buffers[1] + nested_offset;
		}
		start_offset = offsets[0];
		auto list_data = FlatVector::GetData<list_entry_t>(vector);
		for (idx_t i = 0; i < size; i++) {
			auto &le = list_data[i];
			le.offset = cur_offset;
			le.length = offsets[i + 1] - offsets[i];
			cur_offset += le.length;
		}
		list_size = offsets[size];
	} else {
		auto offsets = (uint64_t *)array.buffers[1] + array.offset + scan_state.chunk_offset;
		if (nested_offset != -1) {
			offsets = (uint64_t *)array.buffers[1] + nested_offset;
		}
		start_offset = offsets[0];
		auto list_data = FlatVector::GetData<list_entry_t>(vector);
		for (idx_t i = 0; i < size; i++) {
			auto &le = list_data[i];
			le.offset = cur_offset;
			le.length = offsets[i + 1] - offsets[i];
			cur_offset += le.length;
		}
		list_size = offsets[size];
	}
	list_size -= start_offset;
	ListVector::Reserve(vector, list_size);
	ListVector::SetListSize(vector, list_size);
	auto &child_vector = ListVector::GetEntry(vector);
	SetValidityMask(child_vector, *array.children[0], scan_state, list_size, start_offset);
	auto &list_mask = FlatVector::Validity(vector);
	if (parent_mask) {
		//! Since this List is owned by a struct we must guarantee their validity map matches on Null
		if (!parent_mask->AllValid()) {
			for (idx_t i = 0; i < size; i++) {
				if (!parent_mask->RowIsValid(i)) {
					list_mask.SetInvalid(i);
				}
			}
		}
	}
	if (list_size == 0 && start_offset == 0) {
		ColumnArrowToDuckDB(child_vector, *array.children[0], scan_state, list_size, arrow_convert_data, col_idx,
		                    arrow_convert_idx, -1);
	} else {
		ColumnArrowToDuckDB(child_vector, *array.children[0], scan_state, list_size, arrow_convert_data, col_idx,
		                    arrow_convert_idx, start_offset);
	}
}

void ArrowToDuckDBBlob(Vector &vector, ArrowArray &array, ArrowScanLocalState &scan_state, idx_t size,
                       std::unordered_map<idx_t, unique_ptr<ArrowConvertData>> &arrow_convert_data, idx_t col_idx,
                       std::pair<idx_t, idx_t> &arrow_convert_idx, int64_t nested_offset) {
	auto original_type = arrow_convert_data[col_idx]->variable_sz_type[arrow_convert_idx.first++];
	SetValidityMask(vector, array, scan_state, size, nested_offset);
	if (original_type.first == ArrowVariableSizeType::FIXED_SIZE) {
		//! Have to check validity mask before setting this up
		idx_t offset = (scan_state.chunk_offset + array.offset) * original_type.second;
		if (nested_offset != -1) {
			offset = original_type.second * nested_offset;
		}
		auto cdata = (char *)array.buffers[1];
		for (idx_t row_idx = 0; row_idx < size; row_idx++) {
			if (FlatVector::IsNull(vector, row_idx)) {
				continue;
			}
			auto bptr = cdata + offset;
			auto blob_len = original_type.second;
			FlatVector::GetData<string_t>(vector)[row_idx] = StringVector::AddStringOrBlob(vector, bptr, blob_len);
			offset += blob_len;
		}
	} else if (original_type.first == ArrowVariableSizeType::NORMAL) {
		auto offsets = (uint32_t *)array.buffers[1] + array.offset + scan_state.chunk_offset;
		if (nested_offset != -1) {
			offsets = (uint32_t *)array.buffers[1] + array.offset + nested_offset;
		}
		auto cdata = (char *)array.buffers[2];
		for (idx_t row_idx = 0; row_idx < size; row_idx++) {
			if (FlatVector::IsNull(vector, row_idx)) {
				continue;
			}
			auto bptr = cdata + offsets[row_idx];
			auto blob_len = offsets[row_idx + 1] - offsets[row_idx];
			FlatVector::GetData<string_t>(vector)[row_idx] = StringVector::AddStringOrBlob(vector, bptr, blob_len);
		}
	} else {
		//! Check if last offset is higher than max uint32
		if (((uint64_t *)array.buffers[1])[array.length] > NumericLimits<uint32_t>::Maximum()) { // LCOV_EXCL_START
			throw std::runtime_error("DuckDB does not support Blobs over 4GB");
		} // LCOV_EXCL_STOP
		auto offsets = (uint64_t *)array.buffers[1] + array.offset + scan_state.chunk_offset;
		if (nested_offset != -1) {
			offsets = (uint64_t *)array.buffers[1] + array.offset + nested_offset;
		}
		auto cdata = (char *)array.buffers[2];
		for (idx_t row_idx = 0; row_idx < size; row_idx++) {
			if (FlatVector::IsNull(vector, row_idx)) {
				continue;
			}
			auto bptr = cdata + offsets[row_idx];
			auto blob_len = offsets[row_idx + 1] - offsets[row_idx];
			FlatVector::GetData<string_t>(vector)[row_idx] = StringVector::AddStringOrBlob(vector, bptr, blob_len);
		}
	}
}

void ArrowToDuckDBMapVerify(Vector &vector, idx_t count) {
	auto valid_check = CheckMapValidity(vector, count);
	switch (valid_check) {
	case MapInvalidReason::VALID:
		break;
	case MapInvalidReason::DUPLICATE_KEY: {
		throw InvalidInputException("Arrow map contains duplicate key, which isn't supported by DuckDB map type");
	}
	case MapInvalidReason::NULL_KEY: {
		throw InvalidInputException("Arrow map contains NULL as map key, which isn't supported by DuckDB map type");
	}
	case MapInvalidReason::NULL_KEY_LIST: {
		throw InvalidInputException("Arrow map contains NULL as key list, which isn't supported by DuckDB map type");
	}
	default: {
		throw InternalException("MapInvalidReason not implemented");
	}
	}
}

template <class T>
static void SetVectorString(Vector &vector, idx_t size, char *cdata, T *offsets) {
	auto strings = FlatVector::GetData<string_t>(vector);
	for (idx_t row_idx = 0; row_idx < size; row_idx++) {
		if (FlatVector::IsNull(vector, row_idx)) {
			continue;
		}
		auto cptr = cdata + offsets[row_idx];
		auto str_len = offsets[row_idx + 1] - offsets[row_idx];
		strings[row_idx] = string_t(cptr, str_len);
	}
}

void DirectConversion(Vector &vector, ArrowArray &array, ArrowScanLocalState &scan_state, int64_t nested_offset) {
	auto internal_type = GetTypeIdSize(vector.GetType().InternalType());
	auto data_ptr = (data_ptr_t)array.buffers[1] + internal_type * (scan_state.chunk_offset + array.offset);
	if (nested_offset != -1) {
		data_ptr = (data_ptr_t)array.buffers[1] + internal_type * (array.offset + nested_offset);
	}
	FlatVector::SetData(vector, data_ptr);
}

template <class T>
void TimeConversion(Vector &vector, ArrowArray &array, ArrowScanLocalState &scan_state, int64_t nested_offset,
                    idx_t size, int64_t conversion) {
	auto tgt_ptr = (dtime_t *)FlatVector::GetData(vector);
	auto &validity_mask = FlatVector::Validity(vector);
	auto src_ptr = (T *)array.buffers[1] + scan_state.chunk_offset + array.offset;
	if (nested_offset != -1) {
		src_ptr = (T *)array.buffers[1] + nested_offset + array.offset;
	}
	for (idx_t row = 0; row < size; row++) {
		if (!validity_mask.RowIsValid(row)) {
			continue;
		}
		if (!TryMultiplyOperator::Operation((int64_t)src_ptr[row], conversion, tgt_ptr[row].micros)) {
			throw ConversionException("Could not convert Time to Microsecond");
		}
	}
}

void TimestampTZConversion(Vector &vector, ArrowArray &array, ArrowScanLocalState &scan_state, int64_t nested_offset,
                           idx_t size, int64_t conversion) {
	auto tgt_ptr = (timestamp_t *)FlatVector::GetData(vector);
	auto &validity_mask = FlatVector::Validity(vector);
	auto src_ptr = (int64_t *)array.buffers[1] + scan_state.chunk_offset + array.offset;
	if (nested_offset != -1) {
		src_ptr = (int64_t *)array.buffers[1] + nested_offset + array.offset;
	}
	for (idx_t row = 0; row < size; row++) {
		if (!validity_mask.RowIsValid(row)) {
			continue;
		}
		if (!TryMultiplyOperator::Operation(src_ptr[row], conversion, tgt_ptr[row].value)) {
			throw ConversionException("Could not convert TimestampTZ to Microsecond");
		}
	}
}

void IntervalConversionUs(Vector &vector, ArrowArray &array, ArrowScanLocalState &scan_state, int64_t nested_offset,
                          idx_t size, int64_t conversion) {
	auto tgt_ptr = (interval_t *)FlatVector::GetData(vector);
	auto src_ptr = (int64_t *)array.buffers[1] + scan_state.chunk_offset + array.offset;
	if (nested_offset != -1) {
		src_ptr = (int64_t *)array.buffers[1] + nested_offset + array.offset;
	}
	for (idx_t row = 0; row < size; row++) {
		tgt_ptr[row].days = 0;
		tgt_ptr[row].months = 0;
		if (!TryMultiplyOperator::Operation(src_ptr[row], conversion, tgt_ptr[row].micros)) {
			throw ConversionException("Could not convert Interval to Microsecond");
		}
	}
}

void IntervalConversionMonths(Vector &vector, ArrowArray &array, ArrowScanLocalState &scan_state, int64_t nested_offset,
                              idx_t size) {
	auto tgt_ptr = (interval_t *)FlatVector::GetData(vector);
	auto src_ptr = (int32_t *)array.buffers[1] + scan_state.chunk_offset + array.offset;
	if (nested_offset != -1) {
		src_ptr = (int32_t *)array.buffers[1] + nested_offset + array.offset;
	}
	for (idx_t row = 0; row < size; row++) {
		tgt_ptr[row].days = 0;
		tgt_ptr[row].micros = 0;
		tgt_ptr[row].months = src_ptr[row];
	}
}

void ColumnArrowToDuckDB(Vector &vector, ArrowArray &array, ArrowScanLocalState &scan_state, idx_t size,
                         std::unordered_map<idx_t, unique_ptr<ArrowConvertData>> &arrow_convert_data, idx_t col_idx,
                         std::pair<idx_t, idx_t> &arrow_convert_idx, int64_t nested_offset, ValidityMask *parent_mask) {
	switch (vector.GetType().id()) {
	case LogicalTypeId::SQLNULL:
		vector.Reference(Value());
		break;
	case LogicalTypeId::BOOLEAN: {
		//! Arrow bit-packs boolean values
		//! Lets first figure out where we are in the source array
		auto src_ptr = (uint8_t *)array.buffers[1] + (scan_state.chunk_offset + array.offset) / 8;

		if (nested_offset != -1) {
			src_ptr = (uint8_t *)array.buffers[1] + (nested_offset + array.offset) / 8;
		}
		auto tgt_ptr = (uint8_t *)FlatVector::GetData(vector);
		int src_pos = 0;
		idx_t cur_bit = scan_state.chunk_offset % 8;
		if (nested_offset != -1) {
			cur_bit = nested_offset % 8;
		}
		for (idx_t row = 0; row < size; row++) {
			if ((src_ptr[src_pos] & (1 << cur_bit)) == 0) {
				tgt_ptr[row] = 0;
			} else {
				tgt_ptr[row] = 1;
			}
			cur_bit++;
			if (cur_bit == 8) {
				src_pos++;
				cur_bit = 0;
			}
		}
		break;
	}
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIMESTAMP_NS: {
		DirectConversion(vector, array, scan_state, nested_offset);
		break;
	}
	case LogicalTypeId::VARCHAR: {
		auto original_type = arrow_convert_data[col_idx]->variable_sz_type[arrow_convert_idx.first++];
		auto cdata = (char *)array.buffers[2];
		if (original_type.first == ArrowVariableSizeType::SUPER_SIZE) {
			if (((uint64_t *)array.buffers[1])[array.length] > NumericLimits<uint32_t>::Maximum()) { // LCOV_EXCL_START
				throw std::runtime_error("DuckDB does not support Strings over 4GB");
			} // LCOV_EXCL_STOP
			auto offsets = (uint64_t *)array.buffers[1] + array.offset + scan_state.chunk_offset;
			if (nested_offset != -1) {
				offsets = (uint64_t *)array.buffers[1] + array.offset + nested_offset;
			}
			SetVectorString(vector, size, cdata, offsets);
		} else {
			auto offsets = (uint32_t *)array.buffers[1] + array.offset + scan_state.chunk_offset;
			if (nested_offset != -1) {
				offsets = (uint32_t *)array.buffers[1] + array.offset + nested_offset;
			}
			SetVectorString(vector, size, cdata, offsets);
		}
		break;
	}
	case LogicalTypeId::DATE: {
		auto precision = arrow_convert_data[col_idx]->date_time_precision[arrow_convert_idx.second++];
		switch (precision) {
		case ArrowDateTimeType::DAYS: {
			DirectConversion(vector, array, scan_state, nested_offset);
			break;
		}
		case ArrowDateTimeType::MILLISECONDS: {
			//! convert date from nanoseconds to days
			auto src_ptr = (uint64_t *)array.buffers[1] + scan_state.chunk_offset + array.offset;
			if (nested_offset != -1) {
				src_ptr = (uint64_t *)array.buffers[1] + nested_offset + array.offset;
			}
			auto tgt_ptr = (date_t *)FlatVector::GetData(vector);
			for (idx_t row = 0; row < size; row++) {
				tgt_ptr[row] = date_t(int64_t(src_ptr[row]) / static_cast<int64_t>(1000 * 60 * 60 * 24));
			}
			break;
		}
		default:
			throw std::runtime_error("Unsupported precision for Date Type ");
		}
		break;
	}
	case LogicalTypeId::TIME: {
		auto precision = arrow_convert_data[col_idx]->date_time_precision[arrow_convert_idx.second++];
		switch (precision) {
		case ArrowDateTimeType::SECONDS: {
			TimeConversion<int32_t>(vector, array, scan_state, nested_offset, size, 1000000);
			break;
		}
		case ArrowDateTimeType::MILLISECONDS: {
			TimeConversion<int32_t>(vector, array, scan_state, nested_offset, size, 1000);
			break;
		}
		case ArrowDateTimeType::MICROSECONDS: {
			TimeConversion<int64_t>(vector, array, scan_state, nested_offset, size, 1);
			break;
		}
		case ArrowDateTimeType::NANOSECONDS: {
			auto tgt_ptr = (dtime_t *)FlatVector::GetData(vector);
			auto src_ptr = (int64_t *)array.buffers[1] + scan_state.chunk_offset + array.offset;
			if (nested_offset != -1) {
				src_ptr = (int64_t *)array.buffers[1] + nested_offset + array.offset;
			}
			for (idx_t row = 0; row < size; row++) {
				tgt_ptr[row].micros = src_ptr[row] / 1000;
			}
			break;
		}
		default:
			throw std::runtime_error("Unsupported precision for Time Type ");
		}
		break;
	}
	case LogicalTypeId::TIMESTAMP_TZ: {
		auto precision = arrow_convert_data[col_idx]->date_time_precision[arrow_convert_idx.second++];
		switch (precision) {
		case ArrowDateTimeType::SECONDS: {
			TimestampTZConversion(vector, array, scan_state, nested_offset, size, 1000000);
			break;
		}
		case ArrowDateTimeType::MILLISECONDS: {
			TimestampTZConversion(vector, array, scan_state, nested_offset, size, 1000);
			break;
		}
		case ArrowDateTimeType::MICROSECONDS: {
			DirectConversion(vector, array, scan_state, nested_offset);
			break;
		}
		case ArrowDateTimeType::NANOSECONDS: {
			auto tgt_ptr = (timestamp_t *)FlatVector::GetData(vector);
			auto src_ptr = (int64_t *)array.buffers[1] + scan_state.chunk_offset + array.offset;
			if (nested_offset != -1) {
				src_ptr = (int64_t *)array.buffers[1] + nested_offset + array.offset;
			}
			for (idx_t row = 0; row < size; row++) {
				tgt_ptr[row].value = src_ptr[row] / 1000;
			}
			break;
		}
		default:
			throw std::runtime_error("Unsupported precision for TimestampTZ Type ");
		}
		break;
	}
	case LogicalTypeId::INTERVAL: {
		auto precision = arrow_convert_data[col_idx]->date_time_precision[arrow_convert_idx.second++];
		switch (precision) {
		case ArrowDateTimeType::SECONDS: {
			IntervalConversionUs(vector, array, scan_state, nested_offset, size, 1000000);
			break;
		}
		case ArrowDateTimeType::DAYS:
		case ArrowDateTimeType::MILLISECONDS: {
			IntervalConversionUs(vector, array, scan_state, nested_offset, size, 1000);
			break;
		}
		case ArrowDateTimeType::MICROSECONDS: {
			IntervalConversionUs(vector, array, scan_state, nested_offset, size, 1);
			break;
		}
		case ArrowDateTimeType::NANOSECONDS: {
			auto tgt_ptr = (interval_t *)FlatVector::GetData(vector);
			auto src_ptr = (int64_t *)array.buffers[1] + scan_state.chunk_offset + array.offset;
			if (nested_offset != -1) {
				src_ptr = (int64_t *)array.buffers[1] + nested_offset + array.offset;
			}
			for (idx_t row = 0; row < size; row++) {
				tgt_ptr[row].micros = src_ptr[row] / 1000;
				tgt_ptr[row].days = 0;
				tgt_ptr[row].months = 0;
			}
			break;
		}
		case ArrowDateTimeType::MONTHS: {
			IntervalConversionMonths(vector, array, scan_state, nested_offset, size);
			break;
		}
		default:
			throw std::runtime_error("Unsupported precision for Interval/Duration Type ");
		}
		break;
	}
	case LogicalTypeId::DECIMAL: {
		auto val_mask = FlatVector::Validity(vector);
		//! We have to convert from INT128
		auto src_ptr = (hugeint_t *)array.buffers[1] + scan_state.chunk_offset + array.offset;
		if (nested_offset != -1) {
			src_ptr = (hugeint_t *)array.buffers[1] + nested_offset + array.offset;
		}
		switch (vector.GetType().InternalType()) {
		case PhysicalType::INT16: {
			auto tgt_ptr = (int16_t *)FlatVector::GetData(vector);
			for (idx_t row = 0; row < size; row++) {
				if (val_mask.RowIsValid(row)) {
					auto result = Hugeint::TryCast(src_ptr[row], tgt_ptr[row]);
					D_ASSERT(result);
					(void)result;
				}
			}
			break;
		}
		case PhysicalType::INT32: {
			auto tgt_ptr = (int32_t *)FlatVector::GetData(vector);
			for (idx_t row = 0; row < size; row++) {
				if (val_mask.RowIsValid(row)) {
					auto result = Hugeint::TryCast(src_ptr[row], tgt_ptr[row]);
					D_ASSERT(result);
					(void)result;
				}
			}
			break;
		}
		case PhysicalType::INT64: {
			auto tgt_ptr = (int64_t *)FlatVector::GetData(vector);
			for (idx_t row = 0; row < size; row++) {
				if (val_mask.RowIsValid(row)) {
					auto result = Hugeint::TryCast(src_ptr[row], tgt_ptr[row]);
					D_ASSERT(result);
					(void)result;
				}
			}
			break;
		}
		case PhysicalType::INT128: {
			FlatVector::SetData(vector, (data_ptr_t)array.buffers[1] + GetTypeIdSize(vector.GetType().InternalType()) *
			                                                               (scan_state.chunk_offset + array.offset));
			break;
		}
		default:
			throw std::runtime_error("Unsupported physical type for Decimal: " +
			                         TypeIdToString(vector.GetType().InternalType()));
		}
		break;
	}
	case LogicalTypeId::BLOB: {
		ArrowToDuckDBBlob(vector, array, scan_state, size, arrow_convert_data, col_idx, arrow_convert_idx,
		                  nested_offset);
		break;
	}
	case LogicalTypeId::LIST: {
		ArrowToDuckDBList(vector, array, scan_state, size, arrow_convert_data, col_idx, arrow_convert_idx,
		                  nested_offset, parent_mask);
		break;
	}
	case LogicalTypeId::MAP: {
		ArrowToDuckDBList(vector, array, scan_state, size, arrow_convert_data, col_idx, arrow_convert_idx,
		                  nested_offset, parent_mask);
		ArrowToDuckDBMapVerify(vector, size);
		break;
	}
	case LogicalTypeId::STRUCT: {
		//! Fill the children
		auto &child_entries = StructVector::GetEntries(vector);
		auto &struct_validity_mask = FlatVector::Validity(vector);
		for (idx_t type_idx = 0; type_idx < (idx_t)array.n_children; type_idx++) {
			SetValidityMask(*child_entries[type_idx], *array.children[type_idx], scan_state, size, nested_offset);
			if (!struct_validity_mask.AllValid()) {
				auto &child_validity_mark = FlatVector::Validity(*child_entries[type_idx]);
				for (idx_t i = 0; i < size; i++) {
					if (!struct_validity_mask.RowIsValid(i)) {
						child_validity_mark.SetInvalid(i);
					}
				}
			}
			ColumnArrowToDuckDB(*child_entries[type_idx], *array.children[type_idx], scan_state, size,
			                    arrow_convert_data, col_idx, arrow_convert_idx, nested_offset, &struct_validity_mask);
		}
		break;
	}
	default:
		throw std::runtime_error("Unsupported type " + vector.GetType().ToString());
	}
}

template <class T>
static void SetSelectionVectorLoop(SelectionVector &sel, data_ptr_t indices_p, idx_t size) {
	auto indices = (T *)indices_p;
	for (idx_t row = 0; row < size; row++) {
		sel.set_index(row, indices[row]);
	}
}

template <class T>
static void SetSelectionVectorLoopWithChecks(SelectionVector &sel, data_ptr_t indices_p, idx_t size) {

	auto indices = (T *)indices_p;
	for (idx_t row = 0; row < size; row++) {
		if (indices[row] > NumericLimits<uint32_t>::Maximum()) {
			throw std::runtime_error("DuckDB only supports indices that fit on an uint32");
		}
		sel.set_index(row, indices[row]);
	}
}

template <class T>
static void SetMaskedSelectionVectorLoop(SelectionVector &sel, data_ptr_t indices_p, idx_t size, ValidityMask &mask,
                                         idx_t last_element_pos) {
	auto indices = (T *)indices_p;
	for (idx_t row = 0; row < size; row++) {
		if (mask.RowIsValid(row)) {
			sel.set_index(row, indices[row]);
		} else {
			//! Need to point out to last element
			sel.set_index(row, last_element_pos);
		}
	}
}

void SetSelectionVector(SelectionVector &sel, data_ptr_t indices_p, LogicalType &logical_type, idx_t size,
                        ValidityMask *mask = nullptr, idx_t last_element_pos = 0) {
	sel.Initialize(size);

	if (mask) {
		switch (logical_type.id()) {
		case LogicalTypeId::UTINYINT:
			SetMaskedSelectionVectorLoop<uint8_t>(sel, indices_p, size, *mask, last_element_pos);
			break;
		case LogicalTypeId::TINYINT:
			SetMaskedSelectionVectorLoop<int8_t>(sel, indices_p, size, *mask, last_element_pos);
			break;
		case LogicalTypeId::USMALLINT:
			SetMaskedSelectionVectorLoop<uint16_t>(sel, indices_p, size, *mask, last_element_pos);
			break;
		case LogicalTypeId::SMALLINT:
			SetMaskedSelectionVectorLoop<int16_t>(sel, indices_p, size, *mask, last_element_pos);
			break;
		case LogicalTypeId::UINTEGER:
			if (last_element_pos > NumericLimits<uint32_t>::Maximum()) {
				//! Its guaranteed that our indices will point to the last element, so just throw an error
				throw std::runtime_error("DuckDB only supports indices that fit on an uint32");
			}
			SetMaskedSelectionVectorLoop<uint32_t>(sel, indices_p, size, *mask, last_element_pos);
			break;
		case LogicalTypeId::INTEGER:
			SetMaskedSelectionVectorLoop<int32_t>(sel, indices_p, size, *mask, last_element_pos);
			break;
		case LogicalTypeId::UBIGINT:
			if (last_element_pos > NumericLimits<uint32_t>::Maximum()) {
				//! Its guaranteed that our indices will point to the last element, so just throw an error
				throw std::runtime_error("DuckDB only supports indices that fit on an uint32");
			}
			SetMaskedSelectionVectorLoop<uint64_t>(sel, indices_p, size, *mask, last_element_pos);
			break;
		case LogicalTypeId::BIGINT:
			if (last_element_pos > NumericLimits<uint32_t>::Maximum()) {
				//! Its guaranteed that our indices will point to the last element, so just throw an error
				throw std::runtime_error("DuckDB only supports indices that fit on an uint32");
			}
			SetMaskedSelectionVectorLoop<int64_t>(sel, indices_p, size, *mask, last_element_pos);
			break;

		default:
			throw std::runtime_error("(Arrow) Unsupported type for selection vectors " + logical_type.ToString());
		}

	} else {
		switch (logical_type.id()) {
		case LogicalTypeId::UTINYINT:
			SetSelectionVectorLoop<uint8_t>(sel, indices_p, size);
			break;
		case LogicalTypeId::TINYINT:
			SetSelectionVectorLoop<int8_t>(sel, indices_p, size);
			break;
		case LogicalTypeId::USMALLINT:
			SetSelectionVectorLoop<uint16_t>(sel, indices_p, size);
			break;
		case LogicalTypeId::SMALLINT:
			SetSelectionVectorLoop<int16_t>(sel, indices_p, size);
			break;
		case LogicalTypeId::UINTEGER:
			SetSelectionVectorLoop<uint32_t>(sel, indices_p, size);
			break;
		case LogicalTypeId::INTEGER:
			SetSelectionVectorLoop<int32_t>(sel, indices_p, size);
			break;
		case LogicalTypeId::UBIGINT:
			if (last_element_pos > NumericLimits<uint32_t>::Maximum()) {
				//! We need to check if our indexes fit in a uint32_t
				SetSelectionVectorLoopWithChecks<uint64_t>(sel, indices_p, size);
			} else {
				SetSelectionVectorLoop<uint64_t>(sel, indices_p, size);
			}
			break;
		case LogicalTypeId::BIGINT:
			if (last_element_pos > NumericLimits<uint32_t>::Maximum()) {
				//! We need to check if our indexes fit in a uint32_t
				SetSelectionVectorLoopWithChecks<int64_t>(sel, indices_p, size);
			} else {
				SetSelectionVectorLoop<int64_t>(sel, indices_p, size);
			}
			break;
		default:
			throw std::runtime_error("(Arrow) Unsupported type for selection vectors " + logical_type.ToString());
		}
	}
}

void ColumnArrowToDuckDBDictionary(Vector &vector, ArrowArray &array, ArrowScanLocalState &scan_state, idx_t size,
                                   std::unordered_map<idx_t, unique_ptr<ArrowConvertData>> &arrow_convert_data,
                                   idx_t col_idx, std::pair<idx_t, idx_t> &arrow_convert_idx) {
	SelectionVector sel;
	auto &dict_vectors = scan_state.arrow_dictionary_vectors;
	if (dict_vectors.find(col_idx) == dict_vectors.end()) {
		//! We need to set the dictionary data for this column
		auto base_vector = make_unique<Vector>(vector.GetType(), array.dictionary->length);
		SetValidityMask(*base_vector, *array.dictionary, scan_state, array.dictionary->length, 0, array.null_count > 0);
		ColumnArrowToDuckDB(*base_vector, *array.dictionary, scan_state, array.dictionary->length, arrow_convert_data,
		                    col_idx, arrow_convert_idx);
		dict_vectors[col_idx] = std::move(base_vector);
	}
	auto dictionary_type = arrow_convert_data[col_idx]->dictionary_type;
	//! Get Pointer to Indices of Dictionary
	auto indices = (data_ptr_t)array.buffers[1] +
	               GetTypeIdSize(dictionary_type.InternalType()) * (scan_state.chunk_offset + array.offset);
	if (array.null_count > 0) {
		ValidityMask indices_validity;
		GetValidityMask(indices_validity, array, scan_state, size);
		SetSelectionVector(sel, indices, dictionary_type, size, &indices_validity, array.dictionary->length);
	} else {
		SetSelectionVector(sel, indices, dictionary_type, size);
	}
	vector.Slice(*dict_vectors[col_idx], sel, size);
}

void ArrowTableFunction::ArrowToDuckDB(ArrowScanLocalState &scan_state,
                                       unordered_map<idx_t, unique_ptr<ArrowConvertData>> &arrow_convert_data,
                                       DataChunk &output, idx_t start, bool arrow_scan_is_projected) {
	for (idx_t idx = 0; idx < output.ColumnCount(); idx++) {
		auto col_idx = scan_state.column_ids[idx];

		// If projection was not pushed down into the arrow scanner, but projection pushdown is enabled on the
		// table function, we need to use original column ids here.
		auto arrow_array_idx = arrow_scan_is_projected ? idx : col_idx;

		if (col_idx == COLUMN_IDENTIFIER_ROW_ID) {
			continue;
		}

		std::pair<idx_t, idx_t> arrow_convert_idx {0, 0};
		auto &array = *scan_state.chunk->arrow_array.children[arrow_array_idx];
		if (!array.release) {
			throw InvalidInputException("arrow_scan: released array passed");
		}
		if (array.length != scan_state.chunk->arrow_array.length) {
			throw InvalidInputException("arrow_scan: array length mismatch");
		}
		output.data[idx].GetBuffer()->SetAuxiliaryData(make_unique<ArrowAuxiliaryData>(scan_state.chunk));
		if (array.dictionary) {
			ColumnArrowToDuckDBDictionary(output.data[idx], array, scan_state, output.size(), arrow_convert_data,
			                              col_idx, arrow_convert_idx);
		} else {
			SetValidityMask(output.data[idx], array, scan_state, output.size(), -1);
			ColumnArrowToDuckDB(output.data[idx], array, scan_state, output.size(), arrow_convert_data, col_idx,
			                    arrow_convert_idx);
		}
	}
}

} // namespace duckdb







namespace duckdb {

struct CheckpointBindData : public FunctionData {
	explicit CheckpointBindData(AttachedDatabase *db) : db(db) {
	}

	AttachedDatabase *db;

public:
	unique_ptr<FunctionData> Copy() const override {
		return make_unique<CheckpointBindData>(db);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const CheckpointBindData &)other_p;
		return db == other.db;
	}
};

static unique_ptr<FunctionData> CheckpointBind(ClientContext &context, TableFunctionBindInput &input,
                                               vector<LogicalType> &return_types, vector<string> &names) {
	return_types.emplace_back(LogicalType::BOOLEAN);
	names.emplace_back("Success");

	AttachedDatabase *db;
	auto &db_manager = DatabaseManager::Get(context);
	if (!input.inputs.empty()) {
		db = db_manager.GetDatabase(context, StringValue::Get(input.inputs[0]));
	} else {
		db = db_manager.GetDatabase(context, DatabaseManager::GetDefaultDatabase(context));
	}
	return make_unique<CheckpointBindData>(db);
}

template <bool FORCE>
static void TemplatedCheckpointFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = (CheckpointBindData &)*data_p.bind_data;
	auto &transaction_manager = TransactionManager::Get(*bind_data.db);
	transaction_manager.Checkpoint(context, FORCE);
}

void CheckpointFunction::RegisterFunction(BuiltinFunctions &set) {
	TableFunctionSet checkpoint("checkpoint");
	checkpoint.AddFunction(TableFunction({}, TemplatedCheckpointFunction<false>, CheckpointBind));
	checkpoint.AddFunction(TableFunction({LogicalType::VARCHAR}, TemplatedCheckpointFunction<false>, CheckpointBind));
	set.AddFunction(checkpoint);

	TableFunctionSet force_checkpoint("force_checkpoint");
	force_checkpoint.AddFunction(TableFunction({}, TemplatedCheckpointFunction<true>, CheckpointBind));
	force_checkpoint.AddFunction(
	    TableFunction({LogicalType::VARCHAR}, TemplatedCheckpointFunction<true>, CheckpointBind));
	set.AddFunction(force_checkpoint);
}

} // namespace duckdb













#include <limits>

namespace duckdb {

void SubstringDetection(string &str_1, string &str_2, const string &name_str_1, const string &name_str_2) {
	if (str_1.empty() || str_2.empty()) {
		return;
	}
	if ((str_1.find(str_2) != string::npos || str_2.find(str_1) != std::string::npos)) {
		throw BinderException("%s must not appear in the %s specification and vice versa", name_str_1, name_str_2);
	}
}

//===--------------------------------------------------------------------===//
// Bind
//===--------------------------------------------------------------------===//

void BaseCSVData::Finalize() {
	// verify that the options are correct in the final pass
	if (options.escape.empty()) {
		options.escape = options.quote;
	}
	// escape and delimiter must not be substrings of each other
	if (options.has_delimiter && options.has_escape) {
		SubstringDetection(options.delimiter, options.escape, "DELIMITER", "ESCAPE");
	}
	// delimiter and quote must not be substrings of each other
	if (options.has_quote && options.has_delimiter) {
		SubstringDetection(options.quote, options.delimiter, "DELIMITER", "QUOTE");
	}
	// escape and quote must not be substrings of each other (but can be the same)
	if (options.quote != options.escape && options.has_quote && options.has_escape) {
		SubstringDetection(options.quote, options.escape, "QUOTE", "ESCAPE");
	}
	if (!options.null_str.empty()) {
		// null string and delimiter must not be substrings of each other
		if (options.has_delimiter) {
			SubstringDetection(options.delimiter, options.null_str, "DELIMITER", "NULL");
		}
		// quote/escape and nullstr must not be substrings of each other
		if (options.has_quote) {
			SubstringDetection(options.quote, options.null_str, "QUOTE", "NULL");
		}
		if (options.has_escape) {
			SubstringDetection(options.escape, options.null_str, "ESCAPE", "NULL");
		}
	}
}

static unique_ptr<FunctionData> WriteCSVBind(ClientContext &context, CopyInfo &info, vector<string> &names,
                                             vector<LogicalType> &sql_types) {
	auto bind_data = make_unique<WriteCSVData>(info.file_path, sql_types, names);

	// check all the options in the copy info
	for (auto &option : info.options) {
		auto loption = StringUtil::Lower(option.first);
		auto &set = option.second;
		bind_data->options.SetWriteOption(loption, ConvertVectorToValue(std::move(set)));
	}
	// verify the parsed options
	if (bind_data->options.force_quote.empty()) {
		// no FORCE_QUOTE specified: initialize to false
		bind_data->options.force_quote.resize(names.size(), false);
	}
	bind_data->Finalize();
	bind_data->is_simple = bind_data->options.delimiter.size() == 1 && bind_data->options.escape.size() == 1 &&
	                       bind_data->options.quote.size() == 1;
	return std::move(bind_data);
}

static unique_ptr<FunctionData> ReadCSVBind(ClientContext &context, CopyInfo &info, vector<string> &expected_names,
                                            vector<LogicalType> &expected_types) {
	auto bind_data = make_unique<ReadCSVData>();
	bind_data->sql_types = expected_types;

	string file_pattern = info.file_path;
	vector<string> patterns {file_pattern};

	bind_data->InitializeFiles(context, patterns);

	auto &options = bind_data->options;

	// check all the options in the copy info
	for (auto &option : info.options) {
		auto loption = StringUtil::Lower(option.first);
		auto &set = option.second;
		options.SetReadOption(loption, ConvertVectorToValue(std::move(set)), expected_names);
	}
	// verify the parsed options
	if (options.force_not_null.empty()) {
		// no FORCE_QUOTE specified: initialize to false
		options.force_not_null.resize(expected_types.size(), false);
	}
	bind_data->FinalizeRead(context);
	if (!bind_data->single_threaded && options.auto_detect) {
		options.file_path = bind_data->files[0];
		auto initial_reader = make_unique<BufferedCSVReader>(context, options);
		options = initial_reader->options;
	}
	return std::move(bind_data);
}

//===--------------------------------------------------------------------===//
// Helper writing functions
//===--------------------------------------------------------------------===//
static string AddEscapes(string &to_be_escaped, const string &escape, const string &val) {
	idx_t i = 0;
	string new_val = "";
	idx_t found = val.find(to_be_escaped);

	while (found != string::npos) {
		while (i < found) {
			new_val += val[i];
			i++;
		}
		new_val += escape;
		found = val.find(to_be_escaped, found + escape.length());
	}
	while (i < val.length()) {
		new_val += val[i];
		i++;
	}
	return new_val;
}

static bool RequiresQuotes(WriteCSVData &csv_data, const char *str, idx_t len) {
	auto &options = csv_data.options;
	// check if the string is equal to the null string
	if (len == options.null_str.size() && memcmp(str, options.null_str.c_str(), len) == 0) {
		return true;
	}
	if (csv_data.is_simple) {
		// simple CSV: check for newlines, quotes and delimiter all at once
		for (idx_t i = 0; i < len; i++) {
			if (str[i] == '\n' || str[i] == '\r' || str[i] == options.quote[0] || str[i] == options.delimiter[0]) {
				// newline, write a quoted string
				return true;
			}
		}
		// no newline, quote or delimiter in the string
		// no quoting or escaping necessary
		return false;
	} else {
		// CSV with complex quotes/delimiter (multiple bytes)

		// first check for \n, \r, \n\r in string
		for (idx_t i = 0; i < len; i++) {
			if (str[i] == '\n' || str[i] == '\r') {
				// newline, write a quoted string
				return true;
			}
		}

		// check for delimiter
		if (options.delimiter.length() != 0 &&
		    ContainsFun::Find((const unsigned char *)str, len, (const unsigned char *)options.delimiter.c_str(),
		                      options.delimiter.size()) != DConstants::INVALID_INDEX) {
			return true;
		}
		// check for quote
		if (options.quote.length() != 0 &&
		    ContainsFun::Find((const unsigned char *)str, len, (const unsigned char *)options.quote.c_str(),
		                      options.quote.size()) != DConstants::INVALID_INDEX) {
			return true;
		}
		return false;
	}
}

static void WriteQuotedString(Serializer &serializer, WriteCSVData &csv_data, const char *str, idx_t len,
                              bool force_quote) {
	auto &options = csv_data.options;
	if (!force_quote) {
		// force quote is disabled: check if we need to add quotes anyway
		force_quote = RequiresQuotes(csv_data, str, len);
	}
	if (force_quote) {
		// quoting is enabled: we might need to escape things in the string
		bool requires_escape = false;
		if (csv_data.is_simple) {
			// simple CSV
			// do a single loop to check for a quote or escape value
			for (idx_t i = 0; i < len; i++) {
				if (str[i] == options.quote[0] || str[i] == options.escape[0]) {
					requires_escape = true;
					break;
				}
			}
		} else {
			// complex CSV
			// check for quote or escape separately
			if (options.quote.length() != 0 &&
			    ContainsFun::Find((const unsigned char *)str, len, (const unsigned char *)options.quote.c_str(),
			                      options.quote.size()) != DConstants::INVALID_INDEX) {
				requires_escape = true;
			} else if (options.escape.length() != 0 &&
			           ContainsFun::Find((const unsigned char *)str, len, (const unsigned char *)options.escape.c_str(),
			                             options.escape.size()) != DConstants::INVALID_INDEX) {
				requires_escape = true;
			}
		}
		if (!requires_escape) {
			// fast path: no need to escape anything
			serializer.WriteBufferData(options.quote);
			serializer.WriteData((const_data_ptr_t)str, len);
			serializer.WriteBufferData(options.quote);
			return;
		}

		// slow path: need to add escapes
		string new_val(str, len);
		new_val = AddEscapes(options.escape, options.escape, new_val);
		if (options.escape != options.quote) {
			// need to escape quotes separately
			new_val = AddEscapes(options.quote, options.escape, new_val);
		}
		serializer.WriteBufferData(options.quote);
		serializer.WriteBufferData(new_val);
		serializer.WriteBufferData(options.quote);
	} else {
		serializer.WriteData((const_data_ptr_t)str, len);
	}
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
struct LocalReadCSVData : public LocalFunctionData {
	//! The thread-local buffer to write data into
	BufferedSerializer serializer;
	//! A chunk with VARCHAR columns to cast intermediates into
	DataChunk cast_chunk;
};

struct GlobalWriteCSVData : public GlobalFunctionData {
	GlobalWriteCSVData(FileSystem &fs, const string &file_path, FileOpener *opener, FileCompressionType compression)
	    : fs(fs) {
		handle = fs.OpenFile(file_path, FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_FILE_CREATE_NEW,
		                     FileLockType::WRITE_LOCK, compression, opener);
	}

	void WriteData(const_data_ptr_t data, idx_t size) {
		lock_guard<mutex> flock(lock);
		handle->Write((void *)data, size);
	}

	FileSystem &fs;
	//! The mutex for writing to the physical file
	mutex lock;
	//! The file handle to write to
	unique_ptr<FileHandle> handle;
};

static unique_ptr<LocalFunctionData> WriteCSVInitializeLocal(ExecutionContext &context, FunctionData &bind_data) {
	auto &csv_data = (WriteCSVData &)bind_data;
	auto local_data = make_unique<LocalReadCSVData>();

	// create the chunk with VARCHAR types
	vector<LogicalType> types;
	types.resize(csv_data.options.names.size(), LogicalType::VARCHAR);

	local_data->cast_chunk.Initialize(Allocator::Get(context.client), types);
	return std::move(local_data);
}

static unique_ptr<GlobalFunctionData> WriteCSVInitializeGlobal(ClientContext &context, FunctionData &bind_data,
                                                               const string &file_path) {
	auto &csv_data = (WriteCSVData &)bind_data;
	auto &options = csv_data.options;
	auto global_data = make_unique<GlobalWriteCSVData>(FileSystem::GetFileSystem(context), file_path,
	                                                   FileSystem::GetFileOpener(context), options.compression);

	if (options.header) {
		BufferedSerializer serializer;
		// write the header line to the file
		for (idx_t i = 0; i < csv_data.options.names.size(); i++) {
			if (i != 0) {
				serializer.WriteBufferData(options.delimiter);
			}
			WriteQuotedString(serializer, csv_data, csv_data.options.names[i].c_str(), csv_data.options.names[i].size(),
			                  false);
		}
		serializer.WriteBufferData(csv_data.newline);

		global_data->WriteData(serializer.blob.data.get(), serializer.blob.size);
	}
	return std::move(global_data);
}

static void WriteCSVSink(ExecutionContext &context, FunctionData &bind_data, GlobalFunctionData &gstate,
                         LocalFunctionData &lstate, DataChunk &input) {
	auto &csv_data = (WriteCSVData &)bind_data;
	auto &options = csv_data.options;
	auto &local_data = (LocalReadCSVData &)lstate;
	auto &global_state = (GlobalWriteCSVData &)gstate;

	// write data into the local buffer

	// first cast the columns of the chunk to varchar
	auto &cast_chunk = local_data.cast_chunk;
	cast_chunk.Reset();
	cast_chunk.SetCardinality(input);
	for (idx_t col_idx = 0; col_idx < input.ColumnCount(); col_idx++) {
		if (csv_data.sql_types[col_idx].id() == LogicalTypeId::VARCHAR) {
			// VARCHAR, just reinterpret (cannot reference, because LogicalTypeId::VARCHAR is used by the JSON type too)
			cast_chunk.data[col_idx].Reinterpret(input.data[col_idx]);
		} else if (options.has_format[LogicalTypeId::DATE] && csv_data.sql_types[col_idx].id() == LogicalTypeId::DATE) {
			// use the date format to cast the chunk
			csv_data.options.write_date_format[LogicalTypeId::DATE].ConvertDateVector(
			    input.data[col_idx], cast_chunk.data[col_idx], input.size());
		} else if (options.has_format[LogicalTypeId::TIMESTAMP] &&
		           csv_data.sql_types[col_idx].id() == LogicalTypeId::TIMESTAMP) {
			// use the timestamp format to cast the chunk
			csv_data.options.write_date_format[LogicalTypeId::TIMESTAMP].ConvertTimestampVector(
			    input.data[col_idx], cast_chunk.data[col_idx], input.size());
		} else {
			// non varchar column, perform the cast
			VectorOperations::Cast(context.client, input.data[col_idx], cast_chunk.data[col_idx], input.size());
		}
	}

	cast_chunk.Flatten();
	auto &writer = local_data.serializer;
	// now loop over the vectors and output the values
	for (idx_t row_idx = 0; row_idx < cast_chunk.size(); row_idx++) {
		// write values
		for (idx_t col_idx = 0; col_idx < cast_chunk.ColumnCount(); col_idx++) {
			if (col_idx != 0) {
				writer.WriteBufferData(options.delimiter);
			}
			if (FlatVector::IsNull(cast_chunk.data[col_idx], row_idx)) {
				// write null value
				writer.WriteBufferData(options.null_str);
				continue;
			}

			// non-null value, fetch the string value from the cast chunk
			auto str_data = FlatVector::GetData<string_t>(cast_chunk.data[col_idx]);
			auto str_value = str_data[row_idx];
			// FIXME: we could gain some performance here by checking for certain types if they ever require quotes
			// (e.g. integers only require quotes if the delimiter is a number, decimals only require quotes if the
			// delimiter is a number or "." character)
			WriteQuotedString(writer, csv_data, str_value.GetDataUnsafe(), str_value.GetSize(),
			                  csv_data.options.force_quote[col_idx]);
		}
		writer.WriteBufferData(csv_data.newline);
	}
	// check if we should flush what we have currently written
	if (writer.blob.size >= csv_data.flush_size) {
		global_state.WriteData(writer.blob.data.get(), writer.blob.size);
		writer.Reset();
	}
}

//===--------------------------------------------------------------------===//
// Combine
//===--------------------------------------------------------------------===//
static void WriteCSVCombine(ExecutionContext &context, FunctionData &bind_data, GlobalFunctionData &gstate,
                            LocalFunctionData &lstate) {
	auto &local_data = (LocalReadCSVData &)lstate;
	auto &global_state = (GlobalWriteCSVData &)gstate;
	auto &writer = local_data.serializer;
	// flush the local writer
	if (writer.blob.size > 0) {
		global_state.WriteData(writer.blob.data.get(), writer.blob.size);
		writer.Reset();
	}
}

//===--------------------------------------------------------------------===//
// Finalize
//===--------------------------------------------------------------------===//
void WriteCSVFinalize(ClientContext &context, FunctionData &bind_data, GlobalFunctionData &gstate) {
	auto &global_state = (GlobalWriteCSVData &)gstate;

	global_state.handle->Close();
	global_state.handle.reset();
}

//===--------------------------------------------------------------------===//
// Parallel
//===--------------------------------------------------------------------===//
bool WriteCSVIsParallel(ClientContext &context, FunctionData &bind_data) {
	auto &config = DBConfig::GetConfig(context);
	if (config.options.preserve_insertion_order) {
		return false;
	}
	return true;
}

void CSVCopyFunction::RegisterFunction(BuiltinFunctions &set) {
	CopyFunction info("csv");
	info.copy_to_bind = WriteCSVBind;
	info.copy_to_initialize_local = WriteCSVInitializeLocal;
	info.copy_to_initialize_global = WriteCSVInitializeGlobal;
	info.copy_to_sink = WriteCSVSink;
	info.copy_to_combine = WriteCSVCombine;
	info.copy_to_finalize = WriteCSVFinalize;
	info.parallel = WriteCSVIsParallel;

	info.copy_from_bind = ReadCSVBind;
	info.copy_from_function = ReadCSVTableFunction::GetFunction();

	info.extension = "csv";

	set.AddFunction(info);
}

} // namespace duckdb






namespace duckdb {

struct GlobFunctionBindData : public TableFunctionData {
	vector<string> files;
};

static unique_ptr<FunctionData> GlobFunctionBind(ClientContext &context, TableFunctionBindInput &input,
                                                 vector<LogicalType> &return_types, vector<string> &names) {
	auto &config = DBConfig::GetConfig(context);
	if (!config.options.enable_external_access) {
		throw PermissionException("Globbing is disabled through configuration");
	}
	auto result = make_unique<GlobFunctionBindData>();
	auto &fs = FileSystem::GetFileSystem(context);
	result->files = fs.Glob(StringValue::Get(input.inputs[0]), context);
	return_types.emplace_back(LogicalType::VARCHAR);
	names.emplace_back("file");
	return std::move(result);
}

struct GlobFunctionState : public GlobalTableFunctionState {
	GlobFunctionState() : current_idx(0) {
	}

	idx_t current_idx;
};

static unique_ptr<GlobalTableFunctionState> GlobFunctionInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_unique<GlobFunctionState>();
}

static void GlobFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = (GlobFunctionBindData &)*data_p.bind_data;
	auto &state = (GlobFunctionState &)*data_p.global_state;

	idx_t count = 0;
	idx_t next_idx = MinValue<idx_t>(state.current_idx + STANDARD_VECTOR_SIZE, bind_data.files.size());
	for (; state.current_idx < next_idx; state.current_idx++) {
		output.data[0].SetValue(count, bind_data.files[state.current_idx]);
		count++;
	}
	output.SetCardinality(count);
}

void GlobTableFunction::RegisterFunction(BuiltinFunctions &set) {
	TableFunctionSet glob("glob");
	glob.AddFunction(TableFunction({LogicalType::VARCHAR}, GlobFunction, GlobFunctionBind, GlobFunctionInit));
	set.AddFunction(glob);
}

} // namespace duckdb










namespace duckdb {

struct PragmaDetailedProfilingOutputOperatorData : public GlobalTableFunctionState {
	explicit PragmaDetailedProfilingOutputOperatorData() : initialized(false) {
	}

	ColumnDataScanState scan_state;
	bool initialized;
};

struct PragmaDetailedProfilingOutputData : public TableFunctionData {
	explicit PragmaDetailedProfilingOutputData(vector<LogicalType> &types) : types(types) {
	}
	unique_ptr<ColumnDataCollection> collection;
	vector<LogicalType> types;
};

static unique_ptr<FunctionData> PragmaDetailedProfilingOutputBind(ClientContext &context, TableFunctionBindInput &input,
                                                                  vector<LogicalType> &return_types,
                                                                  vector<string> &names) {
	names.emplace_back("OPERATOR_ID");
	return_types.emplace_back(LogicalType::INTEGER);

	names.emplace_back("ANNOTATION");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("ID");
	return_types.emplace_back(LogicalType::INTEGER);

	names.emplace_back("NAME");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("TIME");
	return_types.emplace_back(LogicalType::DOUBLE);

	names.emplace_back("CYCLES_PER_TUPLE");
	return_types.emplace_back(LogicalType::DOUBLE);

	names.emplace_back("SAMPLE_SIZE");
	return_types.emplace_back(LogicalType::INTEGER);

	names.emplace_back("INPUT_SIZE");
	return_types.emplace_back(LogicalType::INTEGER);

	names.emplace_back("EXTRA_INFO");
	return_types.emplace_back(LogicalType::VARCHAR);

	return make_unique<PragmaDetailedProfilingOutputData>(return_types);
}

unique_ptr<GlobalTableFunctionState> PragmaDetailedProfilingOutputInit(ClientContext &context,
                                                                       TableFunctionInitInput &input) {
	return make_unique<PragmaDetailedProfilingOutputOperatorData>();
}

// Insert a row into the given datachunk
static void SetValue(DataChunk &output, int index, int op_id, string annotation, int id, string name, double time,
                     int sample_counter, int tuple_counter, string extra_info) {
	output.SetValue(0, index, op_id);
	output.SetValue(1, index, std::move(annotation));
	output.SetValue(2, index, id);
	output.SetValue(3, index, std::move(name));
#if defined(RDTSC)
	output.SetValue(4, index, Value(nullptr));
	output.SetValue(5, index, time);
#else
	output.SetValue(4, index, time);
	output.SetValue(5, index, Value(nullptr));

#endif
	output.SetValue(6, index, sample_counter);
	output.SetValue(7, index, tuple_counter);
	output.SetValue(8, index, std::move(extra_info));
}

static void ExtractFunctions(ColumnDataCollection &collection, ExpressionInfo &info, DataChunk &chunk, int op_id,
                             int &fun_id) {
	if (info.hasfunction) {
		D_ASSERT(info.sample_tuples_count != 0);
		SetValue(chunk, chunk.size(), op_id, "Function", fun_id++, info.function_name,
		         int(info.function_time) / double(info.sample_tuples_count), info.sample_tuples_count,
		         info.tuples_count, "");

		chunk.SetCardinality(chunk.size() + 1);
		if (chunk.size() == STANDARD_VECTOR_SIZE) {
			collection.Append(chunk);
			chunk.Reset();
		}
	}
	if (info.children.empty()) {
		return;
	}
	// extract the children of this node
	for (auto &child : info.children) {
		ExtractFunctions(collection, *child, chunk, op_id, fun_id);
	}
}

static void PragmaDetailedProfilingOutputFunction(ClientContext &context, TableFunctionInput &data_p,
                                                  DataChunk &output) {
	auto &state = (PragmaDetailedProfilingOutputOperatorData &)*data_p.global_state;
	auto &data = (PragmaDetailedProfilingOutputData &)*data_p.bind_data;

	if (!state.initialized) {
		// create a ColumnDataCollection
		auto collection = make_unique<ColumnDataCollection>(context, data.types);

		// create a chunk
		DataChunk chunk;
		chunk.Initialize(context, data.types);

		// Initialize ids
		int operator_counter = 1;
		int function_counter = 1;
		int expression_counter = 1;
		if (ClientData::Get(context).query_profiler_history->GetPrevProfilers().empty()) {
			return;
		}
		// For each Operator
		for (auto op :
		     ClientData::Get(context).query_profiler_history->GetPrevProfilers().back().second->GetTreeMap()) {
			// For each Expression Executor
			for (auto &expr_executor : op.second->info.executors_info) {
				// For each Expression tree
				if (!expr_executor) {
					continue;
				}
				for (auto &expr_timer : expr_executor->roots) {
					D_ASSERT(expr_timer->sample_tuples_count != 0);
					SetValue(chunk, chunk.size(), operator_counter, "ExpressionRoot", expression_counter++,
					         // Sometimes, cycle counter is not accurate, too big or too small. return 0 for
					         // those cases
					         expr_timer->name, int(expr_timer->time) / double(expr_timer->sample_tuples_count),
					         expr_timer->sample_tuples_count, expr_timer->tuples_count, expr_timer->extra_info);
					// Increment cardinality
					chunk.SetCardinality(chunk.size() + 1);
					// Check whether data chunk is full or not
					if (chunk.size() == STANDARD_VECTOR_SIZE) {
						collection->Append(chunk);
						chunk.Reset();
					}
					// Extract all functions inside the tree
					ExtractFunctions(*collection, *expr_timer->root, chunk, operator_counter, function_counter);
				}
			}
			operator_counter++;
		}
		collection->Append(chunk);
		data.collection = std::move(collection);
		data.collection->InitializeScan(state.scan_state);
		state.initialized = true;
	}

	data.collection->Scan(state.scan_state, output);
}

void PragmaDetailedProfilingOutput::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("pragma_detailed_profiling_output", {}, PragmaDetailedProfilingOutputFunction,
	                              PragmaDetailedProfilingOutputBind, PragmaDetailedProfilingOutputInit));
}

} // namespace duckdb










namespace duckdb {

struct PragmaLastProfilingOutputOperatorData : public GlobalTableFunctionState {
	PragmaLastProfilingOutputOperatorData() : initialized(false) {
	}

	ColumnDataScanState scan_state;
	bool initialized;
};

struct PragmaLastProfilingOutputData : public TableFunctionData {
	explicit PragmaLastProfilingOutputData(vector<LogicalType> &types) : types(types) {
	}
	unique_ptr<ColumnDataCollection> collection;
	vector<LogicalType> types;
};

static unique_ptr<FunctionData> PragmaLastProfilingOutputBind(ClientContext &context, TableFunctionBindInput &input,
                                                              vector<LogicalType> &return_types,
                                                              vector<string> &names) {
	names.emplace_back("OPERATOR_ID");
	return_types.emplace_back(LogicalType::INTEGER);

	names.emplace_back("NAME");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("TIME");
	return_types.emplace_back(LogicalType::DOUBLE);

	names.emplace_back("CARDINALITY");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("DESCRIPTION");
	return_types.emplace_back(LogicalType::VARCHAR);

	return make_unique<PragmaLastProfilingOutputData>(return_types);
}

static void SetValue(DataChunk &output, int index, int op_id, string name, double time, int64_t car,
                     string description) {
	output.SetValue(0, index, op_id);
	output.SetValue(1, index, std::move(name));
	output.SetValue(2, index, time);
	output.SetValue(3, index, car);
	output.SetValue(4, index, std::move(description));
}

unique_ptr<GlobalTableFunctionState> PragmaLastProfilingOutputInit(ClientContext &context,
                                                                   TableFunctionInitInput &input) {
	return make_unique<PragmaLastProfilingOutputOperatorData>();
}

static void PragmaLastProfilingOutputFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &state = (PragmaLastProfilingOutputOperatorData &)*data_p.global_state;
	auto &data = (PragmaLastProfilingOutputData &)*data_p.bind_data;
	if (!state.initialized) {
		// create a ColumnDataCollection
		auto collection = make_unique<ColumnDataCollection>(context, data.types);

		DataChunk chunk;
		chunk.Initialize(context, data.types);
		int operator_counter = 1;
		if (!ClientData::Get(context).query_profiler_history->GetPrevProfilers().empty()) {
			for (auto op :
			     ClientData::Get(context).query_profiler_history->GetPrevProfilers().back().second->GetTreeMap()) {
				SetValue(chunk, chunk.size(), operator_counter++, op.second->name, op.second->info.time,
				         op.second->info.elements, " ");
				chunk.SetCardinality(chunk.size() + 1);
				if (chunk.size() == STANDARD_VECTOR_SIZE) {
					collection->Append(chunk);
					chunk.Reset();
				}
			}
		}
		collection->Append(chunk);
		data.collection = std::move(collection);
		data.collection->InitializeScan(state.scan_state);
		state.initialized = true;
	}

	data.collection->Scan(state.scan_state, output);
}

void PragmaLastProfilingOutput::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("pragma_last_profiling_output", {}, PragmaLastProfilingOutputFunction,
	                              PragmaLastProfilingOutputBind, PragmaLastProfilingOutputInit));
}

} // namespace duckdb








namespace duckdb {

//===--------------------------------------------------------------------===//
// Range (integers)
//===--------------------------------------------------------------------===//
struct RangeFunctionBindData : public TableFunctionData {
	hugeint_t start;
	hugeint_t end;
	hugeint_t increment;

public:
	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const RangeFunctionBindData &)other_p;
		return other.start == start && other.end == end && other.increment == increment;
	}
};

template <bool GENERATE_SERIES>
static void GenerateRangeParameters(const vector<Value> &inputs, RangeFunctionBindData &result) {
	for (auto &input : inputs) {
		if (input.IsNull()) {
			result.start = GENERATE_SERIES ? 1 : 0;
			result.end = 0;
			result.increment = 1;
			return;
		}
	}
	if (inputs.size() < 2) {
		// single argument: only the end is specified
		result.start = 0;
		result.end = inputs[0].GetValue<int64_t>();
	} else {
		// two arguments: first two arguments are start and end
		result.start = inputs[0].GetValue<int64_t>();
		result.end = inputs[1].GetValue<int64_t>();
	}
	if (inputs.size() < 3) {
		result.increment = 1;
	} else {
		result.increment = inputs[2].GetValue<int64_t>();
	}
	if (result.increment == 0) {
		throw BinderException("interval cannot be 0!");
	}
	if (result.start > result.end && result.increment > 0) {
		throw BinderException("start is bigger than end, but increment is positive: cannot generate infinite series");
	} else if (result.start < result.end && result.increment < 0) {
		throw BinderException("start is smaller than end, but increment is negative: cannot generate infinite series");
	}
}

template <bool GENERATE_SERIES>
static unique_ptr<FunctionData> RangeFunctionBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {
	auto result = make_unique<RangeFunctionBindData>();
	auto &inputs = input.inputs;
	GenerateRangeParameters<GENERATE_SERIES>(inputs, *result);

	return_types.emplace_back(LogicalType::BIGINT);
	if (GENERATE_SERIES) {
		// generate_series has inclusive bounds on the RHS
		if (result->increment < 0) {
			result->end = result->end - 1;
		} else {
			result->end = result->end + 1;
		}
		names.emplace_back("generate_series");
	} else {
		names.emplace_back("range");
	}
	return std::move(result);
}

struct RangeFunctionState : public GlobalTableFunctionState {
	RangeFunctionState() : current_idx(0) {
	}

	int64_t current_idx;
};

static unique_ptr<GlobalTableFunctionState> RangeFunctionInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_unique<RangeFunctionState>();
}

static void RangeFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = (RangeFunctionBindData &)*data_p.bind_data;
	auto &state = (RangeFunctionState &)*data_p.global_state;

	auto increment = bind_data.increment;
	auto end = bind_data.end;
	hugeint_t current_value = bind_data.start + increment * state.current_idx;
	int64_t current_value_i64;
	if (!Hugeint::TryCast<int64_t>(current_value, current_value_i64)) {
		return;
	}
	int64_t offset = increment < 0 ? 1 : -1;
	idx_t remaining = MinValue<idx_t>(Hugeint::Cast<idx_t>((end - current_value + (increment + offset)) / increment),
	                                  STANDARD_VECTOR_SIZE);
	// set the result vector as a sequence vector
	output.data[0].Sequence(current_value_i64, Hugeint::Cast<int64_t>(increment), remaining);
	// increment the index pointer by the remaining count
	state.current_idx += remaining;
	output.SetCardinality(remaining);
}

unique_ptr<NodeStatistics> RangeCardinality(ClientContext &context, const FunctionData *bind_data_p) {
	auto &bind_data = (RangeFunctionBindData &)*bind_data_p;
	idx_t cardinality = Hugeint::Cast<idx_t>((bind_data.end - bind_data.start) / bind_data.increment);
	return make_unique<NodeStatistics>(cardinality, cardinality);
}

//===--------------------------------------------------------------------===//
// Range (timestamp)
//===--------------------------------------------------------------------===//
struct RangeDateTimeBindData : public TableFunctionData {
	timestamp_t start;
	timestamp_t end;
	interval_t increment;
	bool inclusive_bound;
	bool greater_than_check;

public:
	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const RangeDateTimeBindData &)other_p;
		return other.start == start && other.end == end && other.increment == increment &&
		       other.inclusive_bound == inclusive_bound && other.greater_than_check == greater_than_check;
	}

	bool Finished(timestamp_t current_value) {
		if (greater_than_check) {
			if (inclusive_bound) {
				return current_value > end;
			} else {
				return current_value >= end;
			}
		} else {
			if (inclusive_bound) {
				return current_value < end;
			} else {
				return current_value <= end;
			}
		}
	}
};

template <bool GENERATE_SERIES>
static unique_ptr<FunctionData> RangeDateTimeBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {
	auto result = make_unique<RangeDateTimeBindData>();
	auto &inputs = input.inputs;
	D_ASSERT(inputs.size() == 3);
	result->start = inputs[0].GetValue<timestamp_t>();
	result->end = inputs[1].GetValue<timestamp_t>();
	result->increment = inputs[2].GetValue<interval_t>();

	// Infinities either cause errors or infinite loops, so just ban them
	if (!Timestamp::IsFinite(result->start) || !Timestamp::IsFinite(result->end)) {
		throw BinderException("RANGE with infinite bounds is not supported");
	}

	if (result->increment.months == 0 && result->increment.days == 0 && result->increment.micros == 0) {
		throw BinderException("interval cannot be 0!");
	}
	// all elements should point in the same direction
	if (result->increment.months > 0 || result->increment.days > 0 || result->increment.micros > 0) {
		if (result->increment.months < 0 || result->increment.days < 0 || result->increment.micros < 0) {
			throw BinderException("RANGE with composite interval that has mixed signs is not supported");
		}
		result->greater_than_check = true;
		if (result->start > result->end) {
			throw BinderException(
			    "start is bigger than end, but increment is positive: cannot generate infinite series");
		}
	} else {
		result->greater_than_check = false;
		if (result->start < result->end) {
			throw BinderException(
			    "start is smaller than end, but increment is negative: cannot generate infinite series");
		}
	}
	return_types.push_back(inputs[0].type());
	if (GENERATE_SERIES) {
		// generate_series has inclusive bounds on the RHS
		result->inclusive_bound = true;
		names.emplace_back("generate_series");
	} else {
		result->inclusive_bound = false;
		names.emplace_back("range");
	}
	return std::move(result);
}

struct RangeDateTimeState : public GlobalTableFunctionState {
	explicit RangeDateTimeState(timestamp_t start_p) : current_state(start_p) {
	}

	timestamp_t current_state;
	bool finished = false;
};

static unique_ptr<GlobalTableFunctionState> RangeDateTimeInit(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind_data = (RangeDateTimeBindData &)*input.bind_data;
	return make_unique<RangeDateTimeState>(bind_data.start);
}

static void RangeDateTimeFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = (RangeDateTimeBindData &)*data_p.bind_data;
	auto &state = (RangeDateTimeState &)*data_p.global_state;
	if (state.finished) {
		return;
	}

	idx_t size = 0;
	auto data = FlatVector::GetData<timestamp_t>(output.data[0]);
	while (true) {
		data[size++] = state.current_state;
		state.current_state =
		    AddOperator::Operation<timestamp_t, interval_t, timestamp_t>(state.current_state, bind_data.increment);
		if (bind_data.Finished(state.current_state)) {
			state.finished = true;
			break;
		}
		if (size >= STANDARD_VECTOR_SIZE) {
			break;
		}
	}
	output.SetCardinality(size);
}

void RangeTableFunction::RegisterFunction(BuiltinFunctions &set) {
	TableFunctionSet range("range");

	TableFunction range_function({LogicalType::BIGINT}, RangeFunction, RangeFunctionBind<false>, RangeFunctionInit);
	range_function.cardinality = RangeCardinality;

	// single argument range: (end) - implicit start = 0 and increment = 1
	range.AddFunction(range_function);
	// two arguments range: (start, end) - implicit increment = 1
	range_function.arguments = {LogicalType::BIGINT, LogicalType::BIGINT};
	range.AddFunction(range_function);
	// three arguments range: (start, end, increment)
	range_function.arguments = {LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT};
	range.AddFunction(range_function);
	range.AddFunction(TableFunction({LogicalType::TIMESTAMP, LogicalType::TIMESTAMP, LogicalType::INTERVAL},
	                                RangeDateTimeFunction, RangeDateTimeBind<false>, RangeDateTimeInit));
	set.AddFunction(range);
	// generate_series: similar to range, but inclusive instead of exclusive bounds on the RHS
	TableFunctionSet generate_series("generate_series");
	range_function.bind = RangeFunctionBind<true>;
	range_function.arguments = {LogicalType::BIGINT};
	generate_series.AddFunction(range_function);
	range_function.arguments = {LogicalType::BIGINT, LogicalType::BIGINT};
	generate_series.AddFunction(range_function);
	range_function.arguments = {LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT};
	generate_series.AddFunction(range_function);
	generate_series.AddFunction(TableFunction({LogicalType::TIMESTAMP, LogicalType::TIMESTAMP, LogicalType::INTERVAL},
	                                          RangeDateTimeFunction, RangeDateTimeBind<true>, RangeDateTimeInit));
	set.AddFunction(generate_series);
}

void BuiltinFunctions::RegisterTableFunctions() {
	CheckpointFunction::RegisterFunction(*this);
	GlobTableFunction::RegisterFunction(*this);
	RangeTableFunction::RegisterFunction(*this);
	RepeatTableFunction::RegisterFunction(*this);
	SummaryTableFunction::RegisterFunction(*this);
	UnnestTableFunction::RegisterFunction(*this);
}

} // namespace duckdb













#include <limits>

namespace duckdb {

unique_ptr<CSVFileHandle> ReadCSV::OpenCSV(const string &file_path, FileCompressionType compression,
                                           ClientContext &context) {
	auto &fs = FileSystem::GetFileSystem(context);
	auto opener = FileSystem::GetFileOpener(context);
	auto file_handle =
	    fs.OpenFile(file_path.c_str(), FileFlags::FILE_FLAGS_READ, FileLockType::NO_LOCK, compression, opener);
	return make_unique<CSVFileHandle>(std::move(file_handle));
}

void ReadCSVData::InitializeFiles(ClientContext &context, const vector<string> &patterns) {
	auto &fs = FileSystem::GetFileSystem(context);
	for (auto &file_pattern : patterns) {
		auto found_files = fs.Glob(file_pattern, context);
		if (found_files.empty()) {
			throw IOException("No files found that match the pattern \"%s\"", file_pattern);
		}
		files.insert(files.end(), found_files.begin(), found_files.end());
	}
}

void ReadCSVData::FinalizeRead(ClientContext &context) {
	BaseCSVData::Finalize();
	auto &config = DBConfig::GetConfig(context);
	single_threaded = !config.options.experimental_parallel_csv_reader;
	if (options.has_parallel) {
		// Override the option set in the config
		single_threaded = !options.use_parallel;
	}
	bool null_or_empty = options.delimiter.empty() || options.escape.empty() || options.quote.empty() ||
	                     options.delimiter[0] == '\0' || options.escape[0] == '\0' || options.quote[0] == '\0';
	bool complex_options = options.delimiter.size() > 1 || options.escape.size() > 1 || options.quote.size() > 1;
	if (null_or_empty || complex_options || options.new_line == NewLineIdentifier::MIX) {
		// not supported for parallel CSV reading
		single_threaded = true;
	}
}

static unique_ptr<FunctionData> ReadCSVBind(ClientContext &context, TableFunctionBindInput &input,
                                            vector<LogicalType> &return_types, vector<string> &names) {
	auto &config = DBConfig::GetConfig(context);
	if (!config.options.enable_external_access) {
		throw PermissionException("Scanning CSV files is disabled through configuration");
	}

	auto result = make_unique<ReadCSVData>();
	auto &options = result->options;

	vector<string> patterns;
	if (input.inputs[0].type().id() == LogicalTypeId::LIST) {
		// list of globs
		for (auto &val : ListValue::GetChildren(input.inputs[0])) {
			patterns.push_back(StringValue::Get(val));
		}
	} else {
		// single glob pattern
		patterns.push_back(StringValue::Get(input.inputs[0]));
	}

	result->InitializeFiles(context, patterns);

	bool explicitly_set_columns = false;
	for (auto &kv : input.named_parameters) {
		auto loption = StringUtil::Lower(kv.first);
		if (loption == "columns") {
			explicitly_set_columns = true;
			auto &child_type = kv.second.type();
			if (child_type.id() != LogicalTypeId::STRUCT) {
				throw BinderException("read_csv columns requires a struct as input");
			}
			auto &struct_children = StructValue::GetChildren(kv.second);
			D_ASSERT(StructType::GetChildCount(child_type) == struct_children.size());
			for (idx_t i = 0; i < struct_children.size(); i++) {
				auto &name = StructType::GetChildName(child_type, i);
				auto &val = struct_children[i];
				names.push_back(name);
				if (val.type().id() != LogicalTypeId::VARCHAR) {
					throw BinderException("read_csv requires a type specification as string");
				}
				return_types.emplace_back(TransformStringToLogicalType(StringValue::Get(val), context));
			}
			if (names.empty()) {
				throw BinderException("read_csv requires at least a single column as input!");
			}
		} else if (loption == "column_types" || loption == "types" || loption == "dtypes") {
			auto &child_type = kv.second.type();
			if (child_type.id() != LogicalTypeId::STRUCT && child_type.id() != LogicalTypeId::LIST) {
				throw BinderException("read_csv_auto %s requires a struct or list as input", kv.first);
			}
			if (!options.sql_type_list.empty()) {
				throw BinderException("read_csv_auto column_types/types/dtypes can only be supplied once");
			}
			vector<string> sql_type_names;
			if (child_type.id() == LogicalTypeId::STRUCT) {
				auto &struct_children = StructValue::GetChildren(kv.second);
				D_ASSERT(StructType::GetChildCount(child_type) == struct_children.size());
				for (idx_t i = 0; i < struct_children.size(); i++) {
					auto &name = StructType::GetChildName(child_type, i);
					auto &val = struct_children[i];
					if (val.type().id() != LogicalTypeId::VARCHAR) {
						throw BinderException("read_csv_auto %s requires a type specification as string", kv.first);
					}
					sql_type_names.push_back(StringValue::Get(val));
					options.sql_types_per_column[name] = i;
				}
			} else {
				auto &list_child = ListType::GetChildType(child_type);
				if (list_child.id() != LogicalTypeId::VARCHAR) {
					throw BinderException("read_csv_auto %s requires a list of types (varchar) as input", kv.first);
				}
				auto &children = ListValue::GetChildren(kv.second);
				for (auto &child : children) {
					sql_type_names.push_back(StringValue::Get(child));
				}
			}
			options.sql_type_list.reserve(sql_type_names.size());
			for (auto &sql_type : sql_type_names) {
				auto def_type = TransformStringToLogicalType(sql_type);
				if (def_type.id() == LogicalTypeId::USER) {
					throw BinderException("Unrecognized type \"%s\" for read_csv_auto %s definition", sql_type,
					                      kv.first);
				}
				options.sql_type_list.push_back(std::move(def_type));
			}
		} else if (loption == "all_varchar") {
			options.all_varchar = BooleanValue::Get(kv.second);
		} else if (loption == "normalize_names") {
			options.normalize_names = BooleanValue::Get(kv.second);
		} else if (loption == "filename") {
			options.include_file_name = BooleanValue::Get(kv.second);
		} else if (loption == "hive_partitioning") {
			options.include_parsed_hive_partitions = BooleanValue::Get(kv.second);
		} else {
			options.SetReadOption(loption, kv.second, names);
		}
	}
	if (!options.auto_detect && return_types.empty()) {
		throw BinderException("read_csv requires columns to be specified through the 'columns' option. Use "
		                      "read_csv_auto or set read_csv(..., "
		                      "AUTO_DETECT=TRUE) to automatically guess columns.");
	}
	if (options.auto_detect) {
		options.file_path = result->files[0];
		auto initial_reader = make_unique<BufferedCSVReader>(context, options);
		return_types.assign(initial_reader->return_types.begin(), initial_reader->return_types.end());
		if (names.empty()) {
			names.assign(initial_reader->names.begin(), initial_reader->names.end());
		} else {
			if (explicitly_set_columns) {
				// The user has influenced the names, can't assume they are valid anymore
				if (return_types.size() != names.size()) {
					throw BinderException("The amount of names specified (%d) and the observed amount of types (%d) in "
					                      "the file don't match",
					                      names.size(), return_types.size());
				}
			} else {
				D_ASSERT(return_types.size() == names.size());
			}
		}
		options = initial_reader->options;
		result->sql_types = initial_reader->return_types;
		result->initial_reader = std::move(initial_reader);
	} else {
		result->sql_types = return_types;
		D_ASSERT(return_types.size() == names.size());
	}

	// union_col_names will exclude filename and hivepartition
	if (options.union_by_name) {
		case_insensitive_map_t<idx_t> union_names_map;
		vector<string> union_col_names;
		vector<LogicalType> union_col_types;

		auto dummy_readers = UnionByName<BufferedCSVReader, BufferedCSVReaderOptions>::UnionCols(
		    context, result->files, union_col_types, union_col_names, union_names_map, options);

		dummy_readers = UnionByName<BufferedCSVReader, BufferedCSVReaderOptions>::CreateUnionMap(
		    std::move(dummy_readers), union_col_types, union_col_names, union_names_map);

		std::move(dummy_readers.begin(), dummy_readers.end(), std::back_inserter(result->union_readers));
		for (auto &reader : result->union_readers) {
			reader->insert_cols_idx = reader->union_idx_map;
		}

		names.assign(union_col_names.begin(), union_col_names.end());
		return_types.assign(union_col_types.begin(), union_col_types.end());
		const idx_t first_file_index = 0;
		result->initial_reader = std::move(result->union_readers[first_file_index]);
		D_ASSERT(names.size() == return_types.size());

		if (!options.sql_types_per_column.empty()) {
			auto exception = BufferedCSVReader::ColumnTypesError(options.sql_types_per_column, names);
			if (!exception.empty()) {
				throw BinderException(exception);
			}
		}
	}

	if (result->options.include_file_name) {
		result->filename_col_idx = names.size();
		return_types.emplace_back(LogicalType::VARCHAR);
		names.emplace_back("filename");
	}

	if (result->options.include_parsed_hive_partitions) {
		auto partitions = HivePartitioning::Parse(result->files[0]);
		result->hive_partition_col_idx = names.size();
		for (auto &part : partitions) {
			return_types.emplace_back(LogicalType::VARCHAR);
			names.emplace_back(part.first);
		}
	}
	result->options.names = names;
	result->FinalizeRead(context);
	return std::move(result);
}

static unique_ptr<FunctionData> ReadCSVAutoBind(ClientContext &context, TableFunctionBindInput &input,
                                                vector<LogicalType> &return_types, vector<string> &names) {
	input.named_parameters["auto_detect"] = Value::BOOLEAN(true);
	return ReadCSVBind(context, input, return_types, names);
}

//===--------------------------------------------------------------------===//
// Parallel CSV Reader CSV Global State
//===--------------------------------------------------------------------===//
//===--------------------------------------------------------------------===//
// Read CSV Global State
//===--------------------------------------------------------------------===//
struct ParallelCSVGlobalState : public GlobalTableFunctionState {
public:
	ParallelCSVGlobalState(ClientContext &context, unique_ptr<CSVFileHandle> file_handle_p,
	                       vector<string> &files_path_p, idx_t system_threads_p, idx_t buffer_size_p,
	                       idx_t rows_to_skip, bool force_parallelism_p)
	    : file_handle(std::move(file_handle_p)), system_threads(system_threads_p), buffer_size(buffer_size_p),
	      force_parallelism(force_parallelism_p) {
		current_file_path = files_path_p[0];
		for (idx_t i = 0; i < rows_to_skip; i++) {
			file_handle->ReadLine();
		}
		estimated_linenr = rows_to_skip;
		file_size = file_handle->FileSize();
		first_file_size = file_size;
		bytes_read = 0;
		if (buffer_size < file_size) {
			bytes_per_local_state = buffer_size / ParallelCSVGlobalState::MaxThreads();
		} else {
			bytes_per_local_state = file_size / MaxThreads();
		}
		current_buffer = make_shared<CSVBuffer>(context, buffer_size, *file_handle, current_csv_position);
		next_buffer = current_buffer->Next(*file_handle, buffer_size, current_csv_position);
		running_threads = MaxThreads();
	}
	ParallelCSVGlobalState() {
	}

	~ParallelCSVGlobalState() override {
	}

	idx_t MaxThreads() const override;
	//! Returns buffer and index that caller thread should read.
	unique_ptr<CSVBufferRead> Next(ClientContext &context, ReadCSVData &bind_data);
	//! Verify if the CSV File was read correctly
	void Verify();

	void UpdateVerification(VerificationPositions positions);

	void IncrementThread();

	void DecrementThread();

	bool Finished();

	//! How many bytes were read up to this point
	atomic<idx_t> bytes_read;
	//! Size of current file
	idx_t file_size;

	double GetProgress(ReadCSVData &bind_data) const {
		idx_t total_files = bind_data.files.size();

		// get the progress WITHIN the current file
		double progress;
		if (file_size == 0) {
			progress = 1.0;
		} else {
			progress = double(bytes_read) / double(file_size);
		}
		// now get the total percentage of files read
		double percentage = double(file_index) / total_files;
		percentage += (double(1) / double(total_files)) * progress;
		return percentage * 100;
	}

private:
	//! File Handle for current file
	unique_ptr<CSVFileHandle> prev_file_handle;
	unique_ptr<CSVFileHandle> file_handle;
	shared_ptr<CSVBuffer> current_buffer;
	shared_ptr<CSVBuffer> next_buffer;

	//! The index of the next file to read (i.e. current file + 1)
	idx_t file_index = 1;
	string current_file_path;

	//! Mutex to lock when getting next batch of bytes (Parallel Only)
	mutex main_mutex;
	//! Byte set from for last thread
	idx_t next_byte = 0;
	//! The current estimated line number
	idx_t estimated_linenr;
	//! How many bytes we should execute per local state
	idx_t bytes_per_local_state;
	//! Size of first file
	idx_t first_file_size;
	//! Basically max number of threads in DuckDB
	idx_t system_threads;
	//! Size of the buffers
	idx_t buffer_size;
	//! Current batch index
	idx_t batch_index = 0;
	//! Forces parallelism for small CSV Files, should only be used for testing.
	bool force_parallelism;
	//! Current (Global) position of CSV
	idx_t current_csv_position = 0;
	idx_t max_tuple_end = 0;
	//! the vector stores positions where threads ended the last line they read in the CSV File, and the set stores
	//! positions where they started reading the first line.
	vector<idx_t> tuple_end;
	set<idx_t> tuple_start;
	idx_t running_threads = 0;
};

idx_t ParallelCSVGlobalState::MaxThreads() const {
	if (force_parallelism) {
		return system_threads;
	}

	idx_t one_mb = 1000000; // We initialize max one thread per Mb
	idx_t threads_per_mb = first_file_size / one_mb + 1;
	if (threads_per_mb < system_threads) {
		return threads_per_mb;
	}

	return system_threads;
}

void ParallelCSVGlobalState::IncrementThread() {
	lock_guard<mutex> parallel_lock(main_mutex);
	running_threads++;
}

void ParallelCSVGlobalState::DecrementThread() {
	lock_guard<mutex> parallel_lock(main_mutex);
	D_ASSERT(running_threads > 0);
	running_threads--;
}

bool ParallelCSVGlobalState::Finished() {
	lock_guard<mutex> parallel_lock(main_mutex);
	return running_threads == 0;
}

void ParallelCSVGlobalState::Verify() {
	// All threads are done, we run some magic sweet verification code
	if (running_threads == 0) {
		for (auto &last_pos : tuple_end) {
			auto first_pos = tuple_start.find(last_pos);
			if (first_pos == tuple_start.end()) {
				// this might be necessary due to carriage returns outside buffer scopes.
				first_pos = tuple_start.find(last_pos + 1);
			}
			if (first_pos == tuple_start.end() && last_pos != NumericLimits<uint64_t>::Maximum()) {
				string error = "Not possible to read this CSV File with multithreading. Tuple: " + to_string(last_pos) +
				               " does not have a match\n";
				error += "End Lines: \n";
				for (auto &end_line : tuple_end) {
					error += to_string(end_line) + "\n";
				}
				error += "Start Lines: \n";
				for (auto &start_line : tuple_start) {
					error += to_string(start_line) + "\n";
				}
				throw InvalidInputException(
				    "CSV File not supported for multithreading. Please run single-threaded CSV Reading");
			}
		}
	}
}

unique_ptr<CSVBufferRead> ParallelCSVGlobalState::Next(ClientContext &context, ReadCSVData &bind_data) {
	lock_guard<mutex> parallel_lock(main_mutex);
	if (!current_buffer) {
		// This means we are done with the current file, we need to go to the next one (if exists).
		if (file_index < bind_data.files.size()) {
			current_file_path = bind_data.files[file_index++];
			file_handle = ReadCSV::OpenCSV(current_file_path, bind_data.options.compression, context);
			current_csv_position = 0;
			current_buffer = make_shared<CSVBuffer>(context, buffer_size, *file_handle, current_csv_position);
			next_buffer = current_buffer->Next(*file_handle, buffer_size, current_csv_position);
		} else {
			// We are done scanning.
			return nullptr;
		}
	}
	// set up the current buffer
	auto result = make_unique<CSVBufferRead>(current_buffer, next_buffer, next_byte, next_byte + bytes_per_local_state,
	                                         batch_index++, estimated_linenr);
	// move the byte index of the CSV reader to the next buffer
	next_byte += bytes_per_local_state;
	estimated_linenr += bytes_per_local_state / (bind_data.sql_types.size() * 5); // estimate 5 bytes per column
	if (next_byte >= current_buffer->GetBufferSize()) {
		// We replace the current buffer with the next buffer
		next_byte = 0;
		bytes_read += current_buffer->GetBufferSize();
		current_buffer = next_buffer;
		if (next_buffer) {
			// Next buffer gets the next-next buffer
			next_buffer = next_buffer->Next(*file_handle, buffer_size, current_csv_position);
		}
	}
	return result;
}
void ParallelCSVGlobalState::UpdateVerification(VerificationPositions positions) {
	lock_guard<mutex> parallel_lock(main_mutex);
	if (positions.beginning_of_first_line < positions.end_of_last_line) {
		if (positions.end_of_last_line > max_tuple_end) {
			max_tuple_end = positions.end_of_last_line;
		}
		tuple_start.insert(positions.beginning_of_first_line);
		tuple_end.push_back(positions.end_of_last_line);
	}
}

static unique_ptr<GlobalTableFunctionState> ParallelCSVInitGlobal(ClientContext &context,
                                                                  TableFunctionInitInput &input) {
	auto &bind_data = (ReadCSVData &)*input.bind_data;
	if (bind_data.files.empty()) {
		// This can happen when a filename based filter pushdown has eliminated all possible files for this scan.
		return make_unique<ParallelCSVGlobalState>();
	}
	unique_ptr<CSVFileHandle> file_handle;

	bind_data.options.file_path = bind_data.files[0];
	file_handle = ReadCSV::OpenCSV(bind_data.options.file_path, bind_data.options.compression, context);
	idx_t rows_to_skip =
	    bind_data.options.skip_rows + (bind_data.options.has_header && bind_data.options.header ? 1 : 0);
	return make_unique<ParallelCSVGlobalState>(context, std::move(file_handle), bind_data.files,
	                                           context.db->NumberOfThreads(), bind_data.options.buffer_size,
	                                           rows_to_skip, ClientConfig::GetConfig(context).verify_parallelism);
}

//===--------------------------------------------------------------------===//
// Read CSV Local State
//===--------------------------------------------------------------------===//
struct ParallelCSVLocalState : public LocalTableFunctionState {
public:
	explicit ParallelCSVLocalState(unique_ptr<ParallelCSVReader> csv_reader_p) : csv_reader(std::move(csv_reader_p)) {
	}

	//! The CSV reader
	unique_ptr<ParallelCSVReader> csv_reader;
	CSVBufferRead previous_buffer;
	bool done = false;
};

unique_ptr<LocalTableFunctionState> ParallelReadCSVInitLocal(ExecutionContext &context, TableFunctionInitInput &input,
                                                             GlobalTableFunctionState *global_state_p) {
	auto &csv_data = (ReadCSVData &)*input.bind_data;
	auto &global_state = (ParallelCSVGlobalState &)*global_state_p;
	auto next_local_buffer = global_state.Next(context.client, csv_data);
	unique_ptr<ParallelCSVReader> csv_reader;
	if (next_local_buffer) {
		csv_reader = make_unique<ParallelCSVReader>(context.client, csv_data.options, std::move(next_local_buffer),
		                                            csv_data.sql_types);
	} else {
		global_state.DecrementThread();
	}
	return make_unique<ParallelCSVLocalState>(std::move(csv_reader));
}

static void ParallelReadCSVFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = (ReadCSVData &)*data_p.bind_data;
	auto &csv_global_state = (ParallelCSVGlobalState &)*data_p.global_state;
	auto &csv_local_state = (ParallelCSVLocalState &)*data_p.local_state;

	if (!csv_local_state.csv_reader) {
		// no csv_reader was set, this can happen when a filename-based filter has filtered out all possible files
		return;
	}

	do {
		if (output.size() != 0) {
			break;
		}
		if (csv_local_state.csv_reader->finished) {
			auto verification_updates = csv_local_state.csv_reader->GetVerificationPositions();
			if (!csv_local_state.csv_reader->buffer->next_buffer) {
				// if it's the last line of the file we mark as the maximum
				verification_updates.end_of_last_line = NumericLimits<uint64_t>::Maximum();
			}
			csv_global_state.UpdateVerification(verification_updates);
			auto next_chunk = csv_global_state.Next(context, bind_data);
			if (!next_chunk) {
				csv_global_state.DecrementThread();
				break;
			}
			csv_local_state.csv_reader->SetBufferRead(std::move(next_chunk));
		}
		csv_local_state.csv_reader->ParseCSV(output);

	} while (true);
	if (csv_global_state.Finished()) {
		csv_global_state.Verify();
	}
	if (bind_data.options.union_by_name) {
		throw InternalException("FIXME: union by name");
	}
	if (bind_data.options.include_file_name) {
		throw InternalException("FIXME: output file name");
	}
	if (bind_data.options.include_parsed_hive_partitions) {
		throw InternalException("FIXME: hive partitions");
	}
}

//===--------------------------------------------------------------------===//
// Single-Threaded CSV Reader
//===--------------------------------------------------------------------===//
struct SingleThreadedCSVState : public GlobalTableFunctionState {
	explicit SingleThreadedCSVState(idx_t total_files) : total_files(total_files), next_file(0), progress_in_files(0) {
	}

	mutex csv_lock;
	unique_ptr<BufferedCSVReader> initial_reader;
	//! The total number of files to read from
	idx_t total_files;
	//! The index of the next file to read (i.e. current file + 1)
	atomic<idx_t> next_file;
	//! How far along we are in reading the current set of open files
	//! This goes from [0...next_file] * 100
	atomic<idx_t> progress_in_files;
	//! The set of SQL types
	vector<LogicalType> sql_types;

	idx_t MaxThreads() const override {
		return total_files;
	}

	double GetProgress(ReadCSVData &bind_data) const {
		D_ASSERT(total_files == bind_data.files.size());
		D_ASSERT(progress_in_files <= total_files * 100);
		return (double(progress_in_files) / double(total_files));
	}

	unique_ptr<BufferedCSVReader> GetCSVReader(ClientContext &context, ReadCSVData &bind_data, idx_t &file_index,
	                                           idx_t &total_size) {
		BufferedCSVReaderOptions options;
		{
			lock_guard<mutex> l(csv_lock);
			if (initial_reader) {
				return std::move(initial_reader);
			}
			if (next_file >= total_files) {
				return nullptr;
			}
			options = bind_data.options;
			file_index = next_file;
			next_file++;
		}
		// reuse csv_readers was created during binding
		unique_ptr<BufferedCSVReader> result;
		if (options.union_by_name) {
			result = std::move(bind_data.union_readers[file_index]);
		} else {
			options.file_path = bind_data.files[file_index];
			result = make_unique<BufferedCSVReader>(context, std::move(options), sql_types);
		}
		total_size = result->file_handle->FileSize();
		return result;
	}
};

struct SingleThreadedCSVLocalState : public LocalTableFunctionState {
public:
	explicit SingleThreadedCSVLocalState() : bytes_read(0), total_size(0), current_progress(0), file_index(0) {
	}

	//! The CSV reader
	unique_ptr<BufferedCSVReader> csv_reader;
	//! The current amount of bytes read by this reader
	idx_t bytes_read;
	//! The total amount of bytes in the file
	idx_t total_size;
	//! The current progress from 0..100
	idx_t current_progress;
	//! The file index of this reader
	idx_t file_index;
};

static unique_ptr<GlobalTableFunctionState> SingleThreadedCSVInit(ClientContext &context,
                                                                  TableFunctionInitInput &input) {
	auto &bind_data = (ReadCSVData &)*input.bind_data;
	auto result = make_unique<SingleThreadedCSVState>(bind_data.files.size());
	if (bind_data.initial_reader) {
		result->initial_reader = std::move(bind_data.initial_reader);
	} else if (bind_data.files.empty()) {
		// This can happen when a filename based filter pushdown has eliminated all possible files for this scan.
		return std::move(result);
	} else {
		bind_data.options.file_path = bind_data.files[0];
		result->initial_reader = make_unique<BufferedCSVReader>(context, bind_data.options, bind_data.sql_types);
		if (bind_data.options.auto_detect) {
			bind_data.options = result->initial_reader->options;
		}
	}
	if (!bind_data.options.union_by_name) {
		// if we are reading multiple files - run auto-detect only on the first file
		// UNLESS union by name is turned on - in that case we assume that different files have different schemas
		// as such, we need to re-run the auto detection on each file
		bind_data.options.auto_detect = false;
	}
	result->next_file = 1;
	if (result->initial_reader) {
		result->sql_types = result->initial_reader->return_types;
	}
	return std::move(result);
}

unique_ptr<LocalTableFunctionState> SingleThreadedReadCSVInitLocal(ExecutionContext &context,
                                                                   TableFunctionInitInput &input,
                                                                   GlobalTableFunctionState *global_state_p) {
	auto &bind_data = (ReadCSVData &)*input.bind_data;
	auto &data = (SingleThreadedCSVState &)*global_state_p;
	auto result = make_unique<SingleThreadedCSVLocalState>();
	result->csv_reader = data.GetCSVReader(context.client, bind_data, result->file_index, result->total_size);
	return std::move(result);
}

static void SingleThreadedCSVFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = (ReadCSVData &)*data_p.bind_data;
	auto &data = (SingleThreadedCSVState &)*data_p.global_state;
	auto &lstate = (SingleThreadedCSVLocalState &)*data_p.local_state;
	if (!lstate.csv_reader) {
		// no csv_reader was set, this can happen when a filename-based filter has filtered out all possible files
		return;
	}

	do {
		lstate.csv_reader->ParseCSV(output);
		// update the number of bytes read
		D_ASSERT(lstate.bytes_read <= lstate.csv_reader->bytes_in_chunk);
		auto bytes_read = MinValue<idx_t>(lstate.total_size, lstate.csv_reader->bytes_in_chunk);
		auto current_progress = lstate.total_size == 0 ? 100 : 100 * bytes_read / lstate.total_size;
		if (current_progress > lstate.current_progress) {
			if (current_progress > 100) {
				throw InternalException("Progress should never exceed 100");
			}
			data.progress_in_files += current_progress - lstate.current_progress;
			lstate.current_progress = current_progress;
		}
		if (output.size() == 0) {
			// exhausted this file, but we might have more files we can read
			auto csv_reader = data.GetCSVReader(context, bind_data, lstate.file_index, lstate.total_size);
			// add any left-over progress for this file to the progress bar
			if (lstate.current_progress < 100) {
				data.progress_in_files += 100 - lstate.current_progress;
			}
			// reset the current progress
			lstate.current_progress = 0;
			lstate.bytes_read = 0;
			lstate.csv_reader = std::move(csv_reader);
			if (!lstate.csv_reader) {
				// no more files - we are done
				return;
			}
			lstate.bytes_read = 0;
		} else {
			break;
		}
	} while (true);

	if (bind_data.options.union_by_name) {
		UnionByName<BufferedCSVReader, BufferedCSVReaderOptions>::SetNullUnionCols(output,
		                                                                           lstate.csv_reader->union_null_cols);
	}
	if (bind_data.options.include_file_name) {
		auto &col = output.data[bind_data.filename_col_idx];
		col.SetValue(0, Value(lstate.csv_reader->options.file_path));
		col.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
	if (bind_data.options.include_parsed_hive_partitions) {
		auto partitions = HivePartitioning::Parse(lstate.csv_reader->options.file_path);

		idx_t i = bind_data.hive_partition_col_idx;

		if (partitions.size() != (bind_data.options.names.size() - bind_data.hive_partition_col_idx)) {
			throw IOException("Hive partition count mismatch, expected " +
			                  std::to_string(bind_data.options.names.size() - bind_data.hive_partition_col_idx) +
			                  " hive partitions, got " + std::to_string(partitions.size()) + "\n");
		}

		for (auto &part : partitions) {
			if (bind_data.options.names[i] != part.first) {
				throw IOException("Hive partition names mismatch, expected '" + bind_data.options.names[i] +
				                  "' but found '" + part.first + "' for file '" + lstate.csv_reader->options.file_path +
				                  "'");
			}
			auto &col = output.data[i++];
			col.SetValue(0, Value(part.second));
			col.SetVectorType(VectorType::CONSTANT_VECTOR);
		}
	}
}

//===--------------------------------------------------------------------===//
// Read CSV Functions
//===--------------------------------------------------------------------===//
static unique_ptr<GlobalTableFunctionState> ReadCSVInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind_data = (ReadCSVData &)*input.bind_data;
	if (bind_data.single_threaded) {
		return SingleThreadedCSVInit(context, input);
	} else {
		return ParallelCSVInitGlobal(context, input);
	}
}

unique_ptr<LocalTableFunctionState> ReadCSVInitLocal(ExecutionContext &context, TableFunctionInitInput &input,
                                                     GlobalTableFunctionState *global_state_p) {
	auto &csv_data = (ReadCSVData &)*input.bind_data;
	if (csv_data.single_threaded) {
		return SingleThreadedReadCSVInitLocal(context, input, global_state_p);
	} else {
		return ParallelReadCSVInitLocal(context, input, global_state_p);
	}
}

static void ReadCSVFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = (ReadCSVData &)*data_p.bind_data;
	if (bind_data.single_threaded) {
		SingleThreadedCSVFunction(context, data_p, output);
	} else {
		ParallelReadCSVFunction(context, data_p, output);
	}
}

static idx_t CSVReaderGetBatchIndex(ClientContext &context, const FunctionData *bind_data_p,
                                    LocalTableFunctionState *local_state, GlobalTableFunctionState *global_state) {
	auto &bind_data = (ReadCSVData &)*bind_data_p;
	if (bind_data.single_threaded) {
		auto &data = (SingleThreadedCSVLocalState &)*local_state;
		return data.file_index;
	}
	auto &data = (ParallelCSVLocalState &)*local_state;
	return data.csv_reader->buffer->batch_index;
}

static void ReadCSVAddNamedParameters(TableFunction &table_function) {
	table_function.named_parameters["sep"] = LogicalType::VARCHAR;
	table_function.named_parameters["delim"] = LogicalType::VARCHAR;
	table_function.named_parameters["quote"] = LogicalType::VARCHAR;
	table_function.named_parameters["new_line"] = LogicalType::VARCHAR;
	table_function.named_parameters["escape"] = LogicalType::VARCHAR;
	table_function.named_parameters["nullstr"] = LogicalType::VARCHAR;
	table_function.named_parameters["columns"] = LogicalType::ANY;
	table_function.named_parameters["header"] = LogicalType::BOOLEAN;
	table_function.named_parameters["auto_detect"] = LogicalType::BOOLEAN;
	table_function.named_parameters["sample_size"] = LogicalType::BIGINT;
	table_function.named_parameters["sample_chunk_size"] = LogicalType::BIGINT;
	table_function.named_parameters["sample_chunks"] = LogicalType::BIGINT;
	table_function.named_parameters["all_varchar"] = LogicalType::BOOLEAN;
	table_function.named_parameters["dateformat"] = LogicalType::VARCHAR;
	table_function.named_parameters["timestampformat"] = LogicalType::VARCHAR;
	table_function.named_parameters["normalize_names"] = LogicalType::BOOLEAN;
	table_function.named_parameters["compression"] = LogicalType::VARCHAR;
	table_function.named_parameters["filename"] = LogicalType::BOOLEAN;
	table_function.named_parameters["hive_partitioning"] = LogicalType::BOOLEAN;
	table_function.named_parameters["skip"] = LogicalType::BIGINT;
	table_function.named_parameters["max_line_size"] = LogicalType::VARCHAR;
	table_function.named_parameters["maximum_line_size"] = LogicalType::VARCHAR;
	table_function.named_parameters["ignore_errors"] = LogicalType::BOOLEAN;
	table_function.named_parameters["union_by_name"] = LogicalType::BOOLEAN;
	table_function.named_parameters["buffer_size"] = LogicalType::UBIGINT;
	table_function.named_parameters["decimal_separator"] = LogicalType::VARCHAR;
	table_function.named_parameters["parallel"] = LogicalType::BOOLEAN;
}

double CSVReaderProgress(ClientContext &context, const FunctionData *bind_data_p,
                         const GlobalTableFunctionState *global_state) {
	auto &bind_data = (ReadCSVData &)*bind_data_p;
	if (bind_data.single_threaded) {
		auto &data = (SingleThreadedCSVState &)*global_state;
		return data.GetProgress(bind_data);
	} else {
		auto &data = (const ParallelCSVGlobalState &)*global_state;
		return data.GetProgress(bind_data);
	}
}

void CSVComplexFilterPushdown(ClientContext &context, LogicalGet &get, FunctionData *bind_data_p,
                              vector<unique_ptr<Expression>> &filters) {
	auto data = (ReadCSVData *)bind_data_p;

	if (data->options.include_parsed_hive_partitions || data->options.include_file_name) {
		string first_file = data->files[0];

		unordered_map<string, column_t> column_map;
		for (idx_t i = 0; i < get.column_ids.size(); i++) {
			column_map.insert({get.names[get.column_ids[i]], i});
		}

		HivePartitioning::ApplyFiltersToFileList(context, data->files, filters, column_map, get.table_index,
		                                         data->options.include_parsed_hive_partitions,
		                                         data->options.include_file_name);

		if (data->files.empty() || data->files[0] != first_file) {
			data->initial_reader.reset();
		}
	}
}

unique_ptr<NodeStatistics> CSVReaderCardinality(ClientContext &context, const FunctionData *bind_data_p) {
	auto &bind_data = (ReadCSVData &)*bind_data_p;
	idx_t per_file_cardinality = 0;
	if (bind_data.initial_reader && bind_data.initial_reader->file_handle) {
		auto estimated_row_width = (bind_data.sql_types.size() * 5);
		per_file_cardinality = bind_data.initial_reader->file_handle->FileSize() / estimated_row_width;
	} else {
		// determined through the scientific method as the average amount of rows in a CSV file
		per_file_cardinality = 42;
	}
	return make_unique<NodeStatistics>(bind_data.files.size() * per_file_cardinality);
}

void BufferedCSVReaderOptions::Serialize(FieldWriter &writer) const {
	// common options
	writer.WriteField<bool>(has_delimiter);
	writer.WriteString(delimiter);
	writer.WriteField<bool>(has_quote);
	writer.WriteString(quote);
	writer.WriteField<bool>(has_escape);
	writer.WriteString(escape);
	writer.WriteField<bool>(has_header);
	writer.WriteField<bool>(header);
	writer.WriteField<bool>(ignore_errors);
	writer.WriteField<idx_t>(num_cols);
	writer.WriteField<idx_t>(buffer_sample_size);
	writer.WriteString(null_str);
	writer.WriteField<FileCompressionType>(compression);
	// read options
	writer.WriteList<string>(names);
	writer.WriteField<idx_t>(skip_rows);
	writer.WriteField<idx_t>(maximum_line_size);
	writer.WriteField<bool>(normalize_names);
	writer.WriteListNoReference<bool>(force_not_null);
	writer.WriteField<bool>(all_varchar);
	writer.WriteField<idx_t>(sample_chunk_size);
	writer.WriteField<idx_t>(sample_chunks);
	writer.WriteField<bool>(auto_detect);
	writer.WriteString(file_path);
	writer.WriteField<bool>(include_file_name);
	writer.WriteField<bool>(include_parsed_hive_partitions);
	writer.WriteString(decimal_separator);
	// write options
	writer.WriteListNoReference<bool>(force_quote);
}

void BufferedCSVReaderOptions::Deserialize(FieldReader &reader) {
	// common options
	has_delimiter = reader.ReadRequired<bool>();
	delimiter = reader.ReadRequired<string>();
	has_quote = reader.ReadRequired<bool>();
	quote = reader.ReadRequired<string>();
	has_escape = reader.ReadRequired<bool>();
	escape = reader.ReadRequired<string>();
	has_header = reader.ReadRequired<bool>();
	header = reader.ReadRequired<bool>();
	ignore_errors = reader.ReadRequired<bool>();
	num_cols = reader.ReadRequired<idx_t>();
	buffer_sample_size = reader.ReadRequired<idx_t>();
	null_str = reader.ReadRequired<string>();
	compression = reader.ReadRequired<FileCompressionType>();
	// read options
	names = reader.ReadRequiredList<string>();
	skip_rows = reader.ReadRequired<idx_t>();
	maximum_line_size = reader.ReadRequired<idx_t>();
	normalize_names = reader.ReadRequired<bool>();
	force_not_null = reader.ReadRequiredList<bool>();
	all_varchar = reader.ReadRequired<bool>();
	sample_chunk_size = reader.ReadRequired<idx_t>();
	sample_chunks = reader.ReadRequired<idx_t>();
	auto_detect = reader.ReadRequired<bool>();
	file_path = reader.ReadRequired<string>();
	include_file_name = reader.ReadRequired<bool>();
	include_parsed_hive_partitions = reader.ReadRequired<bool>();
	decimal_separator = reader.ReadRequired<string>();
	// write options
	force_quote = reader.ReadRequiredList<bool>();
}

static void CSVReaderSerialize(FieldWriter &writer, const FunctionData *bind_data_p, const TableFunction &function) {
	auto &bind_data = (ReadCSVData &)*bind_data_p;
	writer.WriteList<string>(bind_data.files);
	writer.WriteRegularSerializableList<LogicalType>(bind_data.sql_types);
	writer.WriteField<idx_t>(bind_data.filename_col_idx);
	writer.WriteField<idx_t>(bind_data.hive_partition_col_idx);
	bind_data.options.Serialize(writer);
	writer.WriteField<bool>(bind_data.single_threaded);
}

static unique_ptr<FunctionData> CSVReaderDeserialize(ClientContext &context, FieldReader &reader,
                                                     TableFunction &function) {
	auto result_data = make_unique<ReadCSVData>();
	result_data->files = reader.ReadRequiredList<string>();
	result_data->sql_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
	result_data->filename_col_idx = reader.ReadRequired<idx_t>();
	result_data->hive_partition_col_idx = reader.ReadRequired<idx_t>();
	result_data->options.Deserialize(reader);
	result_data->single_threaded = reader.ReadField<bool>(true);
	return std::move(result_data);
}

TableFunction ReadCSVTableFunction::GetFunction(bool list_parameter) {
	auto parameter = list_parameter ? LogicalType::LIST(LogicalType::VARCHAR) : LogicalType::VARCHAR;
	TableFunction read_csv("read_csv", {parameter}, ReadCSVFunction, ReadCSVBind, ReadCSVInitGlobal, ReadCSVInitLocal);
	read_csv.table_scan_progress = CSVReaderProgress;
	read_csv.pushdown_complex_filter = CSVComplexFilterPushdown;
	read_csv.serialize = CSVReaderSerialize;
	read_csv.deserialize = CSVReaderDeserialize;
	read_csv.get_batch_index = CSVReaderGetBatchIndex;
	read_csv.cardinality = CSVReaderCardinality;
	ReadCSVAddNamedParameters(read_csv);
	return read_csv;
}

TableFunction ReadCSVTableFunction::GetAutoFunction(bool list_parameter) {
	auto parameter = list_parameter ? LogicalType::LIST(LogicalType::VARCHAR) : LogicalType::VARCHAR;
	TableFunction read_csv_auto("read_csv_auto", {parameter}, ReadCSVFunction, ReadCSVAutoBind, ReadCSVInitGlobal,
	                            ReadCSVInitLocal);
	read_csv_auto.table_scan_progress = CSVReaderProgress;
	read_csv_auto.pushdown_complex_filter = CSVComplexFilterPushdown;
	read_csv_auto.serialize = CSVReaderSerialize;
	read_csv_auto.deserialize = CSVReaderDeserialize;
	read_csv_auto.get_batch_index = CSVReaderGetBatchIndex;
	read_csv_auto.cardinality = CSVReaderCardinality;
	ReadCSVAddNamedParameters(read_csv_auto);
	read_csv_auto.named_parameters["column_types"] = LogicalType::ANY;
	read_csv_auto.named_parameters["dtypes"] = LogicalType::ANY;
	read_csv_auto.named_parameters["types"] = LogicalType::ANY;
	return read_csv_auto;
}

void ReadCSVTableFunction::RegisterFunction(BuiltinFunctions &set) {
	TableFunctionSet read_csv("read_csv");
	read_csv.AddFunction(ReadCSVTableFunction::GetFunction());
	read_csv.AddFunction(ReadCSVTableFunction::GetFunction(true));
	set.AddFunction(read_csv);

	TableFunctionSet read_csv_auto("read_csv_auto");
	read_csv_auto.AddFunction(ReadCSVTableFunction::GetAutoFunction());
	read_csv_auto.AddFunction(ReadCSVTableFunction::GetAutoFunction(true));
	set.AddFunction(read_csv_auto);
}

unique_ptr<TableRef> ReadCSVReplacement(ClientContext &context, const string &table_name, ReplacementScanData *data) {
	auto lower_name = StringUtil::Lower(table_name);
	// remove any compression
	if (StringUtil::EndsWith(lower_name, ".gz")) {
		lower_name = lower_name.substr(0, lower_name.size() - 3);
	} else if (StringUtil::EndsWith(lower_name, ".zst")) {
		lower_name = lower_name.substr(0, lower_name.size() - 4);
	}
	if (!StringUtil::EndsWith(lower_name, ".csv") && !StringUtil::Contains(lower_name, ".csv?") &&
	    !StringUtil::EndsWith(lower_name, ".tsv") && !StringUtil::Contains(lower_name, ".tsv?")) {
		return nullptr;
	}
	auto table_function = make_unique<TableFunctionRef>();
	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(make_unique<ConstantExpression>(Value(table_name)));
	table_function->function = make_unique<FunctionExpression>("read_csv_auto", std::move(children));
	return std::move(table_function);
}

void BuiltinFunctions::RegisterReadFunctions() {
	CSVCopyFunction::RegisterFunction(*this);
	ReadCSVTableFunction::RegisterFunction(*this);
	auto &config = DBConfig::GetConfig(*transaction.db);
	config.replacement_scans.emplace_back(ReadCSVReplacement);
}

} // namespace duckdb



namespace duckdb {

struct RepeatFunctionData : public TableFunctionData {
	RepeatFunctionData(Value value, idx_t target_count) : value(std::move(value)), target_count(target_count) {
	}

	Value value;
	idx_t target_count;
};

struct RepeatOperatorData : public GlobalTableFunctionState {
	RepeatOperatorData() : current_count(0) {
	}
	idx_t current_count;
};

static unique_ptr<FunctionData> RepeatBind(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &return_types, vector<string> &names) {
	// the repeat function returns the type of the first argument
	auto &inputs = input.inputs;
	return_types.push_back(inputs[0].type());
	names.push_back(inputs[0].ToString());
	return make_unique<RepeatFunctionData>(inputs[0], inputs[1].GetValue<int64_t>());
}

static unique_ptr<GlobalTableFunctionState> RepeatInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_unique<RepeatOperatorData>();
}

static void RepeatFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = (const RepeatFunctionData &)*data_p.bind_data;
	auto &state = (RepeatOperatorData &)*data_p.global_state;

	idx_t remaining = MinValue<idx_t>(bind_data.target_count - state.current_count, STANDARD_VECTOR_SIZE);
	output.data[0].Reference(bind_data.value);
	output.SetCardinality(remaining);
	state.current_count += remaining;
}

static unique_ptr<NodeStatistics> RepeatCardinality(ClientContext &context, const FunctionData *bind_data_p) {
	auto &bind_data = (const RepeatFunctionData &)*bind_data_p;
	return make_unique<NodeStatistics>(bind_data.target_count, bind_data.target_count);
}

void RepeatTableFunction::RegisterFunction(BuiltinFunctions &set) {
	TableFunction repeat("repeat", {LogicalType::ANY, LogicalType::BIGINT}, RepeatFunction, RepeatBind, RepeatInit);
	repeat.cardinality = RepeatCardinality;
	set.AddFunction(repeat);
}

} // namespace duckdb





// this function makes not that much sense on its own but is a demo for table-parameter table-producing functions

namespace duckdb {

static unique_ptr<FunctionData> SummaryFunctionBind(ClientContext &context, TableFunctionBindInput &input,
                                                    vector<LogicalType> &return_types, vector<string> &names) {

	return_types.emplace_back(LogicalType::VARCHAR);
	names.emplace_back("summary");

	for (idx_t i = 0; i < input.input_table_types.size(); i++) {
		return_types.push_back(input.input_table_types[i]);
		names.emplace_back(input.input_table_names[i]);
	}

	return make_unique<TableFunctionData>();
}

static OperatorResultType SummaryFunction(ExecutionContext &context, TableFunctionInput &data_p, DataChunk &input,
                                          DataChunk &output) {
	output.SetCardinality(input.size());

	for (idx_t row_idx = 0; row_idx < input.size(); row_idx++) {
		string summary_val = "[";

		for (idx_t col_idx = 0; col_idx < input.ColumnCount(); col_idx++) {
			summary_val += input.GetValue(col_idx, row_idx).ToString();
			if (col_idx < input.ColumnCount() - 1) {
				summary_val += ", ";
			}
		}
		summary_val += "]";
		output.SetValue(0, row_idx, Value(summary_val));
	}
	for (idx_t col_idx = 0; col_idx < input.ColumnCount(); col_idx++) {
		output.data[col_idx + 1].Reference(input.data[col_idx]);
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

void SummaryTableFunction::RegisterFunction(BuiltinFunctions &set) {
	TableFunction summary_function("summary", {LogicalType::TABLE}, nullptr, SummaryFunctionBind);
	summary_function.in_out_function = SummaryFunction;
	set.AddFunction(summary_function);
}

} // namespace duckdb










#include <set>

namespace duckdb {

struct DuckDBColumnsData : public GlobalTableFunctionState {
	DuckDBColumnsData() : offset(0), column_offset(0) {
	}

	vector<CatalogEntry *> entries;
	idx_t offset;
	idx_t column_offset;
};

static unique_ptr<FunctionData> DuckDBColumnsBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("database_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("database_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("schema_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("schema_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("table_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("table_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("column_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("column_index");
	return_types.emplace_back(LogicalType::INTEGER);

	names.emplace_back("internal");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("column_default");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("is_nullable");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("data_type");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("data_type_id");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("character_maximum_length");
	return_types.emplace_back(LogicalType::INTEGER);

	names.emplace_back("numeric_precision");
	return_types.emplace_back(LogicalType::INTEGER);

	names.emplace_back("numeric_precision_radix");
	return_types.emplace_back(LogicalType::INTEGER);

	names.emplace_back("numeric_scale");
	return_types.emplace_back(LogicalType::INTEGER);

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBColumnsInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_unique<DuckDBColumnsData>();

	// scan all the schemas for tables and views and collect them
	auto schemas = Catalog::GetAllSchemas(context);
	for (auto &schema : schemas) {
		schema->Scan(context, CatalogType::TABLE_ENTRY, [&](CatalogEntry *entry) { result->entries.push_back(entry); });
	}
	return std::move(result);
}

class ColumnHelper {
public:
	static unique_ptr<ColumnHelper> Create(CatalogEntry *entry);

	virtual ~ColumnHelper() {
	}

	virtual StandardEntry *Entry() = 0;
	virtual idx_t NumColumns() = 0;
	virtual const string &ColumnName(idx_t col) = 0;
	virtual const LogicalType &ColumnType(idx_t col) = 0;
	virtual const Value ColumnDefault(idx_t col) = 0;
	virtual bool IsNullable(idx_t col) = 0;

	void WriteColumns(idx_t index, idx_t start_col, idx_t end_col, DataChunk &output);
};

class TableColumnHelper : public ColumnHelper {
public:
	explicit TableColumnHelper(TableCatalogEntry *entry) : entry(entry) {
		for (auto &constraint : entry->GetConstraints()) {
			if (constraint->type == ConstraintType::NOT_NULL) {
				auto &not_null = *reinterpret_cast<NotNullConstraint *>(constraint.get());
				not_null_cols.insert(not_null.index.index);
			}
		}
	}

	StandardEntry *Entry() override {
		return entry;
	}
	idx_t NumColumns() override {
		return entry->GetColumns().LogicalColumnCount();
	}
	const string &ColumnName(idx_t col) override {
		return entry->GetColumn(LogicalIndex(col)).Name();
	}
	const LogicalType &ColumnType(idx_t col) override {
		return entry->GetColumn(LogicalIndex(col)).Type();
	}
	const Value ColumnDefault(idx_t col) override {
		auto &column = entry->GetColumn(LogicalIndex(col));
		if (column.DefaultValue()) {
			return Value(column.DefaultValue()->ToString());
		}
		return Value();
	}
	bool IsNullable(idx_t col) override {
		return not_null_cols.find(col) == not_null_cols.end();
	}

private:
	TableCatalogEntry *entry;
	std::set<idx_t> not_null_cols;
};

class ViewColumnHelper : public ColumnHelper {
public:
	explicit ViewColumnHelper(ViewCatalogEntry *entry) : entry(entry) {
	}

	StandardEntry *Entry() override {
		return entry;
	}
	idx_t NumColumns() override {
		return entry->types.size();
	}
	const string &ColumnName(idx_t col) override {
		return entry->aliases[col];
	}
	const LogicalType &ColumnType(idx_t col) override {
		return entry->types[col];
	}
	const Value ColumnDefault(idx_t col) override {
		return Value();
	}
	bool IsNullable(idx_t col) override {
		return true;
	}

private:
	ViewCatalogEntry *entry;
};

unique_ptr<ColumnHelper> ColumnHelper::Create(CatalogEntry *entry) {
	switch (entry->type) {
	case CatalogType::TABLE_ENTRY:
		return make_unique<TableColumnHelper>((TableCatalogEntry *)entry);
	case CatalogType::VIEW_ENTRY:
		return make_unique<ViewColumnHelper>((ViewCatalogEntry *)entry);
	default:
		throw NotImplementedException("Unsupported catalog type for duckdb_columns");
	}
}

void ColumnHelper::WriteColumns(idx_t start_index, idx_t start_col, idx_t end_col, DataChunk &output) {
	for (idx_t i = start_col; i < end_col; i++) {
		auto index = start_index + (i - start_col);
		auto &entry = *Entry();

		idx_t col = 0;
		// database_name, VARCHAR
		output.SetValue(col++, index, entry.catalog->GetName());
		// database_oid, BIGINT
		output.SetValue(col++, index, Value::BIGINT(entry.catalog->GetOid()));
		// schema_name, VARCHAR
		output.SetValue(col++, index, entry.schema->name);
		// schema_oid, BIGINT
		output.SetValue(col++, index, Value::BIGINT(entry.schema->oid));
		// table_name, VARCHAR
		output.SetValue(col++, index, entry.name);
		// table_oid, BIGINT
		output.SetValue(col++, index, Value::BIGINT(entry.oid));
		// column_name, VARCHAR
		output.SetValue(col++, index, Value(ColumnName(i)));
		// column_index, INTEGER
		output.SetValue(col++, index, Value::INTEGER(i + 1));
		// internal, BOOLEAN
		output.SetValue(col++, index, Value::BOOLEAN(entry.internal));
		// column_default, VARCHAR
		output.SetValue(col++, index, Value(ColumnDefault(i)));
		// is_nullable, BOOLEAN
		output.SetValue(col++, index, Value::BOOLEAN(IsNullable(i)));
		// data_type, VARCHAR
		const LogicalType &type = ColumnType(i);
		output.SetValue(col++, index, Value(type.ToString()));
		// data_type_id, BIGINT
		output.SetValue(col++, index, Value::BIGINT(int(type.id())));
		if (type == LogicalType::VARCHAR) {
			// FIXME: need check constraints in place to set this correctly
			// character_maximum_length, INTEGER
			output.SetValue(col++, index, Value());
		} else {
			// "character_maximum_length", PhysicalType::INTEGER
			output.SetValue(col++, index, Value());
		}

		Value numeric_precision, numeric_scale, numeric_precision_radix;
		switch (type.id()) {
		case LogicalTypeId::DECIMAL:
			numeric_precision = Value::INTEGER(DecimalType::GetWidth(type));
			numeric_scale = Value::INTEGER(DecimalType::GetScale(type));
			numeric_precision_radix = Value::INTEGER(10);
			break;
		case LogicalTypeId::HUGEINT:
			numeric_precision = Value::INTEGER(128);
			numeric_scale = Value::INTEGER(0);
			numeric_precision_radix = Value::INTEGER(2);
			break;
		case LogicalTypeId::BIGINT:
			numeric_precision = Value::INTEGER(64);
			numeric_scale = Value::INTEGER(0);
			numeric_precision_radix = Value::INTEGER(2);
			break;
		case LogicalTypeId::INTEGER:
			numeric_precision = Value::INTEGER(32);
			numeric_scale = Value::INTEGER(0);
			numeric_precision_radix = Value::INTEGER(2);
			break;
		case LogicalTypeId::SMALLINT:
			numeric_precision = Value::INTEGER(16);
			numeric_scale = Value::INTEGER(0);
			numeric_precision_radix = Value::INTEGER(2);
			break;
		case LogicalTypeId::TINYINT:
			numeric_precision = Value::INTEGER(8);
			numeric_scale = Value::INTEGER(0);
			numeric_precision_radix = Value::INTEGER(2);
			break;
		case LogicalTypeId::FLOAT:
			numeric_precision = Value::INTEGER(24);
			numeric_scale = Value::INTEGER(0);
			numeric_precision_radix = Value::INTEGER(2);
			break;
		case LogicalTypeId::DOUBLE:
			numeric_precision = Value::INTEGER(53);
			numeric_scale = Value::INTEGER(0);
			numeric_precision_radix = Value::INTEGER(2);
			break;
		default:
			numeric_precision = Value();
			numeric_scale = Value();
			numeric_precision_radix = Value();
			break;
		}

		// numeric_precision, INTEGER
		output.SetValue(col++, index, numeric_precision);
		// numeric_precision_radix, INTEGER
		output.SetValue(col++, index, numeric_precision_radix);
		// numeric_scale, INTEGER
		output.SetValue(col++, index, numeric_scale);
	}
}

void DuckDBColumnsFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (DuckDBColumnsData &)*data_p.global_state;
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}

	// We need to track the offset of the relation we're writing as well as the last column
	// we wrote from that relation (if any); it's possible that we can fill up the output
	// with a partial list of columns from a relation and will need to pick up processing the
	// next chunk at the same spot.
	idx_t next = data.offset;
	idx_t column_offset = data.column_offset;
	idx_t index = 0;
	while (next < data.entries.size() && index < STANDARD_VECTOR_SIZE) {
		auto column_helper = ColumnHelper::Create(data.entries[next]);
		idx_t columns = column_helper->NumColumns();

		// Check to see if we are going to exceed the maximum index for a DataChunk
		if (index + (columns - column_offset) > STANDARD_VECTOR_SIZE) {
			idx_t column_limit = column_offset + (STANDARD_VECTOR_SIZE - index);
			output.SetCardinality(STANDARD_VECTOR_SIZE);
			column_helper->WriteColumns(index, column_offset, column_limit, output);

			// Make the current column limit the column offset when we process the next chunk
			column_offset = column_limit;
			break;
		} else {
			// Otherwise, write all of the columns from the current relation and
			// then move on to the next one.
			output.SetCardinality(index + (columns - column_offset));
			column_helper->WriteColumns(index, column_offset, columns, output);
			index += columns - column_offset;
			next++;
			column_offset = 0;
		}
	}
	data.offset = next;
	data.column_offset = column_offset;
}

void DuckDBColumnsFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("duckdb_columns", {}, DuckDBColumnsFunction, DuckDBColumnsBind, DuckDBColumnsInit));
}

} // namespace duckdb


















namespace duckdb {

struct UniqueKeyInfo {
	string schema, table;
	vector<LogicalIndex> columns;

	bool operator==(const UniqueKeyInfo &other) const {
		return (schema == other.schema) && (table == other.table) && (columns == other.columns);
	}
};

} // namespace duckdb

namespace std {

template <>
struct hash<duckdb::UniqueKeyInfo> {
	template <class X>
	static size_t ComputeHash(const X &x) {
		return hash<X>()(x);
	}

	size_t operator()(const duckdb::UniqueKeyInfo &j) const {
		D_ASSERT(j.columns.size() > 0);
		return ComputeHash(j.schema) + ComputeHash(j.table) + ComputeHash(j.columns[0].index);
	}
};

} // namespace std

namespace duckdb {

struct DuckDBConstraintsData : public GlobalTableFunctionState {
	DuckDBConstraintsData() : offset(0), constraint_offset(0), unique_constraint_offset(0) {
	}

	vector<CatalogEntry *> entries;
	idx_t offset;
	idx_t constraint_offset;
	idx_t unique_constraint_offset;
	unordered_map<UniqueKeyInfo, idx_t> known_fk_unique_constraint_offsets;
};

static unique_ptr<FunctionData> DuckDBConstraintsBind(ClientContext &context, TableFunctionBindInput &input,
                                                      vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("database_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("database_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("schema_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("schema_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("table_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("table_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("constraint_index");
	return_types.emplace_back(LogicalType::BIGINT);

	// CHECK, PRIMARY KEY or UNIQUE
	names.emplace_back("constraint_type");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("constraint_text");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("expression");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("constraint_column_indexes");
	return_types.push_back(LogicalType::LIST(LogicalType::BIGINT));

	names.emplace_back("constraint_column_names");
	return_types.push_back(LogicalType::LIST(LogicalType::VARCHAR));

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBConstraintsInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_unique<DuckDBConstraintsData>();

	// scan all the schemas for tables and collect them
	auto schemas = Catalog::GetAllSchemas(context);

	for (auto &schema : schemas) {
		vector<CatalogEntry *> entries;

		schema->Scan(context, CatalogType::TABLE_ENTRY, [&](CatalogEntry *entry) {
			if (entry->type == CatalogType::TABLE_ENTRY) {
				entries.push_back(entry);
			}
		});

		sort(entries.begin(), entries.end(), [&](CatalogEntry *x, CatalogEntry *y) { return (x->name < y->name); });

		result->entries.insert(result->entries.end(), entries.begin(), entries.end());
	};

	return std::move(result);
}

void DuckDBConstraintsFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (DuckDBConstraintsData &)*data_p.global_state;
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = data.entries[data.offset];
		D_ASSERT(entry->type == CatalogType::TABLE_ENTRY);

		auto &table = (TableCatalogEntry &)*entry;
		auto &constraints = table.GetConstraints();
		bool is_duck_table = table.IsDuckTable();
		for (; data.constraint_offset < constraints.size() && count < STANDARD_VECTOR_SIZE; data.constraint_offset++) {
			auto &constraint = constraints[data.constraint_offset];
			// return values:
			// constraint_type, VARCHAR
			// Processing this first due to shortcut (early continue)
			string constraint_type;
			switch (constraint->type) {
			case ConstraintType::CHECK:
				constraint_type = "CHECK";
				break;
			case ConstraintType::UNIQUE: {
				auto &unique = (UniqueConstraint &)*constraint;
				constraint_type = unique.is_primary_key ? "PRIMARY KEY" : "UNIQUE";
				break;
			}
			case ConstraintType::NOT_NULL:
				constraint_type = "NOT NULL";
				break;
			case ConstraintType::FOREIGN_KEY: {
				if (!is_duck_table) {
					continue;
				}
				auto &bound_constraints = table.GetBoundConstraints();
				auto &bound_foreign_key = (const BoundForeignKeyConstraint &)*bound_constraints[data.constraint_offset];
				if (bound_foreign_key.info.type == ForeignKeyType::FK_TYPE_PRIMARY_KEY_TABLE) {
					// Those are already covered by PRIMARY KEY and UNIQUE entries
					continue;
				}
				constraint_type = "FOREIGN KEY";
				break;
			}
			default:
				throw NotImplementedException("Unimplemented constraint for duckdb_constraints");
			}

			idx_t col = 0;
			// database_name, LogicalType::VARCHAR
			output.SetValue(col++, count, Value(table.schema->catalog->GetName()));
			// database_oid, LogicalType::BIGINT
			output.SetValue(col++, count, Value::BIGINT(table.schema->catalog->GetOid()));
			// schema_name, LogicalType::VARCHAR
			output.SetValue(col++, count, Value(table.schema->name));
			// schema_oid, LogicalType::BIGINT
			output.SetValue(col++, count, Value::BIGINT(table.schema->oid));
			// table_name, LogicalType::VARCHAR
			output.SetValue(col++, count, Value(table.name));
			// table_oid, LogicalType::BIGINT
			output.SetValue(col++, count, Value::BIGINT(table.oid));

			// constraint_index, BIGINT
			UniqueKeyInfo uk_info;

			if (is_duck_table) {
				auto &bound_constraint = (BoundConstraint &)*table.GetBoundConstraints()[data.constraint_offset];
				switch (bound_constraint.type) {
				case ConstraintType::UNIQUE: {
					auto &bound_unique = (BoundUniqueConstraint &)bound_constraint;
					uk_info = {table.schema->name, table.name, bound_unique.keys};
					break;
				}
				case ConstraintType::FOREIGN_KEY: {
					const auto &bound_foreign_key = (const BoundForeignKeyConstraint &)bound_constraint;
					const auto &info = bound_foreign_key.info;
					// find the other table
					auto table_entry =
					    Catalog::GetEntry<TableCatalogEntry>(context, INVALID_CATALOG, info.schema, info.table, true);
					if (!table_entry) {
						throw InternalException("dukdb_constraints: entry %s.%s referenced in foreign key not found",
						                        info.schema, info.table);
					}
					vector<LogicalIndex> index;
					for (auto &key : info.pk_keys) {
						index.push_back(table_entry->GetColumns().PhysicalToLogical(key));
					}
					uk_info = {table_entry->schema->name, table_entry->name, index};
					break;
				}
				default:
					break;
				}
			}

			if (uk_info.columns.empty()) {
				output.SetValue(col++, count, Value::BIGINT(data.unique_constraint_offset++));
			} else {
				auto known_unique_constraint_offset = data.known_fk_unique_constraint_offsets.find(uk_info);
				if (known_unique_constraint_offset == data.known_fk_unique_constraint_offsets.end()) {
					data.known_fk_unique_constraint_offsets.insert(make_pair(uk_info, data.unique_constraint_offset));
					output.SetValue(col++, count, Value::BIGINT(data.unique_constraint_offset));
					data.unique_constraint_offset++;
				} else {
					output.SetValue(col++, count, Value::BIGINT(known_unique_constraint_offset->second));
				}
			}
			output.SetValue(col++, count, Value(constraint_type));

			// constraint_text, VARCHAR
			output.SetValue(col++, count, Value(constraint->ToString()));

			// expression, VARCHAR
			Value expression_text;
			if (constraint->type == ConstraintType::CHECK) {
				auto &check = (CheckConstraint &)*constraint;
				expression_text = Value(check.expression->ToString());
			}
			output.SetValue(col++, count, expression_text);

			vector<LogicalIndex> column_index_list;
			if (is_duck_table) {
				auto &bound_constraint = (BoundConstraint &)*table.GetBoundConstraints()[data.constraint_offset];
				switch (bound_constraint.type) {
				case ConstraintType::CHECK: {
					auto &bound_check = (BoundCheckConstraint &)bound_constraint;
					for (auto &col_idx : bound_check.bound_columns) {
						column_index_list.push_back(table.GetColumns().PhysicalToLogical(col_idx));
					}
					break;
				}
				case ConstraintType::UNIQUE: {
					auto &bound_unique = (BoundUniqueConstraint &)bound_constraint;
					for (auto &col_idx : bound_unique.keys) {
						column_index_list.push_back(col_idx);
					}
					break;
				}
				case ConstraintType::NOT_NULL: {
					auto &bound_not_null = (BoundNotNullConstraint &)bound_constraint;
					column_index_list.push_back(table.GetColumns().PhysicalToLogical(bound_not_null.index));
					break;
				}
				case ConstraintType::FOREIGN_KEY: {
					auto &bound_foreign_key = (const BoundForeignKeyConstraint &)bound_constraint;
					for (auto &col_idx : bound_foreign_key.info.fk_keys) {
						column_index_list.push_back(table.GetColumns().PhysicalToLogical(col_idx));
					}
					break;
				}
				default:
					throw NotImplementedException("Unimplemented constraint for duckdb_constraints");
				}
			}

			vector<Value> index_list;
			vector<Value> column_name_list;
			for (auto column_index : column_index_list) {
				index_list.push_back(Value::BIGINT(column_index.index));
				column_name_list.emplace_back(table.GetColumn(column_index).Name());
			}

			// constraint_column_indexes, LIST
			output.SetValue(col++, count, Value::LIST(LogicalType::BIGINT, std::move(index_list)));

			// constraint_column_names, LIST
			output.SetValue(col++, count, Value::LIST(LogicalType::VARCHAR, std::move(column_name_list)));

			count++;
		}
		if (data.constraint_offset >= constraints.size()) {
			data.constraint_offset = 0;
			data.offset++;
		}
	}
	output.SetCardinality(count);
}

void DuckDBConstraintsFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("duckdb_constraints", {}, DuckDBConstraintsFunction, DuckDBConstraintsBind,
	                              DuckDBConstraintsInit));
}

} // namespace duckdb




namespace duckdb {

struct DuckDBDatabasesData : public GlobalTableFunctionState {
	DuckDBDatabasesData() : offset(0) {
	}

	vector<AttachedDatabase *> entries;
	idx_t offset;
};

static unique_ptr<FunctionData> DuckDBDatabasesBind(ClientContext &context, TableFunctionBindInput &input,
                                                    vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("database_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("database_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("path");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("internal");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("type");
	return_types.emplace_back(LogicalType::VARCHAR);

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBDatabasesInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_unique<DuckDBDatabasesData>();

	// scan all the schemas for tables and collect them and collect them
	auto &db_manager = DatabaseManager::Get(context);
	result->entries = db_manager.GetDatabases(context);
	return std::move(result);
}

void DuckDBDatabasesFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (DuckDBDatabasesData &)*data_p.global_state;
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = data.entries[data.offset++];

		auto &attached = (AttachedDatabase &)*entry;
		// return values:

		idx_t col = 0;
		// database_name, VARCHAR
		output.SetValue(col++, count, attached.GetName());
		// database_oid, BIGINT
		output.SetValue(col++, count, Value::BIGINT(attached.oid));
		// path, VARCHAR
		bool is_internal = attached.IsSystem() || attached.IsTemporary();
		Value db_path;
		if (!is_internal) {
			bool in_memory = attached.GetCatalog().InMemory();
			if (!in_memory) {
				db_path = Value(attached.GetCatalog().GetDBPath());
			}
		}
		output.SetValue(col++, count, db_path);
		// internal, BOOLEAN
		output.SetValue(col++, count, Value::BOOLEAN(is_internal));
		// type, VARCHAR
		output.SetValue(col++, count, Value(attached.GetCatalog().GetCatalogType()));

		count++;
	}
	output.SetCardinality(count);
}

void DuckDBDatabasesFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(
	    TableFunction("duckdb_databases", {}, DuckDBDatabasesFunction, DuckDBDatabasesBind, DuckDBDatabasesInit));
}

} // namespace duckdb








namespace duckdb {

struct DependencyInformation {
	CatalogEntry *object;
	CatalogEntry *dependent;
	DependencyType type;
};

struct DuckDBDependenciesData : public GlobalTableFunctionState {
	DuckDBDependenciesData() : offset(0) {
	}

	vector<DependencyInformation> entries;
	idx_t offset;
};

static unique_ptr<FunctionData> DuckDBDependenciesBind(ClientContext &context, TableFunctionBindInput &input,
                                                       vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("classid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("objid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("objsubid");
	return_types.emplace_back(LogicalType::INTEGER);

	names.emplace_back("refclassid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("refobjid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("refobjsubid");
	return_types.emplace_back(LogicalType::INTEGER);

	names.emplace_back("deptype");
	return_types.emplace_back(LogicalType::VARCHAR);

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBDependenciesInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_unique<DuckDBDependenciesData>();

	// scan all the schemas and collect them
	auto &catalog = Catalog::GetCatalog(context, INVALID_CATALOG);
	if (catalog.IsDuckCatalog()) {
		auto &duck_catalog = (DuckCatalog &)catalog;
		auto &dependency_manager = duck_catalog.GetDependencyManager();
		dependency_manager.Scan([&](CatalogEntry *obj, CatalogEntry *dependent, DependencyType type) {
			DependencyInformation info;
			info.object = obj;
			info.dependent = dependent;
			info.type = type;
			result->entries.push_back(info);
		});
	}

	return std::move(result);
}

void DuckDBDependenciesFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (DuckDBDependenciesData &)*data_p.global_state;
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = data.entries[data.offset];

		// return values:
		// classid, LogicalType::BIGINT
		output.SetValue(0, count, Value::BIGINT(0));
		// objid, LogicalType::BIGINT
		output.SetValue(1, count, Value::BIGINT(entry.object->oid));
		// objsubid, LogicalType::INTEGER
		output.SetValue(2, count, Value::INTEGER(0));
		// refclassid, LogicalType::BIGINT
		output.SetValue(3, count, Value::BIGINT(0));
		// refobjid, LogicalType::BIGINT
		output.SetValue(4, count, Value::BIGINT(entry.dependent->oid));
		// refobjsubid, LogicalType::INTEGER
		output.SetValue(5, count, Value::INTEGER(0));
		// deptype, LogicalType::VARCHAR
		string dependency_type_str;
		switch (entry.type) {
		case DependencyType::DEPENDENCY_REGULAR:
			dependency_type_str = "n";
			break;
		case DependencyType::DEPENDENCY_AUTOMATIC:
			dependency_type_str = "a";
			break;
		default:
			throw NotImplementedException("Unimplemented dependency type");
		}
		output.SetValue(6, count, Value(dependency_type_str));

		data.offset++;
		count++;
	}
	output.SetCardinality(count);
}

void DuckDBDependenciesFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("duckdb_dependencies", {}, DuckDBDependenciesFunction, DuckDBDependenciesBind,
	                              DuckDBDependenciesInit));
}

} // namespace duckdb










namespace duckdb {

struct ExtensionInformation {
	string name;
	bool loaded = false;
	bool installed = false;
	string file_path;
	string description;
	vector<Value> aliases;
};

struct DuckDBExtensionsData : public GlobalTableFunctionState {
	DuckDBExtensionsData() : offset(0) {
	}

	vector<ExtensionInformation> entries;
	idx_t offset;
};

static unique_ptr<FunctionData> DuckDBExtensionsBind(ClientContext &context, TableFunctionBindInput &input,
                                                     vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("extension_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("loaded");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("installed");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("install_path");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("description");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("aliases");
	return_types.emplace_back(LogicalType::LIST(LogicalType::VARCHAR));

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBExtensionsInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_unique<DuckDBExtensionsData>();

	auto &fs = FileSystem::GetFileSystem(context);
	auto &db = DatabaseInstance::GetDatabase(context);

	map<string, ExtensionInformation> installed_extensions;
	auto extension_count = ExtensionHelper::DefaultExtensionCount();
	auto alias_count = ExtensionHelper::ExtensionAliasCount();
	for (idx_t i = 0; i < extension_count; i++) {
		auto extension = ExtensionHelper::GetDefaultExtension(i);
		ExtensionInformation info;
		info.name = extension.name;
		info.installed = extension.statically_loaded;
		info.loaded = false;
		info.file_path = extension.statically_loaded ? "(BUILT-IN)" : string();
		info.description = extension.description;
		for (idx_t k = 0; k < alias_count; k++) {
			auto alias = ExtensionHelper::GetExtensionAlias(k);
			if (info.name == alias.extension) {
				info.aliases.emplace_back(alias.alias);
			}
		}
		installed_extensions[info.name] = std::move(info);
	}

	// scan the install directory for installed extensions
	auto ext_directory = ExtensionHelper::ExtensionDirectory(context);
	fs.ListFiles(ext_directory, [&](const string &path, bool is_directory) {
		if (!StringUtil::EndsWith(path, ".duckdb_extension")) {
			return;
		}
		ExtensionInformation info;
		info.name = fs.ExtractBaseName(path);
		info.loaded = false;
		info.file_path = fs.JoinPath(ext_directory, path);
		auto entry = installed_extensions.find(info.name);
		if (entry == installed_extensions.end()) {
			installed_extensions[info.name] = std::move(info);
		} else {
			if (!entry->second.loaded) {
				entry->second.file_path = info.file_path;
			}
			entry->second.installed = true;
		}
	});

	// now check the list of currently loaded extensions
	auto &loaded_extensions = db.LoadedExtensions();
	for (auto &ext_name : loaded_extensions) {
		auto entry = installed_extensions.find(ext_name);
		if (entry == installed_extensions.end()) {
			ExtensionInformation info;
			info.name = ext_name;
			info.loaded = true;
			installed_extensions[ext_name] = std::move(info);
		} else {
			entry->second.loaded = true;
		}
	}

	result->entries.reserve(installed_extensions.size());
	for (auto &kv : installed_extensions) {
		result->entries.push_back(std::move(kv.second));
	}
	return std::move(result);
}

void DuckDBExtensionsFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (DuckDBExtensionsData &)*data_p.global_state;
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = data.entries[data.offset];

		// return values:
		// extension_name LogicalType::VARCHAR
		output.SetValue(0, count, Value(entry.name));
		// loaded LogicalType::BOOLEAN
		output.SetValue(1, count, Value::BOOLEAN(entry.loaded));
		// installed LogicalType::BOOLEAN
		output.SetValue(2, count, !entry.installed && entry.loaded ? Value() : Value::BOOLEAN(entry.installed));
		// install_path LogicalType::VARCHAR
		output.SetValue(3, count, Value(entry.file_path));
		// description LogicalType::VARCHAR
		output.SetValue(4, count, Value(entry.description));
		// aliases     LogicalType::LIST(LogicalType::VARCHAR)
		output.SetValue(5, count, Value::LIST(LogicalType::VARCHAR, entry.aliases));

		data.offset++;
		count++;
	}
	output.SetCardinality(count);
}

void DuckDBExtensionsFun::RegisterFunction(BuiltinFunctions &set) {
	TableFunctionSet functions("duckdb_extensions");
	functions.AddFunction(TableFunction({}, DuckDBExtensionsFunction, DuckDBExtensionsBind, DuckDBExtensionsInit));
	set.AddFunction(functions);
}

} // namespace duckdb

















namespace duckdb {

struct DuckDBFunctionsData : public GlobalTableFunctionState {
	DuckDBFunctionsData() : offset(0), offset_in_entry(0) {
	}

	vector<CatalogEntry *> entries;
	idx_t offset;
	idx_t offset_in_entry;
};

static unique_ptr<FunctionData> DuckDBFunctionsBind(ClientContext &context, TableFunctionBindInput &input,
                                                    vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("database_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("schema_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("function_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("function_type");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("description");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("return_type");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("parameters");
	return_types.push_back(LogicalType::LIST(LogicalType::VARCHAR));

	names.emplace_back("parameter_types");
	return_types.push_back(LogicalType::LIST(LogicalType::VARCHAR));

	names.emplace_back("varargs");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("macro_definition");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("has_side_effects");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("internal");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("function_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	return nullptr;
}

static void ExtractFunctionsFromSchema(ClientContext &context, SchemaCatalogEntry &schema,
                                       DuckDBFunctionsData &result) {
	schema.Scan(context, CatalogType::SCALAR_FUNCTION_ENTRY,
	            [&](CatalogEntry *entry) { result.entries.push_back(entry); });
	schema.Scan(context, CatalogType::TABLE_FUNCTION_ENTRY,
	            [&](CatalogEntry *entry) { result.entries.push_back(entry); });
	schema.Scan(context, CatalogType::PRAGMA_FUNCTION_ENTRY,
	            [&](CatalogEntry *entry) { result.entries.push_back(entry); });
}

unique_ptr<GlobalTableFunctionState> DuckDBFunctionsInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_unique<DuckDBFunctionsData>();

	// scan all the schemas for tables and collect themand collect them
	auto schemas = Catalog::GetAllSchemas(context);
	for (auto &schema : schemas) {
		ExtractFunctionsFromSchema(context, *schema, *result);
	};

	std::sort(result->entries.begin(), result->entries.end(),
	          [&](CatalogEntry *a, CatalogEntry *b) { return (int)a->type < (int)b->type; });
	return std::move(result);
}

struct ScalarFunctionExtractor {
	static idx_t FunctionCount(ScalarFunctionCatalogEntry &entry) {
		return entry.functions.Size();
	}

	static Value GetFunctionType() {
		return Value("scalar");
	}

	static Value GetFunctionDescription(ScalarFunctionCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static Value GetReturnType(ScalarFunctionCatalogEntry &entry, idx_t offset) {
		return Value(entry.functions.GetFunctionByOffset(offset).return_type.ToString());
	}

	static Value GetParameters(ScalarFunctionCatalogEntry &entry, idx_t offset) {
		vector<Value> results;
		for (idx_t i = 0; i < entry.functions.GetFunctionByOffset(offset).arguments.size(); i++) {
			results.emplace_back("col" + to_string(i));
		}
		return Value::LIST(LogicalType::VARCHAR, std::move(results));
	}

	static Value GetParameterTypes(ScalarFunctionCatalogEntry &entry, idx_t offset) {
		vector<Value> results;
		auto fun = entry.functions.GetFunctionByOffset(offset);
		for (idx_t i = 0; i < fun.arguments.size(); i++) {
			results.emplace_back(fun.arguments[i].ToString());
		}
		return Value::LIST(LogicalType::VARCHAR, std::move(results));
	}

	static Value GetVarArgs(ScalarFunctionCatalogEntry &entry, idx_t offset) {
		auto fun = entry.functions.GetFunctionByOffset(offset);
		return !fun.HasVarArgs() ? Value() : Value(fun.varargs.ToString());
	}

	static Value GetMacroDefinition(ScalarFunctionCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static Value HasSideEffects(ScalarFunctionCatalogEntry &entry, idx_t offset) {
		return Value::BOOLEAN(entry.functions.GetFunctionByOffset(offset).side_effects ==
		                      FunctionSideEffects::HAS_SIDE_EFFECTS);
	}
};

struct AggregateFunctionExtractor {
	static idx_t FunctionCount(AggregateFunctionCatalogEntry &entry) {
		return entry.functions.Size();
	}

	static Value GetFunctionType() {
		return Value("aggregate");
	}

	static Value GetFunctionDescription(AggregateFunctionCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static Value GetReturnType(AggregateFunctionCatalogEntry &entry, idx_t offset) {
		return Value(entry.functions.GetFunctionByOffset(offset).return_type.ToString());
	}

	static Value GetParameters(AggregateFunctionCatalogEntry &entry, idx_t offset) {
		vector<Value> results;
		for (idx_t i = 0; i < entry.functions.GetFunctionByOffset(offset).arguments.size(); i++) {
			results.emplace_back("col" + to_string(i));
		}
		return Value::LIST(LogicalType::VARCHAR, std::move(results));
	}

	static Value GetParameterTypes(AggregateFunctionCatalogEntry &entry, idx_t offset) {
		vector<Value> results;
		auto fun = entry.functions.GetFunctionByOffset(offset);
		for (idx_t i = 0; i < fun.arguments.size(); i++) {
			results.emplace_back(fun.arguments[i].ToString());
		}
		return Value::LIST(LogicalType::VARCHAR, std::move(results));
	}

	static Value GetVarArgs(AggregateFunctionCatalogEntry &entry, idx_t offset) {
		auto fun = entry.functions.GetFunctionByOffset(offset);
		return !fun.HasVarArgs() ? Value() : Value(fun.varargs.ToString());
	}

	static Value GetMacroDefinition(AggregateFunctionCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static Value HasSideEffects(AggregateFunctionCatalogEntry &entry, idx_t offset) {
		return Value::BOOLEAN(entry.functions.GetFunctionByOffset(offset).side_effects ==
		                      FunctionSideEffects::HAS_SIDE_EFFECTS);
	}
};

struct MacroExtractor {
	static idx_t FunctionCount(ScalarMacroCatalogEntry &entry) {
		return 1;
	}

	static Value GetFunctionType() {
		return Value("macro");
	}

	static Value GetFunctionDescription(ScalarMacroCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static Value GetReturnType(ScalarMacroCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static Value GetParameters(ScalarMacroCatalogEntry &entry, idx_t offset) {
		vector<Value> results;
		for (auto &param : entry.function->parameters) {
			D_ASSERT(param->type == ExpressionType::COLUMN_REF);
			auto &colref = (ColumnRefExpression &)*param;
			results.emplace_back(colref.GetColumnName());
		}
		for (auto &param_entry : entry.function->default_parameters) {
			results.emplace_back(param_entry.first);
		}
		return Value::LIST(LogicalType::VARCHAR, std::move(results));
	}

	static Value GetParameterTypes(ScalarMacroCatalogEntry &entry, idx_t offset) {
		vector<Value> results;
		for (idx_t i = 0; i < entry.function->parameters.size(); i++) {
			results.emplace_back(LogicalType::VARCHAR);
		}
		for (idx_t i = 0; i < entry.function->default_parameters.size(); i++) {
			results.emplace_back(LogicalType::VARCHAR);
		}
		return Value::LIST(LogicalType::VARCHAR, std::move(results));
	}

	static Value GetVarArgs(ScalarMacroCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static Value GetMacroDefinition(ScalarMacroCatalogEntry &entry, idx_t offset) {
		D_ASSERT(entry.function->type == MacroType::SCALAR_MACRO);
		auto &func = (ScalarMacroFunction &)*entry.function;
		return func.expression->ToString();
	}

	static Value HasSideEffects(ScalarMacroCatalogEntry &entry, idx_t offset) {
		return Value();
	}
};

struct TableMacroExtractor {
	static idx_t FunctionCount(TableMacroCatalogEntry &entry) {
		return 1;
	}

	static Value GetFunctionType() {
		return Value("table_macro");
	}

	static Value GetFunctionDescription(TableMacroCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static Value GetReturnType(TableMacroCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static Value GetParameters(TableMacroCatalogEntry &entry, idx_t offset) {
		vector<Value> results;
		for (auto &param : entry.function->parameters) {
			D_ASSERT(param->type == ExpressionType::COLUMN_REF);
			auto &colref = (ColumnRefExpression &)*param;
			results.emplace_back(colref.GetColumnName());
		}
		for (auto &param_entry : entry.function->default_parameters) {
			results.emplace_back(param_entry.first);
		}
		return Value::LIST(LogicalType::VARCHAR, std::move(results));
	}

	static Value GetParameterTypes(TableMacroCatalogEntry &entry, idx_t offset) {
		vector<Value> results;
		for (idx_t i = 0; i < entry.function->parameters.size(); i++) {
			results.emplace_back(LogicalType::VARCHAR);
		}
		for (idx_t i = 0; i < entry.function->default_parameters.size(); i++) {
			results.emplace_back(LogicalType::VARCHAR);
		}
		return Value::LIST(LogicalType::VARCHAR, std::move(results));
	}

	static Value GetVarArgs(TableMacroCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static Value GetMacroDefinition(TableMacroCatalogEntry &entry, idx_t offset) {
		if (entry.function->type == MacroType::SCALAR_MACRO) {
			auto &func = (ScalarMacroFunction &)*entry.function;
			return func.expression->ToString();
		}
		return Value();
	}

	static Value HasSideEffects(TableMacroCatalogEntry &entry, idx_t offset) {
		return Value();
	}
};

struct TableFunctionExtractor {
	static idx_t FunctionCount(TableFunctionCatalogEntry &entry) {
		return entry.functions.Size();
	}

	static Value GetFunctionType() {
		return Value("table");
	}

	static Value GetFunctionDescription(TableFunctionCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static Value GetReturnType(TableFunctionCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static Value GetParameters(TableFunctionCatalogEntry &entry, idx_t offset) {
		vector<Value> results;
		auto fun = entry.functions.GetFunctionByOffset(offset);
		for (idx_t i = 0; i < fun.arguments.size(); i++) {
			results.emplace_back("col" + to_string(i));
		}
		for (auto &param : fun.named_parameters) {
			results.emplace_back(param.first);
		}
		return Value::LIST(LogicalType::VARCHAR, std::move(results));
	}

	static Value GetParameterTypes(TableFunctionCatalogEntry &entry, idx_t offset) {
		vector<Value> results;
		auto fun = entry.functions.GetFunctionByOffset(offset);

		for (idx_t i = 0; i < fun.arguments.size(); i++) {
			results.emplace_back(fun.arguments[i].ToString());
		}
		for (auto &param : fun.named_parameters) {
			results.emplace_back(param.second.ToString());
		}
		return Value::LIST(LogicalType::VARCHAR, std::move(results));
	}

	static Value GetVarArgs(TableFunctionCatalogEntry &entry, idx_t offset) {
		auto fun = entry.functions.GetFunctionByOffset(offset);
		return !fun.HasVarArgs() ? Value() : Value(fun.varargs.ToString());
	}

	static Value GetMacroDefinition(TableFunctionCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static Value HasSideEffects(TableFunctionCatalogEntry &entry, idx_t offset) {
		return Value();
	}
};

struct PragmaFunctionExtractor {
	static idx_t FunctionCount(PragmaFunctionCatalogEntry &entry) {
		return entry.functions.Size();
	}

	static Value GetFunctionType() {
		return Value("pragma");
	}

	static Value GetFunctionDescription(PragmaFunctionCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static Value GetReturnType(PragmaFunctionCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static Value GetParameters(PragmaFunctionCatalogEntry &entry, idx_t offset) {
		vector<Value> results;
		auto fun = entry.functions.GetFunctionByOffset(offset);

		for (idx_t i = 0; i < fun.arguments.size(); i++) {
			results.emplace_back("col" + to_string(i));
		}
		for (auto &param : fun.named_parameters) {
			results.emplace_back(param.first);
		}
		return Value::LIST(LogicalType::VARCHAR, std::move(results));
	}

	static Value GetParameterTypes(PragmaFunctionCatalogEntry &entry, idx_t offset) {
		vector<Value> results;
		auto fun = entry.functions.GetFunctionByOffset(offset);

		for (idx_t i = 0; i < fun.arguments.size(); i++) {
			results.emplace_back(fun.arguments[i].ToString());
		}
		for (auto &param : fun.named_parameters) {
			results.emplace_back(param.second.ToString());
		}
		return Value::LIST(LogicalType::VARCHAR, std::move(results));
	}

	static Value GetVarArgs(PragmaFunctionCatalogEntry &entry, idx_t offset) {
		auto fun = entry.functions.GetFunctionByOffset(offset);
		return !fun.HasVarArgs() ? Value() : Value(fun.varargs.ToString());
	}

	static Value GetMacroDefinition(PragmaFunctionCatalogEntry &entry, idx_t offset) {
		return Value();
	}

	static Value HasSideEffects(PragmaFunctionCatalogEntry &entry, idx_t offset) {
		return Value();
	}
};

template <class T, class OP>
bool ExtractFunctionData(StandardEntry *entry, idx_t function_idx, DataChunk &output, idx_t output_offset) {
	auto &function = (T &)*entry;
	idx_t col = 0;

	// database_name, LogicalType::VARCHAR
	output.SetValue(col++, output_offset, Value(entry->schema->catalog->GetName()));

	// schema_name, LogicalType::VARCHAR
	output.SetValue(col++, output_offset, Value(entry->schema->name));

	// function_name, LogicalType::VARCHAR
	output.SetValue(col++, output_offset, Value(entry->name));

	// function_type, LogicalType::VARCHAR
	output.SetValue(col++, output_offset, Value(OP::GetFunctionType()));

	// function_description, LogicalType::VARCHAR
	output.SetValue(col++, output_offset, OP::GetFunctionDescription(function, function_idx));

	// return_type, LogicalType::VARCHAR
	output.SetValue(col++, output_offset, OP::GetReturnType(function, function_idx));

	// parameters, LogicalType::LIST(LogicalType::VARCHAR)
	output.SetValue(col++, output_offset, OP::GetParameters(function, function_idx));

	// parameter_types, LogicalType::LIST(LogicalType::VARCHAR)
	output.SetValue(col++, output_offset, OP::GetParameterTypes(function, function_idx));

	// varargs, LogicalType::VARCHAR
	output.SetValue(col++, output_offset, OP::GetVarArgs(function, function_idx));

	// macro_definition, LogicalType::VARCHAR
	output.SetValue(col++, output_offset, OP::GetMacroDefinition(function, function_idx));

	// has_side_effects, LogicalType::BOOLEAN
	output.SetValue(col++, output_offset, OP::HasSideEffects(function, function_idx));

	// internal, LogicalType::BOOLEAN
	output.SetValue(col++, output_offset, Value::BOOLEAN(entry->internal));

	// function_oid, LogicalType::BIGINT
	output.SetValue(col++, output_offset, Value::BIGINT(entry->oid));

	return function_idx + 1 == OP::FunctionCount(function);
}

void DuckDBFunctionsFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (DuckDBFunctionsData &)*data_p.global_state;
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = data.entries[data.offset];
		auto standard_entry = (StandardEntry *)entry;
		bool finished;

		switch (entry->type) {
		case CatalogType::SCALAR_FUNCTION_ENTRY:
			finished = ExtractFunctionData<ScalarFunctionCatalogEntry, ScalarFunctionExtractor>(
			    standard_entry, data.offset_in_entry, output, count);
			break;
		case CatalogType::AGGREGATE_FUNCTION_ENTRY:
			finished = ExtractFunctionData<AggregateFunctionCatalogEntry, AggregateFunctionExtractor>(
			    standard_entry, data.offset_in_entry, output, count);
			break;
		case CatalogType::TABLE_MACRO_ENTRY:
			finished = ExtractFunctionData<TableMacroCatalogEntry, TableMacroExtractor>(
			    standard_entry, data.offset_in_entry, output, count);
			break;

		case CatalogType::MACRO_ENTRY:
			finished = ExtractFunctionData<ScalarMacroCatalogEntry, MacroExtractor>(
			    standard_entry, data.offset_in_entry, output, count);
			break;
		case CatalogType::TABLE_FUNCTION_ENTRY:
			finished = ExtractFunctionData<TableFunctionCatalogEntry, TableFunctionExtractor>(
			    standard_entry, data.offset_in_entry, output, count);
			break;
		case CatalogType::PRAGMA_FUNCTION_ENTRY:
			finished = ExtractFunctionData<PragmaFunctionCatalogEntry, PragmaFunctionExtractor>(
			    standard_entry, data.offset_in_entry, output, count);
			break;
		default:
			throw InternalException("FIXME: unrecognized function type in duckdb_functions");
		}
		if (finished) {
			// finished with this function, move to the next function
			data.offset++;
			data.offset_in_entry = 0;
		} else {
			// more functions remain
			data.offset_in_entry++;
		}
		count++;
	}
	output.SetCardinality(count);
}

void DuckDBFunctionsFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(
	    TableFunction("duckdb_functions", {}, DuckDBFunctionsFunction, DuckDBFunctionsBind, DuckDBFunctionsInit));
}

} // namespace duckdb











namespace duckdb {

struct DuckDBIndexesData : public GlobalTableFunctionState {
	DuckDBIndexesData() : offset(0) {
	}

	vector<CatalogEntry *> entries;
	idx_t offset;
};

static unique_ptr<FunctionData> DuckDBIndexesBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("database_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("database_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("schema_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("schema_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("index_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("index_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("table_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("table_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("is_unique");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("is_primary");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("expressions");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("sql");
	return_types.emplace_back(LogicalType::VARCHAR);

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBIndexesInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_unique<DuckDBIndexesData>();

	// scan all the schemas for tables and collect them and collect them
	auto schemas = Catalog::GetAllSchemas(context);
	for (auto &schema : schemas) {
		schema->Scan(context, CatalogType::INDEX_ENTRY, [&](CatalogEntry *entry) { result->entries.push_back(entry); });
	};
	return std::move(result);
}

void DuckDBIndexesFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (DuckDBIndexesData &)*data_p.global_state;
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = data.entries[data.offset++];

		auto &index = (IndexCatalogEntry &)*entry;
		// return values:

		idx_t col = 0;
		// database_name, VARCHAR
		output.SetValue(col++, count, index.catalog->GetName());
		// database_oid, BIGINT
		output.SetValue(col++, count, Value::BIGINT(index.catalog->GetOid()));
		// schema_name, VARCHAR
		output.SetValue(col++, count, Value(index.schema->name));
		// schema_oid, BIGINT
		output.SetValue(col++, count, Value::BIGINT(index.schema->oid));
		// index_name, VARCHAR
		output.SetValue(col++, count, Value(index.name));
		// index_oid, BIGINT
		output.SetValue(col++, count, Value::BIGINT(index.oid));
		// find the table in the catalog
		auto table_entry =
		    index.schema->catalog->GetEntry<TableCatalogEntry>(context, index.GetSchemaName(), index.GetTableName());
		// table_name, VARCHAR
		output.SetValue(col++, count, Value(table_entry->name));
		// table_oid, BIGINT
		output.SetValue(col++, count, Value::BIGINT(table_entry->oid));
		if (index.index) {
			// is_unique, BOOLEAN
			output.SetValue(col++, count, Value::BOOLEAN(index.index->IsUnique()));
			// is_primary, BOOLEAN
			output.SetValue(col++, count, Value::BOOLEAN(index.index->IsPrimary()));
		} else {
			output.SetValue(col++, count, Value());
			output.SetValue(col++, count, Value());
		}
		// expressions, VARCHAR
		output.SetValue(col++, count, Value());
		// sql, VARCHAR
		auto sql = index.ToSQL();
		output.SetValue(col++, count, sql.empty() ? Value() : Value(std::move(sql)));

		count++;
	}
	output.SetCardinality(count);
}

void DuckDBIndexesFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("duckdb_indexes", {}, DuckDBIndexesFunction, DuckDBIndexesBind, DuckDBIndexesInit));
}

} // namespace duckdb






namespace duckdb {

struct DuckDBKeywordsData : public GlobalTableFunctionState {
	DuckDBKeywordsData() : offset(0) {
	}

	vector<ParserKeyword> entries;
	idx_t offset;
};

static unique_ptr<FunctionData> DuckDBKeywordsBind(ClientContext &context, TableFunctionBindInput &input,
                                                   vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("keyword_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("keyword_category");
	return_types.emplace_back(LogicalType::VARCHAR);

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBKeywordsInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_unique<DuckDBKeywordsData>();
	result->entries = Parser::KeywordList();
	return std::move(result);
}

void DuckDBKeywordsFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (DuckDBKeywordsData &)*data_p.global_state;
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = data.entries[data.offset++];

		// keyword_name, VARCHAR
		output.SetValue(0, count, Value(entry.name));
		// keyword_category, VARCHAR
		string category_name;
		switch (entry.category) {
		case KeywordCategory::KEYWORD_RESERVED:
			category_name = "reserved";
			break;
		case KeywordCategory::KEYWORD_UNRESERVED:
			category_name = "unreserved";
			break;
		case KeywordCategory::KEYWORD_TYPE_FUNC:
			category_name = "type_function";
			break;
		case KeywordCategory::KEYWORD_COL_NAME:
			category_name = "column_name";
			break;
		default:
			throw InternalException("Unrecognized keyword category");
		}
		output.SetValue(1, count, Value(std::move(category_name)));

		count++;
	}
	output.SetCardinality(count);
}

void DuckDBKeywordsFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(
	    TableFunction("duckdb_keywords", {}, DuckDBKeywordsFunction, DuckDBKeywordsBind, DuckDBKeywordsInit));
}

} // namespace duckdb








namespace duckdb {

struct DuckDBSchemasData : public GlobalTableFunctionState {
	DuckDBSchemasData() : offset(0) {
	}

	vector<SchemaCatalogEntry *> entries;
	idx_t offset;
};

static unique_ptr<FunctionData> DuckDBSchemasBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("database_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("database_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("schema_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("internal");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("sql");
	return_types.emplace_back(LogicalType::VARCHAR);

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBSchemasInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_unique<DuckDBSchemasData>();

	// scan all the schemas and collect them
	result->entries = Catalog::GetAllSchemas(context);

	return std::move(result);
}

void DuckDBSchemasFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (DuckDBSchemasData &)*data_p.global_state;
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = data.entries[data.offset];

		// return values:
		idx_t col = 0;
		// "oid", PhysicalType::BIGINT
		output.SetValue(col++, count, Value::BIGINT(entry->oid));
		// database_name, VARCHAR
		output.SetValue(col++, count, entry->catalog->GetName());
		// database_oid, BIGINT
		output.SetValue(col++, count, Value::BIGINT(entry->catalog->GetOid()));
		// "schema_name", PhysicalType::VARCHAR
		output.SetValue(col++, count, Value(entry->name));
		// "internal", PhysicalType::BOOLEAN
		output.SetValue(col++, count, Value::BOOLEAN(entry->internal));
		// "sql", PhysicalType::VARCHAR
		output.SetValue(col++, count, Value());

		data.offset++;
		count++;
	}
	output.SetCardinality(count);
}

void DuckDBSchemasFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("duckdb_schemas", {}, DuckDBSchemasFunction, DuckDBSchemasBind, DuckDBSchemasInit));
}

} // namespace duckdb









namespace duckdb {

struct DuckDBSequencesData : public GlobalTableFunctionState {
	DuckDBSequencesData() : offset(0) {
	}

	vector<SequenceCatalogEntry *> entries;
	idx_t offset;
};

static unique_ptr<FunctionData> DuckDBSequencesBind(ClientContext &context, TableFunctionBindInput &input,
                                                    vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("database_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("database_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("schema_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("schema_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("sequence_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("sequence_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("temporary");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("start_value");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("min_value");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("max_value");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("increment_by");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("cycle");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("last_value");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("sql");
	return_types.emplace_back(LogicalType::VARCHAR);

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBSequencesInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_unique<DuckDBSequencesData>();

	// scan all the schemas for tables and collect themand collect them
	auto schemas = Catalog::GetAllSchemas(context);
	for (auto &schema : schemas) {
		schema->Scan(context, CatalogType::SEQUENCE_ENTRY,
		             [&](CatalogEntry *entry) { result->entries.push_back((SequenceCatalogEntry *)entry); });
	};
	return std::move(result);
}

void DuckDBSequencesFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (DuckDBSequencesData &)*data_p.global_state;
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = data.entries[data.offset++];

		auto &seq = (SequenceCatalogEntry &)*entry;
		// return values:
		idx_t col = 0;
		// database_name, VARCHAR
		output.SetValue(col++, count, entry->catalog->GetName());
		// database_oid, BIGINT
		output.SetValue(col++, count, Value::BIGINT(entry->catalog->GetOid()));
		// schema_name, VARCHAR
		output.SetValue(col++, count, Value(seq.schema->name));
		// schema_oid, BIGINT
		output.SetValue(col++, count, Value::BIGINT(seq.schema->oid));
		// sequence_name, VARCHAR
		output.SetValue(col++, count, Value(seq.name));
		// sequence_oid, BIGINT
		output.SetValue(col++, count, Value::BIGINT(seq.oid));
		// temporary, BOOLEAN
		output.SetValue(col++, count, Value::BOOLEAN(seq.temporary));
		// start_value, BIGINT
		output.SetValue(col++, count, Value::BIGINT(seq.start_value));
		// min_value, BIGINT
		output.SetValue(col++, count, Value::BIGINT(seq.min_value));
		// max_value, BIGINT
		output.SetValue(col++, count, Value::BIGINT(seq.max_value));
		// increment_by, BIGINT
		output.SetValue(col++, count, Value::BIGINT(seq.increment));
		// cycle, BOOLEAN
		output.SetValue(col++, count, Value::BOOLEAN(seq.cycle));
		// last_value, BIGINT
		output.SetValue(col++, count, seq.usage_count == 0 ? Value() : Value::BOOLEAN(seq.last_value));
		// sql, LogicalType::VARCHAR
		output.SetValue(col++, count, Value(seq.ToSQL()));

		count++;
	}
	output.SetCardinality(count);
}

void DuckDBSequencesFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(
	    TableFunction("duckdb_sequences", {}, DuckDBSequencesFunction, DuckDBSequencesBind, DuckDBSequencesInit));
}

} // namespace duckdb





namespace duckdb {

struct DuckDBSettingValue {
	string name;
	string value;
	string description;
	string input_type;
};

struct DuckDBSettingsData : public GlobalTableFunctionState {
	DuckDBSettingsData() : offset(0) {
	}

	vector<DuckDBSettingValue> settings;
	idx_t offset;
};

static unique_ptr<FunctionData> DuckDBSettingsBind(ClientContext &context, TableFunctionBindInput &input,
                                                   vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("value");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("description");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("input_type");
	return_types.emplace_back(LogicalType::VARCHAR);

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBSettingsInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_unique<DuckDBSettingsData>();

	auto &config = DBConfig::GetConfig(context);
	auto options_count = DBConfig::GetOptionCount();
	for (idx_t i = 0; i < options_count; i++) {
		auto option = DBConfig::GetOptionByIndex(i);
		D_ASSERT(option);
		DuckDBSettingValue value;
		value.name = option->name;
		value.value = option->get_setting(context).ToString();
		value.description = option->description;
		value.input_type = LogicalTypeIdToString(option->parameter_type);

		result->settings.push_back(std::move(value));
	}
	for (auto &ext_param : config.extension_parameters) {
		Value setting_val;
		string setting_str_val;
		if (context.TryGetCurrentSetting(ext_param.first, setting_val)) {
			setting_str_val = setting_val.ToString();
		}
		DuckDBSettingValue value;
		value.name = ext_param.first;
		value.value = std::move(setting_str_val);
		value.description = ext_param.second.description;
		value.input_type = ext_param.second.type.ToString();

		result->settings.push_back(std::move(value));
	}
	return std::move(result);
}

void DuckDBSettingsFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (DuckDBSettingsData &)*data_p.global_state;
	if (data.offset >= data.settings.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.settings.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = data.settings[data.offset++];

		// return values:
		// name, LogicalType::VARCHAR
		output.SetValue(0, count, Value(entry.name));
		// value, LogicalType::VARCHAR
		output.SetValue(1, count, Value(entry.value));
		// description, LogicalType::VARCHAR
		output.SetValue(2, count, Value(entry.description));
		// input_type, LogicalType::VARCHAR
		output.SetValue(3, count, Value(entry.input_type));
		count++;
	}
	output.SetCardinality(count);
}

void DuckDBSettingsFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(
	    TableFunction("duckdb_settings", {}, DuckDBSettingsFunction, DuckDBSettingsBind, DuckDBSettingsInit));
}

} // namespace duckdb













namespace duckdb {

struct DuckDBTablesData : public GlobalTableFunctionState {
	DuckDBTablesData() : offset(0) {
	}

	vector<CatalogEntry *> entries;
	idx_t offset;
};

static unique_ptr<FunctionData> DuckDBTablesBind(ClientContext &context, TableFunctionBindInput &input,
                                                 vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("database_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("database_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("schema_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("schema_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("table_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("table_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("internal");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("temporary");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("has_primary_key");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("estimated_size");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("column_count");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("index_count");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("check_constraint_count");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("sql");
	return_types.emplace_back(LogicalType::VARCHAR);

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBTablesInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_unique<DuckDBTablesData>();

	// scan all the schemas for tables and collect themand collect them
	auto schemas = Catalog::GetAllSchemas(context);
	for (auto &schema : schemas) {
		schema->Scan(context, CatalogType::TABLE_ENTRY, [&](CatalogEntry *entry) { result->entries.push_back(entry); });
	};
	return std::move(result);
}

static bool TableHasPrimaryKey(TableCatalogEntry &table) {
	for (auto &constraint : table.GetConstraints()) {
		if (constraint->type == ConstraintType::UNIQUE) {
			auto &unique = (UniqueConstraint &)*constraint;
			if (unique.is_primary_key) {
				return true;
			}
		}
	}
	return false;
}

static idx_t CheckConstraintCount(TableCatalogEntry &table) {
	idx_t check_count = 0;
	for (auto &constraint : table.GetConstraints()) {
		if (constraint->type == ConstraintType::CHECK) {
			check_count++;
		}
	}
	return check_count;
}

void DuckDBTablesFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (DuckDBTablesData &)*data_p.global_state;
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = data.entries[data.offset++];

		if (entry->type != CatalogType::TABLE_ENTRY) {
			continue;
		}
		auto &table = (TableCatalogEntry &)*entry;
		auto storage_info = table.GetStorageInfo(context);
		// return values:
		idx_t col = 0;
		// database_name, VARCHAR
		output.SetValue(col++, count, entry->catalog->GetName());
		// database_oid, BIGINT
		output.SetValue(col++, count, Value::BIGINT(entry->catalog->GetOid()));
		// schema_name, LogicalType::VARCHAR
		output.SetValue(col++, count, Value(table.schema->name));
		// schema_oid, LogicalType::BIGINT
		output.SetValue(col++, count, Value::BIGINT(table.schema->oid));
		// table_name, LogicalType::VARCHAR
		output.SetValue(col++, count, Value(table.name));
		// table_oid, LogicalType::BIGINT
		output.SetValue(col++, count, Value::BIGINT(table.oid));
		// internal, LogicalType::BOOLEAN
		output.SetValue(col++, count, Value::BOOLEAN(table.internal));
		// temporary, LogicalType::BOOLEAN
		output.SetValue(col++, count, Value::BOOLEAN(table.temporary));
		// has_primary_key, LogicalType::BOOLEAN
		output.SetValue(col++, count, Value::BOOLEAN(TableHasPrimaryKey(table)));
		// estimated_size, LogicalType::BIGINT
		Value card_val =
		    storage_info.cardinality == DConstants::INVALID_INDEX ? Value() : Value::BIGINT(storage_info.cardinality);
		output.SetValue(col++, count, card_val);
		// column_count, LogicalType::BIGINT
		output.SetValue(col++, count, Value::BIGINT(table.GetColumns().LogicalColumnCount()));
		// index_count, LogicalType::BIGINT
		output.SetValue(col++, count, Value::BIGINT(storage_info.index_info.size()));
		// check_constraint_count, LogicalType::BIGINT
		output.SetValue(col++, count, Value::BIGINT(CheckConstraintCount(table)));
		// sql, LogicalType::VARCHAR
		output.SetValue(col++, count, Value(table.ToSQL()));

		count++;
	}
	output.SetCardinality(count);
}

void DuckDBTablesFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("duckdb_tables", {}, DuckDBTablesFunction, DuckDBTablesBind, DuckDBTablesInit));
}

} // namespace duckdb









namespace duckdb {

struct DuckDBTypesData : public GlobalTableFunctionState {
	DuckDBTypesData() : offset(0) {
	}

	vector<TypeCatalogEntry *> entries;
	idx_t offset;
	unordered_set<int64_t> oids;
};

static unique_ptr<FunctionData> DuckDBTypesBind(ClientContext &context, TableFunctionBindInput &input,
                                                vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("database_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("database_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("schema_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("schema_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("type_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("type_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("type_size");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("logical_type");
	return_types.emplace_back(LogicalType::VARCHAR);

	// NUMERIC, STRING, DATETIME, BOOLEAN, COMPOSITE, USER
	names.emplace_back("type_category");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("internal");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("labels");
	return_types.emplace_back(LogicalType::LIST(LogicalType::VARCHAR));

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBTypesInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_unique<DuckDBTypesData>();
	auto schemas = Catalog::GetAllSchemas(context);
	for (auto &schema : schemas) {
		schema->Scan(context, CatalogType::TYPE_ENTRY,
		             [&](CatalogEntry *entry) { result->entries.push_back((TypeCatalogEntry *)entry); });
	};
	return std::move(result);
}

void DuckDBTypesFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (DuckDBTypesData &)*data_p.global_state;
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &type_entry = data.entries[data.offset++];
		auto &type = type_entry->user_type;

		// return values:
		idx_t col = 0;
		// database_name, VARCHAR
		output.SetValue(col++, count, type_entry->catalog->GetName());
		// database_oid, BIGINT
		output.SetValue(col++, count, Value::BIGINT(type_entry->catalog->GetOid()));
		// schema_name, LogicalType::VARCHAR
		output.SetValue(col++, count, Value(type_entry->schema->name));
		// schema_oid, LogicalType::BIGINT
		output.SetValue(col++, count, Value::BIGINT(type_entry->schema->oid));
		// type_oid, BIGINT
		int64_t oid;
		if (type_entry->internal) {
			oid = int64_t(type.id());
		} else {
			oid = type_entry->oid;
		}
		Value oid_val;
		if (data.oids.find(oid) == data.oids.end()) {
			data.oids.insert(oid);
			oid_val = Value::BIGINT(oid);
		} else {
			oid_val = Value();
		}
		output.SetValue(col++, count, oid_val);
		// type_name, VARCHAR
		output.SetValue(col++, count, Value(type_entry->name));
		// type_size, BIGINT
		auto internal_type = type.InternalType();
		output.SetValue(col++, count,
		                internal_type == PhysicalType::INVALID ? Value() : Value::BIGINT(GetTypeIdSize(internal_type)));
		// logical_type, VARCHAR
		output.SetValue(col++, count, Value(LogicalTypeIdToString(type.id())));
		// type_category, VARCHAR
		string category;
		switch (type.id()) {
		case LogicalTypeId::TINYINT:
		case LogicalTypeId::SMALLINT:
		case LogicalTypeId::INTEGER:
		case LogicalTypeId::BIGINT:
		case LogicalTypeId::DECIMAL:
		case LogicalTypeId::FLOAT:
		case LogicalTypeId::DOUBLE:
		case LogicalTypeId::UTINYINT:
		case LogicalTypeId::USMALLINT:
		case LogicalTypeId::UINTEGER:
		case LogicalTypeId::UBIGINT:
		case LogicalTypeId::HUGEINT:
			category = "NUMERIC";
			break;
		case LogicalTypeId::DATE:
		case LogicalTypeId::TIME:
		case LogicalTypeId::TIMESTAMP_SEC:
		case LogicalTypeId::TIMESTAMP_MS:
		case LogicalTypeId::TIMESTAMP:
		case LogicalTypeId::TIMESTAMP_NS:
		case LogicalTypeId::INTERVAL:
		case LogicalTypeId::TIME_TZ:
		case LogicalTypeId::TIMESTAMP_TZ:
			category = "DATETIME";
			break;
		case LogicalTypeId::CHAR:
		case LogicalTypeId::VARCHAR:
			category = "STRING";
			break;
		case LogicalTypeId::BOOLEAN:
			category = "BOOLEAN";
			break;
		case LogicalTypeId::STRUCT:
		case LogicalTypeId::LIST:
		case LogicalTypeId::MAP:
		case LogicalTypeId::UNION:
			category = "COMPOSITE";
			break;
		default:
			break;
		}
		output.SetValue(col++, count, category.empty() ? Value() : Value(category));
		// internal, BOOLEAN
		output.SetValue(col++, count, Value::BOOLEAN(type_entry->internal));
		// labels, VARCHAR[]
		if (type.id() == LogicalTypeId::ENUM && type.AuxInfo()) {
			auto data = FlatVector::GetData<string_t>(EnumType::GetValuesInsertOrder(type));
			idx_t size = EnumType::GetSize(type);

			vector<Value> labels;
			for (idx_t i = 0; i < size; i++) {
				labels.emplace_back(data[i]);
			}

			output.SetValue(col++, count, Value::LIST(labels));
		} else {
			output.SetValue(col++, count, Value());
		}

		count++;
	}
	output.SetCardinality(count);
}

void DuckDBTypesFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("duckdb_types", {}, DuckDBTypesFunction, DuckDBTypesBind, DuckDBTypesInit));
}

} // namespace duckdb









namespace duckdb {

struct DuckDBViewsData : public GlobalTableFunctionState {
	DuckDBViewsData() : offset(0) {
	}

	vector<CatalogEntry *> entries;
	idx_t offset;
};

static unique_ptr<FunctionData> DuckDBViewsBind(ClientContext &context, TableFunctionBindInput &input,
                                                vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("database_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("database_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("schema_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("schema_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("view_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("view_oid");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("internal");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("temporary");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("column_count");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("sql");
	return_types.emplace_back(LogicalType::VARCHAR);

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> DuckDBViewsInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_unique<DuckDBViewsData>();

	// scan all the schemas for tables and collect them and collect them
	auto schemas = Catalog::GetAllSchemas(context);
	for (auto &schema : schemas) {
		schema->Scan(context, CatalogType::VIEW_ENTRY, [&](CatalogEntry *entry) { result->entries.push_back(entry); });
	};
	return std::move(result);
}

void DuckDBViewsFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (DuckDBViewsData &)*data_p.global_state;
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = data.entries[data.offset++];

		if (entry->type != CatalogType::VIEW_ENTRY) {
			continue;
		}
		auto &view = (ViewCatalogEntry &)*entry;

		// return values:
		idx_t col = 0;
		// database_name, VARCHAR
		output.SetValue(col++, count, entry->catalog->GetName());
		// database_oid, BIGINT
		output.SetValue(col++, count, Value::BIGINT(entry->catalog->GetOid()));
		// schema_name, LogicalType::VARCHAR
		output.SetValue(col++, count, Value(view.schema->name));
		// schema_oid, LogicalType::BIGINT
		output.SetValue(col++, count, Value::BIGINT(view.schema->oid));
		// view_name, LogicalType::VARCHAR
		output.SetValue(col++, count, Value(view.name));
		// view_oid, LogicalType::BIGINT
		output.SetValue(col++, count, Value::BIGINT(view.oid));
		// internal, LogicalType::BOOLEAN
		output.SetValue(col++, count, Value::BOOLEAN(view.internal));
		// temporary, LogicalType::BOOLEAN
		output.SetValue(col++, count, Value::BOOLEAN(view.temporary));
		// column_count, LogicalType::BIGINT
		output.SetValue(col++, count, Value::BIGINT(view.types.size()));
		// sql, LogicalType::VARCHAR
		output.SetValue(col++, count, Value(view.ToSQL()));

		count++;
	}
	output.SetCardinality(count);
}

void DuckDBViewsFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("duckdb_views", {}, DuckDBViewsFunction, DuckDBViewsBind, DuckDBViewsInit));
}

} // namespace duckdb







namespace duckdb {

struct PragmaCollateData : public GlobalTableFunctionState {
	PragmaCollateData() : offset(0) {
	}

	vector<string> entries;
	idx_t offset;
};

static unique_ptr<FunctionData> PragmaCollateBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("collname");
	return_types.emplace_back(LogicalType::VARCHAR);

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> PragmaCollateInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_unique<PragmaCollateData>();

	auto schemas = Catalog::GetAllSchemas(context);
	for (auto schema : schemas) {
		schema->Scan(context, CatalogType::COLLATION_ENTRY,
		             [&](CatalogEntry *entry) { result->entries.push_back(entry->name); });
	}
	return std::move(result);
}

static void PragmaCollateFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (PragmaCollateData &)*data_p.global_state;
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	idx_t next = MinValue<idx_t>(data.offset + STANDARD_VECTOR_SIZE, data.entries.size());
	output.SetCardinality(next - data.offset);
	for (idx_t i = data.offset; i < next; i++) {
		auto index = i - data.offset;
		output.SetValue(0, index, Value(data.entries[i]));
	}

	data.offset = next;
}

void PragmaCollations::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(
	    TableFunction("pragma_collations", {}, PragmaCollateFunction, PragmaCollateBind, PragmaCollateInit));
}

} // namespace duckdb










namespace duckdb {

struct PragmaDatabaseSizeData : public GlobalTableFunctionState {
	PragmaDatabaseSizeData() : index(0) {
	}

	idx_t index;
	vector<AttachedDatabase *> databases;
	Value memory_usage;
	Value memory_limit;
};

static unique_ptr<FunctionData> PragmaDatabaseSizeBind(ClientContext &context, TableFunctionBindInput &input,
                                                       vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("database_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("database_size");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("block_size");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("total_blocks");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("used_blocks");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("free_blocks");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("wal_size");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("memory_usage");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("memory_limit");
	return_types.emplace_back(LogicalType::VARCHAR);

	return nullptr;
}

unique_ptr<GlobalTableFunctionState> PragmaDatabaseSizeInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_unique<PragmaDatabaseSizeData>();
	result->databases = DatabaseManager::Get(context).GetDatabases(context);
	auto &buffer_manager = BufferManager::GetBufferManager(context);
	result->memory_usage = Value(StringUtil::BytesToHumanReadableString(buffer_manager.GetUsedMemory()));
	auto max_memory = buffer_manager.GetMaxMemory();
	result->memory_limit =
	    max_memory == (idx_t)-1 ? Value("Unlimited") : Value(StringUtil::BytesToHumanReadableString(max_memory));

	return std::move(result);
}

void PragmaDatabaseSizeFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (PragmaDatabaseSizeData &)*data_p.global_state;
	idx_t row = 0;
	for (; data.index < data.databases.size() && row < STANDARD_VECTOR_SIZE; data.index++) {
		auto db = data.databases[data.index];
		if (db->IsSystem() || db->IsTemporary()) {
			continue;
		}
		auto ds = db->GetCatalog().GetDatabaseSize(context);
		idx_t col = 0;
		output.data[col++].SetValue(row, Value(db->GetName()));
		output.data[col++].SetValue(row, Value(StringUtil::BytesToHumanReadableString(ds.bytes)));
		output.data[col++].SetValue(row, Value::BIGINT(ds.block_size));
		output.data[col++].SetValue(row, Value::BIGINT(ds.total_blocks));
		output.data[col++].SetValue(row, Value::BIGINT(ds.used_blocks));
		output.data[col++].SetValue(row, Value::BIGINT(ds.free_blocks));
		output.data[col++].SetValue(
		    row, ds.wal_size == idx_t(-1) ? Value() : Value(StringUtil::BytesToHumanReadableString(ds.wal_size)));
		output.data[col++].SetValue(row, data.memory_usage);
		output.data[col++].SetValue(row, data.memory_limit);
		row++;
	}
	output.SetCardinality(row);
}

void PragmaDatabaseSize::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("pragma_database_size", {}, PragmaDatabaseSizeFunction, PragmaDatabaseSizeBind,
	                              PragmaDatabaseSizeInit));
}

} // namespace duckdb














#include <algorithm>

namespace duckdb {

struct PragmaStorageFunctionData : public TableFunctionData {
	explicit PragmaStorageFunctionData(TableCatalogEntry *table_entry) : table_entry(table_entry) {
	}

	TableCatalogEntry *table_entry;
	TableStorageInfo storage_info;
};

struct PragmaStorageOperatorData : public GlobalTableFunctionState {
	PragmaStorageOperatorData() : offset(0) {
	}

	idx_t offset;
};

static unique_ptr<FunctionData> PragmaStorageInfoBind(ClientContext &context, TableFunctionBindInput &input,
                                                      vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("row_group_id");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("column_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("column_id");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("column_path");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("segment_id");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("segment_type");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("start");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("count");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("compression");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("stats");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("has_updates");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("persistent");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("block_id");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("block_offset");
	return_types.emplace_back(LogicalType::BIGINT);

	auto qname = QualifiedName::Parse(input.inputs[0].GetValue<string>());

	// look up the table name in the catalog
	auto table_entry = Catalog::GetEntry<TableCatalogEntry>(context, qname.catalog, qname.schema, qname.name);
	auto result = make_unique<PragmaStorageFunctionData>(table_entry);
	result->storage_info = table_entry->GetStorageInfo(context);
	return std::move(result);
}

unique_ptr<GlobalTableFunctionState> PragmaStorageInfoInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_unique<PragmaStorageOperatorData>();
}

static void PragmaStorageInfoFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = (PragmaStorageFunctionData &)*data_p.bind_data;
	auto &data = (PragmaStorageOperatorData &)*data_p.global_state;
	idx_t count = 0;
	auto &columns = bind_data.table_entry->GetColumns();
	while (data.offset < bind_data.storage_info.column_segments.size() && count < STANDARD_VECTOR_SIZE) {
		auto &entry = bind_data.storage_info.column_segments[data.offset++];

		idx_t col_idx = 0;
		// row_group_id
		output.SetValue(col_idx++, count, Value::BIGINT(entry.row_group_index));
		// column_name
		auto &col = columns.GetColumn(PhysicalIndex(entry.column_id));
		output.SetValue(col_idx++, count, Value(col.Name()));
		// column_id
		output.SetValue(col_idx++, count, Value::BIGINT(entry.column_id));
		// column_path
		output.SetValue(col_idx++, count, Value(entry.column_path));
		// segment_id
		output.SetValue(col_idx++, count, Value::BIGINT(entry.segment_idx));
		// segment_type
		output.SetValue(col_idx++, count, Value(entry.segment_type));
		// start
		output.SetValue(col_idx++, count, Value::BIGINT(entry.segment_start));
		// count
		output.SetValue(col_idx++, count, Value::BIGINT(entry.segment_count));
		// compression
		output.SetValue(col_idx++, count, Value(entry.compression_type));
		// stats
		output.SetValue(col_idx++, count, Value(entry.segment_stats));
		// has_updates
		output.SetValue(col_idx++, count, Value::BOOLEAN(entry.has_updates));
		// persistent
		output.SetValue(col_idx++, count, Value::BOOLEAN(entry.persistent));
		// block_id
		// block_offset
		if (entry.persistent) {
			output.SetValue(col_idx++, count, Value::BIGINT(entry.block_id));
			output.SetValue(col_idx++, count, Value::BIGINT(entry.block_offset));
		} else {
			output.SetValue(col_idx++, count, Value());
			output.SetValue(col_idx++, count, Value());
		}
		count++;
	}
	output.SetCardinality(count);
}

void PragmaStorageInfo::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("pragma_storage_info", {LogicalType::VARCHAR}, PragmaStorageInfoFunction,
	                              PragmaStorageInfoBind, PragmaStorageInfoInit));
}

} // namespace duckdb














#include <algorithm>

namespace duckdb {

struct PragmaTableFunctionData : public TableFunctionData {
	explicit PragmaTableFunctionData(CatalogEntry *entry_p) : entry(entry_p) {
	}

	CatalogEntry *entry;
};

struct PragmaTableOperatorData : public GlobalTableFunctionState {
	PragmaTableOperatorData() : offset(0) {
	}
	idx_t offset;
};

static unique_ptr<FunctionData> PragmaTableInfoBind(ClientContext &context, TableFunctionBindInput &input,
                                                    vector<LogicalType> &return_types, vector<string> &names) {

	names.emplace_back("cid");
	return_types.emplace_back(LogicalType::INTEGER);

	names.emplace_back("name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("type");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("notnull");
	return_types.emplace_back(LogicalType::BOOLEAN);

	names.emplace_back("dflt_value");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("pk");
	return_types.emplace_back(LogicalType::BOOLEAN);

	auto qname = QualifiedName::Parse(input.inputs[0].GetValue<string>());

	// look up the table name in the catalog
	Binder::BindSchemaOrCatalog(context, qname.catalog, qname.schema);
	auto entry = Catalog::GetEntry(context, CatalogType::TABLE_ENTRY, qname.catalog, qname.schema, qname.name);
	return make_unique<PragmaTableFunctionData>(entry);
}

unique_ptr<GlobalTableFunctionState> PragmaTableInfoInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_unique<PragmaTableOperatorData>();
}

static void CheckConstraints(TableCatalogEntry *table, const ColumnDefinition &column, bool &out_not_null,
                             bool &out_pk) {
	out_not_null = false;
	out_pk = false;
	// check all constraints
	// FIXME: this is pretty inefficient, it probably doesn't matter
	for (auto &constraint : table->GetConstraints()) {
		switch (constraint->type) {
		case ConstraintType::NOT_NULL: {
			auto &not_null = (NotNullConstraint &)*constraint;
			if (not_null.index == column.Logical()) {
				out_not_null = true;
			}
			break;
		}
		case ConstraintType::UNIQUE: {
			auto &unique = (UniqueConstraint &)*constraint;

			if (unique.is_primary_key) {
				if (unique.index == column.Logical()) {
					out_pk = true;
				}
				if (std::find(unique.columns.begin(), unique.columns.end(), column.GetName()) != unique.columns.end()) {
					out_pk = true;
				}
			}
			break;
		}
		default:
			break;
		}
	}
}

static void PragmaTableInfoTable(PragmaTableOperatorData &data, TableCatalogEntry *table, DataChunk &output) {
	if (data.offset >= table->GetColumns().LogicalColumnCount()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t next = MinValue<idx_t>(data.offset + STANDARD_VECTOR_SIZE, table->GetColumns().LogicalColumnCount());
	output.SetCardinality(next - data.offset);

	for (idx_t i = data.offset; i < next; i++) {
		bool not_null, pk;
		auto index = i - data.offset;
		auto &column = table->GetColumn(LogicalIndex(i));
		D_ASSERT(column.Oid() < (idx_t)NumericLimits<int32_t>::Maximum());
		CheckConstraints(table, column, not_null, pk);

		// return values:
		// "cid", PhysicalType::INT32
		output.SetValue(0, index, Value::INTEGER((int32_t)column.Oid()));
		// "name", PhysicalType::VARCHAR
		output.SetValue(1, index, Value(column.Name()));
		// "type", PhysicalType::VARCHAR
		output.SetValue(2, index, Value(column.Type().ToString()));
		// "notnull", PhysicalType::BOOL
		output.SetValue(3, index, Value::BOOLEAN(not_null));
		// "dflt_value", PhysicalType::VARCHAR
		Value def_value = column.DefaultValue() ? Value(column.DefaultValue()->ToString()) : Value();
		output.SetValue(4, index, def_value);
		// "pk", PhysicalType::BOOL
		output.SetValue(5, index, Value::BOOLEAN(pk));
	}
	data.offset = next;
}

static void PragmaTableInfoView(PragmaTableOperatorData &data, ViewCatalogEntry *view, DataChunk &output) {
	if (data.offset >= view->types.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t next = MinValue<idx_t>(data.offset + STANDARD_VECTOR_SIZE, view->types.size());
	output.SetCardinality(next - data.offset);

	for (idx_t i = data.offset; i < next; i++) {
		auto index = i - data.offset;
		auto type = view->types[i];
		auto &name = view->aliases[i];
		// return values:
		// "cid", PhysicalType::INT32

		output.SetValue(0, index, Value::INTEGER((int32_t)i));
		// "name", PhysicalType::VARCHAR
		output.SetValue(1, index, Value(name));
		// "type", PhysicalType::VARCHAR
		output.SetValue(2, index, Value(type.ToString()));
		// "notnull", PhysicalType::BOOL
		output.SetValue(3, index, Value::BOOLEAN(false));
		// "dflt_value", PhysicalType::VARCHAR
		output.SetValue(4, index, Value());
		// "pk", PhysicalType::BOOL
		output.SetValue(5, index, Value::BOOLEAN(false));
	}
	data.offset = next;
}

static void PragmaTableInfoFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = (PragmaTableFunctionData &)*data_p.bind_data;
	auto &state = (PragmaTableOperatorData &)*data_p.global_state;
	switch (bind_data.entry->type) {
	case CatalogType::TABLE_ENTRY:
		PragmaTableInfoTable(state, (TableCatalogEntry *)bind_data.entry, output);
		break;
	case CatalogType::VIEW_ENTRY:
		PragmaTableInfoView(state, (ViewCatalogEntry *)bind_data.entry, output);
		break;
	default:
		throw NotImplementedException("Unimplemented catalog type for pragma_table_info");
	}
}

void PragmaTableInfo::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("pragma_table_info", {LogicalType::VARCHAR}, PragmaTableInfoFunction,
	                              PragmaTableInfoBind, PragmaTableInfoInit));
}

} // namespace duckdb






#include <cmath>
#include <limits>

namespace duckdb {

struct TestAllTypesData : public GlobalTableFunctionState {
	TestAllTypesData() : offset(0) {
	}

	vector<vector<Value>> entries;
	idx_t offset;
};

vector<TestType> TestAllTypesFun::GetTestTypes() {
	vector<TestType> result;
	// scalar types/numerics
	result.emplace_back(LogicalType::BOOLEAN, "bool");
	result.emplace_back(LogicalType::TINYINT, "tinyint");
	result.emplace_back(LogicalType::SMALLINT, "smallint");
	result.emplace_back(LogicalType::INTEGER, "int");
	result.emplace_back(LogicalType::BIGINT, "bigint");
	result.emplace_back(LogicalType::HUGEINT, "hugeint");
	result.emplace_back(LogicalType::UTINYINT, "utinyint");
	result.emplace_back(LogicalType::USMALLINT, "usmallint");
	result.emplace_back(LogicalType::UINTEGER, "uint");
	result.emplace_back(LogicalType::UBIGINT, "ubigint");
	result.emplace_back(LogicalType::DATE, "date");
	result.emplace_back(LogicalType::TIME, "time");
	result.emplace_back(LogicalType::TIMESTAMP, "timestamp");
	result.emplace_back(LogicalType::TIMESTAMP_S, "timestamp_s");
	result.emplace_back(LogicalType::TIMESTAMP_MS, "timestamp_ms");
	result.emplace_back(LogicalType::TIMESTAMP_NS, "timestamp_ns");
	result.emplace_back(LogicalType::TIME_TZ, "time_tz");
	result.emplace_back(LogicalType::TIMESTAMP_TZ, "timestamp_tz");
	result.emplace_back(LogicalType::FLOAT, "float");
	result.emplace_back(LogicalType::DOUBLE, "double");
	result.emplace_back(LogicalType::DECIMAL(4, 1), "dec_4_1");
	result.emplace_back(LogicalType::DECIMAL(9, 4), "dec_9_4");
	result.emplace_back(LogicalType::DECIMAL(18, 6), "dec_18_6");
	result.emplace_back(LogicalType::DECIMAL(38, 10), "dec38_10");
	result.emplace_back(LogicalType::UUID, "uuid");

	// interval
	interval_t min_interval;
	min_interval.months = 0;
	min_interval.days = 0;
	min_interval.micros = 0;

	interval_t max_interval;
	max_interval.months = 999;
	max_interval.days = 999;
	max_interval.micros = 999999999;
	result.emplace_back(LogicalType::INTERVAL, "interval", Value::INTERVAL(min_interval),
	                    Value::INTERVAL(max_interval));
	// strings/blobs/bitstrings
	result.emplace_back(LogicalType::VARCHAR, "varchar", Value("🦆🦆🦆🦆🦆🦆"),
	                    Value(string("goo\x00se", 6)));
	result.emplace_back(LogicalType::BLOB, "blob", Value::BLOB("thisisalongblob\\x00withnullbytes"),
	                    Value::BLOB("\\x00\\x00\\x00a"));
	result.emplace_back(LogicalType::BIT, "bit", Value::BIT("0010001001011100010101011010111"), Value::BIT("10101"));

	// enums
	Vector small_enum(LogicalType::VARCHAR, 2);
	auto small_enum_ptr = FlatVector::GetData<string_t>(small_enum);
	small_enum_ptr[0] = StringVector::AddStringOrBlob(small_enum, "DUCK_DUCK_ENUM");
	small_enum_ptr[1] = StringVector::AddStringOrBlob(small_enum, "GOOSE");
	result.emplace_back(LogicalType::ENUM("small_enum", small_enum, 2), "small_enum");

	Vector medium_enum(LogicalType::VARCHAR, 300);
	auto medium_enum_ptr = FlatVector::GetData<string_t>(medium_enum);
	for (idx_t i = 0; i < 300; i++) {
		medium_enum_ptr[i] = StringVector::AddStringOrBlob(medium_enum, string("enum_") + to_string(i));
	}
	result.emplace_back(LogicalType::ENUM("medium_enum", medium_enum, 300), "medium_enum");

	// this is a big one... not sure if we should push this one here, but it's required for completeness
	Vector large_enum(LogicalType::VARCHAR, 70000);
	auto large_enum_ptr = FlatVector::GetData<string_t>(large_enum);
	for (idx_t i = 0; i < 70000; i++) {
		large_enum_ptr[i] = StringVector::AddStringOrBlob(large_enum, string("enum_") + to_string(i));
	}
	result.emplace_back(LogicalType::ENUM("large_enum", large_enum, 70000), "large_enum");

	// arrays
	auto int_list_type = LogicalType::LIST(LogicalType::INTEGER);
	auto empty_int_list = Value::EMPTYLIST(LogicalType::INTEGER);
	auto int_list = Value::LIST({Value::INTEGER(42), Value::INTEGER(999), Value(LogicalType::INTEGER),
	                             Value(LogicalType::INTEGER), Value::INTEGER(-42)});
	result.emplace_back(int_list_type, "int_array", empty_int_list, int_list);

	auto double_list_type = LogicalType::LIST(LogicalType::DOUBLE);
	auto empty_double_list = Value::EMPTYLIST(LogicalType::DOUBLE);
	auto double_list = Value::LIST(
	    {Value::DOUBLE(42), Value::DOUBLE(NAN), Value::DOUBLE(std::numeric_limits<double>::infinity()),
	     Value::DOUBLE(-std::numeric_limits<double>::infinity()), Value(LogicalType::DOUBLE), Value::DOUBLE(-42)});
	result.emplace_back(double_list_type, "double_array", empty_double_list, double_list);

	auto date_list_type = LogicalType::LIST(LogicalType::DATE);
	auto empty_date_list = Value::EMPTYLIST(LogicalType::DATE);
	auto date_list =
	    Value::LIST({Value::DATE(date_t()), Value::DATE(date_t::infinity()), Value::DATE(date_t::ninfinity()),
	                 Value(LogicalType::DATE), Value::DATE(Date::FromString("2022-05-12"))});
	result.emplace_back(date_list_type, "date_array", empty_date_list, date_list);

	auto timestamp_list_type = LogicalType::LIST(LogicalType::TIMESTAMP);
	auto empty_timestamp_list = Value::EMPTYLIST(LogicalType::TIMESTAMP);
	auto timestamp_list = Value::LIST({Value::TIMESTAMP(timestamp_t()), Value::TIMESTAMP(timestamp_t::infinity()),
	                                   Value::TIMESTAMP(timestamp_t::ninfinity()), Value(LogicalType::TIMESTAMP),
	                                   Value::TIMESTAMP(Timestamp::FromString("2022-05-12 16:23:45"))});
	result.emplace_back(timestamp_list_type, "timestamp_array", empty_timestamp_list, timestamp_list);

	auto timestamptz_list_type = LogicalType::LIST(LogicalType::TIMESTAMP_TZ);
	auto empty_timestamptz_list = Value::EMPTYLIST(LogicalType::TIMESTAMP_TZ);
	auto timestamptz_list = Value::LIST({Value::TIMESTAMPTZ(timestamp_t()), Value::TIMESTAMPTZ(timestamp_t::infinity()),
	                                     Value::TIMESTAMPTZ(timestamp_t::ninfinity()), Value(LogicalType::TIMESTAMP_TZ),
	                                     Value::TIMESTAMPTZ(Timestamp::FromString("2022-05-12 16:23:45-07"))});
	result.emplace_back(timestamptz_list_type, "timestamptz_array", empty_timestamptz_list, timestamptz_list);

	auto varchar_list_type = LogicalType::LIST(LogicalType::VARCHAR);
	auto empty_varchar_list = Value::EMPTYLIST(LogicalType::VARCHAR);
	auto varchar_list =
	    Value::LIST({Value("🦆🦆🦆🦆🦆🦆"), Value("goose"), Value(LogicalType::VARCHAR), Value("")});
	result.emplace_back(varchar_list_type, "varchar_array", empty_varchar_list, varchar_list);

	// nested arrays
	auto nested_list_type = LogicalType::LIST(int_list_type);
	auto empty_nested_list = Value::EMPTYLIST(int_list_type);
	auto nested_int_list = Value::LIST({empty_int_list, int_list, Value(int_list_type), empty_int_list, int_list});
	result.emplace_back(nested_list_type, "nested_int_array", empty_nested_list, nested_int_list);

	// structs
	child_list_t<LogicalType> struct_type_list;
	struct_type_list.push_back(make_pair("a", LogicalType::INTEGER));
	struct_type_list.push_back(make_pair("b", LogicalType::VARCHAR));
	auto struct_type = LogicalType::STRUCT(std::move(struct_type_list));

	child_list_t<Value> min_struct_list;
	min_struct_list.push_back(make_pair("a", Value(LogicalType::INTEGER)));
	min_struct_list.push_back(make_pair("b", Value(LogicalType::VARCHAR)));
	auto min_struct_val = Value::STRUCT(std::move(min_struct_list));

	child_list_t<Value> max_struct_list;
	max_struct_list.push_back(make_pair("a", Value::INTEGER(42)));
	max_struct_list.push_back(make_pair("b", Value("🦆🦆🦆🦆🦆🦆")));
	auto max_struct_val = Value::STRUCT(std::move(max_struct_list));

	result.emplace_back(struct_type, "struct", min_struct_val, max_struct_val);

	// structs with lists
	child_list_t<LogicalType> struct_list_type_list;
	struct_list_type_list.push_back(make_pair("a", int_list_type));
	struct_list_type_list.push_back(make_pair("b", varchar_list_type));
	auto struct_list_type = LogicalType::STRUCT(std::move(struct_list_type_list));

	child_list_t<Value> min_struct_vl_list;
	min_struct_vl_list.push_back(make_pair("a", Value(int_list_type)));
	min_struct_vl_list.push_back(make_pair("b", Value(varchar_list_type)));
	auto min_struct_val_list = Value::STRUCT(std::move(min_struct_vl_list));

	child_list_t<Value> max_struct_vl_list;
	max_struct_vl_list.push_back(make_pair("a", int_list));
	max_struct_vl_list.push_back(make_pair("b", varchar_list));
	auto max_struct_val_list = Value::STRUCT(std::move(max_struct_vl_list));

	result.emplace_back(struct_list_type, "struct_of_arrays", std::move(min_struct_val_list),
	                    std::move(max_struct_val_list));

	// array of structs
	auto array_of_structs_type = LogicalType::LIST(struct_type);
	auto min_array_of_struct_val = Value::EMPTYLIST(struct_type);
	auto max_array_of_struct_val = Value::LIST({min_struct_val, max_struct_val, Value(struct_type)});
	result.emplace_back(array_of_structs_type, "array_of_structs", std::move(min_array_of_struct_val),
	                    std::move(max_array_of_struct_val));

	// map
	auto map_type = LogicalType::MAP(LogicalType::VARCHAR, LogicalType::VARCHAR);
	auto min_map_value = Value::MAP(ListType::GetChildType(map_type), std::vector<Value>());

	child_list_t<Value> map_struct1;
	map_struct1.push_back(make_pair("key", Value("key1")));
	map_struct1.push_back(make_pair("value", Value("🦆🦆🦆🦆🦆🦆")));
	child_list_t<Value> map_struct2;
	map_struct2.push_back(make_pair("key", Value("key2")));
	map_struct2.push_back(make_pair("key", Value("goose")));

	std::vector<Value> map_values;
	map_values.push_back(Value::STRUCT(map_struct1));
	map_values.push_back(Value::STRUCT(map_struct2));

	auto max_map_value = Value::MAP(ListType::GetChildType(map_type), map_values);
	result.emplace_back(map_type, "map", std::move(min_map_value), std::move(max_map_value));

	return result;
}

static unique_ptr<FunctionData> TestAllTypesBind(ClientContext &context, TableFunctionBindInput &input,
                                                 vector<LogicalType> &return_types, vector<string> &names) {
	auto test_types = TestAllTypesFun::GetTestTypes();
	for (auto &test_type : test_types) {
		return_types.push_back(std::move(test_type.type));
		names.push_back(std::move(test_type.name));
	}
	return nullptr;
}

unique_ptr<GlobalTableFunctionState> TestAllTypesInit(ClientContext &context, TableFunctionInitInput &input) {
	auto result = make_unique<TestAllTypesData>();
	auto test_types = TestAllTypesFun::GetTestTypes();
	// 3 rows: min, max and NULL
	result->entries.resize(3);
	// initialize the values
	for (auto &test_type : test_types) {
		result->entries[0].push_back(std::move(test_type.min_value));
		result->entries[1].push_back(std::move(test_type.max_value));
		result->entries[2].emplace_back(std::move(test_type.type));
	}
	return std::move(result);
}

void TestAllTypesFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (TestAllTypesData &)*data_p.global_state;
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	// start returning values
	// either fill up the chunk or return all the remaining columns
	idx_t count = 0;
	while (data.offset < data.entries.size() && count < STANDARD_VECTOR_SIZE) {
		auto &vals = data.entries[data.offset++];
		for (idx_t col_idx = 0; col_idx < vals.size(); col_idx++) {
			output.SetValue(col_idx, count, vals[col_idx]);
		}
		count++;
	}
	output.SetCardinality(count);
}

void TestAllTypesFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("test_all_types", {}, TestAllTypesFunction, TestAllTypesBind, TestAllTypesInit));
}

} // namespace duckdb




namespace duckdb {

// FLAT, CONSTANT, DICTIONARY, SEQUENCE
struct TestVectorBindData : public TableFunctionData {
	LogicalType type;
	bool all_flat;
};

struct TestVectorTypesData : public GlobalTableFunctionState {
	TestVectorTypesData() : offset(0) {
	}

	vector<unique_ptr<DataChunk>> entries;
	idx_t offset;
};

struct TestVectorInfo {
	TestVectorInfo(const LogicalType &type, const map<LogicalTypeId, TestType> &test_type_map,
	               vector<unique_ptr<DataChunk>> &entries)
	    : type(type), test_type_map(test_type_map), entries(entries) {
	}

	const LogicalType &type;
	const map<LogicalTypeId, TestType> &test_type_map;
	vector<unique_ptr<DataChunk>> &entries;
};

struct TestVectorFlat {
	static constexpr const idx_t TEST_VECTOR_CARDINALITY = 3;

	static vector<Value> GenerateValues(TestVectorInfo &info, const LogicalType &type) {
		vector<Value> result;
		switch (type.InternalType()) {
		case PhysicalType::STRUCT: {
			vector<child_list_t<Value>> struct_children;
			auto &child_types = StructType::GetChildTypes(type);

			struct_children.resize(TEST_VECTOR_CARDINALITY);
			for (auto &child_type : child_types) {
				auto child_values = GenerateValues(info, child_type.second);

				for (idx_t i = 0; i < child_values.size(); i++) {
					struct_children[i].push_back(make_pair(child_type.first, std::move(child_values[i])));
				}
			}
			for (auto &struct_child : struct_children) {
				result.push_back(Value::STRUCT(std::move(struct_child)));
			}
			break;
		}
		case PhysicalType::LIST: {
			auto &child_type = ListType::GetChildType(type);
			auto child_values = GenerateValues(info, child_type);

			result.push_back(Value::LIST(child_type, {child_values[0], child_values[1]}));
			result.push_back(Value::LIST(child_type, {}));
			result.push_back(Value::LIST(child_type, {child_values[2]}));
			break;
		}
		default: {
			auto entry = info.test_type_map.find(type.id());
			if (entry == info.test_type_map.end()) {
				throw NotImplementedException("Unimplemented type for test_vector_types %s", type.ToString());
			}
			result.push_back(entry->second.min_value);
			result.push_back(entry->second.max_value);
			result.emplace_back(type);
			break;
		}
		}
		return result;
	}

	static void Generate(TestVectorInfo &info) {
		vector<Value> result_values = GenerateValues(info, info.type);
		for (idx_t cur_row = 0; cur_row < result_values.size(); cur_row += STANDARD_VECTOR_SIZE) {
			auto result = make_unique<DataChunk>();
			result->Initialize(Allocator::DefaultAllocator(), {info.type});
			auto cardinality = MinValue<idx_t>(STANDARD_VECTOR_SIZE, result_values.size() - cur_row);
			for (idx_t i = 0; i < cardinality; i++) {
				result->data[0].SetValue(i, result_values[cur_row + i]);
			}
			result->SetCardinality(cardinality);
			info.entries.push_back(std::move(result));
		}
	}
};

struct TestVectorConstant {
	static void Generate(TestVectorInfo &info) {
		auto values = TestVectorFlat::GenerateValues(info, info.type);
		for (idx_t cur_row = 0; cur_row < TestVectorFlat::TEST_VECTOR_CARDINALITY; cur_row += STANDARD_VECTOR_SIZE) {
			auto result = make_unique<DataChunk>();
			result->Initialize(Allocator::DefaultAllocator(), {info.type});
			auto cardinality = MinValue<idx_t>(STANDARD_VECTOR_SIZE, TestVectorFlat::TEST_VECTOR_CARDINALITY - cur_row);
			result->data[0].SetValue(0, values[0]);
			result->data[0].SetVectorType(VectorType::CONSTANT_VECTOR);
			result->SetCardinality(cardinality);

			info.entries.push_back(std::move(result));
		}
	}
};

struct TestVectorSequence {
	static void GenerateVector(TestVectorInfo &info, const LogicalType &type, Vector &result) {
		D_ASSERT(type == result.GetType());
		switch (type.id()) {
		case LogicalTypeId::TINYINT:
		case LogicalTypeId::SMALLINT:
		case LogicalTypeId::INTEGER:
		case LogicalTypeId::BIGINT:
		case LogicalTypeId::UTINYINT:
		case LogicalTypeId::USMALLINT:
		case LogicalTypeId::UINTEGER:
		case LogicalTypeId::UBIGINT:
			result.Sequence(3, 2, 3);
			return;
		default:
			break;
		}
		switch (type.InternalType()) {
		case PhysicalType::STRUCT: {
			auto &child_entries = StructVector::GetEntries(result);
			for (auto &child_entry : child_entries) {
				GenerateVector(info, child_entry->GetType(), *child_entry);
			}
			break;
		}
		case PhysicalType::LIST: {
			auto data = FlatVector::GetData<list_entry_t>(result);
			data[0].offset = 0;
			data[0].length = 2;
			data[1].offset = 2;
			data[1].length = 0;
			data[2].offset = 2;
			data[2].length = 1;

			GenerateVector(info, ListType::GetChildType(type), ListVector::GetEntry(result));
			ListVector::SetListSize(result, 3);
			break;
		}
		default: {
			auto entry = info.test_type_map.find(type.id());
			if (entry == info.test_type_map.end()) {
				throw NotImplementedException("Unimplemented type for test_vector_types %s", type.ToString());
			}
			result.SetValue(0, entry->second.min_value);
			result.SetValue(1, entry->second.max_value);
			result.SetValue(2, Value(type));
			break;
		}
		}
	}

	static void Generate(TestVectorInfo &info) {
#if STANDARD_VECTOR_SIZE > 2
		auto result = make_unique<DataChunk>();
		result->Initialize(Allocator::DefaultAllocator(), {info.type});

		GenerateVector(info, info.type, result->data[0]);
		result->SetCardinality(3);
		info.entries.push_back(std::move(result));
#endif
	}
};

struct TestVectorDictionary {
	static void Generate(TestVectorInfo &info) {
		idx_t current_chunk = info.entries.size();

		unordered_set<idx_t> slice_entries {1, 2};

		TestVectorFlat::Generate(info);
		idx_t current_idx = 0;
		for (idx_t i = current_chunk; i < info.entries.size(); i++) {
			auto &chunk = *info.entries[i];
			SelectionVector sel(STANDARD_VECTOR_SIZE);
			idx_t sel_idx = 0;
			for (idx_t k = 0; k < chunk.size(); k++) {
				if (slice_entries.count(current_idx + k) > 0) {
					sel.set_index(sel_idx++, k);
				}
			}
			chunk.Slice(sel, sel_idx);
			current_idx += chunk.size();
		}
	}
};

static unique_ptr<FunctionData> TestVectorTypesBind(ClientContext &context, TableFunctionBindInput &input,
                                                    vector<LogicalType> &return_types, vector<string> &names) {
	auto result = make_unique<TestVectorBindData>();
	result->type = input.inputs[0].type();
	result->all_flat = BooleanValue::Get(input.inputs[1]);

	return_types.push_back(result->type);
	names.emplace_back("test_vector");
	return std::move(result);
}

unique_ptr<GlobalTableFunctionState> TestVectorTypesInit(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind_data = (TestVectorBindData &)*input.bind_data;

	auto result = make_unique<TestVectorTypesData>();

	auto test_types = TestAllTypesFun::GetTestTypes();

	map<LogicalTypeId, TestType> test_type_map;
	for (auto &test_type : test_types) {
		test_type_map.insert(make_pair(test_type.type.id(), std::move(test_type)));
	}

	TestVectorInfo info(bind_data.type, test_type_map, result->entries);
	TestVectorFlat::Generate(info);
	TestVectorConstant::Generate(info);
	TestVectorDictionary::Generate(info);
	TestVectorSequence::Generate(info);
	for (auto &entry : result->entries) {
		entry->Verify();
	}
	if (bind_data.all_flat) {
		for (auto &entry : result->entries) {
			entry->Flatten();
			entry->Verify();
		}
	}
	return std::move(result);
}

void TestVectorTypesFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (TestVectorTypesData &)*data_p.global_state;
	if (data.offset >= data.entries.size()) {
		// finished returning values
		return;
	}
	output.Reference(*data.entries[data.offset]);
	data.offset++;
}

void TestVectorTypesFun::RegisterFunction(BuiltinFunctions &set) {
	set.AddFunction(TableFunction("test_vector_types", {LogicalType::ANY, LogicalType::BOOLEAN},
	                              TestVectorTypesFunction, TestVectorTypesBind, TestVectorTypesInit));
}

} // namespace duckdb








namespace duckdb {

void BuiltinFunctions::RegisterSQLiteFunctions() {
	PragmaVersion::RegisterFunction(*this);
	PragmaCollations::RegisterFunction(*this);
	PragmaTableInfo::RegisterFunction(*this);
	PragmaStorageInfo::RegisterFunction(*this);
	PragmaDatabaseSize::RegisterFunction(*this);
	PragmaLastProfilingOutput::RegisterFunction(*this);
	PragmaDetailedProfilingOutput::RegisterFunction(*this);

	DuckDBColumnsFun::RegisterFunction(*this);
	DuckDBConstraintsFun::RegisterFunction(*this);
	DuckDBDatabasesFun::RegisterFunction(*this);
	DuckDBFunctionsFun::RegisterFunction(*this);
	DuckDBKeywordsFun::RegisterFunction(*this);
	DuckDBIndexesFun::RegisterFunction(*this);
	DuckDBSchemasFun::RegisterFunction(*this);
	DuckDBDependenciesFun::RegisterFunction(*this);
	DuckDBExtensionsFun::RegisterFunction(*this);
	DuckDBSequencesFun::RegisterFunction(*this);
	DuckDBSettingsFun::RegisterFunction(*this);
	DuckDBTablesFun::RegisterFunction(*this);
	DuckDBTypesFun::RegisterFunction(*this);
	DuckDBViewsFun::RegisterFunction(*this);
	TestAllTypesFun::RegisterFunction(*this);
	TestVectorTypesFun::RegisterFunction(*this);
}

} // namespace duckdb


















namespace duckdb {

//===--------------------------------------------------------------------===//
// Table Scan
//===--------------------------------------------------------------------===//
bool TableScanParallelStateNext(ClientContext &context, const FunctionData *bind_data_p,
                                LocalTableFunctionState *local_state, GlobalTableFunctionState *gstate);

struct TableScanLocalState : public LocalTableFunctionState {
	//! The current position in the scan
	TableScanState scan_state;
	//! The DataChunk containing all read columns (even filter columns that are immediately removed)
	DataChunk all_columns;
};

static storage_t GetStorageIndex(TableCatalogEntry &table, column_t column_id) {
	if (column_id == DConstants::INVALID_INDEX) {
		return column_id;
	}
	auto &col = table.GetColumn(LogicalIndex(column_id));
	return col.StorageOid();
}

struct TableScanGlobalState : public GlobalTableFunctionState {
	TableScanGlobalState(ClientContext &context, const FunctionData *bind_data_p) : row_count(0) {
		D_ASSERT(bind_data_p);
		auto &bind_data = (const TableScanBindData &)*bind_data_p;
		max_threads = bind_data.table->GetStorage().MaxThreads(context);
	}

	ParallelTableScanState state;
	mutex lock;
	idx_t max_threads;
	//! How many rows we already scanned
	atomic<idx_t> row_count;

	vector<idx_t> projection_ids;
	vector<LogicalType> scanned_types;

	idx_t MaxThreads() const override {
		return max_threads;
	}

	bool CanRemoveFilterColumns() const {
		return !projection_ids.empty();
	}
};

static unique_ptr<LocalTableFunctionState> TableScanInitLocal(ExecutionContext &context, TableFunctionInitInput &input,
                                                              GlobalTableFunctionState *gstate) {
	auto result = make_unique<TableScanLocalState>();
	auto &bind_data = (TableScanBindData &)*input.bind_data;
	vector<column_t> column_ids = input.column_ids;
	for (auto &col : column_ids) {
		auto storage_idx = GetStorageIndex(*bind_data.table, col);
		col = storage_idx;
	}
	result->scan_state.Initialize(std::move(column_ids), input.filters);
	TableScanParallelStateNext(context.client, input.bind_data, result.get(), gstate);
	if (input.CanRemoveFilterColumns()) {
		auto &tsgs = (TableScanGlobalState &)*gstate;
		result->all_columns.Initialize(context.client, tsgs.scanned_types);
	}
	return std::move(result);
}

unique_ptr<GlobalTableFunctionState> TableScanInitGlobal(ClientContext &context, TableFunctionInitInput &input) {

	D_ASSERT(input.bind_data);
	auto &bind_data = (const TableScanBindData &)*input.bind_data;
	auto result = make_unique<TableScanGlobalState>(context, input.bind_data);
	bind_data.table->GetStorage().InitializeParallelScan(context, result->state);
	if (input.CanRemoveFilterColumns()) {
		result->projection_ids = input.projection_ids;
		const auto &columns = bind_data.table->GetColumns();
		for (const auto &col_idx : input.column_ids) {
			if (col_idx == COLUMN_IDENTIFIER_ROW_ID) {
				result->scanned_types.emplace_back(LogicalType::ROW_TYPE);
			} else {
				result->scanned_types.push_back(columns.GetColumn(LogicalIndex(col_idx)).Type());
			}
		}
	}
	return std::move(result);
}

static unique_ptr<BaseStatistics> TableScanStatistics(ClientContext &context, const FunctionData *bind_data_p,
                                                      column_t column_id) {
	auto &bind_data = (const TableScanBindData &)*bind_data_p;
	auto &local_storage = LocalStorage::Get(context, *bind_data.table->catalog);
	if (local_storage.Find(bind_data.table->GetStoragePtr())) {
		// we don't emit any statistics for tables that have outstanding transaction-local data
		return nullptr;
	}
	return bind_data.table->GetStatistics(context, column_id);
}

static void TableScanFunc(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = (TableScanBindData &)*data_p.bind_data;
	auto &gstate = (TableScanGlobalState &)*data_p.global_state;
	auto &state = (TableScanLocalState &)*data_p.local_state;
	auto &transaction = DuckTransaction::Get(context, *bind_data.table->catalog);
	auto &storage = bind_data.table->GetStorage();
	do {
		if (bind_data.is_create_index) {
			storage.CreateIndexScan(state.scan_state, output,
			                        TableScanType::TABLE_SCAN_COMMITTED_ROWS_OMIT_PERMANENTLY_DELETED);
		} else if (gstate.CanRemoveFilterColumns()) {
			state.all_columns.Reset();
			storage.Scan(transaction, state.all_columns, state.scan_state);
			output.ReferenceColumns(state.all_columns, gstate.projection_ids);
		} else {
			storage.Scan(transaction, output, state.scan_state);
		}
		if (output.size() > 0) {
			gstate.row_count += output.size();
			return;
		}
		if (!TableScanParallelStateNext(context, data_p.bind_data, data_p.local_state, data_p.global_state)) {
			return;
		}
	} while (true);
}

bool TableScanParallelStateNext(ClientContext &context, const FunctionData *bind_data_p,
                                LocalTableFunctionState *local_state, GlobalTableFunctionState *global_state) {
	auto &bind_data = (const TableScanBindData &)*bind_data_p;
	auto &parallel_state = (TableScanGlobalState &)*global_state;
	auto &state = (TableScanLocalState &)*local_state;
	auto &storage = bind_data.table->GetStorage();

	lock_guard<mutex> parallel_lock(parallel_state.lock);
	return storage.NextParallelScan(context, parallel_state.state, state.scan_state);
}

double TableScanProgress(ClientContext &context, const FunctionData *bind_data_p,
                         const GlobalTableFunctionState *gstate_p) {
	auto &bind_data = (TableScanBindData &)*bind_data_p;
	auto &gstate = (TableScanGlobalState &)*gstate_p;
	auto &storage = bind_data.table->GetStorage();
	idx_t total_rows = storage.GetTotalRows();
	if (total_rows == 0) {
		//! Table is either empty or smaller than a vector size, so it is finished
		return 100;
	}
	auto percentage = 100 * (double(gstate.row_count) / total_rows);
	if (percentage > 100) {
		//! In case the last chunk has less elements than STANDARD_VECTOR_SIZE, if our percentage is over 100
		//! It means we finished this table.
		return 100;
	}
	return percentage;
}

idx_t TableScanGetBatchIndex(ClientContext &context, const FunctionData *bind_data_p,
                             LocalTableFunctionState *local_state, GlobalTableFunctionState *global_state) {
	auto &state = (TableScanLocalState &)*local_state;
	if (state.scan_state.table_state.row_group_state.row_group) {
		return state.scan_state.table_state.batch_index;
	}
	if (state.scan_state.local_state.row_group_state.row_group) {
		return state.scan_state.table_state.batch_index + state.scan_state.local_state.batch_index;
	}
	return 0;
}

BindInfo TableScanGetBindInfo(const FunctionData *bind_data) {
	return BindInfo(ScanType::TABLE);
}

void TableScanDependency(DependencyList &entries, const FunctionData *bind_data_p) {
	auto &bind_data = (const TableScanBindData &)*bind_data_p;
	entries.AddDependency(bind_data.table);
}

unique_ptr<NodeStatistics> TableScanCardinality(ClientContext &context, const FunctionData *bind_data_p) {
	auto &bind_data = (const TableScanBindData &)*bind_data_p;
	auto &local_storage = LocalStorage::Get(context, *bind_data.table->catalog);
	auto &storage = bind_data.table->GetStorage();
	idx_t estimated_cardinality = storage.info->cardinality + local_storage.AddedRows(bind_data.table->GetStoragePtr());
	return make_unique<NodeStatistics>(storage.info->cardinality, estimated_cardinality);
}

//===--------------------------------------------------------------------===//
// Index Scan
//===--------------------------------------------------------------------===//
struct IndexScanGlobalState : public GlobalTableFunctionState {
	explicit IndexScanGlobalState(data_ptr_t row_id_data) : row_ids(LogicalType::ROW_TYPE, row_id_data) {
	}

	Vector row_ids;
	ColumnFetchState fetch_state;
	TableScanState local_storage_state;
	vector<column_t> column_ids;
	bool finished;
};

static unique_ptr<GlobalTableFunctionState> IndexScanInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind_data = (const TableScanBindData &)*input.bind_data;
	data_ptr_t row_id_data = nullptr;
	if (!bind_data.result_ids.empty()) {
		row_id_data = (data_ptr_t)&bind_data.result_ids[0];
	}
	auto result = make_unique<IndexScanGlobalState>(row_id_data);
	auto &local_storage = LocalStorage::Get(context, *bind_data.table->catalog);
	result->column_ids = input.column_ids;
	result->local_storage_state.Initialize(input.column_ids, input.filters);
	local_storage.InitializeScan(bind_data.table->GetStoragePtr(), result->local_storage_state.local_state,
	                             input.filters);

	result->finished = false;
	return std::move(result);
}

static void IndexScanFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &bind_data = (const TableScanBindData &)*data_p.bind_data;
	auto &state = (IndexScanGlobalState &)*data_p.global_state;
	auto &transaction = DuckTransaction::Get(context, *bind_data.table->catalog);
	auto &local_storage = LocalStorage::Get(transaction);

	if (!state.finished) {
		bind_data.table->GetStorage().Fetch(transaction, output, state.column_ids, state.row_ids,
		                                    bind_data.result_ids.size(), state.fetch_state);
		state.finished = true;
	}
	if (output.size() == 0) {
		local_storage.Scan(state.local_storage_state.local_state, state.column_ids, output);
	}
}

static void RewriteIndexExpression(Index &index, LogicalGet &get, Expression &expr, bool &rewrite_possible) {
	if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
		auto &bound_colref = (BoundColumnRefExpression &)expr;
		// bound column ref: rewrite to fit in the current set of bound column ids
		bound_colref.binding.table_index = get.table_index;
		column_t referenced_column = index.column_ids[bound_colref.binding.column_index];
		// search for the referenced column in the set of column_ids
		for (idx_t i = 0; i < get.column_ids.size(); i++) {
			if (get.column_ids[i] == referenced_column) {
				bound_colref.binding.column_index = i;
				return;
			}
		}
		// column id not found in bound columns in the LogicalGet: rewrite not possible
		rewrite_possible = false;
	}
	ExpressionIterator::EnumerateChildren(
	    expr, [&](Expression &child) { RewriteIndexExpression(index, get, child, rewrite_possible); });
}

void TableScanPushdownComplexFilter(ClientContext &context, LogicalGet &get, FunctionData *bind_data_p,
                                    vector<unique_ptr<Expression>> &filters) {
	auto &bind_data = (TableScanBindData &)*bind_data_p;
	auto table = bind_data.table;
	auto &storage = table->GetStorage();

	auto &config = ClientConfig::GetConfig(context);
	if (!config.enable_optimizer) {
		// we only push index scans if the optimizer is enabled
		return;
	}
	if (bind_data.is_index_scan) {
		return;
	}
	if (filters.empty()) {
		// no indexes or no filters: skip the pushdown
		return;
	}
	// behold
	storage.info->indexes.Scan([&](Index &index) {
		// first rewrite the index expression so the ColumnBindings align with the column bindings of the current table
		if (index.unbound_expressions.size() > 1) {
			return false;
		}
		auto index_expression = index.unbound_expressions[0]->Copy();
		bool rewrite_possible = true;
		RewriteIndexExpression(index, get, *index_expression, rewrite_possible);
		if (!rewrite_possible) {
			// could not rewrite!
			return false;
		}

		Value low_value, high_value, equal_value;
		ExpressionType low_comparison_type = ExpressionType::INVALID, high_comparison_type = ExpressionType::INVALID;
		// try to find a matching index for any of the filter expressions
		for (auto &filter : filters) {
			auto expr = filter.get();

			// create a matcher for a comparison with a constant
			ComparisonExpressionMatcher matcher;
			// match on a comparison type
			matcher.expr_type = make_unique<ComparisonExpressionTypeMatcher>();
			// match on a constant comparison with the indexed expression
			matcher.matchers.push_back(make_unique<ExpressionEqualityMatcher>(index_expression.get()));
			matcher.matchers.push_back(make_unique<ConstantExpressionMatcher>());

			matcher.policy = SetMatcher::Policy::UNORDERED;

			vector<Expression *> bindings;
			if (matcher.Match(expr, bindings)) {
				// range or equality comparison with constant value
				// we can use our index here
				// bindings[0] = the expression
				// bindings[1] = the index expression
				// bindings[2] = the constant
				auto comparison = (BoundComparisonExpression *)bindings[0];
				D_ASSERT(bindings[0]->GetExpressionClass() == ExpressionClass::BOUND_COMPARISON);
				D_ASSERT(bindings[2]->type == ExpressionType::VALUE_CONSTANT);

				auto constant_value = ((BoundConstantExpression *)bindings[2])->value;
				auto comparison_type = comparison->type;
				if (comparison->left->type == ExpressionType::VALUE_CONSTANT) {
					// the expression is on the right side, we flip them around
					comparison_type = FlipComparisionExpression(comparison_type);
				}
				if (comparison_type == ExpressionType::COMPARE_EQUAL) {
					// equality value
					// equality overrides any other bounds so we just break here
					equal_value = constant_value;
					break;
				} else if (comparison_type == ExpressionType::COMPARE_GREATERTHANOREQUALTO ||
				           comparison_type == ExpressionType::COMPARE_GREATERTHAN) {
					// greater than means this is a lower bound
					low_value = constant_value;
					low_comparison_type = comparison_type;
				} else {
					// smaller than means this is an upper bound
					high_value = constant_value;
					high_comparison_type = comparison_type;
				}
			} else if (expr->type == ExpressionType::COMPARE_BETWEEN) {
				// BETWEEN expression
				auto &between = (BoundBetweenExpression &)*expr;
				if (!between.input->Equals(index_expression.get())) {
					// expression doesn't match the current index expression
					continue;
				}
				if (between.lower->type != ExpressionType::VALUE_CONSTANT ||
				    between.upper->type != ExpressionType::VALUE_CONSTANT) {
					// not a constant comparison
					continue;
				}
				low_value = ((BoundConstantExpression &)*between.lower).value;
				low_comparison_type = between.lower_inclusive ? ExpressionType::COMPARE_GREATERTHANOREQUALTO
				                                              : ExpressionType::COMPARE_GREATERTHAN;
				high_value = ((BoundConstantExpression &)*between.upper).value;
				high_comparison_type = between.upper_inclusive ? ExpressionType::COMPARE_LESSTHANOREQUALTO
				                                               : ExpressionType::COMPARE_LESSTHAN;
				break;
			}
		}
		if (!equal_value.IsNull() || !low_value.IsNull() || !high_value.IsNull()) {
			// we can scan this index using this predicate: try a scan
			auto &transaction = Transaction::Get(context, *bind_data.table->catalog);
			unique_ptr<IndexScanState> index_state;
			if (!equal_value.IsNull()) {
				// equality predicate
				index_state =
				    index.InitializeScanSinglePredicate(transaction, equal_value, ExpressionType::COMPARE_EQUAL);
			} else if (!low_value.IsNull() && !high_value.IsNull()) {
				// two-sided predicate
				index_state = index.InitializeScanTwoPredicates(transaction, low_value, low_comparison_type, high_value,
				                                                high_comparison_type);
			} else if (!low_value.IsNull()) {
				// less than predicate
				index_state = index.InitializeScanSinglePredicate(transaction, low_value, low_comparison_type);
			} else {
				D_ASSERT(!high_value.IsNull());
				index_state = index.InitializeScanSinglePredicate(transaction, high_value, high_comparison_type);
			}
			if (index.Scan(transaction, storage, *index_state, STANDARD_VECTOR_SIZE, bind_data.result_ids)) {
				// use an index scan!
				bind_data.is_index_scan = true;
				get.function = TableScanFunction::GetIndexScanFunction();
			} else {
				bind_data.result_ids.clear();
			}
			return true;
		}
		return false;
	});
}

string TableScanToString(const FunctionData *bind_data_p) {
	auto &bind_data = (const TableScanBindData &)*bind_data_p;
	string result = bind_data.table->name;
	return result;
}

static void TableScanSerialize(FieldWriter &writer, const FunctionData *bind_data_p, const TableFunction &function) {
	auto &bind_data = (TableScanBindData &)*bind_data_p;

	writer.WriteString(bind_data.table->schema->name);
	writer.WriteString(bind_data.table->name);
	writer.WriteField<bool>(bind_data.is_index_scan);
	writer.WriteField<bool>(bind_data.is_create_index);
	writer.WriteList<row_t>(bind_data.result_ids);
	writer.WriteString(bind_data.table->schema->catalog->GetName());
}

static unique_ptr<FunctionData> TableScanDeserialize(ClientContext &context, FieldReader &reader,
                                                     TableFunction &function) {
	auto schema_name = reader.ReadRequired<string>();
	auto table_name = reader.ReadRequired<string>();
	auto is_index_scan = reader.ReadRequired<bool>();
	auto is_create_index = reader.ReadRequired<bool>();
	auto result_ids = reader.ReadRequiredList<row_t>();
	auto catalog_name = reader.ReadField<string>(INVALID_CATALOG);

	auto catalog_entry = Catalog::GetEntry<TableCatalogEntry>(context, catalog_name, schema_name, table_name);
	if (!catalog_entry || catalog_entry->type != CatalogType::TABLE_ENTRY) {
		throw SerializationException("Cant find table for %s.%s", schema_name, table_name);
	}

	auto result = make_unique<TableScanBindData>((DuckTableEntry *)catalog_entry);
	result->is_index_scan = is_index_scan;
	result->is_create_index = is_create_index;
	result->result_ids = std::move(result_ids);
	return std::move(result);
}

TableFunction TableScanFunction::GetIndexScanFunction() {
	TableFunction scan_function("index_scan", {}, IndexScanFunction);
	scan_function.init_local = nullptr;
	scan_function.init_global = IndexScanInitGlobal;
	scan_function.statistics = TableScanStatistics;
	scan_function.dependency = TableScanDependency;
	scan_function.cardinality = TableScanCardinality;
	scan_function.pushdown_complex_filter = nullptr;
	scan_function.to_string = TableScanToString;
	scan_function.table_scan_progress = nullptr;
	scan_function.get_batch_index = nullptr;
	scan_function.projection_pushdown = true;
	scan_function.filter_pushdown = false;
	scan_function.serialize = TableScanSerialize;
	scan_function.deserialize = TableScanDeserialize;
	return scan_function;
}

TableFunction TableScanFunction::GetFunction() {
	TableFunction scan_function("seq_scan", {}, TableScanFunc);
	scan_function.init_local = TableScanInitLocal;
	scan_function.init_global = TableScanInitGlobal;
	scan_function.statistics = TableScanStatistics;
	scan_function.dependency = TableScanDependency;
	scan_function.cardinality = TableScanCardinality;
	scan_function.pushdown_complex_filter = TableScanPushdownComplexFilter;
	scan_function.to_string = TableScanToString;
	scan_function.table_scan_progress = TableScanProgress;
	scan_function.get_batch_index = TableScanGetBatchIndex;
	scan_function.get_batch_info = TableScanGetBindInfo;
	scan_function.projection_pushdown = true;
	scan_function.filter_pushdown = true;
	scan_function.filter_prune = true;
	scan_function.serialize = TableScanSerialize;
	scan_function.deserialize = TableScanDeserialize;
	return scan_function;
}

TableCatalogEntry *TableScanFunction::GetTableEntry(const TableFunction &function, const FunctionData *bind_data_p) {
	if (function.function != TableScanFunc || !bind_data_p) {
		return nullptr;
	}
	auto &bind_data = (TableScanBindData &)*bind_data_p;
	return bind_data.table;
}

void TableScanFunction::RegisterFunction(BuiltinFunctions &set) {
	TableFunctionSet table_scan_set("seq_scan");
	table_scan_set.AddFunction(GetFunction());
	set.AddFunction(std::move(table_scan_set));

	set.AddFunction(GetIndexScanFunction());
}

void BuiltinFunctions::RegisterTableScanFunctions() {
	TableScanFunction::RegisterFunction(*this);
}

} // namespace duckdb






namespace duckdb {

struct UnnestBindData : public FunctionData {
	explicit UnnestBindData(LogicalType input_type_p) : input_type(std::move(input_type_p)) {
	}

	LogicalType input_type;

public:
	unique_ptr<FunctionData> Copy() const override {
		return make_unique<UnnestBindData>(input_type);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = (const UnnestBindData &)other_p;
		return input_type == other.input_type;
	}
};

struct UnnestGlobalState : public GlobalTableFunctionState {
	UnnestGlobalState() {
	}

	vector<unique_ptr<Expression>> select_list;

	idx_t MaxThreads() const override {
		return GlobalTableFunctionState::MAX_THREADS;
	}
};

struct UnnestLocalState : public LocalTableFunctionState {
	UnnestLocalState() {
	}

	unique_ptr<OperatorState> operator_state;
};

static unique_ptr<FunctionData> UnnestBind(ClientContext &context, TableFunctionBindInput &input,
                                           vector<LogicalType> &return_types, vector<string> &names) {
	if (input.input_table_types.size() != 1 || input.input_table_types[0].id() != LogicalTypeId::LIST) {
		throw BinderException("UNNEST requires a single list as input");
	}
	return_types.push_back(ListType::GetChildType(input.input_table_types[0]));
	names.push_back(input.input_table_names[0]);
	return make_unique<UnnestBindData>(input.input_table_types[0]);
}

static unique_ptr<LocalTableFunctionState> UnnestLocalInit(ExecutionContext &context, TableFunctionInitInput &input,
                                                           GlobalTableFunctionState *global_state) {
	auto &gstate = (UnnestGlobalState &)*global_state;

	auto result = make_unique<UnnestLocalState>();
	result->operator_state = PhysicalUnnest::GetState(context, gstate.select_list);
	return std::move(result);
}

static unique_ptr<GlobalTableFunctionState> UnnestInit(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind_data = (UnnestBindData &)*input.bind_data;
	auto result = make_unique<UnnestGlobalState>();
	auto ref = make_unique<BoundReferenceExpression>(bind_data.input_type, 0);
	auto bound_unnest = make_unique<BoundUnnestExpression>(ListType::GetChildType(bind_data.input_type));
	bound_unnest->child = std::move(ref);
	result->select_list.push_back(std::move(bound_unnest));
	return std::move(result);
}

static OperatorResultType UnnestFunction(ExecutionContext &context, TableFunctionInput &data_p, DataChunk &input,
                                         DataChunk &output) {
	auto &state = (UnnestGlobalState &)*data_p.global_state;
	auto &lstate = (UnnestLocalState &)*data_p.local_state;
	return PhysicalUnnest::ExecuteInternal(context, input, output, *lstate.operator_state, state.select_list, false);
}

void UnnestTableFunction::RegisterFunction(BuiltinFunctions &set) {
	TableFunction unnest_function("unnest", {LogicalTypeId::TABLE}, nullptr, UnnestBind, UnnestInit, UnnestLocalInit);
	unnest_function.in_out_function = UnnestFunction;
	set.AddFunction(unnest_function);
}

} // namespace duckdb



#include <cstdint>

namespace duckdb {

struct PragmaVersionData : public GlobalTableFunctionState {
	PragmaVersionData() : finished(false) {
	}

	bool finished;
};

static unique_ptr<FunctionData> PragmaVersionBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("library_version");
	return_types.emplace_back(LogicalType::VARCHAR);
	names.emplace_back("source_id");
	return_types.emplace_back(LogicalType::VARCHAR);
	return nullptr;
}

static unique_ptr<GlobalTableFunctionState> PragmaVersionInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_unique<PragmaVersionData>();
}

static void PragmaVersionFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = (PragmaVersionData &)*data_p.global_state;
	if (data.finished) {
		// finished returning values
		return;
	}
	output.SetCardinality(1);
	output.SetValue(0, 0, DuckDB::LibraryVersion());
	output.SetValue(1, 0, DuckDB::SourceID());
	data.finished = true;
}

void PragmaVersion::RegisterFunction(BuiltinFunctions &set) {
	TableFunction pragma_version("pragma_version", {}, PragmaVersionFunction);
	pragma_version.bind = PragmaVersionBind;
	pragma_version.init_global = PragmaVersionInit;
	set.AddFunction(pragma_version);
}

idx_t DuckDB::StandardVectorSize() {
	return STANDARD_VECTOR_SIZE;
}

const char *DuckDB::SourceID() {
	return DUCKDB_SOURCE_ID;
}

const char *DuckDB::LibraryVersion() {
	return DUCKDB_VERSION;
}

string DuckDB::Platform() {
	string os = "linux";
#if INTPTR_MAX == INT64_MAX
	string arch = "amd64";
#elif INTPTR_MAX == INT32_MAX
	string arch = "i686";
#else
#error Unknown pointer size or missing size macros!
#endif
	string postfix = "";

#ifdef _WIN32
	os = "windows";
#elif defined(__APPLE__)
	os = "osx";
#endif
#if defined(__aarch64__) || defined(__ARM_ARCH_ISA_A64)
	arch = "arm64";
#endif

#if !defined(_GLIBCXX_USE_CXX11_ABI) || _GLIBCXX_USE_CXX11_ABI == 0
	if (os == "linux") {
		postfix = "_gcc4";
	}
#endif

	return os + "_" + arch + postfix;
}

} // namespace duckdb


namespace duckdb {

GlobalTableFunctionState::~GlobalTableFunctionState() {
}

LocalTableFunctionState::~LocalTableFunctionState() {
}

TableFunctionInfo::~TableFunctionInfo() {
}

TableFunction::TableFunction(string name, vector<LogicalType> arguments, table_function_t function,
                             table_function_bind_t bind, table_function_init_global_t init_global,
                             table_function_init_local_t init_local)
    : SimpleNamedParameterFunction(std::move(name), std::move(arguments)), bind(bind), init_global(init_global),
      init_local(init_local), function(function), in_out_function(nullptr), in_out_function_final(nullptr),
      statistics(nullptr), dependency(nullptr), cardinality(nullptr), pushdown_complex_filter(nullptr),
      to_string(nullptr), table_scan_progress(nullptr), get_batch_index(nullptr), get_batch_info(nullptr),
      serialize(nullptr), deserialize(nullptr), projection_pushdown(false), filter_pushdown(false),
      filter_prune(false) {
}

TableFunction::TableFunction(const vector<LogicalType> &arguments, table_function_t function,
                             table_function_bind_t bind, table_function_init_global_t init_global,
                             table_function_init_local_t init_local)
    : TableFunction(string(), arguments, function, bind, init_global, init_local) {
}
TableFunction::TableFunction()
    : SimpleNamedParameterFunction("", {}), bind(nullptr), init_global(nullptr), init_local(nullptr), function(nullptr),
      in_out_function(nullptr), statistics(nullptr), dependency(nullptr), cardinality(nullptr),
      pushdown_complex_filter(nullptr), to_string(nullptr), table_scan_progress(nullptr), get_batch_index(nullptr),
      get_batch_info(nullptr), serialize(nullptr), deserialize(nullptr), projection_pushdown(false),
      filter_pushdown(false), filter_prune(false) {
}

} // namespace duckdb
//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/table_macro_function.hpp
//
//
//===----------------------------------------------------------------------===//
//! The SelectStatement of the view





namespace duckdb {

TableMacroFunction::TableMacroFunction(unique_ptr<QueryNode> query_node)
    : MacroFunction(MacroType::TABLE_MACRO), query_node(std::move(query_node)) {
}

TableMacroFunction::TableMacroFunction(void) : MacroFunction(MacroType::TABLE_MACRO) {
}

unique_ptr<MacroFunction> TableMacroFunction::Copy() {
	auto result = make_unique<TableMacroFunction>();
	result->query_node = query_node->Copy();
	this->CopyProperties(*result);
	return std::move(result);
}

string TableMacroFunction::ToSQL(const string &schema, const string &name) {
	return MacroFunction::ToSQL(schema, name) + StringUtil::Format("TABLE (%s);", query_node->ToString());
}

} // namespace duckdb







namespace duckdb {

void UDFWrapper::RegisterFunction(string name, vector<LogicalType> args, LogicalType ret_type,
                                  scalar_function_t udf_function, ClientContext &context, LogicalType varargs) {

	ScalarFunction scalar_function(std::move(name), std::move(args), std::move(ret_type), std::move(udf_function));
	scalar_function.varargs = std::move(varargs);
	scalar_function.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	CreateScalarFunctionInfo info(scalar_function);
	info.schema = DEFAULT_SCHEMA;
	context.RegisterFunction(&info);
}

void UDFWrapper::RegisterAggrFunction(AggregateFunction aggr_function, ClientContext &context, LogicalType varargs) {
	aggr_function.varargs = std::move(varargs);
	CreateAggregateFunctionInfo info(std::move(aggr_function));
	context.RegisterFunction(&info);
}

} // namespace duckdb















namespace duckdb {

BaseAppender::BaseAppender(Allocator &allocator, AppenderType type_p)
    : allocator(allocator), column(0), appender_type(type_p) {
}

BaseAppender::BaseAppender(Allocator &allocator_p, vector<LogicalType> types_p, AppenderType type_p)
    : allocator(allocator_p), types(std::move(types_p)),
      collection(make_unique<ColumnDataCollection>(allocator, types)), column(0), appender_type(type_p) {
	InitializeChunk();
}

BaseAppender::~BaseAppender() {
}

void BaseAppender::Destructor() {
	if (Exception::UncaughtException()) {
		return;
	}
	// flush any remaining chunks, but only if we are not cleaning up the appender as part of an exception stack unwind
	// wrapped in a try/catch because Close() can throw if the table was dropped in the meantime
	try {
		Close();
	} catch (...) {
	}
}

InternalAppender::InternalAppender(ClientContext &context_p, TableCatalogEntry &table_p)
    : BaseAppender(Allocator::DefaultAllocator(), table_p.GetTypes(), AppenderType::PHYSICAL), context(context_p),
      table(table_p) {
}

InternalAppender::~InternalAppender() {
	Destructor();
}

Appender::Appender(Connection &con, const string &schema_name, const string &table_name)
    : BaseAppender(Allocator::DefaultAllocator(), AppenderType::LOGICAL), context(con.context) {
	description = con.TableInfo(schema_name, table_name);
	if (!description) {
		// table could not be found
		throw CatalogException(StringUtil::Format("Table \"%s.%s\" could not be found", schema_name, table_name));
	}
	for (auto &column : description->columns) {
		types.push_back(column.Type());
	}
	InitializeChunk();
	collection = make_unique<ColumnDataCollection>(allocator, types);
}

Appender::Appender(Connection &con, const string &table_name) : Appender(con, DEFAULT_SCHEMA, table_name) {
}

Appender::~Appender() {
	Destructor();
}

void BaseAppender::InitializeChunk() {
	chunk.Initialize(allocator, types);
}

void BaseAppender::BeginRow() {
}

void BaseAppender::EndRow() {
	// check that all rows have been appended to
	if (column != chunk.ColumnCount()) {
		throw InvalidInputException("Call to EndRow before all rows have been appended to!");
	}
	column = 0;
	chunk.SetCardinality(chunk.size() + 1);
	if (chunk.size() >= STANDARD_VECTOR_SIZE) {
		FlushChunk();
	}
}

template <class SRC, class DST>
void BaseAppender::AppendValueInternal(Vector &col, SRC input) {
	FlatVector::GetData<DST>(col)[chunk.size()] = Cast::Operation<SRC, DST>(input);
}

template <class SRC, class DST>
void BaseAppender::AppendDecimalValueInternal(Vector &col, SRC input) {
	switch (appender_type) {
	case AppenderType::LOGICAL: {
		auto &type = col.GetType();
		D_ASSERT(type.id() == LogicalTypeId::DECIMAL);
		auto width = DecimalType::GetWidth(type);
		auto scale = DecimalType::GetScale(type);
		TryCastToDecimal::Operation<SRC, DST>(input, FlatVector::GetData<DST>(col)[chunk.size()], nullptr, width,
		                                      scale);
		return;
	}
	case AppenderType::PHYSICAL: {
		AppendValueInternal<SRC, DST>(col, input);
		return;
	}
	default:
		throw InternalException("Type not implemented for AppenderType");
	}
}

template <class T>
void BaseAppender::AppendValueInternal(T input) {
	if (column >= types.size()) {
		throw InvalidInputException("Too many appends for chunk!");
	}
	auto &col = chunk.data[column];
	switch (col.GetType().id()) {
	case LogicalTypeId::BOOLEAN:
		AppendValueInternal<T, bool>(col, input);
		break;
	case LogicalTypeId::UTINYINT:
		AppendValueInternal<T, uint8_t>(col, input);
		break;
	case LogicalTypeId::TINYINT:
		AppendValueInternal<T, int8_t>(col, input);
		break;
	case LogicalTypeId::USMALLINT:
		AppendValueInternal<T, uint16_t>(col, input);
		break;
	case LogicalTypeId::SMALLINT:
		AppendValueInternal<T, int16_t>(col, input);
		break;
	case LogicalTypeId::UINTEGER:
		AppendValueInternal<T, uint32_t>(col, input);
		break;
	case LogicalTypeId::INTEGER:
		AppendValueInternal<T, int32_t>(col, input);
		break;
	case LogicalTypeId::UBIGINT:
		AppendValueInternal<T, uint64_t>(col, input);
		break;
	case LogicalTypeId::BIGINT:
		AppendValueInternal<T, int64_t>(col, input);
		break;
	case LogicalTypeId::HUGEINT:
		AppendValueInternal<T, hugeint_t>(col, input);
		break;
	case LogicalTypeId::FLOAT:
		AppendValueInternal<T, float>(col, input);
		break;
	case LogicalTypeId::DOUBLE:
		AppendValueInternal<T, double>(col, input);
		break;
	case LogicalTypeId::DECIMAL:
		switch (col.GetType().InternalType()) {
		case PhysicalType::INT16:
			AppendDecimalValueInternal<T, int16_t>(col, input);
			break;
		case PhysicalType::INT32:
			AppendDecimalValueInternal<T, int32_t>(col, input);
			break;
		case PhysicalType::INT64:
			AppendDecimalValueInternal<T, int64_t>(col, input);
			break;
		case PhysicalType::INT128:
			AppendDecimalValueInternal<T, hugeint_t>(col, input);
			break;
		default:
			throw InternalException("Internal type not recognized for Decimal");
		}
		break;
	case LogicalTypeId::DATE:
		AppendValueInternal<T, date_t>(col, input);
		break;
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ:
		AppendValueInternal<T, timestamp_t>(col, input);
		break;
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_TZ:
		AppendValueInternal<T, dtime_t>(col, input);
		break;
	case LogicalTypeId::INTERVAL:
		AppendValueInternal<T, interval_t>(col, input);
		break;
	case LogicalTypeId::VARCHAR:
		FlatVector::GetData<string_t>(col)[chunk.size()] = StringCast::Operation<T>(input, col);
		break;
	default:
		AppendValue(Value::CreateValue<T>(input));
		return;
	}
	column++;
}

template <>
void BaseAppender::Append(bool value) {
	AppendValueInternal<bool>(value);
}

template <>
void BaseAppender::Append(int8_t value) {
	AppendValueInternal<int8_t>(value);
}

template <>
void BaseAppender::Append(int16_t value) {
	AppendValueInternal<int16_t>(value);
}

template <>
void BaseAppender::Append(int32_t value) {
	AppendValueInternal<int32_t>(value);
}

template <>
void BaseAppender::Append(int64_t value) {
	AppendValueInternal<int64_t>(value);
}

template <>
void BaseAppender::Append(hugeint_t value) {
	AppendValueInternal<hugeint_t>(value);
}

template <>
void BaseAppender::Append(uint8_t value) {
	AppendValueInternal<uint8_t>(value);
}

template <>
void BaseAppender::Append(uint16_t value) {
	AppendValueInternal<uint16_t>(value);
}

template <>
void BaseAppender::Append(uint32_t value) {
	AppendValueInternal<uint32_t>(value);
}

template <>
void BaseAppender::Append(uint64_t value) {
	AppendValueInternal<uint64_t>(value);
}

template <>
void BaseAppender::Append(const char *value) {
	AppendValueInternal<string_t>(string_t(value));
}

void BaseAppender::Append(const char *value, uint32_t length) {
	AppendValueInternal<string_t>(string_t(value, length));
}

template <>
void BaseAppender::Append(string_t value) {
	AppendValueInternal<string_t>(value);
}

template <>
void BaseAppender::Append(float value) {
	AppendValueInternal<float>(value);
}

template <>
void BaseAppender::Append(double value) {
	AppendValueInternal<double>(value);
}

template <>
void BaseAppender::Append(date_t value) {
	AppendValueInternal<date_t>(value);
}

template <>
void BaseAppender::Append(dtime_t value) {
	AppendValueInternal<dtime_t>(value);
}

template <>
void BaseAppender::Append(timestamp_t value) {
	AppendValueInternal<timestamp_t>(value);
}

template <>
void BaseAppender::Append(interval_t value) {
	AppendValueInternal<interval_t>(value);
}

template <>
void BaseAppender::Append(Value value) { // NOLINT: template shtuff
	if (column >= chunk.ColumnCount()) {
		throw InvalidInputException("Too many appends for chunk!");
	}
	AppendValue(value);
}

template <>
void BaseAppender::Append(std::nullptr_t value) {
	if (column >= chunk.ColumnCount()) {
		throw InvalidInputException("Too many appends for chunk!");
	}
	auto &col = chunk.data[column++];
	FlatVector::SetNull(col, chunk.size(), true);
}

void BaseAppender::AppendValue(const Value &value) {
	chunk.SetValue(column, chunk.size(), value);
	column++;
}

void BaseAppender::AppendDataChunk(DataChunk &chunk) {
	if (chunk.GetTypes() != types) {
		throw InvalidInputException("Type mismatch in Append DataChunk and the types required for appender");
	}
	collection->Append(chunk);
	if (collection->Count() >= FLUSH_COUNT) {
		Flush();
	}
}

void BaseAppender::FlushChunk() {
	if (chunk.size() == 0) {
		return;
	}
	collection->Append(chunk);
	chunk.Reset();
	if (collection->Count() >= FLUSH_COUNT) {
		Flush();
	}
}

void BaseAppender::Flush() {
	// check that all vectors have the same length before appending
	if (column != 0) {
		throw InvalidInputException("Failed to Flush appender: incomplete append to row!");
	}

	FlushChunk();
	if (collection->Count() == 0) {
		return;
	}
	FlushInternal(*collection);

	collection->Reset();
	column = 0;
}

void Appender::FlushInternal(ColumnDataCollection &collection) {
	context->Append(*description, collection);
}

void InternalAppender::FlushInternal(ColumnDataCollection &collection) {
	table.GetStorage().LocalAppend(table, context, collection);
}

void BaseAppender::Close() {
	if (column == 0 || column == types.size()) {
		Flush();
	}
}

} // namespace duckdb







namespace duckdb {

AttachedDatabase::AttachedDatabase(DatabaseInstance &db, AttachedDatabaseType type)
    : CatalogEntry(CatalogType::DATABASE_ENTRY, nullptr,
                   type == AttachedDatabaseType::SYSTEM_DATABASE ? SYSTEM_CATALOG : TEMP_CATALOG),
      db(db), type(type) {
	D_ASSERT(type == AttachedDatabaseType::TEMP_DATABASE || type == AttachedDatabaseType::SYSTEM_DATABASE);
	if (type == AttachedDatabaseType::TEMP_DATABASE) {
		storage = make_unique<SingleFileStorageManager>(*this, ":memory:", false);
	}
	catalog = make_unique<DuckCatalog>(*this);
	transaction_manager = make_unique<DuckTransactionManager>(*this);
	internal = true;
}

AttachedDatabase::AttachedDatabase(DatabaseInstance &db, Catalog &catalog_p, string name_p, string file_path_p,
                                   AccessMode access_mode)
    : CatalogEntry(CatalogType::DATABASE_ENTRY, &catalog_p, std::move(name_p)), db(db),
      type(access_mode == AccessMode::READ_ONLY ? AttachedDatabaseType::READ_ONLY_DATABASE
                                                : AttachedDatabaseType::READ_WRITE_DATABASE) {
	storage =
	    make_unique<SingleFileStorageManager>(*this, std::move(file_path_p), access_mode == AccessMode::READ_ONLY);
	catalog = make_unique<DuckCatalog>(*this);
	transaction_manager = make_unique<DuckTransactionManager>(*this);
	internal = true;
}

AttachedDatabase::AttachedDatabase(DatabaseInstance &db, Catalog &catalog_p, StorageExtension &storage_extension,
                                   string name_p, AttachInfo &info, AccessMode access_mode)
    : CatalogEntry(CatalogType::DATABASE_ENTRY, &catalog_p, std::move(name_p)), db(db),
      type(access_mode == AccessMode::READ_ONLY ? AttachedDatabaseType::READ_ONLY_DATABASE
                                                : AttachedDatabaseType::READ_WRITE_DATABASE) {
	catalog = storage_extension.attach(storage_extension.storage_info.get(), *this, name, info, access_mode);
	if (!catalog) {
		throw InternalException("AttachedDatabase - attach function did not return a catalog");
	}
	transaction_manager =
	    storage_extension.create_transaction_manager(storage_extension.storage_info.get(), *this, *catalog);
	if (!transaction_manager) {
		throw InternalException(
		    "AttachedDatabase - create_transaction_manager function did not return a transaction manager");
	}
	internal = true;
}

AttachedDatabase::~AttachedDatabase() {
	if (Exception::UncaughtException()) {
		return;
	}
	if (!storage) {
		return;
	}

	// shutting down: attempt to checkpoint the database
	// but only if we are not cleaning up as part of an exception unwind
	try {
		if (!storage->InMemory()) {
			auto &config = DBConfig::GetConfig(db);
			if (!config.options.checkpoint_on_shutdown) {
				return;
			}
			storage->CreateCheckpoint(true);
		}
	} catch (...) {
	}
}

bool AttachedDatabase::IsSystem() const {
	D_ASSERT(!storage || type != AttachedDatabaseType::SYSTEM_DATABASE);
	return type == AttachedDatabaseType::SYSTEM_DATABASE;
}

bool AttachedDatabase::IsTemporary() const {
	return type == AttachedDatabaseType::TEMP_DATABASE;
}
bool AttachedDatabase::IsReadOnly() const {
	return type == AttachedDatabaseType::READ_ONLY_DATABASE;
}

string AttachedDatabase::ExtractDatabaseName(const string &dbpath) {
	if (dbpath.empty() || dbpath == ":memory:") {
		return "memory";
	}
	return FileSystem::ExtractBaseName(dbpath);
}

void AttachedDatabase::Initialize() {
	if (IsSystem()) {
		catalog->Initialize(true);
	} else {
		catalog->Initialize(false);
	}
	if (storage) {
		storage->Initialize();
	}
}

StorageManager &AttachedDatabase::GetStorageManager() {
	if (!storage) {
		throw InternalException("Internal system catalog does not have storage");
	}
	return *storage;
}

Catalog &AttachedDatabase::GetCatalog() {
	return *catalog;
}

TransactionManager &AttachedDatabase::GetTransactionManager() {
	return *transaction_manager;
}

} // namespace duckdb


using duckdb::Appender;
using duckdb::AppenderWrapper;
using duckdb::Connection;
using duckdb::date_t;
using duckdb::dtime_t;
using duckdb::hugeint_t;
using duckdb::interval_t;
using duckdb::string_t;
using duckdb::timestamp_t;

duckdb_state duckdb_appender_create(duckdb_connection connection, const char *schema, const char *table,
                                    duckdb_appender *out_appender) {
	Connection *conn = (Connection *)connection;

	if (!connection || !table || !out_appender) {
		return DuckDBError;
	}
	if (schema == nullptr) {
		schema = DEFAULT_SCHEMA;
	}
	auto wrapper = new AppenderWrapper();
	*out_appender = (duckdb_appender)wrapper;
	try {
		wrapper->appender = duckdb::make_unique<Appender>(*conn, schema, table);
	} catch (std::exception &ex) {
		wrapper->error = ex.what();
		return DuckDBError;
	} catch (...) { // LCOV_EXCL_START
		wrapper->error = "Unknown create appender error";
		return DuckDBError;
	} // LCOV_EXCL_STOP
	return DuckDBSuccess;
}

duckdb_state duckdb_appender_destroy(duckdb_appender *appender) {
	if (!appender || !*appender) {
		return DuckDBError;
	}
	duckdb_appender_close(*appender);
	auto wrapper = (AppenderWrapper *)*appender;
	if (wrapper) {
		delete wrapper;
	}
	*appender = nullptr;
	return DuckDBSuccess;
}

template <class FUN>
duckdb_state duckdb_appender_run_function(duckdb_appender appender, FUN &&function) {
	if (!appender) {
		return DuckDBError;
	}
	auto wrapper = (AppenderWrapper *)appender;
	if (!wrapper->appender) {
		return DuckDBError;
	}
	try {
		function(*wrapper->appender);
	} catch (std::exception &ex) {
		wrapper->error = ex.what();
		return DuckDBError;
	} catch (...) { // LCOV_EXCL_START
		wrapper->error = "Unknown error";
		return DuckDBError;
	} // LCOV_EXCL_STOP
	return DuckDBSuccess;
}

const char *duckdb_appender_error(duckdb_appender appender) {
	if (!appender) {
		return nullptr;
	}
	auto wrapper = (AppenderWrapper *)appender;
	if (wrapper->error.empty()) {
		return nullptr;
	}
	return wrapper->error.c_str();
}

duckdb_state duckdb_appender_begin_row(duckdb_appender appender) {
	return DuckDBSuccess;
}

duckdb_state duckdb_appender_end_row(duckdb_appender appender) {
	return duckdb_appender_run_function(appender, [&](Appender &appender) { appender.EndRow(); });
}

template <class T>
duckdb_state duckdb_append_internal(duckdb_appender appender, T value) {
	if (!appender) {
		return DuckDBError;
	}
	auto *appender_instance = (AppenderWrapper *)appender;
	try {
		appender_instance->appender->Append<T>(value);
	} catch (std::exception &ex) {
		appender_instance->error = ex.what();
		return DuckDBError;
	} catch (...) {
		return DuckDBError;
	}
	return DuckDBSuccess;
}

duckdb_state duckdb_append_bool(duckdb_appender appender, bool value) {
	return duckdb_append_internal<bool>(appender, value);
}

duckdb_state duckdb_append_int8(duckdb_appender appender, int8_t value) {
	return duckdb_append_internal<int8_t>(appender, value);
}

duckdb_state duckdb_append_int16(duckdb_appender appender, int16_t value) {
	return duckdb_append_internal<int16_t>(appender, value);
}

duckdb_state duckdb_append_int32(duckdb_appender appender, int32_t value) {
	return duckdb_append_internal<int32_t>(appender, value);
}

duckdb_state duckdb_append_int64(duckdb_appender appender, int64_t value) {
	return duckdb_append_internal<int64_t>(appender, value);
}

duckdb_state duckdb_append_hugeint(duckdb_appender appender, duckdb_hugeint value) {
	hugeint_t internal;
	internal.lower = value.lower;
	internal.upper = value.upper;
	return duckdb_append_internal<hugeint_t>(appender, internal);
}

duckdb_state duckdb_append_uint8(duckdb_appender appender, uint8_t value) {
	return duckdb_append_internal<uint8_t>(appender, value);
}

duckdb_state duckdb_append_uint16(duckdb_appender appender, uint16_t value) {
	return duckdb_append_internal<uint16_t>(appender, value);
}

duckdb_state duckdb_append_uint32(duckdb_appender appender, uint32_t value) {
	return duckdb_append_internal<uint32_t>(appender, value);
}

duckdb_state duckdb_append_uint64(duckdb_appender appender, uint64_t value) {
	return duckdb_append_internal<uint64_t>(appender, value);
}

duckdb_state duckdb_append_float(duckdb_appender appender, float value) {
	return duckdb_append_internal<float>(appender, value);
}

duckdb_state duckdb_append_double(duckdb_appender appender, double value) {
	return duckdb_append_internal<double>(appender, value);
}

duckdb_state duckdb_append_date(duckdb_appender appender, duckdb_date value) {
	return duckdb_append_internal<date_t>(appender, date_t(value.days));
}

duckdb_state duckdb_append_time(duckdb_appender appender, duckdb_time value) {
	return duckdb_append_internal<dtime_t>(appender, dtime_t(value.micros));
}

duckdb_state duckdb_append_timestamp(duckdb_appender appender, duckdb_timestamp value) {
	return duckdb_append_internal<timestamp_t>(appender, timestamp_t(value.micros));
}

duckdb_state duckdb_append_interval(duckdb_appender appender, duckdb_interval value) {
	interval_t interval;
	interval.months = value.months;
	interval.days = value.days;
	interval.micros = value.micros;
	return duckdb_append_internal<interval_t>(appender, interval);
}

duckdb_state duckdb_append_null(duckdb_appender appender) {
	return duckdb_append_internal<std::nullptr_t>(appender, nullptr);
}

duckdb_state duckdb_append_varchar(duckdb_appender appender, const char *val) {
	return duckdb_append_internal<const char *>(appender, val);
}

duckdb_state duckdb_append_varchar_length(duckdb_appender appender, const char *val, idx_t length) {
	return duckdb_append_internal<string_t>(appender, string_t(val, length));
}
duckdb_state duckdb_append_blob(duckdb_appender appender, const void *data, idx_t length) {
	auto value = duckdb::Value::BLOB((duckdb::const_data_ptr_t)data, length);
	return duckdb_append_internal<duckdb::Value>(appender, value);
}

duckdb_state duckdb_appender_flush(duckdb_appender appender) {
	return duckdb_appender_run_function(appender, [&](Appender &appender) { appender.Flush(); });
}

duckdb_state duckdb_appender_close(duckdb_appender appender) {
	return duckdb_appender_run_function(appender, [&](Appender &appender) { appender.Close(); });
}

duckdb_state duckdb_append_data_chunk(duckdb_appender appender, duckdb_data_chunk chunk) {
	if (!chunk) {
		return DuckDBError;
	}
	auto data_chunk = (duckdb::DataChunk *)chunk;
	return duckdb_appender_run_function(appender, [&](Appender &appender) { appender.AppendDataChunk(*data_chunk); });
}



using duckdb::ArrowConverter;
using duckdb::ArrowResultWrapper;
using duckdb::Connection;
using duckdb::DataChunk;
using duckdb::LogicalType;
using duckdb::MaterializedQueryResult;
using duckdb::PreparedStatementWrapper;
using duckdb::QueryResult;
using duckdb::QueryResultType;

duckdb_state duckdb_query_arrow(duckdb_connection connection, const char *query, duckdb_arrow *out_result) {
	Connection *conn = (Connection *)connection;
	auto wrapper = new ArrowResultWrapper();
	wrapper->result = conn->Query(query);
	*out_result = (duckdb_arrow)wrapper;
	return !wrapper->result->HasError() ? DuckDBSuccess : DuckDBError;
}

duckdb_state duckdb_query_arrow_schema(duckdb_arrow result, duckdb_arrow_schema *out_schema) {
	if (!out_schema) {
		return DuckDBSuccess;
	}
	auto wrapper = (ArrowResultWrapper *)result;
	ArrowConverter::ToArrowSchema((ArrowSchema *)*out_schema, wrapper->result->types, wrapper->result->names,
	                              wrapper->timezone_config);
	return DuckDBSuccess;
}

duckdb_state duckdb_query_arrow_array(duckdb_arrow result, duckdb_arrow_array *out_array) {
	if (!out_array) {
		return DuckDBSuccess;
	}
	auto wrapper = (ArrowResultWrapper *)result;
	auto success = wrapper->result->TryFetch(wrapper->current_chunk, wrapper->result->GetErrorObject());
	if (!success) { // LCOV_EXCL_START
		return DuckDBError;
	} // LCOV_EXCL_STOP
	if (!wrapper->current_chunk || wrapper->current_chunk->size() == 0) {
		return DuckDBSuccess;
	}
	ArrowConverter::ToArrowArray(*wrapper->current_chunk, (ArrowArray *)*out_array);
	return DuckDBSuccess;
}

idx_t duckdb_arrow_row_count(duckdb_arrow result) {
	auto wrapper = (ArrowResultWrapper *)result;
	if (wrapper->result->HasError()) {
		return 0;
	}
	return wrapper->result->RowCount();
}

idx_t duckdb_arrow_column_count(duckdb_arrow result) {
	auto wrapper = (ArrowResultWrapper *)result;
	return wrapper->result->ColumnCount();
}

idx_t duckdb_arrow_rows_changed(duckdb_arrow result) {
	auto wrapper = (ArrowResultWrapper *)result;
	if (wrapper->result->HasError()) {
		return 0;
	}
	idx_t rows_changed = 0;
	auto &collection = wrapper->result->Collection();
	idx_t row_count = collection.Count();
	if (row_count > 0 && wrapper->result->properties.return_type == duckdb::StatementReturnType::CHANGED_ROWS) {
		auto rows = collection.GetRows();
		D_ASSERT(row_count == 1);
		D_ASSERT(rows.size() == 1);
		rows_changed = rows[0].GetValue(0).GetValue<int64_t>();
	}
	return rows_changed;
}

const char *duckdb_query_arrow_error(duckdb_arrow result) {
	auto wrapper = (ArrowResultWrapper *)result;
	return wrapper->result->GetError().c_str();
}

void duckdb_destroy_arrow(duckdb_arrow *result) {
	if (*result) {
		auto wrapper = (ArrowResultWrapper *)*result;
		delete wrapper;
		*result = nullptr;
	}
}

duckdb_state duckdb_execute_prepared_arrow(duckdb_prepared_statement prepared_statement, duckdb_arrow *out_result) {
	auto wrapper = (PreparedStatementWrapper *)prepared_statement;
	if (!wrapper || !wrapper->statement || wrapper->statement->HasError() || !out_result) {
		return DuckDBError;
	}
	auto arrow_wrapper = new ArrowResultWrapper();
	if (wrapper->statement->context->config.set_variables.find("TimeZone") ==
	    wrapper->statement->context->config.set_variables.end()) {
		arrow_wrapper->timezone_config = "UTC";
	} else {
		arrow_wrapper->timezone_config =
		    wrapper->statement->context->config.set_variables["TimeZone"].GetValue<std::string>();
	}

	auto result = wrapper->statement->Execute(wrapper->values, false);
	D_ASSERT(result->type == QueryResultType::MATERIALIZED_RESULT);
	arrow_wrapper->result = duckdb::unique_ptr_cast<QueryResult, MaterializedQueryResult>(std::move(result));
	*out_result = (duckdb_arrow)arrow_wrapper;
	return !arrow_wrapper->result->HasError() ? DuckDBSuccess : DuckDBError;
}



namespace duckdb {

//! DECIMAL -> VARCHAR
template <>
bool CastDecimalCInternal(duckdb_result *source, duckdb_string &result, idx_t col, idx_t row) {
	auto result_data = (duckdb::DuckDBResultData *)source->internal_data;
	auto &query_result = result_data->result;
	auto &source_type = query_result->types[col];
	auto width = duckdb::DecimalType::GetWidth(source_type);
	auto scale = duckdb::DecimalType::GetScale(source_type);
	duckdb::Vector result_vec(duckdb::LogicalType::VARCHAR, false, false);
	duckdb::string_t result_string;
	void *source_address = UnsafeFetchPtr<hugeint_t>(source, col, row);
	switch (source_type.InternalType()) {
	case duckdb::PhysicalType::INT16:
		result_string = duckdb::StringCastFromDecimal::Operation<int16_t>(UnsafeFetchFromPtr<int16_t>(source_address),
		                                                                  width, scale, result_vec);
		break;
	case duckdb::PhysicalType::INT32:
		result_string = duckdb::StringCastFromDecimal::Operation<int32_t>(UnsafeFetchFromPtr<int32_t>(source_address),
		                                                                  width, scale, result_vec);
		break;
	case duckdb::PhysicalType::INT64:
		result_string = duckdb::StringCastFromDecimal::Operation<int64_t>(UnsafeFetchFromPtr<int64_t>(source_address),
		                                                                  width, scale, result_vec);
		break;
	case duckdb::PhysicalType::INT128:
		result_string = duckdb::StringCastFromDecimal::Operation<hugeint_t>(
		    UnsafeFetchFromPtr<hugeint_t>(source_address), width, scale, result_vec);
		break;
	default:
		throw duckdb::InternalException("Unimplemented internal type for decimal");
	}
	result.data = (char *)duckdb_malloc(sizeof(char) * (result_string.GetSize() + 1));
	memcpy(result.data, result_string.GetDataUnsafe(), result_string.GetSize());
	result.data[result_string.GetSize()] = '\0';
	result.size = result_string.GetSize();
	return true;
}

template <class INTERNAL_TYPE>
duckdb_hugeint FetchInternals(void *source_address) {
	throw duckdb::NotImplementedException("FetchInternals not implemented for internal type");
}

template <>
duckdb_hugeint FetchInternals<int16_t>(void *source_address) {
	duckdb_hugeint result;
	int16_t intermediate_result;

	if (!TryCast::Operation<int16_t, int16_t>(UnsafeFetchFromPtr<int16_t>(source_address), intermediate_result)) {
		intermediate_result = FetchDefaultValue::Operation<int16_t>();
	}
	hugeint_t hugeint_result = Hugeint::Cast<int16_t>(intermediate_result);
	result.lower = hugeint_result.lower;
	result.upper = hugeint_result.upper;
	return result;
}
template <>
duckdb_hugeint FetchInternals<int32_t>(void *source_address) {
	duckdb_hugeint result;
	int32_t intermediate_result;

	if (!TryCast::Operation<int32_t, int32_t>(UnsafeFetchFromPtr<int32_t>(source_address), intermediate_result)) {
		intermediate_result = FetchDefaultValue::Operation<int32_t>();
	}
	hugeint_t hugeint_result = Hugeint::Cast<int32_t>(intermediate_result);
	result.lower = hugeint_result.lower;
	result.upper = hugeint_result.upper;
	return result;
}
template <>
duckdb_hugeint FetchInternals<int64_t>(void *source_address) {
	duckdb_hugeint result;
	int64_t intermediate_result;

	if (!TryCast::Operation<int64_t, int64_t>(UnsafeFetchFromPtr<int64_t>(source_address), intermediate_result)) {
		intermediate_result = FetchDefaultValue::Operation<int64_t>();
	}
	hugeint_t hugeint_result = Hugeint::Cast<int64_t>(intermediate_result);
	result.lower = hugeint_result.lower;
	result.upper = hugeint_result.upper;
	return result;
}
template <>
duckdb_hugeint FetchInternals<hugeint_t>(void *source_address) {
	duckdb_hugeint result;
	hugeint_t intermediate_result;

	if (!TryCast::Operation<hugeint_t, hugeint_t>(UnsafeFetchFromPtr<hugeint_t>(source_address), intermediate_result)) {
		intermediate_result = FetchDefaultValue::Operation<hugeint_t>();
	}
	result.lower = intermediate_result.lower;
	result.upper = intermediate_result.upper;
	return result;
}

//! DECIMAL -> DECIMAL (internal fetch)
template <>
bool CastDecimalCInternal(duckdb_result *source, duckdb_decimal &result, idx_t col, idx_t row) {
	auto result_data = (duckdb::DuckDBResultData *)source->internal_data;
	result_data->result->types[col].GetDecimalProperties(result.width, result.scale);
	auto source_address = UnsafeFetchPtr<hugeint_t>(source, col, row);

	if (result.width > duckdb::Decimal::MAX_WIDTH_INT64) {
		result.value = FetchInternals<hugeint_t>(source_address);
	} else if (result.width > duckdb::Decimal::MAX_WIDTH_INT32) {
		result.value = FetchInternals<int64_t>(source_address);
	} else if (result.width > duckdb::Decimal::MAX_WIDTH_INT16) {
		result.value = FetchInternals<int32_t>(source_address);
	} else {
		result.value = FetchInternals<int16_t>(source_address);
	}
	return true;
}

} // namespace duckdb


namespace duckdb {

template <>
duckdb_decimal FetchDefaultValue::Operation() {
	duckdb_decimal result;
	result.scale = 0;
	result.width = 0;
	result.value = {0, 0};
	return result;
}

template <>
date_t FetchDefaultValue::Operation() {
	date_t result;
	result.days = 0;
	return result;
}

template <>
dtime_t FetchDefaultValue::Operation() {
	dtime_t result;
	result.micros = 0;
	return result;
}

template <>
timestamp_t FetchDefaultValue::Operation() {
	timestamp_t result;
	result.value = 0;
	return result;
}

template <>
interval_t FetchDefaultValue::Operation() {
	interval_t result;
	result.months = 0;
	result.days = 0;
	result.micros = 0;
	return result;
}

template <>
char *FetchDefaultValue::Operation() {
	return nullptr;
}

template <>
duckdb_string FetchDefaultValue::Operation() {
	duckdb_string result;
	result.data = nullptr;
	result.size = 0;
	return result;
}

template <>
duckdb_blob FetchDefaultValue::Operation() {
	duckdb_blob result;
	result.data = nullptr;
	result.size = 0;
	return result;
}

//===--------------------------------------------------------------------===//
// Blob Casts
//===--------------------------------------------------------------------===//

template <>
bool FromCBlobCastWrapper::Operation(duckdb_blob input, duckdb_string &result) {
	string_t input_str((const char *)input.data, input.size);
	return ToCStringCastWrapper<duckdb::CastFromBlob>::template Operation<string_t, duckdb_string>(input_str, result);
}

} // namespace duckdb

bool CanUseDeprecatedFetch(duckdb_result *result, idx_t col, idx_t row) {
	if (!result) {
		return false;
	}
	if (!duckdb::deprecated_materialize_result(result)) {
		return false;
	}
	if (col >= result->__deprecated_column_count || row >= result->__deprecated_row_count) {
		return false;
	}
	return true;
}

bool CanFetchValue(duckdb_result *result, idx_t col, idx_t row) {
	if (!CanUseDeprecatedFetch(result, col, row)) {
		return false;
	}
	if (result->__deprecated_columns[col].__deprecated_nullmask[row]) {
		return false;
	}
	return true;
}




using duckdb::DBConfig;
using duckdb::Value;

// config
duckdb_state duckdb_create_config(duckdb_config *out_config) {
	if (!out_config) {
		return DuckDBError;
	}
	DBConfig *config;
	try {
		config = new DBConfig();
	} catch (...) { // LCOV_EXCL_START
		return DuckDBError;
	} // LCOV_EXCL_STOP
	*out_config = (duckdb_config)config;
	return DuckDBSuccess;
}

size_t duckdb_config_count() {
	return DBConfig::GetOptionCount();
}

duckdb_state duckdb_get_config_flag(size_t index, const char **out_name, const char **out_description) {
	auto option = DBConfig::GetOptionByIndex(index);
	if (!option) {
		return DuckDBError;
	}
	if (out_name) {
		*out_name = option->name;
	}
	if (out_description) {
		*out_description = option->description;
	}
	return DuckDBSuccess;
}

duckdb_state duckdb_set_config(duckdb_config config, const char *name, const char *option) {
	if (!config || !name || !option) {
		return DuckDBError;
	}
	auto config_option = DBConfig::GetOptionByName(name);
	if (!config_option) {
		return DuckDBError;
	}
	try {
		auto db_config = (DBConfig *)config;
		db_config->SetOption(*config_option, Value(option));
	} catch (...) {
		return DuckDBError;
	}
	return DuckDBSuccess;
}

void duckdb_destroy_config(duckdb_config *config) {
	if (!config) {
		return;
	}
	if (*config) {
		auto db_config = (DBConfig *)*config;
		delete db_config;
		*config = nullptr;
	}
}




#include <string.h>

duckdb_data_chunk duckdb_create_data_chunk(duckdb_logical_type *ctypes, idx_t column_count) {
	if (!ctypes) {
		return nullptr;
	}
	duckdb::vector<duckdb::LogicalType> types;
	for (idx_t i = 0; i < column_count; i++) {
		auto ltype = (duckdb::LogicalType *)ctypes[i];
		types.push_back(*ltype);
	}

	auto result = new duckdb::DataChunk();
	result->Initialize(duckdb::Allocator::DefaultAllocator(), types);
	return reinterpret_cast<duckdb_data_chunk>(result);
}

void duckdb_destroy_data_chunk(duckdb_data_chunk *chunk) {
	if (chunk && *chunk) {
		auto dchunk = (duckdb::DataChunk *)*chunk;
		delete dchunk;
		*chunk = nullptr;
	}
}

void duckdb_data_chunk_reset(duckdb_data_chunk chunk) {
	if (!chunk) {
		return;
	}
	auto dchunk = (duckdb::DataChunk *)chunk;
	dchunk->Reset();
}

idx_t duckdb_data_chunk_get_column_count(duckdb_data_chunk chunk) {
	if (!chunk) {
		return 0;
	}
	auto dchunk = (duckdb::DataChunk *)chunk;
	return dchunk->ColumnCount();
}

duckdb_vector duckdb_data_chunk_get_vector(duckdb_data_chunk chunk, idx_t col_idx) {
	if (!chunk || col_idx >= duckdb_data_chunk_get_column_count(chunk)) {
		return nullptr;
	}
	auto dchunk = (duckdb::DataChunk *)chunk;
	return reinterpret_cast<duckdb_vector>(&dchunk->data[col_idx]);
}

idx_t duckdb_data_chunk_get_size(duckdb_data_chunk chunk) {
	if (!chunk) {
		return 0;
	}
	auto dchunk = (duckdb::DataChunk *)chunk;
	return dchunk->size();
}

void duckdb_data_chunk_set_size(duckdb_data_chunk chunk, idx_t size) {
	if (!chunk) {
		return;
	}
	auto dchunk = (duckdb::DataChunk *)chunk;
	dchunk->SetCardinality(size);
}

duckdb_logical_type duckdb_vector_get_column_type(duckdb_vector vector) {
	if (!vector) {
		return nullptr;
	}
	auto v = (duckdb::Vector *)vector;
	return reinterpret_cast<duckdb_logical_type>(new duckdb::LogicalType(v->GetType()));
}

void *duckdb_vector_get_data(duckdb_vector vector) {
	if (!vector) {
		return nullptr;
	}
	auto v = (duckdb::Vector *)vector;
	return duckdb::FlatVector::GetData(*v);
}

uint64_t *duckdb_vector_get_validity(duckdb_vector vector) {
	if (!vector) {
		return nullptr;
	}
	auto v = (duckdb::Vector *)vector;
	return duckdb::FlatVector::Validity(*v).GetData();
}

void duckdb_vector_ensure_validity_writable(duckdb_vector vector) {
	if (!vector) {
		return;
	}
	auto v = (duckdb::Vector *)vector;
	auto &validity = duckdb::FlatVector::Validity(*v);
	validity.EnsureWritable();
}

void duckdb_vector_assign_string_element(duckdb_vector vector, idx_t index, const char *str) {
	duckdb_vector_assign_string_element_len(vector, index, str, strlen(str));
}

void duckdb_vector_assign_string_element_len(duckdb_vector vector, idx_t index, const char *str, idx_t str_len) {
	if (!vector) {
		return;
	}
	auto v = (duckdb::Vector *)vector;
	auto data = duckdb::FlatVector::GetData<duckdb::string_t>(*v);
	data[index] = duckdb::StringVector::AddString(*v, str, str_len);
}

duckdb_vector duckdb_list_vector_get_child(duckdb_vector vector) {
	if (!vector) {
		return nullptr;
	}
	auto v = (duckdb::Vector *)vector;
	return reinterpret_cast<duckdb_vector>(&duckdb::ListVector::GetEntry(*v));
}

idx_t duckdb_list_vector_get_size(duckdb_vector vector) {
	if (!vector) {
		return 0;
	}
	auto v = (duckdb::Vector *)vector;
	return duckdb::ListVector::GetListSize(*v);
}

duckdb_state duckdb_list_vector_set_size(duckdb_vector vector, idx_t size) {
	if (!vector) {
		return duckdb_state::DuckDBError;
	}
	auto v = (duckdb::Vector *)vector;
	duckdb::ListVector::SetListSize(*v, size);
	return duckdb_state::DuckDBSuccess;
}

duckdb_state duckdb_list_vector_reserve(duckdb_vector vector, idx_t required_capacity) {
	if (!vector) {
		return duckdb_state::DuckDBError;
	}
	auto v = (duckdb::Vector *)vector;
	duckdb::ListVector::Reserve(*v, required_capacity);
	return duckdb_state::DuckDBSuccess;
}

duckdb_vector duckdb_struct_vector_get_child(duckdb_vector vector, idx_t index) {
	if (!vector) {
		return nullptr;
	}
	auto v = (duckdb::Vector *)vector;
	return reinterpret_cast<duckdb_vector>(duckdb::StructVector::GetEntries(*v)[index].get());
}

bool duckdb_validity_row_is_valid(uint64_t *validity, idx_t row) {
	if (!validity) {
		return true;
	}
	idx_t entry_idx = row / 64;
	idx_t idx_in_entry = row % 64;
	return validity[entry_idx] & ((idx_t)1 << idx_in_entry);
}

void duckdb_validity_set_row_validity(uint64_t *validity, idx_t row, bool valid) {
	if (valid) {
		duckdb_validity_set_row_valid(validity, row);
	} else {
		duckdb_validity_set_row_invalid(validity, row);
	}
}

void duckdb_validity_set_row_invalid(uint64_t *validity, idx_t row) {
	if (!validity) {
		return;
	}
	idx_t entry_idx = row / 64;
	idx_t idx_in_entry = row % 64;
	validity[entry_idx] &= ~((uint64_t)1 << idx_in_entry);
}

void duckdb_validity_set_row_valid(uint64_t *validity, idx_t row) {
	if (!validity) {
		return;
	}
	idx_t entry_idx = row / 64;
	idx_t idx_in_entry = row % 64;
	validity[entry_idx] |= (uint64_t)1 << idx_in_entry;
}





using duckdb::Date;
using duckdb::Time;
using duckdb::Timestamp;

using duckdb::date_t;
using duckdb::dtime_t;
using duckdb::timestamp_t;

duckdb_date_struct duckdb_from_date(duckdb_date date) {
	int32_t year, month, day;
	Date::Convert(date_t(date.days), year, month, day);

	duckdb_date_struct result;
	result.year = year;
	result.month = month;
	result.day = day;
	return result;
}

duckdb_date duckdb_to_date(duckdb_date_struct date) {
	duckdb_date result;
	result.days = Date::FromDate(date.year, date.month, date.day).days;
	return result;
}

duckdb_time_struct duckdb_from_time(duckdb_time time) {
	int32_t hour, minute, second, micros;
	Time::Convert(dtime_t(time.micros), hour, minute, second, micros);

	duckdb_time_struct result;
	result.hour = hour;
	result.min = minute;
	result.sec = second;
	result.micros = micros;
	return result;
}

duckdb_time duckdb_to_time(duckdb_time_struct time) {
	duckdb_time result;
	result.micros = Time::FromTime(time.hour, time.min, time.sec, time.micros).micros;
	return result;
}

duckdb_timestamp_struct duckdb_from_timestamp(duckdb_timestamp ts) {
	date_t date;
	dtime_t time;
	Timestamp::Convert(timestamp_t(ts.micros), date, time);

	duckdb_date ddate;
	ddate.days = date.days;

	duckdb_time dtime;
	dtime.micros = time.micros;

	duckdb_timestamp_struct result;
	result.date = duckdb_from_date(ddate);
	result.time = duckdb_from_time(dtime);
	return result;
}

duckdb_timestamp duckdb_to_timestamp(duckdb_timestamp_struct ts) {
	date_t date = date_t(duckdb_to_date(ts.date).days);
	dtime_t time = dtime_t(duckdb_to_time(ts.time).micros);

	duckdb_timestamp result;
	result.micros = Timestamp::FromDatetime(date, time).value;
	return result;
}


using duckdb::Connection;
using duckdb::DatabaseData;
using duckdb::DBConfig;
using duckdb::DuckDB;

duckdb_state duckdb_open_ext(const char *path, duckdb_database *out, duckdb_config config, char **error) {
	auto wrapper = new DatabaseData();
	try {
		auto db_config = (DBConfig *)config;
		wrapper->database = duckdb::make_unique<DuckDB>(path, db_config);
	} catch (std::exception &ex) {
		if (error) {
			*error = strdup(ex.what());
		}
		delete wrapper;
		return DuckDBError;
	} catch (...) { // LCOV_EXCL_START
		if (error) {
			*error = strdup("Unknown error");
		}
		delete wrapper;
		return DuckDBError;
	} // LCOV_EXCL_STOP
	*out = (duckdb_database)wrapper;
	return DuckDBSuccess;
}

duckdb_state duckdb_open(const char *path, duckdb_database *out) {
	return duckdb_open_ext(path, out, nullptr, nullptr);
}

void duckdb_close(duckdb_database *database) {
	if (database && *database) {
		auto wrapper = (DatabaseData *)*database;
		delete wrapper;
		*database = nullptr;
	}
}

duckdb_state duckdb_connect(duckdb_database database, duckdb_connection *out) {
	if (!database || !out) {
		return DuckDBError;
	}
	auto wrapper = (DatabaseData *)database;
	Connection *connection;
	try {
		connection = new Connection(*wrapper->database);
	} catch (...) { // LCOV_EXCL_START
		return DuckDBError;
	} // LCOV_EXCL_STOP
	*out = (duckdb_connection)connection;
	return DuckDBSuccess;
}

void duckdb_disconnect(duckdb_connection *connection) {
	if (connection && *connection) {
		Connection *conn = (Connection *)*connection;
		delete conn;
		*connection = nullptr;
	}
}

duckdb_state duckdb_query(duckdb_connection connection, const char *query, duckdb_result *out) {
	Connection *conn = (Connection *)connection;
	auto result = conn->Query(query);
	return duckdb_translate_result(std::move(result), out);
}

const char *duckdb_library_version() {
	return DuckDB::LibraryVersion();
}


void duckdb_destroy_value(duckdb_value *value) {
	if (value && *value) {
		auto val = (duckdb::Value *)*value;
		delete val;
		*value = nullptr;
	}
}

duckdb_value duckdb_create_varchar_length(const char *text, idx_t length) {
	return reinterpret_cast<duckdb_value>(new duckdb::Value(std::string(text, length)));
}

duckdb_value duckdb_create_varchar(const char *text) {
	return duckdb_create_varchar_length(text, strlen(text));
}

duckdb_value duckdb_create_int64(int64_t input) {
	auto val = duckdb::Value::BIGINT(input);
	return reinterpret_cast<duckdb_value>(new duckdb::Value(val));
}

char *duckdb_get_varchar(duckdb_value value) {
	auto val = (duckdb::Value *)value;
	auto str_val = val->DefaultCastAs(duckdb::LogicalType::VARCHAR);
	auto &str = duckdb::StringValue::Get(str_val);

	auto result = (char *)malloc(sizeof(char *) * (str.size() + 1));
	memcpy(result, str.c_str(), str.size());
	result[str.size()] = '\0';
	return result;
}

int64_t duckdb_get_int64(duckdb_value value) {
	auto val = (duckdb::Value *)value;
	if (!val->DefaultTryCastAs(duckdb::LogicalType::BIGINT)) {
		return 0;
	}
	return duckdb::BigIntValue::Get(*val);
}


namespace duckdb {

LogicalTypeId ConvertCTypeToCPP(duckdb_type c_type) {
	switch (c_type) {
	case DUCKDB_TYPE_BOOLEAN:
		return LogicalTypeId::BOOLEAN;
	case DUCKDB_TYPE_TINYINT:
		return LogicalTypeId::TINYINT;
	case DUCKDB_TYPE_SMALLINT:
		return LogicalTypeId::SMALLINT;
	case DUCKDB_TYPE_INTEGER:
		return LogicalTypeId::INTEGER;
	case DUCKDB_TYPE_BIGINT:
		return LogicalTypeId::BIGINT;
	case DUCKDB_TYPE_UTINYINT:
		return LogicalTypeId::UTINYINT;
	case DUCKDB_TYPE_USMALLINT:
		return LogicalTypeId::USMALLINT;
	case DUCKDB_TYPE_UINTEGER:
		return LogicalTypeId::UINTEGER;
	case DUCKDB_TYPE_UBIGINT:
		return LogicalTypeId::UBIGINT;
	case DUCKDB_TYPE_HUGEINT:
		return LogicalTypeId::HUGEINT;
	case DUCKDB_TYPE_FLOAT:
		return LogicalTypeId::FLOAT;
	case DUCKDB_TYPE_DOUBLE:
		return LogicalTypeId::DOUBLE;
	case DUCKDB_TYPE_TIMESTAMP:
		return LogicalTypeId::TIMESTAMP;
	case DUCKDB_TYPE_DATE:
		return LogicalTypeId::DATE;
	case DUCKDB_TYPE_TIME:
		return LogicalTypeId::TIME;
	case DUCKDB_TYPE_VARCHAR:
		return LogicalTypeId::VARCHAR;
	case DUCKDB_TYPE_BLOB:
		return LogicalTypeId::BLOB;
	case DUCKDB_TYPE_INTERVAL:
		return LogicalTypeId::INTERVAL;
	case DUCKDB_TYPE_TIMESTAMP_S:
		return LogicalTypeId::TIMESTAMP_SEC;
	case DUCKDB_TYPE_TIMESTAMP_MS:
		return LogicalTypeId::TIMESTAMP_MS;
	case DUCKDB_TYPE_TIMESTAMP_NS:
		return LogicalTypeId::TIMESTAMP_NS;
	case DUCKDB_TYPE_UUID:
		return LogicalTypeId::UUID;
	default: // LCOV_EXCL_START
		D_ASSERT(0);
		return LogicalTypeId::INVALID;
	} // LCOV_EXCL_STOP
}

duckdb_type ConvertCPPTypeToC(const LogicalType &sql_type) {
	switch (sql_type.id()) {
	case LogicalTypeId::BOOLEAN:
		return DUCKDB_TYPE_BOOLEAN;
	case LogicalTypeId::TINYINT:
		return DUCKDB_TYPE_TINYINT;
	case LogicalTypeId::SMALLINT:
		return DUCKDB_TYPE_SMALLINT;
	case LogicalTypeId::INTEGER:
		return DUCKDB_TYPE_INTEGER;
	case LogicalTypeId::BIGINT:
		return DUCKDB_TYPE_BIGINT;
	case LogicalTypeId::UTINYINT:
		return DUCKDB_TYPE_UTINYINT;
	case LogicalTypeId::USMALLINT:
		return DUCKDB_TYPE_USMALLINT;
	case LogicalTypeId::UINTEGER:
		return DUCKDB_TYPE_UINTEGER;
	case LogicalTypeId::UBIGINT:
		return DUCKDB_TYPE_UBIGINT;
	case LogicalTypeId::HUGEINT:
		return DUCKDB_TYPE_HUGEINT;
	case LogicalTypeId::FLOAT:
		return DUCKDB_TYPE_FLOAT;
	case LogicalTypeId::DOUBLE:
		return DUCKDB_TYPE_DOUBLE;
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_TZ:
		return DUCKDB_TYPE_TIMESTAMP;
	case LogicalTypeId::TIMESTAMP_SEC:
		return DUCKDB_TYPE_TIMESTAMP_S;
	case LogicalTypeId::TIMESTAMP_MS:
		return DUCKDB_TYPE_TIMESTAMP_MS;
	case LogicalTypeId::TIMESTAMP_NS:
		return DUCKDB_TYPE_TIMESTAMP_NS;
	case LogicalTypeId::DATE:
		return DUCKDB_TYPE_DATE;
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIME_TZ:
		return DUCKDB_TYPE_TIME;
	case LogicalTypeId::VARCHAR:
		return DUCKDB_TYPE_VARCHAR;
	case LogicalTypeId::BLOB:
		return DUCKDB_TYPE_BLOB;
	case LogicalTypeId::BIT:
		return DUCKDB_TYPE_BIT;
	case LogicalTypeId::INTERVAL:
		return DUCKDB_TYPE_INTERVAL;
	case LogicalTypeId::DECIMAL:
		return DUCKDB_TYPE_DECIMAL;
	case LogicalTypeId::ENUM:
		return DUCKDB_TYPE_ENUM;
	case LogicalTypeId::LIST:
		return DUCKDB_TYPE_LIST;
	case LogicalTypeId::STRUCT:
		return DUCKDB_TYPE_STRUCT;
	case LogicalTypeId::MAP:
		return DUCKDB_TYPE_MAP;
	case LogicalTypeId::UNION:
		return DUCKDB_TYPE_UNION;
	case LogicalTypeId::UUID:
		return DUCKDB_TYPE_UUID;
	default: // LCOV_EXCL_START
		D_ASSERT(0);
		return DUCKDB_TYPE_INVALID;
	} // LCOV_EXCL_STOP
}

idx_t GetCTypeSize(duckdb_type type) {
	switch (type) {
	case DUCKDB_TYPE_BOOLEAN:
		return sizeof(bool);
	case DUCKDB_TYPE_TINYINT:
		return sizeof(int8_t);
	case DUCKDB_TYPE_SMALLINT:
		return sizeof(int16_t);
	case DUCKDB_TYPE_INTEGER:
		return sizeof(int32_t);
	case DUCKDB_TYPE_BIGINT:
		return sizeof(int64_t);
	case DUCKDB_TYPE_UTINYINT:
		return sizeof(uint8_t);
	case DUCKDB_TYPE_USMALLINT:
		return sizeof(uint16_t);
	case DUCKDB_TYPE_UINTEGER:
		return sizeof(uint32_t);
	case DUCKDB_TYPE_UBIGINT:
		return sizeof(uint64_t);
	case DUCKDB_TYPE_HUGEINT:
	case DUCKDB_TYPE_UUID:
		return sizeof(duckdb_hugeint);
	case DUCKDB_TYPE_FLOAT:
		return sizeof(float);
	case DUCKDB_TYPE_DOUBLE:
		return sizeof(double);
	case DUCKDB_TYPE_DATE:
		return sizeof(duckdb_date);
	case DUCKDB_TYPE_TIME:
		return sizeof(duckdb_time);
	case DUCKDB_TYPE_TIMESTAMP:
	case DUCKDB_TYPE_TIMESTAMP_S:
	case DUCKDB_TYPE_TIMESTAMP_MS:
	case DUCKDB_TYPE_TIMESTAMP_NS:
		return sizeof(duckdb_timestamp);
	case DUCKDB_TYPE_VARCHAR:
		return sizeof(const char *);
	case DUCKDB_TYPE_BLOB:
		return sizeof(duckdb_blob);
	case DUCKDB_TYPE_INTERVAL:
		return sizeof(duckdb_interval);
	case DUCKDB_TYPE_DECIMAL:
		return sizeof(duckdb_hugeint);
	default: // LCOV_EXCL_START
		// unsupported type
		D_ASSERT(0);
		return sizeof(const char *);
	} // LCOV_EXCL_STOP
}

} // namespace duckdb

void *duckdb_malloc(size_t size) {
	return malloc(size);
}

void duckdb_free(void *ptr) {
	free(ptr);
}

idx_t duckdb_vector_size() {
	return STANDARD_VECTOR_SIZE;
}







using duckdb::Hugeint;
using duckdb::hugeint_t;
using duckdb::Value;

double duckdb_hugeint_to_double(duckdb_hugeint val) {
	hugeint_t internal;
	internal.lower = val.lower;
	internal.upper = val.upper;
	return Hugeint::Cast<double>(internal);
}

static duckdb_decimal to_decimal_cast(double val, uint8_t width, uint8_t scale) {
	if (width > duckdb::Decimal::MAX_WIDTH_INT64) {
		return duckdb::TryCastToDecimalCInternal<double, duckdb::ToCDecimalCastWrapper<hugeint_t>>(val, width, scale);
	}
	if (width > duckdb::Decimal::MAX_WIDTH_INT32) {
		return duckdb::TryCastToDecimalCInternal<double, duckdb::ToCDecimalCastWrapper<int64_t>>(val, width, scale);
	}
	if (width > duckdb::Decimal::MAX_WIDTH_INT16) {
		return duckdb::TryCastToDecimalCInternal<double, duckdb::ToCDecimalCastWrapper<int32_t>>(val, width, scale);
	}
	return duckdb::TryCastToDecimalCInternal<double, duckdb::ToCDecimalCastWrapper<int16_t>>(val, width, scale);
}

duckdb_decimal duckdb_double_to_decimal(double val, uint8_t width, uint8_t scale) {
	if (scale > width || width > duckdb::Decimal::MAX_WIDTH_INT128) {
		return duckdb::FetchDefaultValue::Operation<duckdb_decimal>();
	}
	return to_decimal_cast(val, width, scale);
}

duckdb_hugeint duckdb_double_to_hugeint(double val) {
	hugeint_t internal_result;
	if (!Value::DoubleIsFinite(val) || !Hugeint::TryConvert<double>(val, internal_result)) {
		internal_result.lower = 0;
		internal_result.upper = 0;
	}

	duckdb_hugeint result;
	result.lower = internal_result.lower;
	result.upper = internal_result.upper;
	return result;
}

double duckdb_decimal_to_double(duckdb_decimal val) {
	double result;
	hugeint_t value;
	value.lower = val.value.lower;
	value.upper = val.value.upper;
	duckdb::TryCastFromDecimal::Operation<hugeint_t, double>(value, result, nullptr, val.width, val.scale);
	return result;
}


static bool AssertLogicalTypeId(duckdb_logical_type type, duckdb::LogicalTypeId type_id) {
	if (!type) {
		return false;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	if (ltype.id() != type_id) {
		return false;
	}
	return true;
}

static bool AssertInternalType(duckdb_logical_type type, duckdb::PhysicalType physical_type) {
	if (!type) {
		return false;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	if (ltype.InternalType() != physical_type) {
		return false;
	}
	return true;
}

duckdb_logical_type duckdb_create_logical_type(duckdb_type type) {
	return reinterpret_cast<duckdb_logical_type>(new duckdb::LogicalType(duckdb::ConvertCTypeToCPP(type)));
}

duckdb_logical_type duckdb_create_list_type(duckdb_logical_type type) {
	if (!type) {
		return nullptr;
	}
	duckdb::LogicalType *ltype = new duckdb::LogicalType;
	*ltype = duckdb::LogicalType::LIST(*(duckdb::LogicalType *)type);
	return reinterpret_cast<duckdb_logical_type>(ltype);
}

duckdb_logical_type duckdb_create_union_type(duckdb_logical_type member_types_p, const char **member_names,
                                             idx_t member_count) {
	if (!member_types_p || !member_names) {
		return nullptr;
	}
	duckdb::LogicalType *member_types = (duckdb::LogicalType *)member_types_p;
	duckdb::LogicalType *mtype = new duckdb::LogicalType;
	duckdb::child_list_t<duckdb::LogicalType> members;

	for (idx_t i = 0; i < member_count; i++) {
		members.push_back(make_pair(member_names[i], member_types[i]));
	}
	*mtype = duckdb::LogicalType::UNION(members);
	return reinterpret_cast<duckdb_logical_type>(mtype);
}

duckdb_logical_type duckdb_create_map_type(duckdb_logical_type key_type, duckdb_logical_type value_type) {
	if (!key_type || !value_type) {
		return nullptr;
	}
	duckdb::LogicalType *mtype = new duckdb::LogicalType;
	*mtype = duckdb::LogicalType::MAP(*(duckdb::LogicalType *)key_type, *(duckdb::LogicalType *)value_type);
	return reinterpret_cast<duckdb_logical_type>(mtype);
}

duckdb_logical_type duckdb_create_decimal_type(uint8_t width, uint8_t scale) {
	return reinterpret_cast<duckdb_logical_type>(new duckdb::LogicalType(duckdb::LogicalType::DECIMAL(width, scale)));
}

duckdb_type duckdb_get_type_id(duckdb_logical_type type) {
	if (!type) {
		return DUCKDB_TYPE_INVALID;
	}
	auto ltype = (duckdb::LogicalType *)type;
	return duckdb::ConvertCPPTypeToC(*ltype);
}

void duckdb_destroy_logical_type(duckdb_logical_type *type) {
	if (type && *type) {
		auto ltype = (duckdb::LogicalType *)*type;
		delete ltype;
		*type = nullptr;
	}
}

uint8_t duckdb_decimal_width(duckdb_logical_type type) {
	if (!AssertLogicalTypeId(type, duckdb::LogicalTypeId::DECIMAL)) {
		return 0;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	return duckdb::DecimalType::GetWidth(ltype);
}

uint8_t duckdb_decimal_scale(duckdb_logical_type type) {
	if (!AssertLogicalTypeId(type, duckdb::LogicalTypeId::DECIMAL)) {
		return 0;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	return duckdb::DecimalType::GetScale(ltype);
}

duckdb_type duckdb_decimal_internal_type(duckdb_logical_type type) {
	if (!AssertLogicalTypeId(type, duckdb::LogicalTypeId::DECIMAL)) {
		return DUCKDB_TYPE_INVALID;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	switch (ltype.InternalType()) {
	case duckdb::PhysicalType::INT16:
		return DUCKDB_TYPE_SMALLINT;
	case duckdb::PhysicalType::INT32:
		return DUCKDB_TYPE_INTEGER;
	case duckdb::PhysicalType::INT64:
		return DUCKDB_TYPE_BIGINT;
	case duckdb::PhysicalType::INT128:
		return DUCKDB_TYPE_HUGEINT;
	default:
		return DUCKDB_TYPE_INVALID;
	}
}

duckdb_type duckdb_enum_internal_type(duckdb_logical_type type) {
	if (!AssertLogicalTypeId(type, duckdb::LogicalTypeId::ENUM)) {
		return DUCKDB_TYPE_INVALID;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	switch (ltype.InternalType()) {
	case duckdb::PhysicalType::UINT8:
		return DUCKDB_TYPE_UTINYINT;
	case duckdb::PhysicalType::UINT16:
		return DUCKDB_TYPE_USMALLINT;
	case duckdb::PhysicalType::UINT32:
		return DUCKDB_TYPE_UINTEGER;
	default:
		return DUCKDB_TYPE_INVALID;
	}
}

uint32_t duckdb_enum_dictionary_size(duckdb_logical_type type) {
	if (!AssertLogicalTypeId(type, duckdb::LogicalTypeId::ENUM)) {
		return 0;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	return duckdb::EnumType::GetSize(ltype);
}

char *duckdb_enum_dictionary_value(duckdb_logical_type type, idx_t index) {
	if (!AssertLogicalTypeId(type, duckdb::LogicalTypeId::ENUM)) {
		return nullptr;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	auto &vector = duckdb::EnumType::GetValuesInsertOrder(ltype);
	auto value = vector.GetValue(index);
	return strdup(duckdb::StringValue::Get(value).c_str());
}

duckdb_logical_type duckdb_list_type_child_type(duckdb_logical_type type) {
	if (!AssertLogicalTypeId(type, duckdb::LogicalTypeId::LIST) &&
	    !AssertLogicalTypeId(type, duckdb::LogicalTypeId::MAP)) {
		return nullptr;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	if (ltype.id() != duckdb::LogicalTypeId::LIST && ltype.id() != duckdb::LogicalTypeId::MAP) {
		return nullptr;
	}
	return reinterpret_cast<duckdb_logical_type>(new duckdb::LogicalType(duckdb::ListType::GetChildType(ltype)));
}

duckdb_logical_type duckdb_map_type_key_type(duckdb_logical_type type) {
	if (!AssertLogicalTypeId(type, duckdb::LogicalTypeId::MAP)) {
		return nullptr;
	}
	auto &mtype = *((duckdb::LogicalType *)type);
	if (mtype.id() != duckdb::LogicalTypeId::MAP) {
		return nullptr;
	}
	return reinterpret_cast<duckdb_logical_type>(new duckdb::LogicalType(duckdb::MapType::KeyType(mtype)));
}

duckdb_logical_type duckdb_map_type_value_type(duckdb_logical_type type) {
	if (!AssertLogicalTypeId(type, duckdb::LogicalTypeId::MAP)) {
		return nullptr;
	}
	auto &mtype = *((duckdb::LogicalType *)type);
	if (mtype.id() != duckdb::LogicalTypeId::MAP) {
		return nullptr;
	}
	return reinterpret_cast<duckdb_logical_type>(new duckdb::LogicalType(duckdb::MapType::ValueType(mtype)));
}

idx_t duckdb_struct_type_child_count(duckdb_logical_type type) {
	if (!AssertInternalType(type, duckdb::PhysicalType::STRUCT)) {
		return 0;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	return duckdb::StructType::GetChildCount(ltype);
}

idx_t duckdb_union_type_member_count(duckdb_logical_type type) {
	if (!AssertLogicalTypeId(type, duckdb::LogicalTypeId::UNION)) {
		return 0;
	}
	idx_t member_count = duckdb_struct_type_child_count(type);
	if (member_count != 0) {
		member_count--;
	}
	return member_count;
}

char *duckdb_union_type_member_name(duckdb_logical_type type, idx_t index) {
	if (!AssertInternalType(type, duckdb::PhysicalType::STRUCT)) {
		return nullptr;
	}
	if (!AssertLogicalTypeId(type, duckdb::LogicalTypeId::UNION)) {
		return nullptr;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	return strdup(duckdb::UnionType::GetMemberName(ltype, index).c_str());
}

duckdb_logical_type duckdb_union_type_member_type(duckdb_logical_type type, idx_t index) {
	if (!AssertInternalType(type, duckdb::PhysicalType::STRUCT)) {
		return nullptr;
	}
	if (!AssertLogicalTypeId(type, duckdb::LogicalTypeId::UNION)) {
		return nullptr;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	return reinterpret_cast<duckdb_logical_type>(
	    new duckdb::LogicalType(duckdb::UnionType::GetMemberType(ltype, index)));
}

char *duckdb_struct_type_child_name(duckdb_logical_type type, idx_t index) {
	if (!AssertInternalType(type, duckdb::PhysicalType::STRUCT)) {
		return nullptr;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	return strdup(duckdb::StructType::GetChildName(ltype, index).c_str());
}

duckdb_logical_type duckdb_struct_type_child_type(duckdb_logical_type type, idx_t index) {
	if (!AssertInternalType(type, duckdb::PhysicalType::STRUCT)) {
		return nullptr;
	}
	auto &ltype = *((duckdb::LogicalType *)type);
	if (ltype.InternalType() != duckdb::PhysicalType::STRUCT) {
		return nullptr;
	}
	return reinterpret_cast<duckdb_logical_type>(
	    new duckdb::LogicalType(duckdb::StructType::GetChildType(ltype, index)));
}
