// See https://raw.githubusercontent.com/duckdb/duckdb/master/LICENSE for licensing information

#include "duckdb.hpp"
#include "duckdb-internal.hpp"
#ifndef DUCKDB_AMALGAMATION
#error header mismatch
#endif



namespace duckdb {

unique_ptr<BaseStatistics> StatisticsPropagator::PropagateExpression(BoundCaseExpression &bound_case,
                                                                     unique_ptr<Expression> *expr_ptr) {
	// propagate in all the children
	auto result_stats = PropagateExpression(bound_case.else_expr);
	for (auto &case_check : bound_case.case_checks) {
		PropagateExpression(case_check.when_expr);
		auto then_stats = PropagateExpression(case_check.then_expr);
		if (!then_stats) {
			result_stats.reset();
		} else if (result_stats) {
			result_stats->Merge(*then_stats);
		}
	}
	return result_stats;
}

} // namespace duckdb




namespace duckdb {

static unique_ptr<BaseStatistics> StatisticsOperationsNumericNumericCast(const BaseStatistics *input_p,
                                                                         const LogicalType &target) {
	auto &input = (NumericStatistics &)*input_p;

	Value min = input.min, max = input.max;
	if (!min.DefaultTryCastAs(target) || !max.DefaultTryCastAs(target)) {
		// overflow in cast: bailout
		return nullptr;
	}
	auto stats = make_unique<NumericStatistics>(target, std::move(min), std::move(max), input.stats_type);
	stats->CopyBase(*input_p);
	return std::move(stats);
}

static unique_ptr<BaseStatistics> StatisticsNumericCastSwitch(const BaseStatistics *input, const LogicalType &target) {
	switch (target.InternalType()) {
	case PhysicalType::INT8:
	case PhysicalType::INT16:
	case PhysicalType::INT32:
	case PhysicalType::INT64:
	case PhysicalType::INT128:
	case PhysicalType::FLOAT:
	case PhysicalType::DOUBLE:
		return StatisticsOperationsNumericNumericCast(input, target);
	default:
		return nullptr;
	}
}

unique_ptr<BaseStatistics> StatisticsPropagator::PropagateExpression(BoundCastExpression &cast,
                                                                     unique_ptr<Expression> *expr_ptr) {
	auto child_stats = PropagateExpression(cast.child);
	if (!child_stats) {
		return nullptr;
	}
	unique_ptr<BaseStatistics> result_stats;
	switch (cast.child->return_type.InternalType()) {
	case PhysicalType::INT8:
	case PhysicalType::INT16:
	case PhysicalType::INT32:
	case PhysicalType::INT64:
	case PhysicalType::INT128:
	case PhysicalType::FLOAT:
	case PhysicalType::DOUBLE:
		result_stats = StatisticsNumericCastSwitch(child_stats.get(), cast.return_type);
		break;
	default:
		return nullptr;
	}
	if (cast.try_cast && result_stats) {
		result_stats->validity_stats = make_unique<ValidityStatistics>(true, true);
	}
	return result_stats;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<BaseStatistics> StatisticsPropagator::PropagateExpression(BoundColumnRefExpression &colref,
                                                                     unique_ptr<Expression> *expr_ptr) {
	auto stats = statistics_map.find(colref.binding);
	if (stats == statistics_map.end()) {
		return nullptr;
	}
	return stats->second->Copy();
}

} // namespace duckdb






namespace duckdb {

FilterPropagateResult StatisticsPropagator::PropagateComparison(BaseStatistics &left, BaseStatistics &right,
                                                                ExpressionType comparison) {
	// only handle numerics for now
	switch (left.type.InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::UINT8:
	case PhysicalType::UINT16:
	case PhysicalType::UINT32:
	case PhysicalType::UINT64:
	case PhysicalType::INT8:
	case PhysicalType::INT16:
	case PhysicalType::INT32:
	case PhysicalType::INT64:
	case PhysicalType::INT128:
	case PhysicalType::FLOAT:
	case PhysicalType::DOUBLE:
		break;
	default:
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	auto &lstats = (NumericStatistics &)left;
	auto &rstats = (NumericStatistics &)right;
	if (lstats.min.IsNull() || lstats.max.IsNull() || rstats.min.IsNull() || rstats.max.IsNull()) {
		// no stats available: nothing to prune
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	// the result of the propagation depend on whether or not either side has null values
	// if there are null values present, we cannot say whether or not
	bool has_null = lstats.CanHaveNull() || rstats.CanHaveNull();
	switch (comparison) {
	case ExpressionType::COMPARE_EQUAL:
		// l = r, if l.min > r.max or r.min > l.max equality is not possible
		if (lstats.min > rstats.max || rstats.min > lstats.max) {
			return has_null ? FilterPropagateResult::FILTER_FALSE_OR_NULL : FilterPropagateResult::FILTER_ALWAYS_FALSE;
		} else {
			return FilterPropagateResult::NO_PRUNING_POSSIBLE;
		}
	case ExpressionType::COMPARE_GREATERTHAN:
		// l > r
		if (lstats.min > rstats.max) {
			// if l.min > r.max, it is always true ONLY if neither side contains nulls
			return has_null ? FilterPropagateResult::FILTER_TRUE_OR_NULL : FilterPropagateResult::FILTER_ALWAYS_TRUE;
		}
		// if r.min is bigger or equal to l.max, the filter is always false
		if (rstats.min >= lstats.max) {
			return has_null ? FilterPropagateResult::FILTER_FALSE_OR_NULL : FilterPropagateResult::FILTER_ALWAYS_FALSE;
		}
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		// l >= r
		if (lstats.min >= rstats.max) {
			// if l.min >= r.max, it is always true ONLY if neither side contains nulls
			return has_null ? FilterPropagateResult::FILTER_TRUE_OR_NULL : FilterPropagateResult::FILTER_ALWAYS_TRUE;
		}
		// if r.min > l.max, the filter is always false
		if (rstats.min > lstats.max) {
			return has_null ? FilterPropagateResult::FILTER_FALSE_OR_NULL : FilterPropagateResult::FILTER_ALWAYS_FALSE;
		}
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	case ExpressionType::COMPARE_LESSTHAN:
		// l < r
		if (lstats.max < rstats.min) {
			// if l.max < r.min, it is always true ONLY if neither side contains nulls
			return has_null ? FilterPropagateResult::FILTER_TRUE_OR_NULL : FilterPropagateResult::FILTER_ALWAYS_TRUE;
		}
		// if l.min >= rstats.max, the filter is always false
		if (lstats.min >= rstats.max) {
			return has_null ? FilterPropagateResult::FILTER_FALSE_OR_NULL : FilterPropagateResult::FILTER_ALWAYS_FALSE;
		}
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		// l <= r
		if (lstats.max <= rstats.min) {
			// if l.max <= r.min, it is always true ONLY if neither side contains nulls
			return has_null ? FilterPropagateResult::FILTER_TRUE_OR_NULL : FilterPropagateResult::FILTER_ALWAYS_TRUE;
		}
		// if l.min > rstats.max, the filter is always false
		if (lstats.min > rstats.max) {
			return has_null ? FilterPropagateResult::FILTER_FALSE_OR_NULL : FilterPropagateResult::FILTER_ALWAYS_FALSE;
		}
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	default:
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
}

unique_ptr<BaseStatistics> StatisticsPropagator::PropagateExpression(BoundComparisonExpression &expr,
                                                                     unique_ptr<Expression> *expr_ptr) {
	auto left_stats = PropagateExpression(expr.left);
	auto right_stats = PropagateExpression(expr.right);
	if (!left_stats || !right_stats) {
		return nullptr;
	}
	// propagate the statistics of the comparison operator
	auto propagate_result = PropagateComparison(*left_stats, *right_stats, expr.type);
	switch (propagate_result) {
	case FilterPropagateResult::FILTER_ALWAYS_TRUE:
		*expr_ptr = make_unique<BoundConstantExpression>(Value::BOOLEAN(true));
		return PropagateExpression(*expr_ptr);
	case FilterPropagateResult::FILTER_ALWAYS_FALSE:
		*expr_ptr = make_unique<BoundConstantExpression>(Value::BOOLEAN(false));
		return PropagateExpression(*expr_ptr);
	case FilterPropagateResult::FILTER_TRUE_OR_NULL: {
		vector<unique_ptr<Expression>> children;
		children.push_back(std::move(expr.left));
		children.push_back(std::move(expr.right));
		*expr_ptr = ExpressionRewriter::ConstantOrNull(std::move(children), Value::BOOLEAN(true));
		return nullptr;
	}
	case FilterPropagateResult::FILTER_FALSE_OR_NULL: {
		vector<unique_ptr<Expression>> children;
		children.push_back(std::move(expr.left));
		children.push_back(std::move(expr.right));
		*expr_ptr = ExpressionRewriter::ConstantOrNull(std::move(children), Value::BOOLEAN(false));
		return nullptr;
	}
	default:
		// FIXME: we can propagate nulls here, i.e. this expression will have nulls only if left and right has nulls
		return nullptr;
	}
}

} // namespace duckdb








namespace duckdb {

unique_ptr<BaseStatistics> StatisticsPropagator::PropagateExpression(BoundConjunctionExpression &expr,
                                                                     unique_ptr<Expression> *expr_ptr) {
	auto is_and = expr.type == ExpressionType::CONJUNCTION_AND;
	for (idx_t expr_idx = 0; expr_idx < expr.children.size(); expr_idx++) {
		auto &child = expr.children[expr_idx];
		auto stats = PropagateExpression(child);
		if (!child->IsFoldable()) {
			continue;
		}
		// we have a constant in a conjunction
		// we (1) either prune the child
		// or (2) replace the entire conjunction with a constant
		auto constant = ExpressionExecutor::EvaluateScalar(context, *child);
		if (constant.IsNull()) {
			continue;
		}
		auto b = BooleanValue::Get(constant);
		bool prune_child = false;
		bool constant_value = true;
		if (b) {
			// true
			if (is_and) {
				// true in and: prune child
				prune_child = true;
			} else {
				// true in OR: replace with TRUE
				constant_value = true;
			}
		} else {
			// false
			if (is_and) {
				// false in AND: replace with FALSE
				constant_value = false;
			} else {
				// false in OR: prune child
				prune_child = true;
			}
		}
		if (prune_child) {
			expr.children.erase(expr.children.begin() + expr_idx);
			expr_idx--;
			continue;
		}
		*expr_ptr = make_unique<BoundConstantExpression>(Value::BOOLEAN(constant_value));
		return PropagateExpression(*expr_ptr);
	}
	if (expr.children.empty()) {
		// if there are no children left, replace the conjunction with TRUE (for AND) or FALSE (for OR)
		*expr_ptr = make_unique<BoundConstantExpression>(Value::BOOLEAN(is_and));
		return PropagateExpression(*expr_ptr);
	} else if (expr.children.size() == 1) {
		// if there is one child left, replace the conjunction with that one child
		*expr_ptr = std::move(expr.children[0]);
	}
	return nullptr;
}

} // namespace duckdb








namespace duckdb {

void UpdateDistinctStats(BaseStatistics &distinct_stats, const Value &input) {
	Vector v(input);
	auto &d_stats = (DistinctStatistics &)distinct_stats;
	d_stats.Update(v, 1);
}

unique_ptr<BaseStatistics> StatisticsPropagator::StatisticsFromValue(const Value &input) {
	switch (input.type().InternalType()) {
	case PhysicalType::BOOL:
	case PhysicalType::UINT8:
	case PhysicalType::UINT16:
	case PhysicalType::UINT32:
	case PhysicalType::UINT64:
	case PhysicalType::INT8:
	case PhysicalType::INT16:
	case PhysicalType::INT32:
	case PhysicalType::INT64:
	case PhysicalType::INT128:
	case PhysicalType::FLOAT:
	case PhysicalType::DOUBLE: {
		auto result = make_unique<NumericStatistics>(input.type(), input, input, StatisticsType::GLOBAL_STATS);
		result->validity_stats = make_unique<ValidityStatistics>(input.IsNull(), !input.IsNull());
		UpdateDistinctStats(*result->distinct_stats, input);
		return std::move(result);
	}
	case PhysicalType::VARCHAR: {
		auto result = make_unique<StringStatistics>(input.type(), StatisticsType::GLOBAL_STATS);
		result->validity_stats = make_unique<ValidityStatistics>(input.IsNull(), !input.IsNull());
		UpdateDistinctStats(*result->distinct_stats, input);
		if (!input.IsNull()) {
			auto &string_value = StringValue::Get(input);
			result->Update(string_t(string_value));
		}
		return std::move(result);
	}
	case PhysicalType::STRUCT: {
		auto result = make_unique<StructStatistics>(input.type());
		result->validity_stats = make_unique<ValidityStatistics>(input.IsNull(), !input.IsNull());
		if (input.IsNull()) {
			for (auto &child_stat : result->child_stats) {
				child_stat.reset();
			}
		} else {
			auto &struct_children = StructValue::GetChildren(input);
			D_ASSERT(result->child_stats.size() == struct_children.size());
			for (idx_t i = 0; i < result->child_stats.size(); i++) {
				result->child_stats[i] = StatisticsFromValue(struct_children[i]);
			}
		}
		return std::move(result);
	}
	case PhysicalType::LIST: {
		auto result = make_unique<ListStatistics>(input.type());
		result->validity_stats = make_unique<ValidityStatistics>(input.IsNull(), !input.IsNull());
		if (input.IsNull()) {
			result->child_stats.reset();
		} else {
			auto &list_children = ListValue::GetChildren(input);
			for (auto &child_element : list_children) {
				auto child_element_stats = StatisticsFromValue(child_element);
				if (child_element_stats) {
					result->child_stats->Merge(*child_element_stats);
				} else {
					result->child_stats.reset();
				}
			}
		}
		return std::move(result);
	}
	default:
		return nullptr;
	}
}

unique_ptr<BaseStatistics> StatisticsPropagator::PropagateExpression(BoundConstantExpression &constant,
                                                                     unique_ptr<Expression> *expr_ptr) {
	return StatisticsFromValue(constant.value);
}

} // namespace duckdb



namespace duckdb {

unique_ptr<BaseStatistics> StatisticsPropagator::PropagateExpression(BoundFunctionExpression &func,
                                                                     unique_ptr<Expression> *expr_ptr) {
	vector<unique_ptr<BaseStatistics>> stats;
	stats.reserve(func.children.size());
	for (idx_t i = 0; i < func.children.size(); i++) {
		stats.push_back(PropagateExpression(func.children[i]));
	}
	if (!func.function.statistics) {
		return nullptr;
	}
	FunctionStatisticsInput input(func, func.bind_info.get(), stats, expr_ptr);
	return func.function.statistics(context, input);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<BaseStatistics> StatisticsPropagator::PropagateExpression(BoundOperatorExpression &expr,
                                                                     unique_ptr<Expression> *expr_ptr) {
	bool all_have_stats = true;
	vector<unique_ptr<BaseStatistics>> child_stats;
	child_stats.reserve(expr.children.size());
	for (auto &child : expr.children) {
		auto stats = PropagateExpression(child);
		if (!stats) {
			all_have_stats = false;
		}
		child_stats.push_back(std::move(stats));
	}
	if (!all_have_stats) {
		return nullptr;
	}
	switch (expr.type) {
	case ExpressionType::OPERATOR_COALESCE:
		// COALESCE, merge stats of all children
		for (idx_t i = 0; i < expr.children.size(); i++) {
			D_ASSERT(child_stats[i]);
			if (!child_stats[i]->CanHaveNoNull()) {
				// this child is always NULL, we can remove it from the coalesce
				// UNLESS there is only one node remaining
				if (expr.children.size() > 1) {
					expr.children.erase(expr.children.begin() + i);
					child_stats.erase(child_stats.begin() + i);
					i--;
				}
			} else if (!child_stats[i]->CanHaveNull()) {
				// coalesce child cannot have NULL entries
				// this is the last coalesce node that influences the result
				// we can erase any children after this node
				if (i + 1 < expr.children.size()) {
					expr.children.erase(expr.children.begin() + i + 1, expr.children.end());
					child_stats.erase(child_stats.begin() + i + 1, child_stats.end());
				}
				break;
			}
		}
		D_ASSERT(!expr.children.empty());
		D_ASSERT(expr.children.size() == child_stats.size());
		if (expr.children.size() == 1) {
			// coalesce of one entry: simply return that entry
			*expr_ptr = std::move(expr.children[0]);
		} else {
			// coalesce of multiple entries
			// merge the stats
			for (idx_t i = 1; i < expr.children.size(); i++) {
				child_stats[0]->Merge(*child_stats[i]);
			}
		}
		return std::move(child_stats[0]);
	case ExpressionType::OPERATOR_IS_NULL:
		if (!child_stats[0]->CanHaveNull()) {
			// child has no null values: x IS NULL will always be false
			*expr_ptr = make_unique<BoundConstantExpression>(Value::BOOLEAN(false));
			return PropagateExpression(*expr_ptr);
		}
		return nullptr;
	case ExpressionType::OPERATOR_IS_NOT_NULL:
		if (!child_stats[0]->CanHaveNull()) {
			// child has no null values: x IS NOT NULL will always be true
			*expr_ptr = make_unique<BoundConstantExpression>(Value::BOOLEAN(true));
			return PropagateExpression(*expr_ptr);
		}
		return nullptr;
	default:
		return nullptr;
	}
}

} // namespace duckdb




namespace duckdb {

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateStatistics(LogicalAggregate &aggr,
                                                                     unique_ptr<LogicalOperator> *node_ptr) {
	// first propagate statistics in the child node
	node_stats = PropagateStatistics(aggr.children[0]);

	// handle the groups: simply propagate statistics and assign the stats to the group binding
	aggr.group_stats.resize(aggr.groups.size());
	for (idx_t group_idx = 0; group_idx < aggr.groups.size(); group_idx++) {
		auto stats = PropagateExpression(aggr.groups[group_idx]);
		aggr.group_stats[group_idx] = stats ? stats->Copy() : nullptr;
		if (!stats) {
			continue;
		}
		if (aggr.grouping_sets.size() > 1) {
			// aggregates with multiple grouping sets can introduce NULL values to certain groups
			// FIXME: actually figure out WHICH groups can have null values introduced
			stats->validity_stats = make_unique<ValidityStatistics>(true, true);
			continue;
		}
		ColumnBinding group_binding(aggr.group_index, group_idx);
		statistics_map[group_binding] = std::move(stats);
	}
	// propagate statistics in the aggregates
	for (idx_t aggregate_idx = 0; aggregate_idx < aggr.expressions.size(); aggregate_idx++) {
		auto stats = PropagateExpression(aggr.expressions[aggregate_idx]);
		if (!stats) {
			continue;
		}
		ColumnBinding aggregate_binding(aggr.aggregate_index, aggregate_idx);
		statistics_map[aggregate_binding] = std::move(stats);
	}
	// the max cardinality of an aggregate is the max cardinality of the input (i.e. when every row is a unique group)
	return std::move(node_stats);
}

} // namespace duckdb



namespace duckdb {

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateStatistics(LogicalCrossProduct &cp,
                                                                     unique_ptr<LogicalOperator> *node_ptr) {
	// first propagate statistics in the child node
	auto left_stats = PropagateStatistics(cp.children[0]);
	auto right_stats = PropagateStatistics(cp.children[1]);
	if (!left_stats || !right_stats) {
		return nullptr;
	}
	MultiplyCardinalities(left_stats, *right_stats);
	return left_stats;
}

} // namespace duckdb









namespace duckdb {

static bool IsCompareDistinct(ExpressionType type) {
	return type == ExpressionType::COMPARE_DISTINCT_FROM || type == ExpressionType::COMPARE_NOT_DISTINCT_FROM;
}

bool StatisticsPropagator::ExpressionIsConstant(Expression &expr, const Value &val) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) {
		return false;
	}
	auto &bound_constant = (BoundConstantExpression &)expr;
	D_ASSERT(bound_constant.value.type() == val.type());
	return Value::NotDistinctFrom(bound_constant.value, val);
}

bool StatisticsPropagator::ExpressionIsConstantOrNull(Expression &expr, const Value &val) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
		return false;
	}
	auto &bound_function = (BoundFunctionExpression &)expr;
	return ConstantOrNull::IsConstantOrNull(bound_function, val);
}

void StatisticsPropagator::SetStatisticsNotNull(ColumnBinding binding) {
	auto entry = statistics_map.find(binding);
	if (entry == statistics_map.end()) {
		return;
	}
	entry->second->validity_stats = make_unique<ValidityStatistics>(false);
}

void StatisticsPropagator::UpdateFilterStatistics(BaseStatistics &stats, ExpressionType comparison_type,
                                                  const Value &constant) {
	// regular comparisons removes all null values
	if (!IsCompareDistinct(comparison_type)) {
		stats.validity_stats = make_unique<ValidityStatistics>(false);
	}
	if (!stats.type.IsNumeric()) {
		// don't handle non-numeric columns here (yet)
		return;
	}
	auto &numeric_stats = (NumericStatistics &)stats;
	if (numeric_stats.min.IsNull() || numeric_stats.max.IsNull()) {
		// no stats available: skip this
		return;
	}
	switch (comparison_type) {
	case ExpressionType::COMPARE_LESSTHAN:
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		// X < constant OR X <= constant
		// max becomes the constant
		numeric_stats.max = constant;
		break;
	case ExpressionType::COMPARE_GREATERTHAN:
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		// X > constant OR X >= constant
		// min becomes the constant
		numeric_stats.min = constant;
		break;
	case ExpressionType::COMPARE_EQUAL:
		// X = constant
		// both min and max become the constant
		numeric_stats.min = constant;
		numeric_stats.max = constant;
		break;
	default:
		break;
	}
}

void StatisticsPropagator::UpdateFilterStatistics(BaseStatistics &lstats, BaseStatistics &rstats,
                                                  ExpressionType comparison_type) {
	// regular comparisons removes all null values
	if (!IsCompareDistinct(comparison_type)) {
		lstats.validity_stats = make_unique<ValidityStatistics>(false);
		rstats.validity_stats = make_unique<ValidityStatistics>(false);
	}
	D_ASSERT(lstats.type == rstats.type);
	if (!lstats.type.IsNumeric()) {
		// don't handle non-numeric columns here (yet)
		return;
	}
	auto &left_stats = (NumericStatistics &)lstats;
	auto &right_stats = (NumericStatistics &)rstats;
	if (left_stats.min.IsNull() || left_stats.max.IsNull() || right_stats.min.IsNull() || right_stats.max.IsNull()) {
		// no stats available: skip this
		return;
	}
	switch (comparison_type) {
	case ExpressionType::COMPARE_LESSTHAN:
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
		// LEFT < RIGHT OR LEFT <= RIGHT
		// we know that every value of left is smaller (or equal to) every value in right
		// i.e. if we have left = [-50, 250] and right = [-100, 100]

		// we know that left.max is AT MOST equal to right.max
		// because any value in left that is BIGGER than right.max will not pass the filter
		if (left_stats.max > right_stats.max) {
			left_stats.max = right_stats.max;
		}

		// we also know that right.min is AT MOST equal to left.min
		// because any value in right that is SMALLER than left.min will not pass the filter
		if (right_stats.min < left_stats.min) {
			right_stats.min = left_stats.min;
		}
		// so in our example, the bounds get updated as follows:
		// left: [-50, 100], right: [-50, 100]
		break;
	case ExpressionType::COMPARE_GREATERTHAN:
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		// LEFT > RIGHT OR LEFT >= RIGHT
		// we know that every value of left is bigger (or equal to) every value in right
		// this is essentially the inverse of the less than (or equal to) scenario
		if (right_stats.max > left_stats.max) {
			right_stats.max = left_stats.max;
		}
		if (left_stats.min < right_stats.min) {
			left_stats.min = right_stats.min;
		}
		break;
	case ExpressionType::COMPARE_EQUAL:
	case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
		// LEFT = RIGHT
		// only the tightest bounds pass
		// so if we have e.g. left = [-50, 250] and right = [-100, 100]
		// the tighest bounds are [-50, 100]
		// select the highest min
		if (left_stats.min > right_stats.min) {
			right_stats.min = left_stats.min;
		} else {
			left_stats.min = right_stats.min;
		}
		// select the lowest max
		if (left_stats.max < right_stats.max) {
			right_stats.max = left_stats.max;
		} else {
			left_stats.max = right_stats.max;
		}
		break;
	default:
		break;
	}
}

void StatisticsPropagator::UpdateFilterStatistics(Expression &left, Expression &right, ExpressionType comparison_type) {
	// first check if either side is a bound column ref
	// any column ref involved in a comparison will not be null after the comparison
	bool compare_distinct = IsCompareDistinct(comparison_type);
	if (!compare_distinct && left.type == ExpressionType::BOUND_COLUMN_REF) {
		SetStatisticsNotNull(((BoundColumnRefExpression &)left).binding);
	}
	if (!compare_distinct && right.type == ExpressionType::BOUND_COLUMN_REF) {
		SetStatisticsNotNull(((BoundColumnRefExpression &)right).binding);
	}
	// check if this is a comparison between a constant and a column ref
	BoundConstantExpression *constant = nullptr;
	BoundColumnRefExpression *columnref = nullptr;
	if (left.type == ExpressionType::VALUE_CONSTANT && right.type == ExpressionType::BOUND_COLUMN_REF) {
		constant = (BoundConstantExpression *)&left;
		columnref = (BoundColumnRefExpression *)&right;
		comparison_type = FlipComparisionExpression(comparison_type);
	} else if (left.type == ExpressionType::BOUND_COLUMN_REF && right.type == ExpressionType::VALUE_CONSTANT) {
		columnref = (BoundColumnRefExpression *)&left;
		constant = (BoundConstantExpression *)&right;
	} else if (left.type == ExpressionType::BOUND_COLUMN_REF && right.type == ExpressionType::BOUND_COLUMN_REF) {
		// comparison between two column refs
		auto &left_column_ref = (BoundColumnRefExpression &)left;
		auto &right_column_ref = (BoundColumnRefExpression &)right;
		auto lentry = statistics_map.find(left_column_ref.binding);
		auto rentry = statistics_map.find(right_column_ref.binding);
		if (lentry == statistics_map.end() || rentry == statistics_map.end()) {
			return;
		}
		UpdateFilterStatistics(*lentry->second, *rentry->second, comparison_type);
	} else {
		// unsupported filter
		return;
	}
	if (constant && columnref) {
		// comparison between columnref
		auto entry = statistics_map.find(columnref->binding);
		if (entry == statistics_map.end()) {
			return;
		}
		UpdateFilterStatistics(*entry->second, comparison_type, constant->value);
	}
}

void StatisticsPropagator::UpdateFilterStatistics(Expression &condition) {
	// in filters, we check for constant comparisons with bound columns
	// if we find a comparison in the form of e.g. "i=3", we can update our statistics for that column
	switch (condition.GetExpressionClass()) {
	case ExpressionClass::BOUND_BETWEEN: {
		auto &between = (BoundBetweenExpression &)condition;
		UpdateFilterStatistics(*between.input, *between.lower, between.LowerComparisonType());
		UpdateFilterStatistics(*between.input, *between.upper, between.UpperComparisonType());
		break;
	}
	case ExpressionClass::BOUND_COMPARISON: {
		auto &comparison = (BoundComparisonExpression &)condition;
		UpdateFilterStatistics(*comparison.left, *comparison.right, comparison.type);
		break;
	}
	default:
		break;
	}
}

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateStatistics(LogicalFilter &filter,
                                                                     unique_ptr<LogicalOperator> *node_ptr) {
	// first propagate to the child
	node_stats = PropagateStatistics(filter.children[0]);
	if (filter.children[0]->type == LogicalOperatorType::LOGICAL_EMPTY_RESULT) {
		ReplaceWithEmptyResult(*node_ptr);
		return make_unique<NodeStatistics>(0, 0);
	}

	// then propagate to each of the expressions
	for (idx_t i = 0; i < filter.expressions.size(); i++) {
		auto &condition = filter.expressions[i];
		PropagateExpression(condition);

		if (ExpressionIsConstant(*condition, Value::BOOLEAN(true))) {
			// filter is always true; it is useless to execute it
			// erase this condition
			filter.expressions.erase(filter.expressions.begin() + i);
			i--;
			if (filter.expressions.empty()) {
				// all conditions have been erased: remove the entire filter
				*node_ptr = std::move(filter.children[0]);
				break;
			}
		} else if (ExpressionIsConstant(*condition, Value::BOOLEAN(false)) ||
		           ExpressionIsConstantOrNull(*condition, Value::BOOLEAN(false))) {
			// filter is always false or null; this entire filter should be replaced by an empty result block
			ReplaceWithEmptyResult(*node_ptr);
			return make_unique<NodeStatistics>(0, 0);
		} else {
			// cannot prune this filter: propagate statistics from the filter
			UpdateFilterStatistics(*condition);
		}
	}
	// the max cardinality of a filter is the cardinality of the input (i.e. no tuples get filtered)
	return std::move(node_stats);
}

} // namespace duckdb






namespace duckdb {

FilterPropagateResult StatisticsPropagator::PropagateTableFilter(BaseStatistics &stats, TableFilter &filter) {
	return filter.CheckStatistics(stats);
}

void StatisticsPropagator::UpdateFilterStatistics(BaseStatistics &input, TableFilter &filter) {
	// FIXME: update stats...
	switch (filter.filter_type) {
	case TableFilterType::CONJUNCTION_AND: {
		auto &conjunction_and = (ConjunctionAndFilter &)filter;
		for (auto &child_filter : conjunction_and.child_filters) {
			UpdateFilterStatistics(input, *child_filter);
		}
		break;
	}
	case TableFilterType::CONSTANT_COMPARISON: {
		auto &constant_filter = (ConstantFilter &)filter;
		UpdateFilterStatistics(input, constant_filter.comparison_type, constant_filter.constant);
		break;
	}
	default:
		break;
	}
}

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateStatistics(LogicalGet &get,
                                                                     unique_ptr<LogicalOperator> *node_ptr) {
	if (get.function.cardinality) {
		node_stats = get.function.cardinality(context, get.bind_data.get());
	}
	if (!get.function.statistics) {
		// no column statistics to get
		return std::move(node_stats);
	}
	for (idx_t i = 0; i < get.column_ids.size(); i++) {
		auto stats = get.function.statistics(context, get.bind_data.get(), get.column_ids[i]);
		if (stats) {
			ColumnBinding binding(get.table_index, i);
			statistics_map.insert(make_pair(binding, std::move(stats)));
		}
	}
	// push table filters into the statistics
	vector<idx_t> column_indexes;
	column_indexes.reserve(get.table_filters.filters.size());
	for (auto &kv : get.table_filters.filters) {
		column_indexes.push_back(kv.first);
	}

	for (auto &table_filter_column : column_indexes) {
		idx_t column_index;
		for (column_index = 0; column_index < get.column_ids.size(); column_index++) {
			if (get.column_ids[column_index] == table_filter_column) {
				break;
			}
		}
		D_ASSERT(column_index < get.column_ids.size());
		D_ASSERT(get.column_ids[column_index] == table_filter_column);

		// find the stats
		ColumnBinding stats_binding(get.table_index, column_index);
		auto entry = statistics_map.find(stats_binding);
		if (entry == statistics_map.end()) {
			// no stats for this entry
			continue;
		}
		auto &stats = *entry->second;

		// fetch the table filter
		D_ASSERT(get.table_filters.filters.count(table_filter_column) > 0);
		auto &filter = get.table_filters.filters[table_filter_column];
		auto propagate_result = PropagateTableFilter(stats, *filter);
		switch (propagate_result) {
		case FilterPropagateResult::FILTER_ALWAYS_TRUE:
			// filter is always true; it is useless to execute it
			// erase this condition
			get.table_filters.filters.erase(table_filter_column);
			break;
		case FilterPropagateResult::FILTER_FALSE_OR_NULL:
		case FilterPropagateResult::FILTER_ALWAYS_FALSE:
			// filter is always false; this entire filter should be replaced by an empty result block
			ReplaceWithEmptyResult(*node_ptr);
			return make_unique<NodeStatistics>(0, 0);
		default:
			// general case: filter can be true or false, update this columns' statistics
			UpdateFilterStatistics(stats, *filter);
			break;
		}
	}
	return std::move(node_stats);
}

} // namespace duckdb











namespace duckdb {

void StatisticsPropagator::PropagateStatistics(LogicalComparisonJoin &join, unique_ptr<LogicalOperator> *node_ptr) {
	for (idx_t i = 0; i < join.conditions.size(); i++) {
		auto &condition = join.conditions[i];
		auto stats_left = PropagateExpression(condition.left);
		auto stats_right = PropagateExpression(condition.right);
		if (stats_left && stats_right) {
			if ((condition.comparison == ExpressionType::COMPARE_DISTINCT_FROM ||
			     condition.comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) &&
			    stats_left->CanHaveNull() && stats_right->CanHaveNull()) {
				// null values are equal in this join, and both sides can have null values
				// nothing to do here
				continue;
			}
			auto prune_result = PropagateComparison(*stats_left, *stats_right, condition.comparison);
			// Add stats to logical_join for perfect hash join
			join.join_stats.push_back(std::move(stats_left));
			join.join_stats.push_back(std::move(stats_right));
			switch (prune_result) {
			case FilterPropagateResult::FILTER_FALSE_OR_NULL:
			case FilterPropagateResult::FILTER_ALWAYS_FALSE:
				// filter is always false or null, none of the join conditions matter
				switch (join.join_type) {
				case JoinType::SEMI:
				case JoinType::INNER:
					// semi or inner join on false; entire node can be pruned
					ReplaceWithEmptyResult(*node_ptr);
					return;
				case JoinType::ANTI: {
					// when the right child has data, return the left child
					// when the right child has no data, return an empty set
					auto limit = make_unique<LogicalLimit>(1, 0, nullptr, nullptr);
					limit->AddChild(std::move(join.children[1]));
					auto cross_product = LogicalCrossProduct::Create(std::move(join.children[0]), std::move(limit));
					*node_ptr = std::move(cross_product);
					return;
				}
				case JoinType::LEFT:
					// anti/left outer join: replace right side with empty node
					ReplaceWithEmptyResult(join.children[1]);
					return;
				case JoinType::RIGHT:
					// right outer join: replace left side with empty node
					ReplaceWithEmptyResult(join.children[0]);
					return;
				default:
					// other join types: can't do much meaningful with this information
					// full outer join requires both sides anyway; we can skip the execution of the actual join, but eh
					// mark/single join requires knowing if the rhs has null values or not
					break;
				}
				break;
			case FilterPropagateResult::FILTER_ALWAYS_TRUE:
				// filter is always true
				if (join.conditions.size() > 1) {
					// there are multiple conditions: erase this condition
					join.conditions.erase(join.conditions.begin() + i);
					// remove the corresponding statistics
					join.join_stats.clear();
					i--;
					continue;
				} else {
					// this is the only condition and it is always true: all conditions are true
					switch (join.join_type) {
					case JoinType::SEMI: {
						// when the right child has data, return the left child
						// when the right child has no data, return an empty set
						auto limit = make_unique<LogicalLimit>(1, 0, nullptr, nullptr);
						limit->AddChild(std::move(join.children[1]));
						auto cross_product = LogicalCrossProduct::Create(std::move(join.children[0]), std::move(limit));
						*node_ptr = std::move(cross_product);
						return;
					}
					case JoinType::INNER:
					case JoinType::LEFT:
					case JoinType::RIGHT:
					case JoinType::OUTER: {
						// inner/left/right/full outer join, replace with cross product
						// since the condition is always true, left/right/outer join are equivalent to inner join here
						auto cross_product =
						    LogicalCrossProduct::Create(std::move(join.children[0]), std::move(join.children[1]));
						*node_ptr = std::move(cross_product);
						return;
					}
					case JoinType::ANTI:
						// anti join on true: empty result
						ReplaceWithEmptyResult(*node_ptr);
						return;
					default:
						// we don't handle mark/single join here yet
						break;
					}
				}
				break;
			default:
				break;
			}
		}
		// after we have propagated, we can update the statistics on both sides
		// note that it is fine to do this now, even if the same column is used again later
		// e.g. if we have i=j AND i=k, and the stats for j and k are disjoint, we know there are no results
		// so if we have e.g. i: [0, 100], j: [0, 25], k: [75, 100]
		// we can set i: [0, 25] after the first comparison, and statically determine that the second comparison is fals

		// note that we can't update statistics the same for all join types
		// mark and single joins don't filter any tuples -> so there is no propagation possible
		// anti joins have inverse statistics propagation
		// (i.e. if we have an anti join on i: [0, 100] and j: [0, 25], the resulting stats are i:[25,100])
		// for now we don't handle anti joins
		if (condition.comparison == ExpressionType::COMPARE_DISTINCT_FROM ||
		    condition.comparison == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
			// skip update when null values are equal (for now?)
			continue;
		}
		switch (join.join_type) {
		case JoinType::INNER:
		case JoinType::SEMI: {
			UpdateFilterStatistics(*condition.left, *condition.right, condition.comparison);
			auto stats_left = PropagateExpression(condition.left);
			auto stats_right = PropagateExpression(condition.right);
			// Update join_stats when is already part of the join
			if (join.join_stats.size() == 2) {
				join.join_stats[0] = std::move(stats_left);
				join.join_stats[1] = std::move(stats_right);
			}
			break;
		}
		default:
			break;
		}
	}
}

void StatisticsPropagator::PropagateStatistics(LogicalAnyJoin &join, unique_ptr<LogicalOperator> *node_ptr) {
	// propagate the expression into the join condition
	PropagateExpression(join.condition);
}

void StatisticsPropagator::MultiplyCardinalities(unique_ptr<NodeStatistics> &stats, NodeStatistics &new_stats) {
	if (!stats->has_estimated_cardinality || !new_stats.has_estimated_cardinality || !stats->has_max_cardinality ||
	    !new_stats.has_max_cardinality) {
		stats = nullptr;
		return;
	}
	stats->estimated_cardinality = MaxValue<idx_t>(stats->estimated_cardinality, new_stats.estimated_cardinality);
	auto new_max = Hugeint::Multiply(stats->max_cardinality, new_stats.max_cardinality);
	if (new_max < NumericLimits<int64_t>::Maximum()) {
		int64_t result;
		if (!Hugeint::TryCast<int64_t>(new_max, result)) {
			throw InternalException("Overflow in cast in statistics propagation");
		}
		D_ASSERT(result >= 0);
		stats->max_cardinality = idx_t(result);
	} else {
		stats = nullptr;
	}
}

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateStatistics(LogicalJoin &join,
                                                                     unique_ptr<LogicalOperator> *node_ptr) {
	// first propagate through the children of the join
	node_stats = PropagateStatistics(join.children[0]);
	for (idx_t child_idx = 1; child_idx < join.children.size(); child_idx++) {
		auto child_stats = PropagateStatistics(join.children[child_idx]);
		if (!child_stats) {
			node_stats = nullptr;
		} else if (node_stats) {
			MultiplyCardinalities(node_stats, *child_stats);
		}
	}

	auto join_type = join.join_type;
	// depending on the join type, we might need to alter the statistics
	// LEFT, FULL, RIGHT OUTER and SINGLE joins can introduce null values
	// this requires us to alter the statistics after this point in the query plan
	bool adds_null_on_left = IsRightOuterJoin(join_type);
	bool adds_null_on_right = IsLeftOuterJoin(join_type) || join_type == JoinType::SINGLE;

	vector<ColumnBinding> left_bindings, right_bindings;
	if (adds_null_on_left) {
		left_bindings = join.children[0]->GetColumnBindings();
	}
	if (adds_null_on_right) {
		right_bindings = join.children[1]->GetColumnBindings();
	}

	// then propagate into the join conditions
	switch (join.type) {
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		PropagateStatistics((LogicalComparisonJoin &)join, node_ptr);
		break;
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
		PropagateStatistics((LogicalAnyJoin &)join, node_ptr);
		break;
	default:
		break;
	}

	if (adds_null_on_right) {
		// left or full outer join: set IsNull() to true for all rhs statistics
		for (auto &binding : right_bindings) {
			auto stats = statistics_map.find(binding);
			if (stats != statistics_map.end()) {
				stats->second->validity_stats = make_unique<ValidityStatistics>(true);
			}
		}
	}
	if (adds_null_on_left) {
		// right or full outer join: set IsNull() to true for all lhs statistics
		for (auto &binding : left_bindings) {
			auto stats = statistics_map.find(binding);
			if (stats != statistics_map.end()) {
				stats->second->validity_stats = make_unique<ValidityStatistics>(true);
			}
		}
	}
	return std::move(node_stats);
}

static void MaxCardinalities(unique_ptr<NodeStatistics> &stats, NodeStatistics &new_stats) {
	if (!stats->has_estimated_cardinality || !new_stats.has_estimated_cardinality || !stats->has_max_cardinality ||
	    !new_stats.has_max_cardinality) {
		stats = nullptr;
		return;
	}
	stats->estimated_cardinality = MaxValue<idx_t>(stats->estimated_cardinality, new_stats.estimated_cardinality);
	stats->max_cardinality = MaxValue<idx_t>(stats->max_cardinality, new_stats.max_cardinality);
}

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateStatistics(LogicalPositionalJoin &join,
                                                                     unique_ptr<LogicalOperator> *node_ptr) {
	D_ASSERT(join.type == LogicalOperatorType::LOGICAL_POSITIONAL_JOIN);

	// first propagate through the children of the join
	node_stats = PropagateStatistics(join.children[0]);
	for (idx_t child_idx = 1; child_idx < join.children.size(); child_idx++) {
		auto child_stats = PropagateStatistics(join.children[child_idx]);
		if (!child_stats) {
			node_stats = nullptr;
		} else if (node_stats) {
			if (!node_stats->has_estimated_cardinality || !child_stats->has_estimated_cardinality ||
			    !node_stats->has_max_cardinality || !child_stats->has_max_cardinality) {
				node_stats = nullptr;
			} else {
				MaxCardinalities(node_stats, *child_stats);
			}
		}
	}

	// No conditions.

	// Positional Joins are always FULL OUTER

	// set IsNull() to true for all lhs statistics
	auto left_bindings = join.children[0]->GetColumnBindings();
	for (auto &binding : left_bindings) {
		auto stats = statistics_map.find(binding);
		if (stats != statistics_map.end()) {
			stats->second->validity_stats = make_unique<ValidityStatistics>(true);
		}
	}

	// set IsNull() to true for all rhs statistics
	auto right_bindings = join.children[1]->GetColumnBindings();
	for (auto &binding : right_bindings) {
		auto stats = statistics_map.find(binding);
		if (stats != statistics_map.end()) {
			stats->second->validity_stats = make_unique<ValidityStatistics>(true);
		}
	}

	return std::move(node_stats);
}

} // namespace duckdb



namespace duckdb {

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateStatistics(LogicalLimit &limit,
                                                                     unique_ptr<LogicalOperator> *node_ptr) {
	// propagate statistics in the child node
	PropagateStatistics(limit.children[0]);
	// return the node stats, with as expected cardinality the amount specified in the limit
	return make_unique<NodeStatistics>(limit.limit_val, limit.limit_val);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateStatistics(LogicalOrder &order,
                                                                     unique_ptr<LogicalOperator> *node_ptr) {
	// first propagate to the child
	node_stats = PropagateStatistics(order.children[0]);

	// then propagate to each of the order expressions
	for (auto &bound_order : order.orders) {
		PropagateAndCompress(bound_order.expression, bound_order.stats);
	}
	return std::move(node_stats);
}

} // namespace duckdb



namespace duckdb {

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateStatistics(LogicalProjection &proj,
                                                                     unique_ptr<LogicalOperator> *node_ptr) {
	// first propagate to the child
	node_stats = PropagateStatistics(proj.children[0]);
	if (proj.children[0]->type == LogicalOperatorType::LOGICAL_EMPTY_RESULT) {
		ReplaceWithEmptyResult(*node_ptr);
		return std::move(node_stats);
	}

	// then propagate to each of the expressions
	for (idx_t i = 0; i < proj.expressions.size(); i++) {
		auto stats = PropagateExpression(proj.expressions[i]);
		if (stats) {
			ColumnBinding binding(proj.table_index, i);
			statistics_map.insert(make_pair(binding, std::move(stats)));
		}
	}
	return std::move(node_stats);
}

} // namespace duckdb



namespace duckdb {

void StatisticsPropagator::AddCardinalities(unique_ptr<NodeStatistics> &stats, NodeStatistics &new_stats) {
	if (!stats->has_estimated_cardinality || !new_stats.has_estimated_cardinality || !stats->has_max_cardinality ||
	    !new_stats.has_max_cardinality) {
		stats = nullptr;
		return;
	}
	stats->estimated_cardinality += new_stats.estimated_cardinality;
	auto new_max = Hugeint::Add(stats->max_cardinality, new_stats.max_cardinality);
	if (new_max < NumericLimits<int64_t>::Maximum()) {
		int64_t result;
		if (!Hugeint::TryCast<int64_t>(new_max, result)) {
			throw InternalException("Overflow in cast in statistics propagation");
		}
		D_ASSERT(result >= 0);
		stats->max_cardinality = idx_t(result);
	} else {
		stats = nullptr;
	}
}

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateStatistics(LogicalSetOperation &setop,
                                                                     unique_ptr<LogicalOperator> *node_ptr) {
	// first propagate statistics in the child nodes
	auto left_stats = PropagateStatistics(setop.children[0]);
	auto right_stats = PropagateStatistics(setop.children[1]);

	// now fetch the column bindings on both sides
	auto left_bindings = setop.children[0]->GetColumnBindings();
	auto right_bindings = setop.children[1]->GetColumnBindings();

	D_ASSERT(left_bindings.size() == right_bindings.size());
	D_ASSERT(left_bindings.size() == setop.column_count);
	for (idx_t i = 0; i < setop.column_count; i++) {
		// for each column binding, we fetch the statistics from both the lhs and the rhs
		auto left_entry = statistics_map.find(left_bindings[i]);
		auto right_entry = statistics_map.find(right_bindings[i]);
		if (left_entry == statistics_map.end() || right_entry == statistics_map.end()) {
			// no statistics on one of the sides: can't propagate stats
			continue;
		}
		unique_ptr<BaseStatistics> new_stats;
		switch (setop.type) {
		case LogicalOperatorType::LOGICAL_UNION:
			// union: merge the stats of the LHS and RHS together
			new_stats = left_entry->second->Copy();
			new_stats->Merge(*right_entry->second);
			break;
		case LogicalOperatorType::LOGICAL_EXCEPT:
			// except: use the stats of the LHS
			new_stats = left_entry->second->Copy();
			break;
		case LogicalOperatorType::LOGICAL_INTERSECT:
			// intersect: intersect the two stats
			// FIXME: for now we just use the stats of the LHS, as this is correct
			// however, the stats can be further refined to the minimal subset of the LHS and RHS
			new_stats = left_entry->second->Copy();
			break;
		default:
			throw InternalException("Unsupported setop type");
		}
		ColumnBinding binding(setop.table_index, i);
		statistics_map[binding] = std::move(new_stats);
	}
	if (!left_stats || !right_stats) {
		return nullptr;
	}
	if (setop.type == LogicalOperatorType::LOGICAL_UNION) {
		AddCardinalities(left_stats, *right_stats);
	}
	return left_stats;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateStatistics(LogicalWindow &window,
                                                                     unique_ptr<LogicalOperator> *node_ptr) {
	// first propagate to the child
	node_stats = PropagateStatistics(window.children[0]);

	// then propagate to each of the order expressions
	for (auto &window_expr : window.expressions) {
		auto over_expr = reinterpret_cast<BoundWindowExpression *>(window_expr.get());
		for (auto &expr : over_expr->partitions) {
			over_expr->partitions_stats.push_back(PropagateExpression(expr));
		}
		for (auto &bound_order : over_expr->orders) {
			bound_order.stats = PropagateExpression(bound_order.expression);
		}
	}
	return std::move(node_stats);
}

} // namespace duckdb







namespace duckdb {

StatisticsPropagator::StatisticsPropagator(ClientContext &context) : context(context) {
}

void StatisticsPropagator::ReplaceWithEmptyResult(unique_ptr<LogicalOperator> &node) {
	node = make_unique<LogicalEmptyResult>(std::move(node));
}

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateChildren(LogicalOperator &node,
                                                                   unique_ptr<LogicalOperator> *node_ptr) {
	for (idx_t child_idx = 0; child_idx < node.children.size(); child_idx++) {
		PropagateStatistics(node.children[child_idx]);
	}
	return nullptr;
}

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateStatistics(LogicalOperator &node,
                                                                     unique_ptr<LogicalOperator> *node_ptr) {
	switch (node.type) {
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		return PropagateStatistics((LogicalAggregate &)node, node_ptr);
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
		return PropagateStatistics((LogicalCrossProduct &)node, node_ptr);
	case LogicalOperatorType::LOGICAL_FILTER:
		return PropagateStatistics((LogicalFilter &)node, node_ptr);
	case LogicalOperatorType::LOGICAL_GET:
		return PropagateStatistics((LogicalGet &)node, node_ptr);
	case LogicalOperatorType::LOGICAL_PROJECTION:
		return PropagateStatistics((LogicalProjection &)node, node_ptr);
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
	case LogicalOperatorType::LOGICAL_JOIN:
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		return PropagateStatistics((LogicalJoin &)node, node_ptr);
	case LogicalOperatorType::LOGICAL_POSITIONAL_JOIN:
		return PropagateStatistics((LogicalPositionalJoin &)node, node_ptr);
	case LogicalOperatorType::LOGICAL_UNION:
	case LogicalOperatorType::LOGICAL_EXCEPT:
	case LogicalOperatorType::LOGICAL_INTERSECT:
		return PropagateStatistics((LogicalSetOperation &)node, node_ptr);
	case LogicalOperatorType::LOGICAL_ORDER_BY:
		return PropagateStatistics((LogicalOrder &)node, node_ptr);
	case LogicalOperatorType::LOGICAL_WINDOW:
		return PropagateStatistics((LogicalWindow &)node, node_ptr);
	default:
		return PropagateChildren(node, node_ptr);
	}
}

unique_ptr<NodeStatistics> StatisticsPropagator::PropagateStatistics(unique_ptr<LogicalOperator> &node_ptr) {
	return PropagateStatistics(*node_ptr, &node_ptr);
}

unique_ptr<BaseStatistics> StatisticsPropagator::PropagateExpression(Expression &expr,
                                                                     unique_ptr<Expression> *expr_ptr) {
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::BOUND_AGGREGATE:
		return PropagateExpression((BoundAggregateExpression &)expr, expr_ptr);
	case ExpressionClass::BOUND_BETWEEN:
		return PropagateExpression((BoundBetweenExpression &)expr, expr_ptr);
	case ExpressionClass::BOUND_CASE:
		return PropagateExpression((BoundCaseExpression &)expr, expr_ptr);
	case ExpressionClass::BOUND_CONJUNCTION:
		return PropagateExpression((BoundConjunctionExpression &)expr, expr_ptr);
	case ExpressionClass::BOUND_FUNCTION:
		return PropagateExpression((BoundFunctionExpression &)expr, expr_ptr);
	case ExpressionClass::BOUND_CAST:
		return PropagateExpression((BoundCastExpression &)expr, expr_ptr);
	case ExpressionClass::BOUND_COMPARISON:
		return PropagateExpression((BoundComparisonExpression &)expr, expr_ptr);
	case ExpressionClass::BOUND_CONSTANT:
		return PropagateExpression((BoundConstantExpression &)expr, expr_ptr);
	case ExpressionClass::BOUND_COLUMN_REF:
		return PropagateExpression((BoundColumnRefExpression &)expr, expr_ptr);
	case ExpressionClass::BOUND_OPERATOR:
		return PropagateExpression((BoundOperatorExpression &)expr, expr_ptr);
	default:
		break;
	}
	ExpressionIterator::EnumerateChildren(expr, [&](unique_ptr<Expression> &child) { PropagateExpression(child); });
	return nullptr;
}

unique_ptr<BaseStatistics> StatisticsPropagator::PropagateExpression(unique_ptr<Expression> &expr) {
	auto stats = PropagateExpression(*expr, &expr);
	if (ClientConfig::GetConfig(context).query_verification_enabled && stats) {
		expr->verification_stats = stats->Copy();
	}
	return stats;
}

} // namespace duckdb






namespace duckdb {

unique_ptr<LogicalOperator> TopN::Optimize(unique_ptr<LogicalOperator> op) {
	if (op->type == LogicalOperatorType::LOGICAL_LIMIT &&
	    op->children[0]->type == LogicalOperatorType::LOGICAL_ORDER_BY) {
		auto &limit = (LogicalLimit &)*op;
		auto &order_by = (LogicalOrder &)*(op->children[0]);

		// This optimization doesn't apply when OFFSET is present without LIMIT
		// Or if offset is not constant
		if (limit.limit_val != NumericLimits<int64_t>::Maximum() || limit.offset) {
			auto topn = make_unique<LogicalTopN>(std::move(order_by.orders), limit.limit_val, limit.offset_val);
			topn->AddChild(std::move(order_by.children[0]));
			op = std::move(topn);
		}
	} else {
		for (auto &child : op->children) {
			child = Optimize(std::move(child));
		}
	}
	return op;
}

} // namespace duckdb











namespace duckdb {

void UnnestRewriterPlanUpdater::VisitOperator(LogicalOperator &op) {
	VisitOperatorChildren(op);
	VisitOperatorExpressions(op);
}

void UnnestRewriterPlanUpdater::VisitExpression(unique_ptr<Expression> *expression) {

	auto &expr = *expression;

	if (expr->expression_class == ExpressionClass::BOUND_COLUMN_REF) {

		auto &bound_column_ref = (BoundColumnRefExpression &)*expr;
		for (idx_t i = 0; i < replace_bindings.size(); i++) {
			if (bound_column_ref.binding == replace_bindings[i].old_binding) {
				bound_column_ref.binding = replace_bindings[i].new_binding;
			}
			// previously pointing to the LOGICAL_DELIM_GET
			if (bound_column_ref.binding.table_index == replace_bindings[i].old_binding.table_index &&
			    replace_bindings[i].old_binding.column_index == DConstants::INVALID_INDEX) {
				bound_column_ref.binding = replace_bindings[i].new_binding;
			}
		}
	}

	VisitExpressionChildren(**expression);
}

unique_ptr<LogicalOperator> UnnestRewriter::Optimize(unique_ptr<LogicalOperator> op) {

	UnnestRewriterPlanUpdater updater;
	vector<unique_ptr<LogicalOperator> *> candidates;
	FindCandidates(&op, candidates);

	// rewrite the plan and update the bindings
	for (auto &candidate : candidates) {

		// rearrange the logical operators
		if (RewriteCandidate(candidate)) {
			// update the bindings of the BOUND_UNNEST expression
			UpdateBoundUnnestBindings(updater, candidate);
			// update the sequence of LOGICAL_PROJECTION(s)
			UpdateRHSBindings(&op, candidate, updater);
			// reset
			delim_columns.clear();
			lhs_bindings.clear();
		}
	}

	return op;
}

void UnnestRewriter::FindCandidates(unique_ptr<LogicalOperator> *op_ptr,
                                    vector<unique_ptr<LogicalOperator> *> &candidates) {
	auto op = op_ptr->get();
	// search children before adding, so that we add candidates bottom-up
	for (auto &child : op->children) {
		FindCandidates(&child, candidates);
	}

	// search for operator that has a LOGICAL_DELIM_JOIN as its child
	if (op->children.size() != 1) {
		return;
	}
	if (op->children[0]->type != LogicalOperatorType::LOGICAL_DELIM_JOIN) {
		return;
	}

	// found a delim join
	auto &delim_join = (LogicalDelimJoin &)*op->children[0];
	// only support INNER delim joins
	if (delim_join.join_type != JoinType::INNER) {
		return;
	}
	// INNER delim join must have exactly one condition
	if (delim_join.conditions.size() != 1) {
		return;
	}

	// LHS child is a window
	if (delim_join.children[0]->type != LogicalOperatorType::LOGICAL_WINDOW) {
		return;
	}

	// RHS child must be projection(s) followed by an UNNEST
	auto curr_op = &delim_join.children[1];
	while (curr_op->get()->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		if (curr_op->get()->children.size() != 1) {
			break;
		}
		curr_op = &curr_op->get()->children[0];
	}

	if (curr_op->get()->type == LogicalOperatorType::LOGICAL_UNNEST) {
		candidates.push_back(op_ptr);
	}
	return;
}

bool UnnestRewriter::RewriteCandidate(unique_ptr<LogicalOperator> *candidate) {

	auto &topmost_op = (LogicalOperator &)**candidate;
	if (topmost_op.type != LogicalOperatorType::LOGICAL_PROJECTION &&
	    topmost_op.type != LogicalOperatorType::LOGICAL_WINDOW &&
	    topmost_op.type != LogicalOperatorType::LOGICAL_FILTER &&
	    topmost_op.type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY &&
	    topmost_op.type != LogicalOperatorType::LOGICAL_UNNEST) {
		return false;
	}

	// get the LOGICAL_DELIM_JOIN, which is a child of the candidate
	D_ASSERT(topmost_op.children.size() == 1);
	auto &delim_join = *(topmost_op.children[0]);
	D_ASSERT(delim_join.type == LogicalOperatorType::LOGICAL_DELIM_JOIN);
	GetDelimColumns(delim_join);

	// LHS of the LOGICAL_DELIM_JOIN is a LOGICAL_WINDOW that contains a LOGICAL_PROJECTION
	// this lhs_proj later becomes the child of the UNNEST
	auto &window = *delim_join.children[0];
	auto &lhs_op = window.children[0];
	GetLHSExpressions(*lhs_op);

	// find the LOGICAL_UNNEST
	// and get the path down to the LOGICAL_UNNEST
	vector<unique_ptr<LogicalOperator> *> path_to_unnest;
	auto curr_op = &(delim_join.children[1]);
	while (curr_op->get()->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		path_to_unnest.push_back(curr_op);
		curr_op = &curr_op->get()->children[0];
	}

	// store the table index of the child of the LOGICAL_UNNEST
	// then update the plan by making the lhs_proj the child of the LOGICAL_UNNEST
	D_ASSERT(curr_op->get()->type == LogicalOperatorType::LOGICAL_UNNEST);
	auto &unnest = (LogicalUnnest &)*curr_op->get();
	D_ASSERT(unnest.children[0]->type == LogicalOperatorType::LOGICAL_DELIM_GET);
	overwritten_tbl_idx = ((LogicalDelimGet &)*unnest.children[0]).table_index;
	unnest.children[0] = std::move(lhs_op);

	// replace the LOGICAL_DELIM_JOIN with its RHS child operator
	topmost_op.children[0] = std::move(*path_to_unnest.front());
	return true;
}

void UnnestRewriter::UpdateRHSBindings(unique_ptr<LogicalOperator> *plan_ptr, unique_ptr<LogicalOperator> *candidate,
                                       UnnestRewriterPlanUpdater &updater) {

	auto &topmost_op = (LogicalOperator &)**candidate;
	idx_t shift = lhs_bindings.size();

	vector<unique_ptr<LogicalOperator> *> path_to_unnest;
	auto curr_op = &(topmost_op.children[0]);
	while (curr_op->get()->type == LogicalOperatorType::LOGICAL_PROJECTION) {

		path_to_unnest.push_back(curr_op);
		D_ASSERT(curr_op->get()->type == LogicalOperatorType::LOGICAL_PROJECTION);
		auto &proj = (LogicalProjection &)*curr_op->get();

		// pop the two last expressions from all projections (delim_idx and UNNEST column)
		D_ASSERT(proj.expressions.size() > 2);
		proj.expressions.pop_back();
		proj.expressions.pop_back();

		// store all shifted current bindings
		idx_t tbl_idx = proj.table_index;
		for (idx_t i = 0; i < proj.expressions.size(); i++) {
			ReplaceBinding replace_binding(ColumnBinding(tbl_idx, i), ColumnBinding(tbl_idx, i + shift));
			updater.replace_bindings.push_back(replace_binding);
		}

		curr_op = &curr_op->get()->children[0];
	}

	// update all bindings by shifting them
	updater.VisitOperator(*plan_ptr->get());
	updater.replace_bindings.clear();

	// update all bindings coming from the LHS to RHS bindings
	D_ASSERT(topmost_op.children[0]->type == LogicalOperatorType::LOGICAL_PROJECTION);
	auto &top_proj = (LogicalProjection &)*topmost_op.children[0];
	for (idx_t i = 0; i < lhs_bindings.size(); i++) {
		ReplaceBinding replace_binding(lhs_bindings[i].binding, ColumnBinding(top_proj.table_index, i));
		updater.replace_bindings.push_back(replace_binding);
	}

	// temporarily remove the BOUND_UNNEST and the child of the LOGICAL_UNNEST from the plan
	D_ASSERT(curr_op->get()->type == LogicalOperatorType::LOGICAL_UNNEST);
	auto &unnest = (LogicalUnnest &)*curr_op->get();
	auto temp_bound_unnest = std::move(unnest.expressions[0]);
	auto temp_unnest_child = std::move(unnest.children[0]);
	unnest.expressions.clear();
	unnest.children.clear();
	// update the bindings of the plan
	updater.VisitOperator(*plan_ptr->get());
	updater.replace_bindings.clear();
	// add the child again
	unnest.expressions.push_back(std::move(temp_bound_unnest));
	unnest.children.push_back(std::move(temp_unnest_child));

	// add the LHS expressions to each LOGICAL_PROJECTION
	for (idx_t i = path_to_unnest.size(); i > 0; i--) {

		D_ASSERT(path_to_unnest[i - 1]->get()->type == LogicalOperatorType::LOGICAL_PROJECTION);
		auto &proj = (LogicalProjection &)*path_to_unnest[i - 1]->get();

		// temporarily store the existing expressions
		vector<unique_ptr<Expression>> existing_expressions;
		for (idx_t expr_idx = 0; expr_idx < proj.expressions.size(); expr_idx++) {
			existing_expressions.push_back(std::move(proj.expressions[expr_idx]));
		}

		proj.expressions.clear();

		// add the new expressions
		for (idx_t expr_idx = 0; expr_idx < lhs_bindings.size(); expr_idx++) {
			auto new_expr = make_unique<BoundColumnRefExpression>(
			    lhs_bindings[expr_idx].alias, lhs_bindings[expr_idx].type, lhs_bindings[expr_idx].binding);
			proj.expressions.push_back(std::move(new_expr));

			// update the table index
			lhs_bindings[expr_idx].binding.table_index = proj.table_index;
			lhs_bindings[expr_idx].binding.column_index = expr_idx;
		}

		// add the existing expressions again
		for (idx_t expr_idx = 0; expr_idx < existing_expressions.size(); expr_idx++) {
			proj.expressions.push_back(std::move(existing_expressions[expr_idx]));
		}
	}
}

void UnnestRewriter::UpdateBoundUnnestBindings(UnnestRewriterPlanUpdater &updater,
                                               unique_ptr<LogicalOperator> *candidate) {

	auto &topmost_op = (LogicalOperator &)**candidate;

	// traverse LOGICAL_PROJECTION(s)
	auto curr_op = &(topmost_op.children[0]);
	while (curr_op->get()->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		curr_op = &curr_op->get()->children[0];
	}

	// found the LOGICAL_UNNEST
	D_ASSERT(curr_op->get()->type == LogicalOperatorType::LOGICAL_UNNEST);
	auto &unnest = (LogicalUnnest &)*curr_op->get();

	auto unnest_child_cols = unnest.children[0]->GetColumnBindings();
	for (idx_t delim_col_idx = 0; delim_col_idx < delim_columns.size(); delim_col_idx++) {
		for (idx_t child_col_idx = 0; child_col_idx < unnest_child_cols.size(); child_col_idx++) {
			if (delim_columns[delim_col_idx].table_index == unnest_child_cols[child_col_idx].table_index) {
				ColumnBinding old_binding(overwritten_tbl_idx, DConstants::INVALID_INDEX);
				updater.replace_bindings.emplace_back(ReplaceBinding(old_binding, delim_columns[delim_col_idx]));
				break;
			}
		}
	}

	// update bindings
	D_ASSERT(unnest.expressions.size() == 1);
	updater.VisitExpression(&unnest.expressions[0]);
	updater.replace_bindings.clear();
}

void UnnestRewriter::GetDelimColumns(LogicalOperator &op) {

	D_ASSERT(op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN);
	auto &delim_join = (LogicalDelimJoin &)op;
	for (idx_t i = 0; i < delim_join.duplicate_eliminated_columns.size(); i++) {
		auto &expr = *delim_join.duplicate_eliminated_columns[i];
		D_ASSERT(expr.type == ExpressionType::BOUND_COLUMN_REF);
		auto &bound_colref_expr = (BoundColumnRefExpression &)expr;
		delim_columns.push_back(bound_colref_expr.binding);
	}
}

void UnnestRewriter::GetLHSExpressions(LogicalOperator &op) {

	op.ResolveOperatorTypes();
	auto col_bindings = op.GetColumnBindings();
	D_ASSERT(op.types.size() == col_bindings.size());

	bool set_alias = false;
	// we can easily extract the alias for LOGICAL_PROJECTION(s)
	if (op.type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = (LogicalProjection &)op;
		if (proj.expressions.size() == op.types.size()) {
			set_alias = true;
		}
	}

	for (idx_t i = 0; i < op.types.size(); i++) {
		lhs_bindings.emplace_back(LHSBinding(col_bindings[i], op.types[i]));
		if (set_alias) {
			auto &proj = (LogicalProjection &)op;
			lhs_bindings.back().alias = proj.expressions[i]->alias;
		}
	}
}

} // namespace duckdb


namespace duckdb {

BasePipelineEvent::BasePipelineEvent(shared_ptr<Pipeline> pipeline_p)
    : Event(pipeline_p->executor), pipeline(std::move(pipeline_p)) {
}

BasePipelineEvent::BasePipelineEvent(Pipeline &pipeline_p)
    : Event(pipeline_p.executor), pipeline(pipeline_p.shared_from_this()) {
}

} // namespace duckdb






namespace duckdb {

Event::Event(Executor &executor_p)
    : executor(executor_p), finished_tasks(0), total_tasks(0), finished_dependencies(0), total_dependencies(0),
      finished(false) {
}

void Event::CompleteDependency() {
	idx_t current_finished = ++finished_dependencies;
	D_ASSERT(current_finished <= total_dependencies);
	if (current_finished == total_dependencies) {
		// all dependencies have been completed: schedule the event
		D_ASSERT(total_tasks == 0);
		Schedule();
		if (total_tasks == 0) {
			Finish();
		}
	}
}

void Event::Finish() {
	D_ASSERT(!finished);
	FinishEvent();
	finished = true;
	// finished processing the pipeline, now we can schedule pipelines that depend on this pipeline
	for (auto &parent_entry : parents) {
		auto parent = parent_entry.lock();
		if (!parent) { // LCOV_EXCL_START
			continue;
		} // LCOV_EXCL_STOP
		// mark a dependency as completed for each of the parents
		parent->CompleteDependency();
	}
	FinalizeFinish();
}

void Event::AddDependency(Event &event) {
	total_dependencies++;
	event.parents.push_back(weak_ptr<Event>(shared_from_this()));
#ifdef DEBUG
	event.parents_raw.push_back(this);
#endif
}

const vector<Event *> &Event::GetParentsVerification() const {
	D_ASSERT(parents.size() == parents_raw.size());
	return parents_raw;
}

void Event::FinishTask() {
	D_ASSERT(finished_tasks.load() < total_tasks.load());
	idx_t current_tasks = total_tasks;
	idx_t current_finished = ++finished_tasks;
	D_ASSERT(current_finished <= current_tasks);
	if (current_finished == current_tasks) {
		Finish();
	}
}

void Event::InsertEvent(shared_ptr<Event> replacement_event) {
	replacement_event->parents = std::move(parents);
#ifdef DEBUG
	replacement_event->parents_raw = std::move(parents_raw);
#endif
	replacement_event->AddDependency(*this);
	executor.AddEvent(std::move(replacement_event));
}

void Event::SetTasks(vector<unique_ptr<Task>> tasks) {
	auto &ts = TaskScheduler::GetScheduler(executor.context);
	D_ASSERT(total_tasks == 0);
	D_ASSERT(!tasks.empty());
	this->total_tasks = tasks.size();
	for (auto &task : tasks) {
		ts.ScheduleTask(executor.GetToken(), std::move(task));
	}
}

} // namespace duckdb

















#include <algorithm>

namespace duckdb {

Executor::Executor(ClientContext &context) : context(context) {
}

Executor::~Executor() {
}

Executor &Executor::Get(ClientContext &context) {
	return context.GetExecutor();
}

void Executor::AddEvent(shared_ptr<Event> event) {
	lock_guard<mutex> elock(executor_lock);
	if (cancelled) {
		return;
	}
	events.push_back(std::move(event));
}

struct PipelineEventStack {
	Event *pipeline_initialize_event;
	Event *pipeline_event;
	Event *pipeline_finish_event;
	Event *pipeline_complete_event;
};

using event_map_t = unordered_map<const Pipeline *, PipelineEventStack>;

struct ScheduleEventData {
	ScheduleEventData(const vector<shared_ptr<MetaPipeline>> &meta_pipelines, vector<shared_ptr<Event>> &events,
	                  bool initial_schedule)
	    : meta_pipelines(meta_pipelines), events(events), initial_schedule(initial_schedule) {
	}

	const vector<shared_ptr<MetaPipeline>> &meta_pipelines;
	vector<shared_ptr<Event>> &events;
	bool initial_schedule;
	event_map_t event_map;
};

void Executor::SchedulePipeline(const shared_ptr<MetaPipeline> &meta_pipeline, ScheduleEventData &event_data) {
	D_ASSERT(meta_pipeline);
	auto &events = event_data.events;
	auto &event_map = event_data.event_map;

	// create events/stack for the base pipeline
	auto base_pipeline = meta_pipeline->GetBasePipeline();
	auto base_initialize_event = make_shared<PipelineInitializeEvent>(base_pipeline);
	auto base_event = make_shared<PipelineEvent>(base_pipeline);
	auto base_finish_event = make_shared<PipelineFinishEvent>(base_pipeline);
	auto base_complete_event = make_shared<PipelineCompleteEvent>(base_pipeline->executor, event_data.initial_schedule);
	PipelineEventStack base_stack {base_initialize_event.get(), base_event.get(), base_finish_event.get(),
	                               base_complete_event.get()};
	events.push_back(std::move(base_initialize_event));
	events.push_back(std::move(base_event));
	events.push_back(std::move(base_finish_event));
	events.push_back(std::move(base_complete_event));

	// dependencies: initialize -> event -> finish -> complete
	base_stack.pipeline_event->AddDependency(*base_stack.pipeline_initialize_event);
	base_stack.pipeline_finish_event->AddDependency(*base_stack.pipeline_event);
	base_stack.pipeline_complete_event->AddDependency(*base_stack.pipeline_finish_event);

	// create an event and stack for all pipelines in the MetaPipeline
	vector<shared_ptr<Pipeline>> pipelines;
	meta_pipeline->GetPipelines(pipelines, false);
	for (idx_t i = 1; i < pipelines.size(); i++) { // loop starts at 1 because 0 is the base pipeline
		auto &pipeline = pipelines[i];
		D_ASSERT(pipeline);

		// create events/stack for this pipeline
		auto pipeline_event = make_shared<PipelineEvent>(pipeline);
		Event *pipeline_finish_event_ptr;
		if (meta_pipeline->HasFinishEvent(pipeline.get())) {
			// this pipeline has its own finish event (despite going into the same sink - Finalize twice!)
			auto pipeline_finish_event = make_unique<PipelineFinishEvent>(pipeline);
			pipeline_finish_event_ptr = pipeline_finish_event.get();
			events.push_back(std::move(pipeline_finish_event));
			base_stack.pipeline_complete_event->AddDependency(*pipeline_finish_event_ptr);
		} else {
			pipeline_finish_event_ptr = base_stack.pipeline_finish_event;
		}
		PipelineEventStack pipeline_stack {base_stack.pipeline_initialize_event, pipeline_event.get(),
		                                   pipeline_finish_event_ptr, base_stack.pipeline_complete_event};
		events.push_back(std::move(pipeline_event));

		// dependencies: base_initialize -> pipeline_event -> base_finish
		pipeline_stack.pipeline_event->AddDependency(*base_stack.pipeline_initialize_event);
		pipeline_stack.pipeline_finish_event->AddDependency(*pipeline_stack.pipeline_event);

		// add pipeline stack to event map
		event_map.insert(make_pair(pipeline.get(), pipeline_stack));
	}

	// add base stack to the event data too
	event_map.insert(make_pair(base_pipeline.get(), base_stack));

	// set up the dependencies within this MetaPipeline
	for (auto &pipeline : pipelines) {
		auto source = pipeline->GetSource();
		if (source->type == PhysicalOperatorType::TABLE_SCAN) {
			// we have to reset the source here (in the main thread), because some of our clients (looking at you, R)
			// do not like it when threads other than the main thread call into R, for e.g., arrow scans
			pipeline->ResetSource(true);
		}

		auto dependencies = meta_pipeline->GetDependencies(pipeline.get());
		if (!dependencies) {
			continue;
		}
		auto &pipeline_stack = event_map[pipeline.get()];
		for (auto &dependency : *dependencies) {
			auto &dependency_stack = event_map[dependency];
			pipeline_stack.pipeline_event->AddDependency(*dependency_stack.pipeline_event);
		}
	}
}

void Executor::ScheduleEventsInternal(ScheduleEventData &event_data) {
	auto &events = event_data.events;
	D_ASSERT(events.empty());

	// create all the required pipeline events
	for (auto &pipeline : event_data.meta_pipelines) {
		SchedulePipeline(pipeline, event_data);
	}

	// set up the dependencies across MetaPipelines
	auto &event_map = event_data.event_map;
	for (auto &entry : event_map) {
		auto pipeline = entry.first;
		for (auto &dependency : pipeline->dependencies) {
			auto dep = dependency.lock();
			D_ASSERT(dep);
			auto event_map_entry = event_map.find(dep.get());
			D_ASSERT(event_map_entry != event_map.end());
			auto &dep_entry = event_map_entry->second;
			D_ASSERT(dep_entry.pipeline_complete_event);
			entry.second.pipeline_event->AddDependency(*dep_entry.pipeline_complete_event);
		}
	}

	// verify that we have no cyclic dependencies
	VerifyScheduledEvents(event_data);

	// schedule the pipelines that do not have dependencies
	for (auto &event : events) {
		if (!event->HasDependencies()) {
			event->Schedule();
		}
	}
}

void Executor::ScheduleEvents(const vector<shared_ptr<MetaPipeline>> &meta_pipelines) {
	ScheduleEventData event_data(meta_pipelines, events, true);
	ScheduleEventsInternal(event_data);
}

void Executor::VerifyScheduledEvents(const ScheduleEventData &event_data) {
#ifdef DEBUG
	const idx_t count = event_data.events.size();
	vector<Event *> vertices;
	vertices.reserve(count);
	for (const auto &event : event_data.events) {
		vertices.push_back(event.get());
	}
	vector<bool> visited(count, false);
	vector<bool> recursion_stack(count, false);
	for (idx_t i = 0; i < count; i++) {
		VerifyScheduledEventsInternal(i, vertices, visited, recursion_stack);
	}
#endif
}

void Executor::VerifyScheduledEventsInternal(const idx_t vertex, const vector<Event *> &vertices, vector<bool> &visited,
                                             vector<bool> &recursion_stack) {
	D_ASSERT(!recursion_stack[vertex]); // this vertex is in the recursion stack: circular dependency!
	if (visited[vertex]) {
		return; // early out: we already visited this vertex
	}

	auto &parents = vertices[vertex]->GetParentsVerification();
	if (parents.empty()) {
		return; // early out: outgoing edges
	}

	// create a vector the indices of the adjacent events
	vector<idx_t> adjacent;
	const idx_t count = vertices.size();
	for (auto parent : parents) {
		idx_t i;
		for (i = 0; i < count; i++) {
			if (vertices[i] == parent) {
				adjacent.push_back(i);
				break;
			}
		}
		D_ASSERT(i != count); // dependency must be in there somewhere
	}

	// mark vertex as visited and add to recursion stack
	visited[vertex] = true;
	recursion_stack[vertex] = true;

	// recurse into adjacent vertices
	for (const auto &i : adjacent) {
		VerifyScheduledEventsInternal(i, vertices, visited, recursion_stack);
	}

	// remove vertex from recursion stack
	recursion_stack[vertex] = false;
}

void Executor::AddRecursiveCTE(PhysicalOperator *rec_cte) {
	recursive_ctes.push_back(rec_cte);
}

void Executor::ReschedulePipelines(const vector<shared_ptr<MetaPipeline>> &pipelines_p,
                                   vector<shared_ptr<Event>> &events_p) {
	ScheduleEventData event_data(pipelines_p, events_p, false);
	ScheduleEventsInternal(event_data);
}

bool Executor::NextExecutor() {
	if (root_pipeline_idx >= root_pipelines.size()) {
		return false;
	}
	root_pipelines[root_pipeline_idx]->Reset();
	root_executor = make_unique<PipelineExecutor>(context, *root_pipelines[root_pipeline_idx]);
	root_pipeline_idx++;
	return true;
}

void Executor::VerifyPipeline(Pipeline &pipeline) {
	D_ASSERT(!pipeline.ToString().empty());
	auto operators = pipeline.GetOperators();
	for (auto &other_pipeline : pipelines) {
		auto other_operators = other_pipeline->GetOperators();
		for (idx_t op_idx = 0; op_idx < operators.size(); op_idx++) {
			for (idx_t other_idx = 0; other_idx < other_operators.size(); other_idx++) {
				auto &left = *operators[op_idx];
				auto &right = *other_operators[other_idx];
				if (left.Equals(right)) {
					D_ASSERT(right.Equals(left));
				} else {
					D_ASSERT(!right.Equals(left));
				}
			}
		}
	}
}

void Executor::VerifyPipelines() {
#ifdef DEBUG
	for (auto &pipeline : pipelines) {
		VerifyPipeline(*pipeline);
	}
#endif
}

void Executor::Initialize(unique_ptr<PhysicalOperator> physical_plan) {
	Reset();
	owned_plan = std::move(physical_plan);
	InitializeInternal(owned_plan.get());
}

void Executor::Initialize(PhysicalOperator *plan) {
	Reset();
	InitializeInternal(plan);
}

void Executor::InitializeInternal(PhysicalOperator *plan) {

	auto &scheduler = TaskScheduler::GetScheduler(context);
	{
		lock_guard<mutex> elock(executor_lock);
		physical_plan = plan;

		this->profiler = ClientData::Get(context).profiler;
		profiler->Initialize(physical_plan);
		this->producer = scheduler.CreateProducer();

		// build and ready the pipelines
		PipelineBuildState state;
		auto root_pipeline = make_shared<MetaPipeline>(*this, state, nullptr);
		root_pipeline->Build(physical_plan);
		root_pipeline->Ready();

		// ready recursive cte pipelines too
		for (auto &rec_cte : recursive_ctes) {
			D_ASSERT(rec_cte->type == PhysicalOperatorType::RECURSIVE_CTE);
			auto &rec_cte_op = (PhysicalRecursiveCTE &)*rec_cte;
			rec_cte_op.recursive_meta_pipeline->Ready();
		}

		// set root pipelines, i.e., all pipelines that end in the final sink
		root_pipeline->GetPipelines(root_pipelines, false);
		root_pipeline_idx = 0;

		// collect all meta-pipelines from the root pipeline
		vector<shared_ptr<MetaPipeline>> to_schedule;
		root_pipeline->GetMetaPipelines(to_schedule, true, true);

		// number of 'PipelineCompleteEvent's is equal to the number of meta pipelines, so we have to set it here
		total_pipelines = to_schedule.size();

		// collect all pipelines from the root pipelines (recursively) for the progress bar and verify them
		root_pipeline->GetPipelines(pipelines, true);

		// finally, verify and schedule
		VerifyPipelines();
		ScheduleEvents(to_schedule);
	}
}

void Executor::CancelTasks() {
	task.reset();
	// we do this by creating weak pointers to all pipelines
	// then clearing our references to the pipelines
	// and waiting until all pipelines have been destroyed
	vector<weak_ptr<Pipeline>> weak_references;
	{
		lock_guard<mutex> elock(executor_lock);
		weak_references.reserve(pipelines.size());
		cancelled = true;
		for (auto &pipeline : pipelines) {
			weak_references.push_back(weak_ptr<Pipeline>(pipeline));
		}
		for (auto op : recursive_ctes) {
			D_ASSERT(op->type == PhysicalOperatorType::RECURSIVE_CTE);
			auto &rec_cte = (PhysicalRecursiveCTE &)*op;
			rec_cte.recursive_meta_pipeline.reset();
		}
		pipelines.clear();
		root_pipelines.clear();
		events.clear();
	}
	WorkOnTasks();
	for (auto &weak_ref : weak_references) {
		while (true) {
			auto weak = weak_ref.lock();
			if (!weak) {
				break;
			}
		}
	}
}

void Executor::WorkOnTasks() {
	auto &scheduler = TaskScheduler::GetScheduler(context);

	unique_ptr<Task> task;
	while (scheduler.GetTaskFromProducer(*producer, task)) {
		task->Execute(TaskExecutionMode::PROCESS_ALL);
		task.reset();
	}
}

bool Executor::ExecutionIsFinished() {
	return completed_pipelines >= total_pipelines || HasError();
}

PendingExecutionResult Executor::ExecuteTask() {
	if (execution_result != PendingExecutionResult::RESULT_NOT_READY) {
		return execution_result;
	}
	// check if there are any incomplete pipelines
	auto &scheduler = TaskScheduler::GetScheduler(context);
	while (completed_pipelines < total_pipelines) {
		// there are! if we don't already have a task, fetch one
		if (!task) {
			scheduler.GetTaskFromProducer(*producer, task);
		}
		if (task) {
			// if we have a task, partially process it
			auto result = task->Execute(TaskExecutionMode::PROCESS_PARTIAL);
			if (result != TaskExecutionResult::TASK_NOT_FINISHED) {
				// if the task is finished, clean it up
				task.reset();
			}
		}
		if (!HasError()) {
			// we (partially) processed a task and no exceptions were thrown
			// give back control to the caller
			return PendingExecutionResult::RESULT_NOT_READY;
		}
		execution_result = PendingExecutionResult::EXECUTION_ERROR;

		// an exception has occurred executing one of the pipelines
		// we need to cancel all tasks associated with this executor
		CancelTasks();
		ThrowException();
	}
	D_ASSERT(!task);

	lock_guard<mutex> elock(executor_lock);
	pipelines.clear();
	NextExecutor();
	if (HasError()) { // LCOV_EXCL_START
		// an exception has occurred executing one of the pipelines
		execution_result = PendingExecutionResult::EXECUTION_ERROR;
		ThrowException();
	} // LCOV_EXCL_STOP
	execution_result = PendingExecutionResult::RESULT_READY;
	return execution_result;
}

void Executor::Reset() {
	lock_guard<mutex> elock(executor_lock);
	physical_plan = nullptr;
	cancelled = false;
	owned_plan.reset();
	root_executor.reset();
	root_pipelines.clear();
	root_pipeline_idx = 0;
	completed_pipelines = 0;
	total_pipelines = 0;
	exceptions.clear();
	pipelines.clear();
	events.clear();
	execution_result = PendingExecutionResult::RESULT_NOT_READY;
}

shared_ptr<Pipeline> Executor::CreateChildPipeline(Pipeline *current, PhysicalOperator *op) {
	D_ASSERT(!current->operators.empty());
	D_ASSERT(op->IsSource());
	// found another operator that is a source, schedule a child pipeline
	// 'op' is the source, and the sink is the same
	auto child_pipeline = make_shared<Pipeline>(*this);
	child_pipeline->sink = current->sink;
	child_pipeline->source = op;

	// the child pipeline has the same operators up until 'op'
	for (auto current_op : current->operators) {
		if (current_op == op) {
			break;
		}
		child_pipeline->operators.push_back(current_op);
	}

	return child_pipeline;
}

vector<LogicalType> Executor::GetTypes() {
	D_ASSERT(physical_plan);
	return physical_plan->GetTypes();
}

void Executor::PushError(PreservedError exception) {
	lock_guard<mutex> elock(error_lock);
	// interrupt execution of any other pipelines that belong to this executor
	context.interrupted = true;
	// push the exception onto the stack
	exceptions.push_back(std::move(exception));
}

bool Executor::HasError() {
	lock_guard<mutex> elock(error_lock);
	return !exceptions.empty();
}

void Executor::ThrowException() {
	lock_guard<mutex> elock(error_lock);
	D_ASSERT(!exceptions.empty());
	auto &entry = exceptions[0];
	entry.Throw();
}

void Executor::Flush(ThreadContext &tcontext) {
	profiler->Flush(tcontext.profiler);
}

bool Executor::GetPipelinesProgress(double &current_progress) { // LCOV_EXCL_START
	lock_guard<mutex> elock(executor_lock);

	vector<double> progress;
	vector<idx_t> cardinality;
	idx_t total_cardinality = 0;
	for (auto &pipeline : pipelines) {
		double child_percentage;
		idx_t child_cardinality;

		if (!pipeline->GetProgress(child_percentage, child_cardinality)) {
			return false;
		}
		progress.push_back(child_percentage);
		cardinality.push_back(child_cardinality);
		total_cardinality += child_cardinality;
	}
	current_progress = 0;
	for (size_t i = 0; i < progress.size(); i++) {
		current_progress += progress[i] * double(cardinality[i]) / double(total_cardinality);
	}
	return true;
} // LCOV_EXCL_STOP

bool Executor::HasResultCollector() {
	return physical_plan->type == PhysicalOperatorType::RESULT_COLLECTOR;
}

unique_ptr<QueryResult> Executor::GetResult() {
	D_ASSERT(HasResultCollector());
	auto &result_collector = (PhysicalResultCollector &)*physical_plan;
	D_ASSERT(result_collector.sink_state);
	return result_collector.GetResult(*result_collector.sink_state);
}

unique_ptr<DataChunk> Executor::FetchChunk() {
	D_ASSERT(physical_plan);

	auto chunk = make_unique<DataChunk>();
	root_executor->InitializeChunk(*chunk);
	while (true) {
		root_executor->ExecutePull(*chunk);
		if (chunk->size() == 0) {
			root_executor->PullFinalize();
			if (NextExecutor()) {
				continue;
			}
			break;
		} else {
			break;
		}
	}
	return chunk;
}

} // namespace duckdb



namespace duckdb {

ExecutorTask::ExecutorTask(Executor &executor_p) : executor(executor_p) {
}

ExecutorTask::ExecutorTask(ClientContext &context) : ExecutorTask(Executor::Get(context)) {
}

ExecutorTask::~ExecutorTask() {
}

TaskExecutionResult ExecutorTask::Execute(TaskExecutionMode mode) {
	try {
		return ExecuteTask(mode);
	} catch (Exception &ex) {
		executor.PushError(PreservedError(ex));
	} catch (std::exception &ex) {
		executor.PushError(PreservedError(ex));
	} catch (...) { // LCOV_EXCL_START
		executor.PushError(PreservedError("Unknown exception in Finalize!"));
	} // LCOV_EXCL_STOP
	return TaskExecutionResult::TASK_ERROR;
}

} // namespace duckdb





namespace duckdb {

MetaPipeline::MetaPipeline(Executor &executor_p, PipelineBuildState &state_p, PhysicalOperator *sink_p)
    : executor(executor_p), state(state_p), sink(sink_p), recursive_cte(false), next_batch_index(0) {
	CreatePipeline();
}

Executor &MetaPipeline::GetExecutor() const {
	return executor;
}

PipelineBuildState &MetaPipeline::GetState() const {
	return state;
}

PhysicalOperator *MetaPipeline::GetSink() const {
	return sink;
}

shared_ptr<Pipeline> &MetaPipeline::GetBasePipeline() {
	return pipelines[0];
}

void MetaPipeline::GetPipelines(vector<shared_ptr<Pipeline>> &result, bool recursive) {
	result.insert(result.end(), pipelines.begin(), pipelines.end());
	if (recursive) {
		for (auto &child : children) {
			child->GetPipelines(result, true);
		}
	}
}

void MetaPipeline::GetMetaPipelines(vector<shared_ptr<MetaPipeline>> &result, bool recursive, bool skip) {
	if (!skip) {
		result.push_back(shared_from_this());
	}
	if (recursive) {
		for (auto &child : children) {
			child->GetMetaPipelines(result, true, false);
		}
	}
}

const vector<Pipeline *> *MetaPipeline::GetDependencies(Pipeline *dependant) const {
	auto it = dependencies.find(dependant);
	if (it == dependencies.end()) {
		return nullptr;
	} else {
		return &it->second;
	}
}

bool MetaPipeline::HasRecursiveCTE() const {
	return recursive_cte;
}

void MetaPipeline::SetRecursiveCTE() {
	recursive_cte = true;
}

void MetaPipeline::AssignNextBatchIndex(Pipeline *pipeline) {
	pipeline->base_batch_index = next_batch_index++ * PipelineBuildState::BATCH_INCREMENT;
}

void MetaPipeline::Build(PhysicalOperator *op) {
	D_ASSERT(pipelines.size() == 1);
	D_ASSERT(children.empty());
	D_ASSERT(final_pipelines.empty());
	op->BuildPipelines(*pipelines.back(), *this);
}

void MetaPipeline::Ready() {
	for (auto &pipeline : pipelines) {
		pipeline->Ready();
	}
	for (auto &child : children) {
		child->Ready();
	}
}

MetaPipeline *MetaPipeline::CreateChildMetaPipeline(Pipeline &current, PhysicalOperator *op) {
	children.push_back(make_unique<MetaPipeline>(executor, state, op));
	auto child_meta_pipeline = children.back().get();
	// child MetaPipeline must finish completely before this MetaPipeline can start
	current.AddDependency(child_meta_pipeline->GetBasePipeline());
	// child meta pipeline is part of the recursive CTE too
	child_meta_pipeline->recursive_cte = recursive_cte;
	return child_meta_pipeline;
}

Pipeline *MetaPipeline::CreatePipeline() {
	pipelines.emplace_back(make_unique<Pipeline>(executor));
	state.SetPipelineSink(*pipelines.back(), sink, next_batch_index++);
	return pipelines.back().get();
}

void MetaPipeline::AddDependenciesFrom(Pipeline *dependant, Pipeline *start, bool including) {
	// find 'start'
	auto it = pipelines.begin();
	for (; it->get() != start; it++) {
	}

	if (!including) {
		it++;
	}

	// collect pipelines that were created from then
	vector<Pipeline *> created_pipelines;
	for (; it != pipelines.end(); it++) {
		if (it->get() == dependant) {
			// cannot depend on itself
			continue;
		}
		created_pipelines.push_back(it->get());
	}

	// add them to the dependencies
	auto &deps = dependencies[dependant];
	deps.insert(deps.begin(), created_pipelines.begin(), created_pipelines.end());
}

void MetaPipeline::AddFinishEvent(Pipeline *pipeline) {
	finish_pipelines.insert(pipeline);
}

bool MetaPipeline::HasFinishEvent(Pipeline *pipeline) {
	return finish_pipelines.find(pipeline) != finish_pipelines.end();
}

Pipeline *MetaPipeline::CreateUnionPipeline(Pipeline &current, bool order_matters) {
	if (HasRecursiveCTE()) {
		throw NotImplementedException("UNIONS are not supported in recursive CTEs yet");
	}

	// create the union pipeline (batch index 0, should be set correctly afterwards)
	auto union_pipeline = CreatePipeline();
	state.SetPipelineOperators(*union_pipeline, state.GetPipelineOperators(current));
	state.SetPipelineSink(*union_pipeline, sink, 0);

	// 'union_pipeline' inherits ALL dependencies of 'current' (within this MetaPipeline, and across MetaPipelines)
	union_pipeline->dependencies = current.dependencies;
	auto current_deps = GetDependencies(&current);
	if (current_deps) {
		dependencies[union_pipeline] = *current_deps;
	}

	if (order_matters) {
		// if we need to preserve order, or if the sink is not parallel, we set a dependency
		dependencies[union_pipeline].push_back(&current);
	}

	return union_pipeline;
}

void MetaPipeline::CreateChildPipeline(Pipeline &current, PhysicalOperator *op, Pipeline *last_pipeline) {
	// rule 2: 'current' must be fully built (down to the source) before creating the child pipeline
	D_ASSERT(current.source);
	if (HasRecursiveCTE()) {
		throw NotImplementedException("Child pipelines are not supported in recursive CTEs yet");
	}

	// create the child pipeline (same batch index)
	pipelines.emplace_back(state.CreateChildPipeline(executor, current, op));
	auto child_pipeline = pipelines.back().get();
	child_pipeline->base_batch_index = current.base_batch_index;

	// child pipeline has a depency (within this MetaPipeline on all pipelines that were scheduled
	// between 'current' and now (including 'current') - set them up
	dependencies[child_pipeline].push_back(&current);
	AddDependenciesFrom(child_pipeline, last_pipeline, false);
	D_ASSERT(!GetDependencies(child_pipeline)->empty());
}

} // namespace duckdb
















namespace duckdb {

class PipelineTask : public ExecutorTask {
	static constexpr const idx_t PARTIAL_CHUNK_COUNT = 50;

public:
	explicit PipelineTask(Pipeline &pipeline_p, shared_ptr<Event> event_p)
	    : ExecutorTask(pipeline_p.executor), pipeline(pipeline_p), event(std::move(event_p)) {
	}

	Pipeline &pipeline;
	shared_ptr<Event> event;
	unique_ptr<PipelineExecutor> pipeline_executor;

public:
	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		if (!pipeline_executor) {
			pipeline_executor = make_unique<PipelineExecutor>(pipeline.GetClientContext(), pipeline);
		}
		if (mode == TaskExecutionMode::PROCESS_PARTIAL) {
			bool finished = pipeline_executor->Execute(PARTIAL_CHUNK_COUNT);
			if (!finished) {
				return TaskExecutionResult::TASK_NOT_FINISHED;
			}
		} else {
			pipeline_executor->Execute();
		}
		event->FinishTask();
		pipeline_executor.reset();
		return TaskExecutionResult::TASK_FINISHED;
	}
};

Pipeline::Pipeline(Executor &executor_p)
    : executor(executor_p), ready(false), initialized(false), source(nullptr), sink(nullptr) {
}

ClientContext &Pipeline::GetClientContext() {
	return executor.context;
}

bool Pipeline::GetProgress(double &current_percentage, idx_t &source_cardinality) {
	D_ASSERT(source);
	source_cardinality = source->estimated_cardinality;
	if (!initialized) {
		current_percentage = 0;
		return true;
	}
	auto &client = executor.context;
	current_percentage = source->GetProgress(client, *source_state);
	return current_percentage >= 0;
}

void Pipeline::ScheduleSequentialTask(shared_ptr<Event> &event) {
	vector<unique_ptr<Task>> tasks;
	tasks.push_back(make_unique<PipelineTask>(*this, event));
	event->SetTasks(std::move(tasks));
}

bool Pipeline::ScheduleParallel(shared_ptr<Event> &event) {
	// check if the sink, source and all intermediate operators support parallelism
	if (!sink->ParallelSink()) {
		return false;
	}
	if (!source->ParallelSource()) {
		return false;
	}
	for (auto &op : operators) {
		if (!op->ParallelOperator()) {
			return false;
		}
	}
	if (sink->RequiresBatchIndex()) {
		if (!source->SupportsBatchIndex()) {
			throw InternalException(
			    "Attempting to schedule a pipeline where the sink requires batch index but source does not support it");
		}
	}
	idx_t max_threads = source_state->MaxThreads();
	return LaunchScanTasks(event, max_threads);
}

bool Pipeline::IsOrderDependent() const {
	auto &config = DBConfig::GetConfig(executor.context);
	if (!config.options.preserve_insertion_order) {
		return false;
	}
	if (sink && sink->IsOrderDependent()) {
		return true;
	}
	if (source && source->IsOrderDependent()) {
		return true;
	}
	for (auto &op : operators) {
		if (op->IsOrderDependent()) {
			return true;
		}
	}
	return false;
}

void Pipeline::Schedule(shared_ptr<Event> &event) {
	D_ASSERT(ready);
	D_ASSERT(sink);
	Reset();
	if (!ScheduleParallel(event)) {
		// could not parallelize this pipeline: push a sequential task instead
		ScheduleSequentialTask(event);
	}
}

bool Pipeline::LaunchScanTasks(shared_ptr<Event> &event, idx_t max_threads) {
	// split the scan up into parts and schedule the parts
	auto &scheduler = TaskScheduler::GetScheduler(executor.context);
	idx_t active_threads = scheduler.NumberOfThreads();
	if (max_threads > active_threads) {
		max_threads = active_threads;
	}
	if (max_threads <= 1) {
		// too small to parallelize
		return false;
	}

	// launch a task for every thread
	vector<unique_ptr<Task>> tasks;
	for (idx_t i = 0; i < max_threads; i++) {
		tasks.push_back(make_unique<PipelineTask>(*this, event));
	}
	event->SetTasks(std::move(tasks));
	return true;
}

void Pipeline::ResetSink() {
	if (sink) {
		lock_guard<mutex> guard(sink->lock);
		if (!sink->sink_state) {
			sink->sink_state = sink->GetGlobalSinkState(GetClientContext());
		}
	}
}

void Pipeline::Reset() {
	ResetSink();
	for (auto &op : operators) {
		if (op) {
			lock_guard<mutex> guard(op->lock);
			if (!op->op_state) {
				op->op_state = op->GetGlobalOperatorState(GetClientContext());
			}
		}
	}
	ResetSource(false);
	// we no longer reset source here because this function is no longer guaranteed to be called by the main thread
	// source reset needs to be called by the main thread because resetting a source may call into clients like R
	initialized = true;
}

void Pipeline::ResetSource(bool force) {
	if (force || !source_state) {
		source_state = source->GetGlobalSourceState(GetClientContext());
	}
}

void Pipeline::Ready() {
	if (ready) {
		return;
	}
	ready = true;
	std::reverse(operators.begin(), operators.end());
}

void Pipeline::Finalize(Event &event) {
	if (executor.HasError()) {
		return;
	}
	D_ASSERT(ready);
	try {
		auto sink_state = sink->Finalize(*this, event, executor.context, *sink->sink_state);
		sink->sink_state->state = sink_state;
	} catch (Exception &ex) { // LCOV_EXCL_START
		executor.PushError(PreservedError(ex));
	} catch (std::exception &ex) {
		executor.PushError(PreservedError(ex));
	} catch (...) {
		executor.PushError(PreservedError("Unknown exception in Finalize!"));
	} // LCOV_EXCL_STOP
}

void Pipeline::AddDependency(shared_ptr<Pipeline> &pipeline) {
	D_ASSERT(pipeline);
	dependencies.push_back(weak_ptr<Pipeline>(pipeline));
	pipeline->parents.push_back(weak_ptr<Pipeline>(shared_from_this()));
}

string Pipeline::ToString() const {
	TreeRenderer renderer;
	return renderer.ToString(*this);
}

void Pipeline::Print() const {
	Printer::Print(ToString());
}

void Pipeline::PrintDependencies() const {
	for (auto &dep : dependencies) {
		shared_ptr<Pipeline>(dep)->Print();
	}
}

vector<PhysicalOperator *> Pipeline::GetOperators() const {
	vector<PhysicalOperator *> result;
	D_ASSERT(source);
	result.push_back(source);
	result.insert(result.end(), operators.begin(), operators.end());
	if (sink) {
		result.push_back(sink);
	}
	return result;
}

//===--------------------------------------------------------------------===//
// Pipeline Build State
//===--------------------------------------------------------------------===//
void PipelineBuildState::SetPipelineSource(Pipeline &pipeline, PhysicalOperator *op) {
	pipeline.source = op;
}

void PipelineBuildState::SetPipelineSink(Pipeline &pipeline, PhysicalOperator *op, idx_t sink_pipeline_count) {
	pipeline.sink = op;
	// set the base batch index of this pipeline based on how many other pipelines have this node as their sink
	pipeline.base_batch_index = BATCH_INCREMENT * sink_pipeline_count;
}

void PipelineBuildState::AddPipelineOperator(Pipeline &pipeline, PhysicalOperator *op) {
	pipeline.operators.push_back(op);
}

PhysicalOperator *PipelineBuildState::GetPipelineSource(Pipeline &pipeline) {
	return pipeline.source;
}

PhysicalOperator *PipelineBuildState::GetPipelineSink(Pipeline &pipeline) {
	return pipeline.sink;
}

void PipelineBuildState::SetPipelineOperators(Pipeline &pipeline, vector<PhysicalOperator *> operators) {
	pipeline.operators = std::move(operators);
}

shared_ptr<Pipeline> PipelineBuildState::CreateChildPipeline(Executor &executor, Pipeline &pipeline,
                                                             PhysicalOperator *op) {
	return executor.CreateChildPipeline(&pipeline, op);
}

vector<PhysicalOperator *> PipelineBuildState::GetPipelineOperators(Pipeline &pipeline) {
	return pipeline.operators;
}

} // namespace duckdb



namespace duckdb {

PipelineCompleteEvent::PipelineCompleteEvent(Executor &executor, bool complete_pipeline_p)
    : Event(executor), complete_pipeline(complete_pipeline_p) {
}

void PipelineCompleteEvent::Schedule() {
}

void PipelineCompleteEvent::FinalizeFinish() {
	if (complete_pipeline) {
		executor.CompletePipeline();
	}
}

} // namespace duckdb



namespace duckdb {

PipelineEvent::PipelineEvent(shared_ptr<Pipeline> pipeline_p) : BasePipelineEvent(std::move(pipeline_p)) {
}

void PipelineEvent::Schedule() {
	auto event = shared_from_this();
	auto &executor = pipeline->executor;
	try {
		pipeline->Schedule(event);
		D_ASSERT(total_tasks > 0);
	} catch (Exception &ex) {
		executor.PushError(PreservedError(ex));
	} catch (std::exception &ex) {
		executor.PushError(PreservedError(ex));
	} catch (...) { // LCOV_EXCL_START
		executor.PushError(PreservedError("Unknown exception in Finalize!"));
	} // LCOV_EXCL_STOP
}

void PipelineEvent::FinishEvent() {
}

} // namespace duckdb




namespace duckdb {

PipelineExecutor::PipelineExecutor(ClientContext &context_p, Pipeline &pipeline_p)
    : pipeline(pipeline_p), thread(context_p), context(context_p, thread, &pipeline_p) {
	D_ASSERT(pipeline.source_state);
	local_source_state = pipeline.source->GetLocalSourceState(context, *pipeline.source_state);
	if (pipeline.sink) {
		local_sink_state = pipeline.sink->GetLocalSinkState(context);
		requires_batch_index = pipeline.sink->RequiresBatchIndex() && pipeline.source->SupportsBatchIndex();
	}

	intermediate_chunks.reserve(pipeline.operators.size());
	intermediate_states.reserve(pipeline.operators.size());
	for (idx_t i = 0; i < pipeline.operators.size(); i++) {
		auto prev_operator = i == 0 ? pipeline.source : pipeline.operators[i - 1];
		auto current_operator = pipeline.operators[i];

		auto chunk = make_unique<DataChunk>();
		chunk->Initialize(Allocator::Get(context.client), prev_operator->GetTypes());
		intermediate_chunks.push_back(std::move(chunk));

		auto op_state = current_operator->GetOperatorState(context);
		intermediate_states.push_back(std::move(op_state));

		if (current_operator->IsSink() && current_operator->sink_state->state == SinkFinalizeType::NO_OUTPUT_POSSIBLE) {
			// one of the operators has already figured out no output is possible
			// we can skip executing the pipeline
			FinishProcessing();
		}
	}
	InitializeChunk(final_chunk);
}

bool PipelineExecutor::Execute(idx_t max_chunks) {
	D_ASSERT(pipeline.sink);
	bool exhausted_source = false;
	auto &source_chunk = pipeline.operators.empty() ? final_chunk : *intermediate_chunks[0];
	for (idx_t i = 0; i < max_chunks; i++) {
		if (IsFinished()) {
			break;
		}
		source_chunk.Reset();
		FetchFromSource(source_chunk);
		if (source_chunk.size() == 0) {
			exhausted_source = true;
			break;
		}
		auto result = ExecutePushInternal(source_chunk);
		if (result == OperatorResultType::FINISHED) {
			D_ASSERT(IsFinished());
			break;
		}
	}
	if (!exhausted_source && !IsFinished()) {
		return false;
	}
	PushFinalize();
	return true;
}

void PipelineExecutor::Execute() {
	Execute(NumericLimits<idx_t>::Maximum());
}

OperatorResultType PipelineExecutor::ExecutePush(DataChunk &input) { // LCOV_EXCL_START
	return ExecutePushInternal(input);
} // LCOV_EXCL_STOP

void PipelineExecutor::FinishProcessing(int32_t operator_idx) {
	finished_processing_idx = operator_idx < 0 ? NumericLimits<int32_t>::Maximum() : operator_idx;
	in_process_operators = stack<idx_t>();
}

bool PipelineExecutor::IsFinished() {
	return finished_processing_idx >= 0;
}

OperatorResultType PipelineExecutor::ExecutePushInternal(DataChunk &input, idx_t initial_idx) {
	D_ASSERT(pipeline.sink);
	if (input.size() == 0) { // LCOV_EXCL_START
		return OperatorResultType::NEED_MORE_INPUT;
	} // LCOV_EXCL_STOP
	while (true) {
		OperatorResultType result;
		// Note: if input is the final_chunk, we don't do any executing, the chunk just needs to be sinked
		if (&input != &final_chunk) {
			final_chunk.Reset();
			result = Execute(input, final_chunk, initial_idx);
			if (result == OperatorResultType::FINISHED) {
				return OperatorResultType::FINISHED;
			}
		} else {
			result = OperatorResultType::NEED_MORE_INPUT;
		}
		auto &sink_chunk = final_chunk;
		if (sink_chunk.size() > 0) {
			StartOperator(pipeline.sink);
			D_ASSERT(pipeline.sink);
			D_ASSERT(pipeline.sink->sink_state);
			auto sink_result = pipeline.sink->Sink(context, *pipeline.sink->sink_state, *local_sink_state, sink_chunk);
			EndOperator(pipeline.sink, nullptr);
			if (sink_result == SinkResultType::FINISHED) {
				FinishProcessing();
				return OperatorResultType::FINISHED;
			}
		}
		if (result == OperatorResultType::NEED_MORE_INPUT) {
			return OperatorResultType::NEED_MORE_INPUT;
		}
	}
}

// Pull a single DataChunk from the pipeline by flushing any operators holding cached output
void PipelineExecutor::FlushCachingOperatorsPull(DataChunk &result) {
	idx_t start_idx = IsFinished() ? idx_t(finished_processing_idx) : 0;
	idx_t op_idx = start_idx;
	while (op_idx < pipeline.operators.size()) {
		if (!pipeline.operators[op_idx]->RequiresFinalExecute()) {
			op_idx++;
			continue;
		}

		OperatorFinalizeResultType finalize_result;
		DataChunk &curr_chunk =
		    op_idx + 1 >= intermediate_chunks.size() ? final_chunk : *intermediate_chunks[op_idx + 1];

		if (pending_final_execute) {
			// Still have a cached chunk from a last pull, reuse chunk
			finalize_result = cached_final_execute_result;
		} else {
			// Flush the current operator
			auto current_operator = pipeline.operators[op_idx];
			StartOperator(current_operator);
			finalize_result = current_operator->FinalExecute(context, curr_chunk, *current_operator->op_state,
			                                                 *intermediate_states[op_idx]);
			EndOperator(current_operator, &curr_chunk);
		}

		auto execute_result = Execute(curr_chunk, result, op_idx + 1);

		if (execute_result == OperatorResultType::HAVE_MORE_OUTPUT) {
			pending_final_execute = true;
			cached_final_execute_result = finalize_result;
		} else {
			pending_final_execute = false;
			if (finalize_result == OperatorFinalizeResultType::FINISHED) {
				FinishProcessing(op_idx);
				op_idx++;
			}
		}

		// Some non-empty result was pulled from some caching operator, we're done for this pull
		if (result.size() > 0) {
			break;
		}
	}
}

// Push all remaining cached operator output through the pipeline
void PipelineExecutor::FlushCachingOperatorsPush() {
	idx_t start_idx = IsFinished() ? idx_t(finished_processing_idx) : 0;
	for (idx_t op_idx = start_idx; op_idx < pipeline.operators.size(); op_idx++) {
		if (!pipeline.operators[op_idx]->RequiresFinalExecute()) {
			continue;
		}

		OperatorFinalizeResultType finalize_result;
		OperatorResultType push_result;

		do {
			auto &curr_chunk =
			    op_idx + 1 >= intermediate_chunks.size() ? final_chunk : *intermediate_chunks[op_idx + 1];
			auto current_operator = pipeline.operators[op_idx];
			StartOperator(current_operator);
			finalize_result = current_operator->FinalExecute(context, curr_chunk, *current_operator->op_state,
			                                                 *intermediate_states[op_idx]);
			EndOperator(current_operator, &curr_chunk);
			push_result = ExecutePushInternal(curr_chunk, op_idx + 1);
		} while (finalize_result != OperatorFinalizeResultType::FINISHED &&
		         push_result != OperatorResultType::FINISHED);

		if (push_result == OperatorResultType::FINISHED) {
			break;
		}
	}
}

void PipelineExecutor::PushFinalize() {
	if (finalized) {
		throw InternalException("Calling PushFinalize on a pipeline that has been finalized already");
	}
	finalized = true;
	// flush all caching operators
	// note that even if an operator has finished, we might still need to flush caches AFTER
	// that operator e.g. if we have SOURCE -> LIMIT -> CROSS_PRODUCT -> SINK, if the
	// LIMIT reports no more rows will be passed on we still need to flush caches from the CROSS_PRODUCT
	D_ASSERT(in_process_operators.empty());

	FlushCachingOperatorsPush();

	D_ASSERT(local_sink_state);
	// run the combine for the sink
	pipeline.sink->Combine(context, *pipeline.sink->sink_state, *local_sink_state);

	// flush all query profiler info
	for (idx_t i = 0; i < intermediate_states.size(); i++) {
		intermediate_states[i]->Finalize(pipeline.operators[i], context);
	}
	pipeline.executor.Flush(thread);
	local_sink_state.reset();
}

void PipelineExecutor::ExecutePull(DataChunk &result) {
	if (IsFinished()) {
		return;
	}
	auto &executor = pipeline.executor;
	try {
		D_ASSERT(!pipeline.sink);
		auto &source_chunk = pipeline.operators.empty() ? result : *intermediate_chunks[0];
		while (result.size() == 0) {
			if (source_empty) {
				FlushCachingOperatorsPull(result);
				break;
			}

			if (in_process_operators.empty()) {
				source_chunk.Reset();
				FetchFromSource(source_chunk);

				if (source_chunk.size() == 0) {
					source_empty = true;
					continue;
				}
			}

			if (!pipeline.operators.empty()) {
				auto state = Execute(source_chunk, result);
				if (state == OperatorResultType::FINISHED) {
					break;
				}
			}
		}
	} catch (const Exception &ex) { // LCOV_EXCL_START
		if (executor.HasError()) {
			executor.ThrowException();
		}
		throw;
	} catch (std::exception &ex) {
		if (executor.HasError()) {
			executor.ThrowException();
		}
		throw;
	} catch (...) {
		if (executor.HasError()) {
			executor.ThrowException();
		}
		throw;
	} // LCOV_EXCL_STOP
}

void PipelineExecutor::PullFinalize() {
	if (finalized) {
		throw InternalException("Calling PullFinalize on a pipeline that has been finalized already");
	}
	finalized = true;
	pipeline.executor.Flush(thread);
}

void PipelineExecutor::GoToSource(idx_t &current_idx, idx_t initial_idx) {
	// we go back to the first operator (the source)
	current_idx = initial_idx;
	if (!in_process_operators.empty()) {
		// ... UNLESS there is an in process operator
		// if there is an in-process operator, we start executing at the latest one
		// for example, if we have a join operator that has tuples left, we first need to emit those tuples
		current_idx = in_process_operators.top();
		in_process_operators.pop();
	}
	D_ASSERT(current_idx >= initial_idx);
}

OperatorResultType PipelineExecutor::Execute(DataChunk &input, DataChunk &result, idx_t initial_idx) {
	if (input.size() == 0) { // LCOV_EXCL_START
		return OperatorResultType::NEED_MORE_INPUT;
	} // LCOV_EXCL_STOP
	D_ASSERT(!pipeline.operators.empty());

	idx_t current_idx;
	GoToSource(current_idx, initial_idx);
	if (current_idx == initial_idx) {
		current_idx++;
	}
	if (current_idx > pipeline.operators.size()) {
		result.Reference(input);
		return OperatorResultType::NEED_MORE_INPUT;
	}
	while (true) {
		if (context.client.interrupted) {
			throw InterruptException();
		}
		// now figure out where to put the chunk
		// if current_idx is the last possible index (>= operators.size()) we write to the result
		// otherwise we write to an intermediate chunk
		auto current_intermediate = current_idx;
		auto &current_chunk =
		    current_intermediate >= intermediate_chunks.size() ? result : *intermediate_chunks[current_intermediate];
		current_chunk.Reset();
		if (current_idx == initial_idx) {
			// we went back to the source: we need more input
			return OperatorResultType::NEED_MORE_INPUT;
		} else {
			auto &prev_chunk =
			    current_intermediate == initial_idx + 1 ? input : *intermediate_chunks[current_intermediate - 1];
			auto operator_idx = current_idx - 1;
			auto current_operator = pipeline.operators[operator_idx];

			// if current_idx > source_idx, we pass the previous' operators output through the Execute of the current
			// operator
			StartOperator(current_operator);
			auto result = current_operator->Execute(context, prev_chunk, current_chunk, *current_operator->op_state,
			                                        *intermediate_states[current_intermediate - 1]);
			EndOperator(current_operator, &current_chunk);
			if (result == OperatorResultType::HAVE_MORE_OUTPUT) {
				// more data remains in this operator
				// push in-process marker
				in_process_operators.push(current_idx);
			} else if (result == OperatorResultType::FINISHED) {
				D_ASSERT(current_chunk.size() == 0);
				FinishProcessing(current_idx);
				return OperatorResultType::FINISHED;
			}
			current_chunk.Verify();
		}

		if (current_chunk.size() == 0) {
			// no output from this operator!
			if (current_idx == initial_idx) {
				// if we got no output from the scan, we are done
				break;
			} else {
				// if we got no output from an intermediate op
				// we go back and try to pull data from the source again
				GoToSource(current_idx, initial_idx);
				continue;
			}
		} else {
			// we got output! continue to the next operator
			current_idx++;
			if (current_idx > pipeline.operators.size()) {
				// if we got output and are at the last operator, we are finished executing for this output chunk
				// return the data and push it into the chunk
				break;
			}
		}
	}
	return in_process_operators.empty() ? OperatorResultType::NEED_MORE_INPUT : OperatorResultType::HAVE_MORE_OUTPUT;
}

void PipelineExecutor::FetchFromSource(DataChunk &result) {
	StartOperator(pipeline.source);
	pipeline.source->GetData(context, result, *pipeline.source_state, *local_source_state);
	if (result.size() != 0 && requires_batch_index) {
		auto next_batch_index =
		    pipeline.source->GetBatchIndex(context, result, *pipeline.source_state, *local_source_state);
		next_batch_index += pipeline.base_batch_index;
		D_ASSERT(local_sink_state->batch_index <= next_batch_index ||
		         local_sink_state->batch_index == DConstants::INVALID_INDEX);
		local_sink_state->batch_index = next_batch_index;
	}
	EndOperator(pipeline.source, &result);
}

void PipelineExecutor::InitializeChunk(DataChunk &chunk) {
	PhysicalOperator *last_op = pipeline.operators.empty() ? pipeline.source : pipeline.operators.back();
	chunk.Initialize(Allocator::DefaultAllocator(), last_op->GetTypes());
}

void PipelineExecutor::StartOperator(PhysicalOperator *op) {
	if (context.client.interrupted) {
		throw InterruptException();
	}
	context.thread.profiler.StartOperator(op);
}

void PipelineExecutor::EndOperator(PhysicalOperator *op, DataChunk *chunk) {
	context.thread.profiler.EndOperator(chunk);

	if (chunk) {
		chunk->Verify();
	}
}

} // namespace duckdb



namespace duckdb {

PipelineFinishEvent::PipelineFinishEvent(shared_ptr<Pipeline> pipeline_p) : BasePipelineEvent(std::move(pipeline_p)) {
}

void PipelineFinishEvent::Schedule() {
}

void PipelineFinishEvent::FinishEvent() {
	pipeline->Finalize(*this);
}

} // namespace duckdb




namespace duckdb {

PipelineInitializeEvent::PipelineInitializeEvent(shared_ptr<Pipeline> pipeline_p)
    : BasePipelineEvent(std::move(pipeline_p)) {
}

class PipelineInitializeTask : public ExecutorTask {
public:
	explicit PipelineInitializeTask(Pipeline &pipeline_p, shared_ptr<Event> event_p)
	    : ExecutorTask(pipeline_p.executor), pipeline(pipeline_p), event(std::move(event_p)) {
	}

	Pipeline &pipeline;
	shared_ptr<Event> event;

public:
	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {
		pipeline.ResetSink();
		event->FinishTask();
		return TaskExecutionResult::TASK_FINISHED;
	}
};

void PipelineInitializeEvent::Schedule() {
	// needs to spawn a task to get the chain of tasks for the query plan going
	vector<unique_ptr<Task>> tasks;
	tasks.push_back(make_unique<PipelineInitializeTask>(*pipeline, shared_from_this()));
	SetTasks(std::move(tasks));
}

void PipelineInitializeEvent::FinishEvent() {
}

} // namespace duckdb






#ifndef DUCKDB_NO_THREADS



#else
#include <queue>
#endif

namespace duckdb {

struct SchedulerThread {
#ifndef DUCKDB_NO_THREADS
	explicit SchedulerThread(unique_ptr<thread> thread_p) : internal_thread(std::move(thread_p)) {
	}

	unique_ptr<thread> internal_thread;
#endif
};

#ifndef DUCKDB_NO_THREADS
typedef duckdb_moodycamel::ConcurrentQueue<unique_ptr<Task>> concurrent_queue_t;
typedef duckdb_moodycamel::LightweightSemaphore lightweight_semaphore_t;

struct ConcurrentQueue {
	concurrent_queue_t q;
	lightweight_semaphore_t semaphore;

	void Enqueue(ProducerToken &token, unique_ptr<Task> task);
	bool DequeueFromProducer(ProducerToken &token, unique_ptr<Task> &task);
};

struct QueueProducerToken {
	explicit QueueProducerToken(ConcurrentQueue &queue) : queue_token(queue.q) {
	}

	duckdb_moodycamel::ProducerToken queue_token;
};

void ConcurrentQueue::Enqueue(ProducerToken &token, unique_ptr<Task> task) {
	lock_guard<mutex> producer_lock(token.producer_lock);
	if (q.enqueue(token.token->queue_token, std::move(task))) {
		semaphore.signal();
	} else {
		throw InternalException("Could not schedule task!");
	}
}

bool ConcurrentQueue::DequeueFromProducer(ProducerToken &token, unique_ptr<Task> &task) {
	lock_guard<mutex> producer_lock(token.producer_lock);
	return q.try_dequeue_from_producer(token.token->queue_token, task);
}

#else
struct ConcurrentQueue {
	std::queue<std::unique_ptr<Task>> q;
	mutex qlock;

	void Enqueue(ProducerToken &token, unique_ptr<Task> task);
	bool DequeueFromProducer(ProducerToken &token, unique_ptr<Task> &task);
};

void ConcurrentQueue::Enqueue(ProducerToken &token, unique_ptr<Task> task) {
	lock_guard<mutex> lock(qlock);
	q.push(std::move(task));
}

bool ConcurrentQueue::DequeueFromProducer(ProducerToken &token, unique_ptr<Task> &task) {
	lock_guard<mutex> lock(qlock);
	if (q.empty()) {
		return false;
	}
	task = std::move(q.front());
	q.pop();
	return true;
}

struct QueueProducerToken {
	QueueProducerToken(ConcurrentQueue &queue) {
	}
};
#endif

ProducerToken::ProducerToken(TaskScheduler &scheduler, unique_ptr<QueueProducerToken> token)
    : scheduler(scheduler), token(std::move(token)) {
}

ProducerToken::~ProducerToken() {
}

TaskScheduler::TaskScheduler(DatabaseInstance &db) : db(db), queue(make_unique<ConcurrentQueue>()) {
}

TaskScheduler::~TaskScheduler() {
#ifndef DUCKDB_NO_THREADS
	SetThreadsInternal(1);
#endif
}

TaskScheduler &TaskScheduler::GetScheduler(ClientContext &context) {
	return TaskScheduler::GetScheduler(DatabaseInstance::GetDatabase(context));
}

TaskScheduler &TaskScheduler::GetScheduler(DatabaseInstance &db) {
	return db.GetScheduler();
}

unique_ptr<ProducerToken> TaskScheduler::CreateProducer() {
	auto token = make_unique<QueueProducerToken>(*queue);
	return make_unique<ProducerToken>(*this, std::move(token));
}

void TaskScheduler::ScheduleTask(ProducerToken &token, unique_ptr<Task> task) {
	// Enqueue a task for the given producer token and signal any sleeping threads
	queue->Enqueue(token, std::move(task));
}

bool TaskScheduler::GetTaskFromProducer(ProducerToken &token, unique_ptr<Task> &task) {
	return queue->DequeueFromProducer(token, task);
}

void TaskScheduler::ExecuteForever(atomic<bool> *marker) {
#ifndef DUCKDB_NO_THREADS
	unique_ptr<Task> task;
	// loop until the marker is set to false
	while (*marker) {
		// wait for a signal with a timeout
		queue->semaphore.wait();
		if (queue->q.try_dequeue(task)) {
			task->Execute(TaskExecutionMode::PROCESS_ALL);
			task.reset();
		}
	}
#else
	throw NotImplementedException("DuckDB was compiled without threads! Background thread loop is not allowed.");
#endif
}

idx_t TaskScheduler::ExecuteTasks(atomic<bool> *marker, idx_t max_tasks) {
#ifndef DUCKDB_NO_THREADS
	idx_t completed_tasks = 0;
	// loop until the marker is set to false
	while (*marker && completed_tasks < max_tasks) {
		unique_ptr<Task> task;
		if (!queue->q.try_dequeue(task)) {
			return completed_tasks;
		}
		task->Execute(TaskExecutionMode::PROCESS_ALL);
		task.reset();
		completed_tasks++;
	}
	return completed_tasks;
#else
	throw NotImplementedException("DuckDB was compiled without threads! Background thread loop is not allowed.");
#endif
}

void TaskScheduler::ExecuteTasks(idx_t max_tasks) {
#ifndef DUCKDB_NO_THREADS
	unique_ptr<Task> task;
	for (idx_t i = 0; i < max_tasks; i++) {
		queue->semaphore.wait(TASK_TIMEOUT_USECS);
		if (!queue->q.try_dequeue(task)) {
			return;
		}
		try {
			task->Execute(TaskExecutionMode::PROCESS_ALL);
			task.reset();
		} catch (...) {
			return;
		}
	}
#else
	throw NotImplementedException("DuckDB was compiled without threads! Background thread loop is not allowed.");
#endif
}

#ifndef DUCKDB_NO_THREADS
static void ThreadExecuteTasks(TaskScheduler *scheduler, atomic<bool> *marker) {
	scheduler->ExecuteForever(marker);
}
#endif

int32_t TaskScheduler::NumberOfThreads() {
	lock_guard<mutex> t(thread_lock);
	auto &config = DBConfig::GetConfig(db);
	return threads.size() + config.options.external_threads + 1;
}

void TaskScheduler::SetThreads(int32_t n) {
#ifndef DUCKDB_NO_THREADS
	lock_guard<mutex> t(thread_lock);
	if (n < 1) {
		throw SyntaxException("Must have at least 1 thread!");
	}
	SetThreadsInternal(n);
#else
	if (n != 1) {
		throw NotImplementedException("DuckDB was compiled without threads! Setting threads > 1 is not allowed.");
	}
#endif
}

void TaskScheduler::Signal(idx_t n) {
#ifndef DUCKDB_NO_THREADS
	queue->semaphore.signal(n);
#endif
}

void TaskScheduler::SetThreadsInternal(int32_t n) {
#ifndef DUCKDB_NO_THREADS
	if (threads.size() == idx_t(n - 1)) {
		return;
	}
	idx_t new_thread_count = n - 1;
	if (threads.size() > new_thread_count) {
		// we are reducing the number of threads: clear all threads first
		for (idx_t i = 0; i < threads.size(); i++) {
			*markers[i] = false;
		}
		Signal(threads.size());
		// now join the threads to ensure they are fully stopped before erasing them
		for (idx_t i = 0; i < threads.size(); i++) {
			threads[i]->internal_thread->join();
		}
		// erase the threads/markers
		threads.clear();
		markers.clear();
	}
	if (threads.size() < new_thread_count) {
		// we are increasing the number of threads: launch them and run tasks on them
		idx_t create_new_threads = new_thread_count - threads.size();
		for (idx_t i = 0; i < create_new_threads; i++) {
			// launch a thread and assign it a cancellation marker
			auto marker = unique_ptr<atomic<bool>>(new atomic<bool>(true));
			auto worker_thread = make_unique<thread>(ThreadExecuteTasks, this, marker.get());
			auto thread_wrapper = make_unique<SchedulerThread>(std::move(worker_thread));

			threads.push_back(std::move(thread_wrapper));
			markers.push_back(std::move(marker));
		}
	}
#endif
}

} // namespace duckdb




namespace duckdb {

ThreadContext::ThreadContext(ClientContext &context) : profiler(QueryProfiler::Get(context).IsEnabled()) {
}

} // namespace duckdb




namespace duckdb {

void BaseExpression::Print() const {
	Printer::Print(ToString());
}

string BaseExpression::GetName() const {
	return !alias.empty() ? alias : ToString();
}

bool BaseExpression::Equals(const BaseExpression *other) const {
	if (!other) {
		return false;
	}
	if (this->expression_class != other->expression_class || this->type != other->type) {
		return false;
	}
	return true;
}

void BaseExpression::Verify() const {
}

} // namespace duckdb







namespace duckdb {

ColumnDefinition::ColumnDefinition(string name_p, LogicalType type_p)
    : name(std::move(name_p)), type(std::move(type_p)) {
}

ColumnDefinition::ColumnDefinition(string name_p, LogicalType type_p, unique_ptr<ParsedExpression> expression,
                                   TableColumnType category)
    : name(std::move(name_p)), type(std::move(type_p)), category(category) {
	switch (category) {
	case TableColumnType::STANDARD: {
		default_value = std::move(expression);
		break;
	}
	case TableColumnType::GENERATED: {
		generated_expression = std::move(expression);
		break;
	}
	default: {
		throw InternalException("Type not implemented for TableColumnType");
	}
	}
}

ColumnDefinition ColumnDefinition::Copy() const {
	ColumnDefinition copy(name, type);
	copy.oid = oid;
	copy.storage_oid = storage_oid;
	copy.SetDefaultValue(default_value ? default_value->Copy() : nullptr);
	copy.generated_expression = generated_expression ? generated_expression->Copy() : nullptr;
	copy.compression_type = compression_type;
	copy.category = category;
	return copy;
}

void ColumnDefinition::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteString(name);
	writer.WriteSerializable(type);
	if (Generated()) {
		writer.WriteOptional(generated_expression);
	} else {
		writer.WriteOptional(default_value);
	}
	writer.WriteField<TableColumnType>(category);
	writer.Finalize();
}

ColumnDefinition ColumnDefinition::Deserialize(Deserializer &source) {
	FieldReader reader(source);
	auto column_name = reader.ReadRequired<string>();
	auto column_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto expression = reader.ReadOptional<ParsedExpression>(nullptr);
	auto category = reader.ReadField<TableColumnType>(TableColumnType::STANDARD);
	reader.Finalize();

	switch (category) {
	case TableColumnType::STANDARD:
		return ColumnDefinition(column_name, column_type, std::move(expression), TableColumnType::STANDARD);
	case TableColumnType::GENERATED:
		return ColumnDefinition(column_name, column_type, std::move(expression), TableColumnType::GENERATED);
	default:
		throw NotImplementedException("Type not implemented for TableColumnType");
	}
}

const unique_ptr<ParsedExpression> &ColumnDefinition::DefaultValue() const {
	return default_value;
}

void ColumnDefinition::SetDefaultValue(unique_ptr<ParsedExpression> default_value) {
	this->default_value = std::move(default_value);
}

const LogicalType &ColumnDefinition::Type() const {
	return type;
}

LogicalType &ColumnDefinition::TypeMutable() {
	return type;
}

void ColumnDefinition::SetType(const LogicalType &type) {
	this->type = type;
}

const string &ColumnDefinition::Name() const {
	return name;
}

void ColumnDefinition::SetName(const string &name) {
	this->name = name;
}

const duckdb::CompressionType &ColumnDefinition::CompressionType() const {
	return compression_type;
}

void ColumnDefinition::SetCompressionType(duckdb::CompressionType compression_type) {
	this->compression_type = compression_type;
}

const storage_t &ColumnDefinition::StorageOid() const {
	return storage_oid;
}

LogicalIndex ColumnDefinition::Logical() const {
	return LogicalIndex(oid);
}

PhysicalIndex ColumnDefinition::Physical() const {
	return PhysicalIndex(storage_oid);
}

void ColumnDefinition::SetStorageOid(storage_t storage_oid) {
	this->storage_oid = storage_oid;
}

const column_t &ColumnDefinition::Oid() const {
	return oid;
}

void ColumnDefinition::SetOid(column_t oid) {
	this->oid = oid;
}

const TableColumnType &ColumnDefinition::Category() const {
	return category;
}

bool ColumnDefinition::Generated() const {
	return category == TableColumnType::GENERATED;
}

//===--------------------------------------------------------------------===//
// Generated Columns (VIRTUAL)
//===--------------------------------------------------------------------===//

static void VerifyColumnRefs(ParsedExpression &expr) {
	if (expr.type == ExpressionType::COLUMN_REF) {
		auto &column_ref = (ColumnRefExpression &)expr;
		if (column_ref.IsQualified()) {
			throw ParserException(
			    "Qualified (tbl.name) column references are not allowed inside of generated column expressions");
		}
	}
	ParsedExpressionIterator::EnumerateChildren(
	    expr, [&](const ParsedExpression &child) { VerifyColumnRefs((ParsedExpression &)child); });
}

static void InnerGetListOfDependencies(ParsedExpression &expr, vector<string> &dependencies) {
	if (expr.type == ExpressionType::COLUMN_REF) {
		auto columnref = (ColumnRefExpression &)expr;
		auto &name = columnref.GetColumnName();
		dependencies.push_back(name);
	}
	ParsedExpressionIterator::EnumerateChildren(expr, [&](const ParsedExpression &child) {
		if (expr.type == ExpressionType::LAMBDA) {
			throw NotImplementedException("Lambda functions are currently not supported in generated columns.");
		}
		InnerGetListOfDependencies((ParsedExpression &)child, dependencies);
	});
}

void ColumnDefinition::GetListOfDependencies(vector<string> &dependencies) const {
	D_ASSERT(Generated());
	InnerGetListOfDependencies(*generated_expression, dependencies);
}

string ColumnDefinition::GetName() const {
	return name;
}

LogicalType ColumnDefinition::GetType() const {
	return type;
}

void ColumnDefinition::SetGeneratedExpression(unique_ptr<ParsedExpression> expression) {
	category = TableColumnType::GENERATED;

	if (expression->HasSubquery()) {
		throw ParserException("Expression of generated column \"%s\" contains a subquery, which isn't allowed", name);
	}

	VerifyColumnRefs(*expression);
	if (type.id() == LogicalTypeId::ANY) {
		generated_expression = std::move(expression);
		return;
	}
	// Always wrap the expression in a cast, that way we can always update the cast when we change the type
	// Except if the type is LogicalType::ANY (no type specified)
	generated_expression = make_unique_base<ParsedExpression, CastExpression>(type, std::move(expression));
}

void ColumnDefinition::ChangeGeneratedExpressionType(const LogicalType &type) {
	D_ASSERT(Generated());
	// First time the type is set, add a cast around the expression
	D_ASSERT(this->type.id() == LogicalTypeId::ANY);
	generated_expression = make_unique_base<ParsedExpression, CastExpression>(type, std::move(generated_expression));
	// Every generated expression should be wrapped in a cast on creation
	// D_ASSERT(generated_expression->type == ExpressionType::OPERATOR_CAST);
	// auto &cast_expr = (CastExpression &)*generated_expression;
	// auto base_expr = std::move(cast_expr.child);
	// generated_expression = make_unique_base<ParsedExpression, CastExpression>(type, std::move(base_expr));
}

const ParsedExpression &ColumnDefinition::GeneratedExpression() const {
	D_ASSERT(Generated());
	return *generated_expression;
}

ParsedExpression &ColumnDefinition::GeneratedExpressionMutable() {
	D_ASSERT(Generated());
	return *generated_expression;
}

} // namespace duckdb




namespace duckdb {

ColumnList::ColumnList(bool allow_duplicate_names) : allow_duplicate_names(allow_duplicate_names) {
}
void ColumnList::AddColumn(ColumnDefinition column) {
	auto oid = columns.size();
	if (!column.Generated()) {
		column.SetStorageOid(physical_columns.size());
		physical_columns.push_back(oid);
	} else {
		column.SetStorageOid(DConstants::INVALID_INDEX);
	}
	column.SetOid(columns.size());
	AddToNameMap(column);
	columns.push_back(std::move(column));
}

void ColumnList::Finalize() {
	// add the "rowid" alias, if there is no rowid column specified in the table
	if (name_map.find("rowid") == name_map.end()) {
		name_map["rowid"] = COLUMN_IDENTIFIER_ROW_ID;
	}
}

void ColumnList::AddToNameMap(ColumnDefinition &col) {
	if (allow_duplicate_names) {
		idx_t index = 1;
		string base_name = col.Name();
		while (name_map.find(col.Name()) != name_map.end()) {
			col.SetName(base_name + ":" + to_string(index++));
		}
	} else {
		if (name_map.find(col.Name()) != name_map.end()) {
			throw CatalogException("Column with name %s already exists!", col.Name());
		}
	}
	name_map[col.Name()] = col.Oid();
}

ColumnDefinition &ColumnList::GetColumnMutable(LogicalIndex logical) {
	if (logical.index >= columns.size()) {
		throw InternalException("Logical column index %lld out of range", logical.index);
	}
	return columns[logical.index];
}

ColumnDefinition &ColumnList::GetColumnMutable(PhysicalIndex physical) {
	if (physical.index >= physical_columns.size()) {
		throw InternalException("Physical column index %lld out of range", physical.index);
	}
	auto logical_index = physical_columns[physical.index];
	D_ASSERT(logical_index < columns.size());
	return columns[logical_index];
}

ColumnDefinition &ColumnList::GetColumnMutable(const string &name) {
	auto entry = name_map.find(name);
	if (entry == name_map.end()) {
		throw InternalException("Column with name \"%s\" does not exist", name);
	}
	auto logical_index = entry->second;
	D_ASSERT(logical_index < columns.size());
	return columns[logical_index];
}

const ColumnDefinition &ColumnList::GetColumn(LogicalIndex logical) const {
	if (logical.index >= columns.size()) {
		throw InternalException("Logical column index %lld out of range", logical.index);
	}
	return columns[logical.index];
}

const ColumnDefinition &ColumnList::GetColumn(PhysicalIndex physical) const {
	if (physical.index >= physical_columns.size()) {
		throw InternalException("Physical column index %lld out of range", physical.index);
	}
	auto logical_index = physical_columns[physical.index];
	D_ASSERT(logical_index < columns.size());
	return columns[logical_index];
}

const ColumnDefinition &ColumnList::GetColumn(const string &name) const {
	auto entry = name_map.find(name);
	if (entry == name_map.end()) {
		throw InternalException("Column with name \"%s\" does not exist", name);
	}
	auto logical_index = entry->second;
	D_ASSERT(logical_index < columns.size());
	return columns[logical_index];
}

vector<string> ColumnList::GetColumnNames() const {
	vector<string> names;
	names.reserve(columns.size());
	for (auto &column : columns) {
		names.push_back(column.Name());
	}
	return names;
}

vector<LogicalType> ColumnList::GetColumnTypes() const {
	vector<LogicalType> types;
	types.reserve(columns.size());
	for (auto &column : columns) {
		types.push_back(column.Type());
	}
	return types;
}

bool ColumnList::ColumnExists(const string &name) const {
	auto entry = name_map.find(name);
	return entry != name_map.end();
}

PhysicalIndex ColumnList::LogicalToPhysical(LogicalIndex logical) const {
	auto &column = GetColumn(logical);
	if (column.Generated()) {
		throw InternalException("Column at position %d is not a physical column", logical.index);
	}
	return column.Physical();
}

LogicalIndex ColumnList::PhysicalToLogical(PhysicalIndex index) const {
	auto &column = GetColumn(index);
	return column.Logical();
}

LogicalIndex ColumnList::GetColumnIndex(string &column_name) const {
	auto entry = name_map.find(column_name);
	if (entry == name_map.end()) {
		return LogicalIndex(DConstants::INVALID_INDEX);
	}
	if (entry->second == COLUMN_IDENTIFIER_ROW_ID) {
		column_name = "rowid";
		return LogicalIndex(COLUMN_IDENTIFIER_ROW_ID);
	}
	column_name = columns[entry->second].Name();
	return LogicalIndex(entry->second);
}

ColumnList ColumnList::Copy() const {
	ColumnList result(allow_duplicate_names);
	for (auto &col : columns) {
		result.AddColumn(col.Copy());
	}
	return result;
}

void ColumnList::Serialize(FieldWriter &writer) const {
	writer.WriteRegularSerializableList(columns);
}

ColumnList ColumnList::Deserialize(FieldReader &reader) {
	ColumnList result;
	auto columns = reader.ReadRequiredSerializableList<ColumnDefinition, ColumnDefinition>();
	for (auto &col : columns) {
		result.AddColumn(std::move(col));
	}
	return result;
}

ColumnList::ColumnListIterator ColumnList::Logical() const {
	return ColumnListIterator(*this, false);
}

ColumnList::ColumnListIterator ColumnList::Physical() const {
	return ColumnListIterator(*this, true);
}

} // namespace duckdb







namespace duckdb {

Constraint::Constraint(ConstraintType type) : type(type) {
}

Constraint::~Constraint() {
}

void Constraint::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteField<ConstraintType>(type);
	Serialize(writer);
	writer.Finalize();
}

unique_ptr<Constraint> Constraint::Deserialize(Deserializer &source) {
	FieldReader reader(source);
	auto type = reader.ReadRequired<ConstraintType>();
	unique_ptr<Constraint> result;
	switch (type) {
	case ConstraintType::NOT_NULL:
		result = NotNullConstraint::Deserialize(reader);
		break;
	case ConstraintType::CHECK:
		result = CheckConstraint::Deserialize(reader);
		break;
	case ConstraintType::UNIQUE:
		result = UniqueConstraint::Deserialize(reader);
		break;
	case ConstraintType::FOREIGN_KEY:
		result = ForeignKeyConstraint::Deserialize(reader);
		break;
	default:
		throw InternalException("Unrecognized constraint type for serialization");
	}
	reader.Finalize();
	return result;
}

void Constraint::Print() const {
	Printer::Print(ToString());
}

} // namespace duckdb




namespace duckdb {

CheckConstraint::CheckConstraint(unique_ptr<ParsedExpression> expression)
    : Constraint(ConstraintType::CHECK), expression(std::move(expression)) {
}

string CheckConstraint::ToString() const {
	return "CHECK(" + expression->ToString() + ")";
}

unique_ptr<Constraint> CheckConstraint::Copy() const {
	return make_unique<CheckConstraint>(expression->Copy());
}

void CheckConstraint::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*expression);
}

unique_ptr<Constraint> CheckConstraint::Deserialize(FieldReader &source) {
	auto expression = source.ReadRequiredSerializable<ParsedExpression>();
	return make_unique<CheckConstraint>(std::move(expression));
}

} // namespace duckdb






namespace duckdb {

ForeignKeyConstraint::ForeignKeyConstraint(vector<string> pk_columns, vector<string> fk_columns, ForeignKeyInfo info)
    : Constraint(ConstraintType::FOREIGN_KEY), pk_columns(std::move(pk_columns)), fk_columns(std::move(fk_columns)),
      info(std::move(info)) {
}

string ForeignKeyConstraint::ToString() const {
	if (info.type == ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE) {
		string base = "FOREIGN KEY (";

		for (idx_t i = 0; i < fk_columns.size(); i++) {
			if (i > 0) {
				base += ", ";
			}
			base += KeywordHelper::WriteOptionallyQuoted(fk_columns[i]);
		}
		base += ") REFERENCES ";
		if (!info.schema.empty()) {
			base += info.schema;
			base += ".";
		}
		base += info.table;
		base += "(";

		for (idx_t i = 0; i < pk_columns.size(); i++) {
			if (i > 0) {
				base += ", ";
			}
			base += KeywordHelper::WriteOptionallyQuoted(pk_columns[i]);
		}
		base += ")";

		return base;
	}

	return "";
}

unique_ptr<Constraint> ForeignKeyConstraint::Copy() const {
	return make_unique<ForeignKeyConstraint>(pk_columns, fk_columns, info);
}

void ForeignKeyConstraint::Serialize(FieldWriter &writer) const {
	D_ASSERT(pk_columns.size() <= NumericLimits<uint32_t>::Maximum());
	writer.WriteList<string>(pk_columns);
	D_ASSERT(fk_columns.size() <= NumericLimits<uint32_t>::Maximum());
	writer.WriteList<string>(fk_columns);
	writer.WriteField<ForeignKeyType>(info.type);
	writer.WriteString(info.schema);
	writer.WriteString(info.table);
	writer.WriteIndexList<PhysicalIndex>(info.pk_keys);
	writer.WriteIndexList<PhysicalIndex>(info.fk_keys);
}

unique_ptr<Constraint> ForeignKeyConstraint::Deserialize(FieldReader &source) {
	ForeignKeyInfo read_info;
	auto pk_columns = source.ReadRequiredList<string>();
	auto fk_columns = source.ReadRequiredList<string>();
	read_info.type = source.ReadRequired<ForeignKeyType>();
	read_info.schema = source.ReadRequired<string>();
	read_info.table = source.ReadRequired<string>();
	read_info.pk_keys = source.ReadRequiredIndexList<PhysicalIndex>();
	read_info.fk_keys = source.ReadRequiredIndexList<PhysicalIndex>();

	// column list parsed constraint
	return make_unique<ForeignKeyConstraint>(pk_columns, fk_columns, std::move(read_info));
}

} // namespace duckdb




namespace duckdb {

NotNullConstraint::NotNullConstraint(LogicalIndex index) : Constraint(ConstraintType::NOT_NULL), index(index) {
}

NotNullConstraint::~NotNullConstraint() {
}

string NotNullConstraint::ToString() const {
	return "NOT NULL";
}

unique_ptr<Constraint> NotNullConstraint::Copy() const {
	return make_unique<NotNullConstraint>(index);
}

void NotNullConstraint::Serialize(FieldWriter &writer) const {
	writer.WriteField<idx_t>(index.index);
}

unique_ptr<Constraint> NotNullConstraint::Deserialize(FieldReader &source) {
	auto index = source.ReadRequired<idx_t>();
	return make_unique_base<Constraint, NotNullConstraint>(LogicalIndex(index));
}

} // namespace duckdb






namespace duckdb {

UniqueConstraint::UniqueConstraint(LogicalIndex index, bool is_primary_key)
    : Constraint(ConstraintType::UNIQUE), index(index), is_primary_key(is_primary_key) {
}
UniqueConstraint::UniqueConstraint(vector<string> columns, bool is_primary_key)
    : Constraint(ConstraintType::UNIQUE), index(DConstants::INVALID_INDEX), columns(std::move(columns)),
      is_primary_key(is_primary_key) {
}

string UniqueConstraint::ToString() const {
	string base = is_primary_key ? "PRIMARY KEY(" : "UNIQUE(";
	for (idx_t i = 0; i < columns.size(); i++) {
		if (i > 0) {
			base += ", ";
		}
		base += KeywordHelper::WriteOptionallyQuoted(columns[i]);
	}
	return base + ")";
}

unique_ptr<Constraint> UniqueConstraint::Copy() const {
	if (index.index == DConstants::INVALID_INDEX) {
		return make_unique<UniqueConstraint>(columns, is_primary_key);
	} else {
		auto result = make_unique<UniqueConstraint>(index, is_primary_key);
		result->columns = columns;
		return std::move(result);
	}
}

void UniqueConstraint::Serialize(FieldWriter &writer) const {
	writer.WriteField<bool>(is_primary_key);
	writer.WriteField<uint64_t>(index.index);
	D_ASSERT(columns.size() <= NumericLimits<uint32_t>::Maximum());
	writer.WriteList<string>(columns);
}

unique_ptr<Constraint> UniqueConstraint::Deserialize(FieldReader &source) {
	auto is_primary_key = source.ReadRequired<bool>();
	auto index = source.ReadRequired<uint64_t>();
	auto columns = source.ReadRequiredList<string>();

	if (index != DConstants::INVALID_INDEX) {
		// single column parsed constraint
		auto result = make_unique<UniqueConstraint>(LogicalIndex(index), is_primary_key);
		result->columns = std::move(columns);
		return std::move(result);
	} else {
		// column list parsed constraint
		return make_unique<UniqueConstraint>(std::move(columns), is_primary_key);
	}
}

} // namespace duckdb



namespace duckdb {

BetweenExpression::BetweenExpression(unique_ptr<ParsedExpression> input_p, unique_ptr<ParsedExpression> lower_p,
                                     unique_ptr<ParsedExpression> upper_p)
    : ParsedExpression(ExpressionType::COMPARE_BETWEEN, ExpressionClass::BETWEEN), input(std::move(input_p)),
      lower(std::move(lower_p)), upper(std::move(upper_p)) {
}

string BetweenExpression::ToString() const {
	return ToString<BetweenExpression, ParsedExpression>(*this);
}

bool BetweenExpression::Equal(const BetweenExpression *a, const BetweenExpression *b) {
	if (!a->input->Equals(b->input.get())) {
		return false;
	}
	if (!a->lower->Equals(b->lower.get())) {
		return false;
	}
	if (!a->upper->Equals(b->upper.get())) {
		return false;
	}
	return true;
}

unique_ptr<ParsedExpression> BetweenExpression::Copy() const {
	auto copy = make_unique<BetweenExpression>(input->Copy(), lower->Copy(), upper->Copy());
	copy->CopyProperties(*this);
	return std::move(copy);
}

void BetweenExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*input);
	writer.WriteSerializable(*lower);
	writer.WriteSerializable(*upper);
}

unique_ptr<ParsedExpression> BetweenExpression::Deserialize(ExpressionType type, FieldReader &source) {
	auto input = source.ReadRequiredSerializable<ParsedExpression>();
	auto lower = source.ReadRequiredSerializable<ParsedExpression>();
	auto upper = source.ReadRequiredSerializable<ParsedExpression>();
	return make_unique<BetweenExpression>(std::move(input), std::move(lower), std::move(upper));
}

} // namespace duckdb





namespace duckdb {

CaseExpression::CaseExpression() : ParsedExpression(ExpressionType::CASE_EXPR, ExpressionClass::CASE) {
}

string CaseExpression::ToString() const {
	return ToString<CaseExpression, ParsedExpression>(*this);
}

bool CaseExpression::Equal(const CaseExpression *a, const CaseExpression *b) {
	if (a->case_checks.size() != b->case_checks.size()) {
		return false;
	}
	for (idx_t i = 0; i < a->case_checks.size(); i++) {
		if (!a->case_checks[i].when_expr->Equals(b->case_checks[i].when_expr.get())) {
			return false;
		}
		if (!a->case_checks[i].then_expr->Equals(b->case_checks[i].then_expr.get())) {
			return false;
		}
	}
	if (!a->else_expr->Equals(b->else_expr.get())) {
		return false;
	}
	return true;
}

unique_ptr<ParsedExpression> CaseExpression::Copy() const {
	auto copy = make_unique<CaseExpression>();
	copy->CopyProperties(*this);
	for (auto &check : case_checks) {
		CaseCheck new_check;
		new_check.when_expr = check.when_expr->Copy();
		new_check.then_expr = check.then_expr->Copy();
		copy->case_checks.push_back(std::move(new_check));
	}
	copy->else_expr = else_expr->Copy();
	return std::move(copy);
}

void CaseExpression::Serialize(FieldWriter &writer) const {
	auto &serializer = writer.GetSerializer();
	// we write a list of multiple expressions here
	// in order to write this as a single field we directly use the field writers' internal serializer
	writer.WriteField<uint32_t>(case_checks.size());
	for (auto &check : case_checks) {
		check.when_expr->Serialize(serializer);
		check.then_expr->Serialize(serializer);
	}
	writer.WriteSerializable<ParsedExpression>(*else_expr);
}

unique_ptr<ParsedExpression> CaseExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto result = make_unique<CaseExpression>();
	auto &source = reader.GetSource();
	auto count = reader.ReadRequired<uint32_t>();
	for (idx_t i = 0; i < count; i++) {
		CaseCheck new_check;
		new_check.when_expr = ParsedExpression::Deserialize(source);
		new_check.then_expr = ParsedExpression::Deserialize(source);
		result->case_checks.push_back(std::move(new_check));
	}
	result->else_expr = reader.ReadRequiredSerializable<ParsedExpression>();
	return std::move(result);
}

} // namespace duckdb





namespace duckdb {

CastExpression::CastExpression(LogicalType target, unique_ptr<ParsedExpression> child, bool try_cast_p)
    : ParsedExpression(ExpressionType::OPERATOR_CAST, ExpressionClass::CAST), cast_type(std::move(target)),
      try_cast(try_cast_p) {
	D_ASSERT(child);
	this->child = std::move(child);
}

string CastExpression::ToString() const {
	return ToString<CastExpression, ParsedExpression>(*this);
}

bool CastExpression::Equal(const CastExpression *a, const CastExpression *b) {
	if (!a->child->Equals(b->child.get())) {
		return false;
	}
	if (a->cast_type != b->cast_type) {
		return false;
	}
	if (a->try_cast != b->try_cast) {
		return false;
	}
	return true;
}

unique_ptr<ParsedExpression> CastExpression::Copy() const {
	auto copy = make_unique<CastExpression>(cast_type, child->Copy(), try_cast);
	copy->CopyProperties(*this);
	return std::move(copy);
}

void CastExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*child);
	writer.WriteSerializable(cast_type);
	writer.WriteField<bool>(try_cast);
}

unique_ptr<ParsedExpression> CastExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto child = reader.ReadRequiredSerializable<ParsedExpression>();
	auto cast_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto try_cast = reader.ReadRequired<bool>();
	return make_unique_base<ParsedExpression, CastExpression>(cast_type, std::move(child), try_cast);
}

} // namespace duckdb





namespace duckdb {

CollateExpression::CollateExpression(string collation_p, unique_ptr<ParsedExpression> child)
    : ParsedExpression(ExpressionType::COLLATE, ExpressionClass::COLLATE), collation(std::move(collation_p)) {
	D_ASSERT(child);
	this->child = std::move(child);
}

string CollateExpression::ToString() const {
	return child->ToString() + " COLLATE " + KeywordHelper::WriteOptionallyQuoted(collation);
}

bool CollateExpression::Equal(const CollateExpression *a, const CollateExpression *b) {
	if (!a->child->Equals(b->child.get())) {
		return false;
	}
	if (a->collation != b->collation) {
		return false;
	}
	return true;
}

unique_ptr<ParsedExpression> CollateExpression::Copy() const {
	auto copy = make_unique<CollateExpression>(collation, child->Copy());
	copy->CopyProperties(*this);
	return std::move(copy);
}

void CollateExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*child);
	writer.WriteString(collation);
}

unique_ptr<ParsedExpression> CollateExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto child = reader.ReadRequiredSerializable<ParsedExpression>();
	auto collation = reader.ReadRequired<string>();
	return make_unique_base<ParsedExpression, CollateExpression>(collation, std::move(child));
}

} // namespace duckdb







namespace duckdb {

ColumnRefExpression::ColumnRefExpression(string column_name, string table_name)
    : ColumnRefExpression(table_name.empty() ? vector<string> {std::move(column_name)}
                                             : vector<string> {std::move(table_name), std::move(column_name)}) {
}

ColumnRefExpression::ColumnRefExpression(string column_name)
    : ColumnRefExpression(vector<string> {std::move(column_name)}) {
}

ColumnRefExpression::ColumnRefExpression(vector<string> column_names_p)
    : ParsedExpression(ExpressionType::COLUMN_REF, ExpressionClass::COLUMN_REF),
      column_names(std::move(column_names_p)) {
#ifdef DEBUG
	for (auto &col_name : column_names) {
		D_ASSERT(!col_name.empty());
	}
#endif
}

bool ColumnRefExpression::IsQualified() const {
	return column_names.size() > 1;
}

const string &ColumnRefExpression::GetColumnName() const {
	D_ASSERT(column_names.size() <= 4);
	return column_names.back();
}

const string &ColumnRefExpression::GetTableName() const {
	D_ASSERT(column_names.size() >= 2 && column_names.size() <= 4);
	if (column_names.size() == 4) {
		return column_names[2];
	}
	if (column_names.size() == 3) {
		return column_names[1];
	}
	return column_names[0];
}

string ColumnRefExpression::GetName() const {
	return !alias.empty() ? alias : column_names.back();
}

string ColumnRefExpression::ToString() const {
	string result;
	for (idx_t i = 0; i < column_names.size(); i++) {
		if (i > 0) {
			result += ".";
		}
		result += KeywordHelper::WriteOptionallyQuoted(column_names[i]);
	}
	return result;
}

bool ColumnRefExpression::Equal(const ColumnRefExpression *a, const ColumnRefExpression *b) {
	if (a->column_names.size() != b->column_names.size()) {
		return false;
	}
	for (idx_t i = 0; i < a->column_names.size(); i++) {
		auto lcase_a = StringUtil::Lower(a->column_names[i]);
		auto lcase_b = StringUtil::Lower(b->column_names[i]);
		if (lcase_a != lcase_b) {
			return false;
		}
	}
	return true;
}

hash_t ColumnRefExpression::Hash() const {
	hash_t result = ParsedExpression::Hash();
	for (auto &column_name : column_names) {
		auto lcase = StringUtil::Lower(column_name);
		result = CombineHash(result, duckdb::Hash<const char *>(lcase.c_str()));
	}
	return result;
}

unique_ptr<ParsedExpression> ColumnRefExpression::Copy() const {
	auto copy = make_unique<ColumnRefExpression>(column_names);
	copy->CopyProperties(*this);
	return std::move(copy);
}

void ColumnRefExpression::Serialize(FieldWriter &writer) const {
	writer.WriteList<string>(column_names);
}

unique_ptr<ParsedExpression> ColumnRefExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto column_names = reader.ReadRequiredList<string>();
	auto expression = make_unique<ColumnRefExpression>(std::move(column_names));
	return std::move(expression);
}

} // namespace duckdb






namespace duckdb {

ComparisonExpression::ComparisonExpression(ExpressionType type, unique_ptr<ParsedExpression> left,
                                           unique_ptr<ParsedExpression> right)
    : ParsedExpression(type, ExpressionClass::COMPARISON), left(std::move(left)), right(std::move(right)) {
}

string ComparisonExpression::ToString() const {
	return ToString<ComparisonExpression, ParsedExpression>(*this);
}

bool ComparisonExpression::Equal(const ComparisonExpression *a, const ComparisonExpression *b) {
	if (!a->left->Equals(b->left.get())) {
		return false;
	}
	if (!a->right->Equals(b->right.get())) {
		return false;
	}
	return true;
}

unique_ptr<ParsedExpression> ComparisonExpression::Copy() const {
	auto copy = make_unique<ComparisonExpression>(type, left->Copy(), right->Copy());
	copy->CopyProperties(*this);
	return std::move(copy);
}

void ComparisonExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*left);
	writer.WriteSerializable(*right);
}

unique_ptr<ParsedExpression> ComparisonExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto left_child = reader.ReadRequiredSerializable<ParsedExpression>();
	auto right_child = reader.ReadRequiredSerializable<ParsedExpression>();
	return make_unique<ComparisonExpression>(type, std::move(left_child), std::move(right_child));
}

} // namespace duckdb





namespace duckdb {

ConjunctionExpression::ConjunctionExpression(ExpressionType type)
    : ParsedExpression(type, ExpressionClass::CONJUNCTION) {
}

ConjunctionExpression::ConjunctionExpression(ExpressionType type, vector<unique_ptr<ParsedExpression>> children)
    : ParsedExpression(type, ExpressionClass::CONJUNCTION) {
	for (auto &child : children) {
		AddExpression(std::move(child));
	}
}

ConjunctionExpression::ConjunctionExpression(ExpressionType type, unique_ptr<ParsedExpression> left,
                                             unique_ptr<ParsedExpression> right)
    : ParsedExpression(type, ExpressionClass::CONJUNCTION) {
	AddExpression(std::move(left));
	AddExpression(std::move(right));
}

void ConjunctionExpression::AddExpression(unique_ptr<ParsedExpression> expr) {
	if (expr->type == type) {
		// expr is a conjunction of the same type: merge the expression lists together
		auto &other = (ConjunctionExpression &)*expr;
		for (auto &child : other.children) {
			children.push_back(std::move(child));
		}
	} else {
		children.push_back(std::move(expr));
	}
}

string ConjunctionExpression::ToString() const {
	return ToString<ConjunctionExpression, ParsedExpression>(*this);
}

bool ConjunctionExpression::Equal(const ConjunctionExpression *a, const ConjunctionExpression *b) {
	return ExpressionUtil::SetEquals(a->children, b->children);
}

unique_ptr<ParsedExpression> ConjunctionExpression::Copy() const {
	vector<unique_ptr<ParsedExpression>> copy_children;
	for (auto &expr : children) {
		copy_children.push_back(expr->Copy());
	}
	auto copy = make_unique<ConjunctionExpression>(type, std::move(copy_children));
	copy->CopyProperties(*this);
	return std::move(copy);
}

void ConjunctionExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializableList(children);
}

unique_ptr<ParsedExpression> ConjunctionExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto result = make_unique<ConjunctionExpression>(type);
	result->children = reader.ReadRequiredSerializableList<ParsedExpression>();
	return std::move(result);
}

} // namespace duckdb







namespace duckdb {

ConstantExpression::ConstantExpression(Value val)
    : ParsedExpression(ExpressionType::VALUE_CONSTANT, ExpressionClass::CONSTANT), value(std::move(val)) {
}

string ConstantExpression::ToString() const {
	return value.ToSQLString();
}

bool ConstantExpression::Equal(const ConstantExpression *a, const ConstantExpression *b) {
	return a->value.type() == b->value.type() && !ValueOperations::DistinctFrom(a->value, b->value);
}

hash_t ConstantExpression::Hash() const {
	return ParsedExpression::Hash();
}

unique_ptr<ParsedExpression> ConstantExpression::Copy() const {
	auto copy = make_unique<ConstantExpression>(value);
	copy->CopyProperties(*this);
	return std::move(copy);
}

void ConstantExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(value);
}

unique_ptr<ParsedExpression> ConstantExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	Value value = reader.ReadRequiredSerializable<Value, Value>();
	return make_unique<ConstantExpression>(std::move(value));
}

} // namespace duckdb




namespace duckdb {

DefaultExpression::DefaultExpression() : ParsedExpression(ExpressionType::VALUE_DEFAULT, ExpressionClass::DEFAULT) {
}

string DefaultExpression::ToString() const {
	return "DEFAULT";
}

unique_ptr<ParsedExpression> DefaultExpression::Copy() const {
	auto copy = make_unique<DefaultExpression>();
	copy->CopyProperties(*this);
	return std::move(copy);
}

void DefaultExpression::Serialize(FieldWriter &writer) const {
}

unique_ptr<ParsedExpression> DefaultExpression::Deserialize(ExpressionType type, FieldReader &source) {
	return make_unique<DefaultExpression>();
}

} // namespace duckdb


#include <utility>





namespace duckdb {

FunctionExpression::FunctionExpression(string catalog, string schema, const string &function_name,
                                       vector<unique_ptr<ParsedExpression>> children_p,
                                       unique_ptr<ParsedExpression> filter, unique_ptr<OrderModifier> order_bys_p,
                                       bool distinct, bool is_operator, bool export_state_p)
    : ParsedExpression(ExpressionType::FUNCTION, ExpressionClass::FUNCTION), catalog(std::move(catalog)),
      schema(std::move(schema)), function_name(StringUtil::Lower(function_name)), is_operator(is_operator),
      children(std::move(children_p)), distinct(distinct), filter(std::move(filter)), order_bys(std::move(order_bys_p)),
      export_state(export_state_p) {
	D_ASSERT(!function_name.empty());
	if (!order_bys) {
		order_bys = make_unique<OrderModifier>();
	}
}

FunctionExpression::FunctionExpression(const string &function_name, vector<unique_ptr<ParsedExpression>> children_p,
                                       unique_ptr<ParsedExpression> filter, unique_ptr<OrderModifier> order_bys,
                                       bool distinct, bool is_operator, bool export_state_p)
    : FunctionExpression(INVALID_CATALOG, INVALID_SCHEMA, function_name, std::move(children_p), std::move(filter),
                         std::move(order_bys), distinct, is_operator, export_state_p) {
}

string FunctionExpression::ToString() const {
	return ToString<FunctionExpression, ParsedExpression>(*this, schema, function_name, is_operator, distinct,
	                                                      filter.get(), order_bys.get(), export_state, true);
}

bool FunctionExpression::Equal(const FunctionExpression *a, const FunctionExpression *b) {
	if (a->catalog != b->catalog || a->schema != b->schema || a->function_name != b->function_name ||
	    b->distinct != a->distinct) {
		return false;
	}
	if (b->children.size() != a->children.size()) {
		return false;
	}
	for (idx_t i = 0; i < a->children.size(); i++) {
		if (!a->children[i]->Equals(b->children[i].get())) {
			return false;
		}
	}
	if (!BaseExpression::Equals(a->filter.get(), b->filter.get())) {
		return false;
	}
	if (!a->order_bys->Equals(b->order_bys.get())) {
		return false;
	}
	if (a->export_state != b->export_state) {
		return false;
	}
	return true;
}

hash_t FunctionExpression::Hash() const {
	hash_t result = ParsedExpression::Hash();
	result = CombineHash(result, duckdb::Hash<const char *>(schema.c_str()));
	result = CombineHash(result, duckdb::Hash<const char *>(function_name.c_str()));
	result = CombineHash(result, duckdb::Hash<bool>(distinct));
	result = CombineHash(result, duckdb::Hash<bool>(export_state));
	return result;
}

unique_ptr<ParsedExpression> FunctionExpression::Copy() const {
	vector<unique_ptr<ParsedExpression>> copy_children;
	unique_ptr<ParsedExpression> filter_copy;
	for (auto &child : children) {
		copy_children.push_back(child->Copy());
	}
	if (filter) {
		filter_copy = filter->Copy();
	}
	unique_ptr<OrderModifier> order_copy;
	if (order_bys) {
		order_copy.reset(static_cast<OrderModifier *>(order_bys->Copy().release()));
	}

	auto copy = make_unique<FunctionExpression>(catalog, schema, function_name, std::move(copy_children),
	                                            std::move(filter_copy), std::move(order_copy), distinct, is_operator,
	                                            export_state);
	copy->CopyProperties(*this);
	return std::move(copy);
}

void FunctionExpression::Serialize(FieldWriter &writer) const {
	writer.WriteString(function_name);
	writer.WriteString(schema);
	writer.WriteSerializableList(children);
	writer.WriteOptional(filter);
	writer.WriteSerializable((ResultModifier &)*order_bys);
	writer.WriteField<bool>(distinct);
	writer.WriteField<bool>(is_operator);
	writer.WriteField<bool>(export_state);
	writer.WriteString(catalog);
}

unique_ptr<ParsedExpression> FunctionExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto function_name = reader.ReadRequired<string>();
	auto schema = reader.ReadRequired<string>();
	auto children = reader.ReadRequiredSerializableList<ParsedExpression>();
	auto filter = reader.ReadOptional<ParsedExpression>(nullptr);
	auto order_bys = unique_ptr_cast<ResultModifier, OrderModifier>(reader.ReadRequiredSerializable<ResultModifier>());
	auto distinct = reader.ReadRequired<bool>();
	auto is_operator = reader.ReadRequired<bool>();
	auto export_state = reader.ReadField<bool>(false);
	auto catalog = reader.ReadField<string>(INVALID_CATALOG);

	unique_ptr<FunctionExpression> function;
	function = make_unique<FunctionExpression>(catalog, schema, function_name, std::move(children), std::move(filter),
	                                           std::move(order_bys), distinct, is_operator, export_state);
	return std::move(function);
}

void FunctionExpression::Verify() const {
	D_ASSERT(!function_name.empty());
}

} // namespace duckdb





namespace duckdb {

LambdaExpression::LambdaExpression(unique_ptr<ParsedExpression> lhs, unique_ptr<ParsedExpression> expr)
    : ParsedExpression(ExpressionType::LAMBDA, ExpressionClass::LAMBDA), lhs(std::move(lhs)), expr(std::move(expr)) {
}

string LambdaExpression::ToString() const {
	return lhs->ToString() + " -> " + expr->ToString();
}

bool LambdaExpression::Equal(const LambdaExpression *a, const LambdaExpression *b) {
	return a->lhs->Equals(b->lhs.get()) && a->expr->Equals(b->expr.get());
}

hash_t LambdaExpression::Hash() const {
	hash_t result = lhs->Hash();
	ParsedExpression::Hash();
	result = CombineHash(result, expr->Hash());
	return result;
}

unique_ptr<ParsedExpression> LambdaExpression::Copy() const {
	auto copy = make_unique<LambdaExpression>(lhs->Copy(), expr->Copy());
	copy->CopyProperties(*this);
	return std::move(copy);
}

void LambdaExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*lhs);
	writer.WriteSerializable(*expr);
}

unique_ptr<ParsedExpression> LambdaExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto lhs = reader.ReadRequiredSerializable<ParsedExpression>();
	auto expr = reader.ReadRequiredSerializable<ParsedExpression>();
	return make_unique<LambdaExpression>(std::move(lhs), std::move(expr));
}

} // namespace duckdb





namespace duckdb {

OperatorExpression::OperatorExpression(ExpressionType type, unique_ptr<ParsedExpression> left,
                                       unique_ptr<ParsedExpression> right)
    : ParsedExpression(type, ExpressionClass::OPERATOR) {
	if (left) {
		children.push_back(std::move(left));
	}
	if (right) {
		children.push_back(std::move(right));
	}
}

OperatorExpression::OperatorExpression(ExpressionType type, vector<unique_ptr<ParsedExpression>> children)
    : ParsedExpression(type, ExpressionClass::OPERATOR), children(std::move(children)) {
}

string OperatorExpression::ToString() const {
	return ToString<OperatorExpression, ParsedExpression>(*this);
}

bool OperatorExpression::Equal(const OperatorExpression *a, const OperatorExpression *b) {
	if (a->children.size() != b->children.size()) {
		return false;
	}
	for (idx_t i = 0; i < a->children.size(); i++) {
		if (!a->children[i]->Equals(b->children[i].get())) {
			return false;
		}
	}
	return true;
}

unique_ptr<ParsedExpression> OperatorExpression::Copy() const {
	auto copy = make_unique<OperatorExpression>(type);
	copy->CopyProperties(*this);
	for (auto &it : children) {
		copy->children.push_back(it->Copy());
	}
	return std::move(copy);
}

void OperatorExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializableList(children);
}

unique_ptr<ParsedExpression> OperatorExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto expression = make_unique<OperatorExpression>(type);
	expression->children = reader.ReadRequiredSerializableList<ParsedExpression>();
	return std::move(expression);
}

} // namespace duckdb







namespace duckdb {

ParameterExpression::ParameterExpression()
    : ParsedExpression(ExpressionType::VALUE_PARAMETER, ExpressionClass::PARAMETER), parameter_nr(0) {
}

string ParameterExpression::ToString() const {
	return "$" + to_string(parameter_nr);
}

unique_ptr<ParsedExpression> ParameterExpression::Copy() const {
	auto copy = make_unique<ParameterExpression>();
	copy->parameter_nr = parameter_nr;
	copy->CopyProperties(*this);
	return std::move(copy);
}

bool ParameterExpression::Equal(const ParameterExpression *a, const ParameterExpression *b) {
	return a->parameter_nr == b->parameter_nr;
}

hash_t ParameterExpression::Hash() const {
	hash_t result = ParsedExpression::Hash();
	return CombineHash(duckdb::Hash(parameter_nr), result);
}

void ParameterExpression::Serialize(FieldWriter &writer) const {
	writer.WriteField<idx_t>(parameter_nr);
}

unique_ptr<ParsedExpression> ParameterExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto expression = make_unique<ParameterExpression>();
	expression->parameter_nr = reader.ReadRequired<idx_t>();
	return std::move(expression);
}

} // namespace duckdb







namespace duckdb {

PositionalReferenceExpression::PositionalReferenceExpression(idx_t index)
    : ParsedExpression(ExpressionType::POSITIONAL_REFERENCE, ExpressionClass::POSITIONAL_REFERENCE), index(index) {
}

string PositionalReferenceExpression::ToString() const {
	return "#" + to_string(index);
}

bool PositionalReferenceExpression::Equal(const PositionalReferenceExpression *a,
                                          const PositionalReferenceExpression *b) {
	return a->index == b->index;
}

unique_ptr<ParsedExpression> PositionalReferenceExpression::Copy() const {
	auto copy = make_unique<PositionalReferenceExpression>(index);
	copy->CopyProperties(*this);
	return std::move(copy);
}

hash_t PositionalReferenceExpression::Hash() const {
	hash_t result = ParsedExpression::Hash();
	return CombineHash(duckdb::Hash(index), result);
}

void PositionalReferenceExpression::Serialize(FieldWriter &writer) const {
	writer.WriteField<idx_t>(index);
}

unique_ptr<ParsedExpression> PositionalReferenceExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto expression = make_unique<PositionalReferenceExpression>(reader.ReadRequired<idx_t>());
	return std::move(expression);
}

} // namespace duckdb





namespace duckdb {

StarExpression::StarExpression(string relation_name_p)
    : ParsedExpression(ExpressionType::STAR, ExpressionClass::STAR), relation_name(std::move(relation_name_p)) {
}

string StarExpression::ToString() const {
	if (!regex.empty()) {
		D_ASSERT(columns);
		return "COLUMNS('" + regex + "')";
	}
	string result;
	if (columns) {
		result += "COLUMNS(";
	}
	result += relation_name.empty() ? "*" : relation_name + ".*";
	if (!exclude_list.empty()) {
		result += " EXCLUDE (";
		bool first_entry = true;
		for (auto &entry : exclude_list) {
			if (!first_entry) {
				result += ", ";
			}
			result += entry;
			first_entry = false;
		}
		result += ")";
	}
	if (!replace_list.empty()) {
		result += " REPLACE (";
		bool first_entry = true;
		for (auto &entry : replace_list) {
			if (!first_entry) {
				result += ", ";
			}
			result += entry.second->ToString();
			result += " AS ";
			result += entry.first;
			first_entry = false;
		}
		result += ")";
	}
	if (columns) {
		result += ")";
	}
	return result;
}

bool StarExpression::Equal(const StarExpression *a, const StarExpression *b) {
	if (a->relation_name != b->relation_name || a->exclude_list != b->exclude_list) {
		return false;
	}
	if (a->columns != b->columns) {
		return false;
	}
	if (a->replace_list.size() != b->replace_list.size()) {
		return false;
	}
	for (auto &entry : a->replace_list) {
		auto other_entry = b->replace_list.find(entry.first);
		if (other_entry == b->replace_list.end()) {
			return false;
		}
		if (!entry.second->Equals(other_entry->second.get())) {
			return false;
		}
	}
	if (a->regex != b->regex) {
		return false;
	}
	return true;
}

void StarExpression::Serialize(FieldWriter &writer) const {
	auto &serializer = writer.GetSerializer();

	writer.WriteString(relation_name);

	// in order to write the exclude_list/replace_list as single fields we directly use the field writers' internal
	// serializer
	writer.WriteField<uint32_t>(exclude_list.size());
	for (auto &exclusion : exclude_list) {
		serializer.WriteString(exclusion);
	}
	writer.WriteField<uint32_t>(replace_list.size());
	for (auto &entry : replace_list) {
		serializer.WriteString(entry.first);
		entry.second->Serialize(serializer);
	}
	writer.WriteField<bool>(columns);
	writer.WriteString(regex);
}

unique_ptr<ParsedExpression> StarExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto &source = reader.GetSource();

	auto result = make_unique<StarExpression>();
	result->relation_name = reader.ReadRequired<string>();
	auto exclusion_count = reader.ReadRequired<uint32_t>();
	for (idx_t i = 0; i < exclusion_count; i++) {
		result->exclude_list.insert(source.Read<string>());
	}
	auto replace_count = reader.ReadRequired<uint32_t>();
	for (idx_t i = 0; i < replace_count; i++) {
		auto name = source.Read<string>();
		auto expr = ParsedExpression::Deserialize(source);
		result->replace_list.insert(make_pair(name, std::move(expr)));
	}
	result->columns = reader.ReadField<bool>(false);
	result->regex = reader.ReadField<string>(string());
	return std::move(result);
}

unique_ptr<ParsedExpression> StarExpression::Copy() const {
	auto copy = make_unique<StarExpression>(relation_name);
	copy->exclude_list = exclude_list;
	for (auto &entry : replace_list) {
		copy->replace_list[entry.first] = entry.second->Copy();
	}
	copy->columns = columns;
	copy->regex = regex;
	copy->CopyProperties(*this);
	return std::move(copy);
}

} // namespace duckdb





namespace duckdb {

SubqueryExpression::SubqueryExpression()
    : ParsedExpression(ExpressionType::SUBQUERY, ExpressionClass::SUBQUERY), subquery_type(SubqueryType::INVALID),
      comparison_type(ExpressionType::INVALID) {
}

string SubqueryExpression::ToString() const {
	switch (subquery_type) {
	case SubqueryType::ANY:
		return "(" + child->ToString() + " " + ExpressionTypeToOperator(comparison_type) + " ANY(" +
		       subquery->ToString() + "))";
	case SubqueryType::EXISTS:
		return "EXISTS(" + subquery->ToString() + ")";
	case SubqueryType::NOT_EXISTS:
		return "NOT EXISTS(" + subquery->ToString() + ")";
	case SubqueryType::SCALAR:
		return "(" + subquery->ToString() + ")";
	default:
		throw InternalException("Unrecognized type for subquery");
	}
}

bool SubqueryExpression::Equal(const SubqueryExpression *a, const SubqueryExpression *b) {
	if (!a->subquery || !b->subquery) {
		return false;
	}
	if (!BaseExpression::Equals(a->child.get(), b->child.get())) {
		return false;
	}
	return a->comparison_type == b->comparison_type && a->subquery_type == b->subquery_type &&
	       a->subquery->Equals(b->subquery.get());
}

unique_ptr<ParsedExpression> SubqueryExpression::Copy() const {
	auto copy = make_unique<SubqueryExpression>();
	copy->CopyProperties(*this);
	copy->subquery = unique_ptr_cast<SQLStatement, SelectStatement>(subquery->Copy());
	copy->subquery_type = subquery_type;
	copy->child = child ? child->Copy() : nullptr;
	copy->comparison_type = comparison_type;
	return std::move(copy);
}

void SubqueryExpression::Serialize(FieldWriter &writer) const {
	auto &serializer = writer.GetSerializer();

	writer.WriteField<SubqueryType>(subquery_type);
	// FIXME: this shouldn't use a serializer (probably)?
	subquery->Serialize(serializer);
	writer.WriteOptional(child);
	writer.WriteField<ExpressionType>(comparison_type);
}

unique_ptr<ParsedExpression> SubqueryExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	// FIXME: this shouldn't use a source
	auto &source = reader.GetSource();

	auto subquery_type = reader.ReadRequired<SubqueryType>();
	auto subquery = SelectStatement::Deserialize(source);

	auto expression = make_unique<SubqueryExpression>();
	expression->subquery_type = subquery_type;
	expression->subquery = std::move(subquery);
	expression->child = reader.ReadOptional<ParsedExpression>(nullptr);
	expression->comparison_type = reader.ReadRequired<ExpressionType>();
	return std::move(expression);
}

} // namespace duckdb






namespace duckdb {

WindowExpression::WindowExpression(ExpressionType type, string catalog_name, string schema, const string &function_name)
    : ParsedExpression(type, ExpressionClass::WINDOW), catalog(std::move(catalog_name)), schema(std::move(schema)),
      function_name(StringUtil::Lower(function_name)), ignore_nulls(false) {
	switch (type) {
	case ExpressionType::WINDOW_AGGREGATE:
	case ExpressionType::WINDOW_ROW_NUMBER:
	case ExpressionType::WINDOW_FIRST_VALUE:
	case ExpressionType::WINDOW_LAST_VALUE:
	case ExpressionType::WINDOW_NTH_VALUE:
	case ExpressionType::WINDOW_RANK:
	case ExpressionType::WINDOW_RANK_DENSE:
	case ExpressionType::WINDOW_PERCENT_RANK:
	case ExpressionType::WINDOW_CUME_DIST:
	case ExpressionType::WINDOW_LEAD:
	case ExpressionType::WINDOW_LAG:
	case ExpressionType::WINDOW_NTILE:
		break;
	default:
		throw NotImplementedException("Window aggregate type %s not supported", ExpressionTypeToString(type).c_str());
	}
}

string WindowExpression::ToString() const {
	return ToString<WindowExpression, ParsedExpression, OrderByNode>(*this, schema, function_name);
}

bool WindowExpression::Equal(const WindowExpression *a, const WindowExpression *b) {
	// check if the child expressions are equivalent
	if (b->children.size() != a->children.size()) {
		return false;
	}
	if (a->ignore_nulls != b->ignore_nulls) {
		return false;
	}
	for (idx_t i = 0; i < a->children.size(); i++) {
		if (!a->children[i]->Equals(b->children[i].get())) {
			return false;
		}
	}
	if (a->start != b->start || a->end != b->end) {
		return false;
	}
	// check if the framing expressions are equivalentbind_
	if (!BaseExpression::Equals(a->start_expr.get(), b->start_expr.get()) ||
	    !BaseExpression::Equals(a->end_expr.get(), b->end_expr.get()) ||
	    !BaseExpression::Equals(a->offset_expr.get(), b->offset_expr.get()) ||
	    !BaseExpression::Equals(a->default_expr.get(), b->default_expr.get())) {
		return false;
	}

	// check if the partitions are equivalent
	if (a->partitions.size() != b->partitions.size()) {
		return false;
	}
	for (idx_t i = 0; i < a->partitions.size(); i++) {
		if (!a->partitions[i]->Equals(b->partitions[i].get())) {
			return false;
		}
	}
	// check if the orderings are equivalent
	if (a->orders.size() != b->orders.size()) {
		return false;
	}
	for (idx_t i = 0; i < a->orders.size(); i++) {
		if (a->orders[i].type != b->orders[i].type) {
			return false;
		}
		if (!a->orders[i].expression->Equals(b->orders[i].expression.get())) {
			return false;
		}
	}
	// check if the filter clauses are equivalent
	if (!BaseExpression::Equals(a->filter_expr.get(), b->filter_expr.get())) {
		return false;
	}

	return true;
}

unique_ptr<ParsedExpression> WindowExpression::Copy() const {
	auto new_window = make_unique<WindowExpression>(type, catalog, schema, function_name);
	new_window->CopyProperties(*this);

	for (auto &child : children) {
		new_window->children.push_back(child->Copy());
	}

	for (auto &e : partitions) {
		new_window->partitions.push_back(e->Copy());
	}

	for (auto &o : orders) {
		new_window->orders.emplace_back(o.type, o.null_order, o.expression->Copy());
	}

	new_window->filter_expr = filter_expr ? filter_expr->Copy() : nullptr;

	new_window->start = start;
	new_window->end = end;
	new_window->start_expr = start_expr ? start_expr->Copy() : nullptr;
	new_window->end_expr = end_expr ? end_expr->Copy() : nullptr;
	new_window->offset_expr = offset_expr ? offset_expr->Copy() : nullptr;
	new_window->default_expr = default_expr ? default_expr->Copy() : nullptr;
	new_window->ignore_nulls = ignore_nulls;

	return std::move(new_window);
}

void WindowExpression::Serialize(FieldWriter &writer) const {
	auto &serializer = writer.GetSerializer();

	writer.WriteString(function_name);
	writer.WriteString(schema);
	writer.WriteSerializableList(children);
	writer.WriteSerializableList(partitions);
	// FIXME: should not use serializer here (probably)?
	D_ASSERT(orders.size() <= NumericLimits<uint32_t>::Maximum());
	writer.WriteField<uint32_t>((uint32_t)orders.size());
	for (auto &order : orders) {
		order.Serialize(serializer);
	}
	writer.WriteField<WindowBoundary>(start);
	writer.WriteField<WindowBoundary>(end);

	writer.WriteOptional(start_expr);
	writer.WriteOptional(end_expr);
	writer.WriteOptional(offset_expr);
	writer.WriteOptional(default_expr);
	writer.WriteField<bool>(ignore_nulls);
	writer.WriteOptional(filter_expr);
	writer.WriteString(catalog);
}

unique_ptr<ParsedExpression> WindowExpression::Deserialize(ExpressionType type, FieldReader &reader) {
	auto function_name = reader.ReadRequired<string>();
	auto schema = reader.ReadRequired<string>();
	auto expr = make_unique<WindowExpression>(type, INVALID_CATALOG, std::move(schema), function_name);
	expr->children = reader.ReadRequiredSerializableList<ParsedExpression>();
	expr->partitions = reader.ReadRequiredSerializableList<ParsedExpression>();

	auto order_count = reader.ReadRequired<uint32_t>();
	auto &source = reader.GetSource();
	for (idx_t i = 0; i < order_count; i++) {
		expr->orders.push_back(OrderByNode::Deserialize(source));
	}
	expr->start = reader.ReadRequired<WindowBoundary>();
	expr->end = reader.ReadRequired<WindowBoundary>();

	expr->start_expr = reader.ReadOptional<ParsedExpression>(nullptr);
	expr->end_expr = reader.ReadOptional<ParsedExpression>(nullptr);
	expr->offset_expr = reader.ReadOptional<ParsedExpression>(nullptr);
	expr->default_expr = reader.ReadOptional<ParsedExpression>(nullptr);
	expr->ignore_nulls = reader.ReadRequired<bool>();
	expr->filter_expr = reader.ReadOptional<ParsedExpression>(nullptr);
	expr->catalog = reader.ReadField<string>(INVALID_CATALOG);
	return std::move(expr);
}

} // namespace duckdb





namespace duckdb {

template <class T>
bool ExpressionUtil::ExpressionListEquals(const vector<unique_ptr<T>> &a, const vector<unique_ptr<T>> &b) {
	if (a.size() != b.size()) {
		return false;
	}
	for (idx_t i = 0; i < a.size(); i++) {
		if (!(*a[i] == *b[i])) {
			return false;
		}
	}
	return true;
}

template <class T>
bool ExpressionUtil::ExpressionSetEquals(const vector<unique_ptr<T>> &a, const vector<unique_ptr<T>> &b) {
	if (a.size() != b.size()) {
		return false;
	}
	// we create a map of expression -> count for the left side
	// we keep the count because the same expression can occur multiple times (e.g. "1 AND 1" is legal)
	// in this case we track the following value: map["Constant(1)"] = 2
	expression_map_t<idx_t> map;
	for (idx_t i = 0; i < a.size(); i++) {
		map[a[i].get()]++;
	}
	// now on the right side we reduce the counts again
	// if the conjunctions are identical, all the counts will be 0 after the
	for (auto &expr : b) {
		auto entry = map.find(expr.get());
		// first we check if we can find the expression in the map at all
		if (entry == map.end()) {
			return false;
		}
		// if we found it we check the count; if the count is already 0 we return false
		// this happens if e.g. the left side contains "1 AND X", and the right side contains "1 AND 1"
		// "1" is contained in the map, however, the right side contains the expression twice
		// hence we know the children are not identical in this case because the LHS and RHS have a different count for
		// the Constant(1) expression
		if (entry->second == 0) {
			return false;
		}
		entry->second--;
	}
	return true;
}

bool ExpressionUtil::ListEquals(const vector<unique_ptr<ParsedExpression>> &a,
                                const vector<unique_ptr<ParsedExpression>> &b) {
	return ExpressionListEquals<ParsedExpression>(a, b);
}

bool ExpressionUtil::ListEquals(const vector<unique_ptr<Expression>> &a, const vector<unique_ptr<Expression>> &b) {
	return ExpressionListEquals<Expression>(a, b);
}

bool ExpressionUtil::SetEquals(const vector<unique_ptr<ParsedExpression>> &a,
                               const vector<unique_ptr<ParsedExpression>> &b) {
	return ExpressionSetEquals<ParsedExpression>(a, b);
}

bool ExpressionUtil::SetEquals(const vector<unique_ptr<Expression>> &a, const vector<unique_ptr<Expression>> &b) {
	return ExpressionSetEquals<Expression>(a, b);
}

} // namespace duckdb




namespace duckdb {

bool KeywordHelper::IsKeyword(const string &text) {
	return Parser::IsKeyword(text);
}

bool KeywordHelper::RequiresQuotes(const string &text, bool allow_caps) {
	for (size_t i = 0; i < text.size(); i++) {
		if (i > 0 && (text[i] >= '0' && text[i] <= '9')) {
			continue;
		}
		if (text[i] >= 'a' && text[i] <= 'z') {
			continue;
		}
		if (allow_caps) {
			if (text[i] >= 'A' && text[i] <= 'Z') {
				continue;
			}
		}
		if (text[i] == '_') {
			continue;
		}
		return true;
	}
	return IsKeyword(text);
}

string KeywordHelper::WriteOptionallyQuoted(const string &text, char quote, bool allow_caps) {
	if (!RequiresQuotes(text, allow_caps)) {
		return text;
	}
	return string(1, quote) + StringUtil::Replace(text, string(1, quote), string(2, quote)) + string(1, quote);
}

} // namespace duckdb





namespace duckdb {

//===--------------------------------------------------------------------===//
// AlterFunctionInfo
//===--------------------------------------------------------------------===//
AlterFunctionInfo::AlterFunctionInfo(AlterFunctionType type, AlterEntryData data)
    : AlterInfo(AlterType::ALTER_FUNCTION, std::move(data.catalog), std::move(data.schema), std::move(data.name),
                data.if_exists),
      alter_function_type(type) {
}
AlterFunctionInfo::~AlterFunctionInfo() {
}

CatalogType AlterFunctionInfo::GetCatalogType() const {
	return CatalogType::SCALAR_FUNCTION_ENTRY;
}

void AlterFunctionInfo::Serialize(FieldWriter &writer) const {
	writer.WriteField<AlterFunctionType>(alter_function_type);
	writer.WriteString(catalog);
	writer.WriteString(schema);
	writer.WriteString(name);
	writer.WriteField(if_exists);
}

unique_ptr<AlterInfo> AlterFunctionInfo::Deserialize(FieldReader &reader) {
	//	auto type = reader.ReadRequired<AlterFunctionType>();
	//	auto schema = reader.ReadRequired<string>();
	//	auto table = reader.ReadRequired<string>();
	//	auto if_exists = reader.ReadRequired<bool>();

	throw NotImplementedException("AlterFunctionInfo cannot be deserialized");
}

//===--------------------------------------------------------------------===//
// AddFunctionOverloadInfo
//===--------------------------------------------------------------------===//
AddFunctionOverloadInfo::AddFunctionOverloadInfo(AlterEntryData data, ScalarFunctionSet new_overloads_p)
    : AlterFunctionInfo(AlterFunctionType::ADD_FUNCTION_OVERLOADS, std::move(data)),
      new_overloads(std::move(new_overloads_p)) {
	this->allow_internal = true;
}
AddFunctionOverloadInfo::~AddFunctionOverloadInfo() {
}

unique_ptr<AlterInfo> AddFunctionOverloadInfo::Copy() const {
	return make_unique_base<AlterInfo, AddFunctionOverloadInfo>(GetAlterEntryData(), new_overloads);
}

} // namespace duckdb






namespace duckdb {

AlterInfo::AlterInfo(AlterType type, string catalog_p, string schema_p, string name_p, bool if_exists)
    : type(type), if_exists(if_exists), catalog(std::move(catalog_p)), schema(std::move(schema_p)),
      name(std::move(name_p)), allow_internal(false) {
}

AlterInfo::~AlterInfo() {
}

void AlterInfo::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteField<AlterType>(type);
	Serialize(writer);
	writer.Finalize();
}

unique_ptr<AlterInfo> AlterInfo::Deserialize(Deserializer &source) {
	FieldReader reader(source);
	auto type = reader.ReadRequired<AlterType>();

	unique_ptr<AlterInfo> result;
	switch (type) {
	case AlterType::ALTER_TABLE:
		result = AlterTableInfo::Deserialize(reader);
		break;
	case AlterType::ALTER_VIEW:
		result = AlterViewInfo::Deserialize(reader);
		break;
	case AlterType::ALTER_FUNCTION:
		result = AlterFunctionInfo::Deserialize(reader);
		break;
	default:
		throw SerializationException("Unknown alter type for deserialization!");
	}
	reader.Finalize();

	return result;
}

AlterEntryData AlterInfo::GetAlterEntryData() const {
	AlterEntryData data;
	data.catalog = catalog;
	data.schema = schema;
	data.name = name;
	data.if_exists = if_exists;
	return data;
}

} // namespace duckdb





namespace duckdb {

//===--------------------------------------------------------------------===//
// ChangeOwnershipInfo
//===--------------------------------------------------------------------===//
ChangeOwnershipInfo::ChangeOwnershipInfo(CatalogType entry_catalog_type, string entry_catalog_p, string entry_schema_p,
                                         string entry_name_p, string owner_schema_p, string owner_name_p,
                                         bool if_exists)
    : AlterInfo(AlterType::CHANGE_OWNERSHIP, std::move(entry_catalog_p), std::move(entry_schema_p),
                std::move(entry_name_p), if_exists),
      entry_catalog_type(entry_catalog_type), owner_schema(std::move(owner_schema_p)),
      owner_name(std::move(owner_name_p)) {
}

CatalogType ChangeOwnershipInfo::GetCatalogType() const {
	return entry_catalog_type;
}

unique_ptr<AlterInfo> ChangeOwnershipInfo::Copy() const {
	return make_unique_base<AlterInfo, ChangeOwnershipInfo>(entry_catalog_type, catalog, schema, name, owner_schema,
	                                                        owner_name, if_exists);
}

void ChangeOwnershipInfo::Serialize(FieldWriter &writer) const {
	throw InternalException("ChangeOwnershipInfo cannot be serialized");
}

//===--------------------------------------------------------------------===//
// AlterTableInfo
//===--------------------------------------------------------------------===//
AlterTableInfo::AlterTableInfo(AlterTableType type, AlterEntryData data)
    : AlterInfo(AlterType::ALTER_TABLE, std::move(data.catalog), std::move(data.schema), std::move(data.name),
                data.if_exists),
      alter_table_type(type) {
}
AlterTableInfo::~AlterTableInfo() {
}

CatalogType AlterTableInfo::GetCatalogType() const {
	return CatalogType::TABLE_ENTRY;
}

void AlterTableInfo::Serialize(FieldWriter &writer) const {
	writer.WriteField<AlterTableType>(alter_table_type);
	writer.WriteString(catalog);
	writer.WriteString(schema);
	writer.WriteString(name);
	writer.WriteField(if_exists);
	SerializeAlterTable(writer);
}

unique_ptr<AlterInfo> AlterTableInfo::Deserialize(FieldReader &reader) {
	auto type = reader.ReadRequired<AlterTableType>();
	AlterEntryData data;
	data.catalog = reader.ReadRequired<string>();
	data.schema = reader.ReadRequired<string>();
	data.name = reader.ReadRequired<string>();
	data.if_exists = reader.ReadRequired<bool>();

	unique_ptr<AlterTableInfo> info;
	switch (type) {
	case AlterTableType::RENAME_COLUMN:
		return RenameColumnInfo::Deserialize(reader, std::move(data));
	case AlterTableType::RENAME_TABLE:
		return RenameTableInfo::Deserialize(reader, std::move(data));
	case AlterTableType::ADD_COLUMN:
		return AddColumnInfo::Deserialize(reader, std::move(data));
	case AlterTableType::REMOVE_COLUMN:
		return RemoveColumnInfo::Deserialize(reader, std::move(data));
	case AlterTableType::ALTER_COLUMN_TYPE:
		return ChangeColumnTypeInfo::Deserialize(reader, std::move(data));
	case AlterTableType::SET_DEFAULT:
		return SetDefaultInfo::Deserialize(reader, std::move(data));
	case AlterTableType::FOREIGN_KEY_CONSTRAINT:
		return AlterForeignKeyInfo::Deserialize(reader, std::move(data));
	case AlterTableType::SET_NOT_NULL:
		return SetNotNullInfo::Deserialize(reader, std::move(data));
	case AlterTableType::DROP_NOT_NULL:
		return DropNotNullInfo::Deserialize(reader, std::move(data));
	default:
		throw SerializationException("Unknown alter table type for deserialization!");
	}
}

//===--------------------------------------------------------------------===//
// RenameColumnInfo
//===--------------------------------------------------------------------===//
RenameColumnInfo::RenameColumnInfo(AlterEntryData data, string old_name_p, string new_name_p)
    : AlterTableInfo(AlterTableType::RENAME_COLUMN, std::move(data)), old_name(std::move(old_name_p)),
      new_name(std::move(new_name_p)) {
}
RenameColumnInfo::~RenameColumnInfo() {
}

unique_ptr<AlterInfo> RenameColumnInfo::Copy() const {
	return make_unique_base<AlterInfo, RenameColumnInfo>(GetAlterEntryData(), old_name, new_name);
}

void RenameColumnInfo::SerializeAlterTable(FieldWriter &writer) const {
	writer.WriteString(old_name);
	writer.WriteString(new_name);
}

unique_ptr<AlterInfo> RenameColumnInfo::Deserialize(FieldReader &reader, AlterEntryData data) {
	auto old_name = reader.ReadRequired<string>();
	auto new_name = reader.ReadRequired<string>();
	return make_unique<RenameColumnInfo>(std::move(data), old_name, new_name);
}

//===--------------------------------------------------------------------===//
// RenameTableInfo
//===--------------------------------------------------------------------===//
RenameTableInfo::RenameTableInfo(AlterEntryData data, string new_name_p)
    : AlterTableInfo(AlterTableType::RENAME_TABLE, std::move(data)), new_table_name(std::move(new_name_p)) {
}
RenameTableInfo::~RenameTableInfo() {
}

unique_ptr<AlterInfo> RenameTableInfo::Copy() const {
	return make_unique_base<AlterInfo, RenameTableInfo>(GetAlterEntryData(), new_table_name);
}

void RenameTableInfo::SerializeAlterTable(FieldWriter &writer) const {
	writer.WriteString(new_table_name);
}

unique_ptr<AlterInfo> RenameTableInfo::Deserialize(FieldReader &reader, AlterEntryData data) {
	auto new_name = reader.ReadRequired<string>();
	return make_unique<RenameTableInfo>(std::move(data), new_name);
}

//===--------------------------------------------------------------------===//
// AddColumnInfo
//===--------------------------------------------------------------------===//
AddColumnInfo::AddColumnInfo(AlterEntryData data, ColumnDefinition new_column, bool if_column_not_exists)
    : AlterTableInfo(AlterTableType::ADD_COLUMN, std::move(data)), new_column(std::move(new_column)),
      if_column_not_exists(if_column_not_exists) {
}

AddColumnInfo::~AddColumnInfo() {
}

unique_ptr<AlterInfo> AddColumnInfo::Copy() const {
	return make_unique_base<AlterInfo, AddColumnInfo>(GetAlterEntryData(), new_column.Copy(), if_column_not_exists);
}

void AddColumnInfo::SerializeAlterTable(FieldWriter &writer) const {
	writer.WriteSerializable(new_column);
	writer.WriteField<bool>(if_column_not_exists);
}

unique_ptr<AlterInfo> AddColumnInfo::Deserialize(FieldReader &reader, AlterEntryData data) {
	auto new_column = reader.ReadRequiredSerializable<ColumnDefinition, ColumnDefinition>();
	auto if_column_not_exists = reader.ReadRequired<bool>();
	return make_unique<AddColumnInfo>(std::move(data), std::move(new_column), if_column_not_exists);
}

//===--------------------------------------------------------------------===//
// RemoveColumnInfo
//===--------------------------------------------------------------------===//
RemoveColumnInfo::RemoveColumnInfo(AlterEntryData data, string removed_column, bool if_column_exists, bool cascade)
    : AlterTableInfo(AlterTableType::REMOVE_COLUMN, std::move(data)), removed_column(std::move(removed_column)),
      if_column_exists(if_column_exists), cascade(cascade) {
}
RemoveColumnInfo::~RemoveColumnInfo() {
}

unique_ptr<AlterInfo> RemoveColumnInfo::Copy() const {
	return make_unique_base<AlterInfo, RemoveColumnInfo>(GetAlterEntryData(), removed_column, if_column_exists,
	                                                     cascade);
}

void RemoveColumnInfo::SerializeAlterTable(FieldWriter &writer) const {
	writer.WriteString(removed_column);
	writer.WriteField<bool>(if_column_exists);
	writer.WriteField<bool>(cascade);
}

unique_ptr<AlterInfo> RemoveColumnInfo::Deserialize(FieldReader &reader, AlterEntryData data) {
	auto new_name = reader.ReadRequired<string>();
	auto if_column_exists = reader.ReadRequired<bool>();
	auto cascade = reader.ReadRequired<bool>();
	return make_unique<RemoveColumnInfo>(std::move(data), std::move(new_name), if_column_exists, cascade);
}

//===--------------------------------------------------------------------===//
// ChangeColumnTypeInfo
//===--------------------------------------------------------------------===//
ChangeColumnTypeInfo::ChangeColumnTypeInfo(AlterEntryData data, string column_name, LogicalType target_type,
                                           unique_ptr<ParsedExpression> expression)
    : AlterTableInfo(AlterTableType::ALTER_COLUMN_TYPE, std::move(data)), column_name(std::move(column_name)),
      target_type(std::move(target_type)), expression(std::move(expression)) {
}
ChangeColumnTypeInfo::~ChangeColumnTypeInfo() {
}

unique_ptr<AlterInfo> ChangeColumnTypeInfo::Copy() const {
	return make_unique_base<AlterInfo, ChangeColumnTypeInfo>(GetAlterEntryData(), column_name, target_type,
	                                                         expression->Copy());
}

void ChangeColumnTypeInfo::SerializeAlterTable(FieldWriter &writer) const {
	writer.WriteString(column_name);
	writer.WriteSerializable(target_type);
	writer.WriteOptional(expression);
}

unique_ptr<AlterInfo> ChangeColumnTypeInfo::Deserialize(FieldReader &reader, AlterEntryData data) {
	auto column_name = reader.ReadRequired<string>();
	auto target_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto expression = reader.ReadOptional<ParsedExpression>(nullptr);
	return make_unique<ChangeColumnTypeInfo>(std::move(data), std::move(column_name), std::move(target_type),
	                                         std::move(expression));
}

//===--------------------------------------------------------------------===//
// SetDefaultInfo
//===--------------------------------------------------------------------===//
SetDefaultInfo::SetDefaultInfo(AlterEntryData data, string column_name_p, unique_ptr<ParsedExpression> new_default)
    : AlterTableInfo(AlterTableType::SET_DEFAULT, std::move(data)), column_name(std::move(column_name_p)),
      expression(std::move(new_default)) {
}
SetDefaultInfo::~SetDefaultInfo() {
}

unique_ptr<AlterInfo> SetDefaultInfo::Copy() const {
	return make_unique_base<AlterInfo, SetDefaultInfo>(GetAlterEntryData(), column_name,
	                                                   expression ? expression->Copy() : nullptr);
}

void SetDefaultInfo::SerializeAlterTable(FieldWriter &writer) const {
	writer.WriteString(column_name);
	writer.WriteOptional(expression);
}

unique_ptr<AlterInfo> SetDefaultInfo::Deserialize(FieldReader &reader, AlterEntryData data) {
	auto column_name = reader.ReadRequired<string>();
	auto new_default = reader.ReadOptional<ParsedExpression>(nullptr);
	return make_unique<SetDefaultInfo>(std::move(data), std::move(column_name), std::move(new_default));
}

//===--------------------------------------------------------------------===//
// SetNotNullInfo
//===--------------------------------------------------------------------===//
SetNotNullInfo::SetNotNullInfo(AlterEntryData data, string column_name_p)
    : AlterTableInfo(AlterTableType::SET_NOT_NULL, std::move(data)), column_name(std::move(column_name_p)) {
}
SetNotNullInfo::~SetNotNullInfo() {
}

unique_ptr<AlterInfo> SetNotNullInfo::Copy() const {
	return make_unique_base<AlterInfo, SetNotNullInfo>(GetAlterEntryData(), column_name);
}

void SetNotNullInfo::SerializeAlterTable(FieldWriter &writer) const {
	writer.WriteString(column_name);
}

unique_ptr<AlterInfo> SetNotNullInfo::Deserialize(FieldReader &reader, AlterEntryData data) {
	auto column_name = reader.ReadRequired<string>();
	return make_unique<SetNotNullInfo>(std::move(data), std::move(column_name));
}

//===--------------------------------------------------------------------===//
// DropNotNullInfo
//===--------------------------------------------------------------------===//
DropNotNullInfo::DropNotNullInfo(AlterEntryData data, string column_name_p)
    : AlterTableInfo(AlterTableType::DROP_NOT_NULL, std::move(data)), column_name(std::move(column_name_p)) {
}
DropNotNullInfo::~DropNotNullInfo() {
}

unique_ptr<AlterInfo> DropNotNullInfo::Copy() const {
	return make_unique_base<AlterInfo, DropNotNullInfo>(GetAlterEntryData(), column_name);
}

void DropNotNullInfo::SerializeAlterTable(FieldWriter &writer) const {
	writer.WriteString(column_name);
}

unique_ptr<AlterInfo> DropNotNullInfo::Deserialize(FieldReader &reader, AlterEntryData data) {
	auto column_name = reader.ReadRequired<string>();
	return make_unique<DropNotNullInfo>(std::move(data), std::move(column_name));
}

//===--------------------------------------------------------------------===//
// AlterForeignKeyInfo
//===--------------------------------------------------------------------===//
AlterForeignKeyInfo::AlterForeignKeyInfo(AlterEntryData data, string fk_table, vector<string> pk_columns,
                                         vector<string> fk_columns, vector<PhysicalIndex> pk_keys,
                                         vector<PhysicalIndex> fk_keys, AlterForeignKeyType type_p)
    : AlterTableInfo(AlterTableType::FOREIGN_KEY_CONSTRAINT, std::move(data)), fk_table(std::move(fk_table)),
      pk_columns(std::move(pk_columns)), fk_columns(std::move(fk_columns)), pk_keys(std::move(pk_keys)),
      fk_keys(std::move(fk_keys)), type(type_p) {
}
AlterForeignKeyInfo::~AlterForeignKeyInfo() {
}

unique_ptr<AlterInfo> AlterForeignKeyInfo::Copy() const {
	return make_unique_base<AlterInfo, AlterForeignKeyInfo>(GetAlterEntryData(), fk_table, pk_columns, fk_columns,
	                                                        pk_keys, fk_keys, type);
}

void AlterForeignKeyInfo::SerializeAlterTable(FieldWriter &writer) const {
	writer.WriteString(fk_table);
	writer.WriteList<string>(pk_columns);
	writer.WriteList<string>(fk_columns);
	writer.WriteIndexList<PhysicalIndex>(pk_keys);
	writer.WriteIndexList<PhysicalIndex>(fk_keys);
	writer.WriteField<AlterForeignKeyType>(type);
}

unique_ptr<AlterInfo> AlterForeignKeyInfo::Deserialize(FieldReader &reader, AlterEntryData data) {
	auto fk_table = reader.ReadRequired<string>();
	auto pk_columns = reader.ReadRequiredList<string>();
	auto fk_columns = reader.ReadRequiredList<string>();
	auto pk_keys = reader.ReadRequiredIndexList<PhysicalIndex>();
	auto fk_keys = reader.ReadRequiredIndexList<PhysicalIndex>();
	auto type = reader.ReadRequired<AlterForeignKeyType>();
	return make_unique<AlterForeignKeyInfo>(std::move(data), std::move(fk_table), std::move(pk_columns),
	                                        std::move(fk_columns), std::move(pk_keys), std::move(fk_keys), type);
}

//===--------------------------------------------------------------------===//
// Alter View
//===--------------------------------------------------------------------===//
AlterViewInfo::AlterViewInfo(AlterViewType type, AlterEntryData data)
    : AlterInfo(AlterType::ALTER_VIEW, std::move(data.catalog), std::move(data.schema), std::move(data.name),
                data.if_exists),
      alter_view_type(type) {
}
AlterViewInfo::~AlterViewInfo() {
}

CatalogType AlterViewInfo::GetCatalogType() const {
	return CatalogType::VIEW_ENTRY;
}

void AlterViewInfo::Serialize(FieldWriter &writer) const {
	writer.WriteField<AlterViewType>(alter_view_type);
	writer.WriteString(catalog);
	writer.WriteString(schema);
	writer.WriteString(name);
	writer.WriteField<bool>(if_exists);
	SerializeAlterView(writer);
}

unique_ptr<AlterInfo> AlterViewInfo::Deserialize(FieldReader &reader) {
	auto type = reader.ReadRequired<AlterViewType>();
	AlterEntryData data;
	data.catalog = reader.ReadRequired<string>();
	data.schema = reader.ReadRequired<string>();
	data.name = reader.ReadRequired<string>();
	data.if_exists = reader.ReadRequired<bool>();
	unique_ptr<AlterViewInfo> info;
	switch (type) {
	case AlterViewType::RENAME_VIEW:
		return RenameViewInfo::Deserialize(reader, std::move(data));
	default:
		throw SerializationException("Unknown alter view type for deserialization!");
	}
}

//===--------------------------------------------------------------------===//
// RenameViewInfo
//===--------------------------------------------------------------------===//
RenameViewInfo::RenameViewInfo(AlterEntryData data, string new_name_p)
    : AlterViewInfo(AlterViewType::RENAME_VIEW, std::move(data)), new_view_name(std::move(new_name_p)) {
}
RenameViewInfo::~RenameViewInfo() {
}

unique_ptr<AlterInfo> RenameViewInfo::Copy() const {
	return make_unique_base<AlterInfo, RenameViewInfo>(GetAlterEntryData(), new_view_name);
}

void RenameViewInfo::SerializeAlterView(FieldWriter &writer) const {
	writer.WriteString(new_view_name);
}

unique_ptr<AlterInfo> RenameViewInfo::Deserialize(FieldReader &reader, AlterEntryData data) {
	auto new_name = reader.ReadRequired<string>();
	return make_unique<RenameViewInfo>(std::move(data), new_name);
}
} // namespace duckdb


namespace duckdb {

CreateAggregateFunctionInfo::CreateAggregateFunctionInfo(AggregateFunction function)
    : CreateFunctionInfo(CatalogType::AGGREGATE_FUNCTION_ENTRY), functions(function.name) {
	name = function.name;
	functions.AddFunction(std::move(function));
	internal = true;
}

CreateAggregateFunctionInfo::CreateAggregateFunctionInfo(AggregateFunctionSet set)
    : CreateFunctionInfo(CatalogType::AGGREGATE_FUNCTION_ENTRY), functions(std::move(set)) {
	name = functions.name;
	for (auto &func : functions.functions) {
		func.name = functions.name;
	}
	internal = true;
}

unique_ptr<CreateInfo> CreateAggregateFunctionInfo::Copy() const {
	auto result = make_unique<CreateAggregateFunctionInfo>(functions);
	CopyProperties(*result);
	return std::move(result);
}

} // namespace duckdb


namespace duckdb {

CreateCollationInfo::CreateCollationInfo(string name_p, ScalarFunction function_p, bool combinable_p,
                                         bool not_required_for_equality_p)
    : CreateInfo(CatalogType::COLLATION_ENTRY), function(std::move(function_p)), combinable(combinable_p),
      not_required_for_equality(not_required_for_equality_p) {
	this->name = std::move(name_p);
	internal = true;
}

void CreateCollationInfo::SerializeInternal(Serializer &) const {
	throw NotImplementedException("Cannot serialize '%s'", CatalogTypeToString(type));
}

unique_ptr<CreateInfo> CreateCollationInfo::Copy() const {
	auto result = make_unique<CreateCollationInfo>(name, function, combinable, not_required_for_equality);
	CopyProperties(*result);
	return std::move(result);
}

} // namespace duckdb


namespace duckdb {

CreateCopyFunctionInfo::CreateCopyFunctionInfo(CopyFunction function_p)
    : CreateInfo(CatalogType::COPY_FUNCTION_ENTRY), function(std::move(function_p)) {
	this->name = function.name;
	internal = true;
}

void CreateCopyFunctionInfo::SerializeInternal(Serializer &) const {
	throw NotImplementedException("Cannot serialize '%s'", CatalogTypeToString(type));
}

unique_ptr<CreateInfo> CreateCopyFunctionInfo::Copy() const {
	auto result = make_unique<CreateCopyFunctionInfo>(function);
	CopyProperties(*result);
	return std::move(result);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<CreateInfo> CreateIndexInfo::Copy() const {

	auto result = make_unique<CreateIndexInfo>();
	CopyProperties(*result);

	result->index_type = index_type;
	result->index_name = index_name;
	result->constraint_type = constraint_type;
	result->table = unique_ptr_cast<TableRef, BaseTableRef>(table->Copy());
	for (auto &expr : expressions) {
		result->expressions.push_back(expr->Copy());
	}

	result->scan_types = scan_types;
	result->names = names;
	result->column_ids = column_ids;
	return std::move(result);
}

void CreateIndexInfo::SerializeInternal(Serializer &serializer) const {

	FieldWriter writer(serializer);
	writer.WriteField(index_type);
	writer.WriteString(index_name);
	writer.WriteField(constraint_type);

	writer.WriteSerializableList<ParsedExpression>(expressions);
	writer.WriteSerializableList<ParsedExpression>(parsed_expressions);

	writer.WriteRegularSerializableList(scan_types);
	writer.WriteList<string>(names);
	writer.WriteList<column_t>(column_ids);

	writer.Finalize();
}

unique_ptr<CreateIndexInfo> CreateIndexInfo::Deserialize(Deserializer &deserializer) {

	auto result = make_unique<CreateIndexInfo>();
	result->DeserializeBase(deserializer);

	FieldReader reader(deserializer);
	result->index_type = reader.ReadRequired<IndexType>();
	result->index_name = reader.ReadRequired<string>();
	result->constraint_type = reader.ReadRequired<IndexConstraintType>();

	result->expressions = reader.ReadRequiredSerializableList<ParsedExpression>();
	result->parsed_expressions = reader.ReadRequiredSerializableList<ParsedExpression>();

	result->scan_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
	result->names = reader.ReadRequiredList<string>();
	result->column_ids = reader.ReadRequiredList<column_t>();

	reader.Finalize();
	return result;
}
} // namespace duckdb









namespace duckdb {
void CreateInfo::DeserializeBase(Deserializer &deserializer) {
	this->catalog = deserializer.Read<string>();
	this->schema = deserializer.Read<string>();
	this->on_conflict = deserializer.Read<OnCreateConflict>();
	this->temporary = deserializer.Read<bool>();
	this->internal = deserializer.Read<bool>();
	this->sql = deserializer.Read<string>();
}

void CreateInfo::Serialize(Serializer &serializer) const {
	serializer.Write(type);
	serializer.WriteString(catalog);
	serializer.WriteString(schema);
	serializer.Write(on_conflict);
	serializer.Write(temporary);
	serializer.Write(internal);
	serializer.WriteString(sql);
	SerializeInternal(serializer);
}

unique_ptr<CreateInfo> CreateInfo::Deserialize(Deserializer &deserializer) {
	auto type = deserializer.Read<CatalogType>();
	switch (type) {
	case CatalogType::INDEX_ENTRY:
		return CreateIndexInfo::Deserialize(deserializer);
	case CatalogType::TABLE_ENTRY:
		return CreateTableInfo::Deserialize(deserializer);
	case CatalogType::SCHEMA_ENTRY:
		return CreateSchemaInfo::Deserialize(deserializer);
	case CatalogType::VIEW_ENTRY:
		return CreateViewInfo::Deserialize(deserializer);
	case CatalogType::DATABASE_ENTRY:
		return CreateDatabaseInfo::Deserialize(deserializer);
	default:
		throw NotImplementedException("Cannot deserialize '%s'", CatalogTypeToString(type));
	}
}

unique_ptr<CreateInfo> CreateInfo::Deserialize(Deserializer &source, PlanDeserializationState &state) {
	return Deserialize(source);
}

void CreateInfo::CopyProperties(CreateInfo &other) const {
	other.type = type;
	other.catalog = catalog;
	other.schema = schema;
	other.on_conflict = on_conflict;
	other.temporary = temporary;
	other.internal = internal;
	other.sql = sql;
}

unique_ptr<AlterInfo> CreateInfo::GetAlterInfo() const {
	throw NotImplementedException("GetAlterInfo not implemented for this type");
}

} // namespace duckdb


namespace duckdb {

CreatePragmaFunctionInfo::CreatePragmaFunctionInfo(PragmaFunction function)
    : CreateFunctionInfo(CatalogType::PRAGMA_FUNCTION_ENTRY), functions(function.name) {
	name = function.name;
	functions.AddFunction(std::move(function));
	internal = true;
}
CreatePragmaFunctionInfo::CreatePragmaFunctionInfo(string name, PragmaFunctionSet functions_p)
    : CreateFunctionInfo(CatalogType::PRAGMA_FUNCTION_ENTRY), functions(std::move(functions_p)) {
	this->name = std::move(name);
	internal = true;
}

unique_ptr<CreateInfo> CreatePragmaFunctionInfo::Copy() const {
	auto result = make_unique<CreatePragmaFunctionInfo>(functions.name, functions);
	CopyProperties(*result);
	return std::move(result);
}

} // namespace duckdb



namespace duckdb {

CreateScalarFunctionInfo::CreateScalarFunctionInfo(ScalarFunction function)
    : CreateFunctionInfo(CatalogType::SCALAR_FUNCTION_ENTRY), functions(function.name) {
	name = function.name;
	functions.AddFunction(std::move(function));
	internal = true;
}
CreateScalarFunctionInfo::CreateScalarFunctionInfo(ScalarFunctionSet set)
    : CreateFunctionInfo(CatalogType::SCALAR_FUNCTION_ENTRY), functions(std::move(set)) {
	name = functions.name;
	for (auto &func : functions.functions) {
		func.name = functions.name;
	}
	internal = true;
}

unique_ptr<CreateInfo> CreateScalarFunctionInfo::Copy() const {
	ScalarFunctionSet set(name);
	set.functions = functions.functions;
	auto result = make_unique<CreateScalarFunctionInfo>(std::move(set));
	CopyProperties(*result);
	return std::move(result);
}

unique_ptr<AlterInfo> CreateScalarFunctionInfo::GetAlterInfo() const {
	return make_unique_base<AlterInfo, AddFunctionOverloadInfo>(AlterEntryData(catalog, schema, name, true), functions);
}

} // namespace duckdb


namespace duckdb {

CreateTableFunctionInfo::CreateTableFunctionInfo(TableFunction function)
    : CreateFunctionInfo(CatalogType::TABLE_FUNCTION_ENTRY), functions(function.name) {
	name = function.name;
	functions.AddFunction(std::move(function));
	internal = true;
}
CreateTableFunctionInfo::CreateTableFunctionInfo(TableFunctionSet set)
    : CreateFunctionInfo(CatalogType::TABLE_FUNCTION_ENTRY), functions(std::move(set)) {
	name = functions.name;
	for (auto &func : functions.functions) {
		func.name = functions.name;
	}
	internal = true;
}

unique_ptr<CreateInfo> CreateTableFunctionInfo::Copy() const {
	TableFunctionSet set(name);
	set.functions = functions.functions;
	auto result = make_unique<CreateTableFunctionInfo>(std::move(set));
	CopyProperties(*result);
	return std::move(result);
}

} // namespace duckdb





namespace duckdb {

CreateTableInfo::CreateTableInfo() : CreateInfo(CatalogType::TABLE_ENTRY, INVALID_SCHEMA) {
}

CreateTableInfo::CreateTableInfo(string catalog_p, string schema_p, string name_p)
    : CreateInfo(CatalogType::TABLE_ENTRY, std::move(schema_p), std::move(catalog_p)), table(std::move(name_p)) {
}

CreateTableInfo::CreateTableInfo(SchemaCatalogEntry *schema, string name_p)
    : CreateTableInfo(schema->catalog->GetName(), schema->name, std::move(name_p)) {
}

void CreateTableInfo::SerializeInternal(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteString(table);
	columns.Serialize(writer);
	writer.WriteSerializableList(constraints);
	writer.WriteOptional(query);
	writer.Finalize();
}

unique_ptr<CreateTableInfo> CreateTableInfo::Deserialize(Deserializer &deserializer) {
	auto result = make_unique<CreateTableInfo>();
	result->DeserializeBase(deserializer);

	FieldReader reader(deserializer);
	result->table = reader.ReadRequired<string>();
	result->columns = ColumnList::Deserialize(reader);
	result->constraints = reader.ReadRequiredSerializableList<Constraint>();
	result->query = reader.ReadOptional<SelectStatement>(nullptr);
	reader.Finalize();

	return result;
}

unique_ptr<CreateInfo> CreateTableInfo::Copy() const {
	auto result = make_unique<CreateTableInfo>(catalog, schema, table);
	CopyProperties(*result);
	result->columns = columns.Copy();
	for (auto &constraint : constraints) {
		result->constraints.push_back(constraint->Copy());
	}
	if (query) {
		result->query = unique_ptr_cast<SQLStatement, SelectStatement>(query->Copy());
	}
	return std::move(result);
}

} // namespace duckdb








namespace duckdb {

CreateViewInfo::CreateViewInfo() : CreateInfo(CatalogType::VIEW_ENTRY, INVALID_SCHEMA) {
}
CreateViewInfo::CreateViewInfo(string catalog_p, string schema_p, string view_name_p)
    : CreateInfo(CatalogType::VIEW_ENTRY, std::move(schema_p), std::move(catalog_p)),
      view_name(std::move(view_name_p)) {
}

CreateViewInfo::CreateViewInfo(SchemaCatalogEntry *schema, string view_name)
    : CreateViewInfo(schema->catalog->GetName(), schema->name, std::move(view_name)) {
}

unique_ptr<CreateInfo> CreateViewInfo::Copy() const {
	auto result = make_unique<CreateViewInfo>(catalog, schema, view_name);
	CopyProperties(*result);
	result->aliases = aliases;
	result->types = types;
	result->query = unique_ptr_cast<SQLStatement, SelectStatement>(query->Copy());
	return std::move(result);
}

unique_ptr<CreateViewInfo> CreateViewInfo::Deserialize(Deserializer &deserializer) {
	auto result = make_unique<CreateViewInfo>();
	result->DeserializeBase(deserializer);

	FieldReader reader(deserializer);
	result->view_name = reader.ReadRequired<string>();
	result->aliases = reader.ReadRequiredList<string>();
	result->types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
	result->query = reader.ReadOptional<SelectStatement>(nullptr);
	reader.Finalize();

	return result;
}

void CreateViewInfo::SerializeInternal(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteString(view_name);
	writer.WriteList<string>(aliases);
	writer.WriteRegularSerializableList(types);
	writer.WriteOptional(query);
	writer.Finalize();
}

unique_ptr<CreateViewInfo> CreateViewInfo::FromSelect(ClientContext &context, unique_ptr<CreateViewInfo> info) {
	D_ASSERT(info);
	D_ASSERT(!info->view_name.empty());
	D_ASSERT(!info->sql.empty());
	D_ASSERT(!info->query);

	Parser parser;
	parser.ParseQuery(info->sql);
	if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
		throw BinderException(
		    "Failed to create view from SQL string - \"%s\" - statement did not contain a single SELECT statement",
		    info->sql);
	}
	D_ASSERT(parser.statements.size() == 1 && parser.statements[0]->type == StatementType::SELECT_STATEMENT);
	info->query = unique_ptr_cast<SQLStatement, SelectStatement>(std::move(parser.statements[0]));

	auto binder = Binder::CreateBinder(context);
	binder->BindCreateViewInfo(*info);

	return info;
}

unique_ptr<CreateViewInfo> CreateViewInfo::FromCreateView(ClientContext &context, const string &sql) {
	D_ASSERT(!sql.empty());

	// parse the SQL statement
	Parser parser;
	parser.ParseQuery(sql);

	if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::CREATE_STATEMENT) {
		throw BinderException(
		    "Failed to create view from SQL string - \"%s\" - statement did not contain a single CREATE VIEW statement",
		    sql);
	}
	auto &create_statement = (CreateStatement &)*parser.statements[0];
	if (create_statement.info->type != CatalogType::VIEW_ENTRY) {
		throw BinderException(
		    "Failed to create view from SQL string - \"%s\" - view did not contain a CREATE VIEW statement", sql);
	}

	auto result = unique_ptr_cast<CreateInfo, CreateViewInfo>(std::move(create_statement.info));

	auto binder = Binder::CreateBinder(context);
	binder->BindCreateViewInfo(*result);

	return result;
}

} // namespace duckdb



namespace duckdb {

string SampleMethodToString(SampleMethod method) {
	switch (method) {
	case SampleMethod::SYSTEM_SAMPLE:
		return "System";
	case SampleMethod::BERNOULLI_SAMPLE:
		return "Bernoulli";
	case SampleMethod::RESERVOIR_SAMPLE:
		return "Reservoir";
	default:
		return "Unknown";
	}
}

void SampleOptions::Serialize(Serializer &serializer) {
	FieldWriter writer(serializer);
	writer.WriteSerializable(sample_size);
	writer.WriteField<bool>(is_percentage);
	writer.WriteField<SampleMethod>(method);
	writer.WriteField<int64_t>(seed);
	writer.Finalize();
}

unique_ptr<SampleOptions> SampleOptions::Deserialize(Deserializer &source) {
	auto result = make_unique<SampleOptions>();

	FieldReader reader(source);
	result->sample_size = reader.ReadRequiredSerializable<Value, Value>();
	result->is_percentage = reader.ReadRequired<bool>();
	result->method = reader.ReadRequired<SampleMethod>();
	result->seed = reader.ReadRequired<int64_t>();
	reader.Finalize();

	return result;
}

unique_ptr<SampleOptions> SampleOptions::Copy() {
	auto result = make_unique<SampleOptions>();
	result->sample_size = sample_size;
	result->is_percentage = is_percentage;
	result->method = method;
	result->seed = seed;
	return result;
}

bool SampleOptions::Equals(SampleOptions *a, SampleOptions *b) {
	if (a == b) {
		return true;
	}
	if (!a || !b) {
		return false;
	}
	if (a->sample_size != b->sample_size || a->is_percentage != b->is_percentage || a->method != b->method ||
	    a->seed != b->seed) {
		return false;
	}
	return true;
}

} // namespace duckdb








namespace duckdb {

bool ParsedExpression::IsAggregate() const {
	bool is_aggregate = false;
	ParsedExpressionIterator::EnumerateChildren(
	    *this, [&](const ParsedExpression &child) { is_aggregate |= child.IsAggregate(); });
	return is_aggregate;
}

bool ParsedExpression::IsWindow() const {
	bool is_window = false;
	ParsedExpressionIterator::EnumerateChildren(*this,
	                                            [&](const ParsedExpression &child) { is_window |= child.IsWindow(); });
	return is_window;
}

bool ParsedExpression::IsScalar() const {
	bool is_scalar = true;
	ParsedExpressionIterator::EnumerateChildren(*this, [&](const ParsedExpression &child) {
		if (!child.IsScalar()) {
			is_scalar = false;
		}
	});
	return is_scalar;
}

bool ParsedExpression::HasParameter() const {
	bool has_parameter = false;
	ParsedExpressionIterator::EnumerateChildren(
	    *this, [&](const ParsedExpression &child) { has_parameter |= child.HasParameter(); });
	return has_parameter;
}

bool ParsedExpression::HasSubquery() const {
	bool has_subquery = false;
	ParsedExpressionIterator::EnumerateChildren(
	    *this, [&](const ParsedExpression &child) { has_subquery |= child.HasSubquery(); });
	return has_subquery;
}

bool ParsedExpression::Equals(const BaseExpression *other) const {
	if (!BaseExpression::Equals(other)) {
		return false;
	}
	switch (expression_class) {
	case ExpressionClass::BETWEEN:
		return BetweenExpression::Equal((BetweenExpression *)this, (BetweenExpression *)other);
	case ExpressionClass::CASE:
		return CaseExpression::Equal((CaseExpression *)this, (CaseExpression *)other);
	case ExpressionClass::CAST:
		return CastExpression::Equal((CastExpression *)this, (CastExpression *)other);
	case ExpressionClass::COLLATE:
		return CollateExpression::Equal((CollateExpression *)this, (CollateExpression *)other);
	case ExpressionClass::COLUMN_REF:
		return ColumnRefExpression::Equal((ColumnRefExpression *)this, (ColumnRefExpression *)other);
	case ExpressionClass::COMPARISON:
		return ComparisonExpression::Equal((ComparisonExpression *)this, (ComparisonExpression *)other);
	case ExpressionClass::CONJUNCTION:
		return ConjunctionExpression::Equal((ConjunctionExpression *)this, (ConjunctionExpression *)other);
	case ExpressionClass::CONSTANT:
		return ConstantExpression::Equal((ConstantExpression *)this, (ConstantExpression *)other);
	case ExpressionClass::DEFAULT:
		return true;
	case ExpressionClass::FUNCTION:
		return FunctionExpression::Equal((FunctionExpression *)this, (FunctionExpression *)other);
	case ExpressionClass::LAMBDA:
		return LambdaExpression::Equal((LambdaExpression *)this, (LambdaExpression *)other);
	case ExpressionClass::OPERATOR:
		return OperatorExpression::Equal((OperatorExpression *)this, (OperatorExpression *)other);
	case ExpressionClass::PARAMETER:
		return ParameterExpression::Equal((ParameterExpression *)this, (ParameterExpression *)other);
	case ExpressionClass::POSITIONAL_REFERENCE:
		return PositionalReferenceExpression::Equal((PositionalReferenceExpression *)this,
		                                            (PositionalReferenceExpression *)other);
	case ExpressionClass::STAR:
		return StarExpression::Equal((StarExpression *)this, (StarExpression *)other);
	case ExpressionClass::SUBQUERY:
		return SubqueryExpression::Equal((SubqueryExpression *)this, (SubqueryExpression *)other);
	case ExpressionClass::WINDOW:
		return WindowExpression::Equal((WindowExpression *)this, (WindowExpression *)other);
	default:
		throw SerializationException("Unsupported type for expression comparison!");
	}
}

hash_t ParsedExpression::Hash() const {
	hash_t hash = duckdb::Hash<uint32_t>((uint32_t)type);
	ParsedExpressionIterator::EnumerateChildren(
	    *this, [&](const ParsedExpression &child) { hash = CombineHash(child.Hash(), hash); });
	return hash;
}

void ParsedExpression::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteField<ExpressionClass>(GetExpressionClass());
	writer.WriteField<ExpressionType>(type);
	writer.WriteString(alias);
	Serialize(writer);
	writer.Finalize();
}

unique_ptr<ParsedExpression> ParsedExpression::Deserialize(Deserializer &source) {
	FieldReader reader(source);
	auto expression_class = reader.ReadRequired<ExpressionClass>();
	auto type = reader.ReadRequired<ExpressionType>();
	auto alias = reader.ReadRequired<string>();
	unique_ptr<ParsedExpression> result;
	switch (expression_class) {
	case ExpressionClass::BETWEEN:
		result = BetweenExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::CASE:
		result = CaseExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::CAST:
		result = CastExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::COLLATE:
		result = CollateExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::COLUMN_REF:
		result = ColumnRefExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::COMPARISON:
		result = ComparisonExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::CONJUNCTION:
		result = ConjunctionExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::CONSTANT:
		result = ConstantExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::DEFAULT:
		result = DefaultExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::FUNCTION:
		result = FunctionExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::LAMBDA:
		result = LambdaExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::OPERATOR:
		result = OperatorExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::PARAMETER:
		result = ParameterExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::POSITIONAL_REFERENCE:
		result = PositionalReferenceExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::STAR:
		result = StarExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::SUBQUERY:
		result = SubqueryExpression::Deserialize(type, reader);
		break;
	case ExpressionClass::WINDOW:
		result = WindowExpression::Deserialize(type, reader);
		break;
	default:
		throw SerializationException("Unsupported type for expression deserialization: '%s'!",
		                             ExpressionClassToString(expression_class));
	}
	result->alias = alias;
	reader.Finalize();
	return result;
}

} // namespace duckdb









namespace duckdb {

void ParsedExpressionIterator::EnumerateChildren(const ParsedExpression &expression,
                                                 const std::function<void(const ParsedExpression &child)> &callback) {
	EnumerateChildren((ParsedExpression &)expression, [&](unique_ptr<ParsedExpression> &child) {
		D_ASSERT(child);
		callback(*child);
	});
}

void ParsedExpressionIterator::EnumerateChildren(ParsedExpression &expr,
                                                 const std::function<void(ParsedExpression &child)> &callback) {
	EnumerateChildren(expr, [&](unique_ptr<ParsedExpression> &child) {
		D_ASSERT(child);
		callback(*child);
	});
}

void ParsedExpressionIterator::EnumerateChildren(
    ParsedExpression &expr, const std::function<void(unique_ptr<ParsedExpression> &child)> &callback) {
	switch (expr.expression_class) {
	case ExpressionClass::BETWEEN: {
		auto &cast_expr = (BetweenExpression &)expr;
		callback(cast_expr.input);
		callback(cast_expr.lower);
		callback(cast_expr.upper);
		break;
	}
	case ExpressionClass::CASE: {
		auto &case_expr = (CaseExpression &)expr;
		for (auto &check : case_expr.case_checks) {
			callback(check.when_expr);
			callback(check.then_expr);
		}
		callback(case_expr.else_expr);
		break;
	}
	case ExpressionClass::CAST: {
		auto &cast_expr = (CastExpression &)expr;
		callback(cast_expr.child);
		break;
	}
	case ExpressionClass::COLLATE: {
		auto &cast_expr = (CollateExpression &)expr;
		callback(cast_expr.child);
		break;
	}
	case ExpressionClass::COMPARISON: {
		auto &comp_expr = (ComparisonExpression &)expr;
		callback(comp_expr.left);
		callback(comp_expr.right);
		break;
	}
	case ExpressionClass::CONJUNCTION: {
		auto &conj_expr = (ConjunctionExpression &)expr;
		for (auto &child : conj_expr.children) {
			callback(child);
		}
		break;
	}

	case ExpressionClass::FUNCTION: {
		auto &func_expr = (FunctionExpression &)expr;
		for (auto &child : func_expr.children) {
			callback(child);
		}
		if (func_expr.filter) {
			callback(func_expr.filter);
		}
		if (func_expr.order_bys) {
			for (auto &order : func_expr.order_bys->orders) {
				callback(order.expression);
			}
		}
		break;
	}
	case ExpressionClass::LAMBDA: {
		auto &lambda_expr = (LambdaExpression &)expr;
		callback(lambda_expr.lhs);
		callback(lambda_expr.expr);
		break;
	}
	case ExpressionClass::OPERATOR: {
		auto &op_expr = (OperatorExpression &)expr;
		for (auto &child : op_expr.children) {
			callback(child);
		}
		break;
	}
	case ExpressionClass::SUBQUERY: {
		auto &subquery_expr = (SubqueryExpression &)expr;
		if (subquery_expr.child) {
			callback(subquery_expr.child);
		}
		break;
	}
	case ExpressionClass::WINDOW: {
		auto &window_expr = (WindowExpression &)expr;
		for (auto &partition : window_expr.partitions) {
			callback(partition);
		}
		for (auto &order : window_expr.orders) {
			callback(order.expression);
		}
		for (auto &child : window_expr.children) {
			callback(child);
		}
		if (window_expr.filter_expr) {
			callback(window_expr.filter_expr);
		}
		if (window_expr.start_expr) {
			callback(window_expr.start_expr);
		}
		if (window_expr.end_expr) {
			callback(window_expr.end_expr);
		}
		if (window_expr.offset_expr) {
			callback(window_expr.offset_expr);
		}
		if (window_expr.default_expr) {
			callback(window_expr.default_expr);
		}
		break;
	}
	case ExpressionClass::BOUND_EXPRESSION:
	case ExpressionClass::COLUMN_REF:
	case ExpressionClass::CONSTANT:
	case ExpressionClass::DEFAULT:
	case ExpressionClass::STAR:
	case ExpressionClass::PARAMETER:
	case ExpressionClass::POSITIONAL_REFERENCE:
		// these node types have no children
		break;
	default:
		// called on non ParsedExpression type!
		throw NotImplementedException("Unimplemented expression class");
	}
}

void ParsedExpressionIterator::EnumerateQueryNodeModifiers(
    QueryNode &node, const std::function<void(unique_ptr<ParsedExpression> &child)> &callback) {

	for (auto &modifier : node.modifiers) {
		switch (modifier->type) {
		case ResultModifierType::LIMIT_MODIFIER: {
			auto &limit_modifier = (LimitModifier &)*modifier;
			if (limit_modifier.limit) {
				callback(limit_modifier.limit);
			}
			if (limit_modifier.offset) {
				callback(limit_modifier.offset);
			}
		} break;

		case ResultModifierType::LIMIT_PERCENT_MODIFIER: {
			auto &limit_modifier = (LimitPercentModifier &)*modifier;
			if (limit_modifier.limit) {
				callback(limit_modifier.limit);
			}
			if (limit_modifier.offset) {
				callback(limit_modifier.offset);
			}
		} break;

		case ResultModifierType::ORDER_MODIFIER: {
			auto &order_modifier = (OrderModifier &)*modifier;
			for (auto &order : order_modifier.orders) {
				callback(order.expression);
			}
		} break;

		case ResultModifierType::DISTINCT_MODIFIER: {
			auto &distinct_modifier = (DistinctModifier &)*modifier;
			for (auto &target : distinct_modifier.distinct_on_targets) {
				callback(target);
			}
		} break;

		// do nothing
		default:
			break;
		}
	}
}

void ParsedExpressionIterator::EnumerateTableRefChildren(
    TableRef &ref, const std::function<void(unique_ptr<ParsedExpression> &child)> &callback) {
	switch (ref.type) {
	case TableReferenceType::EXPRESSION_LIST: {
		auto &el_ref = (ExpressionListRef &)ref;
		for (idx_t i = 0; i < el_ref.values.size(); i++) {
			for (idx_t j = 0; j < el_ref.values[i].size(); j++) {
				callback(el_ref.values[i][j]);
			}
		}
		break;
	}
	case TableReferenceType::JOIN: {
		auto &j_ref = (JoinRef &)ref;
		EnumerateTableRefChildren(*j_ref.left, callback);
		EnumerateTableRefChildren(*j_ref.right, callback);
		if (j_ref.condition) {
			callback(j_ref.condition);
		}
		break;
	}
	case TableReferenceType::SUBQUERY: {
		auto &sq_ref = (SubqueryRef &)ref;
		EnumerateQueryNodeChildren(*sq_ref.subquery->node, callback);
		break;
	}
	case TableReferenceType::TABLE_FUNCTION: {
		auto &tf_ref = (TableFunctionRef &)ref;
		callback(tf_ref.function);
		break;
	}
	case TableReferenceType::BASE_TABLE:
	case TableReferenceType::EMPTY:
		// these TableRefs do not need to be unfolded
		break;
	case TableReferenceType::INVALID:
	case TableReferenceType::CTE:
		throw NotImplementedException("TableRef type not implemented for traversal");
	}
}

void ParsedExpressionIterator::EnumerateQueryNodeChildren(
    QueryNode &node, const std::function<void(unique_ptr<ParsedExpression> &child)> &callback) {
	switch (node.type) {
	case QueryNodeType::RECURSIVE_CTE_NODE: {
		auto &rcte_node = (RecursiveCTENode &)node;
		EnumerateQueryNodeChildren(*rcte_node.left, callback);
		EnumerateQueryNodeChildren(*rcte_node.right, callback);
		break;
	}
	case QueryNodeType::SELECT_NODE: {
		auto &sel_node = (SelectNode &)node;
		for (idx_t i = 0; i < sel_node.select_list.size(); i++) {
			callback(sel_node.select_list[i]);
		}
		for (idx_t i = 0; i < sel_node.groups.group_expressions.size(); i++) {
			callback(sel_node.groups.group_expressions[i]);
		}
		if (sel_node.where_clause) {
			callback(sel_node.where_clause);
		}
		if (sel_node.having) {
			callback(sel_node.having);
		}
		if (sel_node.qualify) {
			callback(sel_node.qualify);
		}

		EnumerateTableRefChildren(*sel_node.from_table.get(), callback);
		break;
	}
	case QueryNodeType::SET_OPERATION_NODE: {
		auto &setop_node = (SetOperationNode &)node;
		EnumerateQueryNodeChildren(*setop_node.left, callback);
		EnumerateQueryNodeChildren(*setop_node.right, callback);
		break;
	}
	default:
		throw NotImplementedException("QueryNode type not implemented for traversal");
	}

	if (!node.modifiers.empty()) {
		EnumerateQueryNodeModifiers(node, callback);
	}

	for (auto &kv : node.cte_map.map) {
		EnumerateQueryNodeChildren(*kv.second->query->node, callback);
	}
}

} // namespace duckdb
















namespace duckdb {

Parser::Parser(ParserOptions options_p) : options(options_p) {
}

struct UnicodeSpace {
	UnicodeSpace(idx_t pos, idx_t bytes) : pos(pos), bytes(bytes) {
	}

	idx_t pos;
	idx_t bytes;
};

static bool ReplaceUnicodeSpaces(const string &query, string &new_query, vector<UnicodeSpace> &unicode_spaces) {
	if (unicode_spaces.empty()) {
		// no unicode spaces found
		return false;
	}
	idx_t prev = 0;
	for (auto &usp : unicode_spaces) {
		new_query += query.substr(prev, usp.pos - prev);
		new_query += " ";
		prev = usp.pos + usp.bytes;
	}
	new_query += query.substr(prev, query.size() - prev);
	return true;
}

// This function strips unicode space characters from the query and replaces them with regular spaces
// It returns true if any unicode space characters were found and stripped
// See here for a list of unicode space characters - https://jkorpela.fi/chars/spaces.html
static bool StripUnicodeSpaces(const string &query_str, string &new_query) {
	const idx_t NBSP_LEN = 2;
	const idx_t USP_LEN = 3;
	idx_t pos = 0;
	unsigned char quote;
	vector<UnicodeSpace> unicode_spaces;
	auto query = (unsigned char *)query_str.c_str();
	auto qsize = query_str.size();

regular:
	for (; pos + 2 < qsize; pos++) {
		if (query[pos] == 0xC2) {
			if (query[pos + 1] == 0xA0) {
				// U+00A0 - C2A0
				unicode_spaces.emplace_back(pos, NBSP_LEN);
			}
		}
		if (query[pos] == 0xE2) {
			if (query[pos + 1] == 0x80) {
				if (query[pos + 2] >= 0x80 && query[pos + 2] <= 0x8B) {
					// U+2000 to U+200B
					// E28080 - E2808B
					unicode_spaces.emplace_back(pos, USP_LEN);
				} else if (query[pos + 2] == 0xAF) {
					// U+202F - E280AF
					unicode_spaces.emplace_back(pos, USP_LEN);
				}
			} else if (query[pos + 1] == 0x81) {
				if (query[pos + 2] == 0x9F) {
					// U+205F - E2819f
					unicode_spaces.emplace_back(pos, USP_LEN);
				} else if (query[pos + 2] == 0xA0) {
					// U+2060 - E281A0
					unicode_spaces.emplace_back(pos, USP_LEN);
				}
			}
		} else if (query[pos] == 0xE3) {
			if (query[pos + 1] == 0x80 && query[pos + 2] == 0x80) {
				// U+3000 - E38080
				unicode_spaces.emplace_back(pos, USP_LEN);
			}
		} else if (query[pos] == 0xEF) {
			if (query[pos + 1] == 0xBB && query[pos + 2] == 0xBF) {
				// U+FEFF - EFBBBF
				unicode_spaces.emplace_back(pos, USP_LEN);
			}
		} else if (query[pos] == '"' || query[pos] == '\'') {
			quote = query[pos];
			pos++;
			goto in_quotes;
		} else if (query[pos] == '-' && query[pos + 1] == '-') {
			goto in_comment;
		}
	}
	goto end;
in_quotes:
	for (; pos + 1 < qsize; pos++) {
		if (query[pos] == quote) {
			if (query[pos + 1] == quote) {
				// escaped quote
				pos++;
				continue;
			}
			pos++;
			goto regular;
		}
	}
	goto end;
in_comment:
	for (; pos < qsize; pos++) {
		if (query[pos] == '\n' || query[pos] == '\r') {
			goto regular;
		}
	}
	goto end;
end:
	return ReplaceUnicodeSpaces(query_str, new_query, unicode_spaces);
}

void Parser::ParseQuery(const string &query) {
	Transformer transformer(options.max_expression_depth);
	string parser_error;
	{
		// check if there are any unicode spaces in the string
		string new_query;
		if (StripUnicodeSpaces(query, new_query)) {
			// there are - strip the unicode spaces and re-run the query
			ParseQuery(new_query);
			return;
		}
	}
	{
		PostgresParser::SetPreserveIdentifierCase(options.preserve_identifier_case);
		PostgresParser parser;
		parser.Parse(query);
		if (parser.success) {
			if (!parser.parse_tree) {
				// empty statement
				return;
			}

			// if it succeeded, we transform the Postgres parse tree into a list of
			// SQLStatements
			transformer.TransformParseTree(parser.parse_tree, statements);
		} else {
			parser_error = QueryErrorContext::Format(query, parser.error_message, parser.error_location - 1);
		}
	}
	if (!parser_error.empty()) {
		if (options.extensions) {
			for (auto &ext : *options.extensions) {
				D_ASSERT(ext.parse_function);
				auto result = ext.parse_function(ext.parser_info.get(), query);
				if (result.type == ParserExtensionResultType::PARSE_SUCCESSFUL) {
					auto statement = make_unique<ExtensionStatement>(ext, std::move(result.parse_data));
					statement->stmt_length = query.size();
					statement->stmt_location = 0;
					statements.push_back(std::move(statement));
					return;
				}
				if (result.type == ParserExtensionResultType::DISPLAY_EXTENSION_ERROR) {
					throw ParserException(result.error);
				}
			}
		}
		throw ParserException(parser_error);
	}
	if (!statements.empty()) {
		auto &last_statement = statements.back();
		last_statement->stmt_length = query.size() - last_statement->stmt_location;
		for (auto &statement : statements) {
			statement->query = query;
			if (statement->type == StatementType::CREATE_STATEMENT) {
				auto &create = (CreateStatement &)*statement;
				create.info->sql = query.substr(statement->stmt_location, statement->stmt_length);
			}
		}
	}
}

vector<SimplifiedToken> Parser::Tokenize(const string &query) {
	auto pg_tokens = PostgresParser::Tokenize(query);
	vector<SimplifiedToken> result;
	result.reserve(pg_tokens.size());
	for (auto &pg_token : pg_tokens) {
		SimplifiedToken token;
		switch (pg_token.type) {
		case duckdb_libpgquery::PGSimplifiedTokenType::PG_SIMPLIFIED_TOKEN_IDENTIFIER:
			token.type = SimplifiedTokenType::SIMPLIFIED_TOKEN_IDENTIFIER;
			break;
		case duckdb_libpgquery::PGSimplifiedTokenType::PG_SIMPLIFIED_TOKEN_NUMERIC_CONSTANT:
			token.type = SimplifiedTokenType::SIMPLIFIED_TOKEN_NUMERIC_CONSTANT;
			break;
		case duckdb_libpgquery::PGSimplifiedTokenType::PG_SIMPLIFIED_TOKEN_STRING_CONSTANT:
			token.type = SimplifiedTokenType::SIMPLIFIED_TOKEN_STRING_CONSTANT;
			break;
		case duckdb_libpgquery::PGSimplifiedTokenType::PG_SIMPLIFIED_TOKEN_OPERATOR:
			token.type = SimplifiedTokenType::SIMPLIFIED_TOKEN_OPERATOR;
			break;
		case duckdb_libpgquery::PGSimplifiedTokenType::PG_SIMPLIFIED_TOKEN_KEYWORD:
			token.type = SimplifiedTokenType::SIMPLIFIED_TOKEN_KEYWORD;
			break;
		// comments are not supported by our tokenizer right now
		case duckdb_libpgquery::PGSimplifiedTokenType::PG_SIMPLIFIED_TOKEN_COMMENT: // LCOV_EXCL_START
			token.type = SimplifiedTokenType::SIMPLIFIED_TOKEN_COMMENT;
			break;
		default:
			throw InternalException("Unrecognized token category");
		} // LCOV_EXCL_STOP
		token.start = pg_token.start;
		result.push_back(token);
	}
	return result;
}

bool Parser::IsKeyword(const string &text) {
	return PostgresParser::IsKeyword(text);
}

vector<ParserKeyword> Parser::KeywordList() {
	auto keywords = PostgresParser::KeywordList();
	vector<ParserKeyword> result;
	for (auto &kw : keywords) {
		ParserKeyword res;
		res.name = kw.text;
		switch (kw.category) {
		case duckdb_libpgquery::PGKeywordCategory::PG_KEYWORD_RESERVED:
			res.category = KeywordCategory::KEYWORD_RESERVED;
			break;
		case duckdb_libpgquery::PGKeywordCategory::PG_KEYWORD_UNRESERVED:
			res.category = KeywordCategory::KEYWORD_UNRESERVED;
			break;
		case duckdb_libpgquery::PGKeywordCategory::PG_KEYWORD_TYPE_FUNC:
			res.category = KeywordCategory::KEYWORD_TYPE_FUNC;
			break;
		case duckdb_libpgquery::PGKeywordCategory::PG_KEYWORD_COL_NAME:
			res.category = KeywordCategory::KEYWORD_COL_NAME;
			break;
		default:
			throw InternalException("Unrecognized keyword category");
		}
		result.push_back(res);
	}
	return result;
}

vector<unique_ptr<ParsedExpression>> Parser::ParseExpressionList(const string &select_list, ParserOptions options) {
	// construct a mock query prefixed with SELECT
	string mock_query = "SELECT " + select_list;
	// parse the query
	Parser parser(options);
	parser.ParseQuery(mock_query);
	// check the statements
	if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
		throw ParserException("Expected a single SELECT statement");
	}
	auto &select = (SelectStatement &)*parser.statements[0];
	if (select.node->type != QueryNodeType::SELECT_NODE) {
		throw ParserException("Expected a single SELECT node");
	}
	auto &select_node = (SelectNode &)*select.node;
	return std::move(select_node.select_list);
}

vector<OrderByNode> Parser::ParseOrderList(const string &select_list, ParserOptions options) {
	// construct a mock query
	string mock_query = "SELECT * FROM tbl ORDER BY " + select_list;
	// parse the query
	Parser parser(options);
	parser.ParseQuery(mock_query);
	// check the statements
	if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
		throw ParserException("Expected a single SELECT statement");
	}
	auto &select = (SelectStatement &)*parser.statements[0];
	if (select.node->type != QueryNodeType::SELECT_NODE) {
		throw ParserException("Expected a single SELECT node");
	}
	auto &select_node = (SelectNode &)*select.node;
	if (select_node.modifiers.empty() || select_node.modifiers[0]->type != ResultModifierType::ORDER_MODIFIER ||
	    select_node.modifiers.size() != 1) {
		throw ParserException("Expected a single ORDER clause");
	}
	auto &order = (OrderModifier &)*select_node.modifiers[0];
	return std::move(order.orders);
}

void Parser::ParseUpdateList(const string &update_list, vector<string> &update_columns,
                             vector<unique_ptr<ParsedExpression>> &expressions, ParserOptions options) {
	// construct a mock query
	string mock_query = "UPDATE tbl SET " + update_list;
	// parse the query
	Parser parser(options);
	parser.ParseQuery(mock_query);
	// check the statements
	if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::UPDATE_STATEMENT) {
		throw ParserException("Expected a single UPDATE statement");
	}
	auto &update = (UpdateStatement &)*parser.statements[0];
	update_columns = std::move(update.set_info->columns);
	expressions = std::move(update.set_info->expressions);
}

vector<vector<unique_ptr<ParsedExpression>>> Parser::ParseValuesList(const string &value_list, ParserOptions options) {
	// construct a mock query
	string mock_query = "VALUES " + value_list;
	// parse the query
	Parser parser(options);
	parser.ParseQuery(mock_query);
	// check the statements
	if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::SELECT_STATEMENT) {
		throw ParserException("Expected a single SELECT statement");
	}
	auto &select = (SelectStatement &)*parser.statements[0];
	if (select.node->type != QueryNodeType::SELECT_NODE) {
		throw ParserException("Expected a single SELECT node");
	}
	auto &select_node = (SelectNode &)*select.node;
	if (!select_node.from_table || select_node.from_table->type != TableReferenceType::EXPRESSION_LIST) {
		throw ParserException("Expected a single VALUES statement");
	}
	auto &values_list = (ExpressionListRef &)*select_node.from_table;
	return std::move(values_list.values);
}

ColumnList Parser::ParseColumnList(const string &column_list, ParserOptions options) {
	string mock_query = "CREATE TABLE blabla (" + column_list + ")";
	Parser parser(options);
	parser.ParseQuery(mock_query);
	if (parser.statements.size() != 1 || parser.statements[0]->type != StatementType::CREATE_STATEMENT) {
		throw ParserException("Expected a single CREATE statement");
	}
	auto &create = (CreateStatement &)*parser.statements[0];
	if (create.info->type != CatalogType::TABLE_ENTRY) {
		throw InternalException("Expected a single CREATE TABLE statement");
	}
	auto &info = ((CreateTableInfo &)*create.info);
	return std::move(info.columns);
}

} // namespace duckdb







namespace duckdb {

string QueryErrorContext::Format(const string &query, const string &error_message, int error_loc) {
	if (error_loc < 0 || size_t(error_loc) >= query.size()) {
		// no location in query provided
		return error_message;
	}
	idx_t error_location = idx_t(error_loc);
	// count the line numbers until the error location
	// and set the start position as the first character of that line
	idx_t start_pos = 0;
	idx_t line_number = 1;
	for (idx_t i = 0; i < error_location; i++) {
		if (StringUtil::CharacterIsNewline(query[i])) {
			line_number++;
			start_pos = i + 1;
		}
	}
	// now find either the next newline token after the query, or find the end of string
	// this is the initial end position
	idx_t end_pos = query.size();
	for (idx_t i = error_location; i < query.size(); i++) {
		if (StringUtil::CharacterIsNewline(query[i])) {
			end_pos = i;
			break;
		}
	}
	// now start scanning from the start pos
	// we want to figure out the start and end pos of what we are going to render
	// we want to render at most 80 characters in total, with the error_location located in the middle
	const char *buf = query.c_str() + start_pos;
	idx_t len = end_pos - start_pos;
	vector<idx_t> render_widths;
	vector<idx_t> positions;
	if (Utf8Proc::IsValid(buf, len)) {
		// for unicode awareness, we traverse the graphemes of the current line and keep track of their render widths
		// and of their position in the string
		for (idx_t cpos = 0; cpos < len;) {
			auto char_render_width = Utf8Proc::RenderWidth(buf, len, cpos);
			positions.push_back(cpos);
			render_widths.push_back(char_render_width);
			cpos = Utf8Proc::NextGraphemeCluster(buf, len, cpos);
		}
	} else { // LCOV_EXCL_START
		// invalid utf-8, we can't do much at this point
		// we just assume every character is a character, and every character has a render width of 1
		for (idx_t cpos = 0; cpos < len; cpos++) {
			positions.push_back(cpos);
			render_widths.push_back(1);
		}
	} // LCOV_EXCL_STOP
	// now we want to find the (unicode aware) start and end position
	idx_t epos = 0;
	// start by finding the error location inside the array
	for (idx_t i = 0; i < positions.size(); i++) {
		if (positions[i] >= (error_location - start_pos)) {
			epos = i;
			break;
		}
	}
	bool truncate_beginning = false;
	bool truncate_end = false;
	idx_t spos = 0;
	// now we iterate backwards from the error location
	// we show max 40 render width before the error location
	idx_t current_render_width = 0;
	for (idx_t i = epos; i > 0; i--) {
		current_render_width += render_widths[i];
		if (current_render_width >= 40) {
			truncate_beginning = true;
			start_pos = positions[i];
			spos = i;
			break;
		}
	}
	// now do the same, but going forward
	current_render_width = 0;
	for (idx_t i = epos; i < positions.size(); i++) {
		current_render_width += render_widths[i];
		if (current_render_width >= 40) {
			truncate_end = true;
			end_pos = positions[i];
			break;
		}
	}
	string line_indicator = "LINE " + to_string(line_number) + ": ";
	string begin_trunc = truncate_beginning ? "..." : "";
	string end_trunc = truncate_end ? "..." : "";

	// get the render width of the error indicator (i.e. how many spaces we need to insert before the ^)
	idx_t error_render_width = 0;
	for (idx_t i = spos; i < epos; i++) {
		error_render_width += render_widths[i];
	}
	error_render_width += line_indicator.size() + begin_trunc.size();

	// now first print the error message plus the current line (or a subset of the line)
	string result = error_message;
	result += "\n" + line_indicator + begin_trunc + query.substr(start_pos, end_pos - start_pos) + end_trunc;
	// print an arrow pointing at the error location
	result += "\n" + string(error_render_width, ' ') + "^";
	return result;
}

string QueryErrorContext::FormatErrorRecursive(const string &msg, vector<ExceptionFormatValue> &values) {
	string error_message = values.empty() ? msg : ExceptionFormatValue::Format(msg, values);
	if (!statement || query_location >= statement->query.size()) {
		// no statement provided or query location out of range
		return error_message;
	}
	return Format(statement->query, error_message, query_location);
}

} // namespace duckdb



namespace duckdb {

string RecursiveCTENode::ToString() const {
	string result;
	result += "(" + left->ToString() + ")";
	result += " UNION ";
	if (union_all) {
		result += " ALL ";
	}
	result += "(" + right->ToString() + ")";
	return result;
}

bool RecursiveCTENode::Equals(const QueryNode *other_p) const {
	if (!QueryNode::Equals(other_p)) {
		return false;
	}
	if (this == other_p) {
		return true;
	}
	auto other = (RecursiveCTENode *)other_p;

	if (other->union_all != union_all) {
		return false;
	}
	if (!left->Equals(other->left.get())) {
		return false;
	}
	if (!right->Equals(other->right.get())) {
		return false;
	}
	return true;
}

unique_ptr<QueryNode> RecursiveCTENode::Copy() const {
	auto result = make_unique<RecursiveCTENode>();
	result->ctename = ctename;
	result->union_all = union_all;
	result->left = left->Copy();
	result->right = right->Copy();
	result->aliases = aliases;
	this->CopyProperties(*result);
	return std::move(result);
}

void RecursiveCTENode::Serialize(FieldWriter &writer) const {
	writer.WriteString(ctename);
	writer.WriteField<bool>(union_all);
	writer.WriteSerializable(*left);
	writer.WriteSerializable(*right);
	writer.WriteList<string>(aliases);
}

unique_ptr<QueryNode> RecursiveCTENode::Deserialize(FieldReader &reader) {
	auto result = make_unique<RecursiveCTENode>();
	result->ctename = reader.ReadRequired<string>();
	result->union_all = reader.ReadRequired<bool>();
	result->left = reader.ReadRequiredSerializable<QueryNode>();
	result->right = reader.ReadRequiredSerializable<QueryNode>();
	result->aliases = reader.ReadRequiredList<string>();
	return std::move(result);
}

} // namespace duckdb





namespace duckdb {

SelectNode::SelectNode()
    : QueryNode(QueryNodeType::SELECT_NODE), aggregate_handling(AggregateHandling::STANDARD_HANDLING) {
}

string SelectNode::ToString() const {
	string result;
	result = cte_map.ToString();
	result += "SELECT ";

	// search for a distinct modifier
	for (idx_t modifier_idx = 0; modifier_idx < modifiers.size(); modifier_idx++) {
		if (modifiers[modifier_idx]->type == ResultModifierType::DISTINCT_MODIFIER) {
			auto &distinct_modifier = (DistinctModifier &)*modifiers[modifier_idx];
			result += "DISTINCT ";
			if (!distinct_modifier.distinct_on_targets.empty()) {
				result += "ON (";
				for (idx_t k = 0; k < distinct_modifier.distinct_on_targets.size(); k++) {
					if (k > 0) {
						result += ", ";
					}
					result += distinct_modifier.distinct_on_targets[k]->ToString();
				}
				result += ") ";
			}
		}
	}
	for (idx_t i = 0; i < select_list.size(); i++) {
		if (i > 0) {
			result += ", ";
		}
		result += select_list[i]->ToString();
		if (!select_list[i]->alias.empty()) {
			result += " AS " + KeywordHelper::WriteOptionallyQuoted(select_list[i]->alias);
		}
	}
	if (from_table && from_table->type != TableReferenceType::EMPTY) {
		result += " FROM " + from_table->ToString();
	}
	if (where_clause) {
		result += " WHERE " + where_clause->ToString();
	}
	if (!groups.grouping_sets.empty()) {
		result += " GROUP BY ";
		// if we are dealing with multiple grouping sets, we have to add a few additional brackets
		bool grouping_sets = groups.grouping_sets.size() > 1;
		if (grouping_sets) {
			result += "GROUPING SETS (";
		}
		for (idx_t i = 0; i < groups.grouping_sets.size(); i++) {
			auto &grouping_set = groups.grouping_sets[i];
			if (i > 0) {
				result += ",";
			}
			if (grouping_set.empty()) {
				result += "()";
				continue;
			}
			if (grouping_sets) {
				result += "(";
			}
			bool first = true;
			for (auto &grp : grouping_set) {
				if (!first) {
					result += ", ";
				}
				result += groups.group_expressions[grp]->ToString();
				first = false;
			}
			if (grouping_sets) {
				result += ")";
			}
		}
		if (grouping_sets) {
			result += ")";
		}
	} else if (aggregate_handling == AggregateHandling::FORCE_AGGREGATES) {
		result += " GROUP BY ALL";
	}
	if (having) {
		result += " HAVING " + having->ToString();
	}
	if (qualify) {
		result += " QUALIFY " + qualify->ToString();
	}
	if (sample) {
		result += " USING SAMPLE ";
		result += sample->sample_size.ToString();
		if (sample->is_percentage) {
			result += "%";
		}
		result += " (" + SampleMethodToString(sample->method);
		if (sample->seed >= 0) {
			result += ", " + std::to_string(sample->seed);
		}
		result += ")";
	}
	return result + ResultModifiersToString();
}

bool SelectNode::Equals(const QueryNode *other_p) const {
	if (!QueryNode::Equals(other_p)) {
		return false;
	}
	if (this == other_p) {
		return true;
	}
	auto other = (SelectNode *)other_p;

	// SELECT
	if (!ExpressionUtil::ListEquals(select_list, other->select_list)) {
		return false;
	}
	// FROM
	if (from_table) {
		// we have a FROM clause, compare to the other one
		if (!from_table->Equals(other->from_table.get())) {
			return false;
		}
	} else if (other->from_table) {
		// we don't have a FROM clause, if the other statement has one they are
		// not equal
		return false;
	}
	// WHERE
	if (!BaseExpression::Equals(where_clause.get(), other->where_clause.get())) {
		return false;
	}
	// GROUP BY
	if (!ExpressionUtil::ListEquals(groups.group_expressions, other->groups.group_expressions)) {
		return false;
	}
	if (groups.grouping_sets != other->groups.grouping_sets) {
		return false;
	}
	if (!SampleOptions::Equals(sample.get(), other->sample.get())) {
		return false;
	}
	// HAVING
	if (!BaseExpression::Equals(having.get(), other->having.get())) {
		return false;
	}
	// QUALIFY
	if (!BaseExpression::Equals(qualify.get(), other->qualify.get())) {
		return false;
	}
	return true;
}

unique_ptr<QueryNode> SelectNode::Copy() const {
	auto result = make_unique<SelectNode>();
	for (auto &child : select_list) {
		result->select_list.push_back(child->Copy());
	}
	result->from_table = from_table ? from_table->Copy() : nullptr;
	result->where_clause = where_clause ? where_clause->Copy() : nullptr;
	// groups
	for (auto &group : groups.group_expressions) {
		result->groups.group_expressions.push_back(group->Copy());
	}
	result->groups.grouping_sets = groups.grouping_sets;
	result->aggregate_handling = aggregate_handling;
	result->having = having ? having->Copy() : nullptr;
	result->qualify = qualify ? qualify->Copy() : nullptr;
	result->sample = sample ? sample->Copy() : nullptr;
	this->CopyProperties(*result);
	return std::move(result);
}

void SelectNode::Serialize(FieldWriter &writer) const {
	writer.WriteSerializableList(select_list);
	writer.WriteOptional(from_table);
	writer.WriteOptional(where_clause);
	writer.WriteSerializableList(groups.group_expressions);
	writer.WriteField<uint32_t>(groups.grouping_sets.size());
	auto &serializer = writer.GetSerializer();
	for (auto &grouping_set : groups.grouping_sets) {
		serializer.Write<idx_t>(grouping_set.size());
		for (auto &idx : grouping_set) {
			serializer.Write<idx_t>(idx);
		}
	}
	writer.WriteField<AggregateHandling>(aggregate_handling);
	writer.WriteOptional(having);
	writer.WriteOptional(sample);
	writer.WriteOptional(qualify);
}

unique_ptr<QueryNode> SelectNode::Deserialize(FieldReader &reader) {
	auto result = make_unique<SelectNode>();
	result->select_list = reader.ReadRequiredSerializableList<ParsedExpression>();
	result->from_table = reader.ReadOptional<TableRef>(nullptr);
	result->where_clause = reader.ReadOptional<ParsedExpression>(nullptr);
	result->groups.group_expressions = reader.ReadRequiredSerializableList<ParsedExpression>();

	auto grouping_set_count = reader.ReadRequired<uint32_t>();
	auto &source = reader.GetSource();
	for (idx_t set_idx = 0; set_idx < grouping_set_count; set_idx++) {
		auto set_entries = source.Read<idx_t>();
		GroupingSet grouping_set;
		for (idx_t i = 0; i < set_entries; i++) {
			grouping_set.insert(source.Read<idx_t>());
		}
		result->groups.grouping_sets.push_back(grouping_set);
	}

	result->aggregate_handling = reader.ReadRequired<AggregateHandling>();
	result->having = reader.ReadOptional<ParsedExpression>(nullptr);
	result->sample = reader.ReadOptional<SampleOptions>(nullptr);
	result->qualify = reader.ReadOptional<ParsedExpression>(nullptr);
	return std::move(result);
}

} // namespace duckdb




namespace duckdb {

string SetOperationNode::ToString() const {
	string result;
	result = cte_map.ToString();
	result += "(" + left->ToString() + ") ";
	bool is_distinct = false;
	for (idx_t modifier_idx = 0; modifier_idx < modifiers.size(); modifier_idx++) {
		if (modifiers[modifier_idx]->type == ResultModifierType::DISTINCT_MODIFIER) {
			is_distinct = true;
			break;
		}
	}

	switch (setop_type) {
	case SetOperationType::UNION:
		result += is_distinct ? "UNION" : "UNION ALL";
		break;
	case SetOperationType::UNION_BY_NAME:
		result += is_distinct ? "UNION BY NAME" : "UNION ALL BY NAME";
		break;
	case SetOperationType::EXCEPT:
		D_ASSERT(is_distinct);
		result += "EXCEPT";
		break;
	case SetOperationType::INTERSECT:
		D_ASSERT(is_distinct);
		result += "INTERSECT";
		break;
	default:
		throw InternalException("Unsupported set operation type");
	}
	result += " (" + right->ToString() + ")";
	return result + ResultModifiersToString();
}

bool SetOperationNode::Equals(const QueryNode *other_p) const {
	if (!QueryNode::Equals(other_p)) {
		return false;
	}
	if (this == other_p) {
		return true;
	}
	auto other = (SetOperationNode *)other_p;
	if (setop_type != other->setop_type) {
		return false;
	}
	if (!left->Equals(other->left.get())) {
		return false;
	}
	if (!right->Equals(other->right.get())) {
		return false;
	}
	return true;
}

unique_ptr<QueryNode> SetOperationNode::Copy() const {
	auto result = make_unique<SetOperationNode>();
	result->setop_type = setop_type;
	result->left = left->Copy();
	result->right = right->Copy();
	this->CopyProperties(*result);
	return std::move(result);
}

void SetOperationNode::Serialize(FieldWriter &writer) const {
	writer.WriteField<SetOperationType>(setop_type);
	writer.WriteSerializable(*left);
	writer.WriteSerializable(*right);
}

unique_ptr<QueryNode> SetOperationNode::Deserialize(FieldReader &reader) {
	auto result = make_unique<SetOperationNode>();
	result->setop_type = reader.ReadRequired<SetOperationType>();
	result->left = reader.ReadRequiredSerializable<QueryNode>();
	result->right = reader.ReadRequiredSerializable<QueryNode>();
	return std::move(result);
}

} // namespace duckdb








namespace duckdb {

CommonTableExpressionMap::CommonTableExpressionMap() {
}

CommonTableExpressionMap CommonTableExpressionMap::Copy() const {
	CommonTableExpressionMap res;
	for (auto &kv : this->map) {
		auto kv_info = make_unique<CommonTableExpressionInfo>();
		for (auto &al : kv.second->aliases) {
			kv_info->aliases.push_back(al);
		}
		kv_info->query = unique_ptr_cast<SQLStatement, SelectStatement>(kv.second->query->Copy());
		res.map[kv.first] = std::move(kv_info);
	}
	return res;
}

string CommonTableExpressionMap::ToString() const {
	if (map.empty()) {
		return string();
	}
	// check if there are any recursive CTEs
	bool has_recursive = false;
	for (auto &kv : map) {
		if (kv.second->query->node->type == QueryNodeType::RECURSIVE_CTE_NODE) {
			has_recursive = true;
			break;
		}
	}
	string result = "WITH ";
	if (has_recursive) {
		result += "RECURSIVE ";
	}
	bool first_cte = true;
	for (auto &kv : map) {
		if (!first_cte) {
			result += ", ";
		}
		auto &cte = *kv.second;
		result += KeywordHelper::WriteOptionallyQuoted(kv.first);
		if (!cte.aliases.empty()) {
			result += " (";
			for (idx_t k = 0; k < cte.aliases.size(); k++) {
				if (k > 0) {
					result += ", ";
				}
				result += KeywordHelper::WriteOptionallyQuoted(cte.aliases[k]);
			}
			result += ")";
		}
		result += " AS (";
		result += cte.query->ToString();
		result += ")";
		first_cte = false;
	}
	return result;
}

string QueryNode::ResultModifiersToString() const {
	string result;
	for (idx_t modifier_idx = 0; modifier_idx < modifiers.size(); modifier_idx++) {
		auto &modifier = *modifiers[modifier_idx];
		if (modifier.type == ResultModifierType::ORDER_MODIFIER) {
			auto &order_modifier = (OrderModifier &)modifier;
			result += " ORDER BY ";
			for (idx_t k = 0; k < order_modifier.orders.size(); k++) {
				if (k > 0) {
					result += ", ";
				}
				result += order_modifier.orders[k].ToString();
			}
		} else if (modifier.type == ResultModifierType::LIMIT_MODIFIER) {
			auto &limit_modifier = (LimitModifier &)modifier;
			if (limit_modifier.limit) {
				result += " LIMIT " + limit_modifier.limit->ToString();
			}
			if (limit_modifier.offset) {
				result += " OFFSET " + limit_modifier.offset->ToString();
			}
		} else if (modifier.type == ResultModifierType::LIMIT_PERCENT_MODIFIER) {
			auto &limit_p_modifier = (LimitPercentModifier &)modifier;
			if (limit_p_modifier.limit) {
				result += " LIMIT (" + limit_p_modifier.limit->ToString() + ") %";
			}
			if (limit_p_modifier.offset) {
				result += " OFFSET " + limit_p_modifier.offset->ToString();
			}
		}
	}
	return result;
}

bool QueryNode::Equals(const QueryNode *other) const {
	if (!other) {
		return false;
	}
	if (this == other) {
		return true;
	}
	if (other->type != this->type) {
		return false;
	}
	if (modifiers.size() != other->modifiers.size()) {
		return false;
	}
	for (idx_t i = 0; i < modifiers.size(); i++) {
		if (!modifiers[i]->Equals(other->modifiers[i].get())) {
			return false;
		}
	}
	// WITH clauses (CTEs)
	if (cte_map.map.size() != other->cte_map.map.size()) {
		return false;
	}
	for (auto &entry : cte_map.map) {
		auto other_entry = other->cte_map.map.find(entry.first);
		if (other_entry == other->cte_map.map.end()) {
			return false;
		}
		if (entry.second->aliases != other_entry->second->aliases) {
			return false;
		}
		if (!entry.second->query->Equals(other_entry->second->query.get())) {
			return false;
		}
	}
	return other->type == type;
}

void QueryNode::CopyProperties(QueryNode &other) const {
	for (auto &modifier : modifiers) {
		other.modifiers.push_back(modifier->Copy());
	}
	for (auto &kv : cte_map.map) {
		auto kv_info = make_unique<CommonTableExpressionInfo>();
		for (auto &al : kv.second->aliases) {
			kv_info->aliases.push_back(al);
		}
		kv_info->query = unique_ptr_cast<SQLStatement, SelectStatement>(kv.second->query->Copy());
		other.cte_map.map[kv.first] = std::move(kv_info);
	}
}

void QueryNode::Serialize(Serializer &main_serializer) const {
	FieldWriter writer(main_serializer);
	writer.WriteField<QueryNodeType>(type);
	writer.WriteSerializableList(modifiers);
	// cte_map
	writer.WriteField<uint32_t>((uint32_t)cte_map.map.size());
	auto &serializer = writer.GetSerializer();
	for (auto &cte : cte_map.map) {
		serializer.WriteString(cte.first);
		serializer.WriteStringVector(cte.second->aliases);
		cte.second->query->Serialize(serializer);
	}
	Serialize(writer);
	writer.Finalize();
}

unique_ptr<QueryNode> QueryNode::Deserialize(Deserializer &main_source) {
	FieldReader reader(main_source);

	auto type = reader.ReadRequired<QueryNodeType>();
	auto modifiers = reader.ReadRequiredSerializableList<ResultModifier>();
	// cte_map
	auto cte_count = reader.ReadRequired<uint32_t>();
	auto &source = reader.GetSource();
	unordered_map<string, unique_ptr<CommonTableExpressionInfo>> new_map;
	for (idx_t i = 0; i < cte_count; i++) {
		auto name = source.Read<string>();
		auto info = make_unique<CommonTableExpressionInfo>();
		source.ReadStringVector(info->aliases);
		info->query = SelectStatement::Deserialize(source);
		new_map[name] = std::move(info);
	}
	unique_ptr<QueryNode> result;
	switch (type) {
	case QueryNodeType::SELECT_NODE:
		result = SelectNode::Deserialize(reader);
		break;
	case QueryNodeType::SET_OPERATION_NODE:
		result = SetOperationNode::Deserialize(reader);
		break;
	case QueryNodeType::RECURSIVE_CTE_NODE:
		result = RecursiveCTENode::Deserialize(reader);
		break;
	default:
		throw SerializationException("Could not deserialize Query Node: unknown type!");
	}
	result->modifiers = std::move(modifiers);
	result->cte_map.map = std::move(new_map);
	reader.Finalize();
	return result;
}

void QueryNode::AddDistinct() {
	// check if we already have a DISTINCT modifier
	for (idx_t modifier_idx = modifiers.size(); modifier_idx > 0; modifier_idx--) {
		auto &modifier = *modifiers[modifier_idx - 1];
		if (modifier.type == ResultModifierType::DISTINCT_MODIFIER) {
			auto &distinct_modifier = (DistinctModifier &)modifier;
			if (distinct_modifier.distinct_on_targets.empty()) {
				// we have a DISTINCT without an ON clause - this distinct does not need to be added
				return;
			}
		} else if (modifier.type == ResultModifierType::LIMIT_MODIFIER ||
		           modifier.type == ResultModifierType::LIMIT_PERCENT_MODIFIER) {
			// we encountered a LIMIT or LIMIT PERCENT - these change the result of DISTINCT, so we do need to push a
			// DISTINCT relation
			break;
		}
	}
	modifiers.push_back(make_unique<DistinctModifier>());
}

} // namespace duckdb




namespace duckdb {

bool ResultModifier::Equals(const ResultModifier *other) const {
	if (!other) {
		return false;
	}
	return type == other->type;
}

void ResultModifier::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteField<ResultModifierType>(type);
	Serialize(writer);
	writer.Finalize();
}

unique_ptr<ResultModifier> ResultModifier::Deserialize(Deserializer &source) {
	FieldReader reader(source);
	auto type = reader.ReadRequired<ResultModifierType>();

	unique_ptr<ResultModifier> result;
	switch (type) {
	case ResultModifierType::LIMIT_MODIFIER:
		result = LimitModifier::Deserialize(reader);
		break;
	case ResultModifierType::ORDER_MODIFIER:
		result = OrderModifier::Deserialize(reader);
		break;
	case ResultModifierType::DISTINCT_MODIFIER:
		result = DistinctModifier::Deserialize(reader);
		break;
	case ResultModifierType::LIMIT_PERCENT_MODIFIER:
		result = LimitPercentModifier::Deserialize(reader);
		break;
	default:
		throw InternalException("Unrecognized ResultModifierType for Deserialization");
	}
	reader.Finalize();
	return result;
}

bool LimitModifier::Equals(const ResultModifier *other_p) const {
	if (!ResultModifier::Equals(other_p)) {
		return false;
	}
	auto &other = (LimitModifier &)*other_p;
	if (!BaseExpression::Equals(limit.get(), other.limit.get())) {
		return false;
	}
	if (!BaseExpression::Equals(offset.get(), other.offset.get())) {
		return false;
	}
	return true;
}

unique_ptr<ResultModifier> LimitModifier::Copy() const {
	auto copy = make_unique<LimitModifier>();
	if (limit) {
		copy->limit = limit->Copy();
	}
	if (offset) {
		copy->offset = offset->Copy();
	}
	return std::move(copy);
}

void LimitModifier::Serialize(FieldWriter &writer) const {
	writer.WriteOptional(limit);
	writer.WriteOptional(offset);
}

unique_ptr<ResultModifier> LimitModifier::Deserialize(FieldReader &reader) {
	auto mod = make_unique<LimitModifier>();
	mod->limit = reader.ReadOptional<ParsedExpression>(nullptr);
	mod->offset = reader.ReadOptional<ParsedExpression>(nullptr);
	return std::move(mod);
}

bool DistinctModifier::Equals(const ResultModifier *other_p) const {
	if (!ResultModifier::Equals(other_p)) {
		return false;
	}
	auto &other = (DistinctModifier &)*other_p;
	if (!ExpressionUtil::ListEquals(distinct_on_targets, other.distinct_on_targets)) {
		return false;
	}
	return true;
}

unique_ptr<ResultModifier> DistinctModifier::Copy() const {
	auto copy = make_unique<DistinctModifier>();
	for (auto &expr : distinct_on_targets) {
		copy->distinct_on_targets.push_back(expr->Copy());
	}
	return std::move(copy);
}

void DistinctModifier::Serialize(FieldWriter &writer) const {
	writer.WriteSerializableList(distinct_on_targets);
}

unique_ptr<ResultModifier> DistinctModifier::Deserialize(FieldReader &reader) {
	auto mod = make_unique<DistinctModifier>();
	mod->distinct_on_targets = reader.ReadRequiredSerializableList<ParsedExpression>();
	return std::move(mod);
}

bool OrderModifier::Equals(const ResultModifier *other_p) const {
	if (!ResultModifier::Equals(other_p)) {
		return false;
	}
	auto &other = (OrderModifier &)*other_p;
	if (orders.size() != other.orders.size()) {
		return false;
	}
	for (idx_t i = 0; i < orders.size(); i++) {
		if (orders[i].type != other.orders[i].type) {
			return false;
		}
		if (!BaseExpression::Equals(orders[i].expression.get(), other.orders[i].expression.get())) {
			return false;
		}
	}
	return true;
}

unique_ptr<ResultModifier> OrderModifier::Copy() const {
	auto copy = make_unique<OrderModifier>();
	for (auto &order : orders) {
		copy->orders.emplace_back(order.type, order.null_order, order.expression->Copy());
	}
	return std::move(copy);
}

string OrderByNode::ToString() const {
	auto str = expression->ToString();
	switch (type) {
	case OrderType::ASCENDING:
		str += " ASC";
		break;
	case OrderType::DESCENDING:
		str += " DESC";
		break;
	default:
		break;
	}

	switch (null_order) {
	case OrderByNullType::NULLS_FIRST:
		str += " NULLS FIRST";
		break;
	case OrderByNullType::NULLS_LAST:
		str += " NULLS LAST";
		break;
	default:
		break;
	}
	return str;
}

void OrderByNode::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteField<OrderType>(type);
	writer.WriteField<OrderByNullType>(null_order);
	writer.WriteSerializable(*expression);
	writer.Finalize();
}

OrderByNode OrderByNode::Deserialize(Deserializer &source) {
	FieldReader reader(source);
	auto type = reader.ReadRequired<OrderType>();
	auto null_order = reader.ReadRequired<OrderByNullType>();
	auto expression = reader.ReadRequiredSerializable<ParsedExpression>();
	reader.Finalize();
	return OrderByNode(type, null_order, std::move(expression));
}

void OrderModifier::Serialize(FieldWriter &writer) const {
	writer.WriteRegularSerializableList(orders);
}

unique_ptr<ResultModifier> OrderModifier::Deserialize(FieldReader &reader) {
	auto mod = make_unique<OrderModifier>();
	mod->orders = reader.ReadRequiredSerializableList<OrderByNode, OrderByNode>();
	return std::move(mod);
}

bool LimitPercentModifier::Equals(const ResultModifier *other_p) const {
	if (!ResultModifier::Equals(other_p)) {
		return false;
	}
	auto &other = (LimitPercentModifier &)*other_p;
	if (!BaseExpression::Equals(limit.get(), other.limit.get())) {
		return false;
	}
	if (!BaseExpression::Equals(offset.get(), other.offset.get())) {
		return false;
	}
	return true;
}

unique_ptr<ResultModifier> LimitPercentModifier::Copy() const {
	auto copy = make_unique<LimitPercentModifier>();
	if (limit) {
		copy->limit = limit->Copy();
	}
	if (offset) {
		copy->offset = offset->Copy();
	}
	return std::move(copy);
}

void LimitPercentModifier::Serialize(FieldWriter &writer) const {
	writer.WriteOptional(limit);
	writer.WriteOptional(offset);
}

unique_ptr<ResultModifier> LimitPercentModifier::Deserialize(FieldReader &reader) {
	auto mod = make_unique<LimitPercentModifier>();
	mod->limit = reader.ReadOptional<ParsedExpression>(nullptr);
	mod->offset = reader.ReadOptional<ParsedExpression>(nullptr);
	return std::move(mod);
}

} // namespace duckdb


namespace duckdb {

AlterStatement::AlterStatement() : SQLStatement(StatementType::ALTER_STATEMENT) {
}

AlterStatement::AlterStatement(const AlterStatement &other) : SQLStatement(other), info(other.info->Copy()) {
}

unique_ptr<SQLStatement> AlterStatement::Copy() const {
	return unique_ptr<AlterStatement>(new AlterStatement(*this));
}

} // namespace duckdb


namespace duckdb {

AttachStatement::AttachStatement() : SQLStatement(StatementType::ATTACH_STATEMENT) {
}

AttachStatement::AttachStatement(const AttachStatement &other) : SQLStatement(other), info(other.info->Copy()) {
}

unique_ptr<SQLStatement> AttachStatement::Copy() const {
	return unique_ptr<AttachStatement>(new AttachStatement(*this));
}

} // namespace duckdb


namespace duckdb {

CallStatement::CallStatement() : SQLStatement(StatementType::CALL_STATEMENT) {
}

CallStatement::CallStatement(const CallStatement &other) : SQLStatement(other), function(other.function->Copy()) {
}

unique_ptr<SQLStatement> CallStatement::Copy() const {
	return unique_ptr<CallStatement>(new CallStatement(*this));
}

} // namespace duckdb


namespace duckdb {

CopyStatement::CopyStatement() : SQLStatement(StatementType::COPY_STATEMENT), info(make_unique<CopyInfo>()) {
}

CopyStatement::CopyStatement(const CopyStatement &other) : SQLStatement(other), info(other.info->Copy()) {
	if (other.select_statement) {
		select_statement = other.select_statement->Copy();
	}
}

string ConvertOptionValueToString(const Value &val) {
	auto type = val.type().id();
	switch (type) {
	case LogicalTypeId::VARCHAR:
		return KeywordHelper::WriteOptionallyQuoted(val.ToString());
	default:
		return val.ToString();
	}
}

string CopyStatement::CopyOptionsToString(const string &format,
                                          const case_insensitive_map_t<vector<Value>> &options) const {
	if (format.empty() && options.empty()) {
		return string();
	}
	string result;

	result += " (";
	if (!format.empty()) {
		result += " FORMAT ";
		result += format;
	}
	for (auto it = options.begin(); it != options.end(); it++) {
		if (!format.empty() || it != options.begin()) {
			result += ", ";
		}
		auto &name = it->first;
		auto &values = it->second;

		result += name + " ";
		if (values.empty()) {
			// Options like HEADER don't need an explicit value
			// just providing the name already sets it to true
		} else if (values.size() == 1) {
			result += ConvertOptionValueToString(values[0]);
		} else {
			result += "( ";
			for (idx_t i = 0; i < values.size(); i++) {
				auto &value = values[i];
				if (i) {
					result += ", ";
				}
				result += KeywordHelper::WriteOptionallyQuoted(value.ToString());
			}
			result += " )";
		}
	}
	result += " )";
	return result;
}

// COPY table-name (c1, c2, ..)
string TablePart(const CopyInfo &info) {
	string result;

	if (!info.catalog.empty()) {
		result += KeywordHelper::WriteOptionallyQuoted(info.catalog) + ".";
	}
	if (!info.schema.empty()) {
		result += KeywordHelper::WriteOptionallyQuoted(info.schema) + ".";
	}
	D_ASSERT(!info.table.empty());
	result += KeywordHelper::WriteOptionallyQuoted(info.table);

	// (c1, c2, ..)
	if (!info.select_list.empty()) {
		result += " (";
		for (idx_t i = 0; i < info.select_list.size(); i++) {
			if (i > 0) {
				result += ", ";
			}
			result += KeywordHelper::WriteOptionallyQuoted(info.select_list[i]);
		}
		result += " )";
	}
	return result;
}

string CopyStatement::ToString() const {
	string result;

	result += "COPY ";
	if (info->is_from) {
		D_ASSERT(!select_statement);
		result += TablePart(*info);
		result += " FROM";
		result += StringUtil::Format(" '%s'", info->file_path);
		result += CopyOptionsToString(info->format, info->options);
	} else {
		if (select_statement) {
			// COPY (select-node) TO ...
			result += "(" + select_statement->ToString() + ")";
		} else {
			result += TablePart(*info);
		}
		result += " TO ";
		result += StringUtil::Format("'%s'", info->file_path);
		result += CopyOptionsToString(info->format, info->options);
	}
	return result;
}

unique_ptr<SQLStatement> CopyStatement::Copy() const {
	return unique_ptr<CopyStatement>(new CopyStatement(*this));
}

} // namespace duckdb


namespace duckdb {

CreateStatement::CreateStatement() : SQLStatement(StatementType::CREATE_STATEMENT) {
}

CreateStatement::CreateStatement(const CreateStatement &other) : SQLStatement(other), info(other.info->Copy()) {
}

unique_ptr<SQLStatement> CreateStatement::Copy() const {
	return unique_ptr<CreateStatement>(new CreateStatement(*this));
}

} // namespace duckdb



namespace duckdb {

DeleteStatement::DeleteStatement() : SQLStatement(StatementType::DELETE_STATEMENT) {
}

DeleteStatement::DeleteStatement(const DeleteStatement &other) : SQLStatement(other), table(other.table->Copy()) {
	if (other.condition) {
		condition = other.condition->Copy();
	}
	for (const auto &using_clause : other.using_clauses) {
		using_clauses.push_back(using_clause->Copy());
	}
	cte_map = other.cte_map.Copy();
}

string DeleteStatement::ToString() const {
	string result;
	result = cte_map.ToString();
	result += "DELETE FROM ";
	result += table->ToString();
	if (!using_clauses.empty()) {
		result += " USING ";
		for (idx_t i = 0; i < using_clauses.size(); i++) {
			if (i > 0) {
				result += ", ";
			}
			result += using_clauses[i]->ToString();
		}
	}
	if (condition) {
		result += " WHERE " + condition->ToString();
	}

	if (!returning_list.empty()) {
		result += " RETURNING ";
		for (idx_t i = 0; i < returning_list.size(); i++) {
			if (i > 0) {
				result += ", ";
			}
			result += returning_list[i]->ToString();
		}
	}
	return result;
}

unique_ptr<SQLStatement> DeleteStatement::Copy() const {
	return unique_ptr<DeleteStatement>(new DeleteStatement(*this));
}

} // namespace duckdb


namespace duckdb {

DropStatement::DropStatement() : SQLStatement(StatementType::DROP_STATEMENT), info(make_unique<DropInfo>()) {
}

DropStatement::DropStatement(const DropStatement &other) : SQLStatement(other), info(other.info->Copy()) {
}

unique_ptr<SQLStatement> DropStatement::Copy() const {
	return unique_ptr<DropStatement>(new DropStatement(*this));
}

} // namespace duckdb


namespace duckdb {

ExecuteStatement::ExecuteStatement() : SQLStatement(StatementType::EXECUTE_STATEMENT) {
}

ExecuteStatement::ExecuteStatement(const ExecuteStatement &other) : SQLStatement(other), name(other.name) {
	for (const auto &value : other.values) {
		values.push_back(value->Copy());
	}
}

unique_ptr<SQLStatement> ExecuteStatement::Copy() const {
	return unique_ptr<ExecuteStatement>(new ExecuteStatement(*this));
}

} // namespace duckdb


namespace duckdb {

ExplainStatement::ExplainStatement(unique_ptr<SQLStatement> stmt, ExplainType explain_type)
    : SQLStatement(StatementType::EXPLAIN_STATEMENT), stmt(std::move(stmt)), explain_type(explain_type) {
}

ExplainStatement::ExplainStatement(const ExplainStatement &other)
    : SQLStatement(other), stmt(other.stmt->Copy()), explain_type(other.explain_type) {
}

unique_ptr<SQLStatement> ExplainStatement::Copy() const {
	return unique_ptr<ExplainStatement>(new ExplainStatement(*this));
}

} // namespace duckdb


namespace duckdb {

ExportStatement::ExportStatement(unique_ptr<CopyInfo> info)
    : SQLStatement(StatementType::EXPORT_STATEMENT), info(std::move(info)) {
}

ExportStatement::ExportStatement(const ExportStatement &other) : SQLStatement(other), info(other.info->Copy()) {
}

unique_ptr<SQLStatement> ExportStatement::Copy() const {
	return unique_ptr<ExportStatement>(new ExportStatement(*this));
}

} // namespace duckdb


namespace duckdb {

ExtensionStatement::ExtensionStatement(ParserExtension extension_p, unique_ptr<ParserExtensionParseData> parse_data_p)
    : SQLStatement(StatementType::EXTENSION_STATEMENT), extension(std::move(extension_p)),
      parse_data(std::move(parse_data_p)) {
}

unique_ptr<SQLStatement> ExtensionStatement::Copy() const {
	return make_unique<ExtensionStatement>(extension, parse_data->Copy());
}

} // namespace duckdb





namespace duckdb {

OnConflictInfo::OnConflictInfo() : action_type(OnConflictAction::THROW) {
}

OnConflictInfo::OnConflictInfo(const OnConflictInfo &other)
    : action_type(other.action_type), indexed_columns(other.indexed_columns) {
	if (other.set_info) {
		set_info = other.set_info->Copy();
	}
}

unique_ptr<OnConflictInfo> OnConflictInfo::Copy() const {
	return unique_ptr<OnConflictInfo>(new OnConflictInfo(*this));
}

InsertStatement::InsertStatement()
    : SQLStatement(StatementType::INSERT_STATEMENT), schema(DEFAULT_SCHEMA), catalog(INVALID_CATALOG) {
}

InsertStatement::InsertStatement(const InsertStatement &other)
    : SQLStatement(other),
      select_statement(unique_ptr_cast<SQLStatement, SelectStatement>(other.select_statement->Copy())),
      columns(other.columns), table(other.table), schema(other.schema), catalog(other.catalog) {
	cte_map = other.cte_map.Copy();
	if (other.on_conflict_info) {
		on_conflict_info = other.on_conflict_info->Copy();
	}
}

string InsertStatement::OnConflictActionToString(OnConflictAction action) {
	switch (action) {
	case OnConflictAction::NOTHING:
		return "DO NOTHING";
	case OnConflictAction::REPLACE:
	case OnConflictAction::UPDATE:
		return "DO UPDATE";
	case OnConflictAction::THROW:
		// Explicitly left empty, for ToString purposes
		return "";
	default: {
		throw NotImplementedException("type not implemented for OnConflictActionType");
	}
	}
}

string InsertStatement::ToString() const {
	bool or_replace_shorthand_set = false;
	string result;

	result = cte_map.ToString();
	result += "INSERT";
	if (on_conflict_info && on_conflict_info->action_type == OnConflictAction::REPLACE) {
		or_replace_shorthand_set = true;
		result += " OR REPLACE";
	}
	result += " INTO ";
	if (!catalog.empty()) {
		result += KeywordHelper::WriteOptionallyQuoted(catalog) + ".";
	}
	if (!schema.empty()) {
		result += KeywordHelper::WriteOptionallyQuoted(schema) + ".";
	}
	result += KeywordHelper::WriteOptionallyQuoted(table);
	// Write the (optional) alias of the insert target
	if (table_ref && !table_ref->alias.empty()) {
		result += StringUtil::Format(" AS %s", KeywordHelper::WriteOptionallyQuoted(table_ref->alias));
	}
	if (!columns.empty()) {
		result += " (";
		for (idx_t i = 0; i < columns.size(); i++) {
			if (i > 0) {
				result += ", ";
			}
			result += KeywordHelper::WriteOptionallyQuoted(columns[i]);
		}
		result += " )";
	}
	result += " ";
	auto values_list = GetValuesList();
	if (values_list) {
		values_list->alias = string();
		result += values_list->ToString();
	} else {
		result += select_statement->ToString();
	}
	if (!or_replace_shorthand_set && on_conflict_info) {
		auto &conflict_info = *on_conflict_info;
		result += " ON CONFLICT ";
		// (optional) conflict target
		if (!conflict_info.indexed_columns.empty()) {
			result += "(";
			auto &columns = conflict_info.indexed_columns;
			for (auto it = columns.begin(); it != columns.end();) {
				result += StringUtil::Lower(*it);
				if (++it != columns.end()) {
					result += ", ";
				}
			}
			result += " )";
		}

		// (optional) where clause
		if (conflict_info.condition) {
			result += " WHERE " + conflict_info.condition->ToString();
		}
		result += " " + OnConflictActionToString(conflict_info.action_type);
		if (conflict_info.set_info) {
			D_ASSERT(conflict_info.action_type == OnConflictAction::UPDATE);
			result += " SET ";
			auto &set_info = *conflict_info.set_info;
			D_ASSERT(set_info.columns.size() == set_info.expressions.size());
			// SET <column_name> = <expression>
			for (idx_t i = 0; i < set_info.columns.size(); i++) {
				auto &column = set_info.columns[i];
				auto &expr = set_info.expressions[i];
				if (i) {
					result += ", ";
				}
				result += StringUtil::Lower(column) + " = " + expr->ToString();
			}
			// (optional) where clause
			if (set_info.condition) {
				result += " WHERE " + set_info.condition->ToString();
			}
		}
	}
	if (!returning_list.empty()) {
		result += " RETURNING ";
		for (idx_t i = 0; i < returning_list.size(); i++) {
			if (i > 0) {
				result += ", ";
			}
			result += returning_list[i]->ToString();
		}
	}
	return result;
}

unique_ptr<SQLStatement> InsertStatement::Copy() const {
	return unique_ptr<InsertStatement>(new InsertStatement(*this));
}

ExpressionListRef *InsertStatement::GetValuesList() const {
	if (select_statement->node->type != QueryNodeType::SELECT_NODE) {
		return nullptr;
	}
	auto &node = (SelectNode &)*select_statement->node;
	if (node.where_clause || node.qualify || node.having) {
		return nullptr;
	}
	if (!node.cte_map.map.empty()) {
		return nullptr;
	}
	if (!node.groups.grouping_sets.empty()) {
		return nullptr;
	}
	if (node.aggregate_handling != AggregateHandling::STANDARD_HANDLING) {
		return nullptr;
	}
	if (node.select_list.size() != 1 || node.select_list[0]->type != ExpressionType::STAR) {
		return nullptr;
	}
	if (!node.from_table || node.from_table->type != TableReferenceType::EXPRESSION_LIST) {
		return nullptr;
	}
	return (ExpressionListRef *)node.from_table.get();
}

} // namespace duckdb


namespace duckdb {

LoadStatement::LoadStatement() : SQLStatement(StatementType::LOAD_STATEMENT) {
}

LoadStatement::LoadStatement(const LoadStatement &other) : SQLStatement(other), info(other.info->Copy()) {
}

unique_ptr<SQLStatement> LoadStatement::Copy() const {
	return unique_ptr<LoadStatement>(new LoadStatement(*this));
}

} // namespace duckdb


namespace duckdb {

PragmaStatement::PragmaStatement() : SQLStatement(StatementType::PRAGMA_STATEMENT), info(make_unique<PragmaInfo>()) {
}

PragmaStatement::PragmaStatement(const PragmaStatement &other) : SQLStatement(other), info(other.info->Copy()) {
}

unique_ptr<SQLStatement> PragmaStatement::Copy() const {
	return unique_ptr<PragmaStatement>(new PragmaStatement(*this));
}

} // namespace duckdb


namespace duckdb {

PrepareStatement::PrepareStatement() : SQLStatement(StatementType::PREPARE_STATEMENT), statement(nullptr), name("") {
}

PrepareStatement::PrepareStatement(const PrepareStatement &other)
    : SQLStatement(other), statement(other.statement->Copy()), name(other.name) {
}

unique_ptr<SQLStatement> PrepareStatement::Copy() const {
	return unique_ptr<PrepareStatement>(new PrepareStatement(*this));
}

} // namespace duckdb


namespace duckdb {

RelationStatement::RelationStatement(shared_ptr<Relation> relation)
    : SQLStatement(StatementType::RELATION_STATEMENT), relation(std::move(relation)) {
}

unique_ptr<SQLStatement> RelationStatement::Copy() const {
	return unique_ptr<RelationStatement>(new RelationStatement(*this));
}

} // namespace duckdb




namespace duckdb {

SelectStatement::SelectStatement(const SelectStatement &other) : SQLStatement(other), node(other.node->Copy()) {
}

unique_ptr<SQLStatement> SelectStatement::Copy() const {
	return unique_ptr<SelectStatement>(new SelectStatement(*this));
}

void SelectStatement::Serialize(Serializer &serializer) const {
	node->Serialize(serializer);
}

unique_ptr<SelectStatement> SelectStatement::Deserialize(Deserializer &source) {
	auto result = make_unique<SelectStatement>();
	result->node = QueryNode::Deserialize(source);
	return result;
}

bool SelectStatement::Equals(const SQLStatement *other_p) const {
	if (type != other_p->type) {
		return false;
	}
	auto other = (SelectStatement *)other_p;
	return node->Equals(other->node.get());
}

string SelectStatement::ToString() const {
	return node->ToString();
}

} // namespace duckdb


namespace duckdb {

SetStatement::SetStatement(std::string name_p, SetScope scope_p, SetType type_p)
    : SQLStatement(StatementType::SET_STATEMENT), name(std::move(name_p)), scope(scope_p), set_type(type_p) {
}

unique_ptr<SQLStatement> SetStatement::Copy() const {
	return unique_ptr<SetStatement>(new SetStatement(*this));
}

// Set Variable

SetVariableStatement::SetVariableStatement(std::string name_p, Value value_p, SetScope scope_p)
    : SetStatement(std::move(name_p), scope_p, SetType::SET), value(std::move(value_p)) {
}

unique_ptr<SQLStatement> SetVariableStatement::Copy() const {
	return unique_ptr<SetVariableStatement>(new SetVariableStatement(*this));
}

// Reset Variable

ResetVariableStatement::ResetVariableStatement(std::string name_p, SetScope scope_p)
    : SetStatement(std::move(name_p), scope_p, SetType::RESET) {
}

} // namespace duckdb


namespace duckdb {

ShowStatement::ShowStatement() : SQLStatement(StatementType::SHOW_STATEMENT), info(make_unique<ShowSelectInfo>()) {
}

ShowStatement::ShowStatement(const ShowStatement &other) : SQLStatement(other), info(other.info->Copy()) {
}

unique_ptr<SQLStatement> ShowStatement::Copy() const {
	return unique_ptr<ShowStatement>(new ShowStatement(*this));
}

} // namespace duckdb


namespace duckdb {

TransactionStatement::TransactionStatement(TransactionType type)
    : SQLStatement(StatementType::TRANSACTION_STATEMENT), info(make_unique<TransactionInfo>(type)) {
}

TransactionStatement::TransactionStatement(const TransactionStatement &other)
    : SQLStatement(other), info(make_unique<TransactionInfo>(other.info->type)) {
}

unique_ptr<SQLStatement> TransactionStatement::Copy() const {
	return unique_ptr<TransactionStatement>(new TransactionStatement(*this));
}

} // namespace duckdb



namespace duckdb {

UpdateSetInfo::UpdateSetInfo() {
}

UpdateSetInfo::UpdateSetInfo(const UpdateSetInfo &other) : columns(other.columns) {
	if (other.condition) {
		condition = other.condition->Copy();
	}
	for (auto &expr : other.expressions) {
		expressions.emplace_back(expr->Copy());
	}
}

unique_ptr<UpdateSetInfo> UpdateSetInfo::Copy() const {
	return unique_ptr<UpdateSetInfo>(new UpdateSetInfo(*this));
}

UpdateStatement::UpdateStatement() : SQLStatement(StatementType::UPDATE_STATEMENT) {
}

UpdateStatement::UpdateStatement(const UpdateStatement &other)
    : SQLStatement(other), table(other.table->Copy()), set_info(other.set_info->Copy()) {
	if (other.from_table) {
		from_table = other.from_table->Copy();
	}
	cte_map = other.cte_map.Copy();
}

string UpdateStatement::ToString() const {
	D_ASSERT(set_info);
	auto &condition = set_info->condition;
	auto &columns = set_info->columns;
	auto &expressions = set_info->expressions;

	string result;
	result = cte_map.ToString();
	result += "UPDATE ";
	result += table->ToString();
	result += " SET ";
	D_ASSERT(columns.size() == expressions.size());
	for (idx_t i = 0; i < columns.size(); i++) {
		if (i > 0) {
			result += ", ";
		}
		result += KeywordHelper::WriteOptionallyQuoted(columns[i]);
		result += " = ";
		result += expressions[i]->ToString();
	}
	if (from_table) {
		result += " FROM " + from_table->ToString();
	}
	if (condition) {
		result += " WHERE " + condition->ToString();
	}
	if (!returning_list.empty()) {
		result += " RETURNING ";
		for (idx_t i = 0; i < returning_list.size(); i++) {
			if (i > 0) {
				result += ", ";
			}
			result += returning_list[i]->ToString();
		}
	}
	return result;
}

unique_ptr<SQLStatement> UpdateStatement::Copy() const {
	return unique_ptr<UpdateStatement>(new UpdateStatement(*this));
}

} // namespace duckdb


namespace duckdb {

VacuumStatement::VacuumStatement(const VacuumOptions &options)
    : SQLStatement(StatementType::VACUUM_STATEMENT), info(make_unique<VacuumInfo>(options)) {
}

VacuumStatement::VacuumStatement(const VacuumStatement &other) : SQLStatement(other), info(other.info->Copy()) {
}

unique_ptr<SQLStatement> VacuumStatement::Copy() const {
	return unique_ptr<VacuumStatement>(new VacuumStatement(*this));
}

} // namespace duckdb





namespace duckdb {

string BaseTableRef::ToString() const {
	string result;
	result += catalog_name.empty() ? "" : (KeywordHelper::WriteOptionallyQuoted(catalog_name) + ".");
	result += schema_name.empty() ? "" : (KeywordHelper::WriteOptionallyQuoted(schema_name) + ".");
	result += KeywordHelper::WriteOptionallyQuoted(table_name);
	return BaseToString(result, column_name_alias);
}

bool BaseTableRef::Equals(const TableRef *other_p) const {
	if (!TableRef::Equals(other_p)) {
		return false;
	}
	auto other = (BaseTableRef *)other_p;
	return other->catalog_name == catalog_name && other->schema_name == schema_name &&
	       other->table_name == table_name && column_name_alias == other->column_name_alias;
}

void BaseTableRef::Serialize(FieldWriter &writer) const {
	writer.WriteString(schema_name);
	writer.WriteString(table_name);
	writer.WriteList<string>(column_name_alias);
	writer.WriteString(catalog_name);
}

unique_ptr<TableRef> BaseTableRef::Deserialize(FieldReader &reader) {
	auto result = make_unique<BaseTableRef>();

	result->schema_name = reader.ReadRequired<string>();
	result->table_name = reader.ReadRequired<string>();
	result->column_name_alias = reader.ReadRequiredList<string>();
	result->catalog_name = reader.ReadField<string>(INVALID_CATALOG);

	return std::move(result);
}

unique_ptr<TableRef> BaseTableRef::Copy() {
	auto copy = make_unique<BaseTableRef>();

	copy->catalog_name = catalog_name;
	copy->schema_name = schema_name;
	copy->table_name = table_name;
	copy->column_name_alias = column_name_alias;
	CopyProperties(*copy);

	return std::move(copy);
}
} // namespace duckdb




namespace duckdb {

string EmptyTableRef::ToString() const {
	return "";
}

bool EmptyTableRef::Equals(const TableRef *other) const {
	return TableRef::Equals(other);
}

unique_ptr<TableRef> EmptyTableRef::Copy() {
	return make_unique<EmptyTableRef>();
}

void EmptyTableRef::Serialize(FieldWriter &writer) const {
}

unique_ptr<TableRef> EmptyTableRef::Deserialize(FieldReader &reader) {
	return make_unique<EmptyTableRef>();
}

} // namespace duckdb




namespace duckdb {

string ExpressionListRef::ToString() const {
	D_ASSERT(!values.empty());
	string result = "(VALUES ";
	for (idx_t row_idx = 0; row_idx < values.size(); row_idx++) {
		if (row_idx > 0) {
			result += ", ";
		}
		auto &row = values[row_idx];
		result += "(";
		for (idx_t col_idx = 0; col_idx < row.size(); col_idx++) {
			if (col_idx > 0) {
				result += ", ";
			}
			result += row[col_idx]->ToString();
		}
		result += ")";
	}
	result += ")";
	return BaseToString(result, expected_names);
}

bool ExpressionListRef::Equals(const TableRef *other_p) const {
	if (!TableRef::Equals(other_p)) {
		return false;
	}
	auto other = (ExpressionListRef *)other_p;
	if (values.size() != other->values.size()) {
		return false;
	}
	for (idx_t i = 0; i < values.size(); i++) {
		if (values[i].size() != other->values[i].size()) {
			return false;
		}
		for (idx_t j = 0; j < values[i].size(); j++) {
			if (!values[i][j]->Equals(other->values[i][j].get())) {
				return false;
			}
		}
	}
	return true;
}

unique_ptr<TableRef> ExpressionListRef::Copy() {
	// value list
	auto result = make_unique<ExpressionListRef>();
	for (auto &val_list : values) {
		vector<unique_ptr<ParsedExpression>> new_val_list;
		new_val_list.reserve(val_list.size());
		for (auto &val : val_list) {
			new_val_list.push_back(val->Copy());
		}
		result->values.push_back(std::move(new_val_list));
	}
	result->expected_names = expected_names;
	result->expected_types = expected_types;
	CopyProperties(*result);
	return std::move(result);
}

void ExpressionListRef::Serialize(FieldWriter &writer) const {
	writer.WriteList<string>(expected_names);
	writer.WriteRegularSerializableList<LogicalType>(expected_types);
	auto &serializer = writer.GetSerializer();
	writer.WriteField<uint32_t>(values.size());
	for (idx_t i = 0; i < values.size(); i++) {
		serializer.WriteList(values[i]);
	}
}

unique_ptr<TableRef> ExpressionListRef::Deserialize(FieldReader &reader) {
	auto result = make_unique<ExpressionListRef>();
	// value list
	result->expected_names = reader.ReadRequiredList<string>();
	result->expected_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
	idx_t value_list_size = reader.ReadRequired<uint32_t>();
	auto &source = reader.GetSource();
	for (idx_t i = 0; i < value_list_size; i++) {
		vector<unique_ptr<ParsedExpression>> value_list;
		source.ReadList<ParsedExpression>(value_list);
		result->values.push_back(std::move(value_list));
	}
	return std::move(result);
}

} // namespace duckdb





namespace duckdb {

string JoinRef::ToString() const {
	string result;
	result = left->ToString() + " ";
	switch (ref_type) {
	case JoinRefType::REGULAR:
		result += JoinTypeToString(type) + " JOIN ";
		break;
	case JoinRefType::NATURAL:
		result += "NATURAL ";
		result += JoinTypeToString(type) + " JOIN ";
		break;
	case JoinRefType::CROSS:
		result += ", ";
		break;
	case JoinRefType::POSITIONAL:
		result += "POSITIONAL JOIN ";
		break;
	}
	result += right->ToString();
	if (condition) {
		D_ASSERT(using_columns.empty());
		result += " ON (";
		result += condition->ToString();
		result += ")";
	} else if (!using_columns.empty()) {
		result += " USING (";
		for (idx_t i = 0; i < using_columns.size(); i++) {
			if (i > 0) {
				result += ", ";
			}
			result += using_columns[i];
		}
		result += ")";
	}
	return result;
}

bool JoinRef::Equals(const TableRef *other_p) const {
	if (!TableRef::Equals(other_p)) {
		return false;
	}
	auto other = (JoinRef *)other_p;
	if (using_columns.size() != other->using_columns.size()) {
		return false;
	}
	for (idx_t i = 0; i < using_columns.size(); i++) {
		if (using_columns[i] != other->using_columns[i]) {
			return false;
		}
	}
	return left->Equals(other->left.get()) && right->Equals(other->right.get()) &&
	       BaseExpression::Equals(condition.get(), other->condition.get()) && type == other->type;
}

unique_ptr<TableRef> JoinRef::Copy() {
	auto copy = make_unique<JoinRef>(ref_type);
	copy->left = left->Copy();
	copy->right = right->Copy();
	if (condition) {
		copy->condition = condition->Copy();
	}
	copy->type = type;
	copy->ref_type = ref_type;
	copy->alias = alias;
	copy->using_columns = using_columns;
	return std::move(copy);
}

void JoinRef::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*left);
	writer.WriteSerializable(*right);
	writer.WriteOptional(condition);
	writer.WriteField<JoinType>(type);
	writer.WriteField<JoinRefType>(ref_type);
	writer.WriteList<string>(using_columns);
}

unique_ptr<TableRef> JoinRef::Deserialize(FieldReader &reader) {
	auto result = make_unique<JoinRef>(JoinRefType::REGULAR);
	result->left = reader.ReadRequiredSerializable<TableRef>();
	result->right = reader.ReadRequiredSerializable<TableRef>();
	result->condition = reader.ReadOptional<ParsedExpression>(nullptr);
	result->type = reader.ReadRequired<JoinType>();
	result->ref_type = reader.ReadRequired<JoinRefType>();
	result->using_columns = reader.ReadRequiredList<string>();
	return std::move(result);
}

} // namespace duckdb





namespace duckdb {

string SubqueryRef::ToString() const {
	string result = "(" + subquery->ToString() + ")";
	return BaseToString(result, column_name_alias);
}

SubqueryRef::SubqueryRef(unique_ptr<SelectStatement> subquery_p, string alias_p)
    : TableRef(TableReferenceType::SUBQUERY), subquery(std::move(subquery_p)) {
	this->alias = std::move(alias_p);
}

bool SubqueryRef::Equals(const TableRef *other_p) const {
	if (!TableRef::Equals(other_p)) {
		return false;
	}
	auto other = (SubqueryRef *)other_p;
	return subquery->Equals(other->subquery.get());
}

unique_ptr<TableRef> SubqueryRef::Copy() {
	auto copy = make_unique<SubqueryRef>(unique_ptr_cast<SQLStatement, SelectStatement>(subquery->Copy()), alias);
	copy->column_name_alias = column_name_alias;
	CopyProperties(*copy);
	return std::move(copy);
}

void SubqueryRef::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*subquery);
	writer.WriteList<string>(column_name_alias);
}

unique_ptr<TableRef> SubqueryRef::Deserialize(FieldReader &reader) {
	auto subquery = reader.ReadRequiredSerializable<SelectStatement>();
	auto result = make_unique<SubqueryRef>(std::move(subquery));
	result->column_name_alias = reader.ReadRequiredList<string>();
	return std::move(result);
}

} // namespace duckdb




namespace duckdb {

TableFunctionRef::TableFunctionRef() : TableRef(TableReferenceType::TABLE_FUNCTION) {
}

string TableFunctionRef::ToString() const {
	return BaseToString(function->ToString(), column_name_alias);
}

bool TableFunctionRef::Equals(const TableRef *other_p) const {
	if (!TableRef::Equals(other_p)) {
		return false;
	}
	auto other = (TableFunctionRef *)other_p;
	return function->Equals(other->function.get());
}

void TableFunctionRef::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*function);
	writer.WriteString(alias);
	writer.WriteList<string>(column_name_alias);
}

unique_ptr<TableRef> TableFunctionRef::Deserialize(FieldReader &reader) {
	auto result = make_unique<TableFunctionRef>();
	result->function = reader.ReadRequiredSerializable<ParsedExpression>();
	result->alias = reader.ReadRequired<string>();
	result->column_name_alias = reader.ReadRequiredList<string>();
	return std::move(result);
}

unique_ptr<TableRef> TableFunctionRef::Copy() {
	auto copy = make_unique<TableFunctionRef>();

	copy->function = function->Copy();
	copy->column_name_alias = column_name_alias;
	CopyProperties(*copy);

	return std::move(copy);
}

} // namespace duckdb







namespace duckdb {

string TableRef::BaseToString(string result) const {
	vector<string> column_name_alias;
	return BaseToString(std::move(result), column_name_alias);
}

string TableRef::BaseToString(string result, const vector<string> &column_name_alias) const {
	if (!alias.empty()) {
		result += " AS " + KeywordHelper::WriteOptionallyQuoted(alias);
	}
	if (!column_name_alias.empty()) {
		D_ASSERT(!alias.empty());
		result += "(";
		for (idx_t i = 0; i < column_name_alias.size(); i++) {
			if (i > 0) {
				result += ", ";
			}
			result += KeywordHelper::WriteOptionallyQuoted(column_name_alias[i]);
		}
		result += ")";
	}
	if (sample) {
		result += " TABLESAMPLE " + SampleMethodToString(sample->method);
		result += "(" + sample->sample_size.ToString() + " " + string(sample->is_percentage ? "PERCENT" : "ROWS") + ")";
		if (sample->seed >= 0) {
			result += "REPEATABLE (" + to_string(sample->seed) + ")";
		}
	}

	return result;
}

bool TableRef::Equals(const TableRef *other) const {
	return other && type == other->type && alias == other->alias &&
	       SampleOptions::Equals(sample.get(), other->sample.get());
}

void TableRef::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteField<TableReferenceType>(type);
	writer.WriteString(alias);
	writer.WriteOptional(sample);
	Serialize(writer);
	writer.Finalize();
}

unique_ptr<TableRef> TableRef::Deserialize(Deserializer &source) {
	FieldReader reader(source);

	auto type = reader.ReadRequired<TableReferenceType>();
	auto alias = reader.ReadRequired<string>();
	auto sample = reader.ReadOptional<SampleOptions>(nullptr);
	unique_ptr<TableRef> result;
	switch (type) {
	case TableReferenceType::BASE_TABLE:
		result = BaseTableRef::Deserialize(reader);
		break;
	case TableReferenceType::JOIN:
		result = JoinRef::Deserialize(reader);
		break;
	case TableReferenceType::SUBQUERY:
		result = SubqueryRef::Deserialize(reader);
		break;
	case TableReferenceType::TABLE_FUNCTION:
		result = TableFunctionRef::Deserialize(reader);
		break;
	case TableReferenceType::EMPTY:
		result = EmptyTableRef::Deserialize(reader);
		break;
	case TableReferenceType::EXPRESSION_LIST:
		result = ExpressionListRef::Deserialize(reader);
		break;
	case TableReferenceType::CTE:
	case TableReferenceType::INVALID:
		throw InternalException("Unsupported type for TableRef::Deserialize");
	}
	reader.Finalize();

	result->alias = alias;
	result->sample = std::move(sample);
	return result;
}

void TableRef::CopyProperties(TableRef &target) const {
	D_ASSERT(type == target.type);
	target.alias = alias;
	target.query_location = query_location;
	target.sample = sample ? sample->Copy() : nullptr;
}

void TableRef::Print() {
	Printer::Print(ToString());
}

} // namespace duckdb





namespace duckdb {

static void ParseSchemaTableNameFK(duckdb_libpgquery::PGRangeVar *input, ForeignKeyInfo &fk_info) {
	if (input->catalogname) {
		throw ParserException("FOREIGN KEY constraints cannot be defined cross-database");
	}
	if (input->schemaname) {
		fk_info.schema = input->schemaname;
	} else {
		fk_info.schema = "";
	};
	fk_info.table = input->relname;
}

unique_ptr<Constraint> Transformer::TransformConstraint(duckdb_libpgquery::PGListCell *cell) {
	auto constraint = reinterpret_cast<duckdb_libpgquery::PGConstraint *>(cell->data.ptr_value);
	switch (constraint->contype) {
	case duckdb_libpgquery::PG_CONSTR_UNIQUE:
	case duckdb_libpgquery::PG_CONSTR_PRIMARY: {
		bool is_primary_key = constraint->contype == duckdb_libpgquery::PG_CONSTR_PRIMARY;
		vector<string> columns;
		for (auto kc = constraint->keys->head; kc; kc = kc->next) {
			columns.emplace_back(reinterpret_cast<duckdb_libpgquery::PGValue *>(kc->data.ptr_value)->val.str);
		}
		return make_unique<UniqueConstraint>(columns, is_primary_key);
	}
	case duckdb_libpgquery::PG_CONSTR_CHECK: {
		auto expression = TransformExpression(constraint->raw_expr);
		if (expression->HasSubquery()) {
			throw ParserException("subqueries prohibited in CHECK constraints");
		}
		return make_unique<CheckConstraint>(TransformExpression(constraint->raw_expr));
	}
	case duckdb_libpgquery::PG_CONSTR_FOREIGN: {
		ForeignKeyInfo fk_info;
		fk_info.type = ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE;
		ParseSchemaTableNameFK(constraint->pktable, fk_info);
		vector<string> pk_columns, fk_columns;
		for (auto kc = constraint->fk_attrs->head; kc; kc = kc->next) {
			fk_columns.emplace_back(reinterpret_cast<duckdb_libpgquery::PGValue *>(kc->data.ptr_value)->val.str);
		}
		if (constraint->pk_attrs) {
			for (auto kc = constraint->pk_attrs->head; kc; kc = kc->next) {
				pk_columns.emplace_back(reinterpret_cast<duckdb_libpgquery::PGValue *>(kc->data.ptr_value)->val.str);
			}
		}
		if (!pk_columns.empty() && pk_columns.size() != fk_columns.size()) {
			throw ParserException("The number of referencing and referenced columns for foreign keys must be the same");
		}
		if (fk_columns.empty()) {
			throw ParserException("The set of referencing and referenced columns for foreign keys must be not empty");
		}
		return make_unique<ForeignKeyConstraint>(pk_columns, fk_columns, std::move(fk_info));
	}
	default:
		throw NotImplementedException("Constraint type not handled yet!");
	}
}

unique_ptr<Constraint> Transformer::TransformConstraint(duckdb_libpgquery::PGListCell *cell, ColumnDefinition &column,
                                                        idx_t index) {
	auto constraint = reinterpret_cast<duckdb_libpgquery::PGConstraint *>(cell->data.ptr_value);
	D_ASSERT(constraint);
	switch (constraint->contype) {
	case duckdb_libpgquery::PG_CONSTR_NOTNULL:
		return make_unique<NotNullConstraint>(LogicalIndex(index));
	case duckdb_libpgquery::PG_CONSTR_CHECK:
		return TransformConstraint(cell);
	case duckdb_libpgquery::PG_CONSTR_PRIMARY:
		return make_unique<UniqueConstraint>(LogicalIndex(index), true);
	case duckdb_libpgquery::PG_CONSTR_UNIQUE:
		return make_unique<UniqueConstraint>(LogicalIndex(index), false);
	case duckdb_libpgquery::PG_CONSTR_NULL:
		return nullptr;
	case duckdb_libpgquery::PG_CONSTR_GENERATED_VIRTUAL: {
		if (column.DefaultValue()) {
			throw InvalidInputException("DEFAULT constraint on GENERATED column \"%s\" is not allowed", column.Name());
		}
		column.SetGeneratedExpression(TransformExpression(constraint->raw_expr));
		return nullptr;
	}
	case duckdb_libpgquery::PG_CONSTR_GENERATED_STORED:
		throw InvalidInputException("Can not create a STORED generated column!");
	case duckdb_libpgquery::PG_CONSTR_DEFAULT:
		column.SetDefaultValue(TransformExpression(constraint->raw_expr));
		return nullptr;
	case duckdb_libpgquery::PG_CONSTR_COMPRESSION:
		column.SetCompressionType(CompressionTypeFromString(constraint->compression_name));
		if (column.CompressionType() == CompressionType::COMPRESSION_AUTO) {
			throw ParserException("Unrecognized option for column compression, expected none, uncompressed, rle, "
			                      "dictionary, pfor, bitpacking or fsst");
		}
		return nullptr;
	case duckdb_libpgquery::PG_CONSTR_FOREIGN: {
		ForeignKeyInfo fk_info;
		fk_info.type = ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE;
		ParseSchemaTableNameFK(constraint->pktable, fk_info);

		vector<string> pk_columns, fk_columns;
		fk_columns.emplace_back(column.Name().c_str());
		if (constraint->pk_attrs) {
			for (auto kc = constraint->pk_attrs->head; kc; kc = kc->next) {
				pk_columns.emplace_back(reinterpret_cast<duckdb_libpgquery::PGValue *>(kc->data.ptr_value)->val.str);
			}
		}
		if (pk_columns.size() != fk_columns.size()) {
			throw ParserException("The number of referencing and referenced columns for foreign keys must be the same");
		}
		return make_unique<ForeignKeyConstraint>(pk_columns, fk_columns, std::move(fk_info));
	}
	default:
		throw NotImplementedException("Constraint not implemented!");
	}
}

} // namespace duckdb






namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformArrayAccess(duckdb_libpgquery::PGAIndirection *indirection_node) {
	// transform the source expression
	unique_ptr<ParsedExpression> result;
	result = TransformExpression(indirection_node->arg);

	// now go over the indices
	// note that a single indirection node can contain multiple indices
	// this happens for e.g. more complex accesses (e.g. (foo).field1[42])
	idx_t list_size = 0;
	for (auto node = indirection_node->indirection->head; node != nullptr; node = node->next) {
		auto target = reinterpret_cast<duckdb_libpgquery::PGNode *>(node->data.ptr_value);
		D_ASSERT(target);

		switch (target->type) {
		case duckdb_libpgquery::T_PGAIndices: {
			// index access (either slice or extract)
			auto index = (duckdb_libpgquery::PGAIndices *)target;
			vector<unique_ptr<ParsedExpression>> children;
			children.push_back(std::move(result));
			if (index->is_slice) {
				// slice
				children.push_back(!index->lidx ? make_unique<ConstantExpression>(Value())
				                                : TransformExpression(index->lidx));
				children.push_back(!index->uidx ? make_unique<ConstantExpression>(Value())
				                                : TransformExpression(index->uidx));
				result = make_unique<OperatorExpression>(ExpressionType::ARRAY_SLICE, std::move(children));
			} else {
				// array access
				D_ASSERT(!index->lidx);
				D_ASSERT(index->uidx);
				children.push_back(TransformExpression(index->uidx));
				result = make_unique<OperatorExpression>(ExpressionType::ARRAY_EXTRACT, std::move(children));
			}
			break;
		}
		case duckdb_libpgquery::T_PGString: {
			auto val = (duckdb_libpgquery::PGValue *)target;
			vector<unique_ptr<ParsedExpression>> children;
			children.push_back(std::move(result));
			children.push_back(TransformValue(*val));
			result = make_unique<OperatorExpression>(ExpressionType::STRUCT_EXTRACT, std::move(children));
			break;
		}
		default:
			throw NotImplementedException("Unimplemented subscript type");
		}
		list_size++;
		StackCheck(list_size);
	}
	return result;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformBoolExpr(duckdb_libpgquery::PGBoolExpr *root) {
	unique_ptr<ParsedExpression> result;
	for (auto node = root->args->head; node != nullptr; node = node->next) {
		auto next = TransformExpression(reinterpret_cast<duckdb_libpgquery::PGNode *>(node->data.ptr_value));

		switch (root->boolop) {
		case duckdb_libpgquery::PG_AND_EXPR: {
			if (!result) {
				result = std::move(next);
			} else {
				result = make_unique<ConjunctionExpression>(ExpressionType::CONJUNCTION_AND, std::move(result),
				                                            std::move(next));
			}
			break;
		}
		case duckdb_libpgquery::PG_OR_EXPR: {
			if (!result) {
				result = std::move(next);
			} else {
				result = make_unique<ConjunctionExpression>(ExpressionType::CONJUNCTION_OR, std::move(result),
				                                            std::move(next));
			}
			break;
		}
		case duckdb_libpgquery::PG_NOT_EXPR: {
			if (next->type == ExpressionType::COMPARE_IN) {
				// convert COMPARE_IN to COMPARE_NOT_IN
				next->type = ExpressionType::COMPARE_NOT_IN;
				result = std::move(next);
			} else if (next->type >= ExpressionType::COMPARE_EQUAL &&
			           next->type <= ExpressionType::COMPARE_GREATERTHANOREQUALTO) {
				// NOT on a comparison: we can negate the comparison
				// e.g. NOT(x > y) is equivalent to x <= y
				next->type = NegateComparisionExpression(next->type);
				result = std::move(next);
			} else {
				result = make_unique<OperatorExpression>(ExpressionType::OPERATOR_NOT, std::move(next));
			}
			break;
		}
		}
	}
	return result;
}

} // namespace duckdb





namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformCase(duckdb_libpgquery::PGCaseExpr *root) {
	D_ASSERT(root);

	auto case_node = make_unique<CaseExpression>();
	for (auto cell = root->args->head; cell != nullptr; cell = cell->next) {
		CaseCheck case_check;

		auto w = reinterpret_cast<duckdb_libpgquery::PGCaseWhen *>(cell->data.ptr_value);
		auto test_raw = TransformExpression(reinterpret_cast<duckdb_libpgquery::PGNode *>(w->expr));
		unique_ptr<ParsedExpression> test;
		auto arg = TransformExpression(reinterpret_cast<duckdb_libpgquery::PGNode *>(root->arg));
		if (arg) {
			case_check.when_expr =
			    make_unique<ComparisonExpression>(ExpressionType::COMPARE_EQUAL, std::move(arg), std::move(test_raw));
		} else {
			case_check.when_expr = std::move(test_raw);
		}
		case_check.then_expr = TransformExpression(reinterpret_cast<duckdb_libpgquery::PGNode *>(w->result));
		case_node->case_checks.push_back(std::move(case_check));
	}

	if (root->defresult) {
		case_node->else_expr = TransformExpression(reinterpret_cast<duckdb_libpgquery::PGNode *>(root->defresult));
	} else {
		case_node->else_expr = make_unique<ConstantExpression>(Value(LogicalType::SQLNULL));
	}
	return std::move(case_node);
}

} // namespace duckdb






namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformTypeCast(duckdb_libpgquery::PGTypeCast *root) {
	D_ASSERT(root);

	// get the type to cast to
	auto type_name = root->typeName;
	LogicalType target_type = TransformTypeName(type_name);

	// check for a constant BLOB value, then return ConstantExpression with BLOB
	if (!root->tryCast && target_type == LogicalType::BLOB && root->arg->type == duckdb_libpgquery::T_PGAConst) {
		auto c = reinterpret_cast<duckdb_libpgquery::PGAConst *>(root->arg);
		if (c->val.type == duckdb_libpgquery::T_PGString) {
			return make_unique<ConstantExpression>(Value::BLOB(string(c->val.val.str)));
		}
	}
	// transform the expression node
	auto expression = TransformExpression(root->arg);
	bool try_cast = root->tryCast;

	// now create a cast operation
	return make_unique<CastExpression>(target_type, std::move(expression), try_cast);
}

} // namespace duckdb



namespace duckdb {

// COALESCE(a,b,c) returns the first argument that is NOT NULL, so
// rewrite into CASE(a IS NOT NULL, a, CASE(b IS NOT NULL, b, c))
unique_ptr<ParsedExpression> Transformer::TransformCoalesce(duckdb_libpgquery::PGAExpr *root) {
	D_ASSERT(root);

	auto coalesce_args = reinterpret_cast<duckdb_libpgquery::PGList *>(root->lexpr);
	D_ASSERT(coalesce_args->length > 0); // parser ensures this already

	auto coalesce_op = make_unique<OperatorExpression>(ExpressionType::OPERATOR_COALESCE);
	for (auto cell = coalesce_args->head; cell; cell = cell->next) {
		// get the value of the COALESCE
		auto value_expr = TransformExpression(reinterpret_cast<duckdb_libpgquery::PGNode *>(cell->data.ptr_value));
		coalesce_op->children.push_back(std::move(value_expr));
	}
	return std::move(coalesce_op);
}

} // namespace duckdb





namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformStarExpression(duckdb_libpgquery::PGNode *node) {
	auto star = (duckdb_libpgquery::PGAStar *)node;
	auto result = make_unique<StarExpression>(star->relation ? star->relation : string());
	if (star->except_list) {
		for (auto head = star->except_list->head; head; head = head->next) {
			auto value = (duckdb_libpgquery::PGValue *)head->data.ptr_value;
			D_ASSERT(value->type == duckdb_libpgquery::T_PGString);
			string exclude_entry = value->val.str;
			if (result->exclude_list.find(exclude_entry) != result->exclude_list.end()) {
				throw ParserException("Duplicate entry \"%s\" in EXCLUDE list", exclude_entry);
			}
			result->exclude_list.insert(std::move(exclude_entry));
		}
	}
	if (star->replace_list) {
		for (auto head = star->replace_list->head; head; head = head->next) {
			auto list = (duckdb_libpgquery::PGList *)head->data.ptr_value;
			D_ASSERT(list->length == 2);
			auto replace_expression = TransformExpression((duckdb_libpgquery::PGNode *)list->head->data.ptr_value);
			auto value = (duckdb_libpgquery::PGValue *)list->tail->data.ptr_value;
			D_ASSERT(value->type == duckdb_libpgquery::T_PGString);
			string exclude_entry = value->val.str;
			if (result->replace_list.find(exclude_entry) != result->replace_list.end()) {
				throw ParserException("Duplicate entry \"%s\" in REPLACE list", exclude_entry);
			}
			if (result->exclude_list.find(exclude_entry) != result->exclude_list.end()) {
				throw ParserException("Column \"%s\" cannot occur in both EXCEPT and REPLACE list", exclude_entry);
			}
			result->replace_list.insert(make_pair(std::move(exclude_entry), std::move(replace_expression)));
		}
	}
	if (star->regex) {
		D_ASSERT(result->relation_name.empty());
		D_ASSERT(result->exclude_list.empty());
		D_ASSERT(result->replace_list.empty());
		result->regex = star->regex;
	}
	result->columns = star->columns;
	return std::move(result);
}

unique_ptr<ParsedExpression> Transformer::TransformColumnRef(duckdb_libpgquery::PGColumnRef *root) {
	auto fields = root->fields;
	auto head_node = (duckdb_libpgquery::PGNode *)fields->head->data.ptr_value;
	switch (head_node->type) {
	case duckdb_libpgquery::T_PGString: {
		if (fields->length < 1) {
			throw InternalException("Unexpected field length");
		}
		vector<string> column_names;
		for (auto node = fields->head; node; node = node->next) {
			column_names.emplace_back(reinterpret_cast<duckdb_libpgquery::PGValue *>(node->data.ptr_value)->val.str);
		}
		auto colref = make_unique<ColumnRefExpression>(std::move(column_names));
		colref->query_location = root->location;
		return std::move(colref);
	}
	case duckdb_libpgquery::T_PGAStar: {
		return TransformStarExpression(head_node);
	}
	default:
		throw NotImplementedException("ColumnRef not implemented!");
	}
}

} // namespace duckdb






namespace duckdb {

unique_ptr<ConstantExpression> Transformer::TransformValue(duckdb_libpgquery::PGValue val) {
	switch (val.type) {
	case duckdb_libpgquery::T_PGInteger:
		D_ASSERT(val.val.ival <= NumericLimits<int32_t>::Maximum());
		return make_unique<ConstantExpression>(Value::INTEGER((int32_t)val.val.ival));
	case duckdb_libpgquery::T_PGBitString: // FIXME: this should actually convert to BLOB
	case duckdb_libpgquery::T_PGString:
		return make_unique<ConstantExpression>(Value(string(val.val.str)));
	case duckdb_libpgquery::T_PGFloat: {
		string_t str_val(val.val.str);
		bool try_cast_as_integer = true;
		bool try_cast_as_decimal = true;
		int decimal_position = -1;
		for (idx_t i = 0; i < str_val.GetSize(); i++) {
			if (val.val.str[i] == '.') {
				// decimal point: cast as either decimal or double
				try_cast_as_integer = false;
				decimal_position = i;
			}
			if (val.val.str[i] == 'e' || val.val.str[i] == 'E') {
				// found exponent, cast as double
				try_cast_as_integer = false;
				try_cast_as_decimal = false;
			}
		}
		if (try_cast_as_integer) {
			int64_t bigint_value;
			// try to cast as bigint first
			if (TryCast::Operation<string_t, int64_t>(str_val, bigint_value)) {
				// successfully cast to bigint: bigint value
				return make_unique<ConstantExpression>(Value::BIGINT(bigint_value));
			}
			hugeint_t hugeint_value;
			// if that is not successful; try to cast as hugeint
			if (TryCast::Operation<string_t, hugeint_t>(str_val, hugeint_value)) {
				// successfully cast to bigint: bigint value
				return make_unique<ConstantExpression>(Value::HUGEINT(hugeint_value));
			}
		}
		idx_t decimal_offset = val.val.str[0] == '-' ? 3 : 2;
		if (try_cast_as_decimal && decimal_position >= 0 &&
		    str_val.GetSize() < Decimal::MAX_WIDTH_DECIMAL + decimal_offset) {
			// figure out the width/scale based on the decimal position
			auto width = uint8_t(str_val.GetSize() - 1);
			auto scale = uint8_t(width - decimal_position);
			if (val.val.str[0] == '-') {
				width--;
			}
			if (width <= Decimal::MAX_WIDTH_DECIMAL) {
				// we can cast the value as a decimal
				Value val = Value(str_val);
				val = val.DefaultCastAs(LogicalType::DECIMAL(width, scale));
				return make_unique<ConstantExpression>(std::move(val));
			}
		}
		// if there is a decimal or the value is too big to cast as either hugeint or bigint
		double dbl_value = Cast::Operation<string_t, double>(str_val);
		return make_unique<ConstantExpression>(Value::DOUBLE(dbl_value));
	}
	case duckdb_libpgquery::T_PGNull:
		return make_unique<ConstantExpression>(Value(LogicalType::SQLNULL));
	default:
		throw NotImplementedException("Value not implemented!");
	}
}

unique_ptr<ParsedExpression> Transformer::TransformConstant(duckdb_libpgquery::PGAConst *c) {
	return TransformValue(c->val);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformResTarget(duckdb_libpgquery::PGResTarget *root) {
	D_ASSERT(root);

	auto expr = TransformExpression(root->val);
	if (!expr) {
		return nullptr;
	}
	if (root->name) {
		expr->alias = string(root->name);
	}
	return expr;
}

unique_ptr<ParsedExpression> Transformer::TransformNamedArg(duckdb_libpgquery::PGNamedArgExpr *root) {
	D_ASSERT(root);

	auto expr = TransformExpression((duckdb_libpgquery::PGNode *)root->arg);
	if (root->name) {
		expr->alias = string(root->name);
	}
	return expr;
}

unique_ptr<ParsedExpression> Transformer::TransformExpression(duckdb_libpgquery::PGNode *node) {
	if (!node) {
		return nullptr;
	}

	auto stack_checker = StackCheck();

	switch (node->type) {
	case duckdb_libpgquery::T_PGColumnRef:
		return TransformColumnRef(reinterpret_cast<duckdb_libpgquery::PGColumnRef *>(node));
	case duckdb_libpgquery::T_PGAConst:
		return TransformConstant(reinterpret_cast<duckdb_libpgquery::PGAConst *>(node));
	case duckdb_libpgquery::T_PGAExpr:
		return TransformAExpr(reinterpret_cast<duckdb_libpgquery::PGAExpr *>(node));
	case duckdb_libpgquery::T_PGFuncCall:
		return TransformFuncCall(reinterpret_cast<duckdb_libpgquery::PGFuncCall *>(node));
	case duckdb_libpgquery::T_PGBoolExpr:
		return TransformBoolExpr(reinterpret_cast<duckdb_libpgquery::PGBoolExpr *>(node));
	case duckdb_libpgquery::T_PGTypeCast:
		return TransformTypeCast(reinterpret_cast<duckdb_libpgquery::PGTypeCast *>(node));
	case duckdb_libpgquery::T_PGCaseExpr:
		return TransformCase(reinterpret_cast<duckdb_libpgquery::PGCaseExpr *>(node));
	case duckdb_libpgquery::T_PGSubLink:
		return TransformSubquery(reinterpret_cast<duckdb_libpgquery::PGSubLink *>(node));
	case duckdb_libpgquery::T_PGCoalesceExpr:
		return TransformCoalesce(reinterpret_cast<duckdb_libpgquery::PGAExpr *>(node));
	case duckdb_libpgquery::T_PGNullTest:
		return TransformNullTest(reinterpret_cast<duckdb_libpgquery::PGNullTest *>(node));
	case duckdb_libpgquery::T_PGResTarget:
		return TransformResTarget(reinterpret_cast<duckdb_libpgquery::PGResTarget *>(node));
	case duckdb_libpgquery::T_PGParamRef:
		return TransformParamRef(reinterpret_cast<duckdb_libpgquery::PGParamRef *>(node));
	case duckdb_libpgquery::T_PGNamedArgExpr:
		return TransformNamedArg(reinterpret_cast<duckdb_libpgquery::PGNamedArgExpr *>(node));
	case duckdb_libpgquery::T_PGSQLValueFunction:
		return TransformSQLValueFunction(reinterpret_cast<duckdb_libpgquery::PGSQLValueFunction *>(node));
	case duckdb_libpgquery::T_PGSetToDefault:
		return make_unique<DefaultExpression>();
	case duckdb_libpgquery::T_PGCollateClause:
		return TransformCollateExpr(reinterpret_cast<duckdb_libpgquery::PGCollateClause *>(node));
	case duckdb_libpgquery::T_PGIntervalConstant:
		return TransformInterval(reinterpret_cast<duckdb_libpgquery::PGIntervalConstant *>(node));
	case duckdb_libpgquery::T_PGLambdaFunction:
		return TransformLambda(reinterpret_cast<duckdb_libpgquery::PGLambdaFunction *>(node));
	case duckdb_libpgquery::T_PGAIndirection:
		return TransformArrayAccess(reinterpret_cast<duckdb_libpgquery::PGAIndirection *>(node));
	case duckdb_libpgquery::T_PGPositionalReference:
		return TransformPositionalReference(reinterpret_cast<duckdb_libpgquery::PGPositionalReference *>(node));
	case duckdb_libpgquery::T_PGGroupingFunc:
		return TransformGroupingFunction(reinterpret_cast<duckdb_libpgquery::PGGroupingFunc *>(node));
	case duckdb_libpgquery::T_PGAStar:
		return TransformStarExpression(node);
	default:
		throw NotImplementedException("Expr of type %d not implemented\n", (int)node->type);
	}
}

void Transformer::TransformExpressionList(duckdb_libpgquery::PGList &list,
                                          vector<unique_ptr<ParsedExpression>> &result) {
	for (auto node = list.head; node != nullptr; node = node->next) {
		auto target = reinterpret_cast<duckdb_libpgquery::PGNode *>(node->data.ptr_value);
		D_ASSERT(target);

		auto expr = TransformExpression(target);
		D_ASSERT(expr);

		result.push_back(std::move(expr));
	}
}

} // namespace duckdb











namespace duckdb {

static ExpressionType WindowToExpressionType(string &fun_name) {
	if (fun_name == "rank") {
		return ExpressionType::WINDOW_RANK;
	} else if (fun_name == "rank_dense" || fun_name == "dense_rank") {
		return ExpressionType::WINDOW_RANK_DENSE;
	} else if (fun_name == "percent_rank") {
		return ExpressionType::WINDOW_PERCENT_RANK;
	} else if (fun_name == "row_number") {
		return ExpressionType::WINDOW_ROW_NUMBER;
	} else if (fun_name == "first_value" || fun_name == "first") {
		return ExpressionType::WINDOW_FIRST_VALUE;
	} else if (fun_name == "last_value" || fun_name == "last") {
		return ExpressionType::WINDOW_LAST_VALUE;
	} else if (fun_name == "nth_value" || fun_name == "last") {
		return ExpressionType::WINDOW_NTH_VALUE;
	} else if (fun_name == "cume_dist") {
		return ExpressionType::WINDOW_CUME_DIST;
	} else if (fun_name == "lead") {
		return ExpressionType::WINDOW_LEAD;
	} else if (fun_name == "lag") {
		return ExpressionType::WINDOW_LAG;
	} else if (fun_name == "ntile") {
		return ExpressionType::WINDOW_NTILE;
	}

	return ExpressionType::WINDOW_AGGREGATE;
}

void Transformer::TransformWindowDef(duckdb_libpgquery::PGWindowDef *window_spec, WindowExpression *expr) {
	D_ASSERT(window_spec);
	D_ASSERT(expr);

	// next: partitioning/ordering expressions
	if (window_spec->partitionClause) {
		TransformExpressionList(*window_spec->partitionClause, expr->partitions);
	}
	TransformOrderBy(window_spec->orderClause, expr->orders);
}

void Transformer::TransformWindowFrame(duckdb_libpgquery::PGWindowDef *window_spec, WindowExpression *expr) {
	D_ASSERT(window_spec);
	D_ASSERT(expr);

	// finally: specifics of bounds
	expr->start_expr = TransformExpression(window_spec->startOffset);
	expr->end_expr = TransformExpression(window_spec->endOffset);

	if ((window_spec->frameOptions & FRAMEOPTION_END_UNBOUNDED_PRECEDING) ||
	    (window_spec->frameOptions & FRAMEOPTION_START_UNBOUNDED_FOLLOWING)) {
		throw InternalException(
		    "Window frames starting with unbounded following or ending in unbounded preceding make no sense");
	}

	const bool rangeMode = (window_spec->frameOptions & FRAMEOPTION_RANGE) != 0;
	if (window_spec->frameOptions & FRAMEOPTION_START_UNBOUNDED_PRECEDING) {
		expr->start = WindowBoundary::UNBOUNDED_PRECEDING;
	} else if (window_spec->frameOptions & FRAMEOPTION_START_VALUE_PRECEDING) {
		expr->start = rangeMode ? WindowBoundary::EXPR_PRECEDING_RANGE : WindowBoundary::EXPR_PRECEDING_ROWS;
	} else if (window_spec->frameOptions & FRAMEOPTION_START_VALUE_FOLLOWING) {
		expr->start = rangeMode ? WindowBoundary::EXPR_FOLLOWING_RANGE : WindowBoundary::EXPR_FOLLOWING_ROWS;
	} else if (window_spec->frameOptions & FRAMEOPTION_START_CURRENT_ROW) {
		expr->start = rangeMode ? WindowBoundary::CURRENT_ROW_RANGE : WindowBoundary::CURRENT_ROW_ROWS;
	}

	if (window_spec->frameOptions & FRAMEOPTION_END_UNBOUNDED_FOLLOWING) {
		expr->end = WindowBoundary::UNBOUNDED_FOLLOWING;
	} else if (window_spec->frameOptions & FRAMEOPTION_END_VALUE_PRECEDING) {
		expr->end = rangeMode ? WindowBoundary::EXPR_PRECEDING_RANGE : WindowBoundary::EXPR_PRECEDING_ROWS;
	} else if (window_spec->frameOptions & FRAMEOPTION_END_VALUE_FOLLOWING) {
		expr->end = rangeMode ? WindowBoundary::EXPR_FOLLOWING_RANGE : WindowBoundary::EXPR_FOLLOWING_ROWS;
	} else if (window_spec->frameOptions & FRAMEOPTION_END_CURRENT_ROW) {
		expr->end = rangeMode ? WindowBoundary::CURRENT_ROW_RANGE : WindowBoundary::CURRENT_ROW_ROWS;
	}

	D_ASSERT(expr->start != WindowBoundary::INVALID && expr->end != WindowBoundary::INVALID);
	if (((window_spec->frameOptions & (FRAMEOPTION_START_VALUE_PRECEDING | FRAMEOPTION_START_VALUE_FOLLOWING)) &&
	     !expr->start_expr) ||
	    ((window_spec->frameOptions & (FRAMEOPTION_END_VALUE_PRECEDING | FRAMEOPTION_END_VALUE_FOLLOWING)) &&
	     !expr->end_expr)) {
		throw InternalException("Failed to transform window boundary expression");
	}
}

unique_ptr<ParsedExpression> Transformer::TransformFuncCall(duckdb_libpgquery::PGFuncCall *root) {
	auto name = root->funcname;
	string catalog, schema, function_name;
	if (name->length == 3) {
		// catalog + schema + name
		catalog = reinterpret_cast<duckdb_libpgquery::PGValue *>(name->head->data.ptr_value)->val.str;
		schema = reinterpret_cast<duckdb_libpgquery::PGValue *>(name->head->next->data.ptr_value)->val.str;
		function_name = reinterpret_cast<duckdb_libpgquery::PGValue *>(name->head->next->next->data.ptr_value)->val.str;
	} else if (name->length == 2) {
		// schema + name
		catalog = INVALID_CATALOG;
		schema = reinterpret_cast<duckdb_libpgquery::PGValue *>(name->head->data.ptr_value)->val.str;
		function_name = reinterpret_cast<duckdb_libpgquery::PGValue *>(name->head->next->data.ptr_value)->val.str;
	} else if (name->length == 1) {
		// unqualified name
		catalog = INVALID_CATALOG;
		schema = INVALID_SCHEMA;
		function_name = reinterpret_cast<duckdb_libpgquery::PGValue *>(name->head->data.ptr_value)->val.str;
	} else {
		throw InternalException("TransformFuncCall - Expected 1, 2 or 3 qualifications");
	}

	auto lowercase_name = StringUtil::Lower(function_name);

	if (root->over) {
		const auto win_fun_type = WindowToExpressionType(lowercase_name);
		if (win_fun_type == ExpressionType::INVALID) {
			throw InternalException("Unknown/unsupported window function");
		}

		if (root->agg_distinct) {
			throw ParserException("DISTINCT is not implemented for window functions!");
		}

		if (root->agg_order) {
			throw ParserException("ORDER BY is not implemented for window functions!");
		}

		if (win_fun_type != ExpressionType::WINDOW_AGGREGATE && root->agg_filter) {
			throw ParserException("FILTER is not implemented for non-aggregate window functions!");
		}
		if (root->export_state) {
			throw ParserException("EXPORT_STATE is not supported for window functions!");
		}

		if (win_fun_type == ExpressionType::WINDOW_AGGREGATE && root->agg_ignore_nulls) {
			throw ParserException("IGNORE NULLS is not supported for windowed aggregates");
		}

		auto expr = make_unique<WindowExpression>(win_fun_type, std::move(catalog), std::move(schema), lowercase_name);
		expr->ignore_nulls = root->agg_ignore_nulls;

		if (root->agg_filter) {
			auto filter_expr = TransformExpression(root->agg_filter);
			expr->filter_expr = std::move(filter_expr);
		}

		if (root->args) {
			vector<unique_ptr<ParsedExpression>> function_list;
			TransformExpressionList(*root->args, function_list);

			if (win_fun_type == ExpressionType::WINDOW_AGGREGATE) {
				for (auto &child : function_list) {
					expr->children.push_back(std::move(child));
				}
			} else {
				if (!function_list.empty()) {
					expr->children.push_back(std::move(function_list[0]));
				}
				if (win_fun_type == ExpressionType::WINDOW_LEAD || win_fun_type == ExpressionType::WINDOW_LAG) {
					if (function_list.size() > 1) {
						expr->offset_expr = std::move(function_list[1]);
					}
					if (function_list.size() > 2) {
						expr->default_expr = std::move(function_list[2]);
					}
					if (function_list.size() > 3) {
						throw ParserException("Incorrect number of parameters for function %s", lowercase_name);
					}
				} else if (win_fun_type == ExpressionType::WINDOW_NTH_VALUE) {
					if (function_list.size() > 1) {
						expr->children.push_back(std::move(function_list[1]));
					}
					if (function_list.size() > 2) {
						throw ParserException("Incorrect number of parameters for function %s", lowercase_name);
					}
				} else {
					if (function_list.size() > 1) {
						throw ParserException("Incorrect number of parameters for function %s", lowercase_name);
					}
				}
			}
		}
		auto window_spec = reinterpret_cast<duckdb_libpgquery::PGWindowDef *>(root->over);
		if (window_spec->name) {
			auto it = window_clauses.find(StringUtil::Lower(string(window_spec->name)));
			if (it == window_clauses.end()) {
				throw ParserException("window \"%s\" does not exist", window_spec->name);
			}
			window_spec = it->second;
			D_ASSERT(window_spec);
		}
		auto window_ref = window_spec;
		if (window_ref->refname) {
			auto it = window_clauses.find(StringUtil::Lower(string(window_spec->refname)));
			if (it == window_clauses.end()) {
				throw ParserException("window \"%s\" does not exist", window_spec->refname);
			}
			window_ref = it->second;
			D_ASSERT(window_ref);
		}
		TransformWindowDef(window_ref, expr.get());
		TransformWindowFrame(window_spec, expr.get());
		expr->query_location = root->location;
		return std::move(expr);
	}

	if (root->agg_ignore_nulls) {
		throw ParserException("IGNORE NULLS is not supported for non-window functions");
	}

	//  TransformExpressionList??
	vector<unique_ptr<ParsedExpression>> children;
	if (root->args != nullptr) {
		for (auto node = root->args->head; node != nullptr; node = node->next) {
			auto child_expr = TransformExpression((duckdb_libpgquery::PGNode *)node->data.ptr_value);
			children.push_back(std::move(child_expr));
		}
	}
	unique_ptr<ParsedExpression> filter_expr;
	if (root->agg_filter) {
		filter_expr = TransformExpression(root->agg_filter);
	}

	auto order_bys = make_unique<OrderModifier>();
	TransformOrderBy(root->agg_order, order_bys->orders);

	// Ordered aggregates can be either WITHIN GROUP or after the function arguments
	if (root->agg_within_group) {
		//	https://www.postgresql.org/docs/current/functions-aggregate.html#FUNCTIONS-ORDEREDSET-TABLE
		//  Since we implement "ordered aggregates" without sorting,
		//  we map all the ones we support to the corresponding aggregate function.
		if (order_bys->orders.size() != 1) {
			throw ParserException("Cannot use multiple ORDER BY clauses with WITHIN GROUP");
		}
		if (lowercase_name == "percentile_cont") {
			if (children.size() != 1) {
				throw ParserException("Wrong number of arguments for PERCENTILE_CONT");
			}
			lowercase_name = "quantile_cont";
		} else if (lowercase_name == "percentile_disc") {
			if (children.size() != 1) {
				throw ParserException("Wrong number of arguments for PERCENTILE_DISC");
			}
			lowercase_name = "quantile_disc";
		} else if (lowercase_name == "mode") {
			if (!children.empty()) {
				throw ParserException("Wrong number of arguments for MODE");
			}
			lowercase_name = "mode";
		} else {
			throw ParserException("Unknown ordered aggregate \"%s\".", function_name);
		}
	}

	// star gets eaten in the parser
	if (lowercase_name == "count" && children.empty()) {
		lowercase_name = "count_star";
	}

	if (lowercase_name == "if") {
		if (children.size() != 3) {
			throw ParserException("Wrong number of arguments to IF.");
		}
		auto expr = make_unique<CaseExpression>();
		CaseCheck check;
		check.when_expr = std::move(children[0]);
		check.then_expr = std::move(children[1]);
		expr->case_checks.push_back(std::move(check));
		expr->else_expr = std::move(children[2]);
		return std::move(expr);
	} else if (lowercase_name == "construct_array") {
		auto construct_array = make_unique<OperatorExpression>(ExpressionType::ARRAY_CONSTRUCTOR);
		construct_array->children = std::move(children);
		return std::move(construct_array);
	} else if (lowercase_name == "ifnull") {
		if (children.size() != 2) {
			throw ParserException("Wrong number of arguments to IFNULL.");
		}

		//  Two-argument COALESCE
		auto coalesce_op = make_unique<OperatorExpression>(ExpressionType::OPERATOR_COALESCE);
		coalesce_op->children.push_back(std::move(children[0]));
		coalesce_op->children.push_back(std::move(children[1]));
		return std::move(coalesce_op);
	}

	auto function = make_unique<FunctionExpression>(std::move(catalog), std::move(schema), lowercase_name.c_str(),
	                                                std::move(children), std::move(filter_expr), std::move(order_bys),
	                                                root->agg_distinct, false, root->export_state);
	function->query_location = root->location;

	return std::move(function);
}

static string SQLValueOpToString(duckdb_libpgquery::PGSQLValueFunctionOp op) {
	switch (op) {
	case duckdb_libpgquery::PG_SVFOP_CURRENT_DATE:
		return "current_date";
	case duckdb_libpgquery::PG_SVFOP_CURRENT_TIME:
		return "get_current_time";
	case duckdb_libpgquery::PG_SVFOP_CURRENT_TIME_N:
		return "current_time_n";
	case duckdb_libpgquery::PG_SVFOP_CURRENT_TIMESTAMP:
		return "get_current_timestamp";
	case duckdb_libpgquery::PG_SVFOP_CURRENT_TIMESTAMP_N:
		return "current_timestamp_n";
	case duckdb_libpgquery::PG_SVFOP_LOCALTIME:
		return "current_localtime";
	case duckdb_libpgquery::PG_SVFOP_LOCALTIME_N:
		return "current_localtime_n";
	case duckdb_libpgquery::PG_SVFOP_LOCALTIMESTAMP:
		return "current_localtimestamp";
	case duckdb_libpgquery::PG_SVFOP_LOCALTIMESTAMP_N:
		return "current_localtimestamp_n";
	case duckdb_libpgquery::PG_SVFOP_CURRENT_ROLE:
		return "current_role";
	case duckdb_libpgquery::PG_SVFOP_CURRENT_USER:
		return "current_user";
	case duckdb_libpgquery::PG_SVFOP_USER:
		return "user";
	case duckdb_libpgquery::PG_SVFOP_SESSION_USER:
		return "session_user";
	case duckdb_libpgquery::PG_SVFOP_CURRENT_CATALOG:
		return "current_catalog";
	case duckdb_libpgquery::PG_SVFOP_CURRENT_SCHEMA:
		return "current_schema";
	default:
		throw InternalException("Could not find named SQL value function specification " + to_string((int)op));
	}
}

unique_ptr<ParsedExpression> Transformer::TransformSQLValueFunction(duckdb_libpgquery::PGSQLValueFunction *node) {
	D_ASSERT(node);
	vector<unique_ptr<ParsedExpression>> children;
	auto fname = SQLValueOpToString(node->op);
	return make_unique<FunctionExpression>(fname, std::move(children));
}

} // namespace duckdb



namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformGroupingFunction(duckdb_libpgquery::PGGroupingFunc *n) {
	auto op = make_unique<OperatorExpression>(ExpressionType::GROUPING_FUNCTION);
	for (auto node = n->args->head; node; node = node->next) {
		auto n = (duckdb_libpgquery::PGNode *)node->data.ptr_value;
		op->children.push_back(TransformExpression(n));
	}
	op->query_location = n->location;
	return std::move(op);
}

} // namespace duckdb






namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformInterval(duckdb_libpgquery::PGIntervalConstant *node) {
	// handle post-fix notation of INTERVAL

	// three scenarios
	// interval (expr) year
	// interval 'string' year
	// interval int year
	unique_ptr<ParsedExpression> expr;
	switch (node->val_type) {
	case duckdb_libpgquery::T_PGAExpr:
		expr = TransformExpression(node->eval);
		break;
	case duckdb_libpgquery::T_PGString:
		expr = make_unique<ConstantExpression>(Value(node->sval));
		break;
	case duckdb_libpgquery::T_PGInteger:
		expr = make_unique<ConstantExpression>(Value(node->ival));
		break;
	default:
		throw InternalException("Unsupported interval transformation");
	}

	if (!node->typmods) {
		return make_unique<CastExpression>(LogicalType::INTERVAL, std::move(expr));
	}

	int32_t mask = ((duckdb_libpgquery::PGAConst *)node->typmods->head->data.ptr_value)->val.val.ival;
	// these seemingly random constants are from datetime.hpp
	// they are copied here to avoid having to include this header
	// the bitshift is from the function INTERVAL_MASK in the parser
	constexpr int32_t MONTH_MASK = 1 << 1;
	constexpr int32_t YEAR_MASK = 1 << 2;
	constexpr int32_t DAY_MASK = 1 << 3;
	constexpr int32_t HOUR_MASK = 1 << 10;
	constexpr int32_t MINUTE_MASK = 1 << 11;
	constexpr int32_t SECOND_MASK = 1 << 12;
	constexpr int32_t MILLISECOND_MASK = 1 << 13;
	constexpr int32_t MICROSECOND_MASK = 1 << 14;

	// we need to check certain combinations
	// because certain interval masks (e.g. INTERVAL '10' HOURS TO DAYS) set multiple bits
	// for now we don't support all of the combined ones
	// (we might add support if someone complains about it)

	string fname;
	LogicalType target_type;
	if (mask & YEAR_MASK && mask & MONTH_MASK) {
		// DAY TO HOUR
		throw ParserException("YEAR TO MONTH is not supported");
	} else if (mask & DAY_MASK && mask & HOUR_MASK) {
		// DAY TO HOUR
		throw ParserException("DAY TO HOUR is not supported");
	} else if (mask & DAY_MASK && mask & MINUTE_MASK) {
		// DAY TO MINUTE
		throw ParserException("DAY TO MINUTE is not supported");
	} else if (mask & DAY_MASK && mask & SECOND_MASK) {
		// DAY TO SECOND
		throw ParserException("DAY TO SECOND is not supported");
	} else if (mask & HOUR_MASK && mask & MINUTE_MASK) {
		// DAY TO SECOND
		throw ParserException("HOUR TO MINUTE is not supported");
	} else if (mask & HOUR_MASK && mask & SECOND_MASK) {
		// DAY TO SECOND
		throw ParserException("HOUR TO SECOND is not supported");
	} else if (mask & MINUTE_MASK && mask & SECOND_MASK) {
		// DAY TO SECOND
		throw ParserException("MINUTE TO SECOND is not supported");
	} else if (mask & YEAR_MASK) {
		// YEAR
		fname = "to_years";
		target_type = LogicalType::INTEGER;
	} else if (mask & MONTH_MASK) {
		// MONTH
		fname = "to_months";
		target_type = LogicalType::INTEGER;
	} else if (mask & DAY_MASK) {
		// DAY
		fname = "to_days";
		target_type = LogicalType::INTEGER;
	} else if (mask & HOUR_MASK) {
		// HOUR
		fname = "to_hours";
		target_type = LogicalType::BIGINT;
	} else if (mask & MINUTE_MASK) {
		// MINUTE
		fname = "to_minutes";
		target_type = LogicalType::BIGINT;
	} else if (mask & SECOND_MASK) {
		// SECOND
		fname = "to_seconds";
		target_type = LogicalType::BIGINT;
	} else if (mask & MILLISECOND_MASK) {
		// MILLISECOND
		fname = "to_milliseconds";
		target_type = LogicalType::BIGINT;
	} else if (mask & MICROSECOND_MASK) {
		// SECOND
		fname = "to_microseconds";
		target_type = LogicalType::BIGINT;
	} else {
		throw InternalException("Unsupported interval post-fix");
	}
	// first push a cast to the target type
	expr = make_unique<CastExpression>(target_type, std::move(expr));
	// now push the operation
	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(std::move(expr));
	return make_unique<FunctionExpression>(fname, std::move(children));
}

} // namespace duckdb




namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformNullTest(duckdb_libpgquery::PGNullTest *root) {
	D_ASSERT(root);
	auto arg = TransformExpression(reinterpret_cast<duckdb_libpgquery::PGNode *>(root->arg));
	if (root->argisrow) {
		throw NotImplementedException("IS NULL argisrow");
	}
	ExpressionType expr_type = (root->nulltesttype == duckdb_libpgquery::PG_IS_NULL)
	                               ? ExpressionType::OPERATOR_IS_NULL
	                               : ExpressionType::OPERATOR_IS_NOT_NULL;

	return unique_ptr<ParsedExpression>(new OperatorExpression(expr_type, std::move(arg)));
}

} // namespace duckdb




namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformLambda(duckdb_libpgquery::PGLambdaFunction *node) {

	D_ASSERT(node->lhs);
	D_ASSERT(node->rhs);

	auto lhs = TransformExpression(node->lhs);
	auto rhs = TransformExpression(node->rhs);
	D_ASSERT(lhs);
	D_ASSERT(rhs);
	return make_unique<LambdaExpression>(std::move(lhs), std::move(rhs));
}

} // namespace duckdb














namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformUnaryOperator(const string &op, unique_ptr<ParsedExpression> child) {
	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(std::move(child));

	// built-in operator function
	auto result = make_unique<FunctionExpression>(op, std::move(children));
	result->is_operator = true;
	return std::move(result);
}

unique_ptr<ParsedExpression> Transformer::TransformBinaryOperator(const string &op, unique_ptr<ParsedExpression> left,
                                                                  unique_ptr<ParsedExpression> right) {
	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(std::move(left));
	children.push_back(std::move(right));

	if (op == "~" || op == "!~") {
		// rewrite 'asdf' SIMILAR TO '.*sd.*' into regexp_full_match('asdf', '.*sd.*')
		bool invert_similar = op == "!~";

		auto result = make_unique<FunctionExpression>("regexp_full_match", std::move(children));
		if (invert_similar) {
			return make_unique<OperatorExpression>(ExpressionType::OPERATOR_NOT, std::move(result));
		} else {
			return std::move(result);
		}
	} else {
		auto target_type = OperatorToExpressionType(op);
		if (target_type != ExpressionType::INVALID) {
			// built-in comparison operator
			return make_unique<ComparisonExpression>(target_type, std::move(children[0]), std::move(children[1]));
		}
		// not a special operator: convert to a function expression
		auto result = make_unique<FunctionExpression>(op, std::move(children));
		result->is_operator = true;
		return std::move(result);
	}
}

unique_ptr<ParsedExpression> Transformer::TransformAExprInternal(duckdb_libpgquery::PGAExpr *root) {
	D_ASSERT(root);
	auto name = string((reinterpret_cast<duckdb_libpgquery::PGValue *>(root->name->head->data.ptr_value))->val.str);

	switch (root->kind) {
	case duckdb_libpgquery::PG_AEXPR_OP_ALL:
	case duckdb_libpgquery::PG_AEXPR_OP_ANY: {
		// left=ANY(right)
		// we turn this into left=ANY((SELECT UNNEST(right)))
		auto left_expr = TransformExpression(root->lexpr);
		auto right_expr = TransformExpression(root->rexpr);

		auto subquery_expr = make_unique<SubqueryExpression>();
		auto select_statement = make_unique<SelectStatement>();
		auto select_node = make_unique<SelectNode>();
		vector<unique_ptr<ParsedExpression>> children;
		children.push_back(std::move(right_expr));

		select_node->select_list.push_back(make_unique<FunctionExpression>("UNNEST", std::move(children)));
		select_node->from_table = make_unique<EmptyTableRef>();
		select_statement->node = std::move(select_node);
		subquery_expr->subquery = std::move(select_statement);
		subquery_expr->subquery_type = SubqueryType::ANY;
		subquery_expr->child = std::move(left_expr);
		subquery_expr->comparison_type = OperatorToExpressionType(name);
		subquery_expr->query_location = root->location;

		if (root->kind == duckdb_libpgquery::PG_AEXPR_OP_ALL) {
			// ALL sublink is equivalent to NOT(ANY) with inverted comparison
			// e.g. [= ALL()] is equivalent to [NOT(<> ANY())]
			// first invert the comparison type
			subquery_expr->comparison_type = NegateComparisionExpression(subquery_expr->comparison_type);
			return make_unique<OperatorExpression>(ExpressionType::OPERATOR_NOT, std::move(subquery_expr));
		}
		return std::move(subquery_expr);
	}
	case duckdb_libpgquery::PG_AEXPR_IN: {
		auto left_expr = TransformExpression(root->lexpr);
		ExpressionType operator_type;
		// this looks very odd, but seems to be the way to find out its NOT IN
		if (name == "<>") {
			// NOT IN
			operator_type = ExpressionType::COMPARE_NOT_IN;
		} else {
			// IN
			operator_type = ExpressionType::COMPARE_IN;
		}
		auto result = make_unique<OperatorExpression>(operator_type, std::move(left_expr));
		result->query_location = root->location;
		TransformExpressionList(*((duckdb_libpgquery::PGList *)root->rexpr), result->children);
		return std::move(result);
	}
	// rewrite NULLIF(a, b) into CASE WHEN a=b THEN NULL ELSE a END
	case duckdb_libpgquery::PG_AEXPR_NULLIF: {
		vector<unique_ptr<ParsedExpression>> children;
		children.push_back(TransformExpression(root->lexpr));
		children.push_back(TransformExpression(root->rexpr));
		return make_unique<FunctionExpression>("nullif", std::move(children));
	}
	// rewrite (NOT) X BETWEEN A AND B into (NOT) AND(GREATERTHANOREQUALTO(X,
	// A), LESSTHANOREQUALTO(X, B))
	case duckdb_libpgquery::PG_AEXPR_BETWEEN:
	case duckdb_libpgquery::PG_AEXPR_NOT_BETWEEN: {
		auto between_args = reinterpret_cast<duckdb_libpgquery::PGList *>(root->rexpr);
		if (between_args->length != 2 || !between_args->head->data.ptr_value || !between_args->tail->data.ptr_value) {
			throw InternalException("(NOT) BETWEEN needs two args");
		}

		auto input = TransformExpression(root->lexpr);
		auto between_left =
		    TransformExpression(reinterpret_cast<duckdb_libpgquery::PGNode *>(between_args->head->data.ptr_value));
		auto between_right =
		    TransformExpression(reinterpret_cast<duckdb_libpgquery::PGNode *>(between_args->tail->data.ptr_value));

		auto compare_between =
		    make_unique<BetweenExpression>(std::move(input), std::move(between_left), std::move(between_right));
		if (root->kind == duckdb_libpgquery::PG_AEXPR_BETWEEN) {
			return std::move(compare_between);
		} else {
			return make_unique<OperatorExpression>(ExpressionType::OPERATOR_NOT, std::move(compare_between));
		}
	}
	// rewrite SIMILAR TO into regexp_full_match('asdf', '.*sd.*')
	case duckdb_libpgquery::PG_AEXPR_SIMILAR: {
		auto left_expr = TransformExpression(root->lexpr);
		auto right_expr = TransformExpression(root->rexpr);

		vector<unique_ptr<ParsedExpression>> children;
		children.push_back(std::move(left_expr));

		auto &similar_func = reinterpret_cast<FunctionExpression &>(*right_expr);
		D_ASSERT(similar_func.function_name == "similar_escape");
		D_ASSERT(similar_func.children.size() == 2);
		if (similar_func.children[1]->type != ExpressionType::VALUE_CONSTANT) {
			throw NotImplementedException("Custom escape in SIMILAR TO");
		}
		auto &constant = (ConstantExpression &)*similar_func.children[1];
		if (!constant.value.IsNull()) {
			throw NotImplementedException("Custom escape in SIMILAR TO");
		}
		// take the child of the similar_func
		children.push_back(std::move(similar_func.children[0]));

		// this looks very odd, but seems to be the way to find out its NOT IN
		bool invert_similar = false;
		if (name == "!~") {
			// NOT SIMILAR TO
			invert_similar = true;
		}
		const auto regex_function = "regexp_full_match";
		auto result = make_unique<FunctionExpression>(regex_function, std::move(children));

		if (invert_similar) {
			return make_unique<OperatorExpression>(ExpressionType::OPERATOR_NOT, std::move(result));
		} else {
			return std::move(result);
		}
	}
	case duckdb_libpgquery::PG_AEXPR_NOT_DISTINCT: {
		auto left_expr = TransformExpression(root->lexpr);
		auto right_expr = TransformExpression(root->rexpr);
		return make_unique<ComparisonExpression>(ExpressionType::COMPARE_NOT_DISTINCT_FROM, std::move(left_expr),
		                                         std::move(right_expr));
	}
	case duckdb_libpgquery::PG_AEXPR_DISTINCT: {
		auto left_expr = TransformExpression(root->lexpr);
		auto right_expr = TransformExpression(root->rexpr);
		return make_unique<ComparisonExpression>(ExpressionType::COMPARE_DISTINCT_FROM, std::move(left_expr),
		                                         std::move(right_expr));
	}

	default:
		break;
	}
	auto left_expr = TransformExpression(root->lexpr);
	auto right_expr = TransformExpression(root->rexpr);

	if (!left_expr) {
		// prefix operator
		return TransformUnaryOperator(name, std::move(right_expr));
	} else if (!right_expr) {
		// postfix operator, only ! is currently supported
		return TransformUnaryOperator(name + "__postfix", std::move(left_expr));
	} else {
		return TransformBinaryOperator(name, std::move(left_expr), std::move(right_expr));
	}
}

unique_ptr<ParsedExpression> Transformer::TransformAExpr(duckdb_libpgquery::PGAExpr *root) {
	auto result = TransformAExprInternal(root);
	if (result) {
		result->query_location = root->location;
	}
	return result;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformParamRef(duckdb_libpgquery::PGParamRef *node) {
	D_ASSERT(node);
	auto expr = make_unique<ParameterExpression>();
	if (node->number < 0) {
		throw ParserException("Parameter numbers cannot be negative");
	}

	if (node->name) {
		// This is a named parameter, try to find an entry for it
		D_ASSERT(node->number == 0);
		int32_t index;
		if (GetNamedParam(node->name, index)) {
			// We've seen this named parameter before and assigned it an index!
			node->number = index;
		}
	}
	if (node->number == 0) {
		expr->parameter_nr = ParamCount() + 1;
		if (node->name && !HasNamedParameters() && ParamCount() != 0) {
			// This parameter is named, but there were other parameter before it, and they were not named
			throw NotImplementedException("Mixing positional and named parameters is not supported yet");
		}
		if (node->name) {
			D_ASSERT(!named_param_map.count(node->name));
			// Add it to the named parameter map so we can find it next time it's referenced
			SetNamedParam(node->name, expr->parameter_nr);
		}
	} else {
		if (!node->name && HasNamedParameters()) {
			// This parameter does not have a name, but the named param map is not empty
			throw NotImplementedException("Mixing positional and named parameters is not supported yet");
		}
		expr->parameter_nr = node->number;
	}
	SetParamCount(MaxValue<idx_t>(ParamCount(), expr->parameter_nr));
	return std::move(expr);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformPositionalReference(duckdb_libpgquery::PGPositionalReference *node) {
	if (node->position <= 0) {
		throw ParserException("Positional reference node needs to be >= 1");
	}
	auto result = make_unique<PositionalReferenceExpression>(node->position);
	result->query_location = node->location;
	return std::move(result);
}

} // namespace duckdb





namespace duckdb {

unique_ptr<ParsedExpression> Transformer::TransformSubquery(duckdb_libpgquery::PGSubLink *root) {
	D_ASSERT(root);
	auto subquery_expr = make_unique<SubqueryExpression>();

	subquery_expr->subquery = TransformSelect(root->subselect);
	D_ASSERT(subquery_expr->subquery);
	D_ASSERT(subquery_expr->subquery->node->GetSelectList().size() > 0);

	switch (root->subLinkType) {
	case duckdb_libpgquery::PG_EXISTS_SUBLINK: {
		subquery_expr->subquery_type = SubqueryType::EXISTS;
		break;
	}
	case duckdb_libpgquery::PG_ANY_SUBLINK:
	case duckdb_libpgquery::PG_ALL_SUBLINK: {
		// comparison with ANY() or ALL()
		subquery_expr->subquery_type = SubqueryType::ANY;
		subquery_expr->child = TransformExpression(root->testexpr);
		// get the operator name
		if (!root->operName) {
			// simple IN
			subquery_expr->comparison_type = ExpressionType::COMPARE_EQUAL;
		} else {
			auto operator_name =
			    string((reinterpret_cast<duckdb_libpgquery::PGValue *>(root->operName->head->data.ptr_value))->val.str);
			subquery_expr->comparison_type = OperatorToExpressionType(operator_name);
		}
		if (subquery_expr->comparison_type != ExpressionType::COMPARE_EQUAL &&
		    subquery_expr->comparison_type != ExpressionType::COMPARE_NOTEQUAL &&
		    subquery_expr->comparison_type != ExpressionType::COMPARE_GREATERTHAN &&
		    subquery_expr->comparison_type != ExpressionType::COMPARE_GREATERTHANOREQUALTO &&
		    subquery_expr->comparison_type != ExpressionType::COMPARE_LESSTHAN &&
		    subquery_expr->comparison_type != ExpressionType::COMPARE_LESSTHANOREQUALTO) {
			throw ParserException("ANY and ALL operators require one of =,<>,>,<,>=,<= comparisons!");
		}
		if (root->subLinkType == duckdb_libpgquery::PG_ALL_SUBLINK) {
			// ALL sublink is equivalent to NOT(ANY) with inverted comparison
			// e.g. [= ALL()] is equivalent to [NOT(<> ANY())]
			// first invert the comparison type
			subquery_expr->comparison_type = NegateComparisionExpression(subquery_expr->comparison_type);
			return make_unique<OperatorExpression>(ExpressionType::OPERATOR_NOT, std::move(subquery_expr));
		}
		break;
	}
	case duckdb_libpgquery::PG_EXPR_SUBLINK: {
		// return a single scalar value from the subquery
		// no child expression to compare to
		subquery_expr->subquery_type = SubqueryType::SCALAR;
		break;
	}
	case duckdb_libpgquery::PG_ARRAY_SUBLINK: {
		auto subquery_table_alias = "__subquery";
		auto subquery_column_alias = "__arr_element";

		// ARRAY expression
		// wrap subquery into "SELECT CASE WHEN ARRAY_AGG(i) IS NULL THEN [] ELSE ARRAY_AGG(i) END FROM (...) tbl(i)"
		auto select_node = make_unique<SelectNode>();

		// ARRAY_AGG(i)
		vector<unique_ptr<ParsedExpression>> children;
		children.push_back(
		    make_unique_base<ParsedExpression, ColumnRefExpression>(subquery_column_alias, subquery_table_alias));
		auto aggr = make_unique<FunctionExpression>("array_agg", std::move(children));
		// ARRAY_AGG(i) IS NULL
		auto agg_is_null = make_unique<OperatorExpression>(ExpressionType::OPERATOR_IS_NULL, aggr->Copy());
		// empty list
		vector<unique_ptr<ParsedExpression>> list_children;
		auto empty_list = make_unique<FunctionExpression>("list_value", std::move(list_children));
		// CASE
		auto case_expr = make_unique<CaseExpression>();
		CaseCheck check;
		check.when_expr = std::move(agg_is_null);
		check.then_expr = std::move(empty_list);
		case_expr->case_checks.push_back(std::move(check));
		case_expr->else_expr = std::move(aggr);

		select_node->select_list.push_back(std::move(case_expr));

		// FROM (...) tbl(i)
		auto child_subquery = make_unique<SubqueryRef>(std::move(subquery_expr->subquery), subquery_table_alias);
		child_subquery->column_name_alias.emplace_back(subquery_column_alias);
		select_node->from_table = std::move(child_subquery);

		auto new_subquery = make_unique<SelectStatement>();
		new_subquery->node = std::move(select_node);
		subquery_expr->subquery = std::move(new_subquery);

		subquery_expr->subquery_type = SubqueryType::SCALAR;
		break;
	}
	default:
		throw NotImplementedException("Subquery of type %d not implemented\n", (int)root->subLinkType);
	}
	subquery_expr->query_location = root->location;
	return std::move(subquery_expr);
}

} // namespace duckdb


namespace duckdb {

std::string Transformer::NodetypeToString(duckdb_libpgquery::PGNodeTag type) { // LCOV_EXCL_START
	switch (type) {
	case duckdb_libpgquery::T_PGInvalid:
		return "T_Invalid";
	case duckdb_libpgquery::T_PGIndexInfo:
		return "T_IndexInfo";
	case duckdb_libpgquery::T_PGExprContext:
		return "T_ExprContext";
	case duckdb_libpgquery::T_PGProjectionInfo:
		return "T_ProjectionInfo";
	case duckdb_libpgquery::T_PGJunkFilter:
		return "T_JunkFilter";
	case duckdb_libpgquery::T_PGResultRelInfo:
		return "T_ResultRelInfo";
	case duckdb_libpgquery::T_PGEState:
		return "T_EState";
	case duckdb_libpgquery::T_PGTupleTableSlot:
		return "T_TupleTableSlot";
	case duckdb_libpgquery::T_PGPlan:
		return "T_Plan";
	case duckdb_libpgquery::T_PGResult:
		return "T_Result";
	case duckdb_libpgquery::T_PGProjectSet:
		return "T_ProjectSet";
	case duckdb_libpgquery::T_PGModifyTable:
		return "T_ModifyTable";
	case duckdb_libpgquery::T_PGAppend:
		return "T_Append";
	case duckdb_libpgquery::T_PGMergeAppend:
		return "T_MergeAppend";
	case duckdb_libpgquery::T_PGRecursiveUnion:
		return "T_RecursiveUnion";
	case duckdb_libpgquery::T_PGBitmapAnd:
		return "T_BitmapAnd";
	case duckdb_libpgquery::T_PGBitmapOr:
		return "T_BitmapOr";
	case duckdb_libpgquery::T_PGScan:
		return "T_Scan";
	case duckdb_libpgquery::T_PGSeqScan:
		return "T_SeqScan";
	case duckdb_libpgquery::T_PGSampleScan:
		return "T_SampleScan";
	case duckdb_libpgquery::T_PGIndexScan:
		return "T_IndexScan";
	case duckdb_libpgquery::T_PGIndexOnlyScan:
		return "T_IndexOnlyScan";
	case duckdb_libpgquery::T_PGBitmapIndexScan:
		return "T_BitmapIndexScan";
	case duckdb_libpgquery::T_PGBitmapHeapScan:
		return "T_BitmapHeapScan";
	case duckdb_libpgquery::T_PGTidScan:
		return "T_TidScan";
	case duckdb_libpgquery::T_PGSubqueryScan:
		return "T_SubqueryScan";
	case duckdb_libpgquery::T_PGFunctionScan:
		return "T_FunctionScan";
	case duckdb_libpgquery::T_PGValuesScan:
		return "T_ValuesScan";
	case duckdb_libpgquery::T_PGTableFuncScan:
		return "T_TableFuncScan";
	case duckdb_libpgquery::T_PGCteScan:
		return "T_CteScan";
	case duckdb_libpgquery::T_PGNamedTuplestoreScan:
		return "T_NamedTuplestoreScan";
	case duckdb_libpgquery::T_PGWorkTableScan:
		return "T_WorkTableScan";
	case duckdb_libpgquery::T_PGForeignScan:
		return "T_ForeignScan";
	case duckdb_libpgquery::T_PGCustomScan:
		return "T_CustomScan";
	case duckdb_libpgquery::T_PGJoin:
		return "T_Join";
	case duckdb_libpgquery::T_PGNestLoop:
		return "T_NestLoop";
	case duckdb_libpgquery::T_PGMergeJoin:
		return "T_MergeJoin";
	case duckdb_libpgquery::T_PGHashJoin:
		return "T_HashJoin";
	case duckdb_libpgquery::T_PGMaterial:
		return "T_Material";
	case duckdb_libpgquery::T_PGSort:
		return "T_Sort";
	case duckdb_libpgquery::T_PGGroup:
		return "T_Group";
	case duckdb_libpgquery::T_PGAgg:
		return "T_Agg";
	case duckdb_libpgquery::T_PGWindowAgg:
		return "T_WindowAgg";
	case duckdb_libpgquery::T_PGUnique:
		return "T_Unique";
	case duckdb_libpgquery::T_PGGather:
		return "T_Gather";
	case duckdb_libpgquery::T_PGGatherMerge:
		return "T_GatherMerge";
	case duckdb_libpgquery::T_PGHash:
		return "T_Hash";
	case duckdb_libpgquery::T_PGSetOp:
		return "T_SetOp";
	case duckdb_libpgquery::T_PGLockRows:
		return "T_LockRows";
	case duckdb_libpgquery::T_PGLimit:
		return "T_Limit";
	case duckdb_libpgquery::T_PGNestLoopParam:
		return "T_NestLoopParam";
	case duckdb_libpgquery::T_PGPlanRowMark:
		return "T_PlanRowMark";
	case duckdb_libpgquery::T_PGPlanInvalItem:
		return "T_PlanInvalItem";
	case duckdb_libpgquery::T_PGPlanState:
		return "T_PlanState";
	case duckdb_libpgquery::T_PGResultState:
		return "T_ResultState";
	case duckdb_libpgquery::T_PGProjectSetState:
		return "T_ProjectSetState";
	case duckdb_libpgquery::T_PGModifyTableState:
		return "T_ModifyTableState";
	case duckdb_libpgquery::T_PGAppendState:
		return "T_AppendState";
	case duckdb_libpgquery::T_PGMergeAppendState:
		return "T_MergeAppendState";
	case duckdb_libpgquery::T_PGRecursiveUnionState:
		return "T_RecursiveUnionState";
	case duckdb_libpgquery::T_PGBitmapAndState:
		return "T_BitmapAndState";
	case duckdb_libpgquery::T_PGBitmapOrState:
		return "T_BitmapOrState";
	case duckdb_libpgquery::T_PGScanState:
		return "T_ScanState";
	case duckdb_libpgquery::T_PGSeqScanState:
		return "T_SeqScanState";
	case duckdb_libpgquery::T_PGSampleScanState:
		return "T_SampleScanState";
	case duckdb_libpgquery::T_PGIndexScanState:
		return "T_IndexScanState";
	case duckdb_libpgquery::T_PGIndexOnlyScanState:
		return "T_IndexOnlyScanState";
	case duckdb_libpgquery::T_PGBitmapIndexScanState:
		return "T_BitmapIndexScanState";
	case duckdb_libpgquery::T_PGBitmapHeapScanState:
		return "T_BitmapHeapScanState";
	case duckdb_libpgquery::T_PGTidScanState:
		return "T_TidScanState";
	case duckdb_libpgquery::T_PGSubqueryScanState:
		return "T_SubqueryScanState";
	case duckdb_libpgquery::T_PGFunctionScanState:
		return "T_FunctionScanState";
	case duckdb_libpgquery::T_PGTableFuncScanState:
		return "T_TableFuncScanState";
	case duckdb_libpgquery::T_PGValuesScanState:
		return "T_ValuesScanState";
	case duckdb_libpgquery::T_PGCteScanState:
		return "T_CteScanState";
	case duckdb_libpgquery::T_PGNamedTuplestoreScanState:
		return "T_NamedTuplestoreScanState";
	case duckdb_libpgquery::T_PGWorkTableScanState:
		return "T_WorkTableScanState";
	case duckdb_libpgquery::T_PGForeignScanState:
		return "T_ForeignScanState";
	case duckdb_libpgquery::T_PGCustomScanState:
		return "T_CustomScanState";
	case duckdb_libpgquery::T_PGJoinState:
		return "T_JoinState";
	case duckdb_libpgquery::T_PGNestLoopState:
		return "T_NestLoopState";
	case duckdb_libpgquery::T_PGMergeJoinState:
		return "T_MergeJoinState";
	case duckdb_libpgquery::T_PGHashJoinState:
		return "T_HashJoinState";
	case duckdb_libpgquery::T_PGMaterialState:
		return "T_MaterialState";
	case duckdb_libpgquery::T_PGSortState:
		return "T_SortState";
	case duckdb_libpgquery::T_PGGroupState:
		return "T_GroupState";
	case duckdb_libpgquery::T_PGAggState:
		return "T_AggState";
	case duckdb_libpgquery::T_PGWindowAggState:
		return "T_WindowAggState";
	case duckdb_libpgquery::T_PGUniqueState:
		return "T_UniqueState";
	case duckdb_libpgquery::T_PGGatherState:
		return "T_GatherState";
	case duckdb_libpgquery::T_PGGatherMergeState:
		return "T_GatherMergeState";
	case duckdb_libpgquery::T_PGHashState:
		return "T_HashState";
	case duckdb_libpgquery::T_PGSetOpState:
		return "T_SetOpState";
	case duckdb_libpgquery::T_PGLockRowsState:
		return "T_LockRowsState";
	case duckdb_libpgquery::T_PGLimitState:
		return "T_LimitState";
	case duckdb_libpgquery::T_PGAlias:
		return "T_Alias";
	case duckdb_libpgquery::T_PGRangeVar:
		return "T_RangeVar";
	case duckdb_libpgquery::T_PGTableFunc:
		return "T_TableFunc";
	case duckdb_libpgquery::T_PGExpr:
		return "T_Expr";
	case duckdb_libpgquery::T_PGVar:
		return "T_Var";
	case duckdb_libpgquery::T_PGConst:
		return "T_Const";
	case duckdb_libpgquery::T_PGParam:
		return "T_Param";
	case duckdb_libpgquery::T_PGAggref:
		return "T_Aggref";
	case duckdb_libpgquery::T_PGGroupingFunc:
		return "T_GroupingFunc";
	case duckdb_libpgquery::T_PGWindowFunc:
		return "T_WindowFunc";
	case duckdb_libpgquery::T_PGArrayRef:
		return "T_ArrayRef";
	case duckdb_libpgquery::T_PGFuncExpr:
		return "T_FuncExpr";
	case duckdb_libpgquery::T_PGNamedArgExpr:
		return "T_NamedArgExpr";
	case duckdb_libpgquery::T_PGOpExpr:
		return "T_OpExpr";
	case duckdb_libpgquery::T_PGDistinctExpr:
		return "T_DistinctExpr";
	case duckdb_libpgquery::T_PGNullIfExpr:
		return "T_NullIfExpr";
	case duckdb_libpgquery::T_PGScalarArrayOpExpr:
		return "T_ScalarArrayOpExpr";
	case duckdb_libpgquery::T_PGBoolExpr:
		return "T_BoolExpr";
	case duckdb_libpgquery::T_PGSubLink:
		return "T_SubLink";
	case duckdb_libpgquery::T_PGSubPlan:
		return "T_SubPlan";
	case duckdb_libpgquery::T_PGAlternativeSubPlan:
		return "T_AlternativeSubPlan";
	case duckdb_libpgquery::T_PGFieldSelect:
		return "T_FieldSelect";
	case duckdb_libpgquery::T_PGFieldStore:
		return "T_FieldStore";
	case duckdb_libpgquery::T_PGRelabelType:
		return "T_RelabelType";
	case duckdb_libpgquery::T_PGCoerceViaIO:
		return "T_CoerceViaIO";
	case duckdb_libpgquery::T_PGArrayCoerceExpr:
		return "T_ArrayCoerceExpr";
	case duckdb_libpgquery::T_PGConvertRowtypeExpr:
		return "T_ConvertRowtypeExpr";
	case duckdb_libpgquery::T_PGCollateExpr:
		return "T_CollateExpr";
	case duckdb_libpgquery::T_PGCaseExpr:
		return "T_CaseExpr";
	case duckdb_libpgquery::T_PGCaseWhen:
		return "T_CaseWhen";
	case duckdb_libpgquery::T_PGCaseTestExpr:
		return "T_CaseTestExpr";
	case duckdb_libpgquery::T_PGArrayExpr:
		return "T_ArrayExpr";
	case duckdb_libpgquery::T_PGRowExpr:
		return "T_RowExpr";
	case duckdb_libpgquery::T_PGRowCompareExpr:
		return "T_RowCompareExpr";
	case duckdb_libpgquery::T_PGCoalesceExpr:
		return "T_CoalesceExpr";
	case duckdb_libpgquery::T_PGMinMaxExpr:
		return "T_MinMaxExpr";
	case duckdb_libpgquery::T_PGSQLValueFunction:
		return "T_SQLValueFunction";
	case duckdb_libpgquery::T_PGXmlExpr:
		return "T_XmlExpr";
	case duckdb_libpgquery::T_PGNullTest:
		return "T_NullTest";
	case duckdb_libpgquery::T_PGBooleanTest:
		return "T_BooleanTest";
	case duckdb_libpgquery::T_PGCoerceToDomain:
		return "T_CoerceToDomain";
	case duckdb_libpgquery::T_PGCoerceToDomainValue:
		return "T_CoerceToDomainValue";
	case duckdb_libpgquery::T_PGSetToDefault:
		return "T_SetToDefault";
	case duckdb_libpgquery::T_PGCurrentOfExpr:
		return "T_CurrentOfExpr";
	case duckdb_libpgquery::T_PGNextValueExpr:
		return "T_NextValueExpr";
	case duckdb_libpgquery::T_PGInferenceElem:
		return "T_InferenceElem";
	case duckdb_libpgquery::T_PGTargetEntry:
		return "T_TargetEntry";
	case duckdb_libpgquery::T_PGRangeTblRef:
		return "T_RangeTblRef";
	case duckdb_libpgquery::T_PGJoinExpr:
		return "T_JoinExpr";
	case duckdb_libpgquery::T_PGFromExpr:
		return "T_FromExpr";
	case duckdb_libpgquery::T_PGOnConflictExpr:
		return "T_OnConflictExpr";
	case duckdb_libpgquery::T_PGIntoClause:
		return "T_IntoClause";
	case duckdb_libpgquery::T_PGExprState:
		return "T_ExprState";
	case duckdb_libpgquery::T_PGAggrefExprState:
		return "T_AggrefExprState";
	case duckdb_libpgquery::T_PGWindowFuncExprState:
		return "T_WindowFuncExprState";
	case duckdb_libpgquery::T_PGSetExprState:
		return "T_SetExprState";
	case duckdb_libpgquery::T_PGSubPlanState:
		return "T_SubPlanState";
	case duckdb_libpgquery::T_PGAlternativeSubPlanState:
		return "T_AlternativeSubPlanState";
	case duckdb_libpgquery::T_PGDomainConstraintState:
		return "T_DomainConstraintState";
	case duckdb_libpgquery::T_PGPlannerInfo:
		return "T_PlannerInfo";
	case duckdb_libpgquery::T_PGPlannerGlobal:
		return "T_PlannerGlobal";
	case duckdb_libpgquery::T_PGRelOptInfo:
		return "T_RelOptInfo";
	case duckdb_libpgquery::T_PGIndexOptInfo:
		return "T_IndexOptInfo";
	case duckdb_libpgquery::T_PGForeignKeyOptInfo:
		return "T_ForeignKeyOptInfo";
	case duckdb_libpgquery::T_PGParamPathInfo:
		return "T_ParamPathInfo";
	case duckdb_libpgquery::T_PGPath:
		return "T_Path";
	case duckdb_libpgquery::T_PGIndexPath:
		return "T_IndexPath";
	case duckdb_libpgquery::T_PGBitmapHeapPath:
		return "T_BitmapHeapPath";
	case duckdb_libpgquery::T_PGBitmapAndPath:
		return "T_BitmapAndPath";
	case duckdb_libpgquery::T_PGBitmapOrPath:
		return "T_BitmapOrPath";
	case duckdb_libpgquery::T_PGTidPath:
		return "T_TidPath";
	case duckdb_libpgquery::T_PGSubqueryScanPath:
		return "T_SubqueryScanPath";
	case duckdb_libpgquery::T_PGForeignPath:
		return "T_ForeignPath";
	case duckdb_libpgquery::T_PGCustomPath:
		return "T_CustomPath";
	case duckdb_libpgquery::T_PGNestPath:
		return "T_NestPath";
	case duckdb_libpgquery::T_PGMergePath:
		return "T_MergePath";
	case duckdb_libpgquery::T_PGHashPath:
		return "T_HashPath";
	case duckdb_libpgquery::T_PGAppendPath:
		return "T_AppendPath";
	case duckdb_libpgquery::T_PGMergeAppendPath:
		return "T_MergeAppendPath";
	case duckdb_libpgquery::T_PGResultPath:
		return "T_ResultPath";
	case duckdb_libpgquery::T_PGMaterialPath:
		return "T_MaterialPath";
	case duckdb_libpgquery::T_PGUniquePath:
		return "T_UniquePath";
	case duckdb_libpgquery::T_PGGatherPath:
		return "T_GatherPath";
	case duckdb_libpgquery::T_PGGatherMergePath:
		return "T_GatherMergePath";
	case duckdb_libpgquery::T_PGProjectionPath:
		return "T_ProjectionPath";
	case duckdb_libpgquery::T_PGProjectSetPath:
		return "T_ProjectSetPath";
	case duckdb_libpgquery::T_PGSortPath:
		return "T_SortPath";
	case duckdb_libpgquery::T_PGGroupPath:
		return "T_GroupPath";
	case duckdb_libpgquery::T_PGUpperUniquePath:
		return "T_UpperUniquePath";
	case duckdb_libpgquery::T_PGAggPath:
		return "T_AggPath";
	case duckdb_libpgquery::T_PGGroupingSetsPath:
		return "T_GroupingSetsPath";
	case duckdb_libpgquery::T_PGMinMaxAggPath:
		return "T_MinMaxAggPath";
	case duckdb_libpgquery::T_PGWindowAggPath:
		return "T_WindowAggPath";
	case duckdb_libpgquery::T_PGSetOpPath:
		return "T_SetOpPath";
	case duckdb_libpgquery::T_PGRecursiveUnionPath:
		return "T_RecursiveUnionPath";
	case duckdb_libpgquery::T_PGLockRowsPath:
		return "T_LockRowsPath";
	case duckdb_libpgquery::T_PGModifyTablePath:
		return "T_ModifyTablePath";
	case duckdb_libpgquery::T_PGLimitPath:
		return "T_LimitPath";
	case duckdb_libpgquery::T_PGEquivalenceClass:
		return "T_EquivalenceClass";
	case duckdb_libpgquery::T_PGEquivalenceMember:
		return "T_EquivalenceMember";
	case duckdb_libpgquery::T_PGPathKey:
		return "T_PathKey";
	case duckdb_libpgquery::T_PGPathTarget:
		return "T_PathTarget";
	case duckdb_libpgquery::T_PGRestrictInfo:
		return "T_RestrictInfo";
	case duckdb_libpgquery::T_PGPlaceHolderVar:
		return "T_PlaceHolderVar";
	case duckdb_libpgquery::T_PGSpecialJoinInfo:
		return "T_SpecialJoinInfo";
	case duckdb_libpgquery::T_PGAppendRelInfo:
		return "T_AppendRelInfo";
	case duckdb_libpgquery::T_PGPartitionedChildRelInfo:
		return "T_PartitionedChildRelInfo";
	case duckdb_libpgquery::T_PGPlaceHolderInfo:
		return "T_PlaceHolderInfo";
	case duckdb_libpgquery::T_PGMinMaxAggInfo:
		return "T_MinMaxAggInfo";
	case duckdb_libpgquery::T_PGPlannerParamItem:
		return "T_PlannerParamItem";
	case duckdb_libpgquery::T_PGRollupData:
		return "T_RollupData";
	case duckdb_libpgquery::T_PGGroupingSetData:
		return "T_GroupingSetData";
	case duckdb_libpgquery::T_PGStatisticExtInfo:
		return "T_StatisticExtInfo";
	case duckdb_libpgquery::T_PGMemoryContext:
		return "T_MemoryContext";
	case duckdb_libpgquery::T_PGAllocSetContext:
		return "T_AllocSetContext";
	case duckdb_libpgquery::T_PGSlabContext:
		return "T_SlabContext";
	case duckdb_libpgquery::T_PGValue:
		return "T_Value";
	case duckdb_libpgquery::T_PGInteger:
		return "T_Integer";
	case duckdb_libpgquery::T_PGFloat:
		return "T_Float";
	case duckdb_libpgquery::T_PGString:
		return "T_String";
	case duckdb_libpgquery::T_PGBitString:
		return "T_BitString";
	case duckdb_libpgquery::T_PGNull:
		return "T_Null";
	case duckdb_libpgquery::T_PGList:
		return "T_List";
	case duckdb_libpgquery::T_PGIntList:
		return "T_IntList";
	case duckdb_libpgquery::T_PGOidList:
		return "T_OidList";
	case duckdb_libpgquery::T_PGExtensibleNode:
		return "T_ExtensibleNode";
	case duckdb_libpgquery::T_PGRawStmt:
		return "T_RawStmt";
	case duckdb_libpgquery::T_PGQuery:
		return "T_Query";
	case duckdb_libpgquery::T_PGPlannedStmt:
		return "T_PlannedStmt";
	case duckdb_libpgquery::T_PGInsertStmt:
		return "T_InsertStmt";
	case duckdb_libpgquery::T_PGDeleteStmt:
		return "T_DeleteStmt";
	case duckdb_libpgquery::T_PGUpdateStmt:
		return "T_UpdateStmt";
	case duckdb_libpgquery::T_PGSelectStmt:
		return "T_SelectStmt";
	case duckdb_libpgquery::T_PGAlterTableStmt:
		return "T_AlterTableStmt";
	case duckdb_libpgquery::T_PGAlterTableCmd:
		return "T_AlterTableCmd";
	case duckdb_libpgquery::T_PGAlterDomainStmt:
		return "T_AlterDomainStmt";
	case duckdb_libpgquery::T_PGSetOperationStmt:
		return "T_SetOperationStmt";
	case duckdb_libpgquery::T_PGGrantStmt:
		return "T_GrantStmt";
	case duckdb_libpgquery::T_PGGrantRoleStmt:
		return "T_GrantRoleStmt";
	case duckdb_libpgquery::T_PGAlterDefaultPrivilegesStmt:
		return "T_AlterDefaultPrivilegesStmt";
	case duckdb_libpgquery::T_PGClosePortalStmt:
		return "T_ClosePortalStmt";
	case duckdb_libpgquery::T_PGClusterStmt:
		return "T_ClusterStmt";
	case duckdb_libpgquery::T_PGCopyStmt:
		return "T_CopyStmt";
	case duckdb_libpgquery::T_PGCreateStmt:
		return "T_CreateStmt";
	case duckdb_libpgquery::T_PGDefineStmt:
		return "T_DefineStmt";
	case duckdb_libpgquery::T_PGDropStmt:
		return "T_DropStmt";
	case duckdb_libpgquery::T_PGTruncateStmt:
		return "T_TruncateStmt";
	case duckdb_libpgquery::T_PGCommentStmt:
		return "T_CommentStmt";
	case duckdb_libpgquery::T_PGFetchStmt:
		return "T_FetchStmt";
	case duckdb_libpgquery::T_PGIndexStmt:
		return "T_IndexStmt";
	case duckdb_libpgquery::T_PGCreateFunctionStmt:
		return "T_CreateFunctionStmt";
	case duckdb_libpgquery::T_PGAlterFunctionStmt:
		return "T_AlterFunctionStmt";
	case duckdb_libpgquery::T_PGDoStmt:
		return "T_DoStmt";
	case duckdb_libpgquery::T_PGRenameStmt:
		return "T_RenameStmt";
	case duckdb_libpgquery::T_PGRuleStmt:
		return "T_RuleStmt";
	case duckdb_libpgquery::T_PGNotifyStmt:
		return "T_NotifyStmt";
	case duckdb_libpgquery::T_PGListenStmt:
		return "T_ListenStmt";
	case duckdb_libpgquery::T_PGUnlistenStmt:
		return "T_UnlistenStmt";
	case duckdb_libpgquery::T_PGTransactionStmt:
		return "T_TransactionStmt";
	case duckdb_libpgquery::T_PGViewStmt:
		return "T_ViewStmt";
	case duckdb_libpgquery::T_PGLoadStmt:
		return "T_LoadStmt";
	case duckdb_libpgquery::T_PGCreateDomainStmt:
		return "T_CreateDomainStmt";
	case duckdb_libpgquery::T_PGCreatedbStmt:
		return "T_CreatedbStmt";
	case duckdb_libpgquery::T_PGDropdbStmt:
		return "T_DropdbStmt";
	case duckdb_libpgquery::T_PGVacuumStmt:
		return "T_VacuumStmt";
	case duckdb_libpgquery::T_PGExplainStmt:
		return "T_ExplainStmt";
	case duckdb_libpgquery::T_PGCreateTableAsStmt:
		return "T_CreateTableAsStmt";
	case duckdb_libpgquery::T_PGCreateSeqStmt:
		return "T_CreateSeqStmt";
	case duckdb_libpgquery::T_PGAlterSeqStmt:
		return "T_AlterSeqStmt";
	case duckdb_libpgquery::T_PGVariableSetStmt:
		return "T_VariableSetStmt";
	case duckdb_libpgquery::T_PGVariableShowStmt:
		return "T_VariableShowStmt";
	case duckdb_libpgquery::T_PGVariableShowSelectStmt:
		return "T_VariableShowSelectStmt";
	case duckdb_libpgquery::T_PGDiscardStmt:
		return "T_DiscardStmt";
	case duckdb_libpgquery::T_PGCreateTrigStmt:
		return "T_CreateTrigStmt";
	case duckdb_libpgquery::T_PGCreatePLangStmt:
		return "T_CreatePLangStmt";
	case duckdb_libpgquery::T_PGCreateRoleStmt:
		return "T_CreateRoleStmt";
	case duckdb_libpgquery::T_PGAlterRoleStmt:
		return "T_AlterRoleStmt";
	case duckdb_libpgquery::T_PGDropRoleStmt:
		return "T_DropRoleStmt";
	case duckdb_libpgquery::T_PGLockStmt:
		return "T_LockStmt";
	case duckdb_libpgquery::T_PGConstraintsSetStmt:
		return "T_ConstraintsSetStmt";
	case duckdb_libpgquery::T_PGReindexStmt:
		return "T_ReindexStmt";
	case duckdb_libpgquery::T_PGCheckPointStmt:
		return "T_CheckPointStmt";
	case duckdb_libpgquery::T_PGCreateSchemaStmt:
		return "T_CreateSchemaStmt";
	case duckdb_libpgquery::T_PGAlterDatabaseStmt:
		return "T_AlterDatabaseStmt";
	case duckdb_libpgquery::T_PGAlterDatabaseSetStmt:
		return "T_AlterDatabaseSetStmt";
	case duckdb_libpgquery::T_PGAlterRoleSetStmt:
		return "T_AlterRoleSetStmt";
	case duckdb_libpgquery::T_PGCreateConversionStmt:
		return "T_CreateConversionStmt";
	case duckdb_libpgquery::T_PGCreateCastStmt:
		return "T_CreateCastStmt";
	case duckdb_libpgquery::T_PGCreateOpClassStmt:
		return "T_CreateOpClassStmt";
	case duckdb_libpgquery::T_PGCreateOpFamilyStmt:
		return "T_CreateOpFamilyStmt";
	case duckdb_libpgquery::T_PGAlterOpFamilyStmt:
		return "T_AlterOpFamilyStmt";
	case duckdb_libpgquery::T_PGPrepareStmt:
		return "T_PrepareStmt";
	case duckdb_libpgquery::T_PGExecuteStmt:
		return "T_ExecuteStmt";
	case duckdb_libpgquery::T_PGCallStmt:
		return "T_CallStmt";
	case duckdb_libpgquery::T_PGDeallocateStmt:
		return "T_DeallocateStmt";
	case duckdb_libpgquery::T_PGDeclareCursorStmt:
		return "T_DeclareCursorStmt";
	case duckdb_libpgquery::T_PGCreateTableSpaceStmt:
		return "T_CreateTableSpaceStmt";
	case duckdb_libpgquery::T_PGDropTableSpaceStmt:
		return "T_DropTableSpaceStmt";
	case duckdb_libpgquery::T_PGAlterObjectDependsStmt:
		return "T_AlterObjectDependsStmt";
	case duckdb_libpgquery::T_PGAlterObjectSchemaStmt:
		return "T_AlterObjectSchemaStmt";
	case duckdb_libpgquery::T_PGAlterOwnerStmt:
		return "T_AlterOwnerStmt";
	case duckdb_libpgquery::T_PGAlterOperatorStmt:
		return "T_AlterOperatorStmt";
	case duckdb_libpgquery::T_PGDropOwnedStmt:
		return "T_DropOwnedStmt";
	case duckdb_libpgquery::T_PGReassignOwnedStmt:
		return "T_ReassignOwnedStmt";
	case duckdb_libpgquery::T_PGCompositeTypeStmt:
		return "T_CompositeTypeStmt";
	case duckdb_libpgquery::T_PGCreateTypeStmt:
		return "T_CreateTypeStmt";
	case duckdb_libpgquery::T_PGCreateRangeStmt:
		return "T_CreateRangeStmt";
	case duckdb_libpgquery::T_PGAlterEnumStmt:
		return "T_AlterEnumStmt";
	case duckdb_libpgquery::T_PGAlterTSDictionaryStmt:
		return "T_AlterTSDictionaryStmt";
	case duckdb_libpgquery::T_PGAlterTSConfigurationStmt:
		return "T_AlterTSConfigurationStmt";
	case duckdb_libpgquery::T_PGCreateFdwStmt:
		return "T_CreateFdwStmt";
	case duckdb_libpgquery::T_PGAlterFdwStmt:
		return "T_AlterFdwStmt";
	case duckdb_libpgquery::T_PGCreateForeignServerStmt:
		return "T_CreateForeignServerStmt";
	case duckdb_libpgquery::T_PGAlterForeignServerStmt:
		return "T_AlterForeignServerStmt";
	case duckdb_libpgquery::T_PGCreateUserMappingStmt:
		return "T_CreateUserMappingStmt";
	case duckdb_libpgquery::T_PGAlterUserMappingStmt:
		return "T_AlterUserMappingStmt";
	case duckdb_libpgquery::T_PGDropUserMappingStmt:
		return "T_DropUserMappingStmt";
	case duckdb_libpgquery::T_PGAlterTableSpaceOptionsStmt:
		return "T_AlterTableSpaceOptionsStmt";
	case duckdb_libpgquery::T_PGAlterTableMoveAllStmt:
		return "T_AlterTableMoveAllStmt";
	case duckdb_libpgquery::T_PGSecLabelStmt:
		return "T_SecLabelStmt";
	case duckdb_libpgquery::T_PGCreateForeignTableStmt:
		return "T_CreateForeignTableStmt";
	case duckdb_libpgquery::T_PGImportForeignSchemaStmt:
		return "T_ImportForeignSchemaStmt";
	case duckdb_libpgquery::T_PGCreateExtensionStmt:
		return "T_CreateExtensionStmt";
	case duckdb_libpgquery::T_PGAlterExtensionStmt:
		return "T_AlterExtensionStmt";
	case duckdb_libpgquery::T_PGAlterExtensionContentsStmt:
		return "T_AlterExtensionContentsStmt";
	case duckdb_libpgquery::T_PGCreateEventTrigStmt:
		return "T_CreateEventTrigStmt";
	case duckdb_libpgquery::T_PGAlterEventTrigStmt:
		return "T_AlterEventTrigStmt";
	case duckdb_libpgquery::T_PGRefreshMatViewStmt:
		return "T_RefreshMatViewStmt";
	case duckdb_libpgquery::T_PGReplicaIdentityStmt:
		return "T_ReplicaIdentityStmt";
	case duckdb_libpgquery::T_PGAlterSystemStmt:
		return "T_AlterSystemStmt";
	case duckdb_libpgquery::T_PGCreatePolicyStmt:
		return "T_CreatePolicyStmt";
	case duckdb_libpgquery::T_PGAlterPolicyStmt:
		return "T_AlterPolicyStmt";
	case duckdb_libpgquery::T_PGCreateTransformStmt:
		return "T_CreateTransformStmt";
	case duckdb_libpgquery::T_PGCreateAmStmt:
		return "T_CreateAmStmt";
	case duckdb_libpgquery::T_PGCreatePublicationStmt:
		return "T_CreatePublicationStmt";
	case duckdb_libpgquery::T_PGAlterPublicationStmt:
		return "T_AlterPublicationStmt";
	case duckdb_libpgquery::T_PGCreateSubscriptionStmt:
		return "T_CreateSubscriptionStmt";
	case duckdb_libpgquery::T_PGAlterSubscriptionStmt:
		return "T_AlterSubscriptionStmt";
	case duckdb_libpgquery::T_PGDropSubscriptionStmt:
		return "T_DropSubscriptionStmt";
	case duckdb_libpgquery::T_PGCreateStatsStmt:
		return "T_CreateStatsStmt";
	case duckdb_libpgquery::T_PGAlterCollationStmt:
		return "T_AlterCollationStmt";
	case duckdb_libpgquery::T_PGAExpr:
		return "TAExpr";
	case duckdb_libpgquery::T_PGColumnRef:
		return "T_ColumnRef";
	case duckdb_libpgquery::T_PGParamRef:
		return "T_ParamRef";
	case duckdb_libpgquery::T_PGAConst:
		return "TAConst";
	case duckdb_libpgquery::T_PGFuncCall:
		return "T_FuncCall";
	case duckdb_libpgquery::T_PGAStar:
		return "TAStar";
	case duckdb_libpgquery::T_PGAIndices:
		return "TAIndices";
	case duckdb_libpgquery::T_PGAIndirection:
		return "TAIndirection";
	case duckdb_libpgquery::T_PGAArrayExpr:
		return "TAArrayExpr";
	case duckdb_libpgquery::T_PGResTarget:
		return "T_ResTarget";
	case duckdb_libpgquery::T_PGMultiAssignRef:
		return "T_MultiAssignRef";
	case duckdb_libpgquery::T_PGTypeCast:
		return "T_TypeCast";
	case duckdb_libpgquery::T_PGCollateClause:
		return "T_CollateClause";
	case duckdb_libpgquery::T_PGSortBy:
		return "T_SortBy";
	case duckdb_libpgquery::T_PGWindowDef:
		return "T_WindowDef";
	case duckdb_libpgquery::T_PGRangeSubselect:
		return "T_RangeSubselect";
	case duckdb_libpgquery::T_PGRangeFunction:
		return "T_RangeFunction";
	case duckdb_libpgquery::T_PGRangeTableSample:
		return "T_RangeTableSample";
	case duckdb_libpgquery::T_PGRangeTableFunc:
		return "T_RangeTableFunc";
	case duckdb_libpgquery::T_PGRangeTableFuncCol:
		return "T_RangeTableFuncCol";
	case duckdb_libpgquery::T_PGTypeName:
		return "T_TypeName";
	case duckdb_libpgquery::T_PGColumnDef:
		return "T_ColumnDef";
	case duckdb_libpgquery::T_PGIndexElem:
		return "T_IndexElem";
	case duckdb_libpgquery::T_PGConstraint:
		return "T_Constraint";
	case duckdb_libpgquery::T_PGDefElem:
		return "T_DefElem";
	case duckdb_libpgquery::T_PGRangeTblEntry:
		return "T_RangeTblEntry";
	case duckdb_libpgquery::T_PGRangeTblFunction:
		return "T_RangeTblFunction";
	case duckdb_libpgquery::T_PGTableSampleClause:
		return "T_TableSampleClause";
	case duckdb_libpgquery::T_PGWithCheckOption:
		return "T_WithCheckOption";
	case duckdb_libpgquery::T_PGSortGroupClause:
		return "T_SortGroupClause";
	case duckdb_libpgquery::T_PGGroupingSet:
		return "T_GroupingSet";
	case duckdb_libpgquery::T_PGWindowClause:
		return "T_WindowClause";
	case duckdb_libpgquery::T_PGObjectWithArgs:
		return "T_ObjectWithArgs";
	case duckdb_libpgquery::T_PGAccessPriv:
		return "T_AccessPriv";
	case duckdb_libpgquery::T_PGCreateOpClassItem:
		return "T_CreateOpClassItem";
	case duckdb_libpgquery::T_PGTableLikeClause:
		return "T_TableLikeClause";
	case duckdb_libpgquery::T_PGFunctionParameter:
		return "T_FunctionParameter";
	case duckdb_libpgquery::T_PGLockingClause:
		return "T_LockingClause";
	case duckdb_libpgquery::T_PGRowMarkClause:
		return "T_RowMarkClause";
	case duckdb_libpgquery::T_PGXmlSerialize:
		return "T_XmlSerialize";
	case duckdb_libpgquery::T_PGWithClause:
		return "T_WithClause";
	case duckdb_libpgquery::T_PGInferClause:
		return "T_InferClause";
	case duckdb_libpgquery::T_PGOnConflictClause:
		return "T_OnConflictClause";
	case duckdb_libpgquery::T_PGCommonTableExpr:
		return "T_CommonTableExpr";
	case duckdb_libpgquery::T_PGRoleSpec:
		return "T_RoleSpec";
	case duckdb_libpgquery::T_PGTriggerTransition:
		return "T_TriggerTransition";
	case duckdb_libpgquery::T_PGPartitionElem:
		return "T_PartitionElem";
	case duckdb_libpgquery::T_PGPartitionSpec:
		return "T_PartitionSpec";
	case duckdb_libpgquery::T_PGPartitionBoundSpec:
		return "T_PartitionBoundSpec";
	case duckdb_libpgquery::T_PGPartitionRangeDatum:
		return "T_PartitionRangeDatum";
	case duckdb_libpgquery::T_PGPartitionCmd:
		return "T_PartitionCmd";
	case duckdb_libpgquery::T_PGIdentifySystemCmd:
		return "T_IdentifySystemCmd";
	case duckdb_libpgquery::T_PGBaseBackupCmd:
		return "T_BaseBackupCmd";
	case duckdb_libpgquery::T_PGCreateReplicationSlotCmd:
		return "T_CreateReplicationSlotCmd";
	case duckdb_libpgquery::T_PGDropReplicationSlotCmd:
		return "T_DropReplicationSlotCmd";
	case duckdb_libpgquery::T_PGStartReplicationCmd:
		return "T_StartReplicationCmd";
	case duckdb_libpgquery::T_PGTimeLineHistoryCmd:
		return "T_TimeLineHistoryCmd";
	case duckdb_libpgquery::T_PGSQLCmd:
		return "T_SQLCmd";
	case duckdb_libpgquery::T_PGTriggerData:
		return "T_TriggerData";
	case duckdb_libpgquery::T_PGEventTriggerData:
		return "T_EventTriggerData";
	case duckdb_libpgquery::T_PGReturnSetInfo:
		return "T_ReturnSetInfo";
	case duckdb_libpgquery::T_PGWindowObjectData:
		return "T_WindowObjectData";
	case duckdb_libpgquery::T_PGTIDBitmap:
		return "T_TIDBitmap";
	case duckdb_libpgquery::T_PGInlineCodeBlock:
		return "T_InlineCodeBlock";
	case duckdb_libpgquery::T_PGFdwRoutine:
		return "T_FdwRoutine";
	case duckdb_libpgquery::T_PGIndexAmRoutine:
		return "T_IndexAmRoutine";
	case duckdb_libpgquery::T_PGTsmRoutine:
		return "T_TsmRoutine";
	case duckdb_libpgquery::T_PGForeignKeyCacheInfo:
		return "T_ForeignKeyCacheInfo";
	case duckdb_libpgquery::T_PGAttachStmt:
		return "T_PGAttachStmt";
	case duckdb_libpgquery::T_PGUseStmt:
		return "T_PGUseStmt";
	case duckdb_libpgquery::T_PGCreateDatabaseStmt:
		return "T_PGCreateDatabaseStmt";
	default:
		return "(UNKNOWN)";
	}
} // LCOV_EXCL_STOP

} // namespace duckdb


namespace duckdb {

string Transformer::TransformAlias(duckdb_libpgquery::PGAlias *root, vector<string> &column_name_alias) {
	if (!root) {
		return "";
	}
	if (root->colnames) {
		for (auto node = root->colnames->head; node != nullptr; node = node->next) {
			column_name_alias.emplace_back(
			    reinterpret_cast<duckdb_libpgquery::PGValue *>(node->data.ptr_value)->val.str);
		}
	}
	return root->aliasname;
}

} // namespace duckdb






namespace duckdb {

void Transformer::TransformCTE(duckdb_libpgquery::PGWithClause *de_with_clause, CommonTableExpressionMap &cte_map) {
	// TODO: might need to update in case of future lawsuit
	D_ASSERT(de_with_clause);

	D_ASSERT(de_with_clause->ctes);
	for (auto cte_ele = de_with_clause->ctes->head; cte_ele != nullptr; cte_ele = cte_ele->next) {
		auto info = make_unique<CommonTableExpressionInfo>();

		auto cte = reinterpret_cast<duckdb_libpgquery::PGCommonTableExpr *>(cte_ele->data.ptr_value);
		if (cte->aliascolnames) {
			for (auto node = cte->aliascolnames->head; node != nullptr; node = node->next) {
				info->aliases.emplace_back(
				    reinterpret_cast<duckdb_libpgquery::PGValue *>(node->data.ptr_value)->val.str);
			}
		}
		// lets throw some errors on unsupported features early
		if (cte->ctecolnames) {
			throw NotImplementedException("Column name setting not supported in CTEs");
		}
		if (cte->ctecoltypes) {
			throw NotImplementedException("Column type setting not supported in CTEs");
		}
		if (cte->ctecoltypmods) {
			throw NotImplementedException("Column type modification not supported in CTEs");
		}
		if (cte->ctecolcollations) {
			throw NotImplementedException("CTE collations not supported");
		}
		// we need a query
		if (!cte->ctequery || cte->ctequery->type != duckdb_libpgquery::T_PGSelectStmt) {
			throw NotImplementedException("A CTE needs a SELECT");
		}

		// CTE transformation can either result in inlining for non recursive CTEs, or in recursive CTE bindings
		// otherwise.
		if (cte->cterecursive || de_with_clause->recursive) {
			info->query = TransformRecursiveCTE(cte, *info);
		} else {
			Transformer cte_transformer(this);
			info->query = cte_transformer.TransformSelect(cte->ctequery);
		}
		D_ASSERT(info->query);
		auto cte_name = string(cte->ctename);

		auto it = cte_map.map.find(cte_name);
		if (it != cte_map.map.end()) {
			// can't have two CTEs with same name
			throw ParserException("Duplicate CTE name \"%s\"", cte_name);
		}
		cte_map.map[cte_name] = std::move(info);
	}
}

unique_ptr<SelectStatement> Transformer::TransformRecursiveCTE(duckdb_libpgquery::PGCommonTableExpr *cte,
                                                               CommonTableExpressionInfo &info) {
	auto stmt = (duckdb_libpgquery::PGSelectStmt *)cte->ctequery;

	unique_ptr<SelectStatement> select;
	switch (stmt->op) {
	case duckdb_libpgquery::PG_SETOP_UNION:
	case duckdb_libpgquery::PG_SETOP_EXCEPT:
	case duckdb_libpgquery::PG_SETOP_INTERSECT: {
		select = make_unique<SelectStatement>();
		select->node = make_unique_base<QueryNode, RecursiveCTENode>();
		auto result = (RecursiveCTENode *)select->node.get();
		result->ctename = string(cte->ctename);
		result->union_all = stmt->all;
		result->left = TransformSelectNode(stmt->larg);
		result->right = TransformSelectNode(stmt->rarg);
		result->aliases = info.aliases;

		D_ASSERT(result->left);
		D_ASSERT(result->right);

		if (stmt->op != duckdb_libpgquery::PG_SETOP_UNION) {
			throw ParserException("Unsupported setop type for recursive CTE: only UNION or UNION ALL are supported");
		}
		break;
	}
	default:
		// This CTE is not recursive. Fallback to regular query transformation.
		return TransformSelect(cte->ctequery);
	}

	if (stmt->limitCount || stmt->limitOffset) {
		throw ParserException("LIMIT or OFFSET in a recursive query is not allowed");
	}
	if (stmt->sortClause) {
		throw ParserException("ORDER BY in a recursive query is not allowed");
	}
	return select;
}

} // namespace duckdb






namespace duckdb {

static void CheckGroupingSetMax(idx_t count) {
	static constexpr const idx_t MAX_GROUPING_SETS = 65535;
	if (count > MAX_GROUPING_SETS) {
		throw ParserException("Maximum grouping set count of %d exceeded", MAX_GROUPING_SETS);
	}
}

static void CheckGroupingSetCubes(idx_t current_count, idx_t cube_count) {
	idx_t combinations = 1;
	for (idx_t i = 0; i < cube_count; i++) {
		combinations *= 2;
		CheckGroupingSetMax(current_count + combinations);
	}
}

struct GroupingExpressionMap {
	expression_map_t<idx_t> map;
};

static GroupingSet VectorToGroupingSet(vector<idx_t> &indexes) {
	GroupingSet result;
	for (idx_t i = 0; i < indexes.size(); i++) {
		result.insert(indexes[i]);
	}
	return result;
}

static void MergeGroupingSet(GroupingSet &result, GroupingSet &other) {
	CheckGroupingSetMax(result.size() + other.size());
	result.insert(other.begin(), other.end());
}

void Transformer::AddGroupByExpression(unique_ptr<ParsedExpression> expression, GroupingExpressionMap &map,
                                       GroupByNode &result, vector<idx_t> &result_set) {
	if (expression->type == ExpressionType::FUNCTION) {
		auto &func = (FunctionExpression &)*expression;
		if (func.function_name == "row") {
			for (auto &child : func.children) {
				AddGroupByExpression(std::move(child), map, result, result_set);
			}
			return;
		}
	}
	auto entry = map.map.find(expression.get());
	idx_t result_idx;
	if (entry == map.map.end()) {
		result_idx = result.group_expressions.size();
		map.map[expression.get()] = result_idx;
		result.group_expressions.push_back(std::move(expression));
	} else {
		result_idx = entry->second;
	}
	result_set.push_back(result_idx);
}

static void AddCubeSets(const GroupingSet &current_set, vector<GroupingSet> &result_set,
                        vector<GroupingSet> &result_sets, idx_t start_idx = 0) {
	CheckGroupingSetMax(result_sets.size());
	result_sets.push_back(current_set);
	for (idx_t k = start_idx; k < result_set.size(); k++) {
		auto child_set = current_set;
		MergeGroupingSet(child_set, result_set[k]);
		AddCubeSets(child_set, result_set, result_sets, k + 1);
	}
}

void Transformer::TransformGroupByExpression(duckdb_libpgquery::PGNode *n, GroupingExpressionMap &map,
                                             GroupByNode &result, vector<idx_t> &indexes) {
	auto expression = TransformExpression(n);
	AddGroupByExpression(std::move(expression), map, result, indexes);
}

// If one GROUPING SETS clause is nested inside another,
// the effect is the same as if all the elements of the inner clause had been written directly in the outer clause.
void Transformer::TransformGroupByNode(duckdb_libpgquery::PGNode *n, GroupingExpressionMap &map, SelectNode &result,
                                       vector<GroupingSet> &result_sets) {
	if (n->type == duckdb_libpgquery::T_PGGroupingSet) {
		auto grouping_set = (duckdb_libpgquery::PGGroupingSet *)n;
		switch (grouping_set->kind) {
		case duckdb_libpgquery::GROUPING_SET_EMPTY:
			result_sets.emplace_back();
			break;
		case duckdb_libpgquery::GROUPING_SET_ALL: {
			result.aggregate_handling = AggregateHandling::FORCE_AGGREGATES;
			break;
		}
		case duckdb_libpgquery::GROUPING_SET_SETS: {
			for (auto node = grouping_set->content->head; node; node = node->next) {
				auto pg_node = (duckdb_libpgquery::PGNode *)node->data.ptr_value;
				TransformGroupByNode(pg_node, map, result, result_sets);
			}
			break;
		}
		case duckdb_libpgquery::GROUPING_SET_ROLLUP: {
			vector<GroupingSet> rollup_sets;
			for (auto node = grouping_set->content->head; node; node = node->next) {
				auto pg_node = (duckdb_libpgquery::PGNode *)node->data.ptr_value;
				vector<idx_t> rollup_set;
				TransformGroupByExpression(pg_node, map, result.groups, rollup_set);
				rollup_sets.push_back(VectorToGroupingSet(rollup_set));
			}
			// generate the subsets of the rollup set and add them to the grouping sets
			GroupingSet current_set;
			result_sets.push_back(current_set);
			for (idx_t i = 0; i < rollup_sets.size(); i++) {
				MergeGroupingSet(current_set, rollup_sets[i]);
				result_sets.push_back(current_set);
			}
			break;
		}
		case duckdb_libpgquery::GROUPING_SET_CUBE: {
			vector<GroupingSet> cube_sets;
			for (auto node = grouping_set->content->head; node; node = node->next) {
				auto pg_node = (duckdb_libpgquery::PGNode *)node->data.ptr_value;
				vector<idx_t> cube_set;
				TransformGroupByExpression(pg_node, map, result.groups, cube_set);
				cube_sets.push_back(VectorToGroupingSet(cube_set));
			}
			// generate the subsets of the rollup set and add them to the grouping sets
			CheckGroupingSetCubes(result_sets.size(), cube_sets.size());

			GroupingSet current_set;
			AddCubeSets(current_set, cube_sets, result_sets, 0);
			break;
		}
		default:
			throw InternalException("Unsupported GROUPING SET type %d", grouping_set->kind);
		}
	} else {
		vector<idx_t> indexes;
		TransformGroupByExpression(n, map, result.groups, indexes);
		result_sets.push_back(VectorToGroupingSet(indexes));
	}
}

// If multiple grouping items are specified in a single GROUP BY clause,
// then the final list of grouping sets is the cross product of the individual items.
bool Transformer::TransformGroupBy(duckdb_libpgquery::PGList *group, SelectNode &select_node) {
	if (!group) {
		return false;
	}
	auto &result = select_node.groups;
	GroupingExpressionMap map;
	for (auto node = group->head; node != nullptr; node = node->next) {
		auto n = reinterpret_cast<duckdb_libpgquery::PGNode *>(node->data.ptr_value);
		vector<GroupingSet> result_sets;
		TransformGroupByNode(n, map, select_node, result_sets);
		CheckGroupingSetMax(result_sets.size());
		if (result.grouping_sets.empty()) {
			// no grouping sets yet: use the current set of grouping sets
			result.grouping_sets = std::move(result_sets);
		} else {
			// compute the cross product
			vector<GroupingSet> new_sets;
			idx_t grouping_set_count = result.grouping_sets.size() * result_sets.size();
			CheckGroupingSetMax(grouping_set_count);
			new_sets.reserve(grouping_set_count);
			for (idx_t current_idx = 0; current_idx < result.grouping_sets.size(); current_idx++) {
				auto &current_set = result.grouping_sets[current_idx];
				for (idx_t new_idx = 0; new_idx < result_sets.size(); new_idx++) {
					auto &new_set = result_sets[new_idx];
					GroupingSet set;
					set.insert(current_set.begin(), current_set.end());
					set.insert(new_set.begin(), new_set.end());
					new_sets.push_back(std::move(set));
				}
			}
			result.grouping_sets = std::move(new_sets);
		}
	}
	return true;
}

} // namespace duckdb





namespace duckdb {

bool Transformer::TransformOrderBy(duckdb_libpgquery::PGList *order, vector<OrderByNode> &result) {
	if (!order) {
		return false;
	}

	for (auto node = order->head; node != nullptr; node = node->next) {
		auto temp = reinterpret_cast<duckdb_libpgquery::PGNode *>(node->data.ptr_value);
		if (temp->type == duckdb_libpgquery::T_PGSortBy) {
			OrderType type;
			OrderByNullType null_order;
			auto sort = reinterpret_cast<duckdb_libpgquery::PGSortBy *>(temp);
			auto target = sort->node;
			if (sort->sortby_dir == duckdb_libpgquery::PG_SORTBY_DEFAULT) {
				type = OrderType::ORDER_DEFAULT;
			} else if (sort->sortby_dir == duckdb_libpgquery::PG_SORTBY_ASC) {
				type = OrderType::ASCENDING;
			} else if (sort->sortby_dir == duckdb_libpgquery::PG_SORTBY_DESC) {
				type = OrderType::DESCENDING;
			} else {
				throw NotImplementedException("Unimplemented order by type");
			}
			if (sort->sortby_nulls == duckdb_libpgquery::PG_SORTBY_NULLS_DEFAULT) {
				null_order = OrderByNullType::ORDER_DEFAULT;
			} else if (sort->sortby_nulls == duckdb_libpgquery::PG_SORTBY_NULLS_FIRST) {
				null_order = OrderByNullType::NULLS_FIRST;
			} else if (sort->sortby_nulls == duckdb_libpgquery::PG_SORTBY_NULLS_LAST) {
				null_order = OrderByNullType::NULLS_LAST;
			} else {
				throw NotImplementedException("Unimplemented order by type");
			}
			auto order_expression = TransformExpression(target);
			if (order_expression->GetExpressionClass() == ExpressionClass::STAR) {
				auto &star_expr = (StarExpression &)*order_expression;
				D_ASSERT(star_expr.relation_name.empty());
				if (star_expr.columns) {
					throw ParserException("COLUMNS expr is not supported in ORDER BY");
				}
			}
			result.emplace_back(type, null_order, std::move(order_expression));
		} else {
			throw NotImplementedException("ORDER BY list member type %d\n", temp->type);
		}
	}
	return true;
}

} // namespace duckdb





namespace duckdb {

static SampleMethod GetSampleMethod(const string &method) {
	auto lmethod = StringUtil::Lower(method);
	if (lmethod == "system") {
		return SampleMethod::SYSTEM_SAMPLE;
	} else if (lmethod == "bernoulli") {
		return SampleMethod::BERNOULLI_SAMPLE;
	} else if (lmethod == "reservoir") {
		return SampleMethod::RESERVOIR_SAMPLE;
	} else {
		throw ParserException("Unrecognized sampling method %s, expected system, bernoulli or reservoir", method);
	}
}

unique_ptr<SampleOptions> Transformer::TransformSampleOptions(duckdb_libpgquery::PGNode *options) {
	if (!options) {
		return nullptr;
	}
	auto result = make_unique<SampleOptions>();
	auto &sample_options = (duckdb_libpgquery::PGSampleOptions &)*options;
	auto &sample_size = (duckdb_libpgquery::PGSampleSize &)*sample_options.sample_size;
	auto sample_value = TransformValue(sample_size.sample_size)->value;
	result->is_percentage = sample_size.is_percentage;
	if (sample_size.is_percentage) {
		// sample size is given in sample_size: use system sampling
		auto percentage = sample_value.GetValue<double>();
		if (percentage < 0 || percentage > 100) {
			throw ParserException("Sample sample_size %llf out of range, must be between 0 and 100", percentage);
		}
		result->sample_size = Value::DOUBLE(percentage);
		result->method = SampleMethod::SYSTEM_SAMPLE;
	} else {
		// sample size is given in rows: use reservoir sampling
		auto rows = sample_value.GetValue<int64_t>();
		if (rows < 0) {
			throw ParserException("Sample rows %lld out of range, must be bigger than or equal to 0", rows);
		}
		result->sample_size = Value::BIGINT(rows);
		result->method = SampleMethod::RESERVOIR_SAMPLE;
	}
	if (sample_options.method) {
		result->method = GetSampleMethod(sample_options.method);
	}
	if (sample_options.has_seed) {
		result->seed = sample_options.seed;
	}
	return result;
}

} // namespace duckdb







namespace duckdb {

LogicalType Transformer::TransformTypeName(duckdb_libpgquery::PGTypeName *type_name) {
	if (!type_name || type_name->type != duckdb_libpgquery::T_PGTypeName) {
		throw ParserException("Expected a type");
	}
	auto stack_checker = StackCheck();

	auto name = (reinterpret_cast<duckdb_libpgquery::PGValue *>(type_name->names->tail->data.ptr_value)->val.str);
	// transform it to the SQL type
	LogicalTypeId base_type = TransformStringToLogicalTypeId(name);

	LogicalType result_type;
	if (base_type == LogicalTypeId::LIST) {
		throw ParserException("LIST is not valid as a stand-alone type");
	} else if (base_type == LogicalTypeId::ENUM) {
		throw ParserException("ENUM is not valid as a stand-alone type");
	} else if (base_type == LogicalTypeId::STRUCT) {
		if (!type_name->typmods || type_name->typmods->length == 0) {
			throw ParserException("Struct needs a name and entries");
		}
		child_list_t<LogicalType> children;
		case_insensitive_set_t name_collision_set;

		for (auto node = type_name->typmods->head; node; node = node->next) {
			auto &type_val = *((duckdb_libpgquery::PGList *)node->data.ptr_value);
			if (type_val.length != 2) {
				throw ParserException("Struct entry needs an entry name and a type name");
			}

			auto entry_name_node = (duckdb_libpgquery::PGValue *)(type_val.head->data.ptr_value);
			D_ASSERT(entry_name_node->type == duckdb_libpgquery::T_PGString);
			auto entry_type_node = (duckdb_libpgquery::PGValue *)(type_val.tail->data.ptr_value);
			D_ASSERT(entry_type_node->type == duckdb_libpgquery::T_PGTypeName);

			auto entry_name = string(entry_name_node->val.str);
			D_ASSERT(!entry_name.empty());

			if (name_collision_set.find(entry_name) != name_collision_set.end()) {
				throw ParserException("Duplicate struct entry name \"%s\"", entry_name);
			}
			name_collision_set.insert(entry_name);

			auto entry_type = TransformTypeName((duckdb_libpgquery::PGTypeName *)entry_type_node);
			children.push_back(make_pair(entry_name, entry_type));
		}
		D_ASSERT(!children.empty());
		result_type = LogicalType::STRUCT(std::move(children));
	} else if (base_type == LogicalTypeId::MAP) {

		if (!type_name->typmods || type_name->typmods->length != 2) {
			throw ParserException("Map type needs exactly two entries, key and value type");
		}
		auto key_type = TransformTypeName((duckdb_libpgquery::PGTypeName *)type_name->typmods->head->data.ptr_value);
		auto value_type = TransformTypeName((duckdb_libpgquery::PGTypeName *)type_name->typmods->tail->data.ptr_value);

		result_type = LogicalType::MAP(std::move(key_type), std::move(value_type));
	} else if (base_type == LogicalTypeId::UNION) {
		if (!type_name->typmods || type_name->typmods->length == 0) {
			throw ParserException("Union type needs at least one member");
		}
		if (type_name->typmods->length > (int)UnionType::MAX_UNION_MEMBERS) {
			throw ParserException("Union types can have at most %d members", UnionType::MAX_UNION_MEMBERS);
		}

		child_list_t<LogicalType> children;
		case_insensitive_set_t name_collision_set;

		for (auto node = type_name->typmods->head; node; node = node->next) {
			auto &type_val = *((duckdb_libpgquery::PGList *)node->data.ptr_value);
			if (type_val.length != 2) {
				throw ParserException("Union type member needs a tag name and a type name");
			}

			auto entry_name_node = (duckdb_libpgquery::PGValue *)(type_val.head->data.ptr_value);
			D_ASSERT(entry_name_node->type == duckdb_libpgquery::T_PGString);
			auto entry_type_node = (duckdb_libpgquery::PGValue *)(type_val.tail->data.ptr_value);
			D_ASSERT(entry_type_node->type == duckdb_libpgquery::T_PGTypeName);

			auto entry_name = string(entry_name_node->val.str);
			D_ASSERT(!entry_name.empty());

			if (name_collision_set.find(entry_name) != name_collision_set.end()) {
				throw ParserException("Duplicate union type tag name \"%s\"", entry_name);
			}

			name_collision_set.insert(entry_name);

			auto entry_type = TransformTypeName((duckdb_libpgquery::PGTypeName *)entry_type_node);
			children.push_back(make_pair(entry_name, entry_type));
		}
		D_ASSERT(!children.empty());
		result_type = LogicalType::UNION(std::move(children));
	} else {
		int64_t width, scale;
		if (base_type == LogicalTypeId::DECIMAL) {
			// default decimal width/scale
			width = 18;
			scale = 3;
		} else {
			width = 0;
			scale = 0;
		}
		// check any modifiers
		int modifier_idx = 0;
		if (type_name->typmods) {
			for (auto node = type_name->typmods->head; node; node = node->next) {
				auto &const_val = *((duckdb_libpgquery::PGAConst *)node->data.ptr_value);
				if (const_val.type != duckdb_libpgquery::T_PGAConst ||
				    const_val.val.type != duckdb_libpgquery::T_PGInteger) {
					throw ParserException("Expected an integer constant as type modifier");
				}
				if (const_val.val.val.ival < 0) {
					throw ParserException("Negative modifier not supported");
				}
				if (modifier_idx == 0) {
					width = const_val.val.val.ival;
					if (base_type == LogicalTypeId::BIT && const_val.location != -1) {
						width = 0;
					}
				} else if (modifier_idx == 1) {
					scale = const_val.val.val.ival;
				} else {
					throw ParserException("A maximum of two modifiers is supported");
				}
				modifier_idx++;
			}
		}
		switch (base_type) {
		case LogicalTypeId::VARCHAR:
			if (modifier_idx > 1) {
				throw ParserException("VARCHAR only supports a single modifier");
			}
			// FIXME: create CHECK constraint based on varchar width
			width = 0;
			result_type = LogicalType::VARCHAR;
			break;
		case LogicalTypeId::DECIMAL:
			if (modifier_idx == 1) {
				// only width is provided: set scale to 0
				scale = 0;
			}
			if (width <= 0 || width > Decimal::MAX_WIDTH_DECIMAL) {
				throw ParserException("Width must be between 1 and %d!", (int)Decimal::MAX_WIDTH_DECIMAL);
			}
			if (scale > width) {
				throw ParserException("Scale cannot be bigger than width");
			}
			result_type = LogicalType::DECIMAL(width, scale);
			break;
		case LogicalTypeId::INTERVAL:
			if (modifier_idx > 1) {
				throw ParserException("INTERVAL only supports a single modifier");
			}
			width = 0;
			result_type = LogicalType::INTERVAL;
			break;
		case LogicalTypeId::USER: {
			string user_type_name {name};
			result_type = LogicalType::USER(user_type_name);
			break;
		}
		case LogicalTypeId::BIT: {
			if (!width && type_name->typmods) {
				throw ParserException("Type %s does not support any modifiers!", LogicalType(base_type).ToString());
			}
			result_type = LogicalType(base_type);
			break;
		}
		case LogicalTypeId::TIMESTAMP:
			if (modifier_idx == 0) {
				result_type = LogicalType::TIMESTAMP;
			} else {
				if (modifier_idx > 1) {
					throw ParserException("TIMESTAMP only supports a single modifier");
				}
				if (width > 10) {
					throw ParserException("TIMESTAMP only supports until nano-second precision (9)");
				}
				if (width == 0) {
					result_type = LogicalType::TIMESTAMP_S;
				} else if (width <= 3) {
					result_type = LogicalType::TIMESTAMP_MS;
				} else if (width <= 6) {
					result_type = LogicalType::TIMESTAMP;
				} else {
					result_type = LogicalType::TIMESTAMP_NS;
				}
			}
			break;
		default:
			if (modifier_idx > 0) {
				throw ParserException("Type %s does not support any modifiers!", LogicalType(base_type).ToString());
			}
			result_type = LogicalType(base_type);
			break;
		}
	}
	if (type_name->arrayBounds) {
		// array bounds: turn the type into a list
		idx_t extra_stack = 0;
		for (auto cell = type_name->arrayBounds->head; cell != nullptr; cell = cell->next) {
			result_type = LogicalType::LIST(std::move(result_type));
			StackCheck(extra_stack++);
		}
	}
	return result_type;
}

} // namespace duckdb







namespace duckdb {

unique_ptr<AlterStatement> Transformer::TransformAlterSequence(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGAlterSeqStmt *>(node);
	D_ASSERT(stmt);
	auto result = make_unique<AlterStatement>();

	auto qname = TransformQualifiedName(stmt->sequence);
	auto sequence_catalog = qname.catalog;
	auto sequence_schema = qname.schema;
	auto sequence_name = qname.name;

	if (!stmt->options) {
		throw InternalException("Expected an argument for ALTER SEQUENCE.");
	}

	unordered_set<SequenceInfo, EnumClassHash> used;
	duckdb_libpgquery::PGListCell *cell = nullptr;
	for_each_cell(cell, stmt->options->head) {
		auto *def_elem = reinterpret_cast<duckdb_libpgquery::PGDefElem *>(cell->data.ptr_value);
		string opt_name = string(def_elem->defname);

		if (opt_name == "owned_by") {
			if (used.find(SequenceInfo::SEQ_OWN) != used.end()) {
				throw ParserException("Owned by value should be passed as most once");
			}
			used.insert(SequenceInfo::SEQ_OWN);

			auto val = (duckdb_libpgquery::PGValue *)def_elem->arg;
			if (!val) {
				throw InternalException("Expected an argument for option %s", opt_name);
			}
			D_ASSERT(val);
			if (val->type != duckdb_libpgquery::T_PGList) {
				throw InternalException("Expected a string argument for option %s", opt_name);
			}
			auto opt_values = vector<string>();

			auto opt_value_list = (duckdb_libpgquery::PGList *)(val);
			for (auto c = opt_value_list->head; c != nullptr; c = lnext(c)) {
				auto target = (duckdb_libpgquery::PGResTarget *)(c->data.ptr_value);
				opt_values.emplace_back(target->name);
			}
			D_ASSERT(!opt_values.empty());
			string owner_schema = INVALID_SCHEMA;
			string owner_name;
			if (opt_values.size() == 2) {
				owner_schema = opt_values[0];
				owner_name = opt_values[1];
			} else if (opt_values.size() == 1) {
				owner_schema = DEFAULT_SCHEMA;
				owner_name = opt_values[0];
			} else {
				throw InternalException("Wrong argument for %s. Expected either <schema>.<name> or <name>", opt_name);
			}
			auto info = make_unique<ChangeOwnershipInfo>(CatalogType::SEQUENCE_ENTRY, sequence_catalog, sequence_schema,
			                                             sequence_name, owner_schema, owner_name, stmt->missing_ok);
			result->info = std::move(info);
		} else {
			throw NotImplementedException("ALTER SEQUENCE option not supported yet!");
		}
	}
	result->info->if_exists = stmt->missing_ok;
	return result;
}
} // namespace duckdb






namespace duckdb {

unique_ptr<AlterStatement> Transformer::TransformAlter(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGAlterTableStmt *>(node);
	D_ASSERT(stmt);
	D_ASSERT(stmt->relation);

	if (stmt->cmds->length != 1) {
		throw ParserException("Only one ALTER command per statement is supported");
	}

	auto result = make_unique<AlterStatement>();
	auto qname = TransformQualifiedName(stmt->relation);

	// first we check the type of ALTER
	for (auto c = stmt->cmds->head; c != nullptr; c = c->next) {
		auto command = reinterpret_cast<duckdb_libpgquery::PGAlterTableCmd *>(lfirst(c));
		AlterEntryData data(qname.catalog, qname.schema, qname.name, stmt->missing_ok);
		// TODO: Include more options for command->subtype
		switch (command->subtype) {
		case duckdb_libpgquery::PG_AT_AddColumn: {
			auto cdef = (duckdb_libpgquery::PGColumnDef *)command->def;

			if (stmt->relkind != duckdb_libpgquery::PG_OBJECT_TABLE) {
				throw ParserException("Adding columns is only supported for tables");
			}
			if (cdef->category == duckdb_libpgquery::COL_GENERATED) {
				throw ParserException("Adding generated columns after table creation is not supported yet");
			}
			auto centry = TransformColumnDefinition(cdef);

			if (cdef->constraints) {
				for (auto constr = cdef->constraints->head; constr != nullptr; constr = constr->next) {
					auto constraint = TransformConstraint(constr, centry, 0);
					if (!constraint) {
						continue;
					}
					throw ParserException("Adding columns with constraints not yet supported");
				}
			}
			result->info = make_unique<AddColumnInfo>(std::move(data), std::move(centry), command->missing_ok);
			break;
		}
		case duckdb_libpgquery::PG_AT_DropColumn: {
			bool cascade = command->behavior == duckdb_libpgquery::PG_DROP_CASCADE;

			if (stmt->relkind != duckdb_libpgquery::PG_OBJECT_TABLE) {
				throw ParserException("Dropping columns is only supported for tables");
			}
			result->info = make_unique<RemoveColumnInfo>(std::move(data), command->name, command->missing_ok, cascade);
			break;
		}
		case duckdb_libpgquery::PG_AT_ColumnDefault: {
			auto expr = TransformExpression(command->def);

			if (stmt->relkind != duckdb_libpgquery::PG_OBJECT_TABLE) {
				throw ParserException("Alter column's default is only supported for tables");
			}
			result->info = make_unique<SetDefaultInfo>(std::move(data), command->name, std::move(expr));
			break;
		}
		case duckdb_libpgquery::PG_AT_AlterColumnType: {
			auto cdef = (duckdb_libpgquery::PGColumnDef *)command->def;
			auto column_definition = TransformColumnDefinition(cdef);
			unique_ptr<ParsedExpression> expr;

			if (stmt->relkind != duckdb_libpgquery::PG_OBJECT_TABLE) {
				throw ParserException("Alter column's type is only supported for tables");
			}
			if (cdef->raw_default) {
				expr = TransformExpression(cdef->raw_default);
			} else {
				auto colref = make_unique<ColumnRefExpression>(command->name);
				expr = make_unique<CastExpression>(column_definition.Type(), std::move(colref));
			}
			result->info = make_unique<ChangeColumnTypeInfo>(std::move(data), command->name, column_definition.Type(),
			                                                 std::move(expr));
			break;
		}
		case duckdb_libpgquery::PG_AT_SetNotNull: {
			result->info = make_unique<SetNotNullInfo>(std::move(data), command->name);
			break;
		}
		case duckdb_libpgquery::PG_AT_DropNotNull: {
			result->info = make_unique<DropNotNullInfo>(std::move(data), command->name);
			break;
		}
		case duckdb_libpgquery::PG_AT_DropConstraint:
		default:
			throw NotImplementedException("ALTER TABLE option not supported yet!");
		}
	}

	return result;
}

} // namespace duckdb





namespace duckdb {

unique_ptr<AttachStatement> Transformer::TransformAttach(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGAttachStmt *>(node);
	auto result = make_unique<AttachStatement>();
	auto info = make_unique<AttachInfo>();
	info->name = stmt->name ? stmt->name : string();
	info->path = stmt->path;

	if (stmt->options) {
		duckdb_libpgquery::PGListCell *cell = nullptr;
		for_each_cell(cell, stmt->options->head) {
			auto *def_elem = reinterpret_cast<duckdb_libpgquery::PGDefElem *>(cell->data.ptr_value);
			Value val;
			if (def_elem->arg) {
				val = TransformValue(*((duckdb_libpgquery::PGValue *)def_elem->arg))->value;
			} else {
				val = Value::BOOLEAN(true);
			}
			info->options[StringUtil::Lower(def_elem->defname)] = std::move(val);
		}
	}
	result->info = std::move(info);
	return result;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<CallStatement> Transformer::TransformCall(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGCallStmt *>(node);
	D_ASSERT(stmt);

	auto result = make_unique<CallStatement>();
	result->function = TransformFuncCall((duckdb_libpgquery::PGFuncCall *)stmt->func);
	return result;
}

} // namespace duckdb





namespace duckdb {

unique_ptr<SQLStatement> Transformer::TransformCheckpoint(duckdb_libpgquery::PGNode *node) {
	auto checkpoint = (duckdb_libpgquery::PGCheckPointStmt *)node;

	vector<unique_ptr<ParsedExpression>> children;
	// transform into "CALL checkpoint()" or "CALL force_checkpoint()"
	auto checkpoint_name = checkpoint->force ? "force_checkpoint" : "checkpoint";
	auto result = make_unique<CallStatement>();
	auto function = make_unique<FunctionExpression>(checkpoint_name, std::move(children));
	if (checkpoint->name) {
		function->children.push_back(make_unique<ConstantExpression>(Value(checkpoint->name)));
	}
	result->function = std::move(function);
	return std::move(result);
}

} // namespace duckdb







#include <cstring>

namespace duckdb {

void Transformer::TransformCopyOptions(CopyInfo &info, duckdb_libpgquery::PGList *options) {
	if (!options) {
		return;
	}
	duckdb_libpgquery::PGListCell *cell = nullptr;

	// iterate over each option
	for_each_cell(cell, options->head) {
		auto *def_elem = reinterpret_cast<duckdb_libpgquery::PGDefElem *>(cell->data.ptr_value);
		if (StringUtil::Lower(def_elem->defname) == "format") {
			// format specifier: interpret this option
			auto *format_val = (duckdb_libpgquery::PGValue *)(def_elem->arg);
			if (!format_val || format_val->type != duckdb_libpgquery::T_PGString) {
				throw ParserException("Unsupported parameter type for FORMAT: expected e.g. FORMAT 'csv', 'parquet'");
			}
			info.format = StringUtil::Lower(format_val->val.str);
			continue;
		}
		// otherwise
		if (info.options.find(def_elem->defname) != info.options.end()) {
			throw ParserException("Unexpected duplicate option \"%s\"", def_elem->defname);
		}
		if (!def_elem->arg) {
			info.options[def_elem->defname] = vector<Value>();
			continue;
		}
		switch (def_elem->arg->type) {
		case duckdb_libpgquery::T_PGList: {
			auto column_list = (duckdb_libpgquery::PGList *)(def_elem->arg);
			for (auto c = column_list->head; c != nullptr; c = lnext(c)) {
				auto target = (duckdb_libpgquery::PGResTarget *)(c->data.ptr_value);
				info.options[def_elem->defname].push_back(Value(target->name));
			}
			break;
		}
		case duckdb_libpgquery::T_PGAStar:
			info.options[def_elem->defname].push_back(Value("*"));
			break;
		default:
			info.options[def_elem->defname].push_back(
			    TransformValue(*((duckdb_libpgquery::PGValue *)def_elem->arg))->value);
			break;
		}
	}
}

unique_ptr<CopyStatement> Transformer::TransformCopy(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGCopyStmt *>(node);
	D_ASSERT(stmt);
	auto result = make_unique<CopyStatement>();
	auto &info = *result->info;

	// get file_path and is_from
	info.is_from = stmt->is_from;
	if (!stmt->filename) {
		// stdin/stdout
		info.file_path = info.is_from ? "/dev/stdin" : "/dev/stdout";
	} else {
		// copy to a file
		info.file_path = stmt->filename;
	}
	if (StringUtil::EndsWith(info.file_path, ".parquet")) {
		info.format = "parquet";
	} else if (StringUtil::EndsWith(info.file_path, ".json") || StringUtil::EndsWith(info.file_path, ".ndjson")) {
		info.format = "json";
	} else {
		info.format = "csv";
	}

	// get select_list
	if (stmt->attlist) {
		for (auto n = stmt->attlist->head; n != nullptr; n = n->next) {
			auto target = reinterpret_cast<duckdb_libpgquery::PGResTarget *>(n->data.ptr_value);
			if (target->name) {
				info.select_list.emplace_back(target->name);
			}
		}
	}

	if (stmt->relation) {
		auto ref = TransformRangeVar(stmt->relation);
		auto &table = *reinterpret_cast<BaseTableRef *>(ref.get());
		info.table = table.table_name;
		info.schema = table.schema_name;
		info.catalog = table.catalog_name;
	} else {
		result->select_statement = TransformSelectNode((duckdb_libpgquery::PGSelectStmt *)stmt->query);
	}

	// handle the different options of the COPY statement
	TransformCopyOptions(info, stmt->options);

	return result;
}

} // namespace duckdb






namespace duckdb {

unique_ptr<CreateStatement> Transformer::TransformCreateDatabase(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGCreateDatabaseStmt *>(node);
	auto result = make_unique<CreateStatement>();
	auto info = make_unique<CreateDatabaseInfo>();

	info->extension_name = stmt->extension ? stmt->extension : string();
	info->path = stmt->path ? stmt->path : string();

	auto qualified_name = TransformQualifiedName(stmt->name);
	if (!IsInvalidCatalog(qualified_name.catalog)) {
		throw ParserException("Expected \"CREATE DATABASE database\" ");
	}

	info->catalog = qualified_name.catalog;
	info->name = qualified_name.name;

	result->info = std::move(info);
	return result;
}

} // namespace duckdb








namespace duckdb {

unique_ptr<CreateStatement> Transformer::TransformCreateFunction(duckdb_libpgquery::PGNode *node) {
	D_ASSERT(node);
	D_ASSERT(node->type == duckdb_libpgquery::T_PGCreateFunctionStmt);

	auto stmt = reinterpret_cast<duckdb_libpgquery::PGCreateFunctionStmt *>(node);
	D_ASSERT(stmt);
	D_ASSERT(stmt->function || stmt->query);

	auto result = make_unique<CreateStatement>();
	auto qname = TransformQualifiedName(stmt->name);

	unique_ptr<MacroFunction> macro_func;

	// function can be null here
	if (stmt->function) {
		auto expression = TransformExpression(stmt->function);
		macro_func = make_unique<ScalarMacroFunction>(std::move(expression));
	} else if (stmt->query) {
		auto query_node = TransformSelect(stmt->query, true)->node->Copy();
		macro_func = make_unique<TableMacroFunction>(std::move(query_node));
	}

	auto info =
	    make_unique<CreateMacroInfo>((stmt->function ? CatalogType::MACRO_ENTRY : CatalogType::TABLE_MACRO_ENTRY));
	info->catalog = qname.catalog;
	info->schema = qname.schema;
	info->name = qname.name;

	// temporary macro
	switch (stmt->name->relpersistence) {
	case duckdb_libpgquery::PG_RELPERSISTENCE_TEMP:
		info->temporary = true;
		break;
	case duckdb_libpgquery::PG_RELPERSISTENCE_UNLOGGED:
		throw ParserException("Unlogged flag not supported for macros: '%s'", qname.name);
		break;
	case duckdb_libpgquery::RELPERSISTENCE_PERMANENT:
		info->temporary = false;
		break;
	}

	// what to do on conflict
	info->on_conflict = TransformOnConflict(stmt->onconflict);

	if (stmt->params) {
		vector<unique_ptr<ParsedExpression>> parameters;
		TransformExpressionList(*stmt->params, parameters);
		for (auto &param : parameters) {
			if (param->type == ExpressionType::VALUE_CONSTANT) {
				// parameters with default value (must have an alias)
				if (param->alias.empty()) {
					throw ParserException("Invalid parameter: '%s'", param->ToString());
				}
				if (macro_func->default_parameters.find(param->alias) != macro_func->default_parameters.end()) {
					throw ParserException("Duplicate default parameter: '%s'", param->alias);
				}
				macro_func->default_parameters[param->alias] = std::move(param);
			} else if (param->GetExpressionClass() == ExpressionClass::COLUMN_REF) {
				// positional parameters
				if (!macro_func->default_parameters.empty()) {
					throw ParserException("Positional parameters cannot come after parameters with a default value!");
				}
				macro_func->parameters.push_back(std::move(param));
			} else {
				throw ParserException("Invalid parameter: '%s'", param->ToString());
			}
		}
	}

	info->function = std::move(macro_func);
	result->info = std::move(info);

	return result;
}

} // namespace duckdb







namespace duckdb {

static IndexType StringToIndexType(const string &str) {
	string upper_str = StringUtil::Upper(str);
	if (upper_str == "INVALID") {
		return IndexType::INVALID;
	} else if (upper_str == "ART") {
		return IndexType::ART;
	} else {
		throw ConversionException("No IndexType conversion from string '%s'", upper_str);
	}
	return IndexType::INVALID;
}

vector<unique_ptr<ParsedExpression>> Transformer::TransformIndexParameters(duckdb_libpgquery::PGList *list,
                                                                           const string &relation_name) {
	vector<unique_ptr<ParsedExpression>> expressions;
	for (auto cell = list->head; cell != nullptr; cell = cell->next) {
		auto index_element = (duckdb_libpgquery::PGIndexElem *)cell->data.ptr_value;
		if (index_element->collation) {
			throw NotImplementedException("Index with collation not supported yet!");
		}
		if (index_element->opclass) {
			throw NotImplementedException("Index with opclass not supported yet!");
		}

		if (index_element->name) {
			// create a column reference expression
			expressions.push_back(make_unique<ColumnRefExpression>(index_element->name, relation_name));
		} else {
			// parse the index expression
			D_ASSERT(index_element->expr);
			expressions.push_back(TransformExpression(index_element->expr));
		}
	}
	return expressions;
}

unique_ptr<CreateStatement> Transformer::TransformCreateIndex(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGIndexStmt *>(node);
	D_ASSERT(stmt);
	auto result = make_unique<CreateStatement>();
	auto info = make_unique<CreateIndexInfo>();
	if (stmt->unique) {
		info->constraint_type = IndexConstraintType::UNIQUE;
	} else {
		info->constraint_type = IndexConstraintType::NONE;
	}

	info->on_conflict = TransformOnConflict(stmt->onconflict);

	info->expressions = TransformIndexParameters(stmt->indexParams, stmt->relation->relname);

	info->index_type = StringToIndexType(string(stmt->accessMethod));
	auto tableref = make_unique<BaseTableRef>();
	tableref->table_name = stmt->relation->relname;
	if (stmt->relation->schemaname) {
		tableref->schema_name = stmt->relation->schemaname;
	}
	info->table = std::move(tableref);
	if (stmt->idxname) {
		info->index_name = stmt->idxname;
	} else {
		throw NotImplementedException("Index without a name not supported yet!");
	}
	for (auto &expr : info->expressions) {
		info->parsed_expressions.emplace_back(expr->Copy());
	}
	result->info = std::move(info);
	return result;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<CreateStatement> Transformer::TransformCreateSchema(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGCreateSchemaStmt *>(node);
	D_ASSERT(stmt);
	auto result = make_unique<CreateStatement>();
	auto info = make_unique<CreateSchemaInfo>();

	D_ASSERT(stmt->schemaname);
	info->catalog = stmt->catalogname ? stmt->catalogname : INVALID_CATALOG;
	info->schema = stmt->schemaname;
	info->on_conflict = TransformOnConflict(stmt->onconflict);

	if (stmt->schemaElts) {
		// schema elements
		for (auto cell = stmt->schemaElts->head; cell != nullptr; cell = cell->next) {
			auto node = reinterpret_cast<duckdb_libpgquery::PGNode *>(cell->data.ptr_value);
			switch (node->type) {
			case duckdb_libpgquery::T_PGCreateStmt:
			case duckdb_libpgquery::T_PGViewStmt:
			default:
				throw NotImplementedException("Schema element not supported yet!");
			}
		}
	}
	result->info = std::move(info);
	return result;
}

} // namespace duckdb







namespace duckdb {

unique_ptr<CreateStatement> Transformer::TransformCreateSequence(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGCreateSeqStmt *>(node);

	auto result = make_unique<CreateStatement>();
	auto info = make_unique<CreateSequenceInfo>();

	auto qname = TransformQualifiedName(stmt->sequence);
	info->catalog = qname.catalog;
	info->schema = qname.schema;
	info->name = qname.name;

	if (stmt->options) {
		unordered_set<SequenceInfo, EnumClassHash> used;
		duckdb_libpgquery::PGListCell *cell = nullptr;
		for_each_cell(cell, stmt->options->head) {
			auto *def_elem = reinterpret_cast<duckdb_libpgquery::PGDefElem *>(cell->data.ptr_value);
			string opt_name = string(def_elem->defname);
			auto val = (duckdb_libpgquery::PGValue *)def_elem->arg;
			bool nodef = def_elem->defaction == duckdb_libpgquery::PG_DEFELEM_UNSPEC && !val; // e.g. NO MINVALUE
			int64_t opt_value = 0;

			if (val) {
				if (val->type == duckdb_libpgquery::T_PGInteger) {
					opt_value = val->val.ival;
				} else if (val->type == duckdb_libpgquery::T_PGFloat) {
					if (!TryCast::Operation<string_t, int64_t>(string_t(val->val.str), opt_value, true)) {
						throw ParserException("Expected an integer argument for option %s", opt_name);
					}
				} else {
					throw ParserException("Expected an integer argument for option %s", opt_name);
				}
			}
			if (opt_name == "increment") {
				if (used.find(SequenceInfo::SEQ_INC) != used.end()) {
					throw ParserException("Increment value should be passed as most once");
				}
				used.insert(SequenceInfo::SEQ_INC);
				if (nodef) {
					continue;
				}

				info->increment = opt_value;
				if (info->increment == 0) {
					throw ParserException("Increment must not be zero");
				}
				if (info->increment < 0) {
					info->start_value = info->max_value = -1;
					info->min_value = NumericLimits<int64_t>::Minimum();
				} else {
					info->start_value = info->min_value = 1;
					info->max_value = NumericLimits<int64_t>::Maximum();
				}
			} else if (opt_name == "minvalue") {
				if (used.find(SequenceInfo::SEQ_MIN) != used.end()) {
					throw ParserException("Minvalue should be passed as most once");
				}
				used.insert(SequenceInfo::SEQ_MIN);
				if (nodef) {
					continue;
				}

				info->min_value = opt_value;
				if (info->increment > 0) {
					info->start_value = info->min_value;
				}
			} else if (opt_name == "maxvalue") {
				if (used.find(SequenceInfo::SEQ_MAX) != used.end()) {
					throw ParserException("Maxvalue should be passed as most once");
				}
				used.insert(SequenceInfo::SEQ_MAX);
				if (nodef) {
					continue;
				}

				info->max_value = opt_value;
				if (info->increment < 0) {
					info->start_value = info->max_value;
				}
			} else if (opt_name == "start") {
				if (used.find(SequenceInfo::SEQ_START) != used.end()) {
					throw ParserException("Start value should be passed as most once");
				}
				used.insert(SequenceInfo::SEQ_START);
				if (nodef) {
					continue;
				}

				info->start_value = opt_value;
			} else if (opt_name == "cycle") {
				if (used.find(SequenceInfo::SEQ_CYCLE) != used.end()) {
					throw ParserException("Cycle value should be passed as most once");
				}
				used.insert(SequenceInfo::SEQ_CYCLE);
				if (nodef) {
					continue;
				}

				info->cycle = opt_value > 0;
			} else {
				throw ParserException("Unrecognized option \"%s\" for CREATE SEQUENCE", opt_name);
			}
		}
	}
	info->temporary = !stmt->sequence->relpersistence;
	info->on_conflict = TransformOnConflict(stmt->onconflict);
	if (info->max_value <= info->min_value) {
		throw ParserException("MINVALUE (%lld) must be less than MAXVALUE (%lld)", info->min_value, info->max_value);
	}
	if (info->start_value < info->min_value) {
		throw ParserException("START value (%lld) cannot be less than MINVALUE (%lld)", info->start_value,
		                      info->min_value);
	}
	if (info->start_value > info->max_value) {
		throw ParserException("START value (%lld) cannot be greater than MAXVALUE (%lld)", info->start_value,
		                      info->max_value);
	}
	result->info = std::move(info);
	return result;
}

} // namespace duckdb








namespace duckdb {

string Transformer::TransformCollation(duckdb_libpgquery::PGCollateClause *collate) {
	if (!collate) {
		return string();
	}
	string collation;
	for (auto c = collate->collname->head; c != nullptr; c = lnext(c)) {
		auto pgvalue = (duckdb_libpgquery::PGValue *)c->data.ptr_value;
		if (pgvalue->type != duckdb_libpgquery::T_PGString) {
			throw ParserException("Expected a string as collation type!");
		}
		auto collation_argument = string(pgvalue->val.str);
		if (collation.empty()) {
			collation = collation_argument;
		} else {
			collation += "." + collation_argument;
		}
	}
	return collation;
}

OnCreateConflict Transformer::TransformOnConflict(duckdb_libpgquery::PGOnCreateConflict conflict) {
	switch (conflict) {
	case duckdb_libpgquery::PG_ERROR_ON_CONFLICT:
		return OnCreateConflict::ERROR_ON_CONFLICT;
	case duckdb_libpgquery::PG_IGNORE_ON_CONFLICT:
		return OnCreateConflict::IGNORE_ON_CONFLICT;
	case duckdb_libpgquery::PG_REPLACE_ON_CONFLICT:
		return OnCreateConflict::REPLACE_ON_CONFLICT;
	default:
		throw InternalException("Unrecognized OnConflict type");
	}
}

unique_ptr<ParsedExpression> Transformer::TransformCollateExpr(duckdb_libpgquery::PGCollateClause *collate) {
	auto child = TransformExpression(collate->arg);
	auto collation = TransformCollation(collate);
	return make_unique<CollateExpression>(collation, std::move(child));
}

ColumnDefinition Transformer::TransformColumnDefinition(duckdb_libpgquery::PGColumnDef *cdef) {
	string colname;
	if (cdef->colname) {
		colname = cdef->colname;
	}
	bool optional_type = cdef->category == duckdb_libpgquery::COL_GENERATED;
	LogicalType target_type = (optional_type && !cdef->typeName) ? LogicalType::ANY : TransformTypeName(cdef->typeName);
	if (cdef->collClause) {
		if (cdef->category == duckdb_libpgquery::COL_GENERATED) {
			throw ParserException("Collations are not supported on generated columns");
		}
		if (target_type.id() != LogicalTypeId::VARCHAR) {
			throw ParserException("Only VARCHAR columns can have collations!");
		}
		target_type = LogicalType::VARCHAR_COLLATION(TransformCollation(cdef->collClause));
	}

	return ColumnDefinition(colname, target_type);
}

unique_ptr<CreateStatement> Transformer::TransformCreateTable(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGCreateStmt *>(node);
	D_ASSERT(stmt);
	auto result = make_unique<CreateStatement>();
	auto info = make_unique<CreateTableInfo>();

	if (stmt->inhRelations) {
		throw NotImplementedException("inherited relations not implemented");
	}
	D_ASSERT(stmt->relation);

	info->catalog = INVALID_CATALOG;
	auto qname = TransformQualifiedName(stmt->relation);
	info->catalog = qname.catalog;
	info->schema = qname.schema;
	info->table = qname.name;
	info->on_conflict = TransformOnConflict(stmt->onconflict);
	info->temporary =
	    stmt->relation->relpersistence == duckdb_libpgquery::PGPostgresRelPersistence::PG_RELPERSISTENCE_TEMP;

	if (info->temporary && stmt->oncommit != duckdb_libpgquery::PGOnCommitAction::PG_ONCOMMIT_PRESERVE_ROWS &&
	    stmt->oncommit != duckdb_libpgquery::PGOnCommitAction::PG_ONCOMMIT_NOOP) {
		throw NotImplementedException("Only ON COMMIT PRESERVE ROWS is supported");
	}
	if (!stmt->tableElts) {
		throw ParserException("Table must have at least one column!");
	}

	idx_t column_count = 0;
	for (auto c = stmt->tableElts->head; c != nullptr; c = lnext(c)) {
		auto node = reinterpret_cast<duckdb_libpgquery::PGNode *>(c->data.ptr_value);
		switch (node->type) {
		case duckdb_libpgquery::T_PGColumnDef: {
			auto cdef = (duckdb_libpgquery::PGColumnDef *)c->data.ptr_value;
			auto centry = TransformColumnDefinition(cdef);
			if (cdef->constraints) {
				for (auto constr = cdef->constraints->head; constr != nullptr; constr = constr->next) {
					auto constraint = TransformConstraint(constr, centry, info->columns.LogicalColumnCount());
					if (constraint) {
						info->constraints.push_back(std::move(constraint));
					}
				}
			}
			info->columns.AddColumn(std::move(centry));
			column_count++;
			break;
		}
		case duckdb_libpgquery::T_PGConstraint: {
			info->constraints.push_back(TransformConstraint(c));
			break;
		}
		default:
			throw NotImplementedException("ColumnDef type not handled yet");
		}
	}

	if (!column_count) {
		throw ParserException("Table must have at least one column!");
	}

	result->info = std::move(info);
	return result;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<CreateStatement> Transformer::TransformCreateTableAs(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGCreateTableAsStmt *>(node);
	D_ASSERT(stmt);
	if (stmt->relkind == duckdb_libpgquery::PG_OBJECT_MATVIEW) {
		throw NotImplementedException("Materialized view not implemented");
	}
	if (stmt->is_select_into || stmt->into->colNames || stmt->into->options) {
		throw NotImplementedException("Unimplemented features for CREATE TABLE as");
	}
	auto qname = TransformQualifiedName(stmt->into->rel);
	if (stmt->query->type != duckdb_libpgquery::T_PGSelectStmt) {
		throw ParserException("CREATE TABLE AS requires a SELECT clause");
	}
	auto query = TransformSelect(stmt->query, false);

	auto result = make_unique<CreateStatement>();
	auto info = make_unique<CreateTableInfo>();
	info->catalog = qname.catalog;
	info->schema = qname.schema;
	info->table = qname.name;
	info->on_conflict = TransformOnConflict(stmt->onconflict);
	info->temporary =
	    stmt->into->rel->relpersistence == duckdb_libpgquery::PGPostgresRelPersistence::PG_RELPERSISTENCE_TEMP;
	info->query = std::move(query);
	result->info = std::move(info);
	return result;
}

} // namespace duckdb






namespace duckdb {

Vector ReadPgListToVector(duckdb_libpgquery::PGList *column_list, idx_t &size) {
	if (!column_list) {
		Vector result(LogicalType::VARCHAR);
		return result;
	}
	// First we discover the size of this list
	for (auto c = column_list->head; c != nullptr; c = lnext(c)) {
		size++;
	}

	Vector result(LogicalType::VARCHAR, size);
	auto result_ptr = FlatVector::GetData<string_t>(result);

	size = 0;
	for (auto c = column_list->head; c != nullptr; c = lnext(c)) {
		auto &type_val = *((duckdb_libpgquery::PGAConst *)c->data.ptr_value);
		auto entry_value_node = (duckdb_libpgquery::PGValue)(type_val.val);
		if (entry_value_node.type != duckdb_libpgquery::T_PGString) {
			throw ParserException("Expected a string constant as value");
		}

		auto entry_value = string(entry_value_node.val.str);
		D_ASSERT(!entry_value.empty());
		result_ptr[size++] = StringVector::AddStringOrBlob(result, entry_value);
	}
	return result;
}

unique_ptr<CreateStatement> Transformer::TransformCreateType(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGCreateTypeStmt *>(node);
	D_ASSERT(stmt);
	auto result = make_unique<CreateStatement>();
	auto info = make_unique<CreateTypeInfo>();

	auto qualified_name = TransformQualifiedName(stmt->typeName);
	info->catalog = qualified_name.catalog;
	info->schema = qualified_name.schema;
	info->name = qualified_name.name;

	switch (stmt->kind) {
	case duckdb_libpgquery::PG_NEWTYPE_ENUM: {
		info->internal = false;
		if (stmt->query) {
			// CREATE TYPE mood AS ENUM (SELECT ...)
			D_ASSERT(stmt->vals == nullptr);
			auto query = TransformSelect(stmt->query, false);
			info->query = std::move(query);
			info->type = LogicalType::INVALID;
		} else {
			D_ASSERT(stmt->query == nullptr);
			idx_t size = 0;
			auto ordered_array = ReadPgListToVector(stmt->vals, size);
			info->type = LogicalType::ENUM(info->name, ordered_array, size);
		}
	} break;

	case duckdb_libpgquery::PG_NEWTYPE_ALIAS: {
		LogicalType target_type = TransformTypeName(stmt->ofType);
		target_type.SetAlias(info->name);
		info->type = target_type;
	} break;

	default:
		throw InternalException("Unknown kind of new type");
	}

	result->info = std::move(info);
	return result;
}
} // namespace duckdb




namespace duckdb {

unique_ptr<CreateStatement> Transformer::TransformCreateView(duckdb_libpgquery::PGNode *node) {
	D_ASSERT(node);
	D_ASSERT(node->type == duckdb_libpgquery::T_PGViewStmt);

	auto stmt = reinterpret_cast<duckdb_libpgquery::PGViewStmt *>(node);
	D_ASSERT(stmt);
	D_ASSERT(stmt->view);

	auto result = make_unique<CreateStatement>();
	auto info = make_unique<CreateViewInfo>();

	auto qname = TransformQualifiedName(stmt->view);
	info->catalog = qname.catalog;
	info->schema = qname.schema;
	info->view_name = qname.name;
	info->temporary = !stmt->view->relpersistence;
	if (info->temporary && IsInvalidCatalog(info->catalog)) {
		info->catalog = TEMP_CATALOG;
	}
	info->on_conflict = TransformOnConflict(stmt->onconflict);

	info->query = TransformSelect(stmt->query, false);

	if (stmt->aliases && stmt->aliases->length > 0) {
		for (auto c = stmt->aliases->head; c != nullptr; c = lnext(c)) {
			auto node = reinterpret_cast<duckdb_libpgquery::PGNode *>(c->data.ptr_value);
			switch (node->type) {
			case duckdb_libpgquery::T_PGString: {
				auto val = (duckdb_libpgquery::PGValue *)node;
				info->aliases.emplace_back(val->val.str);
				break;
			}
			default:
				throw NotImplementedException("View projection type");
			}
		}
		if (info->aliases.empty()) {
			throw ParserException("Need at least one column name in CREATE VIEW projection list");
		}
	}

	if (stmt->options && stmt->options->length > 0) {
		throw NotImplementedException("VIEW options");
	}

	if (stmt->withCheckOption != duckdb_libpgquery::PGViewCheckOption::PG_NO_CHECK_OPTION) {
		throw NotImplementedException("VIEW CHECK options");
	}
	result->info = std::move(info);
	return result;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<DeleteStatement> Transformer::TransformDelete(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGDeleteStmt *>(node);
	D_ASSERT(stmt);
	auto result = make_unique<DeleteStatement>();
	if (stmt->withClause) {
		TransformCTE(reinterpret_cast<duckdb_libpgquery::PGWithClause *>(stmt->withClause), result->cte_map);
	}

	result->condition = TransformExpression(stmt->whereClause);
	result->table = TransformRangeVar(stmt->relation);
	if (result->table->type != TableReferenceType::BASE_TABLE) {
		throw Exception("Can only delete from base tables!");
	}
	if (stmt->usingClause) {
		for (auto n = stmt->usingClause->head; n != nullptr; n = n->next) {
			auto target = reinterpret_cast<duckdb_libpgquery::PGNode *>(n->data.ptr_value);
			auto using_entry = TransformTableRefNode(target);
			result->using_clauses.push_back(std::move(using_entry));
		}
	}

	if (stmt->returningList) {
		Transformer::TransformExpressionList(*(stmt->returningList), result->returning_list);
	}
	return result;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<SQLStatement> Transformer::TransformDrop(duckdb_libpgquery::PGNode *node) {
	auto stmt = (duckdb_libpgquery::PGDropStmt *)(node);
	auto result = make_unique<DropStatement>();
	auto &info = *result->info.get();
	D_ASSERT(stmt);
	if (stmt->objects->length != 1) {
		throw NotImplementedException("Can only drop one object at a time");
	}
	switch (stmt->removeType) {
	case duckdb_libpgquery::PG_OBJECT_TABLE:
		info.type = CatalogType::TABLE_ENTRY;
		break;
	case duckdb_libpgquery::PG_OBJECT_SCHEMA:
		info.type = CatalogType::SCHEMA_ENTRY;
		break;
	case duckdb_libpgquery::PG_OBJECT_INDEX:
		info.type = CatalogType::INDEX_ENTRY;
		break;
	case duckdb_libpgquery::PG_OBJECT_VIEW:
		info.type = CatalogType::VIEW_ENTRY;
		break;
	case duckdb_libpgquery::PG_OBJECT_SEQUENCE:
		info.type = CatalogType::SEQUENCE_ENTRY;
		break;
	case duckdb_libpgquery::PG_OBJECT_FUNCTION:
		info.type = CatalogType::MACRO_ENTRY;
		break;
	case duckdb_libpgquery::PG_OBJECT_TABLE_MACRO:
		info.type = CatalogType::TABLE_MACRO_ENTRY;
		break;
	case duckdb_libpgquery::PG_OBJECT_TYPE:
		info.type = CatalogType::TYPE_ENTRY;
		break;
	case duckdb_libpgquery::PG_OBJECT_DATABASE:
		info.type = CatalogType::DATABASE_ENTRY;
		break;
	default:
		throw NotImplementedException("Cannot drop this type yet");
	}

	switch (stmt->removeType) {
	case duckdb_libpgquery::PG_OBJECT_TYPE: {
		auto view_list = (duckdb_libpgquery::PGList *)stmt->objects;
		auto target = (duckdb_libpgquery::PGTypeName *)(view_list->head->data.ptr_value);
		info.name = (reinterpret_cast<duckdb_libpgquery::PGValue *>(target->names->tail->data.ptr_value)->val.str);
		break;
	}
	case duckdb_libpgquery::PG_OBJECT_SCHEMA: {
		auto view_list = (duckdb_libpgquery::PGList *)stmt->objects->head->data.ptr_value;
		if (view_list->length == 2) {
			info.catalog = ((duckdb_libpgquery::PGValue *)view_list->head->data.ptr_value)->val.str;
			info.name = ((duckdb_libpgquery::PGValue *)view_list->head->next->data.ptr_value)->val.str;
		} else if (view_list->length == 1) {
			info.name = ((duckdb_libpgquery::PGValue *)view_list->head->data.ptr_value)->val.str;
		} else {
			throw ParserException("Expected \"catalog.schema\" or \"schema\"");
		}
		break;
	}
	default: {
		auto view_list = (duckdb_libpgquery::PGList *)stmt->objects->head->data.ptr_value;
		if (view_list->length == 3) {
			info.catalog = ((duckdb_libpgquery::PGValue *)view_list->head->data.ptr_value)->val.str;
			info.schema = ((duckdb_libpgquery::PGValue *)view_list->head->next->data.ptr_value)->val.str;
			info.name = ((duckdb_libpgquery::PGValue *)view_list->head->next->next->data.ptr_value)->val.str;
		} else if (view_list->length == 2) {
			info.schema = ((duckdb_libpgquery::PGValue *)view_list->head->data.ptr_value)->val.str;
			info.name = ((duckdb_libpgquery::PGValue *)view_list->head->next->data.ptr_value)->val.str;
		} else if (view_list->length == 1) {
			info.name = ((duckdb_libpgquery::PGValue *)view_list->head->data.ptr_value)->val.str;
		} else {
			throw ParserException("Expected \"catalog.schema.name\", \"schema.name\"or \"name\"");
		}
		break;
	}
	}
	info.cascade = stmt->behavior == duckdb_libpgquery::PGDropBehavior::PG_DROP_CASCADE;
	info.if_exists = stmt->missing_ok;
	return std::move(result);
}

} // namespace duckdb



namespace duckdb {

unique_ptr<ExplainStatement> Transformer::TransformExplain(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGExplainStmt *>(node);
	D_ASSERT(stmt);
	auto explain_type = ExplainType::EXPLAIN_STANDARD;
	if (stmt->options) {
		for (auto n = stmt->options->head; n; n = n->next) {
			auto def_elem = ((duckdb_libpgquery::PGDefElem *)n->data.ptr_value)->defname;
			string elem(def_elem);
			if (elem == "analyze") {
				explain_type = ExplainType::EXPLAIN_ANALYZE;
			} else {
				throw NotImplementedException("Unimplemented explain type: %s", elem);
			}
		}
	}
	return make_unique<ExplainStatement>(TransformStatement(stmt->query), explain_type);
}

} // namespace duckdb



namespace duckdb {

unique_ptr<ExportStatement> Transformer::TransformExport(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGExportStmt *>(node);
	auto info = make_unique<CopyInfo>();
	info->file_path = stmt->filename;
	info->format = "csv";
	info->is_from = false;
	// handle export options
	TransformCopyOptions(*info, stmt->options);

	auto result = make_unique<ExportStatement>(std::move(info));
	if (stmt->database) {
		result->database = stmt->database;
	}
	return result;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<PragmaStatement> Transformer::TransformImport(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGImportStmt *>(node);
	auto result = make_unique<PragmaStatement>();
	result->info->name = "import_database";
	result->info->parameters.emplace_back(stmt->filename);
	return result;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<TableRef> Transformer::TransformValuesList(duckdb_libpgquery::PGList *list) {
	auto result = make_unique<ExpressionListRef>();
	for (auto value_list = list->head; value_list != nullptr; value_list = value_list->next) {
		auto target = (duckdb_libpgquery::PGList *)(value_list->data.ptr_value);

		vector<unique_ptr<ParsedExpression>> insert_values;
		TransformExpressionList(*target, insert_values);
		if (!result->values.empty()) {
			if (result->values[0].size() != insert_values.size()) {
				throw ParserException("VALUES lists must all be the same length");
			}
		}
		result->values.push_back(std::move(insert_values));
	}
	result->alias = "valueslist";
	return std::move(result);
}

unique_ptr<InsertStatement> Transformer::TransformInsert(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGInsertStmt *>(node);
	D_ASSERT(stmt);

	if (!stmt->selectStmt) {
		// TODO: This should be easy to add, we already support DEFAULT in the values list,
		// this could probably just be transformed into VALUES (DEFAULT, DEFAULT, DEFAULT, ..) in the Binder
		throw ParserException("DEFAULT VALUES clause is not supported!");
	}

	auto result = make_unique<InsertStatement>();
	if (stmt->withClause) {
		TransformCTE(reinterpret_cast<duckdb_libpgquery::PGWithClause *>(stmt->withClause), result->cte_map);
	}

	// first check if there are any columns specified
	if (stmt->cols) {
		for (auto c = stmt->cols->head; c != nullptr; c = lnext(c)) {
			auto target = (duckdb_libpgquery::PGResTarget *)(c->data.ptr_value);
			result->columns.emplace_back(target->name);
		}
	}

	// Grab and transform the returning columns from the parser.
	if (stmt->returningList) {
		Transformer::TransformExpressionList(*(stmt->returningList), result->returning_list);
	}
	result->select_statement = TransformSelect(stmt->selectStmt, false);

	auto qname = TransformQualifiedName(stmt->relation);
	result->table = qname.name;
	result->schema = qname.schema;

	if (stmt->onConflictClause) {
		if (stmt->onConflictAlias != duckdb_libpgquery::PG_ONCONFLICT_ALIAS_NONE) {
			// OR REPLACE | OR IGNORE are shorthands for the ON CONFLICT clause
			throw ParserException("You can not provide both OR REPLACE|IGNORE and an ON CONFLICT clause, please remove "
			                      "the first if you want to have more granual control");
		}
		result->on_conflict_info = TransformOnConflictClause(stmt->onConflictClause, result->schema);
		result->table_ref = TransformRangeVar(stmt->relation);
	}
	if (stmt->onConflictAlias != duckdb_libpgquery::PG_ONCONFLICT_ALIAS_NONE) {
		D_ASSERT(!stmt->onConflictClause);
		result->on_conflict_info = DummyOnConflictClause(stmt->onConflictAlias, result->schema);
		result->table_ref = TransformRangeVar(stmt->relation);
	}
	result->catalog = qname.catalog;
	return result;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<LoadStatement> Transformer::TransformLoad(duckdb_libpgquery::PGNode *node) {
	D_ASSERT(node->type == duckdb_libpgquery::T_PGLoadStmt);
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGLoadStmt *>(node);

	auto load_stmt = make_unique<LoadStatement>();
	auto load_info = make_unique<LoadInfo>();
	load_info->filename = std::string(stmt->filename);
	switch (stmt->load_type) {
	case duckdb_libpgquery::PG_LOAD_TYPE_LOAD:
		load_info->load_type = LoadType::LOAD;
		break;
	case duckdb_libpgquery::PG_LOAD_TYPE_INSTALL:
		load_info->load_type = LoadType::INSTALL;
		break;
	case duckdb_libpgquery::PG_LOAD_TYPE_FORCE_INSTALL:
		load_info->load_type = LoadType::FORCE_INSTALL;
		break;
	}
	load_stmt->info = std::move(load_info);
	return load_stmt;
}

} // namespace duckdb









namespace duckdb {

unique_ptr<SQLStatement> Transformer::TransformPragma(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGPragmaStmt *>(node);

	auto result = make_unique<PragmaStatement>();
	auto &info = *result->info;

	info.name = stmt->name;
	// parse the arguments, if any
	if (stmt->args) {
		for (auto cell = stmt->args->head; cell != nullptr; cell = cell->next) {
			auto node = reinterpret_cast<duckdb_libpgquery::PGNode *>(cell->data.ptr_value);
			auto expr = TransformExpression(node);

			if (expr->type == ExpressionType::COMPARE_EQUAL) {
				auto &comp = (ComparisonExpression &)*expr;
				if (comp.right->type != ExpressionType::VALUE_CONSTANT) {
					throw ParserException("Named parameter requires a constant on the RHS");
				}
				if (comp.left->type != ExpressionType::COLUMN_REF) {
					throw ParserException("Named parameter requires a column reference on the LHS");
				}
				auto &columnref = (ColumnRefExpression &)*comp.left;
				auto &constant = (ConstantExpression &)*comp.right;
				info.named_parameters[columnref.GetName()] = constant.value;
			} else if (node->type == duckdb_libpgquery::T_PGAConst) {
				auto constant = TransformConstant((duckdb_libpgquery::PGAConst *)node);
				info.parameters.push_back(((ConstantExpression &)*constant).value);
			} else if (expr->type == ExpressionType::COLUMN_REF) {
				auto &colref = (ColumnRefExpression &)*expr;
				if (!colref.IsQualified()) {
					info.parameters.emplace_back(colref.GetColumnName());
				} else {
					info.parameters.emplace_back(expr->ToString());
				}
			} else {
				info.parameters.emplace_back(expr->ToString());
			}
		}
	}
	// now parse the pragma type
	switch (stmt->kind) {
	case duckdb_libpgquery::PG_PRAGMA_TYPE_NOTHING: {
		if (!info.parameters.empty() || !info.named_parameters.empty()) {
			throw InternalException("PRAGMA statement that is not a call or assignment cannot contain parameters");
		}
		break;
	case duckdb_libpgquery::PG_PRAGMA_TYPE_ASSIGNMENT:
		if (info.parameters.size() != 1) {
			throw InternalException("PRAGMA statement with assignment should contain exactly one parameter");
		}
		if (!info.named_parameters.empty()) {
			throw InternalException("PRAGMA statement with assignment cannot have named parameters");
		}
		// SQLite does not distinguish between:
		// "PRAGMA table_info='integers'"
		// "PRAGMA table_info('integers')"
		// for compatibility, any pragmas that match the SQLite ones are parsed as calls
		case_insensitive_set_t sqlite_compat_pragmas {"table_info"};
		if (sqlite_compat_pragmas.find(info.name) != sqlite_compat_pragmas.end()) {
			break;
		}
		auto set_statement = make_unique<SetVariableStatement>(info.name, info.parameters[0], SetScope::AUTOMATIC);
		return std::move(set_statement);
	}
	case duckdb_libpgquery::PG_PRAGMA_TYPE_CALL:
		break;
	default:
		throw InternalException("Unknown pragma type");
	}

	return std::move(result);
}

} // namespace duckdb





namespace duckdb {

unique_ptr<PrepareStatement> Transformer::TransformPrepare(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGPrepareStmt *>(node);
	D_ASSERT(stmt);

	if (stmt->argtypes && stmt->argtypes->length > 0) {
		throw NotImplementedException("Prepared statement argument types are not supported, use CAST");
	}

	auto result = make_unique<PrepareStatement>();
	result->name = string(stmt->name);
	result->statement = TransformStatement(stmt->query);
	if (!result->statement->named_param_map.empty()) {
		throw NotImplementedException("Named parameters are not supported in this client yet");
	}
	SetParamCount(0);

	return result;
}

unique_ptr<ExecuteStatement> Transformer::TransformExecute(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGExecuteStmt *>(node);
	D_ASSERT(stmt);

	auto result = make_unique<ExecuteStatement>();
	result->name = string(stmt->name);

	if (stmt->params) {
		TransformExpressionList(*stmt->params, result->values);
	}
	for (auto &expr : result->values) {
		if (!expr->IsScalar()) {
			throw Exception("Only scalar parameters or NULL supported for EXECUTE");
		}
	}
	return result;
}

unique_ptr<DropStatement> Transformer::TransformDeallocate(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGDeallocateStmt *>(node);
	D_ASSERT(stmt);
	if (!stmt->name) {
		throw ParserException("DEALLOCATE requires a name");
	}

	auto result = make_unique<DropStatement>();
	result->info->type = CatalogType::PREPARED_STATEMENT;
	result->info->name = string(stmt->name);
	return result;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<AlterStatement> Transformer::TransformRename(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGRenameStmt *>(node);
	D_ASSERT(stmt);
	D_ASSERT(stmt->relation);

	unique_ptr<AlterInfo> info;

	AlterEntryData data;
	data.if_exists = stmt->missing_ok;
	data.catalog = stmt->relation->catalogname ? stmt->relation->catalogname : INVALID_CATALOG;
	data.schema = stmt->relation->schemaname ? stmt->relation->schemaname : INVALID_SCHEMA;
	if (stmt->relation->relname) {
		data.name = stmt->relation->relname;
	}
	if (stmt->relation->schemaname) {
	}
	// first we check the type of ALTER
	switch (stmt->renameType) {
	case duckdb_libpgquery::PG_OBJECT_COLUMN: {
		// change column name

		// get the old name and the new name
		string old_name = stmt->subname;
		string new_name = stmt->newname;
		info = make_unique<RenameColumnInfo>(std::move(data), old_name, new_name);
		break;
	}
	case duckdb_libpgquery::PG_OBJECT_TABLE: {
		// change table name
		string new_name = stmt->newname;
		info = make_unique<RenameTableInfo>(std::move(data), new_name);
		break;
	}

	case duckdb_libpgquery::PG_OBJECT_VIEW: {
		// change view name
		string new_name = stmt->newname;
		info = make_unique<RenameViewInfo>(std::move(data), new_name);
		break;
	}
	case duckdb_libpgquery::PG_OBJECT_DATABASE:
	default:
		throw NotImplementedException("Schema element not supported yet!");
	}
	D_ASSERT(info);

	auto result = make_unique<AlterStatement>();
	result->info = std::move(info);
	return result;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<SelectStatement> Transformer::TransformSelect(duckdb_libpgquery::PGNode *node, bool is_select) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGSelectStmt *>(node);
	auto result = make_unique<SelectStatement>();

	// Both Insert/Create Table As uses this.
	if (is_select) {
		if (stmt->intoClause) {
			throw ParserException("SELECT INTO not supported!");
		}
		if (stmt->lockingClause) {
			throw ParserException("SELECT locking clause is not supported!");
		}
	}

	result->node = TransformSelectNode(stmt);
	return result;
}

} // namespace duckdb








namespace duckdb {

unique_ptr<QueryNode> Transformer::TransformSelectNode(duckdb_libpgquery::PGSelectStmt *stmt) {
	D_ASSERT(stmt->type == duckdb_libpgquery::T_PGSelectStmt);
	auto stack_checker = StackCheck();

	unique_ptr<QueryNode> node;

	switch (stmt->op) {
	case duckdb_libpgquery::PG_SETOP_NONE: {
		node = make_unique<SelectNode>();
		auto result = (SelectNode *)node.get();
		if (stmt->withClause) {
			TransformCTE(reinterpret_cast<duckdb_libpgquery::PGWithClause *>(stmt->withClause), node->cte_map);
		}
		if (stmt->windowClause) {
			for (auto window_ele = stmt->windowClause->head; window_ele != nullptr; window_ele = window_ele->next) {
				auto window_def = reinterpret_cast<duckdb_libpgquery::PGWindowDef *>(window_ele->data.ptr_value);
				D_ASSERT(window_def);
				D_ASSERT(window_def->name);
				auto window_name = StringUtil::Lower(string(window_def->name));

				auto it = window_clauses.find(window_name);
				if (it != window_clauses.end()) {
					throw ParserException("window \"%s\" is already defined", window_name);
				}
				window_clauses[window_name] = window_def;
			}
		}

		// checks distinct clause
		if (stmt->distinctClause != nullptr) {
			auto modifier = make_unique<DistinctModifier>();
			// checks distinct on clause
			auto target = reinterpret_cast<duckdb_libpgquery::PGNode *>(stmt->distinctClause->head->data.ptr_value);
			if (target) {
				//  add the columns defined in the ON clause to the select list
				TransformExpressionList(*stmt->distinctClause, modifier->distinct_on_targets);
			}
			result->modifiers.push_back(std::move(modifier));
		}

		// do this early so the value lists also have a `FROM`
		if (stmt->valuesLists) {
			// VALUES list, create an ExpressionList
			D_ASSERT(!stmt->fromClause);
			result->from_table = TransformValuesList(stmt->valuesLists);
			result->select_list.push_back(make_unique<StarExpression>());
		} else {
			if (!stmt->targetList) {
				throw ParserException("SELECT clause without selection list");
			}
			// select list
			TransformExpressionList(*stmt->targetList, result->select_list);
			result->from_table = TransformFrom(stmt->fromClause);
		}

		// where
		result->where_clause = TransformExpression(stmt->whereClause);
		// group by
		TransformGroupBy(stmt->groupClause, *result);
		// having
		result->having = TransformExpression(stmt->havingClause);
		// qualify
		result->qualify = TransformExpression(stmt->qualifyClause);
		// sample
		result->sample = TransformSampleOptions(stmt->sampleOptions);
		break;
	}
	case duckdb_libpgquery::PG_SETOP_UNION:
	case duckdb_libpgquery::PG_SETOP_EXCEPT:
	case duckdb_libpgquery::PG_SETOP_INTERSECT:
	case duckdb_libpgquery::PG_SETOP_UNION_BY_NAME: {
		node = make_unique<SetOperationNode>();
		auto result = (SetOperationNode *)node.get();
		if (stmt->withClause) {
			TransformCTE(reinterpret_cast<duckdb_libpgquery::PGWithClause *>(stmt->withClause), node->cte_map);
		}
		result->left = TransformSelectNode(stmt->larg);
		result->right = TransformSelectNode(stmt->rarg);
		if (!result->left || !result->right) {
			throw Exception("Failed to transform setop children.");
		}

		bool select_distinct = true;
		switch (stmt->op) {
		case duckdb_libpgquery::PG_SETOP_UNION:
			select_distinct = !stmt->all;
			result->setop_type = SetOperationType::UNION;
			break;
		case duckdb_libpgquery::PG_SETOP_EXCEPT:
			result->setop_type = SetOperationType::EXCEPT;
			break;
		case duckdb_libpgquery::PG_SETOP_INTERSECT:
			result->setop_type = SetOperationType::INTERSECT;
			break;
		case duckdb_libpgquery::PG_SETOP_UNION_BY_NAME:
			select_distinct = !stmt->all;
			result->setop_type = SetOperationType::UNION_BY_NAME;
			break;
		default:
			throw Exception("Unexpected setop type");
		}
		if (select_distinct) {
			result->modifiers.push_back(make_unique<DistinctModifier>());
		}
		if (stmt->sampleOptions) {
			throw ParserException("SAMPLE clause is only allowed in regular SELECT statements");
		}
		break;
	}
	default:
		throw NotImplementedException("Statement type %d not implemented!", stmt->op);
	}
	// transform the common properties
	// both the set operations and the regular select can have an ORDER BY/LIMIT attached to them
	vector<OrderByNode> orders;
	TransformOrderBy(stmt->sortClause, orders);
	if (!orders.empty()) {
		auto order_modifier = make_unique<OrderModifier>();
		order_modifier->orders = std::move(orders);
		node->modifiers.push_back(std::move(order_modifier));
	}
	if (stmt->limitCount || stmt->limitOffset) {
		if (stmt->limitCount && stmt->limitCount->type == duckdb_libpgquery::T_PGLimitPercent) {
			auto limit_percent_modifier = make_unique<LimitPercentModifier>();
			auto expr_node = reinterpret_cast<duckdb_libpgquery::PGLimitPercent *>(stmt->limitCount)->limit_percent;
			limit_percent_modifier->limit = TransformExpression(expr_node);
			if (stmt->limitOffset) {
				limit_percent_modifier->offset = TransformExpression(stmt->limitOffset);
			}
			node->modifiers.push_back(std::move(limit_percent_modifier));
		} else {
			auto limit_modifier = make_unique<LimitModifier>();
			if (stmt->limitCount) {
				limit_modifier->limit = TransformExpression(stmt->limitCount);
			}
			if (stmt->limitOffset) {
				limit_modifier->offset = TransformExpression(stmt->limitOffset);
			}
			node->modifiers.push_back(std::move(limit_modifier));
		}
	}
	return node;
}

} // namespace duckdb





namespace duckdb {

namespace {

SetScope ToSetScope(duckdb_libpgquery::VariableSetScope pg_scope) {
	switch (pg_scope) {
	case duckdb_libpgquery::VariableSetScope::VAR_SET_SCOPE_LOCAL:
		return SetScope::LOCAL;
	case duckdb_libpgquery::VariableSetScope::VAR_SET_SCOPE_SESSION:
		return SetScope::SESSION;
	case duckdb_libpgquery::VariableSetScope::VAR_SET_SCOPE_GLOBAL:
		return SetScope::GLOBAL;
	case duckdb_libpgquery::VariableSetScope::VAR_SET_SCOPE_DEFAULT:
		return SetScope::AUTOMATIC;
	default:
		throw InternalException("Unexpected pg_scope: %d", pg_scope);
	}
}

SetType ToSetType(duckdb_libpgquery::VariableSetKind pg_kind) {
	switch (pg_kind) {
	case duckdb_libpgquery::VariableSetKind::VAR_SET_VALUE:
		return SetType::SET;
	case duckdb_libpgquery::VariableSetKind::VAR_RESET:
		return SetType::RESET;
	default:
		throw NotImplementedException("Can only SET or RESET a variable");
	}
}

} // namespace

unique_ptr<SetStatement> Transformer::TransformSetVariable(duckdb_libpgquery::PGVariableSetStmt *stmt) {
	D_ASSERT(stmt->kind == duckdb_libpgquery::VariableSetKind::VAR_SET_VALUE);

	if (stmt->scope == duckdb_libpgquery::VariableSetScope::VAR_SET_SCOPE_LOCAL) {
		throw NotImplementedException("SET LOCAL is not implemented.");
	}

	auto name = std::string(stmt->name);
	D_ASSERT(!name.empty()); // parser protect us!
	if (stmt->args->length != 1) {
		throw ParserException("SET needs a single scalar value parameter");
	}
	D_ASSERT(stmt->args->head && stmt->args->head->data.ptr_value);
	D_ASSERT(((duckdb_libpgquery::PGNode *)stmt->args->head->data.ptr_value)->type == duckdb_libpgquery::T_PGAConst);

	auto value = TransformValue(((duckdb_libpgquery::PGAConst *)stmt->args->head->data.ptr_value)->val)->value;

	return make_unique<SetVariableStatement>(name, value, ToSetScope(stmt->scope));
}

unique_ptr<SetStatement> Transformer::TransformResetVariable(duckdb_libpgquery::PGVariableSetStmt *stmt) {
	D_ASSERT(stmt->kind == duckdb_libpgquery::VariableSetKind::VAR_RESET);

	if (stmt->scope == duckdb_libpgquery::VariableSetScope::VAR_SET_SCOPE_LOCAL) {
		throw NotImplementedException("RESET LOCAL is not implemented.");
	}

	auto name = std::string(stmt->name);
	D_ASSERT(!name.empty()); // parser protect us!

	return make_unique<ResetVariableStatement>(name, ToSetScope(stmt->scope));
}

unique_ptr<SetStatement> Transformer::TransformSet(duckdb_libpgquery::PGNode *node) {
	D_ASSERT(node->type == duckdb_libpgquery::T_PGVariableSetStmt);
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGVariableSetStmt *>(node);

	SetType set_type = ToSetType(stmt->kind);

	switch (set_type) {
	case SetType::SET:
		return TransformSetVariable(stmt);
	case SetType::RESET:
		return TransformResetVariable(stmt);
	default:
		throw NotImplementedException("Type not implemented for SetType");
	}
}

} // namespace duckdb







namespace duckdb {

static void TransformShowName(unique_ptr<PragmaStatement> &result, const string &name) {
	auto &info = *result->info;
	auto lname = StringUtil::Lower(name);

	if (lname == "\"databases\"") {
		info.name = "show_databases";
	} else if (lname == "\"tables\"") {
		// show all tables
		info.name = "show_tables";
	} else if (lname == "__show_tables_expanded") {
		info.name = "show_tables_expanded";
	} else {
		// show one specific table
		info.name = "show";
		info.parameters.emplace_back(name);
	}
}

unique_ptr<SQLStatement> Transformer::TransformShow(duckdb_libpgquery::PGNode *node) {
	// we transform SHOW x into PRAGMA SHOW('x')

	auto stmt = reinterpret_cast<duckdb_libpgquery::PGVariableShowStmt *>(node);
	if (stmt->is_summary) {
		auto result = make_unique<ShowStatement>();
		auto &info = *result->info;
		info.is_summary = stmt->is_summary;

		auto select = make_unique<SelectNode>();
		select->select_list.push_back(make_unique<StarExpression>());
		auto basetable = make_unique<BaseTableRef>();
		auto qualified_name = QualifiedName::Parse(stmt->name);
		basetable->schema_name = qualified_name.schema;
		basetable->table_name = qualified_name.name;
		select->from_table = std::move(basetable);

		info.query = std::move(select);
		return std::move(result);
	}

	auto result = make_unique<PragmaStatement>();

	auto show_name = stmt->name;
	TransformShowName(result, show_name);
	return std::move(result);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<ShowStatement> Transformer::TransformShowSelect(duckdb_libpgquery::PGNode *node) {
	// we capture the select statement of SHOW
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGVariableShowSelectStmt *>(node);
	auto select_stmt = reinterpret_cast<duckdb_libpgquery::PGSelectStmt *>(stmt->stmt);

	auto result = make_unique<ShowStatement>();
	auto &info = *result->info;
	info.is_summary = stmt->is_summary;

	info.query = TransformSelectNode(select_stmt);

	return result;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<TransactionStatement> Transformer::TransformTransaction(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGTransactionStmt *>(node);
	D_ASSERT(stmt);
	switch (stmt->kind) {
	case duckdb_libpgquery::PG_TRANS_STMT_BEGIN:
	case duckdb_libpgquery::PG_TRANS_STMT_START:
		return make_unique<TransactionStatement>(TransactionType::BEGIN_TRANSACTION);
	case duckdb_libpgquery::PG_TRANS_STMT_COMMIT:
		return make_unique<TransactionStatement>(TransactionType::COMMIT);
	case duckdb_libpgquery::PG_TRANS_STMT_ROLLBACK:
		return make_unique<TransactionStatement>(TransactionType::ROLLBACK);
	default:
		throw NotImplementedException("Transaction type %d not implemented yet", stmt->kind);
	}
}

} // namespace duckdb



namespace duckdb {

unique_ptr<UpdateSetInfo> Transformer::TransformUpdateSetInfo(duckdb_libpgquery::PGList *target_list,
                                                              duckdb_libpgquery::PGNode *where_clause) {
	auto result = make_unique<UpdateSetInfo>();

	auto root = target_list;
	for (auto cell = root->head; cell != nullptr; cell = cell->next) {
		auto target = (duckdb_libpgquery::PGResTarget *)(cell->data.ptr_value);
		result->columns.emplace_back(target->name);
		result->expressions.push_back(TransformExpression(target->val));
	}
	result->condition = TransformExpression(where_clause);
	return result;
}

unique_ptr<UpdateStatement> Transformer::TransformUpdate(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGUpdateStmt *>(node);
	D_ASSERT(stmt);

	auto result = make_unique<UpdateStatement>();
	if (stmt->withClause) {
		TransformCTE(reinterpret_cast<duckdb_libpgquery::PGWithClause *>(stmt->withClause), result->cte_map);
	}

	result->table = TransformRangeVar(stmt->relation);
	if (stmt->fromClause) {
		result->from_table = TransformFrom(stmt->fromClause);
	}

	result->set_info = TransformUpdateSetInfo(stmt->targetList, stmt->whereClause);

	// Grab and transform the returning columns from the parser.
	if (stmt->returningList) {
		Transformer::TransformExpressionList(*(stmt->returningList), result->returning_list);
	}

	return result;
}

} // namespace duckdb





namespace duckdb {

OnConflictAction TransformOnConflictAction(duckdb_libpgquery::PGOnConflictClause *on_conflict) {
	if (!on_conflict) {
		return OnConflictAction::THROW;
	}
	switch (on_conflict->action) {
	case duckdb_libpgquery::PG_ONCONFLICT_NONE:
		return OnConflictAction::THROW;
	case duckdb_libpgquery::PG_ONCONFLICT_NOTHING:
		return OnConflictAction::NOTHING;
	case duckdb_libpgquery::PG_ONCONFLICT_UPDATE:
		return OnConflictAction::UPDATE;
	default:
		throw InternalException("Type not implemented for OnConflictAction");
	}
}

vector<string> TransformConflictTarget(duckdb_libpgquery::PGList *list) {
	vector<string> columns;
	for (auto cell = list->head; cell != nullptr; cell = cell->next) {
		auto index_element = (duckdb_libpgquery::PGIndexElem *)cell->data.ptr_value;
		if (index_element->collation) {
			throw NotImplementedException("Index with collation not supported yet!");
		}
		if (index_element->opclass) {
			throw NotImplementedException("Index with opclass not supported yet!");
		}
		if (!index_element->name) {
			throw NotImplementedException("Non-column index element not supported yet!");
		}
		if (index_element->nulls_ordering) {
			throw NotImplementedException("Index with null_ordering not supported yet!");
		}
		if (index_element->ordering) {
			throw NotImplementedException("Index with ordering not supported yet!");
		}
		columns.emplace_back(index_element->name);
	}
	return columns;
}

unique_ptr<OnConflictInfo> Transformer::DummyOnConflictClause(duckdb_libpgquery::PGOnConflictActionAlias type,
                                                              const string &relname) {
	switch (type) {
	case duckdb_libpgquery::PGOnConflictActionAlias::PG_ONCONFLICT_ALIAS_REPLACE: {
		// This can not be fully resolved yet until the bind stage
		auto result = make_unique<OnConflictInfo>();
		result->action_type = OnConflictAction::REPLACE;
		return result;
	}
	case duckdb_libpgquery::PGOnConflictActionAlias::PG_ONCONFLICT_ALIAS_IGNORE: {
		// We can just fully replace this with DO NOTHING, and be done with it
		auto result = make_unique<OnConflictInfo>();
		result->action_type = OnConflictAction::NOTHING;
		return result;
	}
	default: {
		throw InternalException("Type not implemented for PGOnConflictActionAlias");
	}
	}
}

unique_ptr<OnConflictInfo> Transformer::TransformOnConflictClause(duckdb_libpgquery::PGOnConflictClause *node,
                                                                  const string &relname) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGOnConflictClause *>(node);
	D_ASSERT(stmt);

	auto result = make_unique<OnConflictInfo>();
	result->action_type = TransformOnConflictAction(stmt);
	if (stmt->infer) {
		// A filter for the ON CONFLICT ... is specified
		if (stmt->infer->indexElems) {
			// Columns are specified
			result->indexed_columns = TransformConflictTarget(stmt->infer->indexElems);
			if (stmt->infer->whereClause) {
				result->condition = TransformExpression(stmt->infer->whereClause);
			}
		} else {
			throw NotImplementedException("ON CONSTRAINT conflict target is not supported yet");
		}
	}

	if (result->action_type == OnConflictAction::UPDATE) {
		result->set_info = TransformUpdateSetInfo(stmt->targetList, stmt->whereClause);
	}
	return result;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<SetStatement> Transformer::TransformUse(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGUseStmt *>(node);
	auto qualified_name = TransformQualifiedName(stmt->name);
	if (!IsInvalidCatalog(qualified_name.catalog)) {
		throw ParserException("Expected \"USE database\" or \"USE database.schema\"");
	}
	string name;
	if (IsInvalidSchema(qualified_name.schema)) {
		name = qualified_name.name;
	} else {
		name = qualified_name.schema + "." + qualified_name.name;
	}
	return make_unique<SetVariableStatement>("schema", std::move(name), SetScope::AUTOMATIC);
}

} // namespace duckdb



namespace duckdb {

VacuumOptions ParseOptions(int options) {
	VacuumOptions result;
	if (options & duckdb_libpgquery::PGVacuumOption::PG_VACOPT_VACUUM) {
		result.vacuum = true;
	}
	if (options & duckdb_libpgquery::PGVacuumOption::PG_VACOPT_ANALYZE) {
		result.analyze = true;
	}
	if (options & duckdb_libpgquery::PGVacuumOption::PG_VACOPT_VERBOSE) {
		throw NotImplementedException("Verbose vacuum option");
	}
	if (options & duckdb_libpgquery::PGVacuumOption::PG_VACOPT_FREEZE) {
		throw NotImplementedException("Freeze vacuum option");
	}
	if (options & duckdb_libpgquery::PGVacuumOption::PG_VACOPT_FULL) {
		throw NotImplementedException("Full vacuum option");
	}
	if (options & duckdb_libpgquery::PGVacuumOption::PG_VACOPT_NOWAIT) {
		throw NotImplementedException("No Wait vacuum option");
	}
	if (options & duckdb_libpgquery::PGVacuumOption::PG_VACOPT_SKIPTOAST) {
		throw NotImplementedException("Skip Toast vacuum option");
	}
	if (options & duckdb_libpgquery::PGVacuumOption::PG_VACOPT_DISABLE_PAGE_SKIPPING) {
		throw NotImplementedException("Disable Page Skipping vacuum option");
	}
	return result;
}

unique_ptr<SQLStatement> Transformer::TransformVacuum(duckdb_libpgquery::PGNode *node) {
	auto stmt = reinterpret_cast<duckdb_libpgquery::PGVacuumStmt *>(node);
	D_ASSERT(stmt);

	auto result = make_unique<VacuumStatement>(ParseOptions(stmt->options));

	if (stmt->relation) {
		result->info->ref = TransformRangeVar(stmt->relation);
		result->info->has_table = true;
	}

	if (stmt->va_cols) {
		D_ASSERT(result->info->has_table);
		for (auto col_node = stmt->va_cols->head; col_node != nullptr; col_node = col_node->next) {
			result->info->columns.emplace_back(
			    reinterpret_cast<duckdb_libpgquery::PGValue *>(col_node->data.ptr_value)->val.str);
		}
	}

	return std::move(result);
}

} // namespace duckdb



namespace duckdb {

unique_ptr<TableRef> Transformer::TransformRangeVar(duckdb_libpgquery::PGRangeVar *root) {
	auto result = make_unique<BaseTableRef>();

	result->alias = TransformAlias(root->alias, result->column_name_alias);
	if (root->relname) {
		result->table_name = root->relname;
	}
	if (root->catalogname) {
		result->catalog_name = root->catalogname;
	}
	if (root->schemaname) {
		result->schema_name = root->schemaname;
	}
	if (root->sample) {
		result->sample = TransformSampleOptions(root->sample);
	}
	result->query_location = root->location;
	return std::move(result);
}

QualifiedName Transformer::TransformQualifiedName(duckdb_libpgquery::PGRangeVar *root) {
	QualifiedName qname;
	if (root->catalogname) {
		qname.catalog = root->catalogname;
	} else {
		qname.catalog = INVALID_CATALOG;
	}
	if (root->schemaname) {
		qname.schema = root->schemaname;
	} else {
		qname.schema = INVALID_SCHEMA;
	}
	if (root->relname) {
		qname.name = root->relname;
	} else {
		qname.name = string();
	}
	return qname;
}

} // namespace duckdb




namespace duckdb {

unique_ptr<TableRef> Transformer::TransformFrom(duckdb_libpgquery::PGList *root) {
	if (!root) {
		return make_unique<EmptyTableRef>();
	}

	if (root->length > 1) {
		// Cross Product
		auto result = make_unique<JoinRef>(JoinRefType::CROSS);
		JoinRef *cur_root = result.get();
		idx_t list_size = 0;
		for (auto node = root->head; node != nullptr; node = node->next) {
			auto n = reinterpret_cast<duckdb_libpgquery::PGNode *>(node->data.ptr_value);
			unique_ptr<TableRef> next = TransformTableRefNode(n);
			if (!cur_root->left) {
				cur_root->left = std::move(next);
			} else if (!cur_root->right) {
				cur_root->right = std::move(next);
			} else {
				auto old_res = std::move(result);
				result = make_unique<JoinRef>(JoinRefType::CROSS);
				result->left = std::move(old_res);
				result->right = std::move(next);
				cur_root = result.get();
			}
			list_size++;
			StackCheck(list_size);
		}
		return std::move(result);
	}

	auto n = reinterpret_cast<duckdb_libpgquery::PGNode *>(root->head->data.ptr_value);
	return TransformTableRefNode(n);
}

} // namespace duckdb





namespace duckdb {

unique_ptr<TableRef> Transformer::TransformJoin(duckdb_libpgquery::PGJoinExpr *root) {
	auto result = make_unique<JoinRef>(JoinRefType::REGULAR);
	switch (root->jointype) {
	case duckdb_libpgquery::PG_JOIN_INNER: {
		result->type = JoinType::INNER;
		break;
	}
	case duckdb_libpgquery::PG_JOIN_LEFT: {
		result->type = JoinType::LEFT;
		break;
	}
	case duckdb_libpgquery::PG_JOIN_FULL: {
		result->type = JoinType::OUTER;
		break;
	}
	case duckdb_libpgquery::PG_JOIN_RIGHT: {
		result->type = JoinType::RIGHT;
		break;
	}
	case duckdb_libpgquery::PG_JOIN_SEMI: {
		result->type = JoinType::SEMI;
		break;
	}
	case duckdb_libpgquery::PG_JOIN_POSITION: {
		result->ref_type = JoinRefType::POSITIONAL;
		break;
	}
	default: {
		throw NotImplementedException("Join type %d not supported\n", root->jointype);
	}
	}

	// Check the type of left arg and right arg before transform
	result->left = TransformTableRefNode(root->larg);
	result->right = TransformTableRefNode(root->rarg);
	if (root->isNatural) {
		result->ref_type = JoinRefType::NATURAL;
	}
	result->query_location = root->location;

	if (root->usingClause && root->usingClause->length > 0) {
		// usingClause is a list of strings
		for (auto node = root->usingClause->head; node != nullptr; node = node->next) {
			auto target = reinterpret_cast<duckdb_libpgquery::PGNode *>(node->data.ptr_value);
			D_ASSERT(target->type == duckdb_libpgquery::T_PGString);
			auto column_name = string(reinterpret_cast<duckdb_libpgquery::PGValue *>(target)->val.str);
			result->using_columns.push_back(column_name);
		}
		return std::move(result);
	}

	if (!root->quals && result->using_columns.empty() && result->ref_type == JoinRefType::REGULAR) { // CROSS PRODUCT
		result->ref_type = JoinRefType::CROSS;
	}
	result->condition = TransformExpression(root->quals);
	return std::move(result);
}

} // namespace duckdb



namespace duckdb {

unique_ptr<TableRef> Transformer::TransformRangeSubselect(duckdb_libpgquery::PGRangeSubselect *root) {
	Transformer subquery_transformer(this);
	auto subquery = subquery_transformer.TransformSelect(root->subquery);
	if (!subquery) {
		return nullptr;
	}
	auto result = make_unique<SubqueryRef>(std::move(subquery));
	result->alias = TransformAlias(root->alias, result->column_name_alias);
	if (root->sample) {
		result->sample = TransformSampleOptions(root->sample);
	}
	return std::move(result);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<TableRef> Transformer::TransformRangeFunction(duckdb_libpgquery::PGRangeFunction *root) {
	if (root->ordinality) {
		throw NotImplementedException("WITH ORDINALITY not implemented");
	}
	if (root->is_rowsfrom) {
		throw NotImplementedException("ROWS FROM() not implemented");
	}
	if (root->functions->length != 1) {
		throw NotImplementedException("Need exactly one function");
	}
	auto function_sublist = (duckdb_libpgquery::PGList *)root->functions->head->data.ptr_value;
	D_ASSERT(function_sublist->length == 2);
	auto call_tree = (duckdb_libpgquery::PGNode *)function_sublist->head->data.ptr_value;
	auto coldef = function_sublist->head->next->data.ptr_value;

	if (coldef) {
		throw NotImplementedException("Explicit column definition not supported yet");
	}
	// transform the function call
	auto result = make_unique<TableFunctionRef>();
	switch (call_tree->type) {
	case duckdb_libpgquery::T_PGFuncCall: {
		auto func_call = (duckdb_libpgquery::PGFuncCall *)call_tree;
		result->function = TransformFuncCall(func_call);
		result->query_location = func_call->location;
		break;
	}
	case duckdb_libpgquery::T_PGSQLValueFunction:
		result->function = TransformSQLValueFunction((duckdb_libpgquery::PGSQLValueFunction *)call_tree);
		break;
	default:
		throw ParserException("Not a function call or value function");
	}
	result->alias = TransformAlias(root->alias, result->column_name_alias);
	if (root->sample) {
		result->sample = TransformSampleOptions(root->sample);
	}
	return std::move(result);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<TableRef> Transformer::TransformTableRefNode(duckdb_libpgquery::PGNode *n) {
	auto stack_checker = StackCheck();

	switch (n->type) {
	case duckdb_libpgquery::T_PGRangeVar:
		return TransformRangeVar(reinterpret_cast<duckdb_libpgquery::PGRangeVar *>(n));
	case duckdb_libpgquery::T_PGJoinExpr:
		return TransformJoin(reinterpret_cast<duckdb_libpgquery::PGJoinExpr *>(n));
	case duckdb_libpgquery::T_PGRangeSubselect:
		return TransformRangeSubselect(reinterpret_cast<duckdb_libpgquery::PGRangeSubselect *>(n));
	case duckdb_libpgquery::T_PGRangeFunction:
		return TransformRangeFunction(reinterpret_cast<duckdb_libpgquery::PGRangeFunction *>(n));
	default:
		throw NotImplementedException("From Type %d not supported", n->type);
	}
}

} // namespace duckdb






namespace duckdb {

StackChecker::StackChecker(Transformer &transformer_p, idx_t stack_usage_p)
    : transformer(transformer_p), stack_usage(stack_usage_p) {
	transformer.stack_depth += stack_usage;
}

StackChecker::~StackChecker() {
	transformer.stack_depth -= stack_usage;
}

StackChecker::StackChecker(StackChecker &&other) noexcept
    : transformer(other.transformer), stack_usage(other.stack_usage) {
	other.stack_usage = 0;
}

Transformer::Transformer(idx_t max_expression_depth_p)
    : parent(nullptr), max_expression_depth(max_expression_depth_p), stack_depth(DConstants::INVALID_INDEX) {
}

Transformer::Transformer(Transformer *parent)
    : parent(parent), max_expression_depth(parent->max_expression_depth), stack_depth(DConstants::INVALID_INDEX) {
}

bool Transformer::TransformParseTree(duckdb_libpgquery::PGList *tree, vector<unique_ptr<SQLStatement>> &statements) {
	InitializeStackCheck();
	for (auto entry = tree->head; entry != nullptr; entry = entry->next) {
		SetParamCount(0);
		auto stmt = TransformStatement((duckdb_libpgquery::PGNode *)entry->data.ptr_value);
		D_ASSERT(stmt);
		stmt->n_param = ParamCount();
		statements.push_back(std::move(stmt));
	}
	return true;
}

void Transformer::InitializeStackCheck() {
	stack_depth = 0;
}

StackChecker Transformer::StackCheck(idx_t extra_stack) {
	auto node = this;
	while (node->parent) {
		node = node->parent;
	}
	D_ASSERT(node->stack_depth != DConstants::INVALID_INDEX);
	if (node->stack_depth + extra_stack >= max_expression_depth) {
		throw ParserException("Max expression depth limit of %lld exceeded. Use \"SET max_expression_depth TO x\" to "
		                      "increase the maximum expression depth.",
		                      max_expression_depth);
	}
	return StackChecker(*node, extra_stack);
}

unique_ptr<SQLStatement> Transformer::TransformStatement(duckdb_libpgquery::PGNode *stmt) {
	auto result = TransformStatementInternal(stmt);
	result->n_param = ParamCount();
	if (!named_param_map.empty()) {
		// Avoid overriding a previous move with nothing
		result->named_param_map = std::move(named_param_map);
	}
	return result;
}

unique_ptr<SQLStatement> Transformer::TransformStatementInternal(duckdb_libpgquery::PGNode *stmt) {
	switch (stmt->type) {
	case duckdb_libpgquery::T_PGRawStmt: {
		auto raw_stmt = (duckdb_libpgquery::PGRawStmt *)stmt;
		auto result = TransformStatement(raw_stmt->stmt);
		if (result) {
			result->stmt_location = raw_stmt->stmt_location;
			result->stmt_length = raw_stmt->stmt_len;
		}
		return result;
	}
	case duckdb_libpgquery::T_PGSelectStmt:
		return TransformSelect(stmt);
	case duckdb_libpgquery::T_PGCreateStmt:
		return TransformCreateTable(stmt);
	case duckdb_libpgquery::T_PGCreateSchemaStmt:
		return TransformCreateSchema(stmt);
	case duckdb_libpgquery::T_PGViewStmt:
		return TransformCreateView(stmt);
	case duckdb_libpgquery::T_PGCreateSeqStmt:
		return TransformCreateSequence(stmt);
	case duckdb_libpgquery::T_PGCreateFunctionStmt:
		return TransformCreateFunction(stmt);
	case duckdb_libpgquery::T_PGDropStmt:
		return TransformDrop(stmt);
	case duckdb_libpgquery::T_PGInsertStmt:
		return TransformInsert(stmt);
	case duckdb_libpgquery::T_PGCopyStmt:
		return TransformCopy(stmt);
	case duckdb_libpgquery::T_PGTransactionStmt:
		return TransformTransaction(stmt);
	case duckdb_libpgquery::T_PGDeleteStmt:
		return TransformDelete(stmt);
	case duckdb_libpgquery::T_PGUpdateStmt:
		return TransformUpdate(stmt);
	case duckdb_libpgquery::T_PGIndexStmt:
		return TransformCreateIndex(stmt);
	case duckdb_libpgquery::T_PGAlterTableStmt:
		return TransformAlter(stmt);
	case duckdb_libpgquery::T_PGRenameStmt:
		return TransformRename(stmt);
	case duckdb_libpgquery::T_PGPrepareStmt:
		return TransformPrepare(stmt);
	case duckdb_libpgquery::T_PGExecuteStmt:
		return TransformExecute(stmt);
	case duckdb_libpgquery::T_PGDeallocateStmt:
		return TransformDeallocate(stmt);
	case duckdb_libpgquery::T_PGCreateTableAsStmt:
		return TransformCreateTableAs(stmt);
	case duckdb_libpgquery::T_PGPragmaStmt:
		return TransformPragma(stmt);
	case duckdb_libpgquery::T_PGExportStmt:
		return TransformExport(stmt);
	case duckdb_libpgquery::T_PGImportStmt:
		return TransformImport(stmt);
	case duckdb_libpgquery::T_PGExplainStmt:
		return TransformExplain(stmt);
	case duckdb_libpgquery::T_PGVacuumStmt:
		return TransformVacuum(stmt);
	case duckdb_libpgquery::T_PGVariableShowStmt:
		return TransformShow(stmt);
	case duckdb_libpgquery::T_PGVariableShowSelectStmt:
		return TransformShowSelect(stmt);
	case duckdb_libpgquery::T_PGCallStmt:
		return TransformCall(stmt);
	case duckdb_libpgquery::T_PGVariableSetStmt:
		return TransformSet(stmt);
	case duckdb_libpgquery::T_PGCheckPointStmt:
		return TransformCheckpoint(stmt);
	case duckdb_libpgquery::T_PGLoadStmt:
		return TransformLoad(stmt);
	case duckdb_libpgquery::T_PGCreateTypeStmt:
		return TransformCreateType(stmt);
	case duckdb_libpgquery::T_PGAlterSeqStmt:
		return TransformAlterSequence(stmt);
	case duckdb_libpgquery::T_PGAttachStmt:
		return TransformAttach(stmt);
	case duckdb_libpgquery::T_PGUseStmt:
		return TransformUse(stmt);
	case duckdb_libpgquery::T_PGCreateDatabaseStmt:
		return TransformCreateDatabase(stmt);
	default:
		throw NotImplementedException(NodetypeToString(stmt->type));
	}
	return nullptr;
}

} // namespace duckdb

















#include <algorithm>

namespace duckdb {

string BindContext::GetMatchingBinding(const string &column_name) {
	string result;
	for (auto &kv : bindings) {
		auto binding = kv.second.get();
		auto is_using_binding = GetUsingBinding(column_name, kv.first);
		if (is_using_binding) {
			continue;
		}
		if (binding->HasMatchingBinding(column_name)) {
			if (!result.empty() || is_using_binding) {
				throw BinderException("Ambiguous reference to column name \"%s\" (use: \"%s.%s\" "
				                      "or \"%s.%s\")",
				                      column_name, result, column_name, kv.first, column_name);
			}
			result = kv.first;
		}
	}
	return result;
}

vector<string> BindContext::GetSimilarBindings(const string &column_name) {
	vector<pair<string, idx_t>> scores;
	for (auto &kv : bindings) {
		auto binding = kv.second.get();
		for (auto &name : binding->names) {
			idx_t distance = StringUtil::LevenshteinDistance(name, column_name);
			scores.emplace_back(binding->alias + "." + name, distance);
		}
	}
	return StringUtil::TopNStrings(scores);
}

void BindContext::AddUsingBinding(const string &column_name, UsingColumnSet *set) {
	using_columns[column_name].insert(set);
}

void BindContext::AddUsingBindingSet(unique_ptr<UsingColumnSet> set) {
	using_column_sets.push_back(std::move(set));
}

bool BindContext::FindUsingBinding(const string &column_name, unordered_set<UsingColumnSet *> **out) {
	auto entry = using_columns.find(column_name);
	if (entry != using_columns.end()) {
		*out = &entry->second;
		return true;
	}
	return false;
}

UsingColumnSet *BindContext::GetUsingBinding(const string &column_name) {
	unordered_set<UsingColumnSet *> *using_bindings;
	if (!FindUsingBinding(column_name, &using_bindings)) {
		return nullptr;
	}
	if (using_bindings->size() > 1) {
		string error = "Ambiguous column reference: column \"" + column_name + "\" can refer to either:\n";
		for (auto &using_set : *using_bindings) {
			string result_bindings;
			for (auto &binding : using_set->bindings) {
				if (result_bindings.empty()) {
					result_bindings = "[";
				} else {
					result_bindings += ", ";
				}
				result_bindings += binding;
				result_bindings += ".";
				result_bindings += GetActualColumnName(binding, column_name);
			}
			error += result_bindings + "]";
		}
		throw BinderException(error);
	}
	for (auto &using_set : *using_bindings) {
		return using_set;
	}
	throw InternalException("Using binding found but no entries");
}

UsingColumnSet *BindContext::GetUsingBinding(const string &column_name, const string &binding_name) {
	if (binding_name.empty()) {
		throw InternalException("GetUsingBinding: expected non-empty binding_name");
	}
	unordered_set<UsingColumnSet *> *using_bindings;
	if (!FindUsingBinding(column_name, &using_bindings)) {
		return nullptr;
	}
	for (auto &using_set : *using_bindings) {
		auto &bindings = using_set->bindings;
		if (bindings.find(binding_name) != bindings.end()) {
			return using_set;
		}
	}
	return nullptr;
}

void BindContext::RemoveUsingBinding(const string &column_name, UsingColumnSet *set) {
	if (!set) {
		return;
	}
	auto entry = using_columns.find(column_name);
	if (entry == using_columns.end()) {
		throw InternalException("Attempting to remove using binding that is not there");
	}
	auto &bindings = entry->second;
	if (bindings.find(set) != bindings.end()) {
		bindings.erase(set);
	}
	if (bindings.empty()) {
		using_columns.erase(column_name);
	}
}

void BindContext::TransferUsingBinding(BindContext &current_context, UsingColumnSet *current_set,
                                       UsingColumnSet *new_set, const string &binding, const string &using_column) {
	AddUsingBinding(using_column, new_set);
	current_context.RemoveUsingBinding(using_column, current_set);
}

string BindContext::GetActualColumnName(const string &binding_name, const string &column_name) {
	string error;
	auto binding = GetBinding(binding_name, error);
	if (!binding) {
		throw InternalException("No binding with name \"%s\"", binding_name);
	}
	column_t binding_index;
	if (!binding->TryGetBindingIndex(column_name, binding_index)) { // LCOV_EXCL_START
		throw InternalException("Binding with name \"%s\" does not have a column named \"%s\"", binding_name,
		                        column_name);
	} // LCOV_EXCL_STOP
	return binding->names[binding_index];
}

unordered_set<string> BindContext::GetMatchingBindings(const string &column_name) {
	unordered_set<string> result;
	for (auto &kv : bindings) {
		auto binding = kv.second.get();
		if (binding->HasMatchingBinding(column_name)) {
			result.insert(kv.first);
		}
	}
	return result;
}

unique_ptr<ParsedExpression> BindContext::ExpandGeneratedColumn(const string &table_name, const string &column_name) {
	string error_message;

	auto binding = GetBinding(table_name, error_message);
	D_ASSERT(binding);
	auto &table_binding = *(TableBinding *)binding;
	auto result = table_binding.ExpandGeneratedColumn(column_name);
	result->alias = column_name;
	return result;
}

unique_ptr<ParsedExpression> BindContext::CreateColumnReference(const string &table_name, const string &column_name) {
	string schema_name;
	return CreateColumnReference(schema_name, table_name, column_name);
}

static bool ColumnIsGenerated(Binding *binding, column_t index) {
	if (binding->binding_type != BindingType::TABLE) {
		return false;
	}
	auto table_binding = (TableBinding *)binding;
	auto catalog_entry = table_binding->GetStandardEntry();
	if (!catalog_entry) {
		return false;
	}
	if (index == COLUMN_IDENTIFIER_ROW_ID) {
		return false;
	}
	D_ASSERT(catalog_entry->type == CatalogType::TABLE_ENTRY);
	auto table_entry = (TableCatalogEntry *)catalog_entry;
	return table_entry->GetColumn(LogicalIndex(index)).Generated();
}

unique_ptr<ParsedExpression> BindContext::CreateColumnReference(const string &catalog_name, const string &schema_name,
                                                                const string &table_name, const string &column_name) {
	string error_message;
	vector<string> names;
	if (!catalog_name.empty()) {
		names.push_back(catalog_name);
	}
	if (!schema_name.empty()) {
		names.push_back(schema_name);
	}
	names.push_back(table_name);
	names.push_back(column_name);

	auto result = make_unique<ColumnRefExpression>(std::move(names));
	auto binding = GetBinding(table_name, error_message);
	if (!binding) {
		return std::move(result);
	}
	auto column_index = binding->GetBindingIndex(column_name);
	if (ColumnIsGenerated(binding, column_index)) {
		return ExpandGeneratedColumn(table_name, column_name);
	} else if (column_index < binding->names.size() && binding->names[column_index] != column_name) {
		// because of case insensitivity in the binder we rename the column to the original name
		// as it appears in the binding itself
		result->alias = binding->names[column_index];
	}
	return std::move(result);
}

unique_ptr<ParsedExpression> BindContext::CreateColumnReference(const string &schema_name, const string &table_name,
                                                                const string &column_name) {
	string catalog_name;
	return CreateColumnReference(catalog_name, schema_name, table_name, column_name);
}

Binding *BindContext::GetCTEBinding(const string &ctename) {
	auto match = cte_bindings.find(ctename);
	if (match == cte_bindings.end()) {
		return nullptr;
	}
	return match->second.get();
}

Binding *BindContext::GetBinding(const string &name, string &out_error) {
	auto match = bindings.find(name);
	if (match == bindings.end()) {
		// alias not found in this BindContext
		vector<string> candidates;
		for (auto &kv : bindings) {
			candidates.push_back(kv.first);
		}
		string candidate_str =
		    StringUtil::CandidatesMessage(StringUtil::TopNLevenshtein(candidates, name), "Candidate tables");
		out_error = StringUtil::Format("Referenced table \"%s\" not found!%s", name, candidate_str);
		return nullptr;
	}
	return match->second.get();
}

BindResult BindContext::BindColumn(ColumnRefExpression &colref, idx_t depth) {
	if (!colref.IsQualified()) {
		throw InternalException("Could not bind alias \"%s\"!", colref.GetColumnName());
	}

	string error;
	auto binding = GetBinding(colref.GetTableName(), error);
	if (!binding) {
		return BindResult(error);
	}
	return binding->Bind(colref, depth);
}

string BindContext::BindColumn(PositionalReferenceExpression &ref, string &table_name, string &column_name) {
	idx_t total_columns = 0;
	idx_t current_position = ref.index - 1;
	for (auto &entry : bindings_list) {
		idx_t entry_column_count = entry.second->names.size();
		if (ref.index == 0) {
			// this is a row id
			table_name = entry.first;
			column_name = "rowid";
			return string();
		}
		if (current_position < entry_column_count) {
			table_name = entry.first;
			column_name = entry.second->names[current_position];
			return string();
		} else {
			total_columns += entry_column_count;
			current_position -= entry_column_count;
		}
	}
	return StringUtil::Format("Positional reference %d out of range (total %d columns)", ref.index, total_columns);
}

BindResult BindContext::BindColumn(PositionalReferenceExpression &ref, idx_t depth) {
	string table_name, column_name;

	string error = BindColumn(ref, table_name, column_name);
	if (!error.empty()) {
		return BindResult(error);
	}
	auto column_ref = make_unique<ColumnRefExpression>(column_name, table_name);
	return BindColumn(*column_ref, depth);
}

bool BindContext::CheckExclusionList(StarExpression &expr, Binding *binding, const string &column_name,
                                     vector<unique_ptr<ParsedExpression>> &new_select_list,
                                     case_insensitive_set_t &excluded_columns) {
	if (expr.exclude_list.find(column_name) != expr.exclude_list.end()) {
		excluded_columns.insert(column_name);
		return true;
	}
	auto entry = expr.replace_list.find(column_name);
	if (entry != expr.replace_list.end()) {
		auto new_entry = entry->second->Copy();
		new_entry->alias = entry->first;
		excluded_columns.insert(entry->first);
		new_select_list.push_back(std::move(new_entry));
		return true;
	}
	return false;
}

bool CheckRegex(const string &column_name, duckdb_re2::RE2 *regex) {
	if (!regex) {
		return true;
	}
	return RE2::PartialMatch(column_name, *regex);
}

void BindContext::GenerateAllColumnExpressions(StarExpression &expr,
                                               vector<unique_ptr<ParsedExpression>> &new_select_list) {
	if (bindings_list.empty()) {
		throw BinderException("SELECT * expression without FROM clause!");
	}
	case_insensitive_set_t excluded_columns;
	if (expr.relation_name.empty()) {
		// SELECT * case
		// bind all expressions of each table in-order
		unique_ptr<duckdb_re2::RE2> regex;
		bool found_match = true;
		if (!expr.regex.empty()) {
			regex = make_unique<duckdb_re2::RE2>(expr.regex);
			if (!regex->error().empty()) {
				throw BinderException("Failed to compile regex \"%s\": %s", expr.regex, regex->error());
			}
			found_match = false;
		}
		unordered_set<UsingColumnSet *> handled_using_columns;
		for (auto &entry : bindings_list) {
			auto binding = entry.second;
			for (auto &column_name : binding->names) {
				if (CheckExclusionList(expr, binding, column_name, new_select_list, excluded_columns)) {
					continue;
				}
				if (!CheckRegex(column_name, regex.get())) {
					continue;
				}
				found_match = true;
				// check if this column is a USING column
				auto using_binding = GetUsingBinding(column_name, binding->alias);
				if (using_binding) {
					// it is!
					// check if we have already emitted the using column
					if (handled_using_columns.find(using_binding) != handled_using_columns.end()) {
						// we have! bail out
						continue;
					}
					// we have not! output the using column
					if (using_binding->primary_binding.empty()) {
						// no primary binding: output a coalesce
						auto coalesce = make_unique<OperatorExpression>(ExpressionType::OPERATOR_COALESCE);
						for (auto &child_binding : using_binding->bindings) {
							coalesce->children.push_back(make_unique<ColumnRefExpression>(column_name, child_binding));
						}
						coalesce->alias = column_name;
						new_select_list.push_back(std::move(coalesce));
					} else {
						// primary binding: output the qualified column ref
						new_select_list.push_back(
						    make_unique<ColumnRefExpression>(column_name, using_binding->primary_binding));
					}
					handled_using_columns.insert(using_binding);
					continue;
				}
				new_select_list.push_back(make_unique<ColumnRefExpression>(column_name, binding->alias));
			}
		}
		if (!found_match) {
			throw BinderException("No matching columns found that match regex \"%s\"", expr.regex);
		}
	} else {
		// SELECT tbl.* case
		// SELECT struct.* case
		string error;
		auto binding = GetBinding(expr.relation_name, error);
		bool is_struct_ref = false;
		if (!binding) {
			auto binding_name = GetMatchingBinding(expr.relation_name);
			if (binding_name.empty()) {
				throw BinderException(error);
			}
			binding = bindings[binding_name].get();
			is_struct_ref = true;
		}

		if (is_struct_ref) {
			auto col_idx = binding->GetBindingIndex(expr.relation_name);
			auto col_type = binding->types[col_idx];
			if (col_type.id() != LogicalTypeId::STRUCT) {
				throw BinderException(StringUtil::Format(
				    "Cannot extract field from expression \"%s\" because it is not a struct", expr.ToString()));
			}
			auto &struct_children = StructType::GetChildTypes(col_type);
			vector<string> column_names(3);
			column_names[0] = binding->alias;
			column_names[1] = expr.relation_name;
			for (auto &child : struct_children) {
				if (CheckExclusionList(expr, binding, child.first, new_select_list, excluded_columns)) {
					continue;
				}
				column_names[2] = child.first;
				new_select_list.push_back(make_unique<ColumnRefExpression>(column_names));
			}
		} else {
			for (auto &column_name : binding->names) {
				if (CheckExclusionList(expr, binding, column_name, new_select_list, excluded_columns)) {
					continue;
				}

				new_select_list.push_back(make_unique<ColumnRefExpression>(column_name, binding->alias));
			}
		}
	}
	for (auto &excluded : expr.exclude_list) {
		if (excluded_columns.find(excluded) == excluded_columns.end()) {
			throw BinderException("Column \"%s\" in EXCLUDE list not found in %s", excluded,
			                      expr.relation_name.empty() ? "FROM clause" : expr.relation_name.c_str());
		}
	}
	for (auto &entry : expr.replace_list) {
		if (excluded_columns.find(entry.first) == excluded_columns.end()) {
			throw BinderException("Column \"%s\" in REPLACE list not found in %s", entry.first,
			                      expr.relation_name.empty() ? "FROM clause" : expr.relation_name.c_str());
		}
	}
}

void BindContext::AddBinding(const string &alias, unique_ptr<Binding> binding) {
	if (bindings.find(alias) != bindings.end()) {
		throw BinderException("Duplicate alias \"%s\" in query!", alias);
	}
	bindings_list.emplace_back(alias, binding.get());
	bindings[alias] = std::move(binding);
}

void BindContext::AddBaseTable(idx_t index, const string &alias, const vector<string> &names,
                               const vector<LogicalType> &types, vector<column_t> &bound_column_ids,
                               StandardEntry *entry, bool add_row_id) {
	AddBinding(alias, make_unique<TableBinding>(alias, types, names, bound_column_ids, entry, index, add_row_id));
}

void BindContext::AddTableFunction(idx_t index, const string &alias, const vector<string> &names,
                                   const vector<LogicalType> &types, vector<column_t> &bound_column_ids,
                                   StandardEntry *entry) {
	AddBinding(alias, make_unique<TableBinding>(alias, types, names, bound_column_ids, entry, index));
}

static string AddColumnNameToBinding(const string &base_name, case_insensitive_set_t &current_names) {
	idx_t index = 1;
	string name = base_name;
	while (current_names.find(name) != current_names.end()) {
		name = base_name + ":" + std::to_string(index++);
	}
	current_names.insert(name);
	return name;
}

vector<string> BindContext::AliasColumnNames(const string &table_name, const vector<string> &names,
                                             const vector<string> &column_aliases) {
	vector<string> result;
	if (column_aliases.size() > names.size()) {
		throw BinderException("table \"%s\" has %lld columns available but %lld columns specified", table_name,
		                      names.size(), column_aliases.size());
	}
	case_insensitive_set_t current_names;
	// use any provided column aliases first
	for (idx_t i = 0; i < column_aliases.size(); i++) {
		result.push_back(AddColumnNameToBinding(column_aliases[i], current_names));
	}
	// if not enough aliases were provided, use the default names for remaining columns
	for (idx_t i = column_aliases.size(); i < names.size(); i++) {
		result.push_back(AddColumnNameToBinding(names[i], current_names));
	}
	return result;
}

void BindContext::AddSubquery(idx_t index, const string &alias, SubqueryRef &ref, BoundQueryNode &subquery) {
	auto names = AliasColumnNames(alias, subquery.names, ref.column_name_alias);
	AddGenericBinding(index, alias, names, subquery.types);
}

void BindContext::AddEntryBinding(idx_t index, const string &alias, const vector<string> &names,
                                  const vector<LogicalType> &types, StandardEntry *entry) {
	D_ASSERT(entry);
	AddBinding(alias, make_unique<EntryBinding>(alias, types, names, index, *entry));
}

void BindContext::AddView(idx_t index, const string &alias, SubqueryRef &ref, BoundQueryNode &subquery,
                          ViewCatalogEntry *view) {
	auto names = AliasColumnNames(alias, subquery.names, ref.column_name_alias);
	AddEntryBinding(index, alias, names, subquery.types, (StandardEntry *)view);
}

void BindContext::AddSubquery(idx_t index, const string &alias, TableFunctionRef &ref, BoundQueryNode &subquery) {
	auto names = AliasColumnNames(alias, subquery.names, ref.column_name_alias);
	AddGenericBinding(index, alias, names, subquery.types);
}

void BindContext::AddGenericBinding(idx_t index, const string &alias, const vector<string> &names,
                                    const vector<LogicalType> &types) {
	AddBinding(alias, make_unique<Binding>(BindingType::BASE, alias, types, names, index));
}

void BindContext::AddCTEBinding(idx_t index, const string &alias, const vector<string> &names,
                                const vector<LogicalType> &types) {
	auto binding = make_shared<Binding>(BindingType::BASE, alias, types, names, index);

	if (cte_bindings.find(alias) != cte_bindings.end()) {
		throw BinderException("Duplicate alias \"%s\" in query!", alias);
	}
	cte_bindings[alias] = std::move(binding);
	cte_references[alias] = std::make_shared<idx_t>(0);
}

void BindContext::AddContext(BindContext other) {
	for (auto &binding : other.bindings) {
		if (bindings.find(binding.first) != bindings.end()) {
			throw BinderException("Duplicate alias \"%s\" in query!", binding.first);
		}
		bindings[binding.first] = std::move(binding.second);
	}
	for (auto &binding : other.bindings_list) {
		bindings_list.push_back(std::move(binding));
	}
	for (auto &entry : other.using_columns) {
		for (auto &alias : entry.second) {
#ifdef DEBUG
			for (auto &other_alias : using_columns[entry.first]) {
				for (auto &col : alias->bindings) {
					D_ASSERT(other_alias->bindings.find(col) == other_alias->bindings.end());
				}
			}
#endif
			using_columns[entry.first].insert(alias);
		}
	}
}

} // namespace duckdb

















namespace duckdb {

static Value NegatePercentileValue(const Value &v, const bool desc) {
	if (v.IsNull()) {
		return v;
	}

	const auto frac = v.GetValue<double>();
	if (frac < 0 || frac > 1) {
		throw BinderException("PERCENTILEs can only take parameters in the range [0, 1]");
	}

	if (!desc) {
		return v;
	}

	const auto &type = v.type();
	switch (type.id()) {
	case LogicalTypeId::DECIMAL: {
		// Negate DECIMALs as DECIMAL.
		const auto integral = IntegralValue::Get(v);
		const auto width = DecimalType::GetWidth(type);
		const auto scale = DecimalType::GetScale(type);
		switch (type.InternalType()) {
		case PhysicalType::INT16:
			return Value::DECIMAL(Cast::Operation<hugeint_t, int16_t>(-integral), width, scale);
		case PhysicalType::INT32:
			return Value::DECIMAL(Cast::Operation<hugeint_t, int32_t>(-integral), width, scale);
		case PhysicalType::INT64:
			return Value::DECIMAL(Cast::Operation<hugeint_t, int64_t>(-integral), width, scale);
		case PhysicalType::INT128:
			return Value::DECIMAL(-integral, width, scale);
		default:
			throw InternalException("Unknown DECIMAL type");
		}
	}
	default:
		// Everything else can just be a DOUBLE
		return Value::DOUBLE(-v.GetValue<double>());
	}
}

static void NegatePercentileFractions(ClientContext &context, unique_ptr<ParsedExpression> &fractions, bool desc) {
	D_ASSERT(fractions.get());
	D_ASSERT(fractions->expression_class == ExpressionClass::BOUND_EXPRESSION);
	auto &bound = (BoundExpression &)*fractions;

	if (!bound.expr->IsFoldable()) {
		return;
	}

	Value value = ExpressionExecutor::EvaluateScalar(context, *bound.expr);
	if (value.type().id() == LogicalTypeId::LIST) {
		vector<Value> values;
		for (const auto &element_val : ListValue::GetChildren(value)) {
			values.push_back(NegatePercentileValue(element_val, desc));
		}
		bound.expr = make_unique<BoundConstantExpression>(Value::LIST(values));
	} else {
		bound.expr = make_unique<BoundConstantExpression>(NegatePercentileValue(value, desc));
	}
}

BindResult SelectBinder::BindAggregate(FunctionExpression &aggr, AggregateFunctionCatalogEntry *func, idx_t depth) {
	// first bind the child of the aggregate expression (if any)
	this->bound_aggregate = true;
	unique_ptr<Expression> bound_filter;
	AggregateBinder aggregate_binder(binder, context);
	string error, filter_error;

	// Now we bind the filter (if any)
	if (aggr.filter) {
		aggregate_binder.BindChild(aggr.filter, 0, error);
	}

	// Handle ordered-set aggregates by moving the single ORDER BY expression to the front of the children.
	//	https://www.postgresql.org/docs/current/functions-aggregate.html#FUNCTIONS-ORDEREDSET-TABLE
	bool ordered_set_agg = false;
	bool negate_fractions = false;
	if (aggr.order_bys && aggr.order_bys->orders.size() == 1) {
		const auto &func_name = aggr.function_name;
		ordered_set_agg = (func_name == "quantile_cont" || func_name == "quantile_disc" || func_name == "mode");

		if (ordered_set_agg) {
			auto &config = DBConfig::GetConfig(context);
			const auto &order = aggr.order_bys->orders[0];
			const auto sense =
			    (order.type == OrderType::ORDER_DEFAULT) ? config.options.default_order_type : order.type;
			negate_fractions = (sense == OrderType::DESCENDING);
		}
	}

	for (auto &child : aggr.children) {
		aggregate_binder.BindChild(child, 0, error);
		// We have to negate the fractions for PERCENTILE_XXXX DESC
		if (error.empty() && ordered_set_agg) {
			NegatePercentileFractions(context, child, negate_fractions);
		}
	}

	// Bind the ORDER BYs, if any
	if (aggr.order_bys && !aggr.order_bys->orders.empty()) {
		for (auto &order : aggr.order_bys->orders) {
			aggregate_binder.BindChild(order.expression, 0, error);
		}
	}

	if (!error.empty()) {
		// failed to bind child
		if (aggregate_binder.HasBoundColumns()) {
			for (idx_t i = 0; i < aggr.children.size(); i++) {
				// however, we bound columns!
				// that means this aggregation belongs to this node
				// check if we have to resolve any errors by binding with parent binders
				bool success = aggregate_binder.BindCorrelatedColumns(aggr.children[i]);
				// if there is still an error after this, we could not successfully bind the aggregate
				if (!success) {
					throw BinderException(error);
				}
				auto &bound_expr = (BoundExpression &)*aggr.children[i];
				ExtractCorrelatedExpressions(binder, *bound_expr.expr);
			}
			if (aggr.filter) {
				bool success = aggregate_binder.BindCorrelatedColumns(aggr.filter);
				// if there is still an error after this, we could not successfully bind the aggregate
				if (!success) {
					throw BinderException(error);
				}
				auto &bound_expr = (BoundExpression &)*aggr.filter;
				ExtractCorrelatedExpressions(binder, *bound_expr.expr);
			}
			if (aggr.order_bys && !aggr.order_bys->orders.empty()) {
				for (auto &order : aggr.order_bys->orders) {
					bool success = aggregate_binder.BindCorrelatedColumns(order.expression);
					if (!success) {
						throw BinderException(error);
					}
					auto &bound_expr = (BoundExpression &)*order.expression;
					ExtractCorrelatedExpressions(binder, *bound_expr.expr);
				}
			}
		} else {
			// we didn't bind columns, try again in children
			return BindResult(error);
		}
	} else if (depth > 0 && !aggregate_binder.HasBoundColumns()) {
		return BindResult("Aggregate with only constant parameters has to be bound in the root subquery");
	}
	if (!filter_error.empty()) {
		return BindResult(filter_error);
	}

	if (aggr.filter) {
		auto &child = (BoundExpression &)*aggr.filter;
		bound_filter = BoundCastExpression::AddCastToType(context, std::move(child.expr), LogicalType::BOOLEAN);
	}

	// all children bound successfully
	// extract the children and types
	vector<LogicalType> types;
	vector<LogicalType> arguments;
	vector<unique_ptr<Expression>> children;

	if (ordered_set_agg) {
		for (auto &order : aggr.order_bys->orders) {
			auto &child = (BoundExpression &)*order.expression;
			types.push_back(child.expr->return_type);
			arguments.push_back(child.expr->return_type);
			children.push_back(std::move(child.expr));
		}
		aggr.order_bys->orders.clear();
	}

	for (idx_t i = 0; i < aggr.children.size(); i++) {
		auto &child = (BoundExpression &)*aggr.children[i];
		types.push_back(child.expr->return_type);
		arguments.push_back(child.expr->return_type);
		children.push_back(std::move(child.expr));
	}

	// bind the aggregate
	FunctionBinder function_binder(context);
	idx_t best_function = function_binder.BindFunction(func->name, func->functions, types, error);
	if (best_function == DConstants::INVALID_INDEX) {
		throw BinderException(binder.FormatError(aggr, error));
	}
	// found a matching function!
	auto bound_function = func->functions.GetFunctionByOffset(best_function);

	// Bind any sort columns, unless the aggregate is order-insensitive
	auto order_bys = make_unique<BoundOrderModifier>();
	if (!aggr.order_bys->orders.empty()) {
		auto &config = DBConfig::GetConfig(context);
		for (auto &order : aggr.order_bys->orders) {
			auto &order_expr = (BoundExpression &)*order.expression;
			const auto sense =
			    (order.type == OrderType::ORDER_DEFAULT) ? config.options.default_order_type : order.type;
			const auto null_order = (order.null_order == OrderByNullType::ORDER_DEFAULT)
			                            ? config.options.default_null_order
			                            : order.null_order;
			order_bys->orders.emplace_back(BoundOrderByNode(sense, null_order, std::move(order_expr.expr)));
		}
	}

	auto aggregate = function_binder.BindAggregateFunction(
	    bound_function, std::move(children), std::move(bound_filter),
	    aggr.distinct ? AggregateType::DISTINCT : AggregateType::NON_DISTINCT, std::move(order_bys));
	if (aggr.export_state) {
		aggregate = ExportAggregateFunction::Bind(std::move(aggregate));
	}

	// check for all the aggregates if this aggregate already exists
	idx_t aggr_index;
	auto entry = node.aggregate_map.find(aggregate.get());
	if (entry == node.aggregate_map.end()) {
		// new aggregate: insert into aggregate list
		aggr_index = node.aggregates.size();
		node.aggregate_map.insert(make_pair(aggregate.get(), aggr_index));
		node.aggregates.push_back(std::move(aggregate));
	} else {
		// duplicate aggregate: simplify refer to this aggregate
		aggr_index = entry->second;
	}

	// now create a column reference referring to the aggregate
	auto colref = make_unique<BoundColumnRefExpression>(
	    aggr.alias.empty() ? node.aggregates[aggr_index]->ToString() : aggr.alias,
	    node.aggregates[aggr_index]->return_type, ColumnBinding(node.aggregate_index, aggr_index), depth);
	// move the aggregate expression into the set of bound aggregates
	return BindResult(std::move(colref));
}
} // namespace duckdb








namespace duckdb {

BindResult ExpressionBinder::BindExpression(BetweenExpression &expr, idx_t depth) {
	// first try to bind the children of the case expression
	string error;
	BindChild(expr.input, depth, error);
	BindChild(expr.lower, depth, error);
	BindChild(expr.upper, depth, error);
	if (!error.empty()) {
		return BindResult(error);
	}
	// the children have been successfully resolved
	auto &input = (BoundExpression &)*expr.input;
	auto &lower = (BoundExpression &)*expr.lower;
	auto &upper = (BoundExpression &)*expr.upper;

	auto input_sql_type = input.expr->return_type;
	auto lower_sql_type = lower.expr->return_type;
	auto upper_sql_type = upper.expr->return_type;

	// cast the input types to the same type
	// now obtain the result type of the input types
	auto input_type = BoundComparisonExpression::BindComparison(input_sql_type, lower_sql_type);
	input_type = BoundComparisonExpression::BindComparison(input_type, upper_sql_type);
	// add casts (if necessary)
	input.expr = BoundCastExpression::AddCastToType(context, std::move(input.expr), input_type);
	lower.expr = BoundCastExpression::AddCastToType(context, std::move(lower.expr), input_type);
	upper.expr = BoundCastExpression::AddCastToType(context, std::move(upper.expr), input_type);
	if (input_type.id() == LogicalTypeId::VARCHAR) {
		// handle collation
		auto collation = StringType::GetCollation(input_type);
		input.expr = PushCollation(context, std::move(input.expr), collation, false);
		lower.expr = PushCollation(context, std::move(lower.expr), collation, false);
		upper.expr = PushCollation(context, std::move(upper.expr), collation, false);
	}
	if (!input.expr->HasSideEffects() && !input.expr->HasParameter() && !input.expr->HasSubquery()) {
		// the expression does not have side effects and can be copied: create two comparisons
		// the reason we do this is that individual comparisons are easier to handle in optimizers
		// if both comparisons remain they will be folded together again into a single BETWEEN in the optimizer
		auto left_compare = make_unique<BoundComparisonExpression>(ExpressionType::COMPARE_GREATERTHANOREQUALTO,
		                                                           input.expr->Copy(), std::move(lower.expr));
		auto right_compare = make_unique<BoundComparisonExpression>(ExpressionType::COMPARE_LESSTHANOREQUALTO,
		                                                            std::move(input.expr), std::move(upper.expr));
		return BindResult(make_unique<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND,
		                                                          std::move(left_compare), std::move(right_compare)));
	} else {
		// expression has side effects: we cannot duplicate it
		// create a bound_between directly
		return BindResult(make_unique<BoundBetweenExpression>(std::move(input.expr), std::move(lower.expr),
		                                                      std::move(upper.expr), true, true));
	}
}

} // namespace duckdb





namespace duckdb {

BindResult ExpressionBinder::BindExpression(CaseExpression &expr, idx_t depth) {
	// first try to bind the children of the case expression
	string error;
	for (auto &check : expr.case_checks) {
		BindChild(check.when_expr, depth, error);
		BindChild(check.then_expr, depth, error);
	}
	BindChild(expr.else_expr, depth, error);
	if (!error.empty()) {
		return BindResult(error);
	}
	// the children have been successfully resolved
	// figure out the result type of the CASE expression
	auto return_type = ((BoundExpression &)*expr.else_expr).expr->return_type;
	for (auto &check : expr.case_checks) {
		auto &then_expr = (BoundExpression &)*check.then_expr;
		return_type = LogicalType::MaxLogicalType(return_type, then_expr.expr->return_type);
	}

	// bind all the individual components of the CASE statement
	auto result = make_unique<BoundCaseExpression>(return_type);
	for (idx_t i = 0; i < expr.case_checks.size(); i++) {
		auto &check = expr.case_checks[i];
		auto &when_expr = (BoundExpression &)*check.when_expr;
		auto &then_expr = (BoundExpression &)*check.then_expr;
		BoundCaseCheck result_check;
		result_check.when_expr =
		    BoundCastExpression::AddCastToType(context, std::move(when_expr.expr), LogicalType::BOOLEAN);
		result_check.then_expr = BoundCastExpression::AddCastToType(context, std::move(then_expr.expr), return_type);
		result->case_checks.push_back(std::move(result_check));
	}
	auto &else_expr = (BoundExpression &)*expr.else_expr;
	result->else_expr = BoundCastExpression::AddCastToType(context, std::move(else_expr.expr), return_type);
	return BindResult(std::move(result));
}
} // namespace duckdb






namespace duckdb {

BindResult ExpressionBinder::BindExpression(CastExpression &expr, idx_t depth) {
	// first try to bind the child of the cast expression
	string error = Bind(&expr.child, depth);
	if (!error.empty()) {
		return BindResult(error);
	}
	// FIXME: We can also implement 'hello'::schema.custom_type; and pass by the schema down here.
	// Right now just considering its DEFAULT_SCHEMA always
	Binder::BindLogicalType(context, expr.cast_type);
	// the children have been successfully resolved
	auto &child = (BoundExpression &)*expr.child;
	if (expr.try_cast) {
		if (child.expr->return_type == expr.cast_type) {
			// no cast required: type matches
			return BindResult(std::move(child.expr));
		}
		child.expr = BoundCastExpression::AddCastToType(context, std::move(child.expr), expr.cast_type, true);
	} else {
		// otherwise add a cast to the target type
		child.expr = BoundCastExpression::AddCastToType(context, std::move(child.expr), expr.cast_type);
	}
	return BindResult(std::move(child.expr));
}
} // namespace duckdb




namespace duckdb {

BindResult ExpressionBinder::BindExpression(CollateExpression &expr, idx_t depth) {
	// first try to bind the child of the cast expression
	string error = Bind(&expr.child, depth);
	if (!error.empty()) {
		return BindResult(error);
	}
	auto &child = (BoundExpression &)*expr.child;
	if (child.expr->HasParameter()) {
		throw ParameterNotResolvedException();
	}
	if (child.expr->return_type.id() != LogicalTypeId::VARCHAR) {
		throw BinderException("collations are only supported for type varchar");
	}
	// Validate the collation, but don't use it
	PushCollation(context, child.expr->Copy(), expr.collation, false);
	child.expr->return_type = LogicalType::VARCHAR_COLLATION(expr.collation);
	return BindResult(std::move(child.expr));
}

} // namespace duckdb

















namespace duckdb {

unique_ptr<ParsedExpression> ExpressionBinder::QualifyColumnName(const string &column_name, string &error_message) {
	auto using_binding = binder.bind_context.GetUsingBinding(column_name);
	if (using_binding) {
		// we are referencing a USING column
		// check if we can refer to one of the base columns directly
		unique_ptr<Expression> expression;
		if (!using_binding->primary_binding.empty()) {
			// we can! just assign the table name and re-bind
			return binder.bind_context.CreateColumnReference(using_binding->primary_binding, column_name);
		} else {
			// // we cannot! we need to bind this as a coalesce between all the relevant columns
			auto coalesce = make_unique<OperatorExpression>(ExpressionType::OPERATOR_COALESCE);
			coalesce->children.reserve(using_binding->bindings.size());
			for (auto &entry : using_binding->bindings) {
				coalesce->children.push_back(make_unique<ColumnRefExpression>(column_name, entry));
			}
			return std::move(coalesce);
		}
	}

	// find a binding that contains this
	string table_name = binder.bind_context.GetMatchingBinding(column_name);

	// throw an error if a macro conflicts with a column name
	auto is_macro_column = false;
	if (binder.macro_binding != nullptr && binder.macro_binding->HasMatchingBinding(column_name)) {
		is_macro_column = true;
		if (!table_name.empty()) {
			throw BinderException("Conflicting column names for column " + column_name + "!");
		}
	}

	if (lambda_bindings) {
		for (idx_t i = 0; i < lambda_bindings->size(); i++) {
			if ((*lambda_bindings)[i].HasMatchingBinding(column_name)) {

				// throw an error if a lambda conflicts with a column name or a macro
				if (!table_name.empty() || is_macro_column) {
					throw BinderException("Conflicting column names for column " + column_name + "!");
				}

				D_ASSERT(!(*lambda_bindings)[i].alias.empty());
				return make_unique<ColumnRefExpression>(column_name, (*lambda_bindings)[i].alias);
			}
		}
	}

	if (is_macro_column) {
		D_ASSERT(!binder.macro_binding->alias.empty());
		return make_unique<ColumnRefExpression>(column_name, binder.macro_binding->alias);
	}
	// see if it's a column
	if (table_name.empty()) {
		// it's not, find candidates and error
		auto similar_bindings = binder.bind_context.GetSimilarBindings(column_name);
		string candidate_str = StringUtil::CandidatesMessage(similar_bindings, "Candidate bindings");
		error_message =
		    StringUtil::Format("Referenced column \"%s\" not found in FROM clause!%s", column_name, candidate_str);
		return nullptr;
	}
	return binder.bind_context.CreateColumnReference(table_name, column_name);
}

void ExpressionBinder::QualifyColumnNames(unique_ptr<ParsedExpression> &expr) {
	switch (expr->type) {
	case ExpressionType::COLUMN_REF: {
		auto &colref = (ColumnRefExpression &)*expr;
		string error_message;
		auto new_expr = QualifyColumnName(colref, error_message);
		if (new_expr) {
			if (!expr->alias.empty()) {
				new_expr->alias = expr->alias;
			}
			new_expr->query_location = colref.query_location;
			expr = std::move(new_expr);
		}
		break;
	}
	case ExpressionType::POSITIONAL_REFERENCE: {
		auto &ref = (PositionalReferenceExpression &)*expr;
		if (ref.alias.empty()) {
			string table_name, column_name;
			auto error = binder.bind_context.BindColumn(ref, table_name, column_name);
			if (error.empty()) {
				ref.alias = column_name;
			}
		}
		break;
	}
	default:
		break;
	}
	ParsedExpressionIterator::EnumerateChildren(
	    *expr, [&](unique_ptr<ParsedExpression> &child) { QualifyColumnNames(child); });
}

void ExpressionBinder::QualifyColumnNames(Binder &binder, unique_ptr<ParsedExpression> &expr) {
	WhereBinder where_binder(binder, binder.context);
	where_binder.QualifyColumnNames(expr);
}

unique_ptr<ParsedExpression> ExpressionBinder::CreateStructExtract(unique_ptr<ParsedExpression> base,
                                                                   string field_name) {

	// we need to transform the struct extract if it is inside a lambda expression
	// because we cannot bind to an existing table, so we remove the dummy table also
	if (lambda_bindings && base->type == ExpressionType::COLUMN_REF) {
		auto &lambda_column_ref = (ColumnRefExpression &)*base;
		D_ASSERT(!lambda_column_ref.column_names.empty());

		if (lambda_column_ref.column_names[0].find(DummyBinding::DUMMY_NAME) != string::npos) {
			D_ASSERT(lambda_column_ref.column_names.size() == 2);
			auto lambda_param_name = lambda_column_ref.column_names.back();
			lambda_column_ref.column_names.clear();
			lambda_column_ref.column_names.push_back(lambda_param_name);
		}
	}

	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(std::move(base));
	children.push_back(make_unique_base<ParsedExpression, ConstantExpression>(Value(std::move(field_name))));
	auto extract_fun = make_unique<OperatorExpression>(ExpressionType::STRUCT_EXTRACT, std::move(children));
	return std::move(extract_fun);
}

unique_ptr<ParsedExpression> ExpressionBinder::CreateStructPack(ColumnRefExpression &colref) {
	D_ASSERT(colref.column_names.size() <= 3);
	string error_message;
	auto &table_name = colref.column_names.back();
	auto binding = binder.bind_context.GetBinding(table_name, error_message);
	if (!binding) {
		return nullptr;
	}
	if (colref.column_names.size() >= 2) {
		// "schema_name.table_name"
		auto catalog_entry = binding->GetStandardEntry();
		if (!catalog_entry) {
			return nullptr;
		}
		if (catalog_entry->name != table_name) {
			return nullptr;
		}
		if (colref.column_names.size() == 2) {
			auto &qualifier = colref.column_names[0];
			if (catalog_entry->catalog->GetName() != qualifier && catalog_entry->schema->name != qualifier) {
				return nullptr;
			}
		} else if (colref.column_names.size() == 3) {
			auto &catalog_name = colref.column_names[0];
			auto &schema_name = colref.column_names[1];
			if (catalog_entry->catalog->GetName() != catalog_name || catalog_entry->schema->name != schema_name) {
				return nullptr;
			}
		} else {
			throw InternalException("Expected 2 or 3 column names for CreateStructPack");
		}
	}
	// We found the table, now create the struct_pack expression
	vector<unique_ptr<ParsedExpression>> child_exprs;
	for (const auto &column_name : binding->names) {
		child_exprs.push_back(make_unique<ColumnRefExpression>(column_name, table_name));
	}
	return make_unique<FunctionExpression>("struct_pack", std::move(child_exprs));
}

unique_ptr<ParsedExpression> ExpressionBinder::QualifyColumnName(ColumnRefExpression &colref, string &error_message) {
	idx_t column_parts = colref.column_names.size();
	// column names can have an arbitrary amount of dots
	// here is how the resolution works:
	if (column_parts == 1) {
		// no dots (i.e. "part1")
		// -> part1 refers to a column
		// check if we can qualify the column name with the table name
		auto qualified_colref = QualifyColumnName(colref.GetColumnName(), error_message);
		if (qualified_colref) {
			// we could: return it
			return qualified_colref;
		}
		// we could not! Try creating an implicit struct_pack
		return CreateStructPack(colref);
	} else if (column_parts == 2) {
		// one dot (i.e. "part1.part2")
		// EITHER:
		// -> part1 is a table, part2 is a column
		// -> part1 is a column, part2 is a property of that column (i.e. struct_extract)

		// first check if part1 is a table, and part2 is a standard column
		if (binder.HasMatchingBinding(colref.column_names[0], colref.column_names[1], error_message)) {
			// it is! return the colref directly
			return binder.bind_context.CreateColumnReference(colref.column_names[0], colref.column_names[1]);
		} else {
			// otherwise check if we can turn this into a struct extract
			auto new_colref = make_unique<ColumnRefExpression>(colref.column_names[0]);
			string other_error;
			auto qualified_colref = QualifyColumnName(colref.column_names[0], other_error);
			if (qualified_colref) {
				// we could: create a struct extract
				return CreateStructExtract(std::move(qualified_colref), colref.column_names[1]);
			}
			// we could not! Try creating an implicit struct_pack
			return CreateStructPack(colref);
		}
	} else {
		// two or more dots (i.e. "part1.part2.part3.part4...")
		// -> part1 is a catalog, part2 is a schema, part3 is a table, part4 is a column name, part 5 and beyond are
		// struct fields
		// -> part1 is a catalog, part2 is a table, part3 is a column name, part4 and beyond are struct fields
		// -> part1 is a schema, part2 is a table, part3 is a column name, part4 and beyond are struct fields
		// -> part1 is a table, part2 is a column name, part3 and beyond are struct fields
		// -> part1 is a column, part2 and beyond are struct fields

		// we always prefer the most top-level view
		// i.e. in case of multiple resolution options, we resolve in order:
		// -> 1. resolve "part1" as a catalog
		// -> 2. resolve "part1" as a schema
		// -> 3. resolve "part1" as a table
		// -> 4. resolve "part1" as a column

		unique_ptr<ParsedExpression> result_expr;
		idx_t struct_extract_start;
		// first check if part1 is a catalog
		if (colref.column_names.size() > 3 &&
		    binder.HasMatchingBinding(colref.column_names[0], colref.column_names[1], colref.column_names[2],
		                              colref.column_names[3], error_message)) {
			// part1 is a catalog - the column reference is "catalog.schema.table.column"
			result_expr = binder.bind_context.CreateColumnReference(colref.column_names[0], colref.column_names[1],
			                                                        colref.column_names[2], colref.column_names[3]);
			struct_extract_start = 4;
		} else if (binder.HasMatchingBinding(colref.column_names[0], INVALID_SCHEMA, colref.column_names[1],
		                                     colref.column_names[2], error_message)) {
			// part1 is a catalog - the column reference is "catalog.table.column"
			result_expr = binder.bind_context.CreateColumnReference(colref.column_names[0], INVALID_SCHEMA,
			                                                        colref.column_names[1], colref.column_names[2]);
			struct_extract_start = 3;
		} else if (binder.HasMatchingBinding(colref.column_names[0], colref.column_names[1], colref.column_names[2],
		                                     error_message)) {
			// part1 is a schema - the column reference is "schema.table.column"
			// any additional fields are turned into struct_extract calls
			result_expr = binder.bind_context.CreateColumnReference(colref.column_names[0], colref.column_names[1],
			                                                        colref.column_names[2]);
			struct_extract_start = 3;
		} else if (binder.HasMatchingBinding(colref.column_names[0], colref.column_names[1], error_message)) {
			// part1 is a table
			// the column reference is "table.column"
			// any additional fields are turned into struct_extract calls
			result_expr = binder.bind_context.CreateColumnReference(colref.column_names[0], colref.column_names[1]);
			struct_extract_start = 2;
		} else {
			// part1 could be a column
			string col_error;
			result_expr = QualifyColumnName(colref.column_names[0], col_error);
			if (!result_expr) {
				// it is not! Try creating an implicit struct_pack
				return CreateStructPack(colref);
			}
			// it is! add the struct extract calls
			struct_extract_start = 1;
		}
		for (idx_t i = struct_extract_start; i < colref.column_names.size(); i++) {
			result_expr = CreateStructExtract(std::move(result_expr), colref.column_names[i]);
		}
		return result_expr;
	}
}

BindResult ExpressionBinder::BindExpression(ColumnRefExpression &colref_p, idx_t depth) {
	if (binder.GetBindingMode() == BindingMode::EXTRACT_NAMES) {
		return BindResult(make_unique<BoundConstantExpression>(Value(LogicalType::SQLNULL)));
	}
	string error_message;
	auto expr = QualifyColumnName(colref_p, error_message);
	if (!expr) {
		return BindResult(binder.FormatError(colref_p, error_message));
	}
	expr->query_location = colref_p.query_location;

	// a generated column returns a generated expression, a struct on a column returns a struct extract
	if (expr->type != ExpressionType::COLUMN_REF) {
		auto alias = expr->alias;
		auto result = BindExpression(&expr, depth);
		if (result.expression) {
			result.expression->alias = std::move(alias);
		}
		return result;
	}

	auto &colref = (ColumnRefExpression &)*expr;
	D_ASSERT(colref.IsQualified());
	auto &table_name = colref.GetTableName();

	// individual column reference
	// resolve to either a base table or a subquery expression
	// if it was a macro parameter, let macro_binding bind it to the argument
	// if it was a lambda parameter, let lambda_bindings bind it to the argument

	BindResult result;

	auto found_lambda_binding = false;
	if (lambda_bindings) {
		for (idx_t i = 0; i < lambda_bindings->size(); i++) {
			if (table_name == (*lambda_bindings)[i].alias) {
				result = (*lambda_bindings)[i].Bind(colref, i, depth);
				found_lambda_binding = true;
				break;
			}
		}
	}

	if (!found_lambda_binding) {
		if (binder.macro_binding && table_name == binder.macro_binding->alias) {
			result = binder.macro_binding->Bind(colref, depth);
		} else {
			result = binder.bind_context.BindColumn(colref, depth);
		}
	}

	if (!result.HasError()) {
		BoundColumnReferenceInfo ref;
		ref.name = colref.column_names.back();
		ref.query_location = colref.query_location;
		bound_columns.push_back(std::move(ref));
	} else {
		result.error = binder.FormatError(colref_p, result.error);
	}
	return result;
}

} // namespace duckdb


















namespace duckdb {

unique_ptr<Expression> ExpressionBinder::PushCollation(ClientContext &context, unique_ptr<Expression> source,
                                                       const string &collation_p, bool equality_only) {
	// replace default collation with system collation
	string collation;
	if (collation_p.empty()) {
		collation = DBConfig::GetConfig(context).options.collation;
	} else {
		collation = collation_p;
	}
	collation = StringUtil::Lower(collation);
	// bind the collation
	if (collation.empty() || collation == "binary" || collation == "c" || collation == "posix") {
		// binary collation: just skip
		return source;
	}
	auto &catalog = Catalog::GetSystemCatalog(context);
	auto splits = StringUtil::Split(StringUtil::Lower(collation), ".");
	vector<CollateCatalogEntry *> entries;
	for (auto &collation_argument : splits) {
		auto collation_entry = catalog.GetEntry<CollateCatalogEntry>(context, DEFAULT_SCHEMA, collation_argument);
		if (collation_entry->combinable) {
			entries.insert(entries.begin(), collation_entry);
		} else {
			if (!entries.empty() && !entries.back()->combinable) {
				throw BinderException("Cannot combine collation types \"%s\" and \"%s\"", entries.back()->name,
				                      collation_entry->name);
			}
			entries.push_back(collation_entry);
		}
	}
	for (auto &collation_entry : entries) {
		if (equality_only && collation_entry->not_required_for_equality) {
			continue;
		}
		vector<unique_ptr<Expression>> children;
		children.push_back(std::move(source));

		FunctionBinder function_binder(context);
		auto function = function_binder.BindScalarFunction(collation_entry->function, std::move(children));
		source = std::move(function);
	}
	return source;
}

void ExpressionBinder::TestCollation(ClientContext &context, const string &collation) {
	PushCollation(context, make_unique<BoundConstantExpression>(Value("")), collation);
}

LogicalType BoundComparisonExpression::BindComparison(LogicalType left_type, LogicalType right_type) {
	auto result_type = LogicalType::MaxLogicalType(left_type, right_type);
	switch (result_type.id()) {
	case LogicalTypeId::DECIMAL: {
		// result is a decimal: we need the maximum width and the maximum scale over width
		vector<LogicalType> argument_types = {left_type, right_type};
		uint8_t max_width = 0, max_scale = 0, max_width_over_scale = 0;
		for (idx_t i = 0; i < argument_types.size(); i++) {
			uint8_t width, scale;
			auto can_convert = argument_types[i].GetDecimalProperties(width, scale);
			if (!can_convert) {
				return result_type;
			}
			max_width = MaxValue<uint8_t>(width, max_width);
			max_scale = MaxValue<uint8_t>(scale, max_scale);
			max_width_over_scale = MaxValue<uint8_t>(width - scale, max_width_over_scale);
		}
		max_width = MaxValue<uint8_t>(max_scale + max_width_over_scale, max_width);
		if (max_width > Decimal::MAX_WIDTH_DECIMAL) {
			// target width does not fit in decimal: truncate the scale (if possible) to try and make it fit
			max_width = Decimal::MAX_WIDTH_DECIMAL;
		}
		return LogicalType::DECIMAL(max_width, max_scale);
	}
	case LogicalTypeId::VARCHAR:
		// for comparison with strings, we prefer to bind to the numeric types
		if (left_type.IsNumeric() || left_type.id() == LogicalTypeId::BOOLEAN) {
			return left_type;
		} else if (right_type.IsNumeric() || right_type.id() == LogicalTypeId::BOOLEAN) {
			return right_type;
		} else {
			// else: check if collations are compatible
			auto left_collation = StringType::GetCollation(left_type);
			auto right_collation = StringType::GetCollation(right_type);
			if (!left_collation.empty() && !right_collation.empty() && left_collation != right_collation) {
				throw BinderException("Cannot combine types with different collation!");
			}
		}
		return result_type;
	default:
		return result_type;
	}
}

BindResult ExpressionBinder::BindExpression(ComparisonExpression &expr, idx_t depth) {
	// first try to bind the children of the case expression
	string error;
	BindChild(expr.left, depth, error);
	BindChild(expr.right, depth, error);
	if (!error.empty()) {
		return BindResult(error);
	}
	// the children have been successfully resolved
	auto &left = (BoundExpression &)*expr.left;
	auto &right = (BoundExpression &)*expr.right;
	auto left_sql_type = left.expr->return_type;
	auto right_sql_type = right.expr->return_type;
	// cast the input types to the same type
	// now obtain the result type of the input types
	auto input_type = BoundComparisonExpression::BindComparison(left_sql_type, right_sql_type);
	// add casts (if necessary)
	left.expr = BoundCastExpression::AddCastToType(context, std::move(left.expr), input_type,
	                                               input_type.id() == LogicalTypeId::ENUM);
	right.expr = BoundCastExpression::AddCastToType(context, std::move(right.expr), input_type,
	                                                input_type.id() == LogicalTypeId::ENUM);

	if (input_type.id() == LogicalTypeId::VARCHAR) {
		// handle collation
		auto collation = StringType::GetCollation(input_type);
		left.expr = PushCollation(context, std::move(left.expr), collation, expr.type == ExpressionType::COMPARE_EQUAL);
		right.expr =
		    PushCollation(context, std::move(right.expr), collation, expr.type == ExpressionType::COMPARE_EQUAL);
	}
	// now create the bound comparison expression
	return BindResult(make_unique<BoundComparisonExpression>(expr.type, std::move(left.expr), std::move(right.expr)));
}

} // namespace duckdb





namespace duckdb {

BindResult ExpressionBinder::BindExpression(ConjunctionExpression &expr, idx_t depth) {
	// first try to bind the children of the case expression
	string error;
	for (idx_t i = 0; i < expr.children.size(); i++) {
		BindChild(expr.children[i], depth, error);
	}
	if (!error.empty()) {
		return BindResult(error);
	}
	// the children have been successfully resolved
	// cast the input types to boolean (if necessary)
	// and construct the bound conjunction expression
	auto result = make_unique<BoundConjunctionExpression>(expr.type);
	for (auto &child_expr : expr.children) {
		auto &child = (BoundExpression &)*child_expr;
		result->children.push_back(
		    BoundCastExpression::AddCastToType(context, std::move(child.expr), LogicalType::BOOLEAN));
	}
	// now create the bound conjunction expression
	return BindResult(std::move(result));
}

} // namespace duckdb




namespace duckdb {

BindResult ExpressionBinder::BindExpression(ConstantExpression &expr, idx_t depth) {
	return BindResult(make_unique<BoundConstantExpression>(expr.value));
}

} // namespace duckdb














namespace duckdb {

BindResult ExpressionBinder::BindExpression(FunctionExpression &function, idx_t depth,
                                            unique_ptr<ParsedExpression> *expr_ptr) {
	// lookup the function in the catalog
	QueryErrorContext error_context(binder.root_statement, function.query_location);

	if (function.function_name == "unnest" || function.function_name == "unlist") {
		// special case, not in catalog
		// TODO make sure someone does not create such a function OR
		// have unnest live in catalog, too
		return BindUnnest(function, depth);
	}
	auto func = Catalog::GetEntry(context, CatalogType::SCALAR_FUNCTION_ENTRY, function.catalog, function.schema,
	                              function.function_name, true, error_context);
	if (!func) {
		// function was not found - check if we this is a table function
		auto table_func = Catalog::GetEntry(context, CatalogType::TABLE_FUNCTION_ENTRY, function.catalog,
		                                    function.schema, function.function_name, true, error_context);
		if (table_func) {
			throw BinderException(binder.FormatError(
			    function,
			    StringUtil::Format("Function \"%s\" is a table function but it was used as a scalar function. This "
			                       "function has to be called in a FROM clause (similar to a table).",
			                       function.function_name)));
		}
		// not a table function - search again without if_exists to throw the error
		Catalog::GetEntry(context, CatalogType::SCALAR_FUNCTION_ENTRY, function.catalog, function.schema,
		                  function.function_name, false, error_context);
		throw InternalException("Catalog::GetEntry for scalar function did not throw a second time");
	}

	if (func->type != CatalogType::AGGREGATE_FUNCTION_ENTRY &&
	    (function.distinct || function.filter || !function.order_bys->orders.empty())) {
		throw InvalidInputException("Function \"%s\" is a %s. \"DISTINCT\", \"FILTER\", and \"ORDER BY\" are only "
		                            "applicable to aggregate functions.",
		                            function.function_name, CatalogTypeToString(func->type));
	}

	switch (func->type) {
	case CatalogType::SCALAR_FUNCTION_ENTRY:
		// scalar function

		// check for lambda parameters, ignore ->> operator (JSON extension)
		if (function.function_name != "->>") {
			for (auto &child : function.children) {
				if (child->expression_class == ExpressionClass::LAMBDA) {
					return BindLambdaFunction(function, (ScalarFunctionCatalogEntry *)func, depth);
				}
			}
		}

		// other scalar function
		return BindFunction(function, (ScalarFunctionCatalogEntry *)func, depth);

	case CatalogType::MACRO_ENTRY:
		// macro function
		return BindMacro(function, (ScalarMacroCatalogEntry *)func, depth, expr_ptr);
	default:
		// aggregate function
		return BindAggregate(function, (AggregateFunctionCatalogEntry *)func, depth);
	}
}

BindResult ExpressionBinder::BindFunction(FunctionExpression &function, ScalarFunctionCatalogEntry *func, idx_t depth) {

	// bind the children of the function expression
	string error;

	// bind of each child
	for (idx_t i = 0; i < function.children.size(); i++) {
		BindChild(function.children[i], depth, error);
	}

	if (!error.empty()) {
		return BindResult(error);
	}
	if (binder.GetBindingMode() == BindingMode::EXTRACT_NAMES) {
		return BindResult(make_unique<BoundConstantExpression>(Value(LogicalType::SQLNULL)));
	}

	// all children bound successfully
	// extract the children and types
	vector<unique_ptr<Expression>> children;
	for (idx_t i = 0; i < function.children.size(); i++) {
		auto &child = (BoundExpression &)*function.children[i];
		D_ASSERT(child.expr);
		children.push_back(std::move(child.expr));
	}

	FunctionBinder function_binder(context);
	unique_ptr<Expression> result =
	    function_binder.BindScalarFunction(*func, std::move(children), error, function.is_operator, &binder);
	if (!result) {
		throw BinderException(binder.FormatError(function, error));
	}
	return BindResult(std::move(result));
}

BindResult ExpressionBinder::BindLambdaFunction(FunctionExpression &function, ScalarFunctionCatalogEntry *func,
                                                idx_t depth) {

	// bind the children of the function expression
	string error;

	if (function.children.size() != 2) {
		throw BinderException("Invalid function arguments!");
	}
	D_ASSERT(function.children[1]->GetExpressionClass() == ExpressionClass::LAMBDA);

	// bind the list parameter
	BindChild(function.children[0], depth, error);
	if (!error.empty()) {
		return BindResult(error);
	}

	// get the logical type of the children of the list
	auto &list_child = (BoundExpression &)*function.children[0];

	if (list_child.expr->return_type.id() != LogicalTypeId::LIST &&
	    list_child.expr->return_type.id() != LogicalTypeId::SQLNULL &&
	    list_child.expr->return_type.id() != LogicalTypeId::UNKNOWN) {
		throw BinderException(" Invalid LIST argument to " + function.function_name + "!");
	}

	LogicalType list_child_type = list_child.expr->return_type.id();
	if (list_child.expr->return_type.id() != LogicalTypeId::SQLNULL &&
	    list_child.expr->return_type.id() != LogicalTypeId::UNKNOWN) {
		list_child_type = ListType::GetChildType(list_child.expr->return_type);
	}

	// bind the lambda parameter
	auto &lambda_expr = (LambdaExpression &)*function.children[1];
	BindResult bind_lambda_result = BindExpression(lambda_expr, depth, true, list_child_type);

	if (bind_lambda_result.HasError()) {
		error = bind_lambda_result.error;
	} else {
		// successfully bound: replace the node with a BoundExpression
		auto alias = function.children[1]->alias;
		function.children[1] = make_unique<BoundExpression>(std::move(bind_lambda_result.expression));
		auto be = (BoundExpression *)function.children[1].get();
		D_ASSERT(be);
		be->alias = alias;
		if (!alias.empty()) {
			be->expr->alias = alias;
		}
	}

	if (!error.empty()) {
		return BindResult(error);
	}
	if (binder.GetBindingMode() == BindingMode::EXTRACT_NAMES) {
		return BindResult(make_unique<BoundConstantExpression>(Value(LogicalType::SQLNULL)));
	}

	// all children bound successfully
	// extract the children and types
	vector<unique_ptr<Expression>> children;
	for (idx_t i = 0; i < function.children.size(); i++) {
		auto &child = (BoundExpression &)*function.children[i];
		D_ASSERT(child.expr);
		children.push_back(std::move(child.expr));
	}

	// capture the (lambda) columns
	auto &bound_lambda_expr = (BoundLambdaExpression &)*children.back();
	CaptureLambdaColumns(bound_lambda_expr.captures, list_child_type, bound_lambda_expr.lambda_expr);

	FunctionBinder function_binder(context);
	unique_ptr<Expression> result =
	    function_binder.BindScalarFunction(*func, std::move(children), error, function.is_operator, &binder);
	if (!result) {
		throw BinderException(binder.FormatError(function, error));
	}

	auto &bound_function_expr = (BoundFunctionExpression &)*result;
	D_ASSERT(bound_function_expr.children.size() == 2);

	// remove the lambda expression from the children
	auto lambda = std::move(bound_function_expr.children.back());
	bound_function_expr.children.pop_back();
	auto &bound_lambda = (BoundLambdaExpression &)*lambda;

	// push back (in reverse order) any nested lambda parameters so that we can later use them in the lambda expression
	// (rhs)
	if (lambda_bindings) {
		for (idx_t i = lambda_bindings->size(); i > 0; i--) {

			idx_t lambda_index = lambda_bindings->size() - i + 1;
			auto &binding = (*lambda_bindings)[i - 1];

			D_ASSERT(binding.names.size() == 1);
			D_ASSERT(binding.types.size() == 1);

			bound_function_expr.function.arguments.push_back(binding.types[0]);
			auto bound_lambda_param =
			    make_unique<BoundReferenceExpression>(binding.names[0], binding.types[0], lambda_index);
			bound_function_expr.children.push_back(std::move(bound_lambda_param));
		}
	}

	// push back the captures into the children vector and the correct return types into the bound_function arguments
	for (auto &capture : bound_lambda.captures) {
		bound_function_expr.function.arguments.push_back(capture->return_type);
		bound_function_expr.children.push_back(std::move(capture));
	}

	return BindResult(std::move(result));
}

BindResult ExpressionBinder::BindAggregate(FunctionExpression &expr, AggregateFunctionCatalogEntry *function,
                                           idx_t depth) {
	return BindResult(binder.FormatError(expr, UnsupportedAggregateMessage()));
}

BindResult ExpressionBinder::BindUnnest(FunctionExpression &expr, idx_t depth) {
	return BindResult(binder.FormatError(expr, UnsupportedUnnestMessage()));
}

string ExpressionBinder::UnsupportedAggregateMessage() {
	return "Aggregate functions are not supported here";
}

string ExpressionBinder::UnsupportedUnnestMessage() {
	return "UNNEST not supported here";
}

} // namespace duckdb












namespace duckdb {

BindResult ExpressionBinder::BindExpression(LambdaExpression &expr, idx_t depth, const bool is_lambda,
                                            const LogicalType &list_child_type) {

	if (!is_lambda) {
		// this is for binding JSON
		auto lhs_expr = expr.lhs->Copy();
		OperatorExpression arrow_expr(ExpressionType::ARROW, std::move(lhs_expr), expr.expr->Copy());
		return BindExpression(arrow_expr, depth);
	}

	// binding the lambda expression
	D_ASSERT(expr.lhs);
	if (expr.lhs->expression_class != ExpressionClass::FUNCTION &&
	    expr.lhs->expression_class != ExpressionClass::COLUMN_REF) {
		throw BinderException(
		    "Invalid parameter list! Parameters must be comma-separated column names, e.g. x or (x, y).");
	}

	// move the lambda parameters to the params vector
	if (expr.lhs->expression_class == ExpressionClass::COLUMN_REF) {
		expr.params.push_back(std::move(expr.lhs));
	} else {
		auto &func_expr = (FunctionExpression &)*expr.lhs;
		for (idx_t i = 0; i < func_expr.children.size(); i++) {
			expr.params.push_back(std::move(func_expr.children[i]));
		}
	}
	D_ASSERT(!expr.params.empty());

	// create dummy columns for the lambda parameters (lhs)
	vector<LogicalType> column_types;
	vector<string> column_names;
	vector<string> params_strings;

	// positional parameters as column references
	for (idx_t i = 0; i < expr.params.size(); i++) {

		if (expr.params[i]->GetExpressionClass() != ExpressionClass::COLUMN_REF) {
			throw BinderException("Parameter must be a column name.");
		}

		auto column_ref = (ColumnRefExpression &)*expr.params[i];
		if (column_ref.IsQualified()) {
			throw BinderException("Invalid parameter name '%s': must be unqualified", column_ref.ToString());
		}

		column_types.emplace_back(list_child_type);
		column_names.push_back(column_ref.GetColumnName());
		params_strings.push_back(expr.params[i]->ToString());
	}

	// base table alias
	auto params_alias = StringUtil::Join(params_strings, ", ");
	if (params_strings.size() > 1) {
		params_alias = "(" + params_alias + ")";
	}

	// create a lambda binding and push it to the lambda bindings vector
	vector<DummyBinding> local_bindings;
	if (!lambda_bindings) {
		lambda_bindings = &local_bindings;
	}
	DummyBinding new_lambda_binding(column_types, column_names, params_alias);
	lambda_bindings->push_back(new_lambda_binding);

	// bind the parameter expressions
	for (idx_t i = 0; i < expr.params.size(); i++) {
		auto result = BindExpression(&expr.params[i], depth, false);
		D_ASSERT(!result.HasError());
	}

	auto result = BindExpression(&expr.expr, depth, false);
	lambda_bindings->pop_back();

	// successfully bound a subtree of nested lambdas, set this to nullptr in case other parts of the
	// query also contain lambdas
	if (lambda_bindings->empty()) {
		lambda_bindings = nullptr;
	}

	if (result.HasError()) {
		throw BinderException(result.error);
	}

	return BindResult(make_unique<BoundLambdaExpression>(ExpressionType::LAMBDA, LogicalType::LAMBDA,
	                                                     std::move(result.expression), params_strings.size()));
}

void ExpressionBinder::TransformCapturedLambdaColumn(unique_ptr<Expression> &original,
                                                     unique_ptr<Expression> &replacement,
                                                     vector<unique_ptr<Expression>> &captures,
                                                     LogicalType &list_child_type) {

	// check if the original expression is a lambda parameter
	if (original->expression_class == ExpressionClass::BOUND_LAMBDA_REF) {

		// determine if this is the lambda parameter
		auto &bound_lambda_ref = (BoundLambdaRefExpression &)*original;
		auto alias = bound_lambda_ref.alias;

		if (lambda_bindings && bound_lambda_ref.lambda_index != lambda_bindings->size()) {

			D_ASSERT(bound_lambda_ref.lambda_index < lambda_bindings->size());
			auto &lambda_binding = (*lambda_bindings)[bound_lambda_ref.lambda_index];

			D_ASSERT(lambda_binding.names.size() == 1);
			D_ASSERT(lambda_binding.types.size() == 1);
			// refers to a lambda parameter outside of the current lambda function
			replacement =
			    make_unique<BoundReferenceExpression>(lambda_binding.names[0], lambda_binding.types[0],
			                                          lambda_bindings->size() - bound_lambda_ref.lambda_index + 1);

		} else {
			// refers to current lambda parameter
			replacement = make_unique<BoundReferenceExpression>(alias, list_child_type, 0);
		}

	} else {
		// always at least the current lambda parameter
		idx_t index_offset = 1;
		if (lambda_bindings) {
			index_offset += lambda_bindings->size();
		}

		// this is not a lambda parameter, so we need to create a new argument for the arguments vector
		replacement = make_unique<BoundReferenceExpression>(original->alias, original->return_type,
		                                                    captures.size() + index_offset + 1);
		captures.push_back(std::move(original));
	}
}

void ExpressionBinder::CaptureLambdaColumns(vector<unique_ptr<Expression>> &captures, LogicalType &list_child_type,
                                            unique_ptr<Expression> &expr) {

	if (expr->expression_class == ExpressionClass::BOUND_SUBQUERY) {
		throw InvalidInputException("Subqueries are not supported in lambda expressions!");
	}

	// these expression classes do not have children, transform them
	if (expr->expression_class == ExpressionClass::BOUND_CONSTANT ||
	    expr->expression_class == ExpressionClass::BOUND_COLUMN_REF ||
	    expr->expression_class == ExpressionClass::BOUND_PARAMETER ||
	    expr->expression_class == ExpressionClass::BOUND_LAMBDA_REF) {

		// move the expr because we are going to replace it
		auto original = std::move(expr);
		unique_ptr<Expression> replacement;

		TransformCapturedLambdaColumn(original, replacement, captures, list_child_type);

		// replace the expression
		expr = std::move(replacement);

	} else {
		// recursively enumerate the children of the expression
		ExpressionIterator::EnumerateChildren(
		    *expr, [&](unique_ptr<Expression> &child) { CaptureLambdaColumns(captures, list_child_type, child); });
	}

	expr->Verify();
}

} // namespace duckdb









namespace duckdb {

void ExpressionBinder::ReplaceMacroParametersRecursive(unique_ptr<ParsedExpression> &expr) {
	switch (expr->GetExpressionClass()) {
	case ExpressionClass::COLUMN_REF: {
		// if expr is a parameter, replace it with its argument
		auto &colref = (ColumnRefExpression &)*expr;
		bool bind_macro_parameter = false;
		if (colref.IsQualified()) {
			bind_macro_parameter = false;
			if (colref.GetTableName().find(DummyBinding::DUMMY_NAME) != string::npos) {
				bind_macro_parameter = true;
			}
		} else {
			bind_macro_parameter = macro_binding->HasMatchingBinding(colref.GetColumnName());
		}
		if (bind_macro_parameter) {
			D_ASSERT(macro_binding->HasMatchingBinding(colref.GetColumnName()));
			expr = macro_binding->ParamToArg(colref);
		}
		return;
	}
	case ExpressionClass::SUBQUERY: {
		// replacing parameters within a subquery is slightly different
		auto &sq = ((SubqueryExpression &)*expr).subquery;
		ParsedExpressionIterator::EnumerateQueryNodeChildren(
		    *sq->node, [&](unique_ptr<ParsedExpression> &child) { ReplaceMacroParametersRecursive(child); });
		break;
	}
	default: // fall through
		break;
	}
	// unfold child expressions
	ParsedExpressionIterator::EnumerateChildren(
	    *expr, [&](unique_ptr<ParsedExpression> &child) { ReplaceMacroParametersRecursive(child); });
}

BindResult ExpressionBinder::BindMacro(FunctionExpression &function, ScalarMacroCatalogEntry *macro_func, idx_t depth,
                                       unique_ptr<ParsedExpression> *expr) {
	// recast function so we can access the scalar member function->expression
	auto &macro_def = (ScalarMacroFunction &)*macro_func->function;

	// validate the arguments and separate positional and default arguments
	vector<unique_ptr<ParsedExpression>> positionals;
	unordered_map<string, unique_ptr<ParsedExpression>> defaults;

	string error =
	    MacroFunction::ValidateArguments(*macro_func->function, macro_func->name, function, positionals, defaults);
	if (!error.empty()) {
		throw BinderException(binder.FormatError(*expr->get(), error));
	}

	// create a MacroBinding to bind this macro's parameters to its arguments
	vector<LogicalType> types;
	vector<string> names;
	// positional parameters
	for (idx_t i = 0; i < macro_def.parameters.size(); i++) {
		types.emplace_back(LogicalType::SQLNULL);
		auto &param = (ColumnRefExpression &)*macro_def.parameters[i];
		names.push_back(param.GetColumnName());
	}
	// default parameters
	for (auto it = macro_def.default_parameters.begin(); it != macro_def.default_parameters.end(); it++) {
		types.emplace_back(LogicalType::SQLNULL);
		names.push_back(it->first);
		// now push the defaults into the positionals
		positionals.push_back(std::move(defaults[it->first]));
	}
	auto new_macro_binding = make_unique<DummyBinding>(types, names, macro_func->name);
	new_macro_binding->arguments = &positionals;
	macro_binding = new_macro_binding.get();

	// replace current expression with stored macro expression, and replace params
	*expr = macro_def.expression->Copy();
	ReplaceMacroParametersRecursive(*expr);

	// bind the unfolded macro
	return BindExpression(expr, depth);
}

} // namespace duckdb








namespace duckdb {

static LogicalType ResolveNotType(OperatorExpression &op, vector<BoundExpression *> &children) {
	// NOT expression, cast child to BOOLEAN
	D_ASSERT(children.size() == 1);
	children[0]->expr = BoundCastExpression::AddDefaultCastToType(std::move(children[0]->expr), LogicalType::BOOLEAN);
	return LogicalType(LogicalTypeId::BOOLEAN);
}

static LogicalType ResolveInType(OperatorExpression &op, vector<BoundExpression *> &children) {
	if (children.empty()) {
		throw InternalException("IN requires at least a single child node");
	}
	// get the maximum type from the children
	LogicalType max_type = children[0]->expr->return_type;
	for (idx_t i = 1; i < children.size(); i++) {
		max_type = LogicalType::MaxLogicalType(max_type, children[i]->expr->return_type);
	}

	// cast all children to the same type
	for (idx_t i = 0; i < children.size(); i++) {
		children[i]->expr = BoundCastExpression::AddDefaultCastToType(std::move(children[i]->expr), max_type);
	}
	// (NOT) IN always returns a boolean
	return LogicalType::BOOLEAN;
}

static LogicalType ResolveOperatorType(OperatorExpression &op, vector<BoundExpression *> &children) {
	switch (op.type) {
	case ExpressionType::OPERATOR_IS_NULL:
	case ExpressionType::OPERATOR_IS_NOT_NULL:
		// IS (NOT) NULL always returns a boolean, and does not cast its children
		if (!children[0]->expr->return_type.IsValid()) {
			throw ParameterNotResolvedException();
		}
		return LogicalType::BOOLEAN;
	case ExpressionType::COMPARE_IN:
	case ExpressionType::COMPARE_NOT_IN:
		return ResolveInType(op, children);
	case ExpressionType::OPERATOR_COALESCE: {
		ResolveInType(op, children);
		return children[0]->expr->return_type;
	}
	case ExpressionType::OPERATOR_NOT:
		return ResolveNotType(op, children);
	default:
		throw InternalException("Unrecognized expression type for ResolveOperatorType");
	}
}

BindResult ExpressionBinder::BindGroupingFunction(OperatorExpression &op, idx_t depth) {
	return BindResult("GROUPING function is not supported here");
}

BindResult ExpressionBinder::BindExpression(OperatorExpression &op, idx_t depth) {
	if (op.type == ExpressionType::GROUPING_FUNCTION) {
		return BindGroupingFunction(op, depth);
	}
	// bind the children of the operator expression
	string error;
	for (idx_t i = 0; i < op.children.size(); i++) {
		BindChild(op.children[i], depth, error);
	}
	if (!error.empty()) {
		return BindResult(error);
	}
	// all children bound successfully
	string function_name;
	switch (op.type) {
	case ExpressionType::ARRAY_EXTRACT: {
		D_ASSERT(op.children[0]->expression_class == ExpressionClass::BOUND_EXPRESSION);
		auto &b_exp = (BoundExpression &)*op.children[0];
		if (b_exp.expr->return_type.id() == LogicalTypeId::MAP) {
			function_name = "map_extract";
		} else {
			function_name = "array_extract";
		}
		break;
	}
	case ExpressionType::ARRAY_SLICE:
		function_name = "array_slice";
		break;
	case ExpressionType::STRUCT_EXTRACT: {
		D_ASSERT(op.children.size() == 2);
		D_ASSERT(op.children[0]->expression_class == ExpressionClass::BOUND_EXPRESSION);
		D_ASSERT(op.children[1]->expression_class == ExpressionClass::BOUND_EXPRESSION);
		auto &extract_exp = (BoundExpression &)*op.children[0];
		auto &name_exp = (BoundExpression &)*op.children[1];
		auto extract_expr_type = extract_exp.expr->return_type.id();
		if (extract_expr_type != LogicalTypeId::STRUCT && extract_expr_type != LogicalTypeId::UNION &&
		    extract_expr_type != LogicalTypeId::SQLNULL) {
			return BindResult(StringUtil::Format(
			    "Cannot extract field %s from expression \"%s\" because it is not a struct or a union",
			    name_exp.ToString(), extract_exp.ToString()));
		}
		function_name = extract_expr_type == LogicalTypeId::UNION ? "union_extract" : "struct_extract";
		break;
	}
	case ExpressionType::ARRAY_CONSTRUCTOR:
		function_name = "list_value";
		break;
	case ExpressionType::ARROW:
		function_name = "json_extract";
		break;
	default:
		break;
	}
	if (!function_name.empty()) {
		auto function = make_unique<FunctionExpression>(function_name, std::move(op.children));
		return BindExpression(*function, depth, nullptr);
	}

	vector<BoundExpression *> children;
	for (idx_t i = 0; i < op.children.size(); i++) {
		D_ASSERT(op.children[i]->expression_class == ExpressionClass::BOUND_EXPRESSION);
		children.push_back((BoundExpression *)op.children[i].get());
	}
	// now resolve the types
	LogicalType result_type = ResolveOperatorType(op, children);
	if (op.type == ExpressionType::OPERATOR_COALESCE) {
		if (children.empty()) {
			throw BinderException("COALESCE needs at least one child");
		}
		if (children.size() == 1) {
			return BindResult(std::move(children[0]->expr));
		}
	}

	auto result = make_unique<BoundOperatorExpression>(op.type, result_type);
	for (auto &child : children) {
		result->children.push_back(std::move(child->expr));
	}
	return BindResult(std::move(result));
}

} // namespace duckdb






namespace duckdb {

BindResult ExpressionBinder::BindExpression(ParameterExpression &expr, idx_t depth) {
	D_ASSERT(expr.parameter_nr > 0);
	auto bound_parameter = make_unique<BoundParameterExpression>(expr.parameter_nr);
	bound_parameter->alias = expr.alias;
	if (!binder.parameters) {
		throw BinderException("Unexpected prepared parameter. This type of statement can't be prepared!");
	}
	auto parameter_idx = expr.parameter_nr;
	// check if a parameter value has already been supplied
	if (parameter_idx <= binder.parameters->parameter_data.size()) {
		// it has! emit a constant directly
		auto &data = binder.parameters->parameter_data[parameter_idx - 1];
		auto constant = make_unique<BoundConstantExpression>(data.value);
		constant->alias = expr.alias;
		return BindResult(std::move(constant));
	}
	auto entry = binder.parameters->parameters.find(parameter_idx);
	if (entry == binder.parameters->parameters.end()) {
		// no entry yet: create a new one
		auto data = make_shared<BoundParameterData>();
		data->return_type = binder.parameters->GetReturnType(parameter_idx - 1);
		bound_parameter->return_type = data->return_type;
		bound_parameter->parameter_data = data;
		binder.parameters->parameters[parameter_idx] = std::move(data);
	} else {
		// a prepared statement with this parameter index was already there: use it
		auto &data = entry->second;
		bound_parameter->parameter_data = data;
		bound_parameter->return_type = binder.parameters->GetReturnType(parameter_idx - 1);
	}
	return BindResult(std::move(bound_parameter));
}

} // namespace duckdb




namespace duckdb {

BindResult ExpressionBinder::BindExpression(PositionalReferenceExpression &ref, idx_t depth) {
	if (depth != 0) {
		return BindResult("Positional reference expression could not be bound");
	}
	return binder.bind_context.BindColumn(ref, depth);
}

} // namespace duckdb







namespace duckdb {

class BoundSubqueryNode : public QueryNode {
public:
	BoundSubqueryNode(shared_ptr<Binder> subquery_binder, unique_ptr<BoundQueryNode> bound_node,
	                  unique_ptr<SelectStatement> subquery)
	    : QueryNode(QueryNodeType::BOUND_SUBQUERY_NODE), subquery_binder(std::move(subquery_binder)),
	      bound_node(std::move(bound_node)), subquery(std::move(subquery)) {
	}

	shared_ptr<Binder> subquery_binder;
	unique_ptr<BoundQueryNode> bound_node;
	unique_ptr<SelectStatement> subquery;

	const vector<unique_ptr<ParsedExpression>> &GetSelectList() const override {
		throw InternalException("Cannot get select list of bound subquery node");
	}

	string ToString() const override {
		throw InternalException("Cannot ToString bound subquery node");
	}
	unique_ptr<QueryNode> Copy() const override {
		throw InternalException("Cannot copy bound subquery node");
	}
	void Serialize(FieldWriter &writer) const override {
		throw InternalException("Cannot serialize bound subquery node");
	}
};

BindResult ExpressionBinder::BindExpression(SubqueryExpression &expr, idx_t depth) {
	if (expr.subquery->node->type != QueryNodeType::BOUND_SUBQUERY_NODE) {
		D_ASSERT(depth == 0);
		// first bind the actual subquery in a new binder
		auto subquery_binder = Binder::CreateBinder(context, &binder);
		subquery_binder->can_contain_nulls = true;
		auto bound_node = subquery_binder->BindNode(*expr.subquery->node);
		// check the correlated columns of the subquery for correlated columns with depth > 1
		for (idx_t i = 0; i < subquery_binder->correlated_columns.size(); i++) {
			CorrelatedColumnInfo corr = subquery_binder->correlated_columns[i];
			if (corr.depth > 1) {
				// depth > 1, the column references the query ABOVE the current one
				// add to the set of correlated columns for THIS query
				corr.depth -= 1;
				binder.AddCorrelatedColumn(corr);
			}
		}
		if (expr.subquery_type != SubqueryType::EXISTS && bound_node->types.size() > 1) {
			throw BinderException(binder.FormatError(
			    expr, StringUtil::Format("Subquery returns %zu columns - expected 1", bound_node->types.size())));
		}
		auto prior_subquery = std::move(expr.subquery);
		expr.subquery = make_unique<SelectStatement>();
		expr.subquery->node = make_unique<BoundSubqueryNode>(std::move(subquery_binder), std::move(bound_node),
		                                                     std::move(prior_subquery));
	}
	// now bind the child node of the subquery
	if (expr.child) {
		// first bind the children of the subquery, if any
		string error = Bind(&expr.child, depth);
		if (!error.empty()) {
			return BindResult(error);
		}
	}
	// both binding the child and binding the subquery was successful
	D_ASSERT(expr.subquery->node->type == QueryNodeType::BOUND_SUBQUERY_NODE);
	auto bound_subquery = (BoundSubqueryNode *)expr.subquery->node.get();
	auto child = (BoundExpression *)expr.child.get();
	auto subquery_binder = std::move(bound_subquery->subquery_binder);
	auto bound_node = std::move(bound_subquery->bound_node);
	LogicalType return_type =
	    expr.subquery_type == SubqueryType::SCALAR ? bound_node->types[0] : LogicalType(LogicalTypeId::BOOLEAN);
	if (return_type.id() == LogicalTypeId::UNKNOWN) {
		return_type = LogicalType::SQLNULL;
	}

	auto result = make_unique<BoundSubqueryExpression>(return_type);
	if (expr.subquery_type == SubqueryType::ANY) {
		// ANY comparison
		// cast child and subquery child to equivalent types
		D_ASSERT(bound_node->types.size() == 1);
		auto compare_type = LogicalType::MaxLogicalType(child->expr->return_type, bound_node->types[0]);
		child->expr = BoundCastExpression::AddCastToType(context, std::move(child->expr), compare_type);
		result->child_type = bound_node->types[0];
		result->child_target = compare_type;
	}
	result->binder = std::move(subquery_binder);
	result->subquery = std::move(bound_node);
	result->subquery_type = expr.subquery_type;
	result->child = child ? std::move(child->expr) : nullptr;
	result->comparison_type = expr.comparison_type;

	return BindResult(std::move(result));
}

} // namespace duckdb












namespace duckdb {

BindResult SelectBinder::BindUnnest(FunctionExpression &function, idx_t depth) {
	// bind the children of the function expression
	string error;
	if (function.children.size() != 1) {
		return BindResult(binder.FormatError(function, "Unnest() needs exactly one child expressions"));
	}
	BindChild(function.children[0], depth, error);
	if (!error.empty()) {
		// failed to bind
		// try to bind correlated columns manually
		if (!BindCorrelatedColumns(function.children[0])) {
			return BindResult(error);
		}
		auto bound_expr = (BoundExpression *)function.children[0].get();
		ExtractCorrelatedExpressions(binder, *bound_expr->expr);
	}
	auto &child = (BoundExpression &)*function.children[0];
	auto &child_type = child.expr->return_type;

	if (child_type.id() != LogicalTypeId::LIST && child_type.id() != LogicalTypeId::SQLNULL &&
	    child_type.id() != LogicalTypeId::UNKNOWN) {
		return BindResult(binder.FormatError(function, "Unnest() can only be applied to lists and NULL"));
	}

	if (depth > 0) {
		throw BinderException(binder.FormatError(function, "Unnest() for correlated expressions is not supported yet"));
	}

	auto return_type = LogicalType(LogicalTypeId::SQLNULL);
	if (child_type.id() == LogicalTypeId::LIST) {
		return_type = ListType::GetChildType(child_type);
	} else if (child_type.id() == LogicalTypeId::UNKNOWN) {
		throw ParameterNotResolvedException();
	}

	auto result = make_unique<BoundUnnestExpression>(return_type);
	result->child = std::move(child.expr);

	auto unnest_index = node.unnests.size();
	node.unnests.push_back(std::move(result));

	// TODO what if we have multiple unnests in the same projection list? ignore for now

	// now create a column reference referring to the unnest
	auto colref = make_unique<BoundColumnRefExpression>(
	    function.alias.empty() ? node.unnests[unnest_index]->ToString() : function.alias, return_type,
	    ColumnBinding(node.unnest_index, unnest_index), depth);

	return BindResult(std::move(colref));
}

} // namespace duckdb
















namespace duckdb {

static LogicalType ResolveWindowExpressionType(ExpressionType window_type, const vector<LogicalType> &child_types) {

	idx_t param_count;
	switch (window_type) {
	case ExpressionType::WINDOW_RANK:
	case ExpressionType::WINDOW_RANK_DENSE:
	case ExpressionType::WINDOW_ROW_NUMBER:
	case ExpressionType::WINDOW_PERCENT_RANK:
	case ExpressionType::WINDOW_CUME_DIST:
		param_count = 0;
		break;
	case ExpressionType::WINDOW_NTILE:
	case ExpressionType::WINDOW_FIRST_VALUE:
	case ExpressionType::WINDOW_LAST_VALUE:
	case ExpressionType::WINDOW_LEAD:
	case ExpressionType::WINDOW_LAG:
		param_count = 1;
		break;
	case ExpressionType::WINDOW_NTH_VALUE:
		param_count = 2;
		break;
	default:
		throw InternalException("Unrecognized window expression type " + ExpressionTypeToString(window_type));
	}
	if (child_types.size() != param_count) {
		throw BinderException("%s needs %d parameter%s, got %d", ExpressionTypeToString(window_type), param_count,
		                      param_count == 1 ? "" : "s", child_types.size());
	}
	switch (window_type) {
	case ExpressionType::WINDOW_PERCENT_RANK:
	case ExpressionType::WINDOW_CUME_DIST:
		return LogicalType(LogicalTypeId::DOUBLE);
	case ExpressionType::WINDOW_ROW_NUMBER:
	case ExpressionType::WINDOW_RANK:
	case ExpressionType::WINDOW_RANK_DENSE:
	case ExpressionType::WINDOW_NTILE:
		return LogicalType::BIGINT;
	case ExpressionType::WINDOW_NTH_VALUE:
	case ExpressionType::WINDOW_FIRST_VALUE:
	case ExpressionType::WINDOW_LAST_VALUE:
	case ExpressionType::WINDOW_LEAD:
	case ExpressionType::WINDOW_LAG:
		return child_types[0];
	default:
		throw InternalException("Unrecognized window expression type " + ExpressionTypeToString(window_type));
	}
}

static inline OrderType ResolveOrderType(const DBConfig &config, OrderType type) {
	return (type == OrderType::ORDER_DEFAULT) ? config.options.default_order_type : type;
}

static inline OrderByNullType ResolveNullOrder(const DBConfig &config, OrderByNullType null_order) {
	return (null_order == OrderByNullType::ORDER_DEFAULT) ? config.options.default_null_order : null_order;
}

static unique_ptr<Expression> GetExpression(unique_ptr<ParsedExpression> &expr) {
	if (!expr) {
		return nullptr;
	}
	D_ASSERT(expr.get());
	D_ASSERT(expr->expression_class == ExpressionClass::BOUND_EXPRESSION);
	return std::move(((BoundExpression &)*expr).expr);
}

static unique_ptr<Expression> CastWindowExpression(unique_ptr<ParsedExpression> &expr, const LogicalType &type) {
	if (!expr) {
		return nullptr;
	}
	D_ASSERT(expr.get());
	D_ASSERT(expr->expression_class == ExpressionClass::BOUND_EXPRESSION);

	auto &bound = (BoundExpression &)*expr;
	bound.expr = BoundCastExpression::AddDefaultCastToType(std::move(bound.expr), type);

	return std::move(bound.expr);
}

static LogicalType BindRangeExpression(ClientContext &context, const string &name, unique_ptr<ParsedExpression> &expr,
                                       unique_ptr<ParsedExpression> &order_expr) {

	vector<unique_ptr<Expression>> children;

	D_ASSERT(order_expr.get());
	D_ASSERT(order_expr->expression_class == ExpressionClass::BOUND_EXPRESSION);
	auto &bound_order = (BoundExpression &)*order_expr;
	children.emplace_back(bound_order.expr->Copy());

	D_ASSERT(expr.get());
	D_ASSERT(expr->expression_class == ExpressionClass::BOUND_EXPRESSION);
	auto &bound = (BoundExpression &)*expr;
	children.emplace_back(std::move(bound.expr));

	string error;
	FunctionBinder function_binder(context);
	auto function = function_binder.BindScalarFunction(DEFAULT_SCHEMA, name, std::move(children), error, true);
	if (!function) {
		throw BinderException(error);
	}
	bound.expr = std::move(function);
	return bound.expr->return_type;
}

BindResult SelectBinder::BindWindow(WindowExpression &window, idx_t depth) {
	auto name = window.GetName();

	QueryErrorContext error_context(binder.root_statement, window.query_location);
	if (inside_window) {
		throw BinderException(error_context.FormatError("window function calls cannot be nested"));
	}
	if (depth > 0) {
		throw BinderException(error_context.FormatError("correlated columns in window functions not supported"));
	}
	// If we have range expressions, then only one order by clause is allowed.
	if ((window.start == WindowBoundary::EXPR_PRECEDING_RANGE || window.start == WindowBoundary::EXPR_FOLLOWING_RANGE ||
	     window.end == WindowBoundary::EXPR_PRECEDING_RANGE || window.end == WindowBoundary::EXPR_FOLLOWING_RANGE) &&
	    window.orders.size() != 1) {
		throw BinderException(error_context.FormatError("RANGE frames must have only one ORDER BY expression"));
	}
	// bind inside the children of the window function
	// we set the inside_window flag to true to prevent binding nested window functions
	this->inside_window = true;
	string error;
	for (auto &child : window.children) {
		BindChild(child, depth, error);
	}
	for (auto &child : window.partitions) {
		BindChild(child, depth, error);
	}
	for (auto &order : window.orders) {
		BindChild(order.expression, depth, error);
	}
	BindChild(window.filter_expr, depth, error);
	BindChild(window.start_expr, depth, error);
	BindChild(window.end_expr, depth, error);
	BindChild(window.offset_expr, depth, error);
	BindChild(window.default_expr, depth, error);

	this->inside_window = false;
	if (!error.empty()) {
		// failed to bind children of window function
		return BindResult(error);
	}
	// successfully bound all children: create bound window function
	vector<LogicalType> types;
	vector<unique_ptr<Expression>> children;
	for (auto &child : window.children) {
		D_ASSERT(child.get());
		D_ASSERT(child->expression_class == ExpressionClass::BOUND_EXPRESSION);
		auto &bound = (BoundExpression &)*child;
		// Add casts for positional arguments
		const auto argno = children.size();
		switch (window.type) {
		case ExpressionType::WINDOW_NTILE:
			// ntile(bigint)
			if (argno == 0) {
				bound.expr = BoundCastExpression::AddCastToType(context, std::move(bound.expr), LogicalType::BIGINT);
			}
			break;
		case ExpressionType::WINDOW_NTH_VALUE:
			// nth_value(<expr>, index)
			if (argno == 1) {
				bound.expr = BoundCastExpression::AddCastToType(context, std::move(bound.expr), LogicalType::BIGINT);
			}
		default:
			break;
		}
		types.push_back(bound.expr->return_type);
		children.push_back(std::move(bound.expr));
	}
	//  Determine the function type.
	LogicalType sql_type;
	unique_ptr<AggregateFunction> aggregate;
	unique_ptr<FunctionData> bind_info;
	if (window.type == ExpressionType::WINDOW_AGGREGATE) {
		//  Look up the aggregate function in the catalog
		auto func = Catalog::GetEntry<AggregateFunctionCatalogEntry>(context, window.catalog, window.schema,
		                                                             window.function_name, false, error_context);
		D_ASSERT(func->type == CatalogType::AGGREGATE_FUNCTION_ENTRY);

		// bind the aggregate
		string error;
		FunctionBinder function_binder(context);
		auto best_function = function_binder.BindFunction(func->name, func->functions, types, error);
		if (best_function == DConstants::INVALID_INDEX) {
			throw BinderException(binder.FormatError(window, error));
		}
		// found a matching function! bind it as an aggregate
		auto bound_function = func->functions.GetFunctionByOffset(best_function);
		auto bound_aggregate = function_binder.BindAggregateFunction(bound_function, std::move(children));
		// create the aggregate
		aggregate = make_unique<AggregateFunction>(bound_aggregate->function);
		bind_info = std::move(bound_aggregate->bind_info);
		children = std::move(bound_aggregate->children);
		sql_type = bound_aggregate->return_type;
	} else {
		// fetch the child of the non-aggregate window function (if any)
		sql_type = ResolveWindowExpressionType(window.type, types);
	}
	auto result = make_unique<BoundWindowExpression>(window.type, sql_type, std::move(aggregate), std::move(bind_info));
	result->children = std::move(children);
	for (auto &child : window.partitions) {
		result->partitions.push_back(GetExpression(child));
	}
	result->ignore_nulls = window.ignore_nulls;

	// Convert RANGE boundary expressions to ORDER +/- expressions.
	// Note that PRECEEDING and FOLLOWING refer to the sequential order in the frame,
	// not the natural ordering of the type. This means that the offset arithmetic must be reversed
	// for ORDER BY DESC.
	auto &config = DBConfig::GetConfig(context);
	auto range_sense = OrderType::INVALID;
	LogicalType start_type = LogicalType::BIGINT;
	if (window.start == WindowBoundary::EXPR_PRECEDING_RANGE) {
		D_ASSERT(window.orders.size() == 1);
		range_sense = ResolveOrderType(config, window.orders[0].type);
		const auto name = (range_sense == OrderType::ASCENDING) ? "-" : "+";
		start_type = BindRangeExpression(context, name, window.start_expr, window.orders[0].expression);
	} else if (window.start == WindowBoundary::EXPR_FOLLOWING_RANGE) {
		D_ASSERT(window.orders.size() == 1);
		range_sense = ResolveOrderType(config, window.orders[0].type);
		const auto name = (range_sense == OrderType::ASCENDING) ? "+" : "-";
		start_type = BindRangeExpression(context, name, window.start_expr, window.orders[0].expression);
	}

	LogicalType end_type = LogicalType::BIGINT;
	if (window.end == WindowBoundary::EXPR_PRECEDING_RANGE) {
		D_ASSERT(window.orders.size() == 1);
		range_sense = ResolveOrderType(config, window.orders[0].type);
		const auto name = (range_sense == OrderType::ASCENDING) ? "-" : "+";
		end_type = BindRangeExpression(context, name, window.end_expr, window.orders[0].expression);
	} else if (window.end == WindowBoundary::EXPR_FOLLOWING_RANGE) {
		D_ASSERT(window.orders.size() == 1);
		range_sense = ResolveOrderType(config, window.orders[0].type);
		const auto name = (range_sense == OrderType::ASCENDING) ? "+" : "-";
		end_type = BindRangeExpression(context, name, window.end_expr, window.orders[0].expression);
	}

	// Cast ORDER and boundary expressions to the same type
	if (range_sense != OrderType::INVALID) {
		D_ASSERT(window.orders.size() == 1);

		auto &order_expr = window.orders[0].expression;
		D_ASSERT(order_expr.get());
		D_ASSERT(order_expr->expression_class == ExpressionClass::BOUND_EXPRESSION);
		auto &bound_order = (BoundExpression &)*order_expr;
		auto order_type = bound_order.expr->return_type;
		if (window.start_expr) {
			order_type = LogicalType::MaxLogicalType(order_type, start_type);
		}
		if (window.end_expr) {
			order_type = LogicalType::MaxLogicalType(order_type, end_type);
		}

		// Cast all three to match
		bound_order.expr = BoundCastExpression::AddCastToType(context, std::move(bound_order.expr), order_type);
		start_type = end_type = order_type;
	}

	for (auto &order : window.orders) {
		auto type = ResolveOrderType(config, order.type);
		auto null_order = ResolveNullOrder(config, order.null_order);
		auto expression = GetExpression(order.expression);
		result->orders.emplace_back(type, null_order, std::move(expression));
	}

	result->filter_expr = CastWindowExpression(window.filter_expr, LogicalType::BOOLEAN);

	result->start_expr = CastWindowExpression(window.start_expr, start_type);
	result->end_expr = CastWindowExpression(window.end_expr, end_type);
	result->offset_expr = CastWindowExpression(window.offset_expr, LogicalType::BIGINT);
	result->default_expr = CastWindowExpression(window.default_expr, result->return_type);
	result->start = window.start;
	result->end = window.end;

	// create a BoundColumnRef that references this entry
	auto colref = make_unique<BoundColumnRefExpression>(std::move(name), result->return_type,
	                                                    ColumnBinding(node.window_index, node.windows.size()), depth);
	// move the WINDOW expression into the set of bound windows
	node.windows.push_back(std::move(result));
	return BindResult(std::move(colref));
}

} // namespace duckdb








namespace duckdb {

unique_ptr<BoundQueryNode> Binder::BindNode(RecursiveCTENode &statement) {
	auto result = make_unique<BoundRecursiveCTENode>();

	// first recursively visit the recursive CTE operations
	// the left side is visited first and is added to the BindContext of the right side
	D_ASSERT(statement.left);
	D_ASSERT(statement.right);

	result->ctename = statement.ctename;
	result->union_all = statement.union_all;
	result->setop_index = GenerateTableIndex();

	result->left_binder = Binder::CreateBinder(context, this);
	result->left = result->left_binder->BindNode(*statement.left);

	// the result types of the CTE are the types of the LHS
	result->types = result->left->types;
	// names are picked from the LHS, unless aliases are explicitly specified
	result->names = result->left->names;
	for (idx_t i = 0; i < statement.aliases.size() && i < result->names.size(); i++) {
		result->names[i] = statement.aliases[i];
	}

	// This allows the right side to reference the CTE recursively
	bind_context.AddGenericBinding(result->setop_index, statement.ctename, result->names, result->types);

	result->right_binder = Binder::CreateBinder(context, this);

	// Add bindings of left side to temporary CTE bindings context
	result->right_binder->bind_context.AddCTEBinding(result->setop_index, statement.ctename, result->names,
	                                                 result->types);
	result->right = result->right_binder->BindNode(*statement.right);

	// move the correlated expressions from the child binders to this binder
	MoveCorrelatedExpressions(*result->left_binder);
	MoveCorrelatedExpressions(*result->right_binder);

	// now both sides have been bound we can resolve types
	if (result->left->types.size() != result->right->types.size()) {
		throw BinderException("Set operations can only apply to expressions with the "
		                      "same number of result columns");
	}

	if (!statement.modifiers.empty()) {
		throw NotImplementedException("FIXME: bind modifiers in recursive CTE");
	}

	return std::move(result);
}

} // namespace duckdb
























namespace duckdb {

unique_ptr<Expression> Binder::BindOrderExpression(OrderBinder &order_binder, unique_ptr<ParsedExpression> expr) {
	// we treat the Distinct list as a order by
	auto bound_expr = order_binder.Bind(std::move(expr));
	if (!bound_expr) {
		// DISTINCT ON non-integer constant
		// remove the expression from the DISTINCT ON list
		return nullptr;
	}
	D_ASSERT(bound_expr->type == ExpressionType::BOUND_COLUMN_REF);
	return bound_expr;
}

unique_ptr<Expression> Binder::BindDelimiter(ClientContext &context, OrderBinder &order_binder,
                                             unique_ptr<ParsedExpression> delimiter, const LogicalType &type,
                                             Value &delimiter_value) {
	auto new_binder = Binder::CreateBinder(context, this, true);
	if (delimiter->HasSubquery()) {
		if (!order_binder.HasExtraList()) {
			throw BinderException("Subquery in LIMIT/OFFSET not supported in set operation");
		}
		return order_binder.CreateExtraReference(std::move(delimiter));
	}
	ExpressionBinder expr_binder(*new_binder, context);
	expr_binder.target_type = type;
	auto expr = expr_binder.Bind(delimiter);
	if (expr->IsFoldable()) {
		//! this is a constant
		delimiter_value = ExpressionExecutor::EvaluateScalar(context, *expr).CastAs(context, type);
		return nullptr;
	}
	if (!new_binder->correlated_columns.empty()) {
		throw BinderException("Correlated columns not supported in LIMIT/OFFSET");
	}
	// move any correlated columns to this binder
	MoveCorrelatedExpressions(*new_binder);
	return expr;
}

unique_ptr<BoundResultModifier> Binder::BindLimit(OrderBinder &order_binder, LimitModifier &limit_mod) {
	auto result = make_unique<BoundLimitModifier>();
	if (limit_mod.limit) {
		Value val;
		result->limit = BindDelimiter(context, order_binder, std::move(limit_mod.limit), LogicalType::BIGINT, val);
		if (!result->limit) {
			result->limit_val = val.IsNull() ? NumericLimits<int64_t>::Maximum() : val.GetValue<int64_t>();
			if (result->limit_val < 0) {
				throw BinderException("LIMIT cannot be negative");
			}
		}
	}
	if (limit_mod.offset) {
		Value val;
		result->offset = BindDelimiter(context, order_binder, std::move(limit_mod.offset), LogicalType::BIGINT, val);
		if (!result->offset) {
			result->offset_val = val.IsNull() ? 0 : val.GetValue<int64_t>();
			if (result->offset_val < 0) {
				throw BinderException("OFFSET cannot be negative");
			}
		}
	}
	return std::move(result);
}

unique_ptr<BoundResultModifier> Binder::BindLimitPercent(OrderBinder &order_binder, LimitPercentModifier &limit_mod) {
	auto result = make_unique<BoundLimitPercentModifier>();
	if (limit_mod.limit) {
		Value val;
		result->limit = BindDelimiter(context, order_binder, std::move(limit_mod.limit), LogicalType::DOUBLE, val);
		if (!result->limit) {
			result->limit_percent = val.IsNull() ? 100 : val.GetValue<double>();
			if (result->limit_percent < 0.0) {
				throw Exception("Limit percentage can't be negative value");
			}
		}
	}
	if (limit_mod.offset) {
		Value val;
		result->offset = BindDelimiter(context, order_binder, std::move(limit_mod.offset), LogicalType::BIGINT, val);
		if (!result->offset) {
			result->offset_val = val.IsNull() ? 0 : val.GetValue<int64_t>();
		}
	}
	return std::move(result);
}

void Binder::BindModifiers(OrderBinder &order_binder, QueryNode &statement, BoundQueryNode &result) {
	for (auto &mod : statement.modifiers) {
		unique_ptr<BoundResultModifier> bound_modifier;
		switch (mod->type) {
		case ResultModifierType::DISTINCT_MODIFIER: {
			auto &distinct = (DistinctModifier &)*mod;
			auto bound_distinct = make_unique<BoundDistinctModifier>();
			if (distinct.distinct_on_targets.empty()) {
				for (idx_t i = 0; i < result.names.size(); i++) {
					distinct.distinct_on_targets.push_back(make_unique<ConstantExpression>(Value::INTEGER(1 + i)));
				}
			}
			for (auto &distinct_on_target : distinct.distinct_on_targets) {
				auto expr = BindOrderExpression(order_binder, std::move(distinct_on_target));
				if (!expr) {
					continue;
				}
				bound_distinct->target_distincts.push_back(std::move(expr));
			}
			bound_modifier = std::move(bound_distinct);
			break;
		}
		case ResultModifierType::ORDER_MODIFIER: {
			auto &order = (OrderModifier &)*mod;
			auto bound_order = make_unique<BoundOrderModifier>();
			auto &config = DBConfig::GetConfig(context);
			D_ASSERT(!order.orders.empty());
			if (order.orders[0].expression->type == ExpressionType::STAR) {
				// ORDER BY ALL
				// replace the order list with the maximum order by count
				D_ASSERT(order.orders.size() == 1);
				auto order_type = order.orders[0].type;
				auto null_order = order.orders[0].null_order;

				vector<OrderByNode> new_orders;
				for (idx_t i = 0; i < order_binder.MaxCount(); i++) {
					new_orders.emplace_back(order_type, null_order,
					                        make_unique<ConstantExpression>(Value::INTEGER(i + 1)));
				}
				order.orders = std::move(new_orders);
			}
			for (auto &order_node : order.orders) {
				auto order_expression = BindOrderExpression(order_binder, std::move(order_node.expression));
				if (!order_expression) {
					continue;
				}
				auto type =
				    order_node.type == OrderType::ORDER_DEFAULT ? config.options.default_order_type : order_node.type;
				auto null_order = order_node.null_order == OrderByNullType::ORDER_DEFAULT
				                      ? config.options.default_null_order
				                      : order_node.null_order;
				bound_order->orders.emplace_back(type, null_order, std::move(order_expression));
			}
			if (!bound_order->orders.empty()) {
				bound_modifier = std::move(bound_order);
			}
			break;
		}
		case ResultModifierType::LIMIT_MODIFIER:
			bound_modifier = BindLimit(order_binder, (LimitModifier &)*mod);
			break;
		case ResultModifierType::LIMIT_PERCENT_MODIFIER:
			bound_modifier = BindLimitPercent(order_binder, (LimitPercentModifier &)*mod);
			break;
		default:
			throw Exception("Unsupported result modifier");
		}
		if (bound_modifier) {
			result.modifiers.push_back(std::move(bound_modifier));
		}
	}
}

static void AssignReturnType(unique_ptr<Expression> &expr, const vector<LogicalType> &sql_types,
                             idx_t projection_index) {
	if (!expr) {
		return;
	}
	if (expr->type != ExpressionType::BOUND_COLUMN_REF) {
		return;
	}
	auto &bound_colref = (BoundColumnRefExpression &)*expr;
	bound_colref.return_type = sql_types[bound_colref.binding.column_index];
}

void Binder::BindModifierTypes(BoundQueryNode &result, const vector<LogicalType> &sql_types, idx_t projection_index) {
	for (auto &bound_mod : result.modifiers) {
		switch (bound_mod->type) {
		case ResultModifierType::DISTINCT_MODIFIER: {
			auto &distinct = (BoundDistinctModifier &)*bound_mod;
			if (distinct.target_distincts.empty()) {
				// DISTINCT without a target: push references to the standard select list
				for (idx_t i = 0; i < sql_types.size(); i++) {
					distinct.target_distincts.push_back(
					    make_unique<BoundColumnRefExpression>(sql_types[i], ColumnBinding(projection_index, i)));
				}
			} else {
				// DISTINCT with target list: set types
				for (auto &expr : distinct.target_distincts) {
					D_ASSERT(expr->type == ExpressionType::BOUND_COLUMN_REF);
					auto &bound_colref = (BoundColumnRefExpression &)*expr;
					if (bound_colref.binding.column_index == DConstants::INVALID_INDEX) {
						throw BinderException("Ambiguous name in DISTINCT ON!");
					}
					D_ASSERT(bound_colref.binding.column_index < sql_types.size());
					bound_colref.return_type = sql_types[bound_colref.binding.column_index];
				}
			}
			for (auto &target_distinct : distinct.target_distincts) {
				auto &bound_colref = (BoundColumnRefExpression &)*target_distinct;
				const auto &sql_type = sql_types[bound_colref.binding.column_index];
				if (sql_type.id() == LogicalTypeId::VARCHAR) {
					target_distinct = ExpressionBinder::PushCollation(context, std::move(target_distinct),
					                                                  StringType::GetCollation(sql_type), true);
				}
			}
			break;
		}
		case ResultModifierType::LIMIT_MODIFIER: {
			auto &limit = (BoundLimitModifier &)*bound_mod;
			AssignReturnType(limit.limit, sql_types, projection_index);
			AssignReturnType(limit.offset, sql_types, projection_index);
			break;
		}
		case ResultModifierType::LIMIT_PERCENT_MODIFIER: {
			auto &limit = (BoundLimitPercentModifier &)*bound_mod;
			AssignReturnType(limit.limit, sql_types, projection_index);
			AssignReturnType(limit.offset, sql_types, projection_index);
			break;
		}
		case ResultModifierType::ORDER_MODIFIER: {
			auto &order = (BoundOrderModifier &)*bound_mod;
			for (auto &order_node : order.orders) {
				auto &expr = order_node.expression;
				D_ASSERT(expr->type == ExpressionType::BOUND_COLUMN_REF);
				auto &bound_colref = (BoundColumnRefExpression &)*expr;
				if (bound_colref.binding.column_index == DConstants::INVALID_INDEX) {
					throw BinderException("Ambiguous name in ORDER BY!");
				}
				D_ASSERT(bound_colref.binding.column_index < sql_types.size());
				const auto &sql_type = sql_types[bound_colref.binding.column_index];
				bound_colref.return_type = sql_types[bound_colref.binding.column_index];
				if (sql_type.id() == LogicalTypeId::VARCHAR) {
					order_node.expression = ExpressionBinder::PushCollation(context, std::move(order_node.expression),
					                                                        StringType::GetCollation(sql_type));
				}
			}
			break;
		}
		default:
			break;
		}
	}
}

bool Binder::FindStarExpression(ParsedExpression &expr, StarExpression **star) {
	if (expr.GetExpressionClass() == ExpressionClass::STAR) {
		auto current_star = (StarExpression *)&expr;
		if (*star) {
			// we can have multiple
			if (!StarExpression::Equal(*star, current_star)) {
				throw BinderException(
				    FormatError(expr, "Multiple different STAR/COLUMNS in the same expression are not supported"));
			}
			return true;
		}
		*star = current_star;
		return true;
	}
	bool has_star = false;
	ParsedExpressionIterator::EnumerateChildren(expr, [&](ParsedExpression &child_expr) {
		if (FindStarExpression(child_expr, star)) {
			has_star = true;
		}
	});
	return has_star;
}

void Binder::ReplaceStarExpression(unique_ptr<ParsedExpression> &expr, unique_ptr<ParsedExpression> &replacement) {
	D_ASSERT(expr);
	if (expr->GetExpressionClass() == ExpressionClass::STAR) {
		D_ASSERT(replacement);
		expr = replacement->Copy();
		return;
	}
	ParsedExpressionIterator::EnumerateChildren(
	    *expr, [&](unique_ptr<ParsedExpression> &child_expr) { ReplaceStarExpression(child_expr, replacement); });
}

void Binder::ExpandStarExpression(unique_ptr<ParsedExpression> expr,
                                  vector<unique_ptr<ParsedExpression>> &new_select_list) {
	StarExpression *star = nullptr;
	if (!FindStarExpression(*expr, &star)) {
		// no star expression: add it as-is
		D_ASSERT(!star);
		new_select_list.push_back(std::move(expr));
		return;
	}
	D_ASSERT(star);
	vector<unique_ptr<ParsedExpression>> star_list;
	// we have star expressions! expand the list of star expressions
	bind_context.GenerateAllColumnExpressions(*star, star_list);

	// now perform the replacement
	for (idx_t i = 0; i < star_list.size(); i++) {
		auto new_expr = expr->Copy();
		ReplaceStarExpression(new_expr, star_list[i]);
		new_select_list.push_back(std::move(new_expr));
	}
}

void Binder::ExpandStarExpressions(vector<unique_ptr<ParsedExpression>> &select_list,
                                   vector<unique_ptr<ParsedExpression>> &new_select_list) {
	for (auto &select_element : select_list) {
		ExpandStarExpression(std::move(select_element), new_select_list);
	}
}

unique_ptr<BoundQueryNode> Binder::BindNode(SelectNode &statement) {
	auto result = make_unique<BoundSelectNode>();
	result->projection_index = GenerateTableIndex();
	result->group_index = GenerateTableIndex();
	result->aggregate_index = GenerateTableIndex();
	result->groupings_index = GenerateTableIndex();
	result->window_index = GenerateTableIndex();
	result->unnest_index = GenerateTableIndex();
	result->prune_index = GenerateTableIndex();

	// first bind the FROM table statement
	result->from_table = Bind(*statement.from_table);

	// bind the sample clause
	if (statement.sample) {
		result->sample_options = std::move(statement.sample);
	}

	// visit the select list and expand any "*" statements
	vector<unique_ptr<ParsedExpression>> new_select_list;
	ExpandStarExpressions(statement.select_list, new_select_list);

	if (new_select_list.empty()) {
		throw BinderException("SELECT list is empty after resolving * expressions!");
	}
	statement.select_list = std::move(new_select_list);

	// create a mapping of (alias -> index) and a mapping of (Expression -> index) for the SELECT list
	case_insensitive_map_t<idx_t> alias_map;
	expression_map_t<idx_t> projection_map;
	for (idx_t i = 0; i < statement.select_list.size(); i++) {
		auto &expr = statement.select_list[i];
		result->names.push_back(expr->GetName());
		ExpressionBinder::QualifyColumnNames(*this, expr);
		if (!expr->alias.empty()) {
			alias_map[expr->alias] = i;
			result->names[i] = expr->alias;
		}
		projection_map[expr.get()] = i;
		result->original_expressions.push_back(expr->Copy());
	}
	result->column_count = statement.select_list.size();

	// first visit the WHERE clause
	// the WHERE clause happens before the GROUP BY, PROJECTION or HAVING clauses
	if (statement.where_clause) {
		ColumnAliasBinder alias_binder(*result, alias_map);
		WhereBinder where_binder(*this, context, &alias_binder);
		unique_ptr<ParsedExpression> condition = std::move(statement.where_clause);
		result->where_clause = where_binder.Bind(condition);
	}

	// now bind all the result modifiers; including DISTINCT and ORDER BY targets
	OrderBinder order_binder({this}, result->projection_index, statement, alias_map, projection_map);
	BindModifiers(order_binder, statement, *result);

	vector<unique_ptr<ParsedExpression>> unbound_groups;
	BoundGroupInformation info;
	auto &group_expressions = statement.groups.group_expressions;
	if (!group_expressions.empty()) {
		// the statement has a GROUP BY clause, bind it
		unbound_groups.resize(group_expressions.size());
		GroupBinder group_binder(*this, context, statement, result->group_index, alias_map, info.alias_map);
		for (idx_t i = 0; i < group_expressions.size(); i++) {

			// we keep a copy of the unbound expression;
			// we keep the unbound copy around to check for group references in the SELECT and HAVING clause
			// the reason we want the unbound copy is because we want to figure out whether an expression
			// is a group reference BEFORE binding in the SELECT/HAVING binder
			group_binder.unbound_expression = group_expressions[i]->Copy();
			group_binder.bind_index = i;

			// bind the groups
			LogicalType group_type;
			auto bound_expr = group_binder.Bind(group_expressions[i], &group_type);
			D_ASSERT(bound_expr->return_type.id() != LogicalTypeId::INVALID);

			// push a potential collation, if necessary
			bound_expr = ExpressionBinder::PushCollation(context, std::move(bound_expr),
			                                             StringType::GetCollation(group_type), true);
			result->groups.group_expressions.push_back(std::move(bound_expr));

			// in the unbound expression we DO bind the table names of any ColumnRefs
			// we do this to make sure that "table.a" and "a" are treated the same
			// if we wouldn't do this then (SELECT test.a FROM test GROUP BY a) would not work because "test.a" <> "a"
			// hence we convert "a" -> "test.a" in the unbound expression
			unbound_groups[i] = std::move(group_binder.unbound_expression);
			ExpressionBinder::QualifyColumnNames(*this, unbound_groups[i]);
			info.map[unbound_groups[i].get()] = i;
		}
	}
	result->groups.grouping_sets = std::move(statement.groups.grouping_sets);

	// bind the HAVING clause, if any
	if (statement.having) {
		HavingBinder having_binder(*this, context, *result, info, alias_map, statement.aggregate_handling);
		ExpressionBinder::QualifyColumnNames(*this, statement.having);
		result->having = having_binder.Bind(statement.having);
	}

	// bind the QUALIFY clause, if any
	if (statement.qualify) {
		if (statement.aggregate_handling == AggregateHandling::FORCE_AGGREGATES) {
			throw BinderException("Combining QUALIFY with GROUP BY ALL is not supported yet");
		}
		QualifyBinder qualify_binder(*this, context, *result, info, alias_map);
		ExpressionBinder::QualifyColumnNames(*this, statement.qualify);
		result->qualify = qualify_binder.Bind(statement.qualify);
		if (qualify_binder.HasBoundColumns() && qualify_binder.BoundAggregates()) {
			throw BinderException("Cannot mix aggregates with non-aggregated columns!");
		}
	}

	// after that, we bind to the SELECT list
	SelectBinder select_binder(*this, context, *result, info, alias_map);
	vector<LogicalType> internal_sql_types;
	for (idx_t i = 0; i < statement.select_list.size(); i++) {
		bool is_window = statement.select_list[i]->IsWindow();
		idx_t unnest_count = result->unnests.size();
		LogicalType result_type;
		auto expr = select_binder.Bind(statement.select_list[i], &result_type);
		if (statement.aggregate_handling == AggregateHandling::FORCE_AGGREGATES && select_binder.HasBoundColumns()) {
			if (select_binder.BoundAggregates()) {
				throw BinderException("Cannot mix aggregates with non-aggregated columns!");
			}
			if (is_window) {
				throw BinderException("Cannot group on a window clause");
			}
			if (result->unnests.size() > unnest_count) {
				throw BinderException("Cannot group on an UNNEST or UNLIST clause");
			}
			// we are forcing aggregates, and the node has columns bound
			// this entry becomes a group
			auto group_ref = make_unique<BoundColumnRefExpression>(
			    expr->return_type, ColumnBinding(result->group_index, result->groups.group_expressions.size()));
			result->groups.group_expressions.push_back(std::move(expr));
			expr = std::move(group_ref);
		}
		result->select_list.push_back(std::move(expr));
		if (i < result->column_count) {
			result->types.push_back(result_type);
		}
		internal_sql_types.push_back(result_type);
		if (statement.aggregate_handling == AggregateHandling::FORCE_AGGREGATES) {
			select_binder.ResetBindings();
		}
	}
	result->need_prune = result->select_list.size() > result->column_count;

	// in the normal select binder, we bind columns as if there is no aggregation
	// i.e. in the query [SELECT i, SUM(i) FROM integers;] the "i" will be bound as a normal column
	// since we have an aggregation, we need to either (1) throw an error, or (2) wrap the column in a FIRST() aggregate
	// we choose the former one [CONTROVERSIAL: this is the PostgreSQL behavior]
	if (!result->groups.group_expressions.empty() || !result->aggregates.empty() || statement.having ||
	    !result->groups.grouping_sets.empty()) {
		if (statement.aggregate_handling == AggregateHandling::NO_AGGREGATES_ALLOWED) {
			throw BinderException("Aggregates cannot be present in a Project relation!");
		} else if (statement.aggregate_handling == AggregateHandling::STANDARD_HANDLING) {
			if (select_binder.HasBoundColumns()) {
				auto &bound_columns = select_binder.GetBoundColumns();
				string error;
				error = "column \"%s\" must appear in the GROUP BY clause or must be part of an aggregate function.";
				error += "\nEither add it to the GROUP BY list, or use \"ANY_VALUE(%s)\" if the exact value of \"%s\" "
				         "is not important.";
				throw BinderException(FormatError(bound_columns[0].query_location, error, bound_columns[0].name,
				                                  bound_columns[0].name, bound_columns[0].name));
			}
		}
	}

	// QUALIFY clause requires at least one window function to be specified in at least one of the SELECT column list or
	// the filter predicate of the QUALIFY clause
	if (statement.qualify && result->windows.empty()) {
		throw BinderException("at least one window function must appear in the SELECT column or QUALIFY clause");
	}

	// now that the SELECT list is bound, we set the types of DISTINCT/ORDER BY expressions
	BindModifierTypes(*result, internal_sql_types, result->projection_index);
	return std::move(result);
}

} // namespace duckdb
