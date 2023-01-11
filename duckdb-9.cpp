// See https://raw.githubusercontent.com/duckdb/duckdb/master/LICENSE for licensing information

#include "duckdb.hpp"
#include "duckdb-internal.hpp"
#ifndef DUCKDB_AMALGAMATION
#error header mismatch
#endif












namespace duckdb {

static void GatherAliases(BoundQueryNode &node, case_insensitive_map_t<idx_t> &aliases,
                          expression_map_t<idx_t> &expressions, const vector<idx_t> &reorder_idx) {
	if (node.type == QueryNodeType::SET_OPERATION_NODE) {
		// setop, recurse
		auto &setop = (BoundSetOperationNode &)node;

		// create new reorder index
		if (setop.setop_type == SetOperationType::UNION_BY_NAME) {
			vector<idx_t> new_left_reorder_idx(setop.left_reorder_idx.size());
			vector<idx_t> new_right_reorder_idx(setop.right_reorder_idx.size());
			for (idx_t i = 0; i < setop.left_reorder_idx.size(); ++i) {
				new_left_reorder_idx[i] = reorder_idx[setop.left_reorder_idx[i]];
			}

			for (idx_t i = 0; i < setop.right_reorder_idx.size(); ++i) {
				new_right_reorder_idx[i] = reorder_idx[setop.right_reorder_idx[i]];
			}

			// use new reorder index
			GatherAliases(*setop.left, aliases, expressions, new_left_reorder_idx);
			GatherAliases(*setop.right, aliases, expressions, new_right_reorder_idx);
			return;
		}

		GatherAliases(*setop.left, aliases, expressions, reorder_idx);
		GatherAliases(*setop.right, aliases, expressions, reorder_idx);
	} else {
		// query node
		D_ASSERT(node.type == QueryNodeType::SELECT_NODE);
		auto &select = (BoundSelectNode &)node;
		// fill the alias lists
		for (idx_t i = 0; i < select.names.size(); i++) {
			auto &name = select.names[i];
			auto &expr = select.original_expressions[i];
			// first check if the alias is already in there
			auto entry = aliases.find(name);

			idx_t index = reorder_idx[i];

			if (entry != aliases.end()) {
				// the alias already exists
				// check if there is a conflict

				if (entry->second != index) {
					// there is a conflict
					// we place "-1" in the aliases map at this location
					// "-1" signifies that there is an ambiguous reference
					aliases[name] = DConstants::INVALID_INDEX;
				}
			} else {
				// the alias is not in there yet, just assign it
				aliases[name] = index;
			}
			// now check if the node is already in the set of expressions
			auto expr_entry = expressions.find(expr.get());
			if (expr_entry != expressions.end()) {
				// the node is in there
				// repeat the same as with the alias: if there is an ambiguity we insert "-1"
				if (expr_entry->second != index) {
					expressions[expr.get()] = DConstants::INVALID_INDEX;
				}
			} else {
				// not in there yet, just place it in there
				expressions[expr.get()] = index;
			}
		}
	}
}

static void BuildUnionByNameInfo(BoundSetOperationNode &result, bool can_contain_nulls) {
	D_ASSERT(result.setop_type == SetOperationType::UNION_BY_NAME);
	case_insensitive_map_t<idx_t> left_names_map;
	case_insensitive_map_t<idx_t> right_names_map;

	BoundQueryNode *left_node = result.left.get();
	BoundQueryNode *right_node = result.right.get();

	// Build a name_map to use to check if a name exists
	// We throw a binder exception if two same name in the SELECT list
	for (idx_t i = 0; i < left_node->names.size(); ++i) {
		if (left_names_map.find(left_node->names[i]) != left_names_map.end()) {
			throw BinderException("UNION(ALL) BY NAME operation doesn't support same name in SELECT list");
		}
		left_names_map[left_node->names[i]] = i;
	}

	for (idx_t i = 0; i < right_node->names.size(); ++i) {
		if (right_names_map.find(right_node->names[i]) != right_names_map.end()) {
			throw BinderException("UNION(ALL) BY NAME operation doesn't support same name in SELECT list");
		}
		if (left_names_map.find(right_node->names[i]) == left_names_map.end()) {
			result.names.push_back(right_node->names[i]);
		}
		right_names_map[right_node->names[i]] = i;
	}

	idx_t new_size = result.names.size();
	bool need_reorder = false;
	vector<idx_t> left_reorder_idx(left_node->names.size());
	vector<idx_t> right_reorder_idx(right_node->names.size());

	// Construct return type and reorder_idxs
	// reorder_idxs is used to gather correct alias_map
	// and expression_map in GatherAlias(...)
	for (idx_t i = 0; i < new_size; ++i) {
		auto left_index = left_names_map.find(result.names[i]);
		auto right_index = right_names_map.find(result.names[i]);
		bool left_exist = left_index != left_names_map.end();
		bool right_exist = right_index != right_names_map.end();
		LogicalType result_type;
		if (left_exist && right_exist) {
			result_type = LogicalType::MaxLogicalType(left_node->types[left_index->second],
			                                          right_node->types[right_index->second]);
			if (left_index->second != i || right_index->second != i) {
				need_reorder = true;
			}
			left_reorder_idx[left_index->second] = i;
			right_reorder_idx[right_index->second] = i;
		} else if (left_exist) {
			result_type = left_node->types[left_index->second];
			need_reorder = true;
			left_reorder_idx[left_index->second] = i;
		} else {
			D_ASSERT(right_exist);
			result_type = right_node->types[right_index->second];
			need_reorder = true;
			right_reorder_idx[right_index->second] = i;
		}

		if (!can_contain_nulls) {
			if (ExpressionBinder::ContainsNullType(result_type)) {
				result_type = ExpressionBinder::ExchangeNullType(result_type);
			}
		}

		result.types.push_back(result_type);
	}

	result.left_reorder_idx = std::move(left_reorder_idx);
	result.right_reorder_idx = std::move(right_reorder_idx);

	// If reorder is required, collect reorder expressions for push projection
	// into the two child nodes of union node
	if (need_reorder) {
		for (idx_t i = 0; i < new_size; ++i) {
			auto left_index = left_names_map.find(result.names[i]);
			auto right_index = right_names_map.find(result.names[i]);
			bool left_exist = left_index != left_names_map.end();
			bool right_exist = right_index != right_names_map.end();
			unique_ptr<Expression> left_reorder_expr;
			unique_ptr<Expression> right_reorder_expr;
			if (left_exist && right_exist) {
				left_reorder_expr = make_unique<BoundColumnRefExpression>(
				    left_node->types[left_index->second], ColumnBinding(left_node->GetRootIndex(), left_index->second));
				right_reorder_expr = make_unique<BoundColumnRefExpression>(
				    right_node->types[right_index->second],
				    ColumnBinding(right_node->GetRootIndex(), right_index->second));
			} else if (left_exist) {
				left_reorder_expr = make_unique<BoundColumnRefExpression>(
				    left_node->types[left_index->second], ColumnBinding(left_node->GetRootIndex(), left_index->second));
				// create null value here
				right_reorder_expr = make_unique<BoundConstantExpression>(Value(result.types[i]));
			} else {
				D_ASSERT(right_exist);
				left_reorder_expr = make_unique<BoundConstantExpression>(Value(result.types[i]));
				right_reorder_expr = make_unique<BoundColumnRefExpression>(
				    right_node->types[right_index->second],
				    ColumnBinding(right_node->GetRootIndex(), right_index->second));
			}
			result.left_reorder_exprs.push_back(std::move(left_reorder_expr));
			result.right_reorder_exprs.push_back(std::move(right_reorder_expr));
		}
	}
}

unique_ptr<BoundQueryNode> Binder::BindNode(SetOperationNode &statement) {
	auto result = make_unique<BoundSetOperationNode>();
	result->setop_type = statement.setop_type;

	// first recursively visit the set operations
	// both the left and right sides have an independent BindContext and Binder
	D_ASSERT(statement.left);
	D_ASSERT(statement.right);

	result->setop_index = GenerateTableIndex();

	result->left_binder = Binder::CreateBinder(context, this);
	result->left_binder->can_contain_nulls = true;
	result->left = result->left_binder->BindNode(*statement.left);
	result->right_binder = Binder::CreateBinder(context, this);
	result->right_binder->can_contain_nulls = true;
	result->right = result->right_binder->BindNode(*statement.right);

	result->names = result->left->names;

	// move the correlated expressions from the child binders to this binder
	MoveCorrelatedExpressions(*result->left_binder);
	MoveCorrelatedExpressions(*result->right_binder);

	// now both sides have been bound we can resolve types
	if (result->setop_type != SetOperationType::UNION_BY_NAME &&
	    result->left->types.size() != result->right->types.size()) {
		throw BinderException("Set operations can only apply to expressions with the "
		                      "same number of result columns");
	}

	if (result->setop_type == SetOperationType::UNION_BY_NAME) {
		BuildUnionByNameInfo(*result, can_contain_nulls);

	} else {
		// figure out the types of the setop result by picking the max of both
		for (idx_t i = 0; i < result->left->types.size(); i++) {
			auto result_type = LogicalType::MaxLogicalType(result->left->types[i], result->right->types[i]);
			if (!can_contain_nulls) {
				if (ExpressionBinder::ContainsNullType(result_type)) {
					result_type = ExpressionBinder::ExchangeNullType(result_type);
				}
			}
			result->types.push_back(result_type);
		}
	}

	if (!statement.modifiers.empty()) {
		// handle the ORDER BY/DISTINCT clauses

		// we recursively visit the children of this node to extract aliases and expressions that can be referenced
		// in the ORDER BY
		case_insensitive_map_t<idx_t> alias_map;
		expression_map_t<idx_t> expression_map;

		if (result->setop_type == SetOperationType::UNION_BY_NAME) {
			GatherAliases(*result->left, alias_map, expression_map, result->left_reorder_idx);
			GatherAliases(*result->right, alias_map, expression_map, result->right_reorder_idx);
		} else {
			vector<idx_t> reorder_idx;
			for (idx_t i = 0; i < result->names.size(); i++) {
				reorder_idx.push_back(i);
			}
			GatherAliases(*result, alias_map, expression_map, reorder_idx);
		}
		// now we perform the actual resolution of the ORDER BY/DISTINCT expressions
		OrderBinder order_binder({result->left_binder.get(), result->right_binder.get()}, result->setop_index,
		                         alias_map, expression_map, result->names.size());
		BindModifiers(order_binder, statement, *result);
	}

	// finally bind the types of the ORDER/DISTINCT clause expressions
	BindModifierTypes(*result, result->types, result->setop_index);
	return std::move(result);
}

} // namespace duckdb
















namespace duckdb {

unique_ptr<QueryNode> Binder::BindTableMacro(FunctionExpression &function, TableMacroCatalogEntry *macro_func,
                                             idx_t depth) {

	auto &macro_def = (TableMacroFunction &)*macro_func->function;
	auto node = macro_def.query_node->Copy();

	// auto &macro_def = *macro_func->function;

	// validate the arguments and separate positional and default arguments
	vector<unique_ptr<ParsedExpression>> positionals;
	unordered_map<string, unique_ptr<ParsedExpression>> defaults;
	string error =
	    MacroFunction::ValidateArguments(*macro_func->function, macro_func->name, function, positionals, defaults);
	if (!error.empty()) {
		// cannot use error below as binder rnot in scope
		// return BindResult(binder. FormatError(*expr->get(), error));
		throw BinderException(FormatError(function, error));
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

	// We need an ExpressionBinder so that we can call ExpressionBinder::ReplaceMacroParametersRecursive()
	auto eb = ExpressionBinder(*this, this->context);

	eb.macro_binding = new_macro_binding.get();

	/* Does it all goes throu every expression in a selectstmt  */
	ParsedExpressionIterator::EnumerateQueryNodeChildren(
	    *node, [&](unique_ptr<ParsedExpression> &child) { eb.ReplaceMacroParametersRecursive(child); });

	return node;
}

} // namespace duckdb







namespace duckdb {

unique_ptr<LogicalOperator> Binder::VisitQueryNode(BoundQueryNode &node, unique_ptr<LogicalOperator> root) {
	D_ASSERT(root);
	for (auto &mod : node.modifiers) {
		switch (mod->type) {
		case ResultModifierType::DISTINCT_MODIFIER: {
			auto &bound = (BoundDistinctModifier &)*mod;
			auto distinct = make_unique<LogicalDistinct>(std::move(bound.target_distincts));
			distinct->AddChild(std::move(root));
			root = std::move(distinct);
			break;
		}
		case ResultModifierType::ORDER_MODIFIER: {
			auto &bound = (BoundOrderModifier &)*mod;
			auto order = make_unique<LogicalOrder>(std::move(bound.orders));
			order->AddChild(std::move(root));
			root = std::move(order);
			break;
		}
		case ResultModifierType::LIMIT_MODIFIER: {
			auto &bound = (BoundLimitModifier &)*mod;
			auto limit = make_unique<LogicalLimit>(bound.limit_val, bound.offset_val, std::move(bound.limit),
			                                       std::move(bound.offset));
			limit->AddChild(std::move(root));
			root = std::move(limit);
			break;
		}
		case ResultModifierType::LIMIT_PERCENT_MODIFIER: {
			auto &bound = (BoundLimitPercentModifier &)*mod;
			auto limit = make_unique<LogicalLimitPercent>(bound.limit_percent, bound.offset_val, std::move(bound.limit),
			                                              std::move(bound.offset));
			limit->AddChild(std::move(root));
			root = std::move(limit);
			break;
		}
		default:
			throw BinderException("Unimplemented modifier type!");
		}
	}
	return root;
}

} // namespace duckdb








namespace duckdb {

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundRecursiveCTENode &node) {
	// Generate the logical plan for the left and right sides of the set operation
	node.left_binder->plan_subquery = plan_subquery;
	node.right_binder->plan_subquery = plan_subquery;

	auto left_node = node.left_binder->CreatePlan(*node.left);
	auto right_node = node.right_binder->CreatePlan(*node.right);

	// check if there are any unplanned subqueries left in either child
	has_unplanned_subqueries =
	    node.left_binder->has_unplanned_subqueries || node.right_binder->has_unplanned_subqueries;

	// for both the left and right sides, cast them to the same types
	left_node = CastLogicalOperatorToTypes(node.left->types, node.types, std::move(left_node));
	right_node = CastLogicalOperatorToTypes(node.right->types, node.types, std::move(right_node));

	if (!node.right_binder->bind_context.cte_references[node.ctename] ||
	    *node.right_binder->bind_context.cte_references[node.ctename] == 0) {
		auto root = make_unique<LogicalSetOperation>(node.setop_index, node.types.size(), std::move(left_node),
		                                             std::move(right_node), LogicalOperatorType::LOGICAL_UNION);
		return VisitQueryNode(node, std::move(root));
	}
	auto root =
	    make_unique<LogicalRecursiveCTE>(node.setop_index, node.types.size(), node.union_all, std::move(left_node),
	                                     std::move(right_node), LogicalOperatorType::LOGICAL_RECURSIVE_CTE);

	return VisitQueryNode(node, std::move(root));
}

} // namespace duckdb










namespace duckdb {

unique_ptr<LogicalOperator> Binder::PlanFilter(unique_ptr<Expression> condition, unique_ptr<LogicalOperator> root) {
	PlanSubqueries(&condition, &root);
	auto filter = make_unique<LogicalFilter>(std::move(condition));
	filter->AddChild(std::move(root));
	return std::move(filter);
}

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundSelectNode &statement) {
	unique_ptr<LogicalOperator> root;
	D_ASSERT(statement.from_table);
	root = CreatePlan(*statement.from_table);
	D_ASSERT(root);

	// plan the sample clause
	if (statement.sample_options) {
		root = make_unique<LogicalSample>(std::move(statement.sample_options), std::move(root));
	}

	if (statement.where_clause) {
		root = PlanFilter(std::move(statement.where_clause), std::move(root));
	}

	if (!statement.aggregates.empty() || !statement.groups.group_expressions.empty()) {
		if (!statement.groups.group_expressions.empty()) {
			// visit the groups
			for (auto &group : statement.groups.group_expressions) {
				PlanSubqueries(&group, &root);
			}
		}
		// now visit all aggregate expressions
		for (auto &expr : statement.aggregates) {
			PlanSubqueries(&expr, &root);
		}
		// finally create the aggregate node with the group_index and aggregate_index as obtained from the binder
		auto aggregate = make_unique<LogicalAggregate>(statement.group_index, statement.aggregate_index,
		                                               std::move(statement.aggregates));
		aggregate->groups = std::move(statement.groups.group_expressions);
		aggregate->groupings_index = statement.groupings_index;
		aggregate->grouping_sets = std::move(statement.groups.grouping_sets);
		aggregate->grouping_functions = std::move(statement.grouping_functions);

		aggregate->AddChild(std::move(root));
		root = std::move(aggregate);
	} else if (!statement.groups.grouping_sets.empty()) {
		// edge case: we have grouping sets but no groups or aggregates
		// this can only happen if we have e.g. select 1 from tbl group by ();
		// just output a dummy scan
		root = make_unique_base<LogicalOperator, LogicalDummyScan>(statement.group_index);
	}

	if (statement.having) {
		PlanSubqueries(&statement.having, &root);
		auto having = make_unique<LogicalFilter>(std::move(statement.having));

		having->AddChild(std::move(root));
		root = std::move(having);
	}

	if (!statement.windows.empty()) {
		auto win = make_unique<LogicalWindow>(statement.window_index);
		win->expressions = std::move(statement.windows);
		// visit the window expressions
		for (auto &expr : win->expressions) {
			PlanSubqueries(&expr, &root);
		}
		D_ASSERT(!win->expressions.empty());
		win->AddChild(std::move(root));
		root = std::move(win);
	}

	if (statement.qualify) {
		PlanSubqueries(&statement.qualify, &root);
		auto qualify = make_unique<LogicalFilter>(std::move(statement.qualify));

		qualify->AddChild(std::move(root));
		root = std::move(qualify);
	}

	if (!statement.unnests.empty()) {
		auto unnest = make_unique<LogicalUnnest>(statement.unnest_index);
		unnest->expressions = std::move(statement.unnests);
		// visit the unnest expressions
		for (auto &expr : unnest->expressions) {
			PlanSubqueries(&expr, &root);
		}
		D_ASSERT(!unnest->expressions.empty());
		unnest->AddChild(std::move(root));
		root = std::move(unnest);
	}

	for (auto &expr : statement.select_list) {
		PlanSubqueries(&expr, &root);
	}

	// create the projection
	auto proj = make_unique<LogicalProjection>(statement.projection_index, std::move(statement.select_list));
	auto &projection = *proj;
	proj->AddChild(std::move(root));
	root = std::move(proj);

	// finish the plan by handling the elements of the QueryNode
	root = VisitQueryNode(statement, std::move(root));

	// add a prune node if necessary
	if (statement.need_prune) {
		D_ASSERT(root);
		vector<unique_ptr<Expression>> prune_expressions;
		for (idx_t i = 0; i < statement.column_count; i++) {
			prune_expressions.push_back(make_unique<BoundColumnRefExpression>(
			    projection.expressions[i]->return_type, ColumnBinding(statement.projection_index, i)));
		}
		auto prune = make_unique<LogicalProjection>(statement.prune_index, std::move(prune_expressions));
		prune->AddChild(std::move(root));
		root = std::move(prune);
	}
	return root;
}

} // namespace duckdb







namespace duckdb {

// Optionally push a PROJECTION operator
unique_ptr<LogicalOperator> Binder::CastLogicalOperatorToTypes(vector<LogicalType> &source_types,
                                                               vector<LogicalType> &target_types,
                                                               unique_ptr<LogicalOperator> op) {
	D_ASSERT(op);
	// first check if we even need to cast
	D_ASSERT(source_types.size() == target_types.size());
	if (source_types == target_types) {
		// source and target types are equal: don't need to cast
		return op;
	}
	// otherwise add casts
	auto node = op.get();
	if (node->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		// "node" is a projection; we can just do the casts in there
		D_ASSERT(node->expressions.size() == source_types.size());
		// add the casts to the selection list
		for (idx_t i = 0; i < target_types.size(); i++) {
			if (source_types[i] != target_types[i]) {
				// differing types, have to add a cast
				string alias = node->expressions[i]->alias;
				node->expressions[i] =
				    BoundCastExpression::AddCastToType(context, std::move(node->expressions[i]), target_types[i]);
				node->expressions[i]->alias = alias;
			}
		}
		return op;
	} else {
		// found a non-projection operator
		// push a new projection containing the casts

		// fetch the set of column bindings
		auto setop_columns = op->GetColumnBindings();
		D_ASSERT(setop_columns.size() == source_types.size());

		// now generate the expression list
		vector<unique_ptr<Expression>> select_list;
		for (idx_t i = 0; i < target_types.size(); i++) {
			unique_ptr<Expression> result = make_unique<BoundColumnRefExpression>(source_types[i], setop_columns[i]);
			if (source_types[i] != target_types[i]) {
				// add a cast only if the source and target types are not equivalent
				result = BoundCastExpression::AddCastToType(context, std::move(result), target_types[i]);
			}
			select_list.push_back(std::move(result));
		}
		auto projection = make_unique<LogicalProjection>(GenerateTableIndex(), std::move(select_list));
		projection->children.push_back(std::move(op));
		return std::move(projection);
	}
}

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundSetOperationNode &node) {
	// Generate the logical plan for the left and right sides of the set operation
	node.left_binder->plan_subquery = plan_subquery;
	node.right_binder->plan_subquery = plan_subquery;

	auto left_node = node.left_binder->CreatePlan(*node.left);
	auto right_node = node.right_binder->CreatePlan(*node.right);

	// Add a new projection to child node
	D_ASSERT(node.left_reorder_exprs.size() == node.right_reorder_exprs.size());
	if (!node.left_reorder_exprs.empty()) {
		D_ASSERT(node.setop_type == SetOperationType::UNION_BY_NAME);
		vector<LogicalType> left_types;
		vector<LogicalType> right_types;
		// We are going to add a new projection operator, so collect the type
		// of reorder exprs in order to call CastLogicalOperatorToTypes()
		for (idx_t i = 0; i < node.left_reorder_exprs.size(); ++i) {
			left_types.push_back(node.left_reorder_exprs[i]->return_type);
			right_types.push_back(node.right_reorder_exprs[i]->return_type);
		}

		auto left_projection = make_unique<LogicalProjection>(GenerateTableIndex(), std::move(node.left_reorder_exprs));
		left_projection->children.push_back(std::move(left_node));
		left_node = std::move(left_projection);

		auto right_projection =
		    make_unique<LogicalProjection>(GenerateTableIndex(), std::move(node.right_reorder_exprs));
		right_projection->children.push_back(std::move(right_node));
		right_node = std::move(right_projection);

		left_node = CastLogicalOperatorToTypes(left_types, node.types, std::move(left_node));
		right_node = CastLogicalOperatorToTypes(right_types, node.types, std::move(right_node));
	} else {
		left_node = CastLogicalOperatorToTypes(node.left->types, node.types, std::move(left_node));
		right_node = CastLogicalOperatorToTypes(node.right->types, node.types, std::move(right_node));
	}

	// check if there are any unplanned subqueries left in either child
	has_unplanned_subqueries =
	    node.left_binder->has_unplanned_subqueries || node.right_binder->has_unplanned_subqueries;

	// create actual logical ops for setops
	LogicalOperatorType logical_type;
	switch (node.setop_type) {
	case SetOperationType::UNION:
	case SetOperationType::UNION_BY_NAME:
		logical_type = LogicalOperatorType::LOGICAL_UNION;
		break;
	case SetOperationType::EXCEPT:
		logical_type = LogicalOperatorType::LOGICAL_EXCEPT;
		break;
	default:
		D_ASSERT(node.setop_type == SetOperationType::INTERSECT);
		logical_type = LogicalOperatorType::LOGICAL_INTERSECT;
		break;
	}

	auto root = make_unique<LogicalSetOperation>(node.setop_index, node.types.size(), std::move(left_node),
	                                             std::move(right_node), logical_type);

	return VisitQueryNode(node, std::move(root));
}

} // namespace duckdb

















namespace duckdb {

static unique_ptr<Expression> PlanUncorrelatedSubquery(Binder &binder, BoundSubqueryExpression &expr,
                                                       unique_ptr<LogicalOperator> &root,
                                                       unique_ptr<LogicalOperator> plan) {
	D_ASSERT(!expr.IsCorrelated());
	switch (expr.subquery_type) {
	case SubqueryType::EXISTS: {
		// uncorrelated EXISTS
		// we only care about existence, hence we push a LIMIT 1 operator
		auto limit = make_unique<LogicalLimit>(1, 0, nullptr, nullptr);
		limit->AddChild(std::move(plan));
		plan = std::move(limit);

		// now we push a COUNT(*) aggregate onto the limit, this will be either 0 or 1 (EXISTS or NOT EXISTS)
		auto count_star_fun = CountStarFun::GetFunction();

		FunctionBinder function_binder(binder.context);
		auto count_star =
		    function_binder.BindAggregateFunction(count_star_fun, {}, nullptr, AggregateType::NON_DISTINCT);
		auto idx_type = count_star->return_type;
		vector<unique_ptr<Expression>> aggregate_list;
		aggregate_list.push_back(std::move(count_star));
		auto aggregate_index = binder.GenerateTableIndex();
		auto aggregate =
		    make_unique<LogicalAggregate>(binder.GenerateTableIndex(), aggregate_index, std::move(aggregate_list));
		aggregate->AddChild(std::move(plan));
		plan = std::move(aggregate);

		// now we push a projection with a comparison to 1
		auto left_child = make_unique<BoundColumnRefExpression>(idx_type, ColumnBinding(aggregate_index, 0));
		auto right_child = make_unique<BoundConstantExpression>(Value::Numeric(idx_type, 1));
		auto comparison = make_unique<BoundComparisonExpression>(ExpressionType::COMPARE_EQUAL, std::move(left_child),
		                                                         std::move(right_child));

		vector<unique_ptr<Expression>> projection_list;
		projection_list.push_back(std::move(comparison));
		auto projection_index = binder.GenerateTableIndex();
		auto projection = make_unique<LogicalProjection>(projection_index, std::move(projection_list));
		projection->AddChild(std::move(plan));
		plan = std::move(projection);

		// we add it to the main query by adding a cross product
		// FIXME: should use something else besides cross product as we always add only one scalar constant
		root = LogicalCrossProduct::Create(std::move(root), std::move(plan));

		// we replace the original subquery with a ColumnRefExpression referring to the result of the projection (either
		// TRUE or FALSE)
		return make_unique<BoundColumnRefExpression>(expr.GetName(), LogicalType::BOOLEAN,
		                                             ColumnBinding(projection_index, 0));
	}
	case SubqueryType::SCALAR: {
		// uncorrelated scalar, we want to return the first entry
		// figure out the table index of the bound table of the entry which we want to return
		auto bindings = plan->GetColumnBindings();
		D_ASSERT(bindings.size() == 1);
		idx_t table_idx = bindings[0].table_index;

		// in the uncorrelated case we are only interested in the first result of the query
		// hence we simply push a LIMIT 1 to get the first row of the subquery
		auto limit = make_unique<LogicalLimit>(1, 0, nullptr, nullptr);
		limit->AddChild(std::move(plan));
		plan = std::move(limit);

		// we push an aggregate that returns the FIRST element
		vector<unique_ptr<Expression>> expressions;
		auto bound = make_unique<BoundColumnRefExpression>(expr.return_type, ColumnBinding(table_idx, 0));
		vector<unique_ptr<Expression>> first_children;
		first_children.push_back(std::move(bound));

		FunctionBinder function_binder(binder.context);
		auto first_agg = function_binder.BindAggregateFunction(
		    FirstFun::GetFunction(expr.return_type), std::move(first_children), nullptr, AggregateType::NON_DISTINCT);

		expressions.push_back(std::move(first_agg));
		auto aggr_index = binder.GenerateTableIndex();
		auto aggr = make_unique<LogicalAggregate>(binder.GenerateTableIndex(), aggr_index, std::move(expressions));
		aggr->AddChild(std::move(plan));
		plan = std::move(aggr);

		// in the uncorrelated case, we add the value to the main query through a cross product
		// FIXME: should use something else besides cross product as we always add only one scalar constant and cross
		// product is not optimized for this.
		D_ASSERT(root);
		root = LogicalCrossProduct::Create(std::move(root), std::move(plan));

		// we replace the original subquery with a BoundColumnRefExpression referring to the first result of the
		// aggregation
		return make_unique<BoundColumnRefExpression>(expr.GetName(), expr.return_type, ColumnBinding(aggr_index, 0));
	}
	default: {
		D_ASSERT(expr.subquery_type == SubqueryType::ANY);
		// we generate a MARK join that results in either (TRUE, FALSE or NULL)
		// subquery has NULL values -> result is (TRUE or NULL)
		// subquery has no NULL values -> result is (TRUE, FALSE or NULL [if input is NULL])
		// fetch the column bindings
		auto plan_columns = plan->GetColumnBindings();

		// then we generate the MARK join with the subquery
		idx_t mark_index = binder.GenerateTableIndex();
		auto join = make_unique<LogicalComparisonJoin>(JoinType::MARK);
		join->mark_index = mark_index;
		join->AddChild(std::move(root));
		join->AddChild(std::move(plan));
		// create the JOIN condition
		JoinCondition cond;
		cond.left = std::move(expr.child);
		cond.right = BoundCastExpression::AddDefaultCastToType(
		    make_unique<BoundColumnRefExpression>(expr.child_type, plan_columns[0]), expr.child_target);
		cond.comparison = expr.comparison_type;
		join->conditions.push_back(std::move(cond));
		root = std::move(join);

		// we replace the original subquery with a BoundColumnRefExpression referring to the mark column
		return make_unique<BoundColumnRefExpression>(expr.GetName(), expr.return_type, ColumnBinding(mark_index, 0));
	}
	}
}

static unique_ptr<LogicalDelimJoin>
CreateDuplicateEliminatedJoin(const vector<CorrelatedColumnInfo> &correlated_columns, JoinType join_type,
                              unique_ptr<LogicalOperator> original_plan, bool perform_delim) {
	auto delim_join = make_unique<LogicalDelimJoin>(join_type);
	if (!perform_delim) {
		// if we are not performing a delim join, we push a row_number() OVER() window operator on the LHS
		// and perform all duplicate elimination on that row number instead
		D_ASSERT(correlated_columns[0].type.id() == LogicalTypeId::BIGINT);
		auto window = make_unique<LogicalWindow>(correlated_columns[0].binding.table_index);
		auto row_number = make_unique<BoundWindowExpression>(ExpressionType::WINDOW_ROW_NUMBER, LogicalType::BIGINT,
		                                                     nullptr, nullptr);
		row_number->start = WindowBoundary::UNBOUNDED_PRECEDING;
		row_number->end = WindowBoundary::CURRENT_ROW_ROWS;
		row_number->alias = "delim_index";
		window->expressions.push_back(std::move(row_number));
		window->AddChild(std::move(original_plan));
		original_plan = std::move(window);
	}
	delim_join->AddChild(std::move(original_plan));
	for (idx_t i = 0; i < correlated_columns.size(); i++) {
		auto &col = correlated_columns[i];
		delim_join->duplicate_eliminated_columns.push_back(
		    make_unique<BoundColumnRefExpression>(col.type, col.binding));
		delim_join->delim_types.push_back(col.type);
	}
	return delim_join;
}

static void CreateDelimJoinConditions(LogicalDelimJoin &delim_join,
                                      const vector<CorrelatedColumnInfo> &correlated_columns,
                                      vector<ColumnBinding> bindings, idx_t base_offset, bool perform_delim) {
	auto col_count = perform_delim ? correlated_columns.size() : 1;
	for (idx_t i = 0; i < col_count; i++) {
		auto &col = correlated_columns[i];
		auto binding_idx = base_offset + i;
		if (binding_idx >= bindings.size()) {
			throw InternalException("Delim join - binding index out of range");
		}
		JoinCondition cond;
		cond.left = make_unique<BoundColumnRefExpression>(col.name, col.type, col.binding);
		cond.right = make_unique<BoundColumnRefExpression>(col.name, col.type, bindings[binding_idx]);
		cond.comparison = ExpressionType::COMPARE_NOT_DISTINCT_FROM;
		delim_join.conditions.push_back(std::move(cond));
	}
}

static bool PerformDelimOnType(const LogicalType &type) {
	if (type.InternalType() == PhysicalType::LIST) {
		return false;
	}
	if (type.InternalType() == PhysicalType::STRUCT) {
		for (auto &entry : StructType::GetChildTypes(type)) {
			if (!PerformDelimOnType(entry.second)) {
				return false;
			}
		}
	}
	return true;
}

static bool PerformDuplicateElimination(Binder &binder, vector<CorrelatedColumnInfo> &correlated_columns) {
	if (!ClientConfig::GetConfig(binder.context).enable_optimizer) {
		// if optimizations are disabled we always do a delim join
		return true;
	}
	bool perform_delim = true;
	for (auto &col : correlated_columns) {
		if (!PerformDelimOnType(col.type)) {
			perform_delim = false;
			break;
		}
	}
	if (perform_delim) {
		return true;
	}
	auto binding = ColumnBinding(binder.GenerateTableIndex(), 0);
	auto type = LogicalType::BIGINT;
	auto name = "delim_index";
	CorrelatedColumnInfo info(binding, type, name, 0);
	correlated_columns.insert(correlated_columns.begin(), std::move(info));
	return false;
}

static unique_ptr<Expression> PlanCorrelatedSubquery(Binder &binder, BoundSubqueryExpression &expr,
                                                     unique_ptr<LogicalOperator> &root,
                                                     unique_ptr<LogicalOperator> plan) {
	auto &correlated_columns = expr.binder->correlated_columns;
	// FIXME: there should be a way of disabling decorrelation for ANY queries as well, but not for now...
	bool perform_delim =
	    expr.subquery_type == SubqueryType::ANY ? true : PerformDuplicateElimination(binder, correlated_columns);
	D_ASSERT(expr.IsCorrelated());
	// correlated subquery
	// for a more in-depth explanation of this code, read the paper "Unnesting Arbitrary Subqueries"
	// we handle three types of correlated subqueries: Scalar, EXISTS and ANY
	// all three cases are very similar with some minor changes (mainly the type of join performed at the end)
	switch (expr.subquery_type) {
	case SubqueryType::SCALAR: {
		// correlated SCALAR query
		// first push a DUPLICATE ELIMINATED join
		// a duplicate eliminated join creates a duplicate eliminated copy of the LHS
		// and pushes it into any DUPLICATE_ELIMINATED SCAN operators on the RHS

		// in the SCALAR case, we create a SINGLE join (because we are only interested in obtaining the value)
		// NULL values are equal in this join because we join on the correlated columns ONLY
		// and e.g. in the query: SELECT (SELECT 42 FROM integers WHERE i1.i IS NULL LIMIT 1) FROM integers i1;
		// the input value NULL will generate the value 42, and we need to join NULL on the LHS with NULL on the RHS
		// the left side is the original plan
		// this is the side that will be duplicate eliminated and pushed into the RHS
		auto delim_join =
		    CreateDuplicateEliminatedJoin(correlated_columns, JoinType::SINGLE, std::move(root), perform_delim);

		// the right side initially is a DEPENDENT join between the duplicate eliminated scan and the subquery
		// HOWEVER: we do not explicitly create the dependent join
		// instead, we eliminate the dependent join by pushing it down into the right side of the plan
		FlattenDependentJoins flatten(binder, correlated_columns, perform_delim);

		// first we check which logical operators have correlated expressions in the first place
		flatten.DetectCorrelatedExpressions(plan.get());
		// now we push the dependent join down
		auto dependent_join = flatten.PushDownDependentJoin(std::move(plan));

		// now the dependent join is fully eliminated
		// we only need to create the join conditions between the LHS and the RHS
		// fetch the set of columns
		auto plan_columns = dependent_join->GetColumnBindings();

		// now create the join conditions
		CreateDelimJoinConditions(*delim_join, correlated_columns, plan_columns, flatten.delim_offset, perform_delim);
		delim_join->AddChild(std::move(dependent_join));
		root = std::move(delim_join);
		// finally push the BoundColumnRefExpression referring to the data element returned by the join
		return make_unique<BoundColumnRefExpression>(expr.GetName(), expr.return_type,
		                                             plan_columns[flatten.data_offset]);
	}
	case SubqueryType::EXISTS: {
		// correlated EXISTS query
		// this query is similar to the correlated SCALAR query, except we use a MARK join here
		idx_t mark_index = binder.GenerateTableIndex();
		auto delim_join =
		    CreateDuplicateEliminatedJoin(correlated_columns, JoinType::MARK, std::move(root), perform_delim);
		delim_join->mark_index = mark_index;
		// RHS
		FlattenDependentJoins flatten(binder, correlated_columns, perform_delim, true);
		flatten.DetectCorrelatedExpressions(plan.get());
		auto dependent_join = flatten.PushDownDependentJoin(std::move(plan));

		// fetch the set of columns
		auto plan_columns = dependent_join->GetColumnBindings();

		// now we create the join conditions between the dependent join and the original table
		CreateDelimJoinConditions(*delim_join, correlated_columns, plan_columns, flatten.delim_offset, perform_delim);
		delim_join->AddChild(std::move(dependent_join));
		root = std::move(delim_join);
		// finally push the BoundColumnRefExpression referring to the marker
		return make_unique<BoundColumnRefExpression>(expr.GetName(), expr.return_type, ColumnBinding(mark_index, 0));
	}
	default: {
		D_ASSERT(expr.subquery_type == SubqueryType::ANY);
		// correlated ANY query
		// this query is similar to the correlated SCALAR query
		// however, in this case we push a correlated MARK join
		// note that in this join null values are NOT equal for ALL columns, but ONLY for the correlated columns
		// the correlated mark join handles this case by itself
		// as the MARK join has one extra join condition (the original condition, of the ANY expression, e.g.
		// [i=ANY(...)])
		idx_t mark_index = binder.GenerateTableIndex();
		auto delim_join =
		    CreateDuplicateEliminatedJoin(correlated_columns, JoinType::MARK, std::move(root), perform_delim);
		delim_join->mark_index = mark_index;
		// RHS
		FlattenDependentJoins flatten(binder, correlated_columns, true, true);
		flatten.DetectCorrelatedExpressions(plan.get());
		auto dependent_join = flatten.PushDownDependentJoin(std::move(plan));

		// fetch the columns
		auto plan_columns = dependent_join->GetColumnBindings();

		// now we create the join conditions between the dependent join and the original table
		CreateDelimJoinConditions(*delim_join, correlated_columns, plan_columns, flatten.delim_offset, perform_delim);
		// add the actual condition based on the ANY/ALL predicate
		JoinCondition compare_cond;
		compare_cond.left = std::move(expr.child);
		compare_cond.right = BoundCastExpression::AddDefaultCastToType(
		    make_unique<BoundColumnRefExpression>(expr.child_type, plan_columns[0]), expr.child_target);
		compare_cond.comparison = expr.comparison_type;
		delim_join->conditions.push_back(std::move(compare_cond));

		delim_join->AddChild(std::move(dependent_join));
		root = std::move(delim_join);
		// finally push the BoundColumnRefExpression referring to the marker
		return make_unique<BoundColumnRefExpression>(expr.GetName(), expr.return_type, ColumnBinding(mark_index, 0));
	}
	}
}

class RecursiveSubqueryPlanner : public LogicalOperatorVisitor {
public:
	explicit RecursiveSubqueryPlanner(Binder &binder) : binder(binder) {
	}
	void VisitOperator(LogicalOperator &op) override {
		if (!op.children.empty()) {
			root = std::move(op.children[0]);
			D_ASSERT(root);
			VisitOperatorExpressions(op);
			op.children[0] = std::move(root);
			for (idx_t i = 0; i < op.children.size(); i++) {
				D_ASSERT(op.children[i]);
				VisitOperator(*op.children[i]);
			}
		}
	}

	unique_ptr<Expression> VisitReplace(BoundSubqueryExpression &expr, unique_ptr<Expression> *expr_ptr) override {
		return binder.PlanSubquery(expr, root);
	}

private:
	unique_ptr<LogicalOperator> root;
	Binder &binder;
};

unique_ptr<Expression> Binder::PlanSubquery(BoundSubqueryExpression &expr, unique_ptr<LogicalOperator> &root) {
	D_ASSERT(root);
	// first we translate the QueryNode of the subquery into a logical plan
	// note that we do not plan nested subqueries yet
	auto sub_binder = Binder::CreateBinder(context, this);
	sub_binder->plan_subquery = false;
	auto subquery_root = sub_binder->CreatePlan(*expr.subquery);
	D_ASSERT(subquery_root);

	// now we actually flatten the subquery
	auto plan = std::move(subquery_root);
	unique_ptr<Expression> result_expression;
	if (!expr.IsCorrelated()) {
		result_expression = PlanUncorrelatedSubquery(*this, expr, root, std::move(plan));
	} else {
		result_expression = PlanCorrelatedSubquery(*this, expr, root, std::move(plan));
	}
	// finally, we recursively plan the nested subqueries (if there are any)
	if (sub_binder->has_unplanned_subqueries) {
		RecursiveSubqueryPlanner plan(*this);
		plan.VisitOperator(*root);
	}
	return result_expression;
}

void Binder::PlanSubqueries(unique_ptr<Expression> *expr_ptr, unique_ptr<LogicalOperator> *root) {
	if (!*expr_ptr) {
		return;
	}
	auto &expr = **expr_ptr;

	// first visit the children of the node, if any
	ExpressionIterator::EnumerateChildren(expr, [&](unique_ptr<Expression> &expr) { PlanSubqueries(&expr, root); });

	// check if this is a subquery node
	if (expr.expression_class == ExpressionClass::BOUND_SUBQUERY) {
		auto &subquery = (BoundSubqueryExpression &)expr;
		// subquery node! plan it
		if (subquery.IsCorrelated() && !plan_subquery) {
			// detected a nested correlated subquery
			// we don't plan it yet here, we are currently planning a subquery
			// nested subqueries will only be planned AFTER the current subquery has been flattened entirely
			has_unplanned_subqueries = true;
			return;
		}
		*expr_ptr = PlanSubquery(subquery, *root);
	}
}

unique_ptr<LogicalOperator> Binder::PlanLateralJoin(unique_ptr<LogicalOperator> left, unique_ptr<LogicalOperator> right,
                                                    vector<CorrelatedColumnInfo> &correlated_columns,
                                                    JoinType join_type, unique_ptr<Expression> condition) {
	// scan the right operator for correlated columns
	// correlated LATERAL JOIN
	vector<JoinCondition> conditions;
	vector<unique_ptr<Expression>> arbitrary_expressions;
	if (condition) {
		// extract join conditions, if there are any
		LogicalComparisonJoin::ExtractJoinConditions(join_type, left, right, std::move(condition), conditions,
		                                             arbitrary_expressions);
	}

	auto perform_delim = PerformDuplicateElimination(*this, correlated_columns);
	auto delim_join = CreateDuplicateEliminatedJoin(correlated_columns, join_type, std::move(left), perform_delim);

	FlattenDependentJoins flatten(*this, correlated_columns, perform_delim);

	// first we check which logical operators have correlated expressions in the first place
	flatten.DetectCorrelatedExpressions(right.get(), true);
	// now we push the dependent join down
	auto dependent_join = flatten.PushDownDependentJoin(std::move(right));

	// now the dependent join is fully eliminated
	// we only need to create the join conditions between the LHS and the RHS
	// fetch the set of columns
	auto plan_columns = dependent_join->GetColumnBindings();

	// now create the join conditions
	// start off with the conditions that were passed in (if any)
	D_ASSERT(delim_join->conditions.empty());
	delim_join->conditions = std::move(conditions);
	// then add the delim join conditions
	CreateDelimJoinConditions(*delim_join, correlated_columns, plan_columns, flatten.delim_offset, perform_delim);
	delim_join->AddChild(std::move(dependent_join));

	// check if there are any arbitrary expressions left
	if (!arbitrary_expressions.empty()) {
		// we can only evaluate scalar arbitrary expressions for inner joins
		if (join_type != JoinType::INNER) {
			throw BinderException(
			    "Join condition for non-inner LATERAL JOIN must be a comparison between the left and right side");
		}
		auto filter = make_unique<LogicalFilter>();
		filter->expressions = std::move(arbitrary_expressions);
		filter->AddChild(std::move(delim_join));
		return std::move(filter);
	}
	return std::move(delim_join);
}

} // namespace duckdb






namespace duckdb {

BoundStatement Binder::Bind(AttachStatement &stmt) {
	BoundStatement result;
	result.types = {LogicalType::BOOLEAN};
	result.names = {"Success"};

	result.plan = make_unique<LogicalSimple>(LogicalOperatorType::LOGICAL_ATTACH, std::move(stmt.info));
	properties.allow_stream_result = false;
	properties.return_type = StatementReturnType::NOTHING;
	return result;
}

} // namespace duckdb






namespace duckdb {

BoundStatement Binder::Bind(CallStatement &stmt) {
	BoundStatement result;

	TableFunctionRef ref;
	ref.function = std::move(stmt.function);

	auto bound_func = Bind(ref);
	auto &bound_table_func = (BoundTableFunction &)*bound_func;
	auto &get = (LogicalGet &)*bound_table_func.get;
	D_ASSERT(get.returned_types.size() > 0);
	for (idx_t i = 0; i < get.returned_types.size(); i++) {
		get.column_ids.push_back(i);
	}

	result.types = get.returned_types;
	result.names = get.names;
	result.plan = CreatePlan(*bound_func);
	properties.return_type = StatementReturnType::QUERY_RESULT;
	return result;
}

} // namespace duckdb





















#include <algorithm>

namespace duckdb {

static vector<idx_t> ColumnListToIndices(const vector<bool> &vec) {
	vector<idx_t> ret;
	for (idx_t i = 0; i < vec.size(); i++) {
		if (vec[i]) {
			ret.push_back(i);
		}
	}
	return ret;
}

vector<string> GetUniqueNames(const vector<string> &original_names) {
	unordered_set<string> name_set;
	vector<string> unique_names;
	unique_names.reserve(original_names.size());

	for (auto &name : original_names) {
		auto insert_result = name_set.insert(name);
		if (insert_result.second == false) {
			// Could not be inserted, name already exists
			idx_t index = 1;
			string postfixed_name;
			while (true) {
				postfixed_name = StringUtil::Format("%s:%d", name, index);
				auto res = name_set.insert(postfixed_name);
				if (!res.second) {
					index++;
					continue;
				}
				break;
			}
			unique_names.push_back(postfixed_name);
		} else {
			unique_names.push_back(name);
		}
	}
	return unique_names;
}

BoundStatement Binder::BindCopyTo(CopyStatement &stmt) {
	// COPY TO a file
	auto &config = DBConfig::GetConfig(context);
	if (!config.options.enable_external_access) {
		throw PermissionException("COPY TO is disabled by configuration");
	}
	BoundStatement result;
	result.types = {LogicalType::BIGINT};
	result.names = {"Count"};

	// lookup the format in the catalog
	auto copy_function =
	    Catalog::GetEntry<CopyFunctionCatalogEntry>(context, INVALID_CATALOG, DEFAULT_SCHEMA, stmt.info->format);
	if (copy_function->function.plan) {
		// plan rewrite COPY TO
		return copy_function->function.plan(*this, stmt);
	}

	// bind the select statement
	auto select_node = Bind(*stmt.select_statement);

	if (!copy_function->function.copy_to_bind) {
		throw NotImplementedException("COPY TO is not supported for FORMAT \"%s\"", stmt.info->format);
	}
	bool use_tmp_file = true;
	bool allow_overwrite = false;
	bool user_set_use_tmp_file = false;
	bool per_thread_output = false;
	vector<idx_t> partition_cols;

	auto original_options = stmt.info->options;
	stmt.info->options.clear();

	for (auto &option : original_options) {
		auto loption = StringUtil::Lower(option.first);
		if (loption == "use_tmp_file") {
			use_tmp_file = option.second[0].CastAs(context, LogicalType::BOOLEAN).GetValue<bool>();
			user_set_use_tmp_file = true;
			continue;
		}
		if (loption == "allow_overwrite") {
			allow_overwrite = option.second[0].CastAs(context, LogicalType::BOOLEAN).GetValue<bool>();
			continue;
		}

		if (loption == "per_thread_output") {
			per_thread_output = option.second[0].CastAs(context, LogicalType::BOOLEAN).GetValue<bool>();
			continue;
		}
		if (loption == "partition_by") {
			auto converted = ConvertVectorToValue(std::move(option.second));
			partition_cols = ColumnListToIndices(ParseColumnList(converted, select_node.names, loption));
			continue;
		}
		stmt.info->options[option.first] = option.second;
	}
	if (user_set_use_tmp_file && per_thread_output) {
		throw NotImplementedException("Can't combine USE_TMP_FILE and PER_THREAD_OUTPUT for COPY");
	}
	if (user_set_use_tmp_file && !partition_cols.empty()) {
		throw NotImplementedException("Can't combine USE_TMP_FILE and PARTITION_BY for COPY");
	}
	if (per_thread_output && !partition_cols.empty()) {
		throw NotImplementedException("Can't combine PER_THREAD_OUTPUT and PARTITION_BY for COPY");
	}
	bool is_file_and_exists = config.file_system->FileExists(stmt.info->file_path);
	bool is_stdout = stmt.info->file_path == "/dev/stdout";
	if (!user_set_use_tmp_file) {
		use_tmp_file = is_file_and_exists && !per_thread_output && partition_cols.empty() && !is_stdout;
	}

	auto unique_column_names = GetUniqueNames(select_node.names);

	auto function_data =
	    copy_function->function.copy_to_bind(context, *stmt.info, unique_column_names, select_node.types);
	// now create the copy information
	auto copy = make_unique<LogicalCopyToFile>(copy_function->function, std::move(function_data));
	copy->file_path = stmt.info->file_path;
	copy->use_tmp_file = use_tmp_file;
	copy->allow_overwrite = allow_overwrite;
	copy->per_thread_output = per_thread_output;
	copy->per_thread_output = per_thread_output;
	copy->partition_output = !partition_cols.empty();
	copy->partition_columns = std::move(partition_cols);

	copy->names = unique_column_names;
	copy->expected_types = select_node.types;

	copy->AddChild(std::move(select_node.plan));

	result.plan = std::move(copy);

	return result;
}

BoundStatement Binder::BindCopyFrom(CopyStatement &stmt) {
	auto &config = DBConfig::GetConfig(context);
	if (!config.options.enable_external_access) {
		throw PermissionException("COPY FROM is disabled by configuration");
	}
	BoundStatement result;
	result.types = {LogicalType::BIGINT};
	result.names = {"Count"};

	D_ASSERT(!stmt.info->table.empty());
	// COPY FROM a file
	// generate an insert statement for the the to-be-inserted table
	InsertStatement insert;
	insert.table = stmt.info->table;
	insert.schema = stmt.info->schema;
	insert.catalog = stmt.info->catalog;
	insert.columns = stmt.info->select_list;

	// bind the insert statement to the base table
	auto insert_statement = Bind(insert);
	D_ASSERT(insert_statement.plan->type == LogicalOperatorType::LOGICAL_INSERT);

	auto &bound_insert = (LogicalInsert &)*insert_statement.plan;

	// lookup the format in the catalog
	auto &catalog = Catalog::GetSystemCatalog(context);
	auto copy_function = catalog.GetEntry<CopyFunctionCatalogEntry>(context, DEFAULT_SCHEMA, stmt.info->format);
	if (!copy_function->function.copy_from_bind) {
		throw NotImplementedException("COPY FROM is not supported for FORMAT \"%s\"", stmt.info->format);
	}
	// lookup the table to copy into
	BindSchemaOrCatalog(stmt.info->catalog, stmt.info->schema);
	auto table = Catalog::GetEntry<TableCatalogEntry>(context, stmt.info->catalog, stmt.info->schema, stmt.info->table);
	vector<string> expected_names;
	if (!bound_insert.column_index_map.empty()) {
		expected_names.resize(bound_insert.expected_types.size());
		for (auto &col : table->GetColumns().Logical()) {
			auto i = col.Physical();
			if (bound_insert.column_index_map[i] != DConstants::INVALID_INDEX) {
				expected_names[bound_insert.column_index_map[i]] = col.Name();
			}
		}
	} else {
		expected_names.reserve(bound_insert.expected_types.size());
		for (auto &col : table->GetColumns().Logical()) {
			expected_names.push_back(col.Name());
		}
	}

	auto function_data =
	    copy_function->function.copy_from_bind(context, *stmt.info, expected_names, bound_insert.expected_types);
	auto get = make_unique<LogicalGet>(GenerateTableIndex(), copy_function->function.copy_from_function,
	                                   std::move(function_data), bound_insert.expected_types, expected_names);
	for (idx_t i = 0; i < bound_insert.expected_types.size(); i++) {
		get->column_ids.push_back(i);
	}
	insert_statement.plan->children.push_back(std::move(get));
	result.plan = std::move(insert_statement.plan);
	return result;
}

BoundStatement Binder::Bind(CopyStatement &stmt) {
	if (!stmt.info->is_from && !stmt.select_statement) {
		// copy table into file without a query
		// generate SELECT * FROM table;
		auto ref = make_unique<BaseTableRef>();
		ref->catalog_name = stmt.info->catalog;
		ref->schema_name = stmt.info->schema;
		ref->table_name = stmt.info->table;

		auto statement = make_unique<SelectNode>();
		statement->from_table = std::move(ref);
		if (!stmt.info->select_list.empty()) {
			for (auto &name : stmt.info->select_list) {
				statement->select_list.push_back(make_unique<ColumnRefExpression>(name));
			}
		} else {
			statement->select_list.push_back(make_unique<StarExpression>());
		}
		stmt.select_statement = std::move(statement);
	}
	properties.allow_stream_result = false;
	properties.return_type = StatementReturnType::CHANGED_ROWS;
	if (stmt.info->is_from) {
		return BindCopyFrom(stmt);
	} else {
		return BindCopyTo(stmt);
	}
}

} // namespace duckdb










































namespace duckdb {

void Binder::BindSchemaOrCatalog(ClientContext &context, string &catalog, string &schema) {
	if (catalog.empty() && !schema.empty()) {
		// schema is specified - but catalog is not
		// try searching for the catalog instead
		auto &db_manager = DatabaseManager::Get(context);
		auto database = db_manager.GetDatabase(context, schema);
		if (database) {
			// we have a database with this name
			// check if there is a schema
			auto schema_obj = Catalog::GetSchema(context, INVALID_CATALOG, schema, true);
			if (schema_obj) {
				auto &attached = schema_obj->catalog->GetAttached();
				throw BinderException(
				    "Ambiguous reference to catalog or schema \"%s\" - use a fully qualified path like \"%s.%s\"",
				    schema, attached.GetName(), schema);
			}
			catalog = schema;
			schema = string();
		}
	}
}

void Binder::BindSchemaOrCatalog(string &catalog, string &schema) {
	BindSchemaOrCatalog(context, catalog, schema);
}

SchemaCatalogEntry *Binder::BindSchema(CreateInfo &info) {
	BindSchemaOrCatalog(info.catalog, info.schema);
	if (IsInvalidCatalog(info.catalog) && info.temporary) {
		info.catalog = TEMP_CATALOG;
	}
	auto &search_path = ClientData::Get(context).catalog_search_path;
	if (IsInvalidCatalog(info.catalog) && IsInvalidSchema(info.schema)) {
		auto &default_entry = search_path->GetDefault();
		info.catalog = default_entry.catalog;
		info.schema = default_entry.schema;
	} else if (IsInvalidSchema(info.schema)) {
		info.schema = search_path->GetDefaultSchema(info.catalog);
	} else if (IsInvalidCatalog(info.catalog)) {
		info.catalog = search_path->GetDefaultCatalog(info.schema);
	}
	if (IsInvalidCatalog(info.catalog)) {
		info.catalog = DatabaseManager::GetDefaultDatabase(context);
	}
	if (!info.temporary) {
		// non-temporary create: not read only
		if (info.catalog == TEMP_CATALOG) {
			throw ParserException("Only TEMPORARY table names can use the \"%s\" catalog", TEMP_CATALOG);
		}
	} else {
		if (info.catalog != TEMP_CATALOG) {
			throw ParserException("TEMPORARY table names can *only* use the \"%s\" catalog", TEMP_CATALOG);
		}
	}
	// fetch the schema in which we want to create the object
	auto schema_obj = Catalog::GetSchema(context, info.catalog, info.schema);
	D_ASSERT(schema_obj->type == CatalogType::SCHEMA_ENTRY);
	info.schema = schema_obj->name;
	if (!info.temporary) {
		properties.modified_databases.insert(schema_obj->catalog->GetName());
	}
	return schema_obj;
}

SchemaCatalogEntry *Binder::BindCreateSchema(CreateInfo &info) {
	auto schema = BindSchema(info);
	if (schema->catalog->IsSystemCatalog()) {
		throw BinderException("Cannot create entry in system catalog");
	}
	return schema;
}

void Binder::BindCreateViewInfo(CreateViewInfo &base) {
	// bind the view as if it were a query so we can catch errors
	// note that we bind the original, and replace the original with a copy
	auto view_binder = Binder::CreateBinder(context);
	view_binder->can_contain_nulls = true;

	auto copy = base.query->Copy();
	auto query_node = view_binder->Bind(*base.query);
	base.query = unique_ptr_cast<SQLStatement, SelectStatement>(std::move(copy));
	if (base.aliases.size() > query_node.names.size()) {
		throw BinderException("More VIEW aliases than columns in query result");
	}
	// fill up the aliases with the remaining names of the bound query
	base.aliases.reserve(query_node.names.size());
	for (idx_t i = base.aliases.size(); i < query_node.names.size(); i++) {
		base.aliases.push_back(query_node.names[i]);
	}
	base.types = query_node.types;
}

static void QualifyFunctionNames(ClientContext &context, unique_ptr<ParsedExpression> &expr) {
	switch (expr->GetExpressionClass()) {
	case ExpressionClass::FUNCTION: {
		auto &func = (FunctionExpression &)*expr;
		auto function = (StandardEntry *)Catalog::GetEntry(context, CatalogType::SCALAR_FUNCTION_ENTRY, func.catalog,
		                                                   func.schema, func.function_name, true);
		if (function) {
			func.catalog = function->catalog->GetName();
			func.schema = function->schema->name;
		}
		break;
	}
	case ExpressionClass::SUBQUERY: {
		// replacing parameters within a subquery is slightly different
		auto &sq = ((SubqueryExpression &)*expr).subquery;
		ParsedExpressionIterator::EnumerateQueryNodeChildren(
		    *sq->node, [&](unique_ptr<ParsedExpression> &child) { QualifyFunctionNames(context, child); });
		break;
	}
	default: // fall through
		break;
	}
	// unfold child expressions
	ParsedExpressionIterator::EnumerateChildren(
	    *expr, [&](unique_ptr<ParsedExpression> &child) { QualifyFunctionNames(context, child); });
}

SchemaCatalogEntry *Binder::BindCreateFunctionInfo(CreateInfo &info) {
	auto &base = (CreateMacroInfo &)info;
	auto &scalar_function = (ScalarMacroFunction &)*base.function;

	if (scalar_function.expression->HasParameter()) {
		throw BinderException("Parameter expressions within macro's are not supported!");
	}

	// create macro binding in order to bind the function
	vector<LogicalType> dummy_types;
	vector<string> dummy_names;
	// positional parameters
	for (idx_t i = 0; i < base.function->parameters.size(); i++) {
		auto param = (ColumnRefExpression &)*base.function->parameters[i];
		if (param.IsQualified()) {
			throw BinderException("Invalid parameter name '%s': must be unqualified", param.ToString());
		}
		dummy_types.emplace_back(LogicalType::SQLNULL);
		dummy_names.push_back(param.GetColumnName());
	}
	// default parameters
	for (auto it = base.function->default_parameters.begin(); it != base.function->default_parameters.end(); it++) {
		auto &val = (ConstantExpression &)*it->second;
		dummy_types.push_back(val.value.type());
		dummy_names.push_back(it->first);
	}
	auto this_macro_binding = make_unique<DummyBinding>(dummy_types, dummy_names, base.name);
	macro_binding = this_macro_binding.get();
	ExpressionBinder::QualifyColumnNames(*this, scalar_function.expression);
	QualifyFunctionNames(context, scalar_function.expression);

	// create a copy of the expression because we do not want to alter the original
	auto expression = scalar_function.expression->Copy();

	// bind it to verify the function was defined correctly
	string error;
	auto sel_node = make_unique<BoundSelectNode>();
	auto group_info = make_unique<BoundGroupInformation>();
	SelectBinder binder(*this, context, *sel_node, *group_info);
	error = binder.Bind(&expression, 0, false);

	if (!error.empty()) {
		throw BinderException(error);
	}

	return BindCreateSchema(info);
}

void Binder::BindLogicalType(ClientContext &context, LogicalType &type, Catalog *catalog, const string &schema) {
	if (type.id() == LogicalTypeId::LIST || type.id() == LogicalTypeId::MAP) {
		auto child_type = ListType::GetChildType(type);
		BindLogicalType(context, child_type, catalog, schema);
		auto alias = type.GetAlias();
		if (type.id() == LogicalTypeId::LIST) {
			type = LogicalType::LIST(child_type);
		} else {
			D_ASSERT(child_type.id() == LogicalTypeId::STRUCT); // map must be list of structs
			type = LogicalType::MAP(child_type);
		}

		type.SetAlias(alias);
	} else if (type.id() == LogicalTypeId::STRUCT) {
		auto child_types = StructType::GetChildTypes(type);
		for (auto &child_type : child_types) {
			BindLogicalType(context, child_type.second, catalog, schema);
		}
		// Generate new Struct Type
		auto alias = type.GetAlias();
		type = LogicalType::STRUCT(child_types);
		type.SetAlias(alias);
	} else if (type.id() == LogicalTypeId::UNION) {
		auto member_types = UnionType::CopyMemberTypes(type);
		for (auto &member_type : member_types) {
			BindLogicalType(context, member_type.second, catalog, schema);
		}
		// Generate new Union Type
		auto alias = type.GetAlias();
		type = LogicalType::UNION(member_types);
		type.SetAlias(alias);
	} else if (type.id() == LogicalTypeId::USER) {
		auto &user_type_name = UserType::GetTypeName(type);
		if (catalog) {
			type = catalog->GetType(context, schema, user_type_name, true);
			if (type.id() == LogicalTypeId::INVALID) {
				// look in the system catalog if the type was not found
				type = Catalog::GetType(context, SYSTEM_CATALOG, schema, user_type_name);
			}
		} else {
			type = Catalog::GetType(context, INVALID_CATALOG, schema, user_type_name);
		}
	} else if (type.id() == LogicalTypeId::ENUM) {
		auto &enum_type_name = EnumType::GetTypeName(type);
		TypeCatalogEntry *enum_type_catalog;
		if (catalog) {
			enum_type_catalog = catalog->GetEntry<TypeCatalogEntry>(context, schema, enum_type_name, true);
			if (!enum_type_catalog) {
				// look in the system catalog if the type was not found
				enum_type_catalog =
				    Catalog::GetEntry<TypeCatalogEntry>(context, SYSTEM_CATALOG, schema, enum_type_name, true);
			}
		} else {
			enum_type_catalog =
			    Catalog::GetEntry<TypeCatalogEntry>(context, INVALID_CATALOG, schema, enum_type_name, true);
		}

		LogicalType::SetCatalog(type, enum_type_catalog);
	}
}

static void FindMatchingPrimaryKeyColumns(const ColumnList &columns, const vector<unique_ptr<Constraint>> &constraints,
                                          ForeignKeyConstraint &fk) {
	// find the matching primary key constraint
	bool found_constraint = false;
	// if no columns are defined, we will automatically try to bind to the primary key
	bool find_primary_key = fk.pk_columns.empty();
	for (auto &constr : constraints) {
		if (constr->type != ConstraintType::UNIQUE) {
			continue;
		}
		auto &unique = (UniqueConstraint &)*constr;
		if (find_primary_key && !unique.is_primary_key) {
			continue;
		}
		found_constraint = true;

		vector<string> pk_names;
		if (unique.index.index != DConstants::INVALID_INDEX) {
			pk_names.push_back(columns.GetColumn(LogicalIndex(unique.index)).Name());
		} else {
			pk_names = unique.columns;
		}
		if (pk_names.size() != fk.fk_columns.size()) {
			// the number of referencing and referenced columns for foreign keys must be the same
			continue;
		}
		if (find_primary_key) {
			// found matching primary key
			fk.pk_columns = pk_names;
			return;
		}
		if (fk.pk_columns != pk_names) {
			// Name mismatch
			continue;
		}
		// found match
		return;
	}
	// no match found! examine why
	if (!found_constraint) {
		// no unique constraint or primary key
		string search_term = find_primary_key ? "primary key" : "primary key or unique constraint";
		throw BinderException("Failed to create foreign key: there is no %s for referenced table \"%s\"", search_term,
		                      fk.info.table);
	}
	// check if all the columns exist
	for (auto &name : fk.pk_columns) {
		bool found = columns.ColumnExists(name);
		if (!found) {
			throw BinderException(
			    "Failed to create foreign key: referenced table \"%s\" does not have a column named \"%s\"",
			    fk.info.table, name);
		}
	}
	auto fk_names = StringUtil::Join(fk.pk_columns, ",");
	throw BinderException("Failed to create foreign key: referenced table \"%s\" does not have a primary key or unique "
	                      "constraint on the columns %s",
	                      fk.info.table, fk_names);
}

static void FindForeignKeyIndexes(const ColumnList &columns, const vector<string> &names,
                                  vector<PhysicalIndex> &indexes) {
	D_ASSERT(indexes.empty());
	D_ASSERT(!names.empty());
	for (auto &name : names) {
		if (!columns.ColumnExists(name)) {
			throw BinderException("column \"%s\" named in key does not exist", name);
		}
		auto &column = columns.GetColumn(name);
		if (column.Generated()) {
			throw BinderException("Failed to create foreign key: referenced column \"%s\" is a generated column",
			                      column.Name());
		}
		indexes.push_back(column.Physical());
	}
}

static void CheckForeignKeyTypes(const ColumnList &pk_columns, const ColumnList &fk_columns, ForeignKeyConstraint &fk) {
	D_ASSERT(fk.info.pk_keys.size() == fk.info.fk_keys.size());
	for (idx_t c_idx = 0; c_idx < fk.info.pk_keys.size(); c_idx++) {
		auto &pk_col = pk_columns.GetColumn(fk.info.pk_keys[c_idx]);
		auto &fk_col = fk_columns.GetColumn(fk.info.fk_keys[c_idx]);
		if (pk_col.Type() != fk_col.Type()) {
			throw BinderException("Failed to create foreign key: incompatible types between column \"%s\" (\"%s\") and "
			                      "column \"%s\" (\"%s\")",
			                      pk_col.Name(), pk_col.Type().ToString(), fk_col.Name(), fk_col.Type().ToString());
		}
	}
}

void ExpressionContainsGeneratedColumn(const ParsedExpression &expr, const unordered_set<string> &gcols,
                                       bool &contains_gcol) {
	if (contains_gcol) {
		return;
	}
	if (expr.type == ExpressionType::COLUMN_REF) {
		auto &column_ref = (ColumnRefExpression &)expr;
		auto &name = column_ref.GetColumnName();
		if (gcols.count(name)) {
			contains_gcol = true;
			return;
		}
	}
	ParsedExpressionIterator::EnumerateChildren(
	    expr, [&](const ParsedExpression &child) { ExpressionContainsGeneratedColumn(child, gcols, contains_gcol); });
}

static bool AnyConstraintReferencesGeneratedColumn(CreateTableInfo &table_info) {
	unordered_set<string> generated_columns;
	for (auto &col : table_info.columns.Logical()) {
		if (!col.Generated()) {
			continue;
		}
		generated_columns.insert(col.Name());
	}
	if (generated_columns.empty()) {
		return false;
	}

	for (auto &constr : table_info.constraints) {
		switch (constr->type) {
		case ConstraintType::CHECK: {
			auto &constraint = (CheckConstraint &)*constr;
			auto &expr = constraint.expression;
			bool contains_generated_column = false;
			ExpressionContainsGeneratedColumn(*expr, generated_columns, contains_generated_column);
			if (contains_generated_column) {
				return true;
			}
			break;
		}
		case ConstraintType::NOT_NULL: {
			auto &constraint = (NotNullConstraint &)*constr;
			if (table_info.columns.GetColumn(constraint.index).Generated()) {
				return true;
			}
			break;
		}
		case ConstraintType::UNIQUE: {
			auto &constraint = (UniqueConstraint &)*constr;
			auto index = constraint.index;
			if (index.index == DConstants::INVALID_INDEX) {
				for (auto &col : constraint.columns) {
					if (generated_columns.count(col)) {
						return true;
					}
				}
			} else {
				if (table_info.columns.GetColumn(index).Generated()) {
					return true;
				}
			}
			break;
		}
		case ConstraintType::FOREIGN_KEY: {
			// If it contained a generated column, an exception would have been thrown inside AddDataTableIndex earlier
			break;
		}
		default: {
			throw NotImplementedException("ConstraintType not implemented");
		}
		}
	}
	return false;
}

unique_ptr<LogicalOperator> DuckCatalog::BindCreateIndex(Binder &binder, CreateStatement &stmt,
                                                         TableCatalogEntry &table, unique_ptr<LogicalOperator> plan) {
	D_ASSERT(plan->type == LogicalOperatorType::LOGICAL_GET);
	auto &base = (CreateIndexInfo &)*stmt.info;

	auto &get = (LogicalGet &)*plan;
	// bind the index expressions
	vector<unique_ptr<Expression>> expressions;
	IndexBinder index_binder(binder, binder.context);
	for (auto &expr : base.expressions) {
		expressions.push_back(index_binder.Bind(expr));
	}

	auto create_index_info = unique_ptr_cast<CreateInfo, CreateIndexInfo>(std::move(stmt.info));
	for (auto &column_id : get.column_ids) {
		if (column_id == COLUMN_IDENTIFIER_ROW_ID) {
			throw BinderException("Cannot create an index on the rowid!");
		}
		create_index_info->scan_types.push_back(get.returned_types[column_id]);
	}
	create_index_info->scan_types.emplace_back(LogicalType::ROW_TYPE);
	create_index_info->names = get.names;
	create_index_info->column_ids = get.column_ids;

	// the logical CREATE INDEX also needs all fields to scan the referenced table
	return make_unique<LogicalCreateIndex>(std::move(get.bind_data), std::move(create_index_info),
	                                       std::move(expressions), table, std::move(get.function));
}

BoundStatement Binder::Bind(CreateStatement &stmt) {
	BoundStatement result;
	result.names = {"Count"};
	result.types = {LogicalType::BIGINT};
	properties.return_type = StatementReturnType::NOTHING;

	auto catalog_type = stmt.info->type;
	switch (catalog_type) {
	case CatalogType::SCHEMA_ENTRY:
		result.plan = make_unique<LogicalCreate>(LogicalOperatorType::LOGICAL_CREATE_SCHEMA, std::move(stmt.info));
		break;
	case CatalogType::VIEW_ENTRY: {
		auto &base = (CreateViewInfo &)*stmt.info;
		// bind the schema
		auto schema = BindCreateSchema(*stmt.info);
		BindCreateViewInfo(base);
		result.plan =
		    make_unique<LogicalCreate>(LogicalOperatorType::LOGICAL_CREATE_VIEW, std::move(stmt.info), schema);
		break;
	}
	case CatalogType::SEQUENCE_ENTRY: {
		auto schema = BindCreateSchema(*stmt.info);
		result.plan =
		    make_unique<LogicalCreate>(LogicalOperatorType::LOGICAL_CREATE_SEQUENCE, std::move(stmt.info), schema);
		break;
	}
	case CatalogType::TABLE_MACRO_ENTRY: {
		auto schema = BindCreateSchema(*stmt.info);
		result.plan =
		    make_unique<LogicalCreate>(LogicalOperatorType::LOGICAL_CREATE_MACRO, std::move(stmt.info), schema);
		break;
	}
	case CatalogType::MACRO_ENTRY: {
		auto schema = BindCreateFunctionInfo(*stmt.info);
		result.plan =
		    make_unique<LogicalCreate>(LogicalOperatorType::LOGICAL_CREATE_MACRO, std::move(stmt.info), schema);
		break;
	}
	case CatalogType::INDEX_ENTRY: {
		auto &base = (CreateIndexInfo &)*stmt.info;

		// visit the table reference
		auto bound_table = Bind(*base.table);
		if (bound_table->type != TableReferenceType::BASE_TABLE) {
			throw BinderException("Can only create an index over a base table!");
		}
		auto &table_binding = (BoundBaseTableRef &)*bound_table;
		auto table = table_binding.table;
		if (table->temporary) {
			stmt.info->temporary = true;
		}
		// create a plan over the bound table
		auto plan = CreatePlan(*bound_table);
		if (plan->type != LogicalOperatorType::LOGICAL_GET) {
			throw BinderException("Cannot create index on a view!");
		}

		result.plan = table->catalog->BindCreateIndex(*this, stmt, *table, std::move(plan));
		break;
	}
	case CatalogType::TABLE_ENTRY: {
		auto &create_info = (CreateTableInfo &)*stmt.info;
		// If there is a foreign key constraint, resolve primary key column's index from primary key column's name
		unordered_set<SchemaCatalogEntry *> fk_schemas;
		for (idx_t i = 0; i < create_info.constraints.size(); i++) {
			auto &cond = create_info.constraints[i];
			if (cond->type != ConstraintType::FOREIGN_KEY) {
				continue;
			}
			auto &fk = (ForeignKeyConstraint &)*cond;
			if (fk.info.type != ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE) {
				continue;
			}
			D_ASSERT(fk.info.pk_keys.empty());
			D_ASSERT(fk.info.fk_keys.empty());
			FindForeignKeyIndexes(create_info.columns, fk.fk_columns, fk.info.fk_keys);
			if (create_info.table == fk.info.table) {
				// self-referential foreign key constraint
				fk.info.type = ForeignKeyType::FK_TYPE_SELF_REFERENCE_TABLE;
				FindMatchingPrimaryKeyColumns(create_info.columns, create_info.constraints, fk);
				FindForeignKeyIndexes(create_info.columns, fk.pk_columns, fk.info.pk_keys);
				CheckForeignKeyTypes(create_info.columns, create_info.columns, fk);
			} else {
				// have to resolve referenced table
				auto pk_table_entry_ptr =
				    Catalog::GetEntry<TableCatalogEntry>(context, INVALID_CATALOG, fk.info.schema, fk.info.table);
				fk_schemas.insert(pk_table_entry_ptr->schema);
				FindMatchingPrimaryKeyColumns(pk_table_entry_ptr->GetColumns(), pk_table_entry_ptr->GetConstraints(),
				                              fk);
				FindForeignKeyIndexes(pk_table_entry_ptr->GetColumns(), fk.pk_columns, fk.info.pk_keys);
				CheckForeignKeyTypes(pk_table_entry_ptr->GetColumns(), create_info.columns, fk);
				auto &storage = pk_table_entry_ptr->GetStorage();
				auto index = storage.info->indexes.FindForeignKeyIndex(fk.info.pk_keys,
				                                                       ForeignKeyType::FK_TYPE_PRIMARY_KEY_TABLE);
				if (!index) {
					auto fk_column_names = StringUtil::Join(fk.pk_columns, ",");
					throw BinderException("Failed to create foreign key on %s(%s): no UNIQUE or PRIMARY KEY constraint "
					                      "present on these columns",
					                      pk_table_entry_ptr->name, fk_column_names);
				}
			}
			D_ASSERT(fk.info.pk_keys.size() == fk.info.fk_keys.size());
			D_ASSERT(fk.info.pk_keys.size() == fk.pk_columns.size());
			D_ASSERT(fk.info.fk_keys.size() == fk.fk_columns.size());
		}
		if (AnyConstraintReferencesGeneratedColumn(create_info)) {
			throw BinderException("Constraints on generated columns are not supported yet");
		}
		auto bound_info = BindCreateTableInfo(std::move(stmt.info));
		auto root = std::move(bound_info->query);
		for (auto &fk_schema : fk_schemas) {
			if (fk_schema != bound_info->schema) {
				throw BinderException("Creating foreign keys across different schemas or catalogs is not supported");
			}
		}

		// create the logical operator
		auto &schema = bound_info->schema;
		auto create_table = make_unique<LogicalCreateTable>(schema, std::move(bound_info));
		if (root) {
			// CREATE TABLE AS
			properties.return_type = StatementReturnType::CHANGED_ROWS;
			create_table->children.push_back(std::move(root));
		}
		result.plan = std::move(create_table);
		break;
	}
	case CatalogType::TYPE_ENTRY: {
		auto schema = BindCreateSchema(*stmt.info);
		auto &create_type_info = (CreateTypeInfo &)(*stmt.info);
		result.plan =
		    make_unique<LogicalCreate>(LogicalOperatorType::LOGICAL_CREATE_TYPE, std::move(stmt.info), schema);
		if (create_type_info.query) {
			// CREATE TYPE mood AS ENUM (SELECT 'happy')
			auto &select_stmt = (SelectStatement &)*create_type_info.query;
			auto &query_node = *select_stmt.node;

			// We always add distinct modifier implicitly
			bool need_to_add = true;
			if (!query_node.modifiers.empty()) {
				if (query_node.modifiers[0]->type == ResultModifierType::DISTINCT_MODIFIER) {
					// There are cases where the same column is grouped repeatedly
					// CREATE TYPE mood AS ENUM (SELECT DISTINCT ON(x) x FROM test);
					// When we push into a constant expression
					// => CREATE TYPE mood AS ENUM (SELECT DISTINCT ON(x, x) x FROM test);
					auto &distinct_modifier = (DistinctModifier &)*query_node.modifiers[0];
					distinct_modifier.distinct_on_targets.push_back(make_unique<ConstantExpression>(Value::INTEGER(1)));
					need_to_add = false;
				}
			}

			// Add distinct modifier
			if (need_to_add) {
				auto distinct_modifier = make_unique<DistinctModifier>();
				distinct_modifier->distinct_on_targets.push_back(make_unique<ConstantExpression>(Value::INTEGER(1)));
				query_node.modifiers.emplace(query_node.modifiers.begin(), std::move(distinct_modifier));
			}

			auto query_obj = Bind(*create_type_info.query);
			auto query = std::move(query_obj.plan);

			auto &sql_types = query_obj.types;
			if (sql_types.size() != 1 || sql_types[0].id() != LogicalType::VARCHAR) {
				// add cast expression?
				throw BinderException("The query must return one varchar column");
			}

			result.plan->AddChild(std::move(query));
		} else if (create_type_info.type.id() == LogicalTypeId::USER) {
			// two cases:
			// 1: create a type with a non-existant type as source, catalog.GetType(...) will throw exception.
			// 2: create a type alias with a custom type.
			// eg. CREATE TYPE a AS INT; CREATE TYPE b AS a;
			// We set b to be an alias for the underlying type of a
			auto inner_type = Catalog::GetType(context, schema->catalog->GetName(), schema->name,
			                                   UserType::GetTypeName(create_type_info.type));
			// clear to nullptr, we don't need this
			LogicalType::SetCatalog(inner_type, nullptr);
			inner_type.SetAlias(create_type_info.name);
			create_type_info.type = inner_type;
		}
		break;
	}
	case CatalogType::DATABASE_ENTRY: {
		// not supported in DuckDB yet but allow extensions to intercept and implement this functionality
		auto &base = (CreateDatabaseInfo &)*stmt.info;
		string extension_name = base.extension_name;
		string database_name = base.name;
		string source_path = base.path;

		auto &config = DBConfig::GetConfig(context);
		for (auto &extension : config.create_database_extensions) {
			auto create_database_function_ref =
			    extension.function(context, extension_name, database_name, source_path, extension.data.get());
			if (create_database_function_ref) {
				auto bound_create_database_func = Bind(*create_database_function_ref);
				result.plan = CreatePlan(*bound_create_database_func);
				break;
			}
		}
		if (!result.plan) {
			throw NotImplementedException("CREATE DATABASE not supported in DuckDB yet");
		}
		break;
	}
	default:
		throw Exception("Unrecognized type!");
	}
	properties.allow_stream_result = false;
	return result;
}

} // namespace duckdb



















#include <algorithm>

namespace duckdb {

static void CreateColumnDependencyManager(BoundCreateTableInfo &info) {
	auto &base = (CreateTableInfo &)*info.base;
	for (auto &col : base.columns.Logical()) {
		if (!col.Generated()) {
			continue;
		}
		info.column_dependency_manager.AddGeneratedColumn(col, base.columns);
	}
}

static void BindCheckConstraint(Binder &binder, BoundCreateTableInfo &info, const unique_ptr<Constraint> &cond) {
	auto &base = (CreateTableInfo &)*info.base;

	auto bound_constraint = make_unique<BoundCheckConstraint>();
	// check constraint: bind the expression
	CheckBinder check_binder(binder, binder.context, base.table, base.columns, bound_constraint->bound_columns);
	auto &check = (CheckConstraint &)*cond;
	// create a copy of the unbound expression because the binding destroys the constraint
	auto unbound_expression = check.expression->Copy();
	// now bind the constraint and create a new BoundCheckConstraint
	bound_constraint->expression = check_binder.Bind(check.expression);
	info.bound_constraints.push_back(std::move(bound_constraint));
	// move the unbound constraint back into the original check expression
	check.expression = std::move(unbound_expression);
}

static void BindConstraints(Binder &binder, BoundCreateTableInfo &info) {
	auto &base = (CreateTableInfo &)*info.base;

	bool has_primary_key = false;
	logical_index_set_t not_null_columns;
	vector<LogicalIndex> primary_keys;
	for (idx_t i = 0; i < base.constraints.size(); i++) {
		auto &cond = base.constraints[i];
		switch (cond->type) {
		case ConstraintType::CHECK: {
			BindCheckConstraint(binder, info, cond);
			break;
		}
		case ConstraintType::NOT_NULL: {
			auto &not_null = (NotNullConstraint &)*cond;
			auto &col = base.columns.GetColumn(LogicalIndex(not_null.index));
			info.bound_constraints.push_back(make_unique<BoundNotNullConstraint>(PhysicalIndex(col.StorageOid())));
			not_null_columns.insert(not_null.index);
			break;
		}
		case ConstraintType::UNIQUE: {
			auto &unique = (UniqueConstraint &)*cond;
			// have to resolve columns of the unique constraint
			vector<LogicalIndex> keys;
			logical_index_set_t key_set;
			if (unique.index.index != DConstants::INVALID_INDEX) {
				D_ASSERT(unique.index.index < base.columns.LogicalColumnCount());
				// unique constraint is given by single index
				unique.columns.push_back(base.columns.GetColumn(unique.index).Name());
				keys.push_back(unique.index);
				key_set.insert(unique.index);
			} else {
				// unique constraint is given by list of names
				// have to resolve names
				D_ASSERT(!unique.columns.empty());
				for (auto &keyname : unique.columns) {
					if (!base.columns.ColumnExists(keyname)) {
						throw ParserException("column \"%s\" named in key does not exist", keyname);
					}
					auto &column = base.columns.GetColumn(keyname);
					auto column_index = column.Logical();
					if (key_set.find(column_index) != key_set.end()) {
						throw ParserException("column \"%s\" appears twice in "
						                      "primary key constraint",
						                      keyname);
					}
					keys.push_back(column_index);
					key_set.insert(column_index);
				}
			}

			if (unique.is_primary_key) {
				// we can only have one primary key per table
				if (has_primary_key) {
					throw ParserException("table \"%s\" has more than one primary key", base.table);
				}
				has_primary_key = true;
				primary_keys = keys;
			}
			info.bound_constraints.push_back(
			    make_unique<BoundUniqueConstraint>(std::move(keys), std::move(key_set), unique.is_primary_key));
			break;
		}
		case ConstraintType::FOREIGN_KEY: {
			auto &fk = (ForeignKeyConstraint &)*cond;
			D_ASSERT((fk.info.type == ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE && !fk.info.pk_keys.empty()) ||
			         (fk.info.type == ForeignKeyType::FK_TYPE_PRIMARY_KEY_TABLE && !fk.info.pk_keys.empty()) ||
			         fk.info.type == ForeignKeyType::FK_TYPE_SELF_REFERENCE_TABLE);
			physical_index_set_t fk_key_set, pk_key_set;
			for (idx_t i = 0; i < fk.info.pk_keys.size(); i++) {
				pk_key_set.insert(fk.info.pk_keys[i]);
			}
			for (idx_t i = 0; i < fk.info.fk_keys.size(); i++) {
				fk_key_set.insert(fk.info.fk_keys[i]);
			}
			info.bound_constraints.push_back(
			    make_unique<BoundForeignKeyConstraint>(fk.info, std::move(pk_key_set), std::move(fk_key_set)));
			break;
		}
		default:
			throw NotImplementedException("unrecognized constraint type in bind");
		}
	}
	if (has_primary_key) {
		// if there is a primary key index, also create a NOT NULL constraint for each of the columns
		for (auto &column_index : primary_keys) {
			if (not_null_columns.count(column_index)) {
				//! No need to create a NotNullConstraint, it's already present
				continue;
			}
			auto physical_index = base.columns.LogicalToPhysical(column_index);
			base.constraints.push_back(make_unique<NotNullConstraint>(column_index));
			info.bound_constraints.push_back(make_unique<BoundNotNullConstraint>(physical_index));
		}
	}
}

void Binder::BindGeneratedColumns(BoundCreateTableInfo &info) {
	auto &base = (CreateTableInfo &)*info.base;

	vector<string> names;
	vector<LogicalType> types;

	D_ASSERT(base.type == CatalogType::TABLE_ENTRY);
	for (auto &col : base.columns.Logical()) {
		names.push_back(col.Name());
		types.push_back(col.Type());
	}
	auto table_index = GenerateTableIndex();

	// Create a new binder because we dont need (or want) these bindings in this scope
	auto binder = Binder::CreateBinder(context);
	binder->bind_context.AddGenericBinding(table_index, base.table, names, types);
	auto expr_binder = ExpressionBinder(*binder, context);
	string ignore;
	auto table_binding = binder->bind_context.GetBinding(base.table, ignore);
	D_ASSERT(table_binding && ignore.empty());

	auto bind_order = info.column_dependency_manager.GetBindOrder(base.columns);
	logical_index_set_t bound_indices;

	while (!bind_order.empty()) {
		auto i = bind_order.top();
		bind_order.pop();
		auto &col = base.columns.GetColumnMutable(i);

		//! Already bound this previously
		//! This can not be optimized out of the GetBindOrder function
		//! These occurrences happen because we need to make sure that ALL dependencies of a column are resolved before
		//! it gets resolved
		if (bound_indices.count(i)) {
			continue;
		}
		D_ASSERT(col.Generated());
		auto expression = col.GeneratedExpression().Copy();

		auto bound_expression = expr_binder.Bind(expression);
		D_ASSERT(bound_expression);
		D_ASSERT(!bound_expression->HasSubquery());
		if (col.Type().id() == LogicalTypeId::ANY) {
			// Do this before changing the type, so we know it's the first time the type is set
			col.ChangeGeneratedExpressionType(bound_expression->return_type);
			col.SetType(bound_expression->return_type);

			// Update the type in the binding, for future expansions
			string ignore;
			table_binding->types[i.index] = col.Type();
		}
		bound_indices.insert(i);
	}
}

void Binder::BindDefaultValues(const ColumnList &columns, vector<unique_ptr<Expression>> &bound_defaults) {
	for (auto &column : columns.Physical()) {
		unique_ptr<Expression> bound_default;
		if (column.DefaultValue()) {
			// we bind a copy of the DEFAULT value because binding is destructive
			// and we want to keep the original expression around for serialization
			auto default_copy = column.DefaultValue()->Copy();
			ConstantBinder default_binder(*this, context, "DEFAULT value");
			default_binder.target_type = column.Type();
			bound_default = default_binder.Bind(default_copy);
		} else {
			// no default value specified: push a default value of constant null
			bound_default = make_unique<BoundConstantExpression>(Value(column.Type()));
		}
		bound_defaults.push_back(std::move(bound_default));
	}
}

static void ExtractExpressionDependencies(Expression &expr, DependencyList &dependencies) {
	if (expr.type == ExpressionType::BOUND_FUNCTION) {
		auto &function = (BoundFunctionExpression &)expr;
		if (function.function.dependency) {
			function.function.dependency(function, dependencies);
		}
	}
	ExpressionIterator::EnumerateChildren(
	    expr, [&](Expression &child) { ExtractExpressionDependencies(child, dependencies); });
}

static void ExtractDependencies(BoundCreateTableInfo &info) {
	for (auto &default_value : info.bound_defaults) {
		if (default_value) {
			ExtractExpressionDependencies(*default_value, info.dependencies);
		}
	}
	for (auto &constraint : info.bound_constraints) {
		if (constraint->type == ConstraintType::CHECK) {
			auto &bound_check = (BoundCheckConstraint &)*constraint;
			ExtractExpressionDependencies(*bound_check.expression, info.dependencies);
		}
	}
}
unique_ptr<BoundCreateTableInfo> Binder::BindCreateTableInfo(unique_ptr<CreateInfo> info, SchemaCatalogEntry *schema) {
	auto &base = (CreateTableInfo &)*info;
	auto result = make_unique<BoundCreateTableInfo>(std::move(info));
	result->schema = schema;
	if (base.query) {
		// construct the result object
		auto query_obj = Bind(*base.query);
		result->query = std::move(query_obj.plan);

		// construct the set of columns based on the names and types of the query
		auto &names = query_obj.names;
		auto &sql_types = query_obj.types;
		D_ASSERT(names.size() == sql_types.size());
		base.columns.SetAllowDuplicates(true);
		for (idx_t i = 0; i < names.size(); i++) {
			base.columns.AddColumn(ColumnDefinition(names[i], sql_types[i]));
		}
		CreateColumnDependencyManager(*result);
		// bind the generated column expressions
		BindGeneratedColumns(*result);
	} else {
		CreateColumnDependencyManager(*result);
		// bind the generated column expressions
		BindGeneratedColumns(*result);
		// bind any constraints
		BindConstraints(*this, *result);
		// bind the default values
		BindDefaultValues(base.columns, result->bound_defaults);
	}
	// extract dependencies from any default values or CHECK constraints
	ExtractDependencies(*result);

	if (base.columns.PhysicalColumnCount() == 0) {
		throw BinderException("Creating a table without physical (non-generated) columns is not supported");
	}
	// bind collations to detect any unsupported collation errors
	for (idx_t i = 0; i < base.columns.PhysicalColumnCount(); i++) {
		auto &column = base.columns.GetColumnMutable(PhysicalIndex(i));
		if (column.Type().id() == LogicalTypeId::VARCHAR) {
			ExpressionBinder::TestCollation(context, StringType::GetCollation(column.Type()));
		}
		BindLogicalType(context, column.TypeMutable(), result->schema->catalog);
		// We add a catalog dependency
		auto type_dependency = LogicalType::GetCatalog(column.Type());
		if (type_dependency) {
			// Only if the USER comes from a create type
			result->dependencies.AddDependency(type_dependency);
		}
	}
	properties.allow_stream_result = false;
	return result;
}

unique_ptr<BoundCreateTableInfo> Binder::BindCreateTableInfo(unique_ptr<CreateInfo> info) {
	auto &base = (CreateTableInfo &)*info;
	auto schema = BindCreateSchema(base);
	return BindCreateTableInfo(std::move(info), schema);
}

} // namespace duckdb












namespace duckdb {

BoundStatement Binder::Bind(DeleteStatement &stmt) {
	BoundStatement result;

	// visit the table reference
	auto bound_table = Bind(*stmt.table);
	if (bound_table->type != TableReferenceType::BASE_TABLE) {
		throw BinderException("Can only delete from base table!");
	}
	auto &table_binding = (BoundBaseTableRef &)*bound_table;
	auto table = table_binding.table;

	auto root = CreatePlan(*bound_table);
	auto &get = (LogicalGet &)*root;
	D_ASSERT(root->type == LogicalOperatorType::LOGICAL_GET);

	if (!table->temporary) {
		// delete from persistent table: not read only!
		properties.modified_databases.insert(table->catalog->GetName());
	}

	// Add CTEs as bindable
	AddCTEMap(stmt.cte_map);

	// plan any tables from the various using clauses
	if (!stmt.using_clauses.empty()) {
		unique_ptr<LogicalOperator> child_operator;
		for (auto &using_clause : stmt.using_clauses) {
			// bind the using clause
			auto using_binder = Binder::CreateBinder(context, this);
			auto bound_node = using_binder->Bind(*using_clause);
			auto op = CreatePlan(*bound_node);
			if (child_operator) {
				// already bound a child: create a cross product to unify the two
				child_operator = LogicalCrossProduct::Create(std::move(child_operator), std::move(op));
			} else {
				child_operator = std::move(op);
			}
			bind_context.AddContext(std::move(using_binder->bind_context));
		}
		if (child_operator) {
			root = LogicalCrossProduct::Create(std::move(root), std::move(child_operator));
		}
	}

	// project any additional columns required for the condition
	unique_ptr<Expression> condition;
	if (stmt.condition) {
		WhereBinder binder(*this, context);
		condition = binder.Bind(stmt.condition);

		PlanSubqueries(&condition, &root);
		auto filter = make_unique<LogicalFilter>(std::move(condition));
		filter->AddChild(std::move(root));
		root = std::move(filter);
	}
	// create the delete node
	auto del = make_unique<LogicalDelete>(table, GenerateTableIndex());
	del->AddChild(std::move(root));

	// set up the delete expression
	del->expressions.push_back(make_unique<BoundColumnRefExpression>(
	    LogicalType::ROW_TYPE, ColumnBinding(get.table_index, get.column_ids.size())));
	get.column_ids.push_back(COLUMN_IDENTIFIER_ROW_ID);

	if (!stmt.returning_list.empty()) {
		del->return_chunk = true;

		auto update_table_index = GenerateTableIndex();
		del->table_index = update_table_index;

		unique_ptr<LogicalOperator> del_as_logicaloperator = std::move(del);
		return BindReturning(std::move(stmt.returning_list), table, update_table_index,
		                     std::move(del_as_logicaloperator), std::move(result));
	}
	result.plan = std::move(del);
	result.names = {"Count"};
	result.types = {LogicalType::BIGINT};
	properties.allow_stream_result = false;
	properties.return_type = StatementReturnType::CHANGED_ROWS;

	return result;
}

} // namespace duckdb







namespace duckdb {

BoundStatement Binder::Bind(DropStatement &stmt) {
	BoundStatement result;

	switch (stmt.info->type) {
	case CatalogType::PREPARED_STATEMENT:
		// dropping prepared statements is always possible
		// it also does not require a valid transaction
		properties.requires_valid_transaction = false;
		break;
	case CatalogType::SCHEMA_ENTRY: {
		// dropping a schema is never read-only because there are no temporary schemas
		auto &catalog = Catalog::GetCatalog(context, stmt.info->catalog);
		properties.modified_databases.insert(catalog.GetName());
		break;
	}
	case CatalogType::VIEW_ENTRY:
	case CatalogType::SEQUENCE_ENTRY:
	case CatalogType::MACRO_ENTRY:
	case CatalogType::TABLE_MACRO_ENTRY:
	case CatalogType::INDEX_ENTRY:
	case CatalogType::TABLE_ENTRY:
	case CatalogType::TYPE_ENTRY: {
		BindSchemaOrCatalog(stmt.info->catalog, stmt.info->schema);
		auto entry = (StandardEntry *)Catalog::GetEntry(context, stmt.info->type, stmt.info->catalog, stmt.info->schema,
		                                                stmt.info->name, true);
		if (!entry) {
			break;
		}
		stmt.info->catalog = entry->catalog->GetName();
		if (!entry->temporary) {
			// we can only drop temporary tables in read-only mode
			properties.modified_databases.insert(stmt.info->catalog);
		}
		stmt.info->schema = entry->schema->name;
		break;
	}
	case CatalogType::DATABASE_ENTRY:
		// attaching and detaching is read-only
		stmt.info->catalog = SYSTEM_CATALOG;
		break;
	default:
		throw BinderException("Unknown catalog type for drop statement!");
	}
	result.plan = make_unique<LogicalSimple>(LogicalOperatorType::LOGICAL_DROP, std::move(stmt.info));
	result.names = {"Success"};
	result.types = {LogicalType::BOOLEAN};
	properties.allow_stream_result = false;
	properties.return_type = StatementReturnType::NOTHING;
	return result;
}

} // namespace duckdb









namespace duckdb {

BoundStatement Binder::Bind(ExecuteStatement &stmt) {
	auto parameter_count = stmt.n_param;

	// bind the prepared statement
	auto &client_data = ClientData::Get(context);

	auto entry = client_data.prepared_statements.find(stmt.name);
	if (entry == client_data.prepared_statements.end()) {
		throw BinderException("Prepared statement \"%s\" does not exist", stmt.name);
	}

	// check if we need to rebind the prepared statement
	// this happens if the catalog changes, since in this case e.g. tables we relied on may have been deleted
	auto prepared = entry->second;

	// bind any supplied parameters
	vector<Value> bind_values;
	auto constant_binder = Binder::CreateBinder(context);
	constant_binder->SetCanContainNulls(true);
	for (idx_t i = 0; i < stmt.values.size(); i++) {
		ConstantBinder cbinder(*constant_binder, context, "EXECUTE statement");
		auto bound_expr = cbinder.Bind(stmt.values[i]);

		Value value = ExpressionExecutor::EvaluateScalar(context, *bound_expr, true);
		bind_values.push_back(std::move(value));
	}
	unique_ptr<LogicalOperator> rebound_plan;
	if (prepared->RequireRebind(context, bind_values)) {
		// catalog was modified or statement does not have clear types: rebind the statement before running the execute
		Planner prepared_planner(context);
		for (idx_t i = 0; i < bind_values.size(); i++) {
			prepared_planner.parameter_data.emplace_back(bind_values[i]);
		}
		prepared = prepared_planner.PrepareSQLStatement(entry->second->unbound_statement->Copy());
		rebound_plan = std::move(prepared_planner.plan);
		D_ASSERT(prepared->properties.bound_all_parameters);
		this->bound_tables = prepared_planner.binder->bound_tables;
	}
	// copy the properties of the prepared statement into the planner
	this->properties = prepared->properties;
	this->properties.parameter_count = parameter_count;
	BoundStatement result;
	result.names = prepared->names;
	result.types = prepared->types;

	prepared->Bind(std::move(bind_values));
	if (rebound_plan) {
		auto execute_plan = make_unique<LogicalExecute>(std::move(prepared));
		execute_plan->children.push_back(std::move(rebound_plan));
		result.plan = std::move(execute_plan);
	} else {
		result.plan = make_unique<LogicalExecute>(std::move(prepared));
	}
	return result;
}

} // namespace duckdb




namespace duckdb {

BoundStatement Binder::Bind(ExplainStatement &stmt) {
	BoundStatement result;

	// bind the underlying statement
	auto plan = Bind(*stmt.stmt);
	// get the unoptimized logical plan, and create the explain statement
	auto logical_plan_unopt = plan.plan->ToString();
	auto explain = make_unique<LogicalExplain>(std::move(plan.plan), stmt.explain_type);
	explain->logical_plan_unopt = logical_plan_unopt;

	result.plan = std::move(explain);
	result.names = {"explain_key", "explain_value"};
	result.types = {LogicalType::VARCHAR, LogicalType::VARCHAR};
	properties.return_type = StatementReturnType::QUERY_RESULT;
	return result;
}

} // namespace duckdb















#include <algorithm>

namespace duckdb {

//! Sanitizes a string to have only low case chars and underscores
string SanitizeExportIdentifier(const string &str) {
	// Copy the original string to result
	string result(str);

	for (idx_t i = 0; i < str.length(); ++i) {
		auto c = str[i];
		if (c >= 'a' && c <= 'z') {
			// If it is lower case just continue
			continue;
		}

		if (c >= 'A' && c <= 'Z') {
			// To lowercase
			result[i] = tolower(c);
		} else {
			// Substitute to underscore
			result[i] = '_';
		}
	}

	return result;
}

bool IsExistMainKeyTable(string &table_name, vector<TableCatalogEntry *> &unordered) {
	for (idx_t i = 0; i < unordered.size(); i++) {
		if (unordered[i]->name == table_name) {
			return true;
		}
	}
	return false;
}

void ScanForeignKeyTable(vector<TableCatalogEntry *> &ordered, vector<TableCatalogEntry *> &unordered,
                         bool move_only_pk_table) {
	for (auto i = unordered.begin(); i != unordered.end();) {
		auto table_entry = *i;
		bool move_to_ordered = true;
		auto &constraints = table_entry->GetConstraints();
		for (idx_t j = 0; j < constraints.size(); j++) {
			auto &cond = constraints[j];
			if (cond->type == ConstraintType::FOREIGN_KEY) {
				auto &fk = (ForeignKeyConstraint &)*cond;
				if ((move_only_pk_table && fk.info.type == ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE) ||
				    (!move_only_pk_table && fk.info.type == ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE &&
				     IsExistMainKeyTable(fk.info.table, unordered))) {
					move_to_ordered = false;
					break;
				}
			}
		}
		if (move_to_ordered) {
			ordered.push_back(table_entry);
			i = unordered.erase(i);
		} else {
			i++;
		}
	}
}

void ReorderTableEntries(vector<TableCatalogEntry *> &tables) {
	vector<TableCatalogEntry *> ordered;
	vector<TableCatalogEntry *> unordered = tables;
	ScanForeignKeyTable(ordered, unordered, true);
	while (!unordered.empty()) {
		ScanForeignKeyTable(ordered, unordered, false);
	}
	tables = ordered;
}

string CreateFileName(const string &id_suffix, TableCatalogEntry *table, const string &extension) {
	auto name = SanitizeExportIdentifier(table->name);
	if (table->schema->name == DEFAULT_SCHEMA) {
		return StringUtil::Format("%s%s.%s", name, id_suffix, extension);
	}
	auto schema = SanitizeExportIdentifier(table->schema->name);
	return StringUtil::Format("%s_%s%s.%s", schema, name, id_suffix, extension);
}

BoundStatement Binder::Bind(ExportStatement &stmt) {
	// COPY TO a file
	auto &config = DBConfig::GetConfig(context);
	if (!config.options.enable_external_access) {
		throw PermissionException("COPY TO is disabled through configuration");
	}
	BoundStatement result;
	result.types = {LogicalType::BOOLEAN};
	result.names = {"Success"};

	// lookup the format in the catalog
	auto copy_function =
	    Catalog::GetEntry<CopyFunctionCatalogEntry>(context, INVALID_CATALOG, DEFAULT_SCHEMA, stmt.info->format);
	if (!copy_function->function.copy_to_bind && !copy_function->function.plan) {
		throw NotImplementedException("COPY TO is not supported for FORMAT \"%s\"", stmt.info->format);
	}

	// gather a list of all the tables
	string catalog = stmt.database.empty() ? INVALID_CATALOG : stmt.database;
	vector<TableCatalogEntry *> tables;
	auto schemas = Catalog::GetSchemas(context, catalog);
	for (auto &schema : schemas) {
		schema->Scan(context, CatalogType::TABLE_ENTRY, [&](CatalogEntry *entry) {
			if (entry->type == CatalogType::TABLE_ENTRY) {
				tables.push_back((TableCatalogEntry *)entry);
			}
		});
	}

	// reorder tables because of foreign key constraint
	ReorderTableEntries(tables);

	// now generate the COPY statements for each of the tables
	auto &fs = FileSystem::GetFileSystem(context);
	unique_ptr<LogicalOperator> child_operator;

	BoundExportData exported_tables;

	unordered_set<string> table_name_index;
	for (auto &table : tables) {
		auto info = make_unique<CopyInfo>();
		// we copy the options supplied to the EXPORT
		info->format = stmt.info->format;
		info->options = stmt.info->options;
		// set up the file name for the COPY TO

		auto exported_data = ExportedTableData();
		idx_t id = 0;
		while (true) {
			string id_suffix = id == 0 ? string() : "_" + to_string(id);
			auto name = CreateFileName(id_suffix, table, copy_function->function.extension);
			auto directory = stmt.info->file_path;
			auto full_path = fs.JoinPath(directory, name);
			info->file_path = full_path;
			auto insert_result = table_name_index.insert(info->file_path);
			if (insert_result.second == true) {
				// this name was not yet taken: take it
				break;
			}
			id++;
		}
		info->is_from = false;
		info->catalog = catalog;
		info->schema = table->schema->name;
		info->table = table->name;

		// We can not export generated columns
		for (auto &col : table->GetColumns().Physical()) {
			info->select_list.push_back(col.GetName());
		}

		exported_data.database_name = catalog;
		exported_data.table_name = info->table;
		exported_data.schema_name = info->schema;

		exported_data.file_path = info->file_path;

		ExportedTableInfo table_info;
		table_info.entry = table;
		table_info.table_data = exported_data;
		exported_tables.data.push_back(table_info);
		id++;

		// generate the copy statement and bind it
		CopyStatement copy_stmt;
		copy_stmt.info = std::move(info);

		auto copy_binder = Binder::CreateBinder(context, this);
		auto bound_statement = copy_binder->Bind(copy_stmt);
		if (child_operator) {
			// use UNION ALL to combine the individual copy statements into a single node
			auto copy_union =
			    make_unique<LogicalSetOperation>(GenerateTableIndex(), 1, std::move(child_operator),
			                                     std::move(bound_statement.plan), LogicalOperatorType::LOGICAL_UNION);
			child_operator = std::move(copy_union);
		} else {
			child_operator = std::move(bound_statement.plan);
		}
	}

	// try to create the directory, if it doesn't exist yet
	// a bit hacky to do it here, but we need to create the directory BEFORE the copy statements run
	if (!fs.DirectoryExists(stmt.info->file_path)) {
		fs.CreateDirectory(stmt.info->file_path);
	}

	// create the export node
	auto export_node = make_unique<LogicalExport>(copy_function->function, std::move(stmt.info), exported_tables);

	if (child_operator) {
		export_node->children.push_back(std::move(child_operator));
	}

	result.plan = std::move(export_node);
	properties.allow_stream_result = false;
	properties.return_type = StatementReturnType::NOTHING;
	return result;
}

} // namespace duckdb




namespace duckdb {

BoundStatement Binder::Bind(ExtensionStatement &stmt) {
	BoundStatement result;

	// perform the planning of the function
	D_ASSERT(stmt.extension.plan_function);
	auto parse_result =
	    stmt.extension.plan_function(stmt.extension.parser_info.get(), context, std::move(stmt.parse_data));

	properties.modified_databases = parse_result.modified_databases;
	properties.requires_valid_transaction = parse_result.requires_valid_transaction;
	properties.return_type = parse_result.return_type;

	// create the plan as a scan of the given table function
	result.plan = BindTableFunction(parse_result.function, std::move(parse_result.parameters));
	D_ASSERT(result.plan->type == LogicalOperatorType::LOGICAL_GET);
	auto &get = (LogicalGet &)*result.plan;
	result.names = get.names;
	result.types = get.returned_types;
	get.column_ids.clear();
	for (idx_t i = 0; i < get.returned_types.size(); i++) {
		get.column_ids.push_back(i);
	}
	return result;
}

} // namespace duckdb



























namespace duckdb {

static void CheckInsertColumnCountMismatch(int64_t expected_columns, int64_t result_columns, bool columns_provided,
                                           const char *tname) {
	if (result_columns != expected_columns) {
		string msg = StringUtil::Format(!columns_provided ? "table %s has %lld columns but %lld values were supplied"
		                                                  : "Column name/value mismatch for insert on %s: "
		                                                    "expected %lld columns but %lld values were supplied",
		                                tname, expected_columns, result_columns);
		throw BinderException(msg);
	}
}

unique_ptr<ParsedExpression> ExpandDefaultExpression(const ColumnDefinition &column) {
	if (column.DefaultValue()) {
		return column.DefaultValue()->Copy();
	} else {
		return make_unique<ConstantExpression>(Value(column.Type()));
	}
}

void ReplaceDefaultExpression(unique_ptr<ParsedExpression> &expr, const ColumnDefinition &column) {
	D_ASSERT(expr->type == ExpressionType::VALUE_DEFAULT);
	expr = ExpandDefaultExpression(column);
}

void QualifyColumnReferences(unique_ptr<ParsedExpression> &expr, const string &table_name) {
	// To avoid ambiguity with 'excluded', we explicitly qualify all column references
	if (expr->type == ExpressionType::COLUMN_REF) {
		auto &column_ref = (ColumnRefExpression &)*expr;
		if (column_ref.IsQualified()) {
			return;
		}
		auto column_name = column_ref.GetColumnName();
		expr = make_unique<ColumnRefExpression>(column_name, table_name);
	}
	ParsedExpressionIterator::EnumerateChildren(
	    *expr, [&](unique_ptr<ParsedExpression> &child) { QualifyColumnReferences(child, table_name); });
}

// Replace binding.table_index with 'dest' if it's 'source'
void ReplaceColumnBindings(Expression &expr, idx_t source, idx_t dest) {
	if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
		auto &bound_columnref = (BoundColumnRefExpression &)expr;
		if (bound_columnref.binding.table_index == source) {
			bound_columnref.binding.table_index = dest;
		}
	}
	ExpressionIterator::EnumerateChildren(
	    expr, [&](unique_ptr<Expression> &child) { ReplaceColumnBindings(*child, source, dest); });
}

void Binder::BindDoUpdateSetExpressions(const string &table_alias, LogicalInsert *insert, UpdateSetInfo &set_info,
                                        TableCatalogEntry &table) {
	D_ASSERT(insert->children.size() == 1);
	D_ASSERT(insert->children[0]->type == LogicalOperatorType::LOGICAL_PROJECTION);

	vector<column_t> logical_column_ids;
	vector<string> column_names;
	D_ASSERT(set_info.columns.size() == set_info.expressions.size());

	for (idx_t i = 0; i < set_info.columns.size(); i++) {
		auto &colname = set_info.columns[i];
		auto &expr = set_info.expressions[i];
		if (!table.ColumnExists(colname)) {
			throw BinderException("Referenced update column %s not found in table!", colname);
		}
		auto &column = table.GetColumn(colname);
		if (column.Generated()) {
			throw BinderException("Cant update column \"%s\" because it is a generated column!", column.Name());
		}
		if (std::find(insert->set_columns.begin(), insert->set_columns.end(), column.Physical()) !=
		    insert->set_columns.end()) {
			throw BinderException("Multiple assignments to same column \"%s\"", colname);
		}
		insert->set_columns.push_back(column.Physical());
		logical_column_ids.push_back(column.Oid());
		insert->set_types.push_back(column.Type());
		column_names.push_back(colname);
		if (expr->type == ExpressionType::VALUE_DEFAULT) {
			expr = ExpandDefaultExpression(column);
		}
		UpdateBinder binder(*this, context);
		binder.target_type = column.Type();

		// Avoid ambiguity issues
		QualifyColumnReferences(expr, table_alias);

		auto bound_expr = binder.Bind(expr);
		D_ASSERT(bound_expr);
		if (bound_expr->expression_class == ExpressionClass::BOUND_SUBQUERY) {
			throw BinderException("Expression in the DO UPDATE SET clause can not be a subquery");
		}

		insert->expressions.push_back(std::move(bound_expr));
	}

	// Figure out which columns are indexed on
	unordered_set<column_t> indexed_columns;
	auto &indexes = table.GetStorage().info->indexes.Indexes();
	for (auto &index : indexes) {
		for (auto &column_id : index->column_id_set) {
			indexed_columns.insert(column_id);
		}
	}

	// Verify that none of the columns that are targeted with a SET expression are indexed on
	for (idx_t i = 0; i < logical_column_ids.size(); i++) {
		auto &column = logical_column_ids[i];
		if (indexed_columns.count(column)) {
			throw BinderException("Can not assign to column '%s' because it has a UNIQUE/PRIMARY KEY constraint",
			                      column_names[i]);
		}
	}
}

unique_ptr<UpdateSetInfo> CreateSetInfoForReplace(TableCatalogEntry &table, InsertStatement &insert) {
	auto set_info = make_unique<UpdateSetInfo>();

	auto &columns = set_info->columns;
	// Figure out which columns are indexed on

	unordered_set<column_t> indexed_columns;
	auto &indexes = table.GetStorage().info->indexes.Indexes();
	for (auto &index : indexes) {
		for (auto &column_id : index->column_id_set) {
			indexed_columns.insert(column_id);
		}
	}

	auto &column_list = table.GetColumns();
	if (insert.columns.empty()) {
		for (auto &column : column_list.Physical()) {
			auto &name = column.Name();
			// FIXME: can these column names be aliased somehow?
			if (indexed_columns.count(column.Oid())) {
				continue;
			}
			columns.push_back(name);
		}
	} else {
		// a list of columns was explicitly supplied, only update those
		for (auto &name : insert.columns) {
			auto &column = column_list.GetColumn(name);
			if (indexed_columns.count(column.Oid())) {
				continue;
			}
			columns.push_back(name);
		}
	}

	// Create 'excluded' qualified column references of these columns
	for (auto &column : columns) {
		set_info->expressions.push_back(make_unique<ColumnRefExpression>(column, "excluded"));
	}

	return set_info;
}

void Binder::BindOnConflictClause(LogicalInsert &insert, TableCatalogEntry &table, InsertStatement &stmt) {
	if (!stmt.on_conflict_info) {
		insert.action_type = OnConflictAction::THROW;
		return;
	}
	if (!table.IsDuckTable()) {
		throw BinderException("ON CONFLICT clause is not yet supported for non-DuckDB tables");
	}
	D_ASSERT(stmt.table_ref->type == TableReferenceType::BASE_TABLE);

	// visit the table reference
	auto bound_table = Bind(*stmt.table_ref);
	if (bound_table->type != TableReferenceType::BASE_TABLE) {
		throw BinderException("Can only update base table!");
	}

	auto &table_ref = (BaseTableRef &)*stmt.table_ref;
	const string &table_alias = !table_ref.alias.empty() ? table_ref.alias : table_ref.table_name;

	auto &on_conflict = *stmt.on_conflict_info;
	D_ASSERT(on_conflict.action_type != OnConflictAction::THROW);
	insert.action_type = on_conflict.action_type;

	auto &columns = table.GetColumns();
	if (!on_conflict.indexed_columns.empty()) {
		// Bind the ON CONFLICT (<columns>)

		// create a mapping of (list index) -> (column index)
		case_insensitive_map_t<idx_t> specified_columns;
		for (idx_t i = 0; i < on_conflict.indexed_columns.size(); i++) {
			specified_columns[on_conflict.indexed_columns[i]] = i;
			auto column_index = table.GetColumnIndex(on_conflict.indexed_columns[i]);
			if (column_index.index == COLUMN_IDENTIFIER_ROW_ID) {
				throw BinderException("Cannot specify ROWID as ON CONFLICT target");
			}
			auto &col = columns.GetColumn(column_index);
			if (col.Generated()) {
				throw BinderException("Cannot specify a generated column as ON CONFLICT target");
			}
		}
		for (auto &col : columns.Physical()) {
			auto entry = specified_columns.find(col.Name());
			if (entry != specified_columns.end()) {
				// column was specified, set to the index
				insert.on_conflict_filter.insert(col.Oid());
			}
		}
		auto &indexes = table.GetStorage().info->indexes;
		bool index_references_columns = false;
		indexes.Scan([&](Index &index) {
			if (!index.IsUnique()) {
				return false;
			}
			bool index_matches = insert.on_conflict_filter == index.column_id_set;
			if (index_matches) {
				index_references_columns = true;
			}
			return index_matches;
		});
		if (!index_references_columns) {
			// Same as before, this is essentially a no-op, turning this into a DO THROW instead
			// But since this makes no logical sense, it's probably better to throw an error
			throw BinderException(
			    "The specified columns as conflict target are not referenced by a UNIQUE/PRIMARY KEY CONSTRAINT");
		}
	} else {
		// When omitting the conflict target, the ON CONFLICT applies to every UNIQUE/PRIMARY KEY on the table

		// We check if there are any constraints on the table, if there aren't we throw an error.
		auto &indexes = table.GetStorage().info->indexes;
		idx_t found_matching_indexes = 0;
		indexes.Scan([&](Index &index) {
			if (!index.IsUnique()) {
				return false;
			}
			// does this work with multi-column indexes?
			auto &indexed_columns = index.column_id_set;
			for (auto &column : table.GetColumns().Physical()) {
				if (indexed_columns.count(column.Physical().index)) {
					found_matching_indexes++;
				}
			}
			return false;
		});
		if (!found_matching_indexes) {
			throw BinderException(
			    "There are no UNIQUE/PRIMARY KEY Indexes that refer to this table, ON CONFLICT is a no-op");
		}
		if (insert.action_type != OnConflictAction::NOTHING && found_matching_indexes != 1) {
			// When no conflict target is provided, and the action type is UPDATE,
			// we only allow the operation when only a single Index exists
			throw BinderException("Conflict target has to be provided for a DO UPDATE operation when the table has "
			                      "multiple UNIQUE/PRIMARY KEY constraints");
		}
	}

	// add the 'excluded' dummy table binding
	AddTableName("excluded");
	// add a bind context entry for it
	auto excluded_index = GenerateTableIndex();
	insert.excluded_table_index = excluded_index;
	auto table_column_names = columns.GetColumnNames();
	auto table_column_types = columns.GetColumnTypes();
	bind_context.AddGenericBinding(excluded_index, "excluded", table_column_names, table_column_types);

	if (on_conflict.condition) {
		// Avoid ambiguity between <table_name> binding and 'excluded'
		QualifyColumnReferences(on_conflict.condition, table_alias);
		// Bind the ON CONFLICT ... WHERE clause
		WhereBinder where_binder(*this, context);
		auto condition = where_binder.Bind(on_conflict.condition);
		if (condition && condition->expression_class == ExpressionClass::BOUND_SUBQUERY) {
			throw BinderException("conflict_target WHERE clause can not be a subquery");
		}
		insert.on_conflict_condition = std::move(condition);
	}

	auto projection_index = insert.children[0]->GetTableIndex()[0];

	string unused;
	auto original_binding = bind_context.GetBinding(table_alias, unused);
	D_ASSERT(original_binding);

	auto table_index = original_binding->index;

	// Replace any column bindings to refer to the projection table_index, rather than the source table
	if (insert.on_conflict_condition) {
		ReplaceColumnBindings(*insert.on_conflict_condition, table_index, projection_index);
	}

	if (insert.action_type == OnConflictAction::NOTHING) {
		if (!insert.on_conflict_condition) {
			return;
		}
		// Get the column_ids we need to fetch later on from the conflicting tuples
		// of the original table, to execute the expressions
		D_ASSERT(original_binding->binding_type == BindingType::TABLE);
		auto table_binding = (TableBinding *)original_binding;
		insert.columns_to_fetch = table_binding->GetBoundColumnIds();
		return;
	}
	if (insert.action_type == OnConflictAction::REPLACE) {
		D_ASSERT(on_conflict.set_info == nullptr);
		on_conflict.set_info = CreateSetInfoForReplace(table, stmt);
		insert.action_type = OnConflictAction::UPDATE;
	}

	D_ASSERT(on_conflict.set_info);
	auto &set_info = *on_conflict.set_info;
	D_ASSERT(!set_info.columns.empty());
	D_ASSERT(set_info.columns.size() == set_info.expressions.size());

	if (set_info.condition) {
		// Avoid ambiguity between <table_name> binding and 'excluded'
		QualifyColumnReferences(set_info.condition, table_alias);
		// Bind the SET ... WHERE clause
		WhereBinder where_binder(*this, context);
		auto condition = where_binder.Bind(set_info.condition);
		if (condition && condition->expression_class == ExpressionClass::BOUND_SUBQUERY) {
			throw BinderException("conflict_target WHERE clause can not be a subquery");
		}
		insert.do_update_condition = std::move(condition);
	}

	BindDoUpdateSetExpressions(table_alias, &insert, set_info, table);

	// Get the column_ids we need to fetch later on from the conflicting tuples
	// of the original table, to execute the expressions
	D_ASSERT(original_binding->binding_type == BindingType::TABLE);
	auto table_binding = (TableBinding *)original_binding;
	insert.columns_to_fetch = table_binding->GetBoundColumnIds();

	// Replace the column bindings to refer to the child operator
	for (auto &expr : insert.expressions) {
		// Change the non-excluded column references to refer to the projection index
		ReplaceColumnBindings(*expr, table_index, projection_index);
	}
	// Do the same for the (optional) DO UPDATE condition
	if (insert.do_update_condition) {
		ReplaceColumnBindings(*insert.do_update_condition, table_index, projection_index);
	}
}

BoundStatement Binder::Bind(InsertStatement &stmt) {
	BoundStatement result;
	result.names = {"Count"};
	result.types = {LogicalType::BIGINT};

	BindSchemaOrCatalog(stmt.catalog, stmt.schema);
	auto table = Catalog::GetEntry<TableCatalogEntry>(context, stmt.catalog, stmt.schema, stmt.table);
	D_ASSERT(table);
	if (!table->temporary) {
		// inserting into a non-temporary table: alters underlying database
		properties.modified_databases.insert(table->catalog->GetName());
	}

	auto insert = make_unique<LogicalInsert>(table, GenerateTableIndex());
	// Add CTEs as bindable
	AddCTEMap(stmt.cte_map);

	vector<LogicalIndex> named_column_map;
	if (!stmt.columns.empty()) {
		// insertion statement specifies column list

		// create a mapping of (list index) -> (column index)
		case_insensitive_map_t<idx_t> column_name_map;
		for (idx_t i = 0; i < stmt.columns.size(); i++) {
			column_name_map[stmt.columns[i]] = i;
			auto column_index = table->GetColumnIndex(stmt.columns[i]);
			if (column_index.index == COLUMN_IDENTIFIER_ROW_ID) {
				throw BinderException("Cannot explicitly insert values into rowid column");
			}
			auto &col = table->GetColumn(column_index);
			if (col.Generated()) {
				throw BinderException("Cannot insert into a generated column");
			}
			insert->expected_types.push_back(col.Type());
			named_column_map.push_back(column_index);
		}
		for (auto &col : table->GetColumns().Physical()) {
			auto entry = column_name_map.find(col.Name());
			if (entry == column_name_map.end()) {
				// column not specified, set index to DConstants::INVALID_INDEX
				insert->column_index_map.push_back(DConstants::INVALID_INDEX);
			} else {
				// column was specified, set to the index
				insert->column_index_map.push_back(entry->second);
			}
		}
	} else {
		// No columns specified, assume insertion into all columns
		// Intentionally don't populate 'column_index_map' as an indication of this
		for (auto &col : table->GetColumns().Physical()) {
			named_column_map.push_back(col.Logical());
			insert->expected_types.push_back(col.Type());
		}
	}

	// bind the default values
	BindDefaultValues(table->GetColumns(), insert->bound_defaults);
	if (!stmt.select_statement) {
		result.plan = std::move(insert);
		return result;
	}

	// Exclude the generated columns from this amount
	idx_t expected_columns = stmt.columns.empty() ? table->GetColumns().PhysicalColumnCount() : stmt.columns.size();

	// special case: check if we are inserting from a VALUES statement
	auto values_list = stmt.GetValuesList();
	if (values_list) {
		auto &expr_list = (ExpressionListRef &)*values_list;
		expr_list.expected_types.resize(expected_columns);
		expr_list.expected_names.resize(expected_columns);

		D_ASSERT(expr_list.values.size() > 0);
		CheckInsertColumnCountMismatch(expected_columns, expr_list.values[0].size(), !stmt.columns.empty(),
		                               table->name.c_str());

		// VALUES list!
		for (idx_t col_idx = 0; col_idx < expected_columns; col_idx++) {
			D_ASSERT(named_column_map.size() >= col_idx);
			auto &table_col_idx = named_column_map[col_idx];

			// set the expected types as the types for the INSERT statement
			auto &column = table->GetColumn(table_col_idx);
			expr_list.expected_types[col_idx] = column.Type();
			expr_list.expected_names[col_idx] = column.Name();

			// now replace any DEFAULT values with the corresponding default expression
			for (idx_t list_idx = 0; list_idx < expr_list.values.size(); list_idx++) {
				if (expr_list.values[list_idx][col_idx]->type == ExpressionType::VALUE_DEFAULT) {
					// DEFAULT value! replace the entry
					ReplaceDefaultExpression(expr_list.values[list_idx][col_idx], column);
				}
			}
		}
	}

	// parse select statement and add to logical plan
	auto select_binder = Binder::CreateBinder(context, this);
	auto root_select = select_binder->Bind(*stmt.select_statement);
	MoveCorrelatedExpressions(*select_binder);

	CheckInsertColumnCountMismatch(expected_columns, root_select.types.size(), !stmt.columns.empty(),
	                               table->name.c_str());

	auto root = CastLogicalOperatorToTypes(root_select.types, insert->expected_types, std::move(root_select.plan));
	insert->AddChild(std::move(root));

	BindOnConflictClause(*insert, *table, stmt);

	if (!stmt.returning_list.empty()) {
		insert->return_chunk = true;
		result.types.clear();
		result.names.clear();
		auto insert_table_index = GenerateTableIndex();
		insert->table_index = insert_table_index;
		unique_ptr<LogicalOperator> index_as_logicaloperator = std::move(insert);

		return BindReturning(std::move(stmt.returning_list), table, insert_table_index,
		                     std::move(index_as_logicaloperator), std::move(result));
	}

	D_ASSERT(result.types.size() == result.names.size());
	result.plan = std::move(insert);
	properties.allow_stream_result = false;
	properties.return_type = StatementReturnType::CHANGED_ROWS;
	return result;
}

} // namespace duckdb



#include <algorithm>

namespace duckdb {

BoundStatement Binder::Bind(LoadStatement &stmt) {
	BoundStatement result;
	result.types = {LogicalType::BOOLEAN};
	result.names = {"Success"};

	result.plan = make_unique<LogicalSimple>(LogicalOperatorType::LOGICAL_LOAD, std::move(stmt.info));
	properties.allow_stream_result = false;
	properties.return_type = StatementReturnType::NOTHING;
	return result;
}

} // namespace duckdb


#include <algorithm>

namespace duckdb {

BoundStatement Binder::Bind(LogicalPlanStatement &stmt) {
	BoundStatement result;
	result.types = stmt.plan->types;
	for (idx_t i = 0; i < result.types.size(); i++) {
		result.names.push_back(StringUtil::Format("col%d", i));
	}
	result.plan = std::move(stmt.plan);
	properties.allow_stream_result = true;
	properties.return_type = StatementReturnType::QUERY_RESULT; // TODO could also be something else

	return result;
}

} // namespace duckdb







namespace duckdb {

BoundStatement Binder::Bind(PragmaStatement &stmt) {
	// bind the pragma function
	auto entry =
	    Catalog::GetEntry<PragmaFunctionCatalogEntry>(context, INVALID_CATALOG, DEFAULT_SCHEMA, stmt.info->name, false);
	string error;
	FunctionBinder function_binder(context);
	idx_t bound_idx = function_binder.BindFunction(entry->name, entry->functions, *stmt.info, error);
	if (bound_idx == DConstants::INVALID_INDEX) {
		throw BinderException(FormatError(stmt.stmt_location, error));
	}
	auto bound_function = entry->functions.GetFunctionByOffset(bound_idx);
	if (!bound_function.function) {
		throw BinderException("PRAGMA function does not have a function specified");
	}

	// bind and check named params
	QueryErrorContext error_context(root_statement, stmt.stmt_location);
	BindNamedParameters(bound_function.named_parameters, stmt.info->named_parameters, error_context,
	                    bound_function.name);

	BoundStatement result;
	result.names = {"Success"};
	result.types = {LogicalType::BOOLEAN};
	result.plan = make_unique<LogicalPragma>(bound_function, *stmt.info);
	properties.return_type = StatementReturnType::QUERY_RESULT;
	return result;
}

} // namespace duckdb





namespace duckdb {

BoundStatement Binder::Bind(PrepareStatement &stmt) {
	Planner prepared_planner(context);
	auto prepared_data = prepared_planner.PrepareSQLStatement(std::move(stmt.statement));
	this->bound_tables = prepared_planner.binder->bound_tables;

	auto prepare = make_unique<LogicalPrepare>(stmt.name, std::move(prepared_data), std::move(prepared_planner.plan));
	// we can always prepare, even if the transaction has been invalidated
	// this is required because most clients ALWAYS invoke prepared statements
	properties.requires_valid_transaction = false;
	properties.allow_stream_result = false;
	properties.bound_all_parameters = true;
	properties.parameter_count = 0;
	properties.return_type = StatementReturnType::NOTHING;

	BoundStatement result;
	result.names = {"Success"};
	result.types = {LogicalType::BOOLEAN};
	result.plan = std::move(prepare);
	return result;
}

} // namespace duckdb







namespace duckdb {

BoundStatement Binder::Bind(RelationStatement &stmt) {
	return stmt.relation->Bind(*this);
}

} // namespace duckdb




namespace duckdb {

BoundStatement Binder::Bind(SelectStatement &stmt) {
	properties.allow_stream_result = true;
	properties.return_type = StatementReturnType::QUERY_RESULT;
	return Bind(*stmt.node);
}

} // namespace duckdb




#include <algorithm>

namespace duckdb {

BoundStatement Binder::Bind(SetVariableStatement &stmt) {
	BoundStatement result;
	result.types = {LogicalType::BOOLEAN};
	result.names = {"Success"};

	result.plan = make_unique<LogicalSet>(stmt.name, stmt.value, stmt.scope);
	properties.return_type = StatementReturnType::NOTHING;
	return result;
}

BoundStatement Binder::Bind(ResetVariableStatement &stmt) {
	BoundStatement result;
	result.types = {LogicalType::BOOLEAN};
	result.names = {"Success"};

	result.plan = make_unique<LogicalReset>(stmt.name, stmt.scope);
	properties.return_type = StatementReturnType::NOTHING;
	return result;
}

BoundStatement Binder::Bind(SetStatement &stmt) {
	switch (stmt.set_type) {
	case SetType::SET: {
		auto &set_stmt = (SetVariableStatement &)stmt;
		return Bind(set_stmt);
	}
	case SetType::RESET: {
		auto &set_stmt = (ResetVariableStatement &)stmt;
		return Bind(set_stmt);
	}
	default:
		throw NotImplementedException("Type not implemented for SetType");
	}
}

} // namespace duckdb




namespace duckdb {

BoundStatement Binder::Bind(ShowStatement &stmt) {
	BoundStatement result;

	if (stmt.info->is_summary) {
		return BindSummarize(stmt);
	}
	auto plan = Bind(*stmt.info->query);
	stmt.info->types = plan.types;
	stmt.info->aliases = plan.names;

	auto show = make_unique<LogicalShow>(std::move(plan.plan));
	show->types_select = plan.types;
	show->aliases = plan.names;

	result.plan = std::move(show);

	result.names = {"column_name", "column_type", "null", "key", "default", "extra"};
	result.types = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
	                LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR};
	properties.return_type = StatementReturnType::QUERY_RESULT;
	return result;
}

} // namespace duckdb









//! This file contains the binder definitions for statements that do not need to be bound at all and only require a
//! straightforward conversion

namespace duckdb {

BoundStatement Binder::Bind(AlterStatement &stmt) {
	BoundStatement result;
	result.names = {"Success"};
	result.types = {LogicalType::BOOLEAN};
	BindSchemaOrCatalog(stmt.info->catalog, stmt.info->schema);
	auto entry = Catalog::GetEntry(context, stmt.info->GetCatalogType(), stmt.info->catalog, stmt.info->schema,
	                               stmt.info->name, stmt.info->if_exists);
	if (entry) {
		if (!entry->temporary) {
			// we can only alter temporary tables/views in read-only mode
			properties.modified_databases.insert(entry->catalog->GetName());
		}
		stmt.info->catalog = entry->catalog->GetName();
		stmt.info->schema = ((StandardEntry *)entry)->schema->name;
	}
	result.plan = make_unique<LogicalSimple>(LogicalOperatorType::LOGICAL_ALTER, std::move(stmt.info));
	properties.return_type = StatementReturnType::NOTHING;
	return result;
}

BoundStatement Binder::Bind(TransactionStatement &stmt) {
	// transaction statements do not require a valid transaction
	properties.requires_valid_transaction = stmt.info->type == TransactionType::BEGIN_TRANSACTION;

	BoundStatement result;
	result.names = {"Success"};
	result.types = {LogicalType::BOOLEAN};
	result.plan = make_unique<LogicalSimple>(LogicalOperatorType::LOGICAL_TRANSACTION, std::move(stmt.info));
	properties.return_type = StatementReturnType::NOTHING;
	return result;
}

} // namespace duckdb









namespace duckdb {

static unique_ptr<ParsedExpression> SummarizeWrapUnnest(vector<unique_ptr<ParsedExpression>> &children,
                                                        const string &alias) {
	auto list_function = make_unique<FunctionExpression>("list_value", std::move(children));
	vector<unique_ptr<ParsedExpression>> unnest_children;
	unnest_children.push_back(std::move(list_function));
	auto unnest_function = make_unique<FunctionExpression>("unnest", std::move(unnest_children));
	unnest_function->alias = alias;
	return std::move(unnest_function);
}

static unique_ptr<ParsedExpression> SummarizeCreateAggregate(const string &aggregate, string column_name) {
	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(make_unique<ColumnRefExpression>(std::move(column_name)));
	auto aggregate_function = make_unique<FunctionExpression>(aggregate, std::move(children));
	auto cast_function = make_unique<CastExpression>(LogicalType::VARCHAR, std::move(aggregate_function));
	return std::move(cast_function);
}

static unique_ptr<ParsedExpression> SummarizeCreateAggregate(const string &aggregate, string column_name,
                                                             const Value &modifier) {
	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(make_unique<ColumnRefExpression>(std::move(column_name)));
	children.push_back(make_unique<ConstantExpression>(modifier));
	auto aggregate_function = make_unique<FunctionExpression>(aggregate, std::move(children));
	auto cast_function = make_unique<CastExpression>(LogicalType::VARCHAR, std::move(aggregate_function));
	return std::move(cast_function);
}

static unique_ptr<ParsedExpression> SummarizeCreateCountStar() {
	vector<unique_ptr<ParsedExpression>> children;
	auto aggregate_function = make_unique<FunctionExpression>("count_star", std::move(children));
	return std::move(aggregate_function);
}

static unique_ptr<ParsedExpression> SummarizeCreateBinaryFunction(const string &op, unique_ptr<ParsedExpression> left,
                                                                  unique_ptr<ParsedExpression> right) {
	vector<unique_ptr<ParsedExpression>> children;
	children.push_back(std::move(left));
	children.push_back(std::move(right));
	auto binary_function = make_unique<FunctionExpression>(op, std::move(children));
	return std::move(binary_function);
}

static unique_ptr<ParsedExpression> SummarizeCreateNullPercentage(string column_name) {
	auto count_star = make_unique<CastExpression>(LogicalType::DOUBLE, SummarizeCreateCountStar());
	auto count =
	    make_unique<CastExpression>(LogicalType::DOUBLE, SummarizeCreateAggregate("count", std::move(column_name)));
	auto null_percentage = SummarizeCreateBinaryFunction("/", std::move(count), std::move(count_star));
	auto negate_x = SummarizeCreateBinaryFunction("-", make_unique<ConstantExpression>(Value::DOUBLE(1)),
	                                              std::move(null_percentage));
	auto percentage_x =
	    SummarizeCreateBinaryFunction("*", std::move(negate_x), make_unique<ConstantExpression>(Value::DOUBLE(100)));
	auto round_x = SummarizeCreateBinaryFunction("round", std::move(percentage_x),
	                                             make_unique<ConstantExpression>(Value::INTEGER(2)));
	auto concat_x =
	    SummarizeCreateBinaryFunction("concat", std::move(round_x), make_unique<ConstantExpression>(Value("%")));

	return concat_x;
}

BoundStatement Binder::BindSummarize(ShowStatement &stmt) {
	auto query_copy = stmt.info->query->Copy();

	// we bind the plan once in a child-node to figure out the column names and column types
	auto child_binder = Binder::CreateBinder(context);
	auto plan = child_binder->Bind(*stmt.info->query);
	D_ASSERT(plan.types.size() == plan.names.size());
	vector<unique_ptr<ParsedExpression>> name_children;
	vector<unique_ptr<ParsedExpression>> type_children;
	vector<unique_ptr<ParsedExpression>> min_children;
	vector<unique_ptr<ParsedExpression>> max_children;
	vector<unique_ptr<ParsedExpression>> unique_children;
	vector<unique_ptr<ParsedExpression>> avg_children;
	vector<unique_ptr<ParsedExpression>> std_children;
	vector<unique_ptr<ParsedExpression>> q25_children;
	vector<unique_ptr<ParsedExpression>> q50_children;
	vector<unique_ptr<ParsedExpression>> q75_children;
	vector<unique_ptr<ParsedExpression>> count_children;
	vector<unique_ptr<ParsedExpression>> null_percentage_children;
	auto select = make_unique<SelectStatement>();
	select->node = std::move(query_copy);
	for (idx_t i = 0; i < plan.names.size(); i++) {
		name_children.push_back(make_unique<ConstantExpression>(Value(plan.names[i])));
		type_children.push_back(make_unique<ConstantExpression>(Value(plan.types[i].ToString())));
		min_children.push_back(SummarizeCreateAggregate("min", plan.names[i]));
		max_children.push_back(SummarizeCreateAggregate("max", plan.names[i]));
		unique_children.push_back(SummarizeCreateAggregate("approx_count_distinct", plan.names[i]));
		if (plan.types[i].IsNumeric()) {
			avg_children.push_back(SummarizeCreateAggregate("avg", plan.names[i]));
			std_children.push_back(SummarizeCreateAggregate("stddev", plan.names[i]));
			q25_children.push_back(SummarizeCreateAggregate("approx_quantile", plan.names[i], Value::FLOAT(0.25)));
			q50_children.push_back(SummarizeCreateAggregate("approx_quantile", plan.names[i], Value::FLOAT(0.50)));
			q75_children.push_back(SummarizeCreateAggregate("approx_quantile", plan.names[i], Value::FLOAT(0.75)));
		} else {
			avg_children.push_back(make_unique<ConstantExpression>(Value()));
			std_children.push_back(make_unique<ConstantExpression>(Value()));
			q25_children.push_back(make_unique<ConstantExpression>(Value()));
			q50_children.push_back(make_unique<ConstantExpression>(Value()));
			q75_children.push_back(make_unique<ConstantExpression>(Value()));
		}
		count_children.push_back(SummarizeCreateCountStar());
		null_percentage_children.push_back(SummarizeCreateNullPercentage(plan.names[i]));
	}
	auto subquery_ref = make_unique<SubqueryRef>(std::move(select), "summarize_tbl");
	subquery_ref->column_name_alias = plan.names;

	auto select_node = make_unique<SelectNode>();
	select_node->select_list.push_back(SummarizeWrapUnnest(name_children, "column_name"));
	select_node->select_list.push_back(SummarizeWrapUnnest(type_children, "column_type"));
	select_node->select_list.push_back(SummarizeWrapUnnest(min_children, "min"));
	select_node->select_list.push_back(SummarizeWrapUnnest(max_children, "max"));
	select_node->select_list.push_back(SummarizeWrapUnnest(unique_children, "approx_unique"));
	select_node->select_list.push_back(SummarizeWrapUnnest(avg_children, "avg"));
	select_node->select_list.push_back(SummarizeWrapUnnest(std_children, "std"));
	select_node->select_list.push_back(SummarizeWrapUnnest(q25_children, "q25"));
	select_node->select_list.push_back(SummarizeWrapUnnest(q50_children, "q50"));
	select_node->select_list.push_back(SummarizeWrapUnnest(q75_children, "q75"));
	select_node->select_list.push_back(SummarizeWrapUnnest(count_children, "count"));
	select_node->select_list.push_back(SummarizeWrapUnnest(null_percentage_children, "null_percentage"));
	select_node->from_table = std::move(subquery_ref);

	properties.return_type = StatementReturnType::QUERY_RESULT;
	return Bind(*select_node);
}

} // namespace duckdb


















#include <algorithm>

namespace duckdb {

static void BindExtraColumns(TableCatalogEntry &table, LogicalGet &get, LogicalProjection &proj, LogicalUpdate &update,
                             physical_index_set_t &bound_columns) {
	if (bound_columns.size() <= 1) {
		return;
	}
	idx_t found_column_count = 0;
	physical_index_set_t found_columns;
	for (idx_t i = 0; i < update.columns.size(); i++) {
		if (bound_columns.find(update.columns[i]) != bound_columns.end()) {
			// this column is referenced in the CHECK constraint
			found_column_count++;
			found_columns.insert(update.columns[i]);
		}
	}
	if (found_column_count > 0 && found_column_count != bound_columns.size()) {
		// columns in this CHECK constraint were referenced, but not all were part of the UPDATE
		// add them to the scan and update set
		for (auto &check_column_id : bound_columns) {
			if (found_columns.find(check_column_id) != found_columns.end()) {
				// column is already projected
				continue;
			}
			// column is not projected yet: project it by adding the clause "i=i" to the set of updated columns
			auto &column = table.GetColumns().GetColumn(check_column_id);
			update.expressions.push_back(make_unique<BoundColumnRefExpression>(
			    column.Type(), ColumnBinding(proj.table_index, proj.expressions.size())));
			proj.expressions.push_back(make_unique<BoundColumnRefExpression>(
			    column.Type(), ColumnBinding(get.table_index, get.column_ids.size())));
			get.column_ids.push_back(check_column_id.index);
			update.columns.push_back(check_column_id);
		}
	}
}

static bool TypeSupportsRegularUpdate(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::LIST:
	case LogicalTypeId::MAP:
	case LogicalTypeId::UNION:
		// lists and maps and unions don't support updates directly
		return false;
	case LogicalTypeId::STRUCT: {
		auto &child_types = StructType::GetChildTypes(type);
		for (auto &entry : child_types) {
			if (!TypeSupportsRegularUpdate(entry.second)) {
				return false;
			}
		}
		return true;
	}
	default:
		return true;
	}
}

static void BindUpdateConstraints(TableCatalogEntry &table, LogicalGet &get, LogicalProjection &proj,
                                  LogicalUpdate &update) {
	if (!table.IsDuckTable()) {
		return;
	}
	// check the constraints and indexes of the table to see if we need to project any additional columns
	// we do this for indexes with multiple columns and CHECK constraints in the UPDATE clause
	// suppose we have a constraint CHECK(i + j < 10); now we need both i and j to check the constraint
	// if we are only updating one of the two columns we add the other one to the UPDATE set
	// with a "useless" update (i.e. i=i) so we can verify that the CHECK constraint is not violated
	for (auto &constraint : table.GetBoundConstraints()) {
		if (constraint->type == ConstraintType::CHECK) {
			auto &check = *reinterpret_cast<BoundCheckConstraint *>(constraint.get());
			// check constraint! check if we need to add any extra columns to the UPDATE clause
			BindExtraColumns(table, get, proj, update, check.bound_columns);
		}
	}
	auto &storage = table.GetStorage();
	if (update.return_chunk) {
		physical_index_set_t all_columns;
		for (idx_t i = 0; i < storage.column_definitions.size(); i++) {
			all_columns.insert(PhysicalIndex(i));
		}
		BindExtraColumns(table, get, proj, update, all_columns);
	}
	// for index updates we always turn any update into an insert and a delete
	// we thus need all the columns to be available, hence we check if the update touches any index columns
	// If the returning keyword is used, we need access to the whole row in case the user requests it.
	// Therefore switch the update to a delete and insert.
	update.update_is_del_and_insert = false;
	storage.info->indexes.Scan([&](Index &index) {
		if (index.IndexIsUpdated(update.columns)) {
			update.update_is_del_and_insert = true;
			return true;
		}
		return false;
	});

	// we also convert any updates on LIST columns into delete + insert
	for (auto &col_index : update.columns) {
		auto &column = table.GetColumns().GetColumn(col_index);
		if (!TypeSupportsRegularUpdate(column.Type())) {
			update.update_is_del_and_insert = true;
			break;
		}
	}

	if (update.update_is_del_and_insert) {
		// the update updates a column required by an index or requires returning the updated rows,
		// push projections for all columns
		physical_index_set_t all_columns;
		for (idx_t i = 0; i < storage.column_definitions.size(); i++) {
			all_columns.insert(PhysicalIndex(i));
		}
		BindExtraColumns(table, get, proj, update, all_columns);
	}
}

// This creates a LogicalProjection and moves 'root' into it as a child
// unless there are no expressions to project, in which case it just returns 'root'
unique_ptr<LogicalOperator> Binder::BindUpdateSet(LogicalOperator *op, unique_ptr<LogicalOperator> root,
                                                  UpdateSetInfo &set_info, TableCatalogEntry *table,
                                                  vector<PhysicalIndex> &columns) {
	auto proj_index = GenerateTableIndex();

	vector<unique_ptr<Expression>> projection_expressions;
	D_ASSERT(set_info.columns.size() == set_info.expressions.size());
	for (idx_t i = 0; i < set_info.columns.size(); i++) {
		auto &colname = set_info.columns[i];
		auto &expr = set_info.expressions[i];
		if (!table->ColumnExists(colname)) {
			throw BinderException("Referenced update column %s not found in table!", colname);
		}
		auto &column = table->GetColumn(colname);
		if (column.Generated()) {
			throw BinderException("Cant update column \"%s\" because it is a generated column!", column.Name());
		}
		if (std::find(columns.begin(), columns.end(), column.Physical()) != columns.end()) {
			throw BinderException("Multiple assignments to same column \"%s\"", colname);
		}
		columns.push_back(column.Physical());
		if (expr->type == ExpressionType::VALUE_DEFAULT) {
			op->expressions.push_back(make_unique<BoundDefaultExpression>(column.Type()));
		} else {
			UpdateBinder binder(*this, context);
			binder.target_type = column.Type();
			auto bound_expr = binder.Bind(expr);
			PlanSubqueries(&bound_expr, &root);

			op->expressions.push_back(make_unique<BoundColumnRefExpression>(
			    bound_expr->return_type, ColumnBinding(proj_index, projection_expressions.size())));
			projection_expressions.push_back(std::move(bound_expr));
		}
	}
	if (op->type != LogicalOperatorType::LOGICAL_UPDATE && projection_expressions.empty()) {
		return root;
	}
	// now create the projection
	auto proj = make_unique<LogicalProjection>(proj_index, std::move(projection_expressions));
	proj->AddChild(std::move(root));
	return unique_ptr_cast<LogicalProjection, LogicalOperator>(std::move(proj));
}

BoundStatement Binder::Bind(UpdateStatement &stmt) {
	BoundStatement result;
	unique_ptr<LogicalOperator> root;
	LogicalGet *get;

	// visit the table reference
	auto bound_table = Bind(*stmt.table);
	if (bound_table->type != TableReferenceType::BASE_TABLE) {
		throw BinderException("Can only update base table!");
	}
	auto &table_binding = (BoundBaseTableRef &)*bound_table;
	auto table = table_binding.table;

	// Add CTEs as bindable
	AddCTEMap(stmt.cte_map);

	if (stmt.from_table) {
		BoundJoinRef bound_crossproduct(JoinRefType::CROSS);
		bound_crossproduct.left = std::move(bound_table);
		bound_crossproduct.right = Bind(*stmt.from_table);
		root = CreatePlan(bound_crossproduct);
		get = (LogicalGet *)root->children[0].get();
	} else {
		root = CreatePlan(*bound_table);
		get = (LogicalGet *)root.get();
	}

	if (!table->temporary) {
		// update of persistent table: not read only!
		properties.modified_databases.insert(table->catalog->GetName());
	}
	auto update = make_unique<LogicalUpdate>(table);

	// set return_chunk boolean early because it needs uses update_is_del_and_insert logic
	if (!stmt.returning_list.empty()) {
		update->return_chunk = true;
	}
	// bind the default values
	BindDefaultValues(table->GetColumns(), update->bound_defaults);

	// project any additional columns required for the condition/expressions
	if (stmt.set_info->condition) {
		WhereBinder binder(*this, context);
		auto condition = binder.Bind(stmt.set_info->condition);

		PlanSubqueries(&condition, &root);
		auto filter = make_unique<LogicalFilter>(std::move(condition));
		filter->AddChild(std::move(root));
		root = std::move(filter);
	}

	D_ASSERT(stmt.set_info);
	D_ASSERT(stmt.set_info->columns.size() == stmt.set_info->expressions.size());

	auto proj_tmp = BindUpdateSet(update.get(), std::move(root), *stmt.set_info, table, update->columns);
	D_ASSERT(proj_tmp->type == LogicalOperatorType::LOGICAL_PROJECTION);
	auto proj = unique_ptr_cast<LogicalOperator, LogicalProjection>(std::move(proj_tmp));

	// bind any extra columns necessary for CHECK constraints or indexes
	BindUpdateConstraints(*table, *get, *proj, *update);
	// finally add the row id column to the projection list
	proj->expressions.push_back(make_unique<BoundColumnRefExpression>(
	    LogicalType::ROW_TYPE, ColumnBinding(get->table_index, get->column_ids.size())));
	get->column_ids.push_back(COLUMN_IDENTIFIER_ROW_ID);

	// set the projection as child of the update node and finalize the result
	update->AddChild(std::move(proj));

	auto update_table_index = GenerateTableIndex();
	update->table_index = update_table_index;
	if (!stmt.returning_list.empty()) {
		unique_ptr<LogicalOperator> update_as_logicaloperator = std::move(update);

		return BindReturning(std::move(stmt.returning_list), table, update_table_index,
		                     std::move(update_as_logicaloperator), std::move(result));
	}

	result.names = {"Count"};
	result.types = {LogicalType::BIGINT};
	result.plan = std::move(update);
	properties.allow_stream_result = false;
	properties.return_type = StatementReturnType::CHANGED_ROWS;
	return result;
}

} // namespace duckdb








namespace duckdb {

BoundStatement Binder::Bind(VacuumStatement &stmt) {
	BoundStatement result;

	unique_ptr<LogicalOperator> root;

	if (stmt.info->has_table) {
		D_ASSERT(!stmt.info->table);
		D_ASSERT(stmt.info->column_id_map.empty());
		auto bound_table = Bind(*stmt.info->ref);
		if (bound_table->type != TableReferenceType::BASE_TABLE) {
			throw InvalidInputException("Can only vacuum/analyze base tables!");
		}
		auto ref = unique_ptr_cast<BoundTableRef, BoundBaseTableRef>(std::move(bound_table));
		stmt.info->table = ref->table;

		auto &columns = stmt.info->columns;
		vector<unique_ptr<Expression>> select_list;
		if (columns.empty()) {
			// Empty means ALL columns should be vacuumed/analyzed
			auto &get = (LogicalGet &)*ref->get;
			columns.insert(columns.end(), get.names.begin(), get.names.end());
		}

		case_insensitive_set_t column_name_set;
		vector<string> non_generated_column_names;
		for (auto &col_name : columns) {
			if (column_name_set.count(col_name) > 0) {
				throw BinderException("Vacuum the same column twice(same name in column name list)");
			}
			column_name_set.insert(col_name);
			if (!ref->table->ColumnExists(col_name)) {
				throw BinderException("Column with name \"%s\" does not exist", col_name);
			}
			auto &col = ref->table->GetColumn(col_name);
			// ignore generated column
			if (col.Generated()) {
				continue;
			}
			non_generated_column_names.push_back(col_name);
			ColumnRefExpression colref(col_name, ref->table->name);
			auto result = bind_context.BindColumn(colref, 0);
			if (result.HasError()) {
				throw BinderException(result.error);
			}
			select_list.push_back(std::move(result.expression));
		}
		stmt.info->columns = std::move(non_generated_column_names);
		if (!select_list.empty()) {
			auto table_scan = CreatePlan(*ref);
			D_ASSERT(table_scan->type == LogicalOperatorType::LOGICAL_GET);

			auto &get = (LogicalGet &)*table_scan;

			D_ASSERT(select_list.size() == get.column_ids.size());
			D_ASSERT(stmt.info->columns.size() == get.column_ids.size());
			for (idx_t i = 0; i < get.column_ids.size(); i++) {
				stmt.info->column_id_map[i] =
				    ref->table->GetColumns().LogicalToPhysical(LogicalIndex(get.column_ids[i])).index;
			}

			auto projection = make_unique<LogicalProjection>(GenerateTableIndex(), std::move(select_list));
			projection->children.push_back(std::move(table_scan));

			root = std::move(projection);
		} else {
			// eg. CREATE TABLE test (x AS (1));
			//     ANALYZE test;
			// Make it not a SINK so it doesn't have to do anything
			stmt.info->has_table = false;
		}
	}
	auto vacuum = make_unique<LogicalSimple>(LogicalOperatorType::LOGICAL_VACUUM, std::move(stmt.info));
	if (root) {
		vacuum->children.push_back(std::move(root));
	}

	result.names = {"Success"};
	result.types = {LogicalType::BOOLEAN};
	result.plan = std::move(vacuum);
	properties.return_type = StatementReturnType::NOTHING;
	return result;
}

} // namespace duckdb

















namespace duckdb {

unique_ptr<BoundTableRef> Binder::Bind(BaseTableRef &ref) {
	QueryErrorContext error_context(root_statement, ref.query_location);
	// CTEs and views are also referred to using BaseTableRefs, hence need to distinguish here
	// check if the table name refers to a CTE
	auto cte = FindCTE(ref.table_name, ref.table_name == alias);
	if (cte) {
		// Check if there is a CTE binding in the BindContext
		auto ctebinding = bind_context.GetCTEBinding(ref.table_name);
		if (!ctebinding) {
			if (CTEIsAlreadyBound(cte)) {
				throw BinderException("Circular reference to CTE \"%s\", use WITH RECURSIVE to use recursive CTEs",
				                      ref.table_name);
			}
			// Move CTE to subquery and bind recursively
			SubqueryRef subquery(unique_ptr_cast<SQLStatement, SelectStatement>(cte->query->Copy()));
			subquery.alias = ref.alias.empty() ? ref.table_name : ref.alias;
			subquery.column_name_alias = cte->aliases;
			for (idx_t i = 0; i < ref.column_name_alias.size(); i++) {
				if (i < subquery.column_name_alias.size()) {
					subquery.column_name_alias[i] = ref.column_name_alias[i];
				} else {
					subquery.column_name_alias.push_back(ref.column_name_alias[i]);
				}
			}
			return Bind(subquery, cte);
		} else {
			// There is a CTE binding in the BindContext.
			// This can only be the case if there is a recursive CTE present.
			auto index = GenerateTableIndex();
			auto result = make_unique<BoundCTERef>(index, ctebinding->index);
			auto b = ctebinding;
			auto alias = ref.alias.empty() ? ref.table_name : ref.alias;
			auto names = BindContext::AliasColumnNames(alias, b->names, ref.column_name_alias);

			bind_context.AddGenericBinding(index, alias, names, b->types);
			// Update references to CTE
			auto cteref = bind_context.cte_references[ref.table_name];
			(*cteref)++;

			result->types = b->types;
			result->bound_columns = std::move(names);
			return std::move(result);
		}
	}
	// not a CTE
	// extract a table or view from the catalog
	BindSchemaOrCatalog(ref.catalog_name, ref.schema_name);
	auto table_or_view = Catalog::GetEntry(context, CatalogType::TABLE_ENTRY, ref.catalog_name, ref.schema_name,
	                                       ref.table_name, true, error_context);
	if (!table_or_view) {
		string table_name = ref.catalog_name;
		if (!ref.schema_name.empty()) {
			table_name += (!table_name.empty() ? "." : "") + ref.schema_name;
		}
		table_name += (!table_name.empty() ? "." : "") + ref.table_name;
		// table could not be found: try to bind a replacement scan
		auto &config = DBConfig::GetConfig(context);
		if (context.config.use_replacement_scans) {
			for (auto &scan : config.replacement_scans) {
				auto replacement_function = scan.function(context, table_name, scan.data.get());
				if (replacement_function) {
					replacement_function->alias = ref.alias.empty() ? ref.table_name : ref.alias;
					if (replacement_function->type == TableReferenceType::TABLE_FUNCTION) {
						auto &table_function = (TableFunctionRef &)*replacement_function;
						table_function.column_name_alias = ref.column_name_alias;
						;
					} else if (replacement_function->type == TableReferenceType::SUBQUERY) {
						auto &subquery = (SubqueryRef &)*replacement_function;
						subquery.column_name_alias = ref.column_name_alias;
					} else {
						throw InternalException("Replacement scan should return either a table function or a subquery");
					}
					return Bind(*replacement_function);
				}
			}
		}

		// we still didn't find the table
		if (GetBindingMode() == BindingMode::EXTRACT_NAMES) {
			// if we are in EXTRACT_NAMES, we create a dummy table ref
			AddTableName(table_name);

			// add a bind context entry
			auto table_index = GenerateTableIndex();
			auto alias = ref.alias.empty() ? table_name : ref.alias;
			vector<LogicalType> types {LogicalType::INTEGER};
			vector<string> names {"__dummy_col" + to_string(table_index)};
			bind_context.AddGenericBinding(table_index, alias, names, types);
			return make_unique_base<BoundTableRef, BoundEmptyTableRef>(table_index);
		}
		// could not find an alternative: bind again to get the error
		table_or_view = Catalog::GetEntry(context, CatalogType::TABLE_ENTRY, ref.catalog_name, ref.schema_name,
		                                  ref.table_name, false, error_context);
	}
	switch (table_or_view->type) {
	case CatalogType::TABLE_ENTRY: {
		// base table: create the BoundBaseTableRef node
		auto table_index = GenerateTableIndex();
		auto table = (TableCatalogEntry *)table_or_view;

		unique_ptr<FunctionData> bind_data;
		auto scan_function = table->GetScanFunction(context, bind_data);
		auto alias = ref.alias.empty() ? ref.table_name : ref.alias;
		// TODO: bundle the type and name vector in a struct (e.g PackedColumnMetadata)
		vector<LogicalType> table_types;
		vector<string> table_names;
		vector<TableColumnType> table_categories;

		vector<LogicalType> return_types;
		vector<string> return_names;
		for (auto &col : table->GetColumns().Logical()) {
			table_types.push_back(col.Type());
			table_names.push_back(col.Name());
			return_types.push_back(col.Type());
			return_names.push_back(col.Name());
		}
		table_names = BindContext::AliasColumnNames(alias, table_names, ref.column_name_alias);

		auto logical_get = make_unique<LogicalGet>(table_index, scan_function, std::move(bind_data),
		                                           std::move(return_types), std::move(return_names));
		bind_context.AddBaseTable(table_index, alias, table_names, table_types, logical_get->column_ids,
		                          logical_get->GetTable());
		return make_unique_base<BoundTableRef, BoundBaseTableRef>(table, std::move(logical_get));
	}
	case CatalogType::VIEW_ENTRY: {
		// the node is a view: get the query that the view represents
		auto view_catalog_entry = (ViewCatalogEntry *)table_or_view;
		// We need to use a new binder for the view that doesn't reference any CTEs
		// defined for this binder so there are no collisions between the CTEs defined
		// for the view and for the current query
		bool inherit_ctes = false;
		auto view_binder = Binder::CreateBinder(context, this, inherit_ctes);
		view_binder->can_contain_nulls = true;
		SubqueryRef subquery(unique_ptr_cast<SQLStatement, SelectStatement>(view_catalog_entry->query->Copy()));
		subquery.alias = ref.alias.empty() ? ref.table_name : ref.alias;
		subquery.column_name_alias =
		    BindContext::AliasColumnNames(subquery.alias, view_catalog_entry->aliases, ref.column_name_alias);
		// bind the child subquery
		view_binder->AddBoundView(view_catalog_entry);
		auto bound_child = view_binder->Bind(subquery);
		if (!view_binder->correlated_columns.empty()) {
			throw BinderException("Contents of view were altered - view bound correlated columns");
		}

		D_ASSERT(bound_child->type == TableReferenceType::SUBQUERY);
		// verify that the types and names match up with the expected types and names
		auto &bound_subquery = (BoundSubqueryRef &)*bound_child;
		if (bound_subquery.subquery->types != view_catalog_entry->types) {
			throw BinderException("Contents of view were altered: types don't match!");
		}
		bind_context.AddView(bound_subquery.subquery->GetRootIndex(), subquery.alias, subquery,
		                     *bound_subquery.subquery, view_catalog_entry);
		return bound_child;
	}
	default:
		throw InternalException("Catalog entry type");
	}
}
} // namespace duckdb




namespace duckdb {

unique_ptr<BoundTableRef> Binder::Bind(EmptyTableRef &ref) {
	return make_unique<BoundEmptyTableRef>(GenerateTableIndex());
}

} // namespace duckdb







namespace duckdb {

unique_ptr<BoundTableRef> Binder::Bind(ExpressionListRef &expr) {
	auto result = make_unique<BoundExpressionListRef>();
	result->types = expr.expected_types;
	result->names = expr.expected_names;
	// bind value list
	InsertBinder binder(*this, context);
	binder.target_type = LogicalType(LogicalTypeId::INVALID);
	for (idx_t list_idx = 0; list_idx < expr.values.size(); list_idx++) {
		auto &expression_list = expr.values[list_idx];
		if (result->names.empty()) {
			// no names provided, generate them
			for (idx_t val_idx = 0; val_idx < expression_list.size(); val_idx++) {
				result->names.push_back("col" + to_string(val_idx));
			}
		}

		vector<unique_ptr<Expression>> list;
		for (idx_t val_idx = 0; val_idx < expression_list.size(); val_idx++) {
			if (!result->types.empty()) {
				D_ASSERT(result->types.size() == expression_list.size());
				binder.target_type = result->types[val_idx];
			}
			auto expr = binder.Bind(expression_list[val_idx]);
			list.push_back(std::move(expr));
		}
		result->values.push_back(std::move(list));
	}
	if (result->types.empty() && !expr.values.empty()) {
		// there are no types specified
		// we have to figure out the result types
		// for each column, we iterate over all of the expressions and select the max logical type
		// we initialize all types to SQLNULL
		result->types.resize(expr.values[0].size(), LogicalType::SQLNULL);
		// now loop over the lists and select the max logical type
		for (idx_t list_idx = 0; list_idx < result->values.size(); list_idx++) {
			auto &list = result->values[list_idx];
			for (idx_t val_idx = 0; val_idx < list.size(); val_idx++) {
				result->types[val_idx] =
				    LogicalType::MaxLogicalType(result->types[val_idx], list[val_idx]->return_type);
			}
		}
		// finally do another loop over the expressions and add casts where required
		for (idx_t list_idx = 0; list_idx < result->values.size(); list_idx++) {
			auto &list = result->values[list_idx];
			for (idx_t val_idx = 0; val_idx < list.size(); val_idx++) {
				list[val_idx] =
				    BoundCastExpression::AddCastToType(context, std::move(list[val_idx]), result->types[val_idx]);
			}
		}
	}
	result->bind_index = GenerateTableIndex();
	bind_context.AddGenericBinding(result->bind_index, expr.alias, result->names, result->types);
	return std::move(result);
}

} // namespace duckdb













namespace duckdb {

static unique_ptr<ParsedExpression> BindColumn(Binder &binder, ClientContext &context, const string &alias,
                                               const string &column_name) {
	auto expr = make_unique_base<ParsedExpression, ColumnRefExpression>(column_name, alias);
	ExpressionBinder expr_binder(binder, context);
	auto result = expr_binder.Bind(expr);
	return make_unique<BoundExpression>(std::move(result));
}

static unique_ptr<ParsedExpression> AddCondition(ClientContext &context, Binder &left_binder, Binder &right_binder,
                                                 const string &left_alias, const string &right_alias,
                                                 const string &column_name) {
	ExpressionBinder expr_binder(left_binder, context);
	auto left = BindColumn(left_binder, context, left_alias, column_name);
	auto right = BindColumn(right_binder, context, right_alias, column_name);
	return make_unique<ComparisonExpression>(ExpressionType::COMPARE_EQUAL, std::move(left), std::move(right));
}

bool Binder::TryFindBinding(const string &using_column, const string &join_side, string &result) {
	// for each using column, get the matching binding
	auto bindings = bind_context.GetMatchingBindings(using_column);
	if (bindings.empty()) {
		return false;
	}
	// find the join binding
	for (auto &binding : bindings) {
		if (!result.empty()) {
			string error = "Column name \"";
			error += using_column;
			error += "\" is ambiguous: it exists more than once on ";
			error += join_side;
			error += " side of join.\nCandidates:";
			for (auto &binding : bindings) {
				error += "\n\t";
				error += binding;
				error += ".";
				error += bind_context.GetActualColumnName(binding, using_column);
			}
			throw BinderException(error);
		} else {
			result = binding;
		}
	}
	return true;
}

string Binder::FindBinding(const string &using_column, const string &join_side) {
	string result;
	if (!TryFindBinding(using_column, join_side, result)) {
		throw BinderException("Column \"%s\" does not exist on %s side of join!", using_column, join_side);
	}
	return result;
}

static void AddUsingBindings(UsingColumnSet &set, UsingColumnSet *input_set, const string &input_binding) {
	if (input_set) {
		for (auto &entry : input_set->bindings) {
			set.bindings.insert(entry);
		}
	} else {
		set.bindings.insert(input_binding);
	}
}

static void SetPrimaryBinding(UsingColumnSet &set, JoinType join_type, const string &left_binding,
                              const string &right_binding) {
	switch (join_type) {
	case JoinType::LEFT:
	case JoinType::INNER:
	case JoinType::SEMI:
	case JoinType::ANTI:
		set.primary_binding = left_binding;
		break;
	case JoinType::RIGHT:
		set.primary_binding = right_binding;
		break;
	default:
		break;
	}
}

string Binder::RetrieveUsingBinding(Binder &current_binder, UsingColumnSet *current_set, const string &using_column,
                                    const string &join_side, UsingColumnSet *new_set) {
	string binding;
	if (!current_set) {
		binding = current_binder.FindBinding(using_column, join_side);
	} else {
		binding = current_set->primary_binding;
	}
	return binding;
}

static vector<string> RemoveDuplicateUsingColumns(const vector<string> &using_columns) {
	vector<string> result;
	case_insensitive_set_t handled_columns;
	for (auto &using_column : using_columns) {
		if (handled_columns.find(using_column) == handled_columns.end()) {
			handled_columns.insert(using_column);
			result.push_back(using_column);
		}
	}
	return result;
}

unique_ptr<BoundTableRef> Binder::Bind(JoinRef &ref) {
	auto result = make_unique<BoundJoinRef>(ref.ref_type);
	result->left_binder = Binder::CreateBinder(context, this);
	result->right_binder = Binder::CreateBinder(context, this);
	auto &left_binder = *result->left_binder;
	auto &right_binder = *result->right_binder;

	result->type = ref.type;
	result->left = left_binder.Bind(*ref.left);
	{
		LateralBinder binder(left_binder, context);
		result->right = right_binder.Bind(*ref.right);
		result->lateral = binder.HasCorrelatedColumns();
		if (result->lateral) {
			// lateral join: can only be an INNER or LEFT join
			if (ref.type != JoinType::INNER && ref.type != JoinType::LEFT) {
				throw BinderException("The combining JOIN type must be INNER or LEFT for a LATERAL reference");
			}
		}
		result->correlated_columns = binder.ExtractCorrelatedColumns(right_binder);
	}

	vector<unique_ptr<ParsedExpression>> extra_conditions;
	vector<string> extra_using_columns;
	switch (ref.ref_type) {
	case JoinRefType::NATURAL: {
		// natural join, figure out which column names are present in both sides of the join
		// first bind the left hand side and get a list of all the tables and column names
		case_insensitive_set_t lhs_columns;
		auto &lhs_binding_list = left_binder.bind_context.GetBindingsList();
		for (auto &binding : lhs_binding_list) {
			for (auto &column_name : binding.second->names) {
				lhs_columns.insert(column_name);
			}
		}
		// now bind the rhs
		for (auto &column_name : lhs_columns) {
			auto right_using_binding = right_binder.bind_context.GetUsingBinding(column_name);

			string right_binding;
			// loop over the set of lhs columns, and figure out if there is a table in the rhs with the same name
			if (!right_using_binding) {
				if (!right_binder.TryFindBinding(column_name, "right", right_binding)) {
					// no match found for this column on the rhs: skip
					continue;
				}
			}
			extra_using_columns.push_back(column_name);
		}
		if (extra_using_columns.empty()) {
			// no matching bindings found in natural join: throw an exception
			string error_msg = "No columns found to join on in NATURAL JOIN.\n";
			error_msg += "Use CROSS JOIN if you intended for this to be a cross-product.";
			// gather all left/right candidates
			string left_candidates, right_candidates;
			auto &rhs_binding_list = right_binder.bind_context.GetBindingsList();
			for (auto &binding : lhs_binding_list) {
				for (auto &column_name : binding.second->names) {
					if (!left_candidates.empty()) {
						left_candidates += ", ";
					}
					left_candidates += binding.first + "." + column_name;
				}
			}
			for (auto &binding : rhs_binding_list) {
				for (auto &column_name : binding.second->names) {
					if (!right_candidates.empty()) {
						right_candidates += ", ";
					}
					right_candidates += binding.first + "." + column_name;
				}
			}
			error_msg += "\n   Left candidates: " + left_candidates;
			error_msg += "\n   Right candidates: " + right_candidates;
			throw BinderException(FormatError(ref, error_msg));
		}
		break;
	}
	case JoinRefType::REGULAR:
		if (!ref.using_columns.empty()) {
			// USING columns
			D_ASSERT(!result->condition);
			extra_using_columns = ref.using_columns;
		}
		break;
	case JoinRefType::CROSS:
	case JoinRefType::POSITIONAL:
		break;
	}
	extra_using_columns = RemoveDuplicateUsingColumns(extra_using_columns);

	if (!extra_using_columns.empty()) {
		vector<UsingColumnSet *> left_using_bindings;
		vector<UsingColumnSet *> right_using_bindings;
		for (idx_t i = 0; i < extra_using_columns.size(); i++) {
			auto &using_column = extra_using_columns[i];
			// we check if there is ALREADY a using column of the same name in the left and right set
			// this can happen if we chain USING clauses
			// e.g. x JOIN y USING (c) JOIN z USING (c)
			auto left_using_binding = left_binder.bind_context.GetUsingBinding(using_column);
			auto right_using_binding = right_binder.bind_context.GetUsingBinding(using_column);
			if (!left_using_binding) {
				left_binder.bind_context.GetMatchingBinding(using_column);
			}
			if (!right_using_binding) {
				right_binder.bind_context.GetMatchingBinding(using_column);
			}
			left_using_bindings.push_back(left_using_binding);
			right_using_bindings.push_back(right_using_binding);
		}

		for (idx_t i = 0; i < extra_using_columns.size(); i++) {
			auto &using_column = extra_using_columns[i];
			string left_binding;
			string right_binding;

			auto set = make_unique<UsingColumnSet>();
			auto left_using_binding = left_using_bindings[i];
			auto right_using_binding = right_using_bindings[i];
			left_binding = RetrieveUsingBinding(left_binder, left_using_binding, using_column, "left", set.get());
			right_binding = RetrieveUsingBinding(right_binder, right_using_binding, using_column, "right", set.get());

			extra_conditions.push_back(
			    AddCondition(context, left_binder, right_binder, left_binding, right_binding, using_column));

			AddUsingBindings(*set, left_using_binding, left_binding);
			AddUsingBindings(*set, right_using_binding, right_binding);
			SetPrimaryBinding(*set, ref.type, left_binding, right_binding);
			bind_context.TransferUsingBinding(left_binder.bind_context, left_using_binding, set.get(), left_binding,
			                                  using_column);
			bind_context.TransferUsingBinding(right_binder.bind_context, right_using_binding, set.get(), right_binding,
			                                  using_column);
			AddUsingBindingSet(std::move(set));
		}
	}

	bind_context.AddContext(std::move(left_binder.bind_context));
	bind_context.AddContext(std::move(right_binder.bind_context));
	MoveCorrelatedExpressions(left_binder);
	MoveCorrelatedExpressions(right_binder);
	for (auto &condition : extra_conditions) {
		if (ref.condition) {
			ref.condition = make_unique<ConjunctionExpression>(ExpressionType::CONJUNCTION_AND,
			                                                   std::move(ref.condition), std::move(condition));
		} else {
			ref.condition = std::move(condition);
		}
	}
	if (ref.condition) {
		WhereBinder binder(*this, context);
		result->condition = binder.Bind(ref.condition);
	}
	return std::move(result);
}

} // namespace duckdb


namespace duckdb {

void Binder::BindNamedParameters(named_parameter_type_map_t &types, named_parameter_map_t &values,
                                 QueryErrorContext &error_context, string &func_name) {
	for (auto &kv : values) {
		auto entry = types.find(kv.first);
		if (entry == types.end()) {
			// create a list of named parameters for the error
			string named_params;
			for (auto &kv : types) {
				named_params += "    ";
				named_params += kv.first;
				named_params += " ";
				named_params += kv.second.ToString();
				named_params += "\n";
			}
			string error_msg;
			if (named_params.empty()) {
				error_msg = "Function does not accept any named parameters.";
			} else {
				error_msg = "Candidates: " + named_params;
			}
			throw BinderException(error_context.FormatError("Invalid named parameter \"%s\" for function %s\n%s",
			                                                kv.first, func_name, error_msg));
		}
		if (entry->second.id() != LogicalTypeId::ANY) {
			kv.second = kv.second.DefaultCastAs(entry->second);
		}
	}
}

} // namespace duckdb




namespace duckdb {

unique_ptr<BoundTableRef> Binder::Bind(SubqueryRef &ref, CommonTableExpressionInfo *cte) {
	auto binder = Binder::CreateBinder(context, this);
	binder->can_contain_nulls = true;
	if (cte) {
		binder->bound_ctes.insert(cte);
	}
	binder->alias = ref.alias.empty() ? "unnamed_subquery" : ref.alias;
	auto subquery = binder->BindNode(*ref.subquery->node);
	idx_t bind_index = subquery->GetRootIndex();
	string alias;
	if (ref.alias.empty()) {
		alias = "unnamed_subquery" + to_string(bind_index);
	} else {
		alias = ref.alias;
	}
	auto result = make_unique<BoundSubqueryRef>(std::move(binder), std::move(subquery));
	bind_context.AddSubquery(bind_index, alias, ref, *result->subquery);
	MoveCorrelatedExpressions(*result->binder);
	return std::move(result);
}

} // namespace duckdb






















namespace duckdb {

static bool IsTableInTableOutFunction(TableFunctionCatalogEntry &table_function) {
	auto fun = table_function.functions.GetFunctionByOffset(0);
	return table_function.functions.Size() == 1 && fun.arguments.size() == 1 &&
	       fun.arguments[0].id() == LogicalTypeId::TABLE;
}

bool Binder::BindTableInTableOutFunction(vector<unique_ptr<ParsedExpression>> &expressions,
                                         unique_ptr<BoundSubqueryRef> &subquery, string &error) {
	auto binder = Binder::CreateBinder(this->context, this, true);
	unique_ptr<QueryNode> subquery_node;
	if (expressions.size() == 1 && expressions[0]->type == ExpressionType::SUBQUERY) {
		// general case: argument is a subquery, bind it as part of the node
		auto &se = (SubqueryExpression &)*expressions[0];
		subquery_node = std::move(se.subquery->node);
	} else {
		// special case: non-subquery parameter to table-in table-out function
		// generate a subquery and bind that (i.e. UNNEST([1,2,3]) becomes UNNEST((SELECT [1,2,3]))
		auto select_node = make_unique<SelectNode>();
		select_node->select_list = std::move(expressions);
		select_node->from_table = make_unique<EmptyTableRef>();
		subquery_node = std::move(select_node);
	}
	auto node = binder->BindNode(*subquery_node);
	subquery = make_unique<BoundSubqueryRef>(std::move(binder), std::move(node));
	MoveCorrelatedExpressions(*subquery->binder);
	return true;
}

bool Binder::BindTableFunctionParameters(TableFunctionCatalogEntry &table_function,
                                         vector<unique_ptr<ParsedExpression>> &expressions,
                                         vector<LogicalType> &arguments, vector<Value> &parameters,
                                         named_parameter_map_t &named_parameters,
                                         unique_ptr<BoundSubqueryRef> &subquery, string &error) {
	if (IsTableInTableOutFunction(table_function)) {
		// special case binding for table-in table-out function
		arguments.emplace_back(LogicalTypeId::TABLE);
		return BindTableInTableOutFunction(expressions, subquery, error);
	}
	bool seen_subquery = false;
	for (auto &child : expressions) {
		string parameter_name;

		// hack to make named parameters work
		if (child->type == ExpressionType::COMPARE_EQUAL) {
			// comparison, check if the LHS is a columnref
			auto &comp = (ComparisonExpression &)*child;
			if (comp.left->type == ExpressionType::COLUMN_REF) {
				auto &colref = (ColumnRefExpression &)*comp.left;
				if (!colref.IsQualified()) {
					parameter_name = colref.GetColumnName();
					child = std::move(comp.right);
				}
			}
		}
		if (child->type == ExpressionType::SUBQUERY) {
			if (seen_subquery) {
				error = "Table function can have at most one subquery parameter ";
				return false;
			}
			auto binder = Binder::CreateBinder(this->context, this, true);
			auto &se = (SubqueryExpression &)*child;
			auto node = binder->BindNode(*se.subquery->node);
			subquery = make_unique<BoundSubqueryRef>(std::move(binder), std::move(node));
			seen_subquery = true;
			arguments.emplace_back(LogicalTypeId::TABLE);
			continue;
		}

		TableFunctionBinder binder(*this, context);
		LogicalType sql_type;
		auto expr = binder.Bind(child, &sql_type);
		if (expr->HasParameter()) {
			throw ParameterNotResolvedException();
		}
		if (!expr->IsScalar()) {
			error = "Table function requires a constant parameter";
			return false;
		}
		auto constant = ExpressionExecutor::EvaluateScalar(context, *expr, true);
		if (parameter_name.empty()) {
			// unnamed parameter
			if (!named_parameters.empty()) {
				error = "Unnamed parameters cannot come after named parameters";
				return false;
			}
			arguments.emplace_back(sql_type);
			parameters.emplace_back(std::move(constant));
		} else {
			named_parameters[parameter_name] = std::move(constant);
		}
	}
	return true;
}

unique_ptr<LogicalOperator>
Binder::BindTableFunctionInternal(TableFunction &table_function, const string &function_name, vector<Value> parameters,
                                  named_parameter_map_t named_parameters, vector<LogicalType> input_table_types,
                                  vector<string> input_table_names, const vector<string> &column_name_alias,
                                  unique_ptr<ExternalDependency> external_dependency) {
	auto bind_index = GenerateTableIndex();
	// perform the binding
	unique_ptr<FunctionData> bind_data;
	vector<LogicalType> return_types;
	vector<string> return_names;
	if (table_function.bind) {
		TableFunctionBindInput bind_input(parameters, named_parameters, input_table_types, input_table_names,
		                                  table_function.function_info.get());
		bind_data = table_function.bind(context, bind_input, return_types, return_names);
		if (table_function.name == "pandas_scan" || table_function.name == "arrow_scan") {
			auto arrow_bind = (PyTableFunctionData *)bind_data.get();
			arrow_bind->external_dependency = std::move(external_dependency);
		}
	}
	if (return_types.size() != return_names.size()) {
		throw InternalException(
		    "Failed to bind \"%s\": Table function return_types and return_names must be of the same size",
		    table_function.name);
	}
	if (return_types.empty()) {
		throw InternalException("Failed to bind \"%s\": Table function must return at least one column",
		                        table_function.name);
	}
	// overwrite the names with any supplied aliases
	for (idx_t i = 0; i < column_name_alias.size() && i < return_names.size(); i++) {
		return_names[i] = column_name_alias[i];
	}
	for (idx_t i = 0; i < return_names.size(); i++) {
		if (return_names[i].empty()) {
			return_names[i] = "C" + to_string(i);
		}
	}
	auto get = make_unique<LogicalGet>(bind_index, table_function, std::move(bind_data), return_types, return_names);
	get->parameters = parameters;
	get->named_parameters = named_parameters;
	get->input_table_types = input_table_types;
	get->input_table_names = input_table_names;
	if (table_function.in_out_function && !table_function.projection_pushdown) {
		get->column_ids.reserve(return_types.size());
		for (idx_t i = 0; i < return_types.size(); i++) {
			get->column_ids.push_back(i);
		}
	}
	// now add the table function to the bind context so its columns can be bound
	bind_context.AddTableFunction(bind_index, function_name, return_names, return_types, get->column_ids,
	                              get->GetTable());
	return std::move(get);
}

unique_ptr<LogicalOperator> Binder::BindTableFunction(TableFunction &function, vector<Value> parameters) {
	named_parameter_map_t named_parameters;
	vector<LogicalType> input_table_types;
	vector<string> input_table_names;
	vector<string> column_name_aliases;
	return BindTableFunctionInternal(function, function.name, std::move(parameters), std::move(named_parameters),
	                                 std::move(input_table_types), std::move(input_table_names), column_name_aliases,
	                                 nullptr);
}

unique_ptr<BoundTableRef> Binder::Bind(TableFunctionRef &ref) {
	QueryErrorContext error_context(root_statement, ref.query_location);

	D_ASSERT(ref.function->type == ExpressionType::FUNCTION);
	auto fexpr = (FunctionExpression *)ref.function.get();

	TableFunctionCatalogEntry *function = nullptr;

	// fetch the function from the catalog
	auto func_catalog = Catalog::GetEntry(context, CatalogType::TABLE_FUNCTION_ENTRY, fexpr->catalog, fexpr->schema,
	                                      fexpr->function_name, false, error_context);

	if (func_catalog->type == CatalogType::TABLE_FUNCTION_ENTRY) {
		function = (TableFunctionCatalogEntry *)func_catalog;
	} else if (func_catalog->type == CatalogType::TABLE_MACRO_ENTRY) {
		auto macro_func = (TableMacroCatalogEntry *)func_catalog;
		auto query_node = BindTableMacro(*fexpr, macro_func, 0);
		D_ASSERT(query_node);

		auto binder = Binder::CreateBinder(context, this);
		binder->can_contain_nulls = true;

		binder->alias = ref.alias.empty() ? "unnamed_query" : ref.alias;
		auto query = binder->BindNode(*query_node);

		idx_t bind_index = query->GetRootIndex();
		// string alias;
		string alias = (ref.alias.empty() ? "unnamed_query" + to_string(bind_index) : ref.alias);

		auto result = make_unique<BoundSubqueryRef>(std::move(binder), std::move(query));
		// remember ref here is TableFunctionRef and NOT base class
		bind_context.AddSubquery(bind_index, alias, ref, *result->subquery);
		MoveCorrelatedExpressions(*result->binder);
		return std::move(result);
	}

	// evaluate the input parameters to the function
	vector<LogicalType> arguments;
	vector<Value> parameters;
	named_parameter_map_t named_parameters;
	unique_ptr<BoundSubqueryRef> subquery;
	string error;
	if (!BindTableFunctionParameters(*function, fexpr->children, arguments, parameters, named_parameters, subquery,
	                                 error)) {
		throw BinderException(FormatError(ref, error));
	}

	// select the function based on the input parameters
	FunctionBinder function_binder(context);
	idx_t best_function_idx = function_binder.BindFunction(function->name, function->functions, arguments, error);
	if (best_function_idx == DConstants::INVALID_INDEX) {
		throw BinderException(FormatError(ref, error));
	}
	auto table_function = function->functions.GetFunctionByOffset(best_function_idx);

	// now check the named parameters
	BindNamedParameters(table_function.named_parameters, named_parameters, error_context, table_function.name);

	// cast the parameters to the type of the function
	for (idx_t i = 0; i < arguments.size(); i++) {
		if (table_function.arguments[i] != LogicalType::ANY && table_function.arguments[i] != LogicalType::TABLE &&
		    table_function.arguments[i] != LogicalType::POINTER &&
		    table_function.arguments[i].id() != LogicalTypeId::LIST) {
			parameters[i] = parameters[i].CastAs(context, table_function.arguments[i]);
		}
	}

	vector<LogicalType> input_table_types;
	vector<string> input_table_names;

	if (subquery) {
		input_table_types = subquery->subquery->types;
		input_table_names = subquery->subquery->names;
	}
	auto get = BindTableFunctionInternal(table_function, ref.alias.empty() ? fexpr->function_name : ref.alias,
	                                     std::move(parameters), std::move(named_parameters),
	                                     std::move(input_table_types), std::move(input_table_names),
	                                     ref.column_name_alias, std::move(ref.external_dependency));
	if (subquery) {
		get->children.push_back(Binder::CreatePlan(*subquery));
	}

	return make_unique_base<BoundTableRef, BoundTableFunction>(std::move(get));
}

} // namespace duckdb




namespace duckdb {

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundBaseTableRef &ref) {
	return std::move(ref.get);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundCTERef &ref) {
	auto index = ref.bind_index;

	vector<LogicalType> types;
	for (auto &type : ref.types) {
		types.push_back(type);
	}

	return make_unique<LogicalCTERef>(index, ref.cte_index, types, ref.bound_columns);
}

} // namespace duckdb




namespace duckdb {

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundEmptyTableRef &ref) {
	return make_unique<LogicalDummyScan>(ref.bind_index);
}

} // namespace duckdb





namespace duckdb {

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundExpressionListRef &ref) {
	auto root = make_unique_base<LogicalOperator, LogicalDummyScan>(GenerateTableIndex());
	// values list, first plan any subqueries in the list
	for (auto &expr_list : ref.values) {
		for (auto &expr : expr_list) {
			PlanSubqueries(&expr, &root);
		}
	}
	// now create a LogicalExpressionGet from the set of expressions
	// fetch the types
	vector<LogicalType> types;
	for (auto &expr : ref.values[0]) {
		types.push_back(expr->return_type);
	}
	auto expr_get = make_unique<LogicalExpressionGet>(ref.bind_index, types, std::move(ref.values));
	expr_get->AddChild(std::move(root));
	return std::move(expr_get);
}

} // namespace duckdb

















namespace duckdb {

//! Create a JoinCondition from a comparison
static bool CreateJoinCondition(Expression &expr, const unordered_set<idx_t> &left_bindings,
                                const unordered_set<idx_t> &right_bindings, vector<JoinCondition> &conditions) {
	// comparison
	auto &comparison = (BoundComparisonExpression &)expr;
	auto left_side = JoinSide::GetJoinSide(*comparison.left, left_bindings, right_bindings);
	auto right_side = JoinSide::GetJoinSide(*comparison.right, left_bindings, right_bindings);
	if (left_side != JoinSide::BOTH && right_side != JoinSide::BOTH) {
		// join condition can be divided in a left/right side
		JoinCondition condition;
		condition.comparison = expr.type;
		auto left = std::move(comparison.left);
		auto right = std::move(comparison.right);
		if (left_side == JoinSide::RIGHT) {
			// left = right, right = left, flip the comparison symbol and reverse sides
			swap(left, right);
			condition.comparison = FlipComparisionExpression(expr.type);
		}
		condition.left = std::move(left);
		condition.right = std::move(right);
		conditions.push_back(std::move(condition));
		return true;
	}
	return false;
}

void LogicalComparisonJoin::ExtractJoinConditions(JoinType type, unique_ptr<LogicalOperator> &left_child,
                                                  unique_ptr<LogicalOperator> &right_child,
                                                  const unordered_set<idx_t> &left_bindings,
                                                  const unordered_set<idx_t> &right_bindings,
                                                  vector<unique_ptr<Expression>> &expressions,
                                                  vector<JoinCondition> &conditions,
                                                  vector<unique_ptr<Expression>> &arbitrary_expressions) {
	for (auto &expr : expressions) {
		auto total_side = JoinSide::GetJoinSide(*expr, left_bindings, right_bindings);
		if (total_side != JoinSide::BOTH) {
			// join condition does not reference both sides, add it as filter under the join
			if (type == JoinType::LEFT && total_side == JoinSide::RIGHT) {
				// filter is on RHS and the join is a LEFT OUTER join, we can push it in the right child
				if (right_child->type != LogicalOperatorType::LOGICAL_FILTER) {
					// not a filter yet, push a new empty filter
					auto filter = make_unique<LogicalFilter>();
					filter->AddChild(std::move(right_child));
					right_child = std::move(filter);
				}
				// push the expression into the filter
				auto &filter = (LogicalFilter &)*right_child;
				filter.expressions.push_back(std::move(expr));
				continue;
			}
		} else if ((expr->type >= ExpressionType::COMPARE_EQUAL &&
		            expr->type <= ExpressionType::COMPARE_GREATERTHANOREQUALTO) ||
		           expr->type == ExpressionType::COMPARE_DISTINCT_FROM ||
		           expr->type == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
			// comparison, check if we can create a comparison JoinCondition
			if (CreateJoinCondition(*expr, left_bindings, right_bindings, conditions)) {
				// successfully created the join condition
				continue;
			}
		}
		arbitrary_expressions.push_back(std::move(expr));
	}
}

void LogicalComparisonJoin::ExtractJoinConditions(JoinType type, unique_ptr<LogicalOperator> &left_child,
                                                  unique_ptr<LogicalOperator> &right_child,
                                                  vector<unique_ptr<Expression>> &expressions,
                                                  vector<JoinCondition> &conditions,
                                                  vector<unique_ptr<Expression>> &arbitrary_expressions) {
	unordered_set<idx_t> left_bindings, right_bindings;
	LogicalJoin::GetTableReferences(*left_child, left_bindings);
	LogicalJoin::GetTableReferences(*right_child, right_bindings);
	return ExtractJoinConditions(type, left_child, right_child, left_bindings, right_bindings, expressions, conditions,
	                             arbitrary_expressions);
}

void LogicalComparisonJoin::ExtractJoinConditions(JoinType type, unique_ptr<LogicalOperator> &left_child,
                                                  unique_ptr<LogicalOperator> &right_child,
                                                  unique_ptr<Expression> condition, vector<JoinCondition> &conditions,
                                                  vector<unique_ptr<Expression>> &arbitrary_expressions) {
	// split the expressions by the AND clause
	vector<unique_ptr<Expression>> expressions;
	expressions.push_back(std::move(condition));
	LogicalFilter::SplitPredicates(expressions);
	return ExtractJoinConditions(type, left_child, right_child, expressions, conditions, arbitrary_expressions);
}

unique_ptr<LogicalOperator> LogicalComparisonJoin::CreateJoin(JoinType type, unique_ptr<LogicalOperator> left_child,
                                                              unique_ptr<LogicalOperator> right_child,
                                                              vector<JoinCondition> conditions,
                                                              vector<unique_ptr<Expression>> arbitrary_expressions) {
	bool need_to_consider_arbitrary_expressions = true;
	if (type == JoinType::INNER) {
		// for inner joins we can push arbitrary expressions as a filter
		// here we prefer to create a comparison join if possible
		// that way we can use the much faster hash join to process the main join
		// rather than doing a nested loop join to handle arbitrary expressions

		// for left and full outer joins we HAVE to process all join conditions
		// because pushing a filter will lead to an incorrect result, as non-matching tuples cannot be filtered out
		need_to_consider_arbitrary_expressions = false;
	}
	if ((need_to_consider_arbitrary_expressions && !arbitrary_expressions.empty()) || conditions.empty()) {
		if (arbitrary_expressions.empty()) {
			// all conditions were pushed down, add TRUE predicate
			arbitrary_expressions.push_back(make_unique<BoundConstantExpression>(Value::BOOLEAN(true)));
		}
		for (auto &condition : conditions) {
			arbitrary_expressions.push_back(JoinCondition::CreateExpression(std::move(condition)));
		}
		// if we get here we could not create any JoinConditions
		// turn this into an arbitrary expression join
		auto any_join = make_unique<LogicalAnyJoin>(type);
		// create the condition
		any_join->children.push_back(std::move(left_child));
		any_join->children.push_back(std::move(right_child));
		// AND all the arbitrary expressions together
		// do the same with any remaining conditions
		any_join->condition = std::move(arbitrary_expressions[0]);
		for (idx_t i = 1; i < arbitrary_expressions.size(); i++) {
			any_join->condition = make_unique<BoundConjunctionExpression>(
			    ExpressionType::CONJUNCTION_AND, std::move(any_join->condition), std::move(arbitrary_expressions[i]));
		}
		return std::move(any_join);
	} else {
		// we successfully converted expressions into JoinConditions
		// create a LogicalComparisonJoin
		auto comp_join = make_unique<LogicalComparisonJoin>(type);
		comp_join->conditions = std::move(conditions);
		comp_join->children.push_back(std::move(left_child));
		comp_join->children.push_back(std::move(right_child));
		if (!arbitrary_expressions.empty()) {
			// we have some arbitrary expressions as well
			// add them to a filter
			auto filter = make_unique<LogicalFilter>();
			for (auto &expr : arbitrary_expressions) {
				filter->expressions.push_back(std::move(expr));
			}
			LogicalFilter::SplitPredicates(filter->expressions);
			filter->children.push_back(std::move(comp_join));
			return std::move(filter);
		}
		return std::move(comp_join);
	}
}

static bool HasCorrelatedColumns(Expression &expression) {
	if (expression.type == ExpressionType::BOUND_COLUMN_REF) {
		auto &colref = (BoundColumnRefExpression &)expression;
		if (colref.depth > 0) {
			return true;
		}
	}
	bool has_correlated_columns = false;
	ExpressionIterator::EnumerateChildren(expression, [&](Expression &child) {
		if (HasCorrelatedColumns(child)) {
			has_correlated_columns = true;
		}
	});
	return has_correlated_columns;
}

unique_ptr<LogicalOperator> LogicalComparisonJoin::CreateJoin(JoinType type, unique_ptr<LogicalOperator> left_child,
                                                              unique_ptr<LogicalOperator> right_child,
                                                              unique_ptr<Expression> condition) {
	vector<JoinCondition> conditions;
	vector<unique_ptr<Expression>> arbitrary_expressions;
	LogicalComparisonJoin::ExtractJoinConditions(type, left_child, right_child, std::move(condition), conditions,
	                                             arbitrary_expressions);
	return LogicalComparisonJoin::CreateJoin(type, std::move(left_child), std::move(right_child), std::move(conditions),
	                                         std::move(arbitrary_expressions));
}

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundJoinRef &ref) {
	auto left = CreatePlan(*ref.left);
	auto right = CreatePlan(*ref.right);
	if (!ref.lateral && !ref.correlated_columns.empty()) {
		// non-lateral join with correlated columns
		// this happens if there is a join (or cross product) in a correlated subquery
		// due to the lateral binder the expression depth of all correlated columns in the "ref.correlated_columns" set
		// is 1 too high
		// we reduce expression depth of all columns in the "ref.correlated_columns" set by 1
		LateralBinder::ReduceExpressionDepth(*right, ref.correlated_columns);
	}
	if (ref.type == JoinType::RIGHT && ClientConfig::GetConfig(context).enable_optimizer) {
		// we turn any right outer joins into left outer joins for optimization purposes
		// they are the same but with sides flipped, so treating them the same simplifies life
		ref.type = JoinType::LEFT;
		std::swap(left, right);
	}
	if (ref.lateral) {
		// lateral join
		return PlanLateralJoin(std::move(left), std::move(right), ref.correlated_columns, ref.type,
		                       std::move(ref.condition));
	}
	switch (ref.ref_type) {
	case JoinRefType::CROSS:
		return LogicalCrossProduct::Create(std::move(left), std::move(right));
	case JoinRefType::POSITIONAL:
		return LogicalPositionalJoin::Create(std::move(left), std::move(right));
	default:
		break;
	}
	if (ref.type == JoinType::INNER && (ref.condition->HasSubquery() || HasCorrelatedColumns(*ref.condition))) {
		// inner join, generate a cross product + filter
		// this will be later turned into a proper join by the join order optimizer
		auto root = LogicalCrossProduct::Create(std::move(left), std::move(right));

		auto filter = make_unique<LogicalFilter>(std::move(ref.condition));
		// visit the expressions in the filter
		for (auto &expression : filter->expressions) {
			PlanSubqueries(&expression, &root);
		}
		filter->AddChild(std::move(root));
		return std::move(filter);
	}

	// now create the join operator from the join condition
	auto result =
	    LogicalComparisonJoin::CreateJoin(ref.type, std::move(left), std::move(right), std::move(ref.condition));

	LogicalOperator *join;
	if (result->type == LogicalOperatorType::LOGICAL_FILTER) {
		join = result->children[0].get();
	} else {
		join = result.get();
	}
	for (auto &child : join->children) {
		if (child->type == LogicalOperatorType::LOGICAL_FILTER) {
			auto &filter = (LogicalFilter &)*child;
			for (auto &expr : filter.expressions) {
				PlanSubqueries(&expr, &filter.children[0]);
			}
		}
	}

	// we visit the expressions depending on the type of join
	if (join->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		// comparison join
		// in this join we visit the expressions on the LHS with the LHS as root node
		// and the expressions on the RHS with the RHS as root node
		auto &comp_join = (LogicalComparisonJoin &)*join;
		for (idx_t i = 0; i < comp_join.conditions.size(); i++) {
			PlanSubqueries(&comp_join.conditions[i].left, &comp_join.children[0]);
			PlanSubqueries(&comp_join.conditions[i].right, &comp_join.children[1]);
		}
	} else if (join->type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
		auto &any_join = (LogicalAnyJoin &)*join;
		// for the any join we just visit the condition
		if (any_join.condition->HasSubquery()) {
			throw NotImplementedException("Cannot perform non-inner join on subquery!");
		}
	}
	return result;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundSubqueryRef &ref) {
	// generate the logical plan for the subquery
	// this happens separately from the current LogicalPlan generation
	ref.binder->plan_subquery = plan_subquery;
	auto subquery = ref.binder->CreatePlan(*ref.subquery);
	if (ref.binder->has_unplanned_subqueries) {
		has_unplanned_subqueries = true;
	}
	return subquery;
}

} // namespace duckdb



namespace duckdb {

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundTableFunction &ref) {
	return std::move(ref.get);
}

} // namespace duckdb

















#include <algorithm>

namespace duckdb {

shared_ptr<Binder> Binder::CreateBinder(ClientContext &context, Binder *parent, bool inherit_ctes) {
	return make_shared<Binder>(true, context, parent ? parent->shared_from_this() : nullptr, inherit_ctes);
}

Binder::Binder(bool, ClientContext &context, shared_ptr<Binder> parent_p, bool inherit_ctes_p)
    : context(context), parent(std::move(parent_p)), bound_tables(0), inherit_ctes(inherit_ctes_p) {
	parameters = nullptr;
	if (parent) {

		// We have to inherit macro and lambda parameter bindings and from the parent binder, if there is a parent.
		macro_binding = parent->macro_binding;
		lambda_bindings = parent->lambda_bindings;

		if (inherit_ctes) {
			// We have to inherit CTE bindings from the parent bind_context, if there is a parent.
			bind_context.SetCTEBindings(parent->bind_context.GetCTEBindings());
			bind_context.cte_references = parent->bind_context.cte_references;
			parameters = parent->parameters;
		}
	}
}

BoundStatement Binder::Bind(SQLStatement &statement) {
	root_statement = &statement;
	switch (statement.type) {
	case StatementType::SELECT_STATEMENT:
		return Bind((SelectStatement &)statement);
	case StatementType::INSERT_STATEMENT:
		return Bind((InsertStatement &)statement);
	case StatementType::COPY_STATEMENT:
		return Bind((CopyStatement &)statement);
	case StatementType::DELETE_STATEMENT:
		return Bind((DeleteStatement &)statement);
	case StatementType::UPDATE_STATEMENT:
		return Bind((UpdateStatement &)statement);
	case StatementType::RELATION_STATEMENT:
		return Bind((RelationStatement &)statement);
	case StatementType::CREATE_STATEMENT:
		return Bind((CreateStatement &)statement);
	case StatementType::DROP_STATEMENT:
		return Bind((DropStatement &)statement);
	case StatementType::ALTER_STATEMENT:
		return Bind((AlterStatement &)statement);
	case StatementType::TRANSACTION_STATEMENT:
		return Bind((TransactionStatement &)statement);
	case StatementType::PRAGMA_STATEMENT:
		return Bind((PragmaStatement &)statement);
	case StatementType::EXPLAIN_STATEMENT:
		return Bind((ExplainStatement &)statement);
	case StatementType::VACUUM_STATEMENT:
		return Bind((VacuumStatement &)statement);
	case StatementType::SHOW_STATEMENT:
		return Bind((ShowStatement &)statement);
	case StatementType::CALL_STATEMENT:
		return Bind((CallStatement &)statement);
	case StatementType::EXPORT_STATEMENT:
		return Bind((ExportStatement &)statement);
	case StatementType::SET_STATEMENT:
		return Bind((SetStatement &)statement);
	case StatementType::LOAD_STATEMENT:
		return Bind((LoadStatement &)statement);
	case StatementType::EXTENSION_STATEMENT:
		return Bind((ExtensionStatement &)statement);
	case StatementType::PREPARE_STATEMENT:
		return Bind((PrepareStatement &)statement);
	case StatementType::EXECUTE_STATEMENT:
		return Bind((ExecuteStatement &)statement);
	case StatementType::LOGICAL_PLAN_STATEMENT:
		return Bind((LogicalPlanStatement &)statement);
	case StatementType::ATTACH_STATEMENT:
		return Bind((AttachStatement &)statement);
	default: // LCOV_EXCL_START
		throw NotImplementedException("Unimplemented statement type \"%s\" for Bind",
		                              StatementTypeToString(statement.type));
	} // LCOV_EXCL_STOP
}

void Binder::AddCTEMap(CommonTableExpressionMap &cte_map) {
	for (auto &cte_it : cte_map.map) {
		AddCTE(cte_it.first, cte_it.second.get());
	}
}

unique_ptr<BoundQueryNode> Binder::BindNode(QueryNode &node) {
	// first we visit the set of CTEs and add them to the bind context
	AddCTEMap(node.cte_map);
	// now we bind the node
	unique_ptr<BoundQueryNode> result;
	switch (node.type) {
	case QueryNodeType::SELECT_NODE:
		result = BindNode((SelectNode &)node);
		break;
	case QueryNodeType::RECURSIVE_CTE_NODE:
		result = BindNode((RecursiveCTENode &)node);
		break;
	default:
		D_ASSERT(node.type == QueryNodeType::SET_OPERATION_NODE);
		result = BindNode((SetOperationNode &)node);
		break;
	}
	return result;
}

BoundStatement Binder::Bind(QueryNode &node) {
	auto bound_node = BindNode(node);

	BoundStatement result;
	result.names = bound_node->names;
	result.types = bound_node->types;

	// and plan it
	result.plan = CreatePlan(*bound_node);
	return result;
}

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundQueryNode &node) {
	switch (node.type) {
	case QueryNodeType::SELECT_NODE:
		return CreatePlan((BoundSelectNode &)node);
	case QueryNodeType::SET_OPERATION_NODE:
		return CreatePlan((BoundSetOperationNode &)node);
	case QueryNodeType::RECURSIVE_CTE_NODE:
		return CreatePlan((BoundRecursiveCTENode &)node);
	default:
		throw InternalException("Unsupported bound query node type");
	}
}

unique_ptr<BoundTableRef> Binder::Bind(TableRef &ref) {
	unique_ptr<BoundTableRef> result;
	switch (ref.type) {
	case TableReferenceType::BASE_TABLE:
		result = Bind((BaseTableRef &)ref);
		break;
	case TableReferenceType::JOIN:
		result = Bind((JoinRef &)ref);
		break;
	case TableReferenceType::SUBQUERY:
		result = Bind((SubqueryRef &)ref);
		break;
	case TableReferenceType::EMPTY:
		result = Bind((EmptyTableRef &)ref);
		break;
	case TableReferenceType::TABLE_FUNCTION:
		result = Bind((TableFunctionRef &)ref);
		break;
	case TableReferenceType::EXPRESSION_LIST:
		result = Bind((ExpressionListRef &)ref);
		break;
	case TableReferenceType::CTE:
	case TableReferenceType::INVALID:
		throw InternalException("Unknown table ref type");
	}
	result->sample = std::move(ref.sample);
	return result;
}

unique_ptr<LogicalOperator> Binder::CreatePlan(BoundTableRef &ref) {
	unique_ptr<LogicalOperator> root;
	switch (ref.type) {
	case TableReferenceType::BASE_TABLE:
		root = CreatePlan((BoundBaseTableRef &)ref);
		break;
	case TableReferenceType::SUBQUERY:
		root = CreatePlan((BoundSubqueryRef &)ref);
		break;
	case TableReferenceType::JOIN:
		root = CreatePlan((BoundJoinRef &)ref);
		break;
	case TableReferenceType::TABLE_FUNCTION:
		root = CreatePlan((BoundTableFunction &)ref);
		break;
	case TableReferenceType::EMPTY:
		root = CreatePlan((BoundEmptyTableRef &)ref);
		break;
	case TableReferenceType::EXPRESSION_LIST:
		root = CreatePlan((BoundExpressionListRef &)ref);
		break;
	case TableReferenceType::CTE:
		root = CreatePlan((BoundCTERef &)ref);
		break;
	case TableReferenceType::INVALID:
		throw InternalException("Unsupported bound table ref type");
	}
	// plan the sample clause
	if (ref.sample) {
		root = make_unique<LogicalSample>(std::move(ref.sample), std::move(root));
	}
	return root;
}

void Binder::AddCTE(const string &name, CommonTableExpressionInfo *info) {
	D_ASSERT(info);
	D_ASSERT(!name.empty());
	auto entry = CTE_bindings.find(name);
	if (entry != CTE_bindings.end()) {
		throw InternalException("Duplicate CTE \"%s\" in query!", name);
	}
	CTE_bindings[name] = info;
}

CommonTableExpressionInfo *Binder::FindCTE(const string &name, bool skip) {
	auto entry = CTE_bindings.find(name);
	if (entry != CTE_bindings.end()) {
		if (!skip || entry->second->query->node->type == QueryNodeType::RECURSIVE_CTE_NODE) {
			return entry->second;
		}
	}
	if (parent && inherit_ctes) {
		return parent->FindCTE(name, name == alias);
	}
	return nullptr;
}

bool Binder::CTEIsAlreadyBound(CommonTableExpressionInfo *cte) {
	if (bound_ctes.find(cte) != bound_ctes.end()) {
		return true;
	}
	if (parent && inherit_ctes) {
		return parent->CTEIsAlreadyBound(cte);
	}
	return false;
}

void Binder::AddBoundView(ViewCatalogEntry *view) {
	// check if the view is already bound
	auto current = this;
	while (current) {
		if (current->bound_views.find(view) != current->bound_views.end()) {
			throw BinderException("infinite recursion detected: attempting to recursively bind view \"%s\"",
			                      view->name);
		}
		current = current->parent.get();
	}
	bound_views.insert(view);
}

idx_t Binder::GenerateTableIndex() {
	D_ASSERT(parent.get() != this);
	if (parent) {
		return parent->GenerateTableIndex();
	}
	return bound_tables++;
}

void Binder::PushExpressionBinder(ExpressionBinder *binder) {
	GetActiveBinders().push_back(binder);
}

void Binder::PopExpressionBinder() {
	D_ASSERT(HasActiveBinder());
	GetActiveBinders().pop_back();
}

void Binder::SetActiveBinder(ExpressionBinder *binder) {
	D_ASSERT(HasActiveBinder());
	GetActiveBinders().back() = binder;
}

ExpressionBinder *Binder::GetActiveBinder() {
	return GetActiveBinders().back();
}

bool Binder::HasActiveBinder() {
	return !GetActiveBinders().empty();
}

vector<ExpressionBinder *> &Binder::GetActiveBinders() {
	if (parent) {
		return parent->GetActiveBinders();
	}
	return active_binders;
}

void Binder::AddUsingBindingSet(unique_ptr<UsingColumnSet> set) {
	if (parent) {
		parent->AddUsingBindingSet(std::move(set));
		return;
	}
	bind_context.AddUsingBindingSet(std::move(set));
}

void Binder::MoveCorrelatedExpressions(Binder &other) {
	MergeCorrelatedColumns(other.correlated_columns);
	other.correlated_columns.clear();
}

void Binder::MergeCorrelatedColumns(vector<CorrelatedColumnInfo> &other) {
	for (idx_t i = 0; i < other.size(); i++) {
		AddCorrelatedColumn(other[i]);
	}
}

void Binder::AddCorrelatedColumn(const CorrelatedColumnInfo &info) {
	// we only add correlated columns to the list if they are not already there
	if (std::find(correlated_columns.begin(), correlated_columns.end(), info) == correlated_columns.end()) {
		correlated_columns.push_back(info);
	}
}

bool Binder::HasMatchingBinding(const string &table_name, const string &column_name, string &error_message) {
	string empty_schema;
	return HasMatchingBinding(empty_schema, table_name, column_name, error_message);
}

bool Binder::HasMatchingBinding(const string &schema_name, const string &table_name, const string &column_name,
                                string &error_message) {
	string empty_catalog;
	return HasMatchingBinding(empty_catalog, schema_name, table_name, column_name, error_message);
}

bool Binder::HasMatchingBinding(const string &catalog_name, const string &schema_name, const string &table_name,
                                const string &column_name, string &error_message) {
	Binding *binding = nullptr;
	D_ASSERT(!lambda_bindings);
	if (macro_binding && table_name == macro_binding->alias) {
		binding = macro_binding;
	} else {
		binding = bind_context.GetBinding(table_name, error_message);
	}

	if (!binding) {
		return false;
	}
	if (!catalog_name.empty() || !schema_name.empty()) {
		auto catalog_entry = binding->GetStandardEntry();
		if (!catalog_entry) {
			return false;
		}
		if (!catalog_name.empty() && catalog_entry->catalog->GetName() != catalog_name) {
			return false;
		}
		if (!schema_name.empty() && catalog_entry->schema->name != schema_name) {
			return false;
		}
		if (catalog_entry->name != table_name) {
			return false;
		}
	}
	bool binding_found;
	binding_found = binding->HasMatchingBinding(column_name);
	if (!binding_found) {
		error_message = binding->ColumnNotFoundError(column_name);
	}
	return binding_found;
}

void Binder::SetBindingMode(BindingMode mode) {
	if (parent) {
		parent->SetBindingMode(mode);
	}
	this->mode = mode;
}

BindingMode Binder::GetBindingMode() {
	if (parent) {
		return parent->GetBindingMode();
	}
	return mode;
}

void Binder::SetCanContainNulls(bool can_contain_nulls_p) {
	can_contain_nulls = can_contain_nulls_p;
}

void Binder::AddTableName(string table_name) {
	if (parent) {
		parent->AddTableName(std::move(table_name));
		return;
	}
	table_names.insert(std::move(table_name));
}

const unordered_set<string> &Binder::GetTableNames() {
	if (parent) {
		return parent->GetTableNames();
	}
	return table_names;
}

string Binder::FormatError(ParsedExpression &expr_context, const string &message) {
	return FormatError(expr_context.query_location, message);
}

string Binder::FormatError(TableRef &ref_context, const string &message) {
	return FormatError(ref_context.query_location, message);
}

string Binder::FormatErrorRecursive(idx_t query_location, const string &message, vector<ExceptionFormatValue> &values) {
	QueryErrorContext context(root_statement, query_location);
	return context.FormatErrorRecursive(message, values);
}

// FIXME: this is extremely naive
void VerifyNotExcluded(ParsedExpression &expr) {
	if (expr.type == ExpressionType::COLUMN_REF) {
		auto &column_ref = (ColumnRefExpression &)expr;
		if (!column_ref.IsQualified()) {
			return;
		}
		auto &table_name = column_ref.GetTableName();
		if (table_name == "excluded") {
			throw NotImplementedException("'excluded' qualified columns are not supported in the RETURNING clause yet");
		}
		return;
	}
	ParsedExpressionIterator::EnumerateChildren(
	    expr, [&](const ParsedExpression &child) { VerifyNotExcluded((ParsedExpression &)child); });
}

BoundStatement Binder::BindReturning(vector<unique_ptr<ParsedExpression>> returning_list, TableCatalogEntry *table,
                                     idx_t update_table_index, unique_ptr<LogicalOperator> child_operator,
                                     BoundStatement result) {

	vector<LogicalType> types;
	vector<std::string> names;

	auto binder = Binder::CreateBinder(context);

	vector<column_t> bound_columns;
	idx_t column_count = 0;
	for (auto &col : table->GetColumns().Logical()) {
		names.push_back(col.Name());
		types.push_back(col.Type());
		if (!col.Generated()) {
			bound_columns.push_back(column_count);
		}
		column_count++;
	}

	binder->bind_context.AddBaseTable(update_table_index, table->name, names, types, bound_columns, table, false);
	ReturningBinder returning_binder(*binder, context);

	vector<unique_ptr<Expression>> projection_expressions;
	LogicalType result_type;
	for (auto &returning_expr : returning_list) {
		auto expr_type = returning_expr->GetExpressionType();
		if (expr_type == ExpressionType::STAR) {
			auto generated_star_list = vector<unique_ptr<ParsedExpression>>();
			binder->bind_context.GenerateAllColumnExpressions((StarExpression &)*returning_expr, generated_star_list);

			for (auto &star_column : generated_star_list) {
				auto star_expr = returning_binder.Bind(star_column, &result_type);
				result.types.push_back(result_type);
				result.names.push_back(star_expr->GetName());
				projection_expressions.push_back(std::move(star_expr));
			}
		} else {
			// TODO: accept 'excluded' in the RETURNING clause
			VerifyNotExcluded(*returning_expr);
			auto expr = returning_binder.Bind(returning_expr, &result_type);
			result.names.push_back(expr->GetName());
			result.types.push_back(result_type);
			projection_expressions.push_back(std::move(expr));
		}
	}

	auto projection = make_unique<LogicalProjection>(GenerateTableIndex(), std::move(projection_expressions));
	projection->AddChild(std::move(child_operator));
	D_ASSERT(result.types.size() == result.names.size());
	result.plan = std::move(projection);
	properties.allow_stream_result = true;
	properties.return_type = StatementReturnType::QUERY_RESULT;
	return result;
}

} // namespace duckdb



namespace duckdb {

BoundResultModifier::BoundResultModifier(ResultModifierType type) : type(type) {
}

BoundResultModifier::~BoundResultModifier() {
}

BoundOrderByNode::BoundOrderByNode(OrderType type, OrderByNullType null_order, unique_ptr<Expression> expression)
    : type(type), null_order(null_order), expression(std::move(expression)) {
}
BoundOrderByNode::BoundOrderByNode(OrderType type, OrderByNullType null_order, unique_ptr<Expression> expression,
                                   unique_ptr<BaseStatistics> stats)
    : type(type), null_order(null_order), expression(std::move(expression)), stats(std::move(stats)) {
}

BoundOrderByNode BoundOrderByNode::Copy() const {
	if (stats) {
		return BoundOrderByNode(type, null_order, expression->Copy(), stats->Copy());
	} else {
		return BoundOrderByNode(type, null_order, expression->Copy());
	}
}

bool BoundOrderByNode::Equals(const BoundOrderByNode &other) const {
	if (type != other.type || null_order != other.null_order) {
		return false;
	}
	if (!expression->Equals(other.expression.get())) {
		return false;
	}

	return true;
}

string BoundOrderByNode::ToString() const {
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

void BoundOrderByNode::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteField(type);
	writer.WriteField(null_order);
	writer.WriteSerializable(*expression);
	// TODO statistics
	writer.Finalize();
}

BoundOrderByNode BoundOrderByNode::Deserialize(Deserializer &source, PlanDeserializationState &state) {
	FieldReader reader(source);
	auto type = reader.ReadRequired<OrderType>();
	auto null_order = reader.ReadRequired<OrderByNullType>();
	auto expression = reader.ReadRequiredSerializable<Expression>(state);
	reader.Finalize();
	return BoundOrderByNode(type, null_order, std::move(expression));
}

BoundLimitModifier::BoundLimitModifier() : BoundResultModifier(ResultModifierType::LIMIT_MODIFIER) {
}

BoundOrderModifier::BoundOrderModifier() : BoundResultModifier(ResultModifierType::ORDER_MODIFIER) {
}

BoundDistinctModifier::BoundDistinctModifier() : BoundResultModifier(ResultModifierType::DISTINCT_MODIFIER) {
}

BoundLimitPercentModifier::BoundLimitPercentModifier()
    : BoundResultModifier(ResultModifierType::LIMIT_PERCENT_MODIFIER) {
}

} // namespace duckdb









namespace duckdb {

BoundAggregateExpression::BoundAggregateExpression(AggregateFunction function, vector<unique_ptr<Expression>> children,
                                                   unique_ptr<Expression> filter, unique_ptr<FunctionData> bind_info,
                                                   AggregateType aggr_type)
    : Expression(ExpressionType::BOUND_AGGREGATE, ExpressionClass::BOUND_AGGREGATE, function.return_type),
      function(std::move(function)), children(std::move(children)), bind_info(std::move(bind_info)),
      aggr_type(aggr_type), filter(std::move(filter)) {
	D_ASSERT(!function.name.empty());
}

string BoundAggregateExpression::ToString() const {
	return FunctionExpression::ToString<BoundAggregateExpression, Expression>(*this, string(), function.name, false,
	                                                                          IsDistinct(), filter.get());
}

hash_t BoundAggregateExpression::Hash() const {
	hash_t result = Expression::Hash();
	result = CombineHash(result, function.Hash());
	result = CombineHash(result, duckdb::Hash(IsDistinct()));
	return result;
}

bool BoundAggregateExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto other = (BoundAggregateExpression *)other_p;
	if (other->aggr_type != aggr_type) {
		return false;
	}
	if (other->function != function) {
		return false;
	}
	if (children.size() != other->children.size()) {
		return false;
	}
	if (!Expression::Equals(other->filter.get(), filter.get())) {
		return false;
	}
	for (idx_t i = 0; i < children.size(); i++) {
		if (!Expression::Equals(children[i].get(), other->children[i].get())) {
			return false;
		}
	}
	if (!FunctionData::Equals(bind_info.get(), other->bind_info.get())) {
		return false;
	}
	return true;
}

bool BoundAggregateExpression::PropagatesNullValues() const {
	return function.null_handling == FunctionNullHandling::SPECIAL_HANDLING ? false
	                                                                        : Expression::PropagatesNullValues();
}

unique_ptr<Expression> BoundAggregateExpression::Copy() {
	vector<unique_ptr<Expression>> new_children;
	new_children.reserve(children.size());
	for (auto &child : children) {
		new_children.push_back(child->Copy());
	}
	auto new_bind_info = bind_info ? bind_info->Copy() : nullptr;
	auto new_filter = filter ? filter->Copy() : nullptr;
	auto copy = make_unique<BoundAggregateExpression>(function, std::move(new_children), std::move(new_filter),
	                                                  std::move(new_bind_info), aggr_type);
	copy->CopyProperties(*this);
	return std::move(copy);
}

void BoundAggregateExpression::Serialize(FieldWriter &writer) const {
	writer.WriteField(IsDistinct());
	writer.WriteOptional(filter);
	FunctionSerializer::Serialize<AggregateFunction>(writer, function, return_type, children, bind_info.get());
}

unique_ptr<Expression> BoundAggregateExpression::Deserialize(ExpressionDeserializationState &state,
                                                             FieldReader &reader) {
	auto distinct = reader.ReadRequired<bool>();
	auto filter = reader.ReadOptional<Expression>(nullptr, state.gstate);
	vector<unique_ptr<Expression>> children;
	unique_ptr<FunctionData> bind_info;
	auto function = FunctionSerializer::Deserialize<AggregateFunction, AggregateFunctionCatalogEntry>(
	    reader, state, CatalogType::AGGREGATE_FUNCTION_ENTRY, children, bind_info);

	return make_unique<BoundAggregateExpression>(function, std::move(children), std::move(filter), std::move(bind_info),
	                                             distinct ? AggregateType::DISTINCT : AggregateType::NON_DISTINCT);
}

} // namespace duckdb




namespace duckdb {

BoundBetweenExpression::BoundBetweenExpression(unique_ptr<Expression> input, unique_ptr<Expression> lower,
                                               unique_ptr<Expression> upper, bool lower_inclusive, bool upper_inclusive)
    : Expression(ExpressionType::COMPARE_BETWEEN, ExpressionClass::BOUND_BETWEEN, LogicalType::BOOLEAN),
      input(std::move(input)), lower(std::move(lower)), upper(std::move(upper)), lower_inclusive(lower_inclusive),
      upper_inclusive(upper_inclusive) {
}

string BoundBetweenExpression::ToString() const {
	return BetweenExpression::ToString<BoundBetweenExpression, Expression>(*this);
}

bool BoundBetweenExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto other = (BoundBetweenExpression *)other_p;
	if (!Expression::Equals(input.get(), other->input.get())) {
		return false;
	}
	if (!Expression::Equals(lower.get(), other->lower.get())) {
		return false;
	}
	if (!Expression::Equals(upper.get(), other->upper.get())) {
		return false;
	}
	return lower_inclusive == other->lower_inclusive && upper_inclusive == other->upper_inclusive;
}

unique_ptr<Expression> BoundBetweenExpression::Copy() {
	auto copy = make_unique<BoundBetweenExpression>(input->Copy(), lower->Copy(), upper->Copy(), lower_inclusive,
	                                                upper_inclusive);
	copy->CopyProperties(*this);
	return std::move(copy);
}

void BoundBetweenExpression::Serialize(FieldWriter &writer) const {
	writer.WriteOptional(input);
	writer.WriteOptional(lower);
	writer.WriteOptional(upper);
	writer.WriteField(lower_inclusive);
	writer.WriteField(upper_inclusive);
}

unique_ptr<Expression> BoundBetweenExpression::Deserialize(ExpressionDeserializationState &state, FieldReader &reader) {
	auto input = reader.ReadOptional<Expression>(nullptr, state.gstate);
	auto lower = reader.ReadOptional<Expression>(nullptr, state.gstate);
	auto upper = reader.ReadOptional<Expression>(nullptr, state.gstate);
	auto lower_inclusive = reader.ReadRequired<bool>();
	auto upper_inclusive = reader.ReadRequired<bool>();
	return make_unique<BoundBetweenExpression>(std::move(input), std::move(lower), std::move(upper), lower_inclusive,
	                                           upper_inclusive);
}

} // namespace duckdb




namespace duckdb {

BoundCaseExpression::BoundCaseExpression(LogicalType type)
    : Expression(ExpressionType::CASE_EXPR, ExpressionClass::BOUND_CASE, std::move(type)) {
}

BoundCaseExpression::BoundCaseExpression(unique_ptr<Expression> when_expr, unique_ptr<Expression> then_expr,
                                         unique_ptr<Expression> else_expr_p)
    : Expression(ExpressionType::CASE_EXPR, ExpressionClass::BOUND_CASE, then_expr->return_type),
      else_expr(std::move(else_expr_p)) {
	BoundCaseCheck check;
	check.when_expr = std::move(when_expr);
	check.then_expr = std::move(then_expr);
	case_checks.push_back(std::move(check));
}

string BoundCaseExpression::ToString() const {
	return CaseExpression::ToString<BoundCaseExpression, Expression>(*this);
}

bool BoundCaseExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto &other = (BoundCaseExpression &)*other_p;
	if (case_checks.size() != other.case_checks.size()) {
		return false;
	}
	for (idx_t i = 0; i < case_checks.size(); i++) {
		if (!Expression::Equals(case_checks[i].when_expr.get(), other.case_checks[i].when_expr.get())) {
			return false;
		}
		if (!Expression::Equals(case_checks[i].then_expr.get(), other.case_checks[i].then_expr.get())) {
			return false;
		}
	}
	if (!Expression::Equals(else_expr.get(), other.else_expr.get())) {
		return false;
	}
	return true;
}

unique_ptr<Expression> BoundCaseExpression::Copy() {
	auto new_case = make_unique<BoundCaseExpression>(return_type);
	for (auto &check : case_checks) {
		BoundCaseCheck new_check;
		new_check.when_expr = check.when_expr->Copy();
		new_check.then_expr = check.then_expr->Copy();
		new_case->case_checks.push_back(std::move(new_check));
	}
	new_case->else_expr = else_expr->Copy();

	new_case->CopyProperties(*this);
	return std::move(new_case);
}

void BoundCaseCheck::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteSerializable(*when_expr);
	writer.WriteSerializable(*then_expr);
	writer.Finalize();
}

BoundCaseCheck BoundCaseCheck::Deserialize(Deserializer &source, PlanDeserializationState &state) {
	FieldReader reader(source);
	auto when_expr = reader.ReadRequiredSerializable<Expression>(state);
	auto then_expr = reader.ReadRequiredSerializable<Expression>(state);
	reader.Finalize();
	BoundCaseCheck result;
	result.when_expr = std::move(when_expr);
	result.then_expr = std::move(then_expr);
	return result;
}

void BoundCaseExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(return_type);
	writer.WriteRegularSerializableList(case_checks);
	writer.WriteSerializable(*else_expr);
}

unique_ptr<Expression> BoundCaseExpression::Deserialize(ExpressionDeserializationState &state, FieldReader &reader) {
	auto return_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto case_checks = reader.ReadRequiredSerializableList<BoundCaseCheck, BoundCaseCheck>(state.gstate);
	auto else_expr = reader.ReadRequiredSerializable<Expression>(state.gstate);

	auto result = make_unique<BoundCaseExpression>(return_type);
	result->else_expr = std::move(else_expr);
	result->case_checks = std::move(case_checks);
	return std::move(result);
}

} // namespace duckdb








namespace duckdb {

BoundCastExpression::BoundCastExpression(unique_ptr<Expression> child_p, LogicalType target_type_p,
                                         BoundCastInfo bound_cast_p, bool try_cast_p)
    : Expression(ExpressionType::OPERATOR_CAST, ExpressionClass::BOUND_CAST, std::move(target_type_p)),
      child(std::move(child_p)), try_cast(try_cast_p), bound_cast(std::move(bound_cast_p)) {
}

unique_ptr<Expression> AddCastExpressionInternal(unique_ptr<Expression> expr, const LogicalType &target_type,
                                                 BoundCastInfo bound_cast, bool try_cast) {
	if (expr->return_type == target_type) {
		return expr;
	}
	auto &expr_type = expr->return_type;
	if (target_type.id() == LogicalTypeId::LIST && expr_type.id() == LogicalTypeId::LIST) {
		auto &target_list = ListType::GetChildType(target_type);
		auto &expr_list = ListType::GetChildType(expr_type);
		if (target_list.id() == LogicalTypeId::ANY || expr_list == target_list) {
			return expr;
		}
	}
	return make_unique<BoundCastExpression>(std::move(expr), target_type, std::move(bound_cast), try_cast);
}

static BoundCastInfo BindCastFunction(ClientContext &context, const LogicalType &source, const LogicalType &target) {
	auto &cast_functions = DBConfig::GetConfig(context).GetCastFunctions();
	GetCastFunctionInput input(context);
	return cast_functions.GetCastFunction(source, target, input);
}

unique_ptr<Expression> AddCastToTypeInternal(unique_ptr<Expression> expr, const LogicalType &target_type,
                                             CastFunctionSet &cast_functions, GetCastFunctionInput &get_input,
                                             bool try_cast) {
	D_ASSERT(expr);
	if (expr->expression_class == ExpressionClass::BOUND_PARAMETER) {
		auto &parameter = (BoundParameterExpression &)*expr;
		if (!target_type.IsValid()) {
			// invalidate the parameter
			parameter.parameter_data->return_type = LogicalType::INVALID;
			parameter.return_type = target_type;
			return expr;
		}
		if (parameter.parameter_data->return_type.id() == LogicalTypeId::INVALID) {
			// we don't know the type of this parameter
			parameter.return_type = target_type;
			return expr;
		}
		if (parameter.parameter_data->return_type.id() == LogicalTypeId::UNKNOWN) {
			// prepared statement parameter cast - but there is no type, convert the type
			parameter.parameter_data->return_type = target_type;
			parameter.return_type = target_type;
			return expr;
		}
		// prepared statement parameter already has a type
		if (parameter.parameter_data->return_type == target_type) {
			// this type! we are done
			parameter.return_type = parameter.parameter_data->return_type;
			return expr;
		}
		// invalidate the type
		parameter.parameter_data->return_type = LogicalType::INVALID;
		parameter.return_type = target_type;
		return expr;
	} else if (expr->expression_class == ExpressionClass::BOUND_DEFAULT) {
		D_ASSERT(target_type.IsValid());
		auto &def = (BoundDefaultExpression &)*expr;
		def.return_type = target_type;
	}
	if (!target_type.IsValid()) {
		return expr;
	}

	auto cast_function = cast_functions.GetCastFunction(expr->return_type, target_type, get_input);
	return AddCastExpressionInternal(std::move(expr), target_type, std::move(cast_function), try_cast);
}

unique_ptr<Expression> BoundCastExpression::AddDefaultCastToType(unique_ptr<Expression> expr,
                                                                 const LogicalType &target_type, bool try_cast) {
	CastFunctionSet default_set;
	GetCastFunctionInput get_input;
	return AddCastToTypeInternal(std::move(expr), target_type, default_set, get_input, try_cast);
}

unique_ptr<Expression> BoundCastExpression::AddCastToType(ClientContext &context, unique_ptr<Expression> expr,
                                                          const LogicalType &target_type, bool try_cast) {
	auto &cast_functions = DBConfig::GetConfig(context).GetCastFunctions();
	GetCastFunctionInput get_input(context);
	return AddCastToTypeInternal(std::move(expr), target_type, cast_functions, get_input, try_cast);
}

bool BoundCastExpression::CastIsInvertible(const LogicalType &source_type, const LogicalType &target_type) {
	D_ASSERT(source_type.IsValid() && target_type.IsValid());
	if (source_type.id() == LogicalTypeId::BOOLEAN || target_type.id() == LogicalTypeId::BOOLEAN) {
		return false;
	}
	if (source_type.id() == LogicalTypeId::FLOAT || target_type.id() == LogicalTypeId::FLOAT) {
		return false;
	}
	if (source_type.id() == LogicalTypeId::DOUBLE || target_type.id() == LogicalTypeId::DOUBLE) {
		return false;
	}
	if (source_type.id() == LogicalTypeId::DECIMAL || target_type.id() == LogicalTypeId::DECIMAL) {
		uint8_t source_width, target_width;
		uint8_t source_scale, target_scale;
		// cast to or from decimal
		// cast is only invertible if the cast is strictly widening
		if (!source_type.GetDecimalProperties(source_width, source_scale)) {
			return false;
		}
		if (!target_type.GetDecimalProperties(target_width, target_scale)) {
			return false;
		}
		if (target_scale < source_scale) {
			return false;
		}
		return true;
	}
	if (source_type.id() == LogicalTypeId::TIMESTAMP || source_type.id() == LogicalTypeId::TIMESTAMP_TZ) {
		switch (target_type.id()) {
		case LogicalTypeId::DATE:
		case LogicalTypeId::TIME:
		case LogicalTypeId::TIME_TZ:
			return false;
		default:
			break;
		}
	}
	if (source_type.id() == LogicalTypeId::VARCHAR) {
		switch (target_type.id()) {
		case LogicalTypeId::TIME:
		case LogicalTypeId::TIMESTAMP:
		case LogicalTypeId::TIMESTAMP_NS:
		case LogicalTypeId::TIMESTAMP_MS:
		case LogicalTypeId::TIMESTAMP_SEC:
		case LogicalTypeId::TIME_TZ:
		case LogicalTypeId::TIMESTAMP_TZ:
			return true;
		default:
			return false;
		}
	}
	if (target_type.id() == LogicalTypeId::VARCHAR) {
		switch (source_type.id()) {
		case LogicalTypeId::DATE:
		case LogicalTypeId::TIME:
		case LogicalTypeId::TIMESTAMP:
		case LogicalTypeId::TIMESTAMP_NS:
		case LogicalTypeId::TIMESTAMP_MS:
		case LogicalTypeId::TIMESTAMP_SEC:
		case LogicalTypeId::TIME_TZ:
		case LogicalTypeId::TIMESTAMP_TZ:
			return true;
		default:
			return false;
		}
	}
	return true;
}

string BoundCastExpression::ToString() const {
	return (try_cast ? "TRY_CAST(" : "CAST(") + child->GetName() + " AS " + return_type.ToString() + ")";
}

bool BoundCastExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto other = (BoundCastExpression *)other_p;
	if (!Expression::Equals(child.get(), other->child.get())) {
		return false;
	}
	if (try_cast != other->try_cast) {
		return false;
	}
	return true;
}

unique_ptr<Expression> BoundCastExpression::Copy() {
	auto copy = make_unique<BoundCastExpression>(child->Copy(), return_type, bound_cast.Copy(), try_cast);
	copy->CopyProperties(*this);
	return std::move(copy);
}

void BoundCastExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*child);
	writer.WriteSerializable(return_type);
	writer.WriteField(try_cast);
}

unique_ptr<Expression> BoundCastExpression::Deserialize(ExpressionDeserializationState &state, FieldReader &reader) {
	auto child = reader.ReadRequiredSerializable<Expression>(state.gstate);
	auto target_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto try_cast = reader.ReadRequired<bool>();
	auto cast_function = BindCastFunction(state.gstate.context, child->return_type, target_type);
	return make_unique<BoundCastExpression>(std::move(child), std::move(target_type), std::move(cast_function),
	                                        try_cast);
}

} // namespace duckdb






namespace duckdb {

BoundColumnRefExpression::BoundColumnRefExpression(string alias_p, LogicalType type, ColumnBinding binding, idx_t depth)
    : Expression(ExpressionType::BOUND_COLUMN_REF, ExpressionClass::BOUND_COLUMN_REF, std::move(type)),
      binding(binding), depth(depth) {
	this->alias = std::move(alias_p);
}

BoundColumnRefExpression::BoundColumnRefExpression(LogicalType type, ColumnBinding binding, idx_t depth)
    : BoundColumnRefExpression(string(), std::move(type), binding, depth) {
}

unique_ptr<Expression> BoundColumnRefExpression::Copy() {
	return make_unique<BoundColumnRefExpression>(alias, return_type, binding, depth);
}

hash_t BoundColumnRefExpression::Hash() const {
	auto result = Expression::Hash();
	result = CombineHash(result, duckdb::Hash<uint64_t>(binding.column_index));
	result = CombineHash(result, duckdb::Hash<uint64_t>(binding.table_index));
	return CombineHash(result, duckdb::Hash<uint64_t>(depth));
}

bool BoundColumnRefExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto other = (BoundColumnRefExpression *)other_p;
	return other->binding == binding && other->depth == depth;
}

string BoundColumnRefExpression::ToString() const {
	if (!alias.empty()) {
		return alias;
	}
	return "#[" + to_string(binding.table_index) + "." + to_string(binding.column_index) + "]";
}

void BoundColumnRefExpression::Serialize(FieldWriter &writer) const {
	writer.WriteString(alias);
	writer.WriteSerializable(return_type);
	writer.WriteField(binding.table_index);
	writer.WriteField(binding.column_index);
	writer.WriteField(depth);
}

unique_ptr<Expression> BoundColumnRefExpression::Deserialize(ExpressionDeserializationState &state,
                                                             FieldReader &reader) {
	auto alias = reader.ReadRequired<string>();
	auto return_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto table_index = reader.ReadRequired<idx_t>();
	auto column_index = reader.ReadRequired<idx_t>();
	auto depth = reader.ReadRequired<idx_t>();

	return make_unique<BoundColumnRefExpression>(alias, return_type, ColumnBinding(table_index, column_index), depth);
}

} // namespace duckdb




namespace duckdb {

BoundComparisonExpression::BoundComparisonExpression(ExpressionType type, unique_ptr<Expression> left,
                                                     unique_ptr<Expression> right)
    : Expression(type, ExpressionClass::BOUND_COMPARISON, LogicalType::BOOLEAN), left(std::move(left)),
      right(std::move(right)) {
}

string BoundComparisonExpression::ToString() const {
	return ComparisonExpression::ToString<BoundComparisonExpression, Expression>(*this);
}

bool BoundComparisonExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto other = (BoundComparisonExpression *)other_p;
	if (!Expression::Equals(left.get(), other->left.get())) {
		return false;
	}
	if (!Expression::Equals(right.get(), other->right.get())) {
		return false;
	}

	return true;
}

unique_ptr<Expression> BoundComparisonExpression::Copy() {
	auto copy = make_unique<BoundComparisonExpression>(type, left->Copy(), right->Copy());
	copy->CopyProperties(*this);
	return std::move(copy);
}

void BoundComparisonExpression::Serialize(FieldWriter &writer) const {
	writer.WriteOptional(left);
	writer.WriteOptional(right);
}

unique_ptr<Expression> BoundComparisonExpression::Deserialize(ExpressionDeserializationState &state,
                                                              FieldReader &reader) {
	auto left = reader.ReadOptional<Expression>(nullptr, state.gstate);
	auto right = reader.ReadOptional<Expression>(nullptr, state.gstate);
	return make_unique<BoundComparisonExpression>(state.type, std::move(left), std::move(right));
}

} // namespace duckdb





namespace duckdb {

BoundConjunctionExpression::BoundConjunctionExpression(ExpressionType type)
    : Expression(type, ExpressionClass::BOUND_CONJUNCTION, LogicalType::BOOLEAN) {
}

BoundConjunctionExpression::BoundConjunctionExpression(ExpressionType type, unique_ptr<Expression> left,
                                                       unique_ptr<Expression> right)
    : BoundConjunctionExpression(type) {
	children.push_back(std::move(left));
	children.push_back(std::move(right));
}

string BoundConjunctionExpression::ToString() const {
	return ConjunctionExpression::ToString<BoundConjunctionExpression, Expression>(*this);
}

bool BoundConjunctionExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto other = (BoundConjunctionExpression *)other_p;
	return ExpressionUtil::SetEquals(children, other->children);
}

bool BoundConjunctionExpression::PropagatesNullValues() const {
	return false;
}

unique_ptr<Expression> BoundConjunctionExpression::Copy() {
	auto copy = make_unique<BoundConjunctionExpression>(type);
	for (auto &expr : children) {
		copy->children.push_back(expr->Copy());
	}
	copy->CopyProperties(*this);
	return std::move(copy);
}

void BoundConjunctionExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializableList(children);
}

unique_ptr<Expression> BoundConjunctionExpression::Deserialize(ExpressionDeserializationState &state,
                                                               FieldReader &reader) {
	auto children = reader.ReadRequiredSerializableList<Expression>(state.gstate);
	auto res = make_unique<BoundConjunctionExpression>(state.type);
	res->children = std::move(children);
	return std::move(res);
}

} // namespace duckdb





namespace duckdb {

BoundConstantExpression::BoundConstantExpression(Value value_p)
    : Expression(ExpressionType::VALUE_CONSTANT, ExpressionClass::BOUND_CONSTANT, value_p.type()),
      value(std::move(value_p)) {
}

string BoundConstantExpression::ToString() const {
	return value.ToSQLString();
}

bool BoundConstantExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto other = (BoundConstantExpression *)other_p;
	return value.type() == other->value.type() && !ValueOperations::DistinctFrom(value, other->value);
}

hash_t BoundConstantExpression::Hash() const {
	hash_t result = Expression::Hash();
	return CombineHash(value.Hash(), result);
}

unique_ptr<Expression> BoundConstantExpression::Copy() {
	auto copy = make_unique<BoundConstantExpression>(value);
	copy->CopyProperties(*this);
	return std::move(copy);
}

void BoundConstantExpression::Serialize(FieldWriter &writer) const {
	value.Serialize(writer.GetSerializer());
}

unique_ptr<Expression> BoundConstantExpression::Deserialize(ExpressionDeserializationState &state,
                                                            FieldReader &reader) {
	auto value = Value::Deserialize(reader.GetSource());
	return make_unique<BoundConstantExpression>(value);
}

} // namespace duckdb



namespace duckdb {

void BoundDefaultExpression::Serialize(FieldWriter &writer) const {
	throw NotImplementedException(ExpressionTypeToString(type));
}

} // namespace duckdb


namespace duckdb {

BoundExpression::BoundExpression(unique_ptr<Expression> expr)
    : ParsedExpression(ExpressionType::INVALID, ExpressionClass::BOUND_EXPRESSION), expr(std::move(expr)) {
}

string BoundExpression::ToString() const {
	if (!expr) {
		throw InternalException("ToString(): BoundExpression does not have a child");
	}
	return expr->ToString();
}

bool BoundExpression::Equals(const BaseExpression *other) const {
	return false;
}
hash_t BoundExpression::Hash() const {
	return 0;
}

unique_ptr<ParsedExpression> BoundExpression::Copy() const {
	throw SerializationException("Cannot copy or serialize bound expression");
}

void BoundExpression::Serialize(FieldWriter &writer) const {
	throw SerializationException("Cannot copy or serialize bound expression");
}

} // namespace duckdb







namespace duckdb {

BoundFunctionExpression::BoundFunctionExpression(LogicalType return_type, ScalarFunction bound_function,
                                                 vector<unique_ptr<Expression>> arguments,
                                                 unique_ptr<FunctionData> bind_info, bool is_operator)
    : Expression(ExpressionType::BOUND_FUNCTION, ExpressionClass::BOUND_FUNCTION, std::move(return_type)),
      function(std::move(bound_function)), children(std::move(arguments)), bind_info(std::move(bind_info)),
      is_operator(is_operator) {
	D_ASSERT(!function.name.empty());
}

bool BoundFunctionExpression::HasSideEffects() const {
	return function.side_effects == FunctionSideEffects::HAS_SIDE_EFFECTS ? true : Expression::HasSideEffects();
}

bool BoundFunctionExpression::IsFoldable() const {
	// functions with side effects cannot be folded: they have to be executed once for every row
	return function.side_effects == FunctionSideEffects::HAS_SIDE_EFFECTS ? false : Expression::IsFoldable();
}

string BoundFunctionExpression::ToString() const {
	return FunctionExpression::ToString<BoundFunctionExpression, Expression>(*this, string(), function.name,
	                                                                         is_operator);
}
bool BoundFunctionExpression::PropagatesNullValues() const {
	return function.null_handling == FunctionNullHandling::SPECIAL_HANDLING ? false
	                                                                        : Expression::PropagatesNullValues();
}

hash_t BoundFunctionExpression::Hash() const {
	hash_t result = Expression::Hash();
	return CombineHash(result, function.Hash());
}

bool BoundFunctionExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto other = (BoundFunctionExpression *)other_p;
	if (other->function != function) {
		return false;
	}
	if (!ExpressionUtil::ListEquals(children, other->children)) {
		return false;
	}
	if (!FunctionData::Equals(bind_info.get(), other->bind_info.get())) {
		return false;
	}
	return true;
}

unique_ptr<Expression> BoundFunctionExpression::Copy() {
	vector<unique_ptr<Expression>> new_children;
	new_children.reserve(children.size());
	for (auto &child : children) {
		new_children.push_back(child->Copy());
	}
	unique_ptr<FunctionData> new_bind_info = bind_info ? bind_info->Copy() : nullptr;

	auto copy = make_unique<BoundFunctionExpression>(return_type, function, std::move(new_children),
	                                                 std::move(new_bind_info), is_operator);
	copy->CopyProperties(*this);
	return std::move(copy);
}

void BoundFunctionExpression::Verify() const {
	D_ASSERT(!function.name.empty());
}

void BoundFunctionExpression::Serialize(FieldWriter &writer) const {
	D_ASSERT(!function.name.empty());
	D_ASSERT(return_type == function.return_type);
	writer.WriteField(is_operator);
	FunctionSerializer::Serialize<ScalarFunction>(writer, function, return_type, children, bind_info.get());
}

unique_ptr<Expression> BoundFunctionExpression::Deserialize(ExpressionDeserializationState &state,
                                                            FieldReader &reader) {
	auto is_operator = reader.ReadRequired<bool>();
	vector<unique_ptr<Expression>> children;
	unique_ptr<FunctionData> bind_info;
	auto function = FunctionSerializer::Deserialize<ScalarFunction, ScalarFunctionCatalogEntry>(
	    reader, state, CatalogType::SCALAR_FUNCTION_ENTRY, children, bind_info);

	auto return_type = function.return_type;
	return make_unique<BoundFunctionExpression>(std::move(return_type), std::move(function), std::move(children),
	                                            std::move(bind_info), is_operator);
}
} // namespace duckdb




namespace duckdb {

BoundLambdaExpression::BoundLambdaExpression(ExpressionType type_p, LogicalType return_type_p,
                                             unique_ptr<Expression> lambda_expr_p, idx_t parameter_count_p)
    : Expression(type_p, ExpressionClass::BOUND_LAMBDA, std::move(return_type_p)),
      lambda_expr(std::move(lambda_expr_p)), parameter_count(parameter_count_p) {
}

string BoundLambdaExpression::ToString() const {
	return lambda_expr->ToString();
}

bool BoundLambdaExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto other = (BoundLambdaExpression *)other_p;
	if (!Expression::Equals(lambda_expr.get(), other->lambda_expr.get())) {
		return false;
	}
	if (!ExpressionUtil::ListEquals(captures, other->captures)) {
		return false;
	}
	if (parameter_count != other->parameter_count) {
		return false;
	}
	return true;
}

unique_ptr<Expression> BoundLambdaExpression::Copy() {
	auto copy = make_unique<BoundLambdaExpression>(type, return_type, lambda_expr->Copy(), parameter_count);
	for (auto &capture : captures) {
		copy->captures.push_back(capture->Copy());
	}
	return std::move(copy);
}

void BoundLambdaExpression::Serialize(FieldWriter &writer) const {
	throw NotImplementedException(ExpressionTypeToString(type));
}

} // namespace duckdb






namespace duckdb {

BoundLambdaRefExpression::BoundLambdaRefExpression(string alias_p, LogicalType type, ColumnBinding binding,
                                                   idx_t lambda_index, idx_t depth)
    : Expression(ExpressionType::BOUND_LAMBDA_REF, ExpressionClass::BOUND_LAMBDA_REF, std::move(type)),
      binding(binding), lambda_index(lambda_index), depth(depth) {
	this->alias = std::move(alias_p);
}

BoundLambdaRefExpression::BoundLambdaRefExpression(LogicalType type, ColumnBinding binding, idx_t lambda_index,
                                                   idx_t depth)
    : BoundLambdaRefExpression(string(), std::move(type), binding, lambda_index, depth) {
}

unique_ptr<Expression> BoundLambdaRefExpression::Copy() {
	return make_unique<BoundLambdaRefExpression>(alias, return_type, binding, lambda_index, depth);
}

hash_t BoundLambdaRefExpression::Hash() const {
	auto result = Expression::Hash();
	result = CombineHash(result, duckdb::Hash<uint64_t>(lambda_index));
	result = CombineHash(result, duckdb::Hash<uint64_t>(binding.column_index));
	result = CombineHash(result, duckdb::Hash<uint64_t>(binding.table_index));
	return CombineHash(result, duckdb::Hash<uint64_t>(depth));
}

bool BoundLambdaRefExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto other = (BoundLambdaRefExpression *)other_p;
	return other->binding == binding && other->lambda_index == lambda_index && other->depth == depth;
}

string BoundLambdaRefExpression::ToString() const {
	if (!alias.empty()) {
		return alias;
	}
	return "#[" + to_string(binding.table_index) + "." + to_string(binding.column_index) + "." +
	       to_string(lambda_index) + "]";
}

void BoundLambdaRefExpression::Serialize(FieldWriter &writer) const {
	writer.WriteString(alias);
	writer.WriteSerializable(return_type);
	writer.WriteField(lambda_index);
	writer.WriteField(binding.table_index);
	writer.WriteField(binding.column_index);
	writer.WriteField(depth);
}

unique_ptr<Expression> BoundLambdaRefExpression::Deserialize(ExpressionDeserializationState &state,
                                                             FieldReader &reader) {
	auto alias = reader.ReadRequired<string>();
	auto return_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto lambda_index = reader.ReadRequired<idx_t>();
	auto table_index = reader.ReadRequired<idx_t>();
	auto column_index = reader.ReadRequired<idx_t>();
	auto depth = reader.ReadRequired<idx_t>();

	return make_unique<BoundLambdaRefExpression>(alias, return_type, ColumnBinding(table_index, column_index),
	                                             lambda_index, depth);
}

} // namespace duckdb






namespace duckdb {

BoundOperatorExpression::BoundOperatorExpression(ExpressionType type, LogicalType return_type)
    : Expression(type, ExpressionClass::BOUND_OPERATOR, std::move(return_type)) {
}

string BoundOperatorExpression::ToString() const {
	return OperatorExpression::ToString<BoundOperatorExpression, Expression>(*this);
}

bool BoundOperatorExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto other = (BoundOperatorExpression *)other_p;
	if (!ExpressionUtil::ListEquals(children, other->children)) {
		return false;
	}
	return true;
}

unique_ptr<Expression> BoundOperatorExpression::Copy() {
	auto copy = make_unique<BoundOperatorExpression>(type, return_type);
	copy->CopyProperties(*this);
	for (auto &child : children) {
		copy->children.push_back(child->Copy());
	}
	return std::move(copy);
}

void BoundOperatorExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(return_type);
	writer.WriteSerializableList(children);
}

unique_ptr<Expression> BoundOperatorExpression::Deserialize(ExpressionDeserializationState &state,
                                                            FieldReader &reader) {
	auto return_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto children = reader.ReadRequiredSerializableList<Expression>(state.gstate);

	auto result = make_unique<BoundOperatorExpression>(state.type, return_type);
	result->children = std::move(children);
	return std::move(result);
}

} // namespace duckdb






namespace duckdb {

BoundParameterExpression::BoundParameterExpression(idx_t parameter_nr)
    : Expression(ExpressionType::VALUE_PARAMETER, ExpressionClass::BOUND_PARAMETER,
                 LogicalType(LogicalTypeId::UNKNOWN)),
      parameter_nr(parameter_nr) {
}

void BoundParameterExpression::Invalidate(Expression &expr) {
	if (expr.type != ExpressionType::VALUE_PARAMETER) {
		throw InternalException("BoundParameterExpression::Invalidate requires a parameter as input");
	}
	auto &bound_parameter = (BoundParameterExpression &)expr;
	bound_parameter.return_type = LogicalTypeId::SQLNULL;
	bound_parameter.parameter_data->return_type = LogicalTypeId::INVALID;
}

void BoundParameterExpression::InvalidateRecursive(Expression &expr) {
	if (expr.type == ExpressionType::VALUE_PARAMETER) {
		Invalidate(expr);
		return;
	}
	ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) { InvalidateRecursive(child); });
}

bool BoundParameterExpression::IsScalar() const {
	return true;
}
bool BoundParameterExpression::HasParameter() const {
	return true;
}
bool BoundParameterExpression::IsFoldable() const {
	return false;
}

string BoundParameterExpression::ToString() const {
	return "$" + to_string(parameter_nr);
}

bool BoundParameterExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto other = (BoundParameterExpression *)other_p;
	return parameter_nr == other->parameter_nr;
}

hash_t BoundParameterExpression::Hash() const {
	hash_t result = Expression::Hash();
	result = CombineHash(duckdb::Hash(parameter_nr), result);
	return result;
}

unique_ptr<Expression> BoundParameterExpression::Copy() {
	auto result = make_unique<BoundParameterExpression>(parameter_nr);
	result->parameter_data = parameter_data;
	result->return_type = return_type;
	result->CopyProperties(*this);
	return std::move(result);
}

void BoundParameterExpression::Serialize(FieldWriter &writer) const {
	writer.WriteField(parameter_nr);
	writer.WriteSerializable(return_type);
	writer.WriteSerializable(*parameter_data);
}

unique_ptr<Expression> BoundParameterExpression::Deserialize(ExpressionDeserializationState &state,
                                                             FieldReader &reader) {
	auto &global_parameter_set = state.gstate.parameter_data;
	auto parameter_nr = reader.ReadRequired<idx_t>();
	auto result = make_unique<BoundParameterExpression>(parameter_nr);
	result->return_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto parameter_data = reader.ReadRequiredSerializable<BoundParameterData, shared_ptr<BoundParameterData>>();
	// check if we have already deserialized a parameter with this number
	auto entry = global_parameter_set.find(parameter_nr);
	if (entry == global_parameter_set.end()) {
		// we have not - store the entry we deserialized from this parameter expression
		global_parameter_set[parameter_nr] = parameter_data;
	} else {
		// we have! use the previously deserialized entry
		parameter_data = entry->second;
	}
	result->parameter_data = std::move(parameter_data);
	return std::move(result);
}

} // namespace duckdb







namespace duckdb {

BoundReferenceExpression::BoundReferenceExpression(string alias, LogicalType type, idx_t index)
    : Expression(ExpressionType::BOUND_REF, ExpressionClass::BOUND_REF, std::move(type)), index(index) {
	this->alias = std::move(alias);
}
BoundReferenceExpression::BoundReferenceExpression(LogicalType type, idx_t index)
    : BoundReferenceExpression(string(), std::move(type), index) {
}

string BoundReferenceExpression::ToString() const {
	if (!alias.empty()) {
		return alias;
	}
	return "#" + to_string(index);
}

bool BoundReferenceExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto other = (BoundReferenceExpression *)other_p;
	return other->index == index;
}

hash_t BoundReferenceExpression::Hash() const {
	return CombineHash(Expression::Hash(), duckdb::Hash<idx_t>(index));
}

unique_ptr<Expression> BoundReferenceExpression::Copy() {
	return make_unique<BoundReferenceExpression>(alias, return_type, index);
}

void BoundReferenceExpression::Serialize(FieldWriter &writer) const {
	writer.WriteString(alias);
	writer.WriteSerializable(return_type);
	writer.WriteField(index);
}

unique_ptr<Expression> BoundReferenceExpression::Deserialize(ExpressionDeserializationState &state,
                                                             FieldReader &reader) {
	auto alias = reader.ReadRequired<string>();
	auto return_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto index = reader.ReadRequired<idx_t>();
	return make_unique<BoundReferenceExpression>(alias, return_type, index);
}

} // namespace duckdb




namespace duckdb {

BoundSubqueryExpression::BoundSubqueryExpression(LogicalType return_type)
    : Expression(ExpressionType::SUBQUERY, ExpressionClass::BOUND_SUBQUERY, std::move(return_type)) {
}

string BoundSubqueryExpression::ToString() const {
	return "SUBQUERY";
}

bool BoundSubqueryExpression::Equals(const BaseExpression *other_p) const {
	// equality between bound subqueries not implemented currently
	return false;
}

unique_ptr<Expression> BoundSubqueryExpression::Copy() {
	throw SerializationException("Cannot copy BoundSubqueryExpression");
}

bool BoundSubqueryExpression::PropagatesNullValues() const {
	// TODO this can be optimized further by checking the actual subquery node
	return false;
}

void BoundSubqueryExpression::Serialize(FieldWriter &writer) const {
	throw NotImplementedException(ExpressionTypeToString(type));
}

} // namespace duckdb






namespace duckdb {

BoundUnnestExpression::BoundUnnestExpression(LogicalType return_type)
    : Expression(ExpressionType::BOUND_UNNEST, ExpressionClass::BOUND_UNNEST, std::move(return_type)) {
}

bool BoundUnnestExpression::IsFoldable() const {
	return false;
}

string BoundUnnestExpression::ToString() const {
	return "UNNEST(" + child->ToString() + ")";
}

hash_t BoundUnnestExpression::Hash() const {
	hash_t result = Expression::Hash();
	return CombineHash(result, duckdb::Hash("unnest"));
}

bool BoundUnnestExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto other = (BoundUnnestExpression *)other_p;
	if (!Expression::Equals(child.get(), other->child.get())) {
		return false;
	}
	return true;
}

unique_ptr<Expression> BoundUnnestExpression::Copy() {
	auto copy = make_unique<BoundUnnestExpression>(return_type);
	copy->child = child->Copy();
	return std::move(copy);
}

void BoundUnnestExpression::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(return_type);
	writer.WriteSerializable(*child);
}

unique_ptr<Expression> BoundUnnestExpression::Deserialize(ExpressionDeserializationState &state, FieldReader &reader) {
	auto return_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	auto child = reader.ReadRequiredSerializable<Expression>(state.gstate);

	auto result = make_unique<BoundUnnestExpression>(return_type);
	result->child = std::move(child);
	return std::move(result);
}

} // namespace duckdb








namespace duckdb {

BoundWindowExpression::BoundWindowExpression(ExpressionType type, LogicalType return_type,
                                             unique_ptr<AggregateFunction> aggregate,
                                             unique_ptr<FunctionData> bind_info)
    : Expression(type, ExpressionClass::BOUND_WINDOW, std::move(return_type)), aggregate(std::move(aggregate)),
      bind_info(std::move(bind_info)), ignore_nulls(false) {
}

string BoundWindowExpression::ToString() const {
	string function_name = aggregate.get() ? aggregate->name : ExpressionTypeToString(type);
	return WindowExpression::ToString<BoundWindowExpression, Expression, BoundOrderByNode>(*this, string(),
	                                                                                       function_name);
}

bool BoundWindowExpression::Equals(const BaseExpression *other_p) const {
	if (!Expression::Equals(other_p)) {
		return false;
	}
	auto other = (BoundWindowExpression *)other_p;

	if (ignore_nulls != other->ignore_nulls) {
		return false;
	}
	if (start != other->start || end != other->end) {
		return false;
	}
	// check if the child expressions are equivalent
	if (other->children.size() != children.size()) {
		return false;
	}
	for (idx_t i = 0; i < children.size(); i++) {
		if (!Expression::Equals(children[i].get(), other->children[i].get())) {
			return false;
		}
	}
	// check if the filter expressions are equivalent
	if (!Expression::Equals(filter_expr.get(), other->filter_expr.get())) {
		return false;
	}

	// check if the framing expressions are equivalent
	if (!Expression::Equals(start_expr.get(), other->start_expr.get()) ||
	    !Expression::Equals(end_expr.get(), other->end_expr.get()) ||
	    !Expression::Equals(offset_expr.get(), other->offset_expr.get()) ||
	    !Expression::Equals(default_expr.get(), other->default_expr.get())) {
		return false;
	}

	return KeysAreCompatible(other);
}

bool BoundWindowExpression::KeysAreCompatible(const BoundWindowExpression *other) const {
	// check if the partitions are equivalent
	if (partitions.size() != other->partitions.size()) {
		return false;
	}
	for (idx_t i = 0; i < partitions.size(); i++) {
		if (!Expression::Equals(partitions[i].get(), other->partitions[i].get())) {
			return false;
		}
	}
	// check if the orderings are equivalent
	if (orders.size() != other->orders.size()) {
		return false;
	}
	for (idx_t i = 0; i < orders.size(); i++) {
		if (orders[i].type != other->orders[i].type) {
			return false;
		}
		if (!BaseExpression::Equals((BaseExpression *)orders[i].expression.get(),
		                            (BaseExpression *)other->orders[i].expression.get())) {
			return false;
		}
	}
	return true;
}

unique_ptr<Expression> BoundWindowExpression::Copy() {
	auto new_window = make_unique<BoundWindowExpression>(type, return_type, nullptr, nullptr);
	new_window->CopyProperties(*this);

	if (aggregate) {
		new_window->aggregate = make_unique<AggregateFunction>(*aggregate);
	}
	if (bind_info) {
		new_window->bind_info = bind_info->Copy();
	}
	for (auto &child : children) {
		new_window->children.push_back(child->Copy());
	}
	for (auto &e : partitions) {
		new_window->partitions.push_back(e->Copy());
	}
	for (auto &ps : partitions_stats) {
		if (ps) {
			new_window->partitions_stats.push_back(ps->Copy());
		} else {
			new_window->partitions_stats.push_back(nullptr);
		}
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

void BoundWindowExpression::Serialize(FieldWriter &writer) const {
	writer.WriteField<bool>(aggregate.get());
	if (aggregate) {
		D_ASSERT(return_type == aggregate->return_type);
		FunctionSerializer::Serialize<AggregateFunction>(writer, *aggregate, return_type, children, bind_info.get());
	} else {
		// children and return_type are written as part of the aggregate function otherwise
		writer.WriteSerializableList(children);
		writer.WriteSerializable(return_type);
	}
	writer.WriteSerializableList(partitions);
	writer.WriteRegularSerializableList(orders);
	// FIXME: partitions_stats
	writer.WriteOptional(filter_expr);
	writer.WriteField<bool>(ignore_nulls);
	writer.WriteField<WindowBoundary>(start);
	writer.WriteField<WindowBoundary>(end);
	writer.WriteOptional(start_expr);
	writer.WriteOptional(end_expr);
	writer.WriteOptional(offset_expr);
	writer.WriteOptional(default_expr);
}

unique_ptr<Expression> BoundWindowExpression::Deserialize(ExpressionDeserializationState &state, FieldReader &reader) {
	auto has_aggregate = reader.ReadRequired<bool>();
	unique_ptr<AggregateFunction> aggregate;
	unique_ptr<FunctionData> bind_info;
	vector<unique_ptr<Expression>> children;
	LogicalType return_type;
	if (has_aggregate) {
		auto aggr_function = FunctionSerializer::Deserialize<AggregateFunction, AggregateFunctionCatalogEntry>(
		    reader, state, CatalogType::AGGREGATE_FUNCTION_ENTRY, children, bind_info);
		aggregate = make_unique<AggregateFunction>(std::move(aggr_function));
		return_type = aggregate->return_type;
	} else {
		children = reader.ReadRequiredSerializableList<Expression>(state.gstate);
		return_type = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
	}
	auto result =
	    make_unique<BoundWindowExpression>(state.type, return_type, std::move(aggregate), std::move(bind_info));

	result->partitions = reader.ReadRequiredSerializableList<Expression>(state.gstate);
	result->orders = reader.ReadRequiredSerializableList<BoundOrderByNode, BoundOrderByNode>(state.gstate);
	result->filter_expr = reader.ReadOptional<Expression>(nullptr, state.gstate);
	result->ignore_nulls = reader.ReadRequired<bool>();
	result->start = reader.ReadRequired<WindowBoundary>();
	result->end = reader.ReadRequired<WindowBoundary>();
	result->start_expr = reader.ReadOptional<Expression>(nullptr, state.gstate);
	result->end_expr = reader.ReadOptional<Expression>(nullptr, state.gstate);
	result->offset_expr = reader.ReadOptional<Expression>(nullptr, state.gstate);
	result->default_expr = reader.ReadOptional<Expression>(nullptr, state.gstate);
	result->children = std::move(children);
	return std::move(result);
}

} // namespace duckdb









namespace duckdb {

Expression::Expression(ExpressionType type, ExpressionClass expression_class, LogicalType return_type)
    : BaseExpression(type, expression_class), return_type(std::move(return_type)) {
}

Expression::~Expression() {
}

bool Expression::IsAggregate() const {
	bool is_aggregate = false;
	ExpressionIterator::EnumerateChildren(*this, [&](const Expression &child) { is_aggregate |= child.IsAggregate(); });
	return is_aggregate;
}

bool Expression::IsWindow() const {
	bool is_window = false;
	ExpressionIterator::EnumerateChildren(*this, [&](const Expression &child) { is_window |= child.IsWindow(); });
	return is_window;
}

bool Expression::IsScalar() const {
	bool is_scalar = true;
	ExpressionIterator::EnumerateChildren(*this, [&](const Expression &child) {
		if (!child.IsScalar()) {
			is_scalar = false;
		}
	});
	return is_scalar;
}

bool Expression::HasSideEffects() const {
	bool has_side_effects = false;
	ExpressionIterator::EnumerateChildren(*this, [&](const Expression &child) {
		if (child.HasSideEffects()) {
			has_side_effects = true;
		}
	});
	return has_side_effects;
}

bool Expression::PropagatesNullValues() const {
	if (type == ExpressionType::OPERATOR_IS_NULL || type == ExpressionType::OPERATOR_IS_NOT_NULL ||
	    type == ExpressionType::COMPARE_NOT_DISTINCT_FROM || type == ExpressionType::COMPARE_DISTINCT_FROM ||
	    type == ExpressionType::CONJUNCTION_OR || type == ExpressionType::CONJUNCTION_AND) {
		return false;
	}
	bool propagate_null_values = true;
	ExpressionIterator::EnumerateChildren(*this, [&](const Expression &child) {
		if (!child.PropagatesNullValues()) {
			propagate_null_values = false;
		}
	});
	return propagate_null_values;
}

bool Expression::IsFoldable() const {
	bool is_foldable = true;
	ExpressionIterator::EnumerateChildren(*this, [&](const Expression &child) {
		if (!child.IsFoldable()) {
			is_foldable = false;
		}
	});
	return is_foldable;
}

bool Expression::HasParameter() const {
	bool has_parameter = false;
	ExpressionIterator::EnumerateChildren(*this,
	                                      [&](const Expression &child) { has_parameter |= child.HasParameter(); });
	return has_parameter;
}

bool Expression::HasSubquery() const {
	bool has_subquery = false;
	ExpressionIterator::EnumerateChildren(*this, [&](const Expression &child) { has_subquery |= child.HasSubquery(); });
	return has_subquery;
}

hash_t Expression::Hash() const {
	hash_t hash = duckdb::Hash<uint32_t>((uint32_t)type);
	hash = CombineHash(hash, return_type.Hash());
	ExpressionIterator::EnumerateChildren(*this,
	                                      [&](const Expression &child) { hash = CombineHash(child.Hash(), hash); });
	return hash;
}

void Expression::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteField<ExpressionClass>(expression_class);
	writer.WriteField<ExpressionType>(type);
	writer.WriteString(alias);
	Serialize(writer);
	writer.Finalize();
}

unique_ptr<Expression> Expression::Deserialize(Deserializer &source, PlanDeserializationState &gstate) {
	FieldReader reader(source);
	auto expression_class = reader.ReadRequired<ExpressionClass>();
	auto type = reader.ReadRequired<ExpressionType>();
	auto alias = reader.ReadRequired<string>();

	ExpressionDeserializationState state(gstate, type);

	unique_ptr<Expression> result;
	switch (expression_class) {
	case ExpressionClass::BOUND_REF:
		result = BoundReferenceExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_COLUMN_REF:
		result = BoundColumnRefExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_AGGREGATE:
		result = BoundAggregateExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_BETWEEN:
		result = BoundBetweenExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_CONSTANT:
		result = BoundConstantExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_FUNCTION:
		result = BoundFunctionExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_CAST:
		result = BoundCastExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_CASE:
		result = BoundCaseExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_CONJUNCTION:
		result = BoundConjunctionExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_COMPARISON:
		result = BoundComparisonExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_OPERATOR:
		result = BoundOperatorExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_WINDOW:
		result = BoundWindowExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_UNNEST:
		result = BoundUnnestExpression::Deserialize(state, reader);
		break;
	case ExpressionClass::BOUND_PARAMETER:
		result = BoundParameterExpression::Deserialize(state, reader);
		break;
	default:
		throw SerializationException("Unsupported type for expression deserialization %s",
		                             ExpressionTypeToString(type));
	}
	result->alias = alias;
	reader.Finalize();
	return result;
}

} // namespace duckdb




namespace duckdb {

AggregateBinder::AggregateBinder(Binder &binder, ClientContext &context) : ExpressionBinder(binder, context, true) {
}

BindResult AggregateBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
	switch (expr.expression_class) {
	case ExpressionClass::WINDOW:
		throw ParserException("aggregate function calls cannot contain window function calls");
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

string AggregateBinder::UnsupportedAggregateMessage() {
	return "aggregate function calls cannot be nested";
}
} // namespace duckdb






namespace duckdb {

AlterBinder::AlterBinder(Binder &binder, ClientContext &context, TableCatalogEntry &table,
                         vector<LogicalIndex> &bound_columns, LogicalType target_type)
    : ExpressionBinder(binder, context), table(table), bound_columns(bound_columns) {
	this->target_type = std::move(target_type);
}

BindResult AlterBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::WINDOW:
		return BindResult("window functions are not allowed in alter statement");
	case ExpressionClass::SUBQUERY:
		return BindResult("cannot use subquery in alter statement");
	case ExpressionClass::COLUMN_REF:
		return BindColumn((ColumnRefExpression &)expr);
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

string AlterBinder::UnsupportedAggregateMessage() {
	return "aggregate functions are not allowed in alter statement";
}

BindResult AlterBinder::BindColumn(ColumnRefExpression &colref) {
	if (colref.column_names.size() > 1) {
		return BindQualifiedColumnName(colref, table.name);
	}
	auto idx = table.GetColumnIndex(colref.column_names[0], true);
	if (!idx.IsValid()) {
		throw BinderException("Table does not contain column %s referenced in alter statement!",
		                      colref.column_names[0]);
	}
	if (table.GetColumn(idx).Generated()) {
		throw BinderException("Using generated columns in alter statement not supported");
	}
	bound_columns.push_back(idx);
	return BindResult(make_unique<BoundReferenceExpression>(table.GetColumn(idx).Type(), bound_columns.size() - 1));
}

} // namespace duckdb






namespace duckdb {

CheckBinder::CheckBinder(Binder &binder, ClientContext &context, string table_p, const ColumnList &columns,
                         physical_index_set_t &bound_columns)
    : ExpressionBinder(binder, context), table(std::move(table_p)), columns(columns), bound_columns(bound_columns) {
	target_type = LogicalType::INTEGER;
}

BindResult CheckBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::WINDOW:
		return BindResult("window functions are not allowed in check constraints");
	case ExpressionClass::SUBQUERY:
		return BindResult("cannot use subquery in check constraint");
	case ExpressionClass::COLUMN_REF:
		return BindCheckColumn((ColumnRefExpression &)expr);
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

string CheckBinder::UnsupportedAggregateMessage() {
	return "aggregate functions are not allowed in check constraints";
}

BindResult ExpressionBinder::BindQualifiedColumnName(ColumnRefExpression &colref, const string &table_name) {
	idx_t struct_start = 0;
	if (colref.column_names[0] == table_name) {
		struct_start++;
	}
	auto result = make_unique_base<ParsedExpression, ColumnRefExpression>(colref.column_names.back());
	for (idx_t i = struct_start; i + 1 < colref.column_names.size(); i++) {
		result = CreateStructExtract(std::move(result), colref.column_names[i]);
	}
	return BindExpression(&result, 0);
}

BindResult CheckBinder::BindCheckColumn(ColumnRefExpression &colref) {

	// if this is a lambda parameters, then we temporarily add a BoundLambdaRef,
	// which we capture and remove later
	if (lambda_bindings) {
		for (idx_t i = 0; i < lambda_bindings->size(); i++) {
			if (colref.GetColumnName() == (*lambda_bindings)[i].dummy_name) {
				// FIXME: support lambdas in CHECK constraints
				// FIXME: like so: return (*lambda_bindings)[i].Bind(colref, i, depth);
				throw NotImplementedException("Lambda functions are currently not supported in CHECK constraints.");
			}
		}
	}

	if (colref.column_names.size() > 1) {
		return BindQualifiedColumnName(colref, table);
	}
	if (!columns.ColumnExists(colref.column_names[0])) {
		throw BinderException("Table does not contain column %s referenced in check constraint!",
		                      colref.column_names[0]);
	}
	auto &col = columns.GetColumn(colref.column_names[0]);
	if (col.Generated()) {
		auto bound_expression = col.GeneratedExpression().Copy();
		return BindExpression(&bound_expression, 0, false);
	}
	bound_columns.insert(col.Physical());
	D_ASSERT(col.StorageOid() != DConstants::INVALID_INDEX);
	return BindResult(make_unique<BoundReferenceExpression>(col.Type(), col.StorageOid()));
}

} // namespace duckdb








namespace duckdb {

ColumnAliasBinder::ColumnAliasBinder(BoundSelectNode &node, const case_insensitive_map_t<idx_t> &alias_map)
    : node(node), alias_map(alias_map), visited_select_indexes() {
}

BindResult ColumnAliasBinder::BindAlias(ExpressionBinder &enclosing_binder, ColumnRefExpression &expr, idx_t depth,
                                        bool root_expression) {
	if (expr.IsQualified()) {
		return BindResult(StringUtil::Format("Alias %s cannot be qualified.", expr.ToString()));
	}

	auto alias_entry = alias_map.find(expr.column_names[0]);
	if (alias_entry == alias_map.end()) {
		return BindResult(StringUtil::Format("Alias %s is not found.", expr.ToString()));
	}

	if (visited_select_indexes.find(alias_entry->second) != visited_select_indexes.end()) {
		return BindResult("Cannot resolve self-referential alias");
	}

	// found an alias: bind the alias expression
	auto expression = node.original_expressions[alias_entry->second]->Copy();
	visited_select_indexes.insert(alias_entry->second);

	// since the alias has been found, pass a depth of 0. See Issue 4978 (#16)
	// ColumnAliasBinders are only in Having, Qualify and Where Binders
	auto result = enclosing_binder.BindExpression(&expression, 0, root_expression);
	visited_select_indexes.erase(alias_entry->second);
	return result;
}

} // namespace duckdb


namespace duckdb {

ConstantBinder::ConstantBinder(Binder &binder, ClientContext &context, string clause)
    : ExpressionBinder(binder, context), clause(std::move(clause)) {
}

BindResult ConstantBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::COLUMN_REF:
		return BindResult(clause + " cannot contain column names");
	case ExpressionClass::SUBQUERY:
		throw BinderException(clause + " cannot contain subqueries");
	case ExpressionClass::DEFAULT:
		return BindResult(clause + " cannot contain DEFAULT clause");
	case ExpressionClass::WINDOW:
		return BindResult(clause + " cannot contain window functions!");
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

string ConstantBinder::UnsupportedAggregateMessage() {
	return clause + " cannot contain aggregates!";
}

} // namespace duckdb








namespace duckdb {

GroupBinder::GroupBinder(Binder &binder, ClientContext &context, SelectNode &node, idx_t group_index,
                         case_insensitive_map_t<idx_t> &alias_map, case_insensitive_map_t<idx_t> &group_alias_map)
    : ExpressionBinder(binder, context), node(node), alias_map(alias_map), group_alias_map(group_alias_map),
      group_index(group_index) {
}

BindResult GroupBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
	if (root_expression && depth == 0) {
		switch (expr.expression_class) {
		case ExpressionClass::COLUMN_REF:
			return BindColumnRef((ColumnRefExpression &)expr);
		case ExpressionClass::CONSTANT:
			return BindConstant((ConstantExpression &)expr);
		case ExpressionClass::PARAMETER:
			throw ParameterNotAllowedException("Parameter not supported in GROUP BY clause");
		default:
			break;
		}
	}
	switch (expr.expression_class) {
	case ExpressionClass::DEFAULT:
		return BindResult("GROUP BY clause cannot contain DEFAULT clause");
	case ExpressionClass::WINDOW:
		return BindResult("GROUP BY clause cannot contain window functions!");
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

string GroupBinder::UnsupportedAggregateMessage() {
	return "GROUP BY clause cannot contain aggregates!";
}

BindResult GroupBinder::BindSelectRef(idx_t entry) {
	if (used_aliases.find(entry) != used_aliases.end()) {
		// the alias has already been bound to before!
		// this happens if we group on the same alias twice
		// e.g. GROUP BY k, k or GROUP BY 1, 1
		// in this case, we can just replace the grouping with a constant since the second grouping has no effect
		// (the constant grouping will be optimized out later)
		return BindResult(make_unique<BoundConstantExpression>(Value::INTEGER(42)));
	}
	if (entry >= node.select_list.size()) {
		throw BinderException("GROUP BY term out of range - should be between 1 and %d", (int)node.select_list.size());
	}
	// we replace the root expression, also replace the unbound expression
	unbound_expression = node.select_list[entry]->Copy();
	// move the expression that this refers to here and bind it
	auto select_entry = std::move(node.select_list[entry]);
	auto binding = Bind(select_entry, nullptr, false);
	// now replace the original expression in the select list with a reference to this group
	group_alias_map[to_string(entry)] = bind_index;
	node.select_list[entry] = make_unique<ColumnRefExpression>(to_string(entry));
	// insert into the set of used aliases
	used_aliases.insert(entry);
	return BindResult(std::move(binding));
}

BindResult GroupBinder::BindConstant(ConstantExpression &constant) {
	// constant as root expression
	if (!constant.value.type().IsIntegral()) {
		// non-integral expression, we just leave the constant here.
		return ExpressionBinder::BindExpression(constant, 0);
	}
	// INTEGER constant: we use the integer as an index into the select list (e.g. GROUP BY 1)
	auto index = (idx_t)constant.value.GetValue<int64_t>();
	return BindSelectRef(index - 1);
}

BindResult GroupBinder::BindColumnRef(ColumnRefExpression &colref) {
	// columns in GROUP BY clauses:
	// FIRST refer to the original tables, and
	// THEN if no match is found refer to aliases in the SELECT list
	// THEN if no match is found, refer to outer queries

	// first try to bind to the base columns (original tables)
	auto result = ExpressionBinder::BindExpression(colref, 0);
	if (result.HasError()) {
		if (colref.IsQualified()) {
			// explicit table name: not an alias reference
			return result;
		}
		// failed to bind the column and the node is the root expression with depth = 0
		// check if refers to an alias in the select clause
		auto alias_name = colref.column_names[0];
		auto entry = alias_map.find(alias_name);
		if (entry == alias_map.end()) {
			// no matching alias found
			return result;
		}
		result = BindResult(BindSelectRef(entry->second));
		if (!result.HasError()) {
			group_alias_map[alias_name] = bind_index;
		}
	}
	return result;
}

} // namespace duckdb








namespace duckdb {

HavingBinder::HavingBinder(Binder &binder, ClientContext &context, BoundSelectNode &node, BoundGroupInformation &info,
                           case_insensitive_map_t<idx_t> &alias_map, AggregateHandling aggregate_handling)
    : SelectBinder(binder, context, node, info), column_alias_binder(node, alias_map),
      aggregate_handling(aggregate_handling) {
	target_type = LogicalType(LogicalTypeId::BOOLEAN);
}

BindResult HavingBinder::BindColumnRef(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = (ColumnRefExpression &)**expr_ptr;
	auto alias_result = column_alias_binder.BindAlias(*this, expr, depth, root_expression);
	if (!alias_result.HasError()) {
		return alias_result;
	}
	if (aggregate_handling == AggregateHandling::FORCE_AGGREGATES) {
		if (depth > 0) {
			throw BinderException("Having clause cannot reference column in correlated subquery and group by all");
		}
		auto expr = duckdb::SelectBinder::BindExpression(expr_ptr, depth);
		if (expr.HasError()) {
			return expr;
		}
		auto group_ref = make_unique<BoundColumnRefExpression>(
		    expr.expression->return_type, ColumnBinding(node.group_index, node.groups.group_expressions.size()));
		node.groups.group_expressions.push_back(std::move(expr.expression));
		return BindResult(std::move(group_ref));
	}
	return BindResult(StringUtil::Format(
	    "column %s must appear in the GROUP BY clause or be used in an aggregate function", expr.ToString()));
}

BindResult HavingBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
	// check if the expression binds to one of the groups
	auto group_index = TryBindGroup(expr, depth);
	if (group_index != DConstants::INVALID_INDEX) {
		return BindGroup(expr, depth, group_index);
	}
	switch (expr.expression_class) {
	case ExpressionClass::WINDOW:
		return BindResult("HAVING clause cannot contain window functions!");
	case ExpressionClass::COLUMN_REF:
		return BindColumnRef(expr_ptr, depth, root_expression);
	default:
		return duckdb::SelectBinder::BindExpression(expr_ptr, depth);
	}
}

} // namespace duckdb


namespace duckdb {

IndexBinder::IndexBinder(Binder &binder, ClientContext &context) : ExpressionBinder(binder, context) {
}

BindResult IndexBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
	switch (expr.expression_class) {
	case ExpressionClass::WINDOW:
		return BindResult("window functions are not allowed in index expressions");
	case ExpressionClass::SUBQUERY:
		return BindResult("cannot use subquery in index expressions");
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

string IndexBinder::UnsupportedAggregateMessage() {
	return "aggregate functions are not allowed in index expressions";
}

} // namespace duckdb




namespace duckdb {

InsertBinder::InsertBinder(Binder &binder, ClientContext &context) : ExpressionBinder(binder, context) {
}

BindResult InsertBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::DEFAULT:
		return BindResult("DEFAULT is not allowed here!");
	case ExpressionClass::WINDOW:
		return BindResult("INSERT statement cannot contain window functions!");
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

string InsertBinder::UnsupportedAggregateMessage() {
	return "INSERT statement cannot contain aggregates!";
}

} // namespace duckdb






namespace duckdb {

LateralBinder::LateralBinder(Binder &binder, ClientContext &context) : ExpressionBinder(binder, context) {
}

void LateralBinder::ExtractCorrelatedColumns(Expression &expr) {
	if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
		auto &bound_colref = (BoundColumnRefExpression &)expr;
		if (bound_colref.depth > 0) {
			// add the correlated column info
			CorrelatedColumnInfo info(bound_colref);
			if (std::find(correlated_columns.begin(), correlated_columns.end(), info) == correlated_columns.end()) {
				correlated_columns.push_back(std::move(info));
			}
		}
	}
	ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) { ExtractCorrelatedColumns(child); });
}

BindResult LateralBinder::BindColumnRef(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	if (depth == 0) {
		throw InternalException("Lateral binder can only bind correlated columns");
	}
	auto result = ExpressionBinder::BindExpression(expr_ptr, depth);
	if (result.HasError()) {
		return result;
	}
	if (depth > 1) {
		throw BinderException("Nested lateral joins are not supported yet");
	}
	ExtractCorrelatedColumns(*result.expression);
	return result;
}

vector<CorrelatedColumnInfo> LateralBinder::ExtractCorrelatedColumns(Binder &binder) {
	auto all_correlated_columns = binder.correlated_columns;
	for (auto &correlated : correlated_columns) {
		auto entry = std::find(binder.correlated_columns.begin(), binder.correlated_columns.end(), correlated);
		if (entry == binder.correlated_columns.end()) {
			throw InternalException("Lateral Binder: could not find correlated column in binder");
		}
		binder.correlated_columns.erase(entry);
	}
	return all_correlated_columns;
}

BindResult LateralBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::DEFAULT:
		return BindResult("LATERAL join cannot contain DEFAULT clause");
	case ExpressionClass::WINDOW:
		return BindResult("LATERAL join cannot contain window functions!");
	case ExpressionClass::COLUMN_REF:
		return BindColumnRef(expr_ptr, depth, root_expression);
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

string LateralBinder::UnsupportedAggregateMessage() {
	return "LATERAL join cannot contain aggregates!";
}

class ExpressionDepthReducer : public LogicalOperatorVisitor {
public:
	explicit ExpressionDepthReducer(const vector<CorrelatedColumnInfo> &correlated) : correlated_columns(correlated) {
	}

protected:
	void ReduceColumnRefDepth(BoundColumnRefExpression &expr) {
		// don't need to reduce this
		if (expr.depth == 0) {
			return;
		}
		for (auto &correlated : correlated_columns) {
			if (correlated.binding == expr.binding) {
				D_ASSERT(expr.depth > 1);
				expr.depth--;
				break;
			}
		}
	}

	unique_ptr<Expression> VisitReplace(BoundColumnRefExpression &expr, unique_ptr<Expression> *expr_ptr) override {
		ReduceColumnRefDepth(expr);
		return nullptr;
	}

	void ReduceExpressionSubquery(BoundSubqueryExpression &expr) {
		for (auto &s_correlated : expr.binder->correlated_columns) {
			for (auto &correlated : correlated_columns) {
				if (correlated == s_correlated) {
					s_correlated.depth--;
					break;
				}
			}
		}
	}

	void ReduceExpressionDepth(Expression &expr) {
		if (expr.GetExpressionType() == ExpressionType::BOUND_COLUMN_REF) {
			ReduceColumnRefDepth((BoundColumnRefExpression &)expr);
		}
		if (expr.GetExpressionClass() == ExpressionClass::BOUND_SUBQUERY) {
			ReduceExpressionSubquery((BoundSubqueryExpression &)expr);
		}
	}

	unique_ptr<Expression> VisitReplace(BoundSubqueryExpression &expr, unique_ptr<Expression> *expr_ptr) override {
		ReduceExpressionSubquery(expr);
		ExpressionIterator::EnumerateQueryNodeChildren(
		    *expr.subquery, [&](Expression &child_expr) { ReduceExpressionDepth(child_expr); });
		return nullptr;
	}

	const vector<CorrelatedColumnInfo> &correlated_columns;
};

void LateralBinder::ReduceExpressionDepth(LogicalOperator &op, const vector<CorrelatedColumnInfo> &correlated) {
	ExpressionDepthReducer depth_reducer(correlated);
	depth_reducer.VisitOperator(op);
}

} // namespace duckdb












namespace duckdb {

OrderBinder::OrderBinder(vector<Binder *> binders, idx_t projection_index, case_insensitive_map_t<idx_t> &alias_map,
                         expression_map_t<idx_t> &projection_map, idx_t max_count)
    : binders(std::move(binders)), projection_index(projection_index), max_count(max_count), extra_list(nullptr),
      alias_map(alias_map), projection_map(projection_map) {
}
OrderBinder::OrderBinder(vector<Binder *> binders, idx_t projection_index, SelectNode &node,
                         case_insensitive_map_t<idx_t> &alias_map, expression_map_t<idx_t> &projection_map)
    : binders(std::move(binders)), projection_index(projection_index), alias_map(alias_map),
      projection_map(projection_map) {
	this->max_count = node.select_list.size();
	this->extra_list = &node.select_list;
}

unique_ptr<Expression> OrderBinder::CreateProjectionReference(ParsedExpression &expr, idx_t index) {
	string alias;
	if (extra_list && index < extra_list->size()) {
		alias = extra_list->at(index)->ToString();
	} else {
		if (!expr.alias.empty()) {
			alias = expr.alias;
		}
	}
	return make_unique<BoundColumnRefExpression>(std::move(alias), LogicalType::INVALID,
	                                             ColumnBinding(projection_index, index));
}

unique_ptr<Expression> OrderBinder::CreateExtraReference(unique_ptr<ParsedExpression> expr) {
	if (!extra_list) {
		throw InternalException("CreateExtraReference called without extra_list");
	}
	auto result = CreateProjectionReference(*expr, extra_list->size());
	extra_list->push_back(std::move(expr));
	return result;
}

unique_ptr<Expression> OrderBinder::BindConstant(ParsedExpression &expr, const Value &val) {
	// ORDER BY a constant
	if (!val.type().IsIntegral()) {
		// non-integral expression, we just leave the constant here.
		// ORDER BY <constant> has no effect
		// CONTROVERSIAL: maybe we should throw an error
		return nullptr;
	}
	// INTEGER constant: we use the integer as an index into the select list (e.g. ORDER BY 1)
	auto index = (idx_t)val.GetValue<int64_t>();
	if (index < 1 || index > max_count) {
		throw BinderException("ORDER term out of range - should be between 1 and %lld", (idx_t)max_count);
	}
	return CreateProjectionReference(expr, index - 1);
}

unique_ptr<Expression> OrderBinder::Bind(unique_ptr<ParsedExpression> expr) {
	// in the ORDER BY clause we do not bind children
	// we bind ONLY to the select list
	// if there is no matching entry in the SELECT list already, we add the expression to the SELECT list and refer the
	// new expression the new entry will then be bound later during the binding of the SELECT list we also don't do type
	// resolution here: this only happens after the SELECT list has been bound
	switch (expr->expression_class) {
	case ExpressionClass::CONSTANT: {
		// ORDER BY constant
		// is the ORDER BY expression a constant integer? (e.g. ORDER BY 1)
		auto &constant = (ConstantExpression &)*expr;
		return BindConstant(*expr, constant.value);
	}
	case ExpressionClass::COLUMN_REF: {
		// COLUMN REF expression
		// check if we can bind it to an alias in the select list
		auto &colref = (ColumnRefExpression &)*expr;
		// if there is an explicit table name we can't bind to an alias
		if (colref.IsQualified()) {
			break;
		}
		// check the alias list
		auto entry = alias_map.find(colref.column_names[0]);
		if (entry != alias_map.end()) {
			// it does! point it to that entry
			return CreateProjectionReference(*expr, entry->second);
		}
		break;
	}
	case ExpressionClass::POSITIONAL_REFERENCE: {
		auto &posref = (PositionalReferenceExpression &)*expr;
		if (posref.index < 1 || posref.index > max_count) {
			throw BinderException("ORDER term out of range - should be between 1 and %lld", (idx_t)max_count);
		}
		return CreateProjectionReference(*expr, posref.index - 1);
	}
	case ExpressionClass::PARAMETER: {
		throw ParameterNotAllowedException("Parameter not supported in ORDER BY clause");
	}
	default:
		break;
	}
	// general case
	// first bind the table names of this entry
	for (auto &binder : binders) {
		ExpressionBinder::QualifyColumnNames(*binder, expr);
	}
	// first check if the ORDER BY clause already points to an entry in the projection list
	auto entry = projection_map.find(expr.get());
	if (entry != projection_map.end()) {
		if (entry->second == DConstants::INVALID_INDEX) {
			throw BinderException("Ambiguous reference to column");
		}
		// there is a matching entry in the projection list
		// just point to that entry
		return CreateProjectionReference(*expr, entry->second);
	}
	if (!extra_list) {
		// no extra list specified: we cannot push an extra ORDER BY clause
		throw BinderException("Could not ORDER BY column \"%s\": add the expression/function to every SELECT, or move "
		                      "the UNION into a FROM clause.",
		                      expr->ToString());
	}
	// otherwise we need to push the ORDER BY entry into the select list
	return CreateExtraReference(std::move(expr));
}

} // namespace duckdb








namespace duckdb {

QualifyBinder::QualifyBinder(Binder &binder, ClientContext &context, BoundSelectNode &node, BoundGroupInformation &info,
                             case_insensitive_map_t<idx_t> &alias_map)
    : SelectBinder(binder, context, node, info), column_alias_binder(node, alias_map) {
	target_type = LogicalType(LogicalTypeId::BOOLEAN);
}

BindResult QualifyBinder::BindColumnRef(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = (ColumnRefExpression &)**expr_ptr;
	auto result = duckdb::SelectBinder::BindExpression(expr_ptr, depth);
	if (!result.HasError()) {
		return result;
	}

	auto alias_result = column_alias_binder.BindAlias(*this, expr, depth, root_expression);
	if (!alias_result.HasError()) {
		return alias_result;
	}

	return BindResult(StringUtil::Format("Referenced column %s not found in FROM clause and can't find in alias map.",
	                                     expr.ToString()));
}

BindResult QualifyBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
	// check if the expression binds to one of the groups
	auto group_index = TryBindGroup(expr, depth);
	if (group_index != DConstants::INVALID_INDEX) {
		return BindGroup(expr, depth, group_index);
	}
	switch (expr.expression_class) {
	case ExpressionClass::WINDOW:
		return BindWindow((WindowExpression &)expr, depth);
	case ExpressionClass::COLUMN_REF:
		return BindColumnRef(expr_ptr, depth, root_expression);
	default:
		return duckdb::SelectBinder::BindExpression(expr_ptr, depth);
	}
}

} // namespace duckdb


namespace duckdb {

RelationBinder::RelationBinder(Binder &binder, ClientContext &context, string op)
    : ExpressionBinder(binder, context), op(std::move(op)) {
}

BindResult RelationBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
	switch (expr.expression_class) {
	case ExpressionClass::AGGREGATE:
		return BindResult("aggregate functions are not allowed in " + op);
	case ExpressionClass::DEFAULT:
		return BindResult(op + " cannot contain DEFAULT clause");
	case ExpressionClass::SUBQUERY:
		return BindResult("subqueries are not allowed in " + op);
	case ExpressionClass::WINDOW:
		return BindResult("window functions are not allowed in " + op);
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

string RelationBinder::UnsupportedAggregateMessage() {
	return "aggregate functions are not allowed in " + op;
}

} // namespace duckdb




namespace duckdb {

ReturningBinder::ReturningBinder(Binder &binder, ClientContext &context) : ExpressionBinder(binder, context) {
}

BindResult ReturningBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::SUBQUERY:
		return BindResult("SUBQUERY is not supported in returning statements");
	case ExpressionClass::BOUND_SUBQUERY:
		return BindResult("BOUND SUBQUERY is not supported in returning statements");
	case ExpressionClass::COLUMN_REF:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

} // namespace duckdb













namespace duckdb {

SelectBinder::SelectBinder(Binder &binder, ClientContext &context, BoundSelectNode &node, BoundGroupInformation &info,
                           case_insensitive_map_t<idx_t> alias_map)
    : ExpressionBinder(binder, context), inside_window(false), node(node), info(info), alias_map(std::move(alias_map)) {
}

SelectBinder::SelectBinder(Binder &binder, ClientContext &context, BoundSelectNode &node, BoundGroupInformation &info)
    : SelectBinder(binder, context, node, info, case_insensitive_map_t<idx_t>()) {
}

BindResult SelectBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
	// check if the expression binds to one of the groups
	auto group_index = TryBindGroup(expr, depth);
	if (group_index != DConstants::INVALID_INDEX) {
		return BindGroup(expr, depth, group_index);
	}
	switch (expr.expression_class) {
	case ExpressionClass::COLUMN_REF:
		return BindColumnRef(expr_ptr, depth);
	case ExpressionClass::DEFAULT:
		return BindResult("SELECT clause cannot contain DEFAULT clause");
	case ExpressionClass::WINDOW:
		return BindWindow((WindowExpression &)expr, depth);
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

idx_t SelectBinder::TryBindGroup(ParsedExpression &expr, idx_t depth) {
	// first check the group alias map, if expr is a ColumnRefExpression
	if (expr.type == ExpressionType::COLUMN_REF) {
		auto &colref = (ColumnRefExpression &)expr;
		if (!colref.IsQualified()) {
			auto alias_entry = info.alias_map.find(colref.column_names[0]);
			if (alias_entry != info.alias_map.end()) {
				// found entry!
				return alias_entry->second;
			}
		}
	}
	// no alias reference found
	// check the list of group columns for a match
	auto entry = info.map.find(&expr);
	if (entry != info.map.end()) {
		return entry->second;
	}
#ifdef DEBUG
	for (auto entry : info.map) {
		D_ASSERT(!entry.first->Equals(&expr));
		D_ASSERT(!expr.Equals(entry.first));
	}
#endif
	return DConstants::INVALID_INDEX;
}

BindResult SelectBinder::BindColumnRef(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth) {
	// first try to bind the column reference regularly
	auto result = ExpressionBinder::BindExpression(expr_ptr, depth);
	if (!result.HasError()) {
		return result;
	}
	// binding failed
	// check in the alias map
	auto &colref = (ColumnRefExpression &)**expr_ptr;
	if (!colref.IsQualified()) {
		auto alias_entry = alias_map.find(colref.column_names[0]);
		if (alias_entry != alias_map.end()) {
			// found entry!
			auto index = alias_entry->second;
			if (index >= node.select_list.size()) {
				throw BinderException("Column \"%s\" referenced that exists in the SELECT clause - but this column "
				                      "cannot be referenced before it is defined",
				                      colref.column_names[0]);
			}
			if (node.select_list[index]->HasSideEffects()) {
				throw BinderException("Alias \"%s\" referenced in a SELECT clause - but the expression has side "
				                      "effects. This is not yet supported.",
				                      colref.column_names[0]);
			}
			if (node.select_list[index]->HasSubquery()) {
				throw BinderException("Alias \"%s\" referenced in a SELECT clause - but the expression has a subquery."
				                      " This is not yet supported.",
				                      colref.column_names[0]);
			}
			auto result = BindResult(node.select_list[index]->Copy());
			if (result.expression->type == ExpressionType::BOUND_COLUMN_REF) {
				auto &result_expr = (BoundColumnRefExpression &)*result.expression;
				result_expr.depth = depth;
			}
			return result;
		}
	}
	// entry was not found in the alias map: return the original error
	return result;
}

BindResult SelectBinder::BindGroupingFunction(OperatorExpression &op, idx_t depth) {
	if (op.children.empty()) {
		throw InternalException("GROUPING requires at least one child");
	}
	if (node.groups.group_expressions.empty()) {
		return BindResult(binder.FormatError(op, "GROUPING statement cannot be used without groups"));
	}
	if (op.children.size() >= 64) {
		return BindResult(binder.FormatError(op, "GROUPING statement cannot have more than 64 groups"));
	}
	vector<idx_t> group_indexes;
	group_indexes.reserve(op.children.size());
	for (auto &child : op.children) {
		ExpressionBinder::QualifyColumnNames(binder, child);
		auto idx = TryBindGroup(*child, depth);
		if (idx == DConstants::INVALID_INDEX) {
			return BindResult(binder.FormatError(
			    op, StringUtil::Format("GROUPING child \"%s\" must be a grouping column", child->GetName())));
		}
		group_indexes.push_back(idx);
	}
	auto col_idx = node.grouping_functions.size();
	node.grouping_functions.push_back(std::move(group_indexes));
	return BindResult(make_unique<BoundColumnRefExpression>(op.GetName(), LogicalType::BIGINT,
	                                                        ColumnBinding(node.groupings_index, col_idx), depth));
}

BindResult SelectBinder::BindGroup(ParsedExpression &expr, idx_t depth, idx_t group_index) {
	auto &group = node.groups.group_expressions[group_index];
	return BindResult(make_unique<BoundColumnRefExpression>(expr.GetName(), group->return_type,
	                                                        ColumnBinding(node.group_index, group_index), depth));
}

} // namespace duckdb





namespace duckdb {

TableFunctionBinder::TableFunctionBinder(Binder &binder, ClientContext &context) : ExpressionBinder(binder, context) {
}

BindResult TableFunctionBinder::BindColumnReference(ColumnRefExpression &expr, idx_t depth) {

	// if this is a lambda parameters, then we temporarily add a BoundLambdaRef,
	// which we capture and remove later
	if (lambda_bindings) {
		auto &colref = (ColumnRefExpression &)expr;
		for (idx_t i = 0; i < lambda_bindings->size(); i++) {
			if (colref.GetColumnName() == (*lambda_bindings)[i].dummy_name) {
				return (*lambda_bindings)[i].Bind(colref, i, depth);
			}
		}
	}

	auto result_name = StringUtil::Join(expr.column_names, ".");
	return BindResult(make_unique<BoundConstantExpression>(Value(result_name)));
}

BindResult TableFunctionBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth,
                                               bool root_expression) {
	auto &expr = **expr_ptr;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::COLUMN_REF:
		return BindColumnReference((ColumnRefExpression &)expr, depth);
	case ExpressionClass::SUBQUERY:
		throw BinderException("Table function cannot contain subqueries");
	case ExpressionClass::DEFAULT:
		return BindResult("Table function cannot contain DEFAULT clause");
	case ExpressionClass::WINDOW:
		return BindResult("Table function cannot contain window functions!");
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

string TableFunctionBinder::UnsupportedAggregateMessage() {
	return "Table function cannot contain aggregates!";
}

} // namespace duckdb


namespace duckdb {

UpdateBinder::UpdateBinder(Binder &binder, ClientContext &context) : ExpressionBinder(binder, context) {
}

BindResult UpdateBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
	switch (expr.expression_class) {
	case ExpressionClass::WINDOW:
		return BindResult("window functions are not allowed in UPDATE");
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

string UpdateBinder::UnsupportedAggregateMessage() {
	return "aggregate functions are not allowed in UPDATE";
}

} // namespace duckdb



namespace duckdb {

WhereBinder::WhereBinder(Binder &binder, ClientContext &context, ColumnAliasBinder *column_alias_binder)
    : ExpressionBinder(binder, context), column_alias_binder(column_alias_binder) {
	target_type = LogicalType(LogicalTypeId::BOOLEAN);
}

BindResult WhereBinder::BindColumnRef(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = (ColumnRefExpression &)**expr_ptr;
	auto result = ExpressionBinder::BindExpression(expr_ptr, depth);
	if (!result.HasError() || !column_alias_binder) {
		return result;
	}

	BindResult alias_result = column_alias_binder->BindAlias(*this, expr, depth, root_expression);
	// This code path cannot be exercised at thispoint. #1547 might change that.
	if (!alias_result.HasError()) {
		return alias_result;
	}

	return result;
}

BindResult WhereBinder::BindExpression(unique_ptr<ParsedExpression> *expr_ptr, idx_t depth, bool root_expression) {
	auto &expr = **expr_ptr;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::DEFAULT:
		return BindResult("WHERE clause cannot contain DEFAULT clause");
	case ExpressionClass::WINDOW:
		return BindResult("WHERE clause cannot contain window functions!");
	case ExpressionClass::COLUMN_REF:
		return BindColumnRef(expr_ptr, depth, root_expression);
	default:
		return ExpressionBinder::BindExpression(expr_ptr, depth);
	}
}

string WhereBinder::UnsupportedAggregateMessage() {
	return "WHERE clause cannot contain aggregates!";
}

} // namespace duckdb













namespace duckdb {

ExpressionBinder::ExpressionBinder(Binder &binder, ClientContext &context, bool replace_binder)
    : binder(binder), context(context), stored_binder(nullptr) {
	if (replace_binder) {
		stored_binder = binder.GetActiveBinder();
		binder.SetActiveBinder(this);
	} else {
		binder.PushExpressionBinder(this);
	}
}

ExpressionBinder::~ExpressionBinder() {
	if (binder.HasActiveBinder()) {
		if (stored_binder) {
			binder.SetActiveBinder(stored_binder);
		} else {
			binder.PopExpressionBinder();
		}
	}
}

BindResult ExpressionBinder::BindExpression(unique_ptr<ParsedExpression> *expr, idx_t depth, bool root_expression) {
	auto &expr_ref = **expr;
	switch (expr_ref.expression_class) {
	case ExpressionClass::BETWEEN:
		return BindExpression((BetweenExpression &)expr_ref, depth);
	case ExpressionClass::CASE:
		return BindExpression((CaseExpression &)expr_ref, depth);
	case ExpressionClass::CAST:
		return BindExpression((CastExpression &)expr_ref, depth);
	case ExpressionClass::COLLATE:
		return BindExpression((CollateExpression &)expr_ref, depth);
	case ExpressionClass::COLUMN_REF:
		return BindExpression((ColumnRefExpression &)expr_ref, depth);
	case ExpressionClass::COMPARISON:
		return BindExpression((ComparisonExpression &)expr_ref, depth);
	case ExpressionClass::CONJUNCTION:
		return BindExpression((ConjunctionExpression &)expr_ref, depth);
	case ExpressionClass::CONSTANT:
		return BindExpression((ConstantExpression &)expr_ref, depth);
	case ExpressionClass::FUNCTION:
		// binding function expression has extra parameter needed for macro's
		return BindExpression((FunctionExpression &)expr_ref, depth, expr);
	case ExpressionClass::LAMBDA:
		return BindExpression((LambdaExpression &)expr_ref, depth, false, LogicalTypeId::INVALID);
	case ExpressionClass::OPERATOR:
		return BindExpression((OperatorExpression &)expr_ref, depth);
	case ExpressionClass::SUBQUERY:
		return BindExpression((SubqueryExpression &)expr_ref, depth);
	case ExpressionClass::PARAMETER:
		return BindExpression((ParameterExpression &)expr_ref, depth);
	case ExpressionClass::POSITIONAL_REFERENCE:
		return BindExpression((PositionalReferenceExpression &)expr_ref, depth);
	case ExpressionClass::STAR:
		return BindResult(binder.FormatError(expr_ref, "STAR expression is not supported here"));
	default:
		throw NotImplementedException("Unimplemented expression class");
	}
}

bool ExpressionBinder::BindCorrelatedColumns(unique_ptr<ParsedExpression> &expr) {
	// try to bind in one of the outer queries, if the binding error occurred in a subquery
	auto &active_binders = binder.GetActiveBinders();
	// make a copy of the set of binders, so we can restore it later
	auto binders = active_binders;
	active_binders.pop_back();
	idx_t depth = 1;
	bool success = false;
	while (!active_binders.empty()) {
		auto &next_binder = active_binders.back();
		ExpressionBinder::QualifyColumnNames(next_binder->binder, expr);
		auto bind_result = next_binder->Bind(&expr, depth);
		if (bind_result.empty()) {
			success = true;
			break;
		}
		depth++;
		active_binders.pop_back();
	}
	active_binders = binders;
	return success;
}

void ExpressionBinder::BindChild(unique_ptr<ParsedExpression> &expr, idx_t depth, string &error) {
	if (expr) {
		string bind_error = Bind(&expr, depth);
		if (error.empty()) {
			error = bind_error;
		}
	}
}

void ExpressionBinder::ExtractCorrelatedExpressions(Binder &binder, Expression &expr) {
	if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
		auto &bound_colref = (BoundColumnRefExpression &)expr;
		if (bound_colref.depth > 0) {
			binder.AddCorrelatedColumn(CorrelatedColumnInfo(bound_colref));
		}
	}
	ExpressionIterator::EnumerateChildren(expr,
	                                      [&](Expression &child) { ExtractCorrelatedExpressions(binder, child); });
}

bool ExpressionBinder::ContainsType(const LogicalType &type, LogicalTypeId target) {
	if (type.id() == target) {
		return true;
	}
	switch (type.id()) {
	case LogicalTypeId::STRUCT: {
		auto child_count = StructType::GetChildCount(type);
		for (idx_t i = 0; i < child_count; i++) {
			if (ContainsType(StructType::GetChildType(type, i), target)) {
				return true;
			}
		}
		return false;
	}
	case LogicalTypeId::UNION: {
		auto member_count = UnionType::GetMemberCount(type);
		for (idx_t i = 0; i < member_count; i++) {
			if (ContainsType(UnionType::GetMemberType(type, i), target)) {
				return true;
			}
		}
		return false;
	}
	case LogicalTypeId::LIST:
	case LogicalTypeId::MAP:
		return ContainsType(ListType::GetChildType(type), target);
	default:
		return false;
	}
}

LogicalType ExpressionBinder::ExchangeType(const LogicalType &type, LogicalTypeId target, LogicalType new_type) {
	if (type.id() == target) {
		return new_type;
	}
	switch (type.id()) {
	case LogicalTypeId::STRUCT: {
		// we make a copy of the child types of the struct here
		auto child_types = StructType::GetChildTypes(type);
		for (auto &child_type : child_types) {
			child_type.second = ExchangeType(child_type.second, target, new_type);
		}
		return LogicalType::STRUCT(std::move(child_types));
	}
	case LogicalTypeId::UNION: {
		auto member_types = UnionType::CopyMemberTypes(type);
		for (auto &member_type : member_types) {
			member_type.second = ExchangeType(member_type.second, target, new_type);
		}
		return LogicalType::UNION(std::move(member_types));
	}
	case LogicalTypeId::LIST:
		return LogicalType::LIST(ExchangeType(ListType::GetChildType(type), target, new_type));
	case LogicalTypeId::MAP:
		return LogicalType::MAP(ExchangeType(ListType::GetChildType(type), target, new_type));
	default:
		return type;
	}
}

bool ExpressionBinder::ContainsNullType(const LogicalType &type) {
	return ContainsType(type, LogicalTypeId::SQLNULL);
}

LogicalType ExpressionBinder::ExchangeNullType(const LogicalType &type) {
	return ExchangeType(type, LogicalTypeId::SQLNULL, LogicalType::INTEGER);
}

unique_ptr<Expression> ExpressionBinder::Bind(unique_ptr<ParsedExpression> &expr, LogicalType *result_type,
                                              bool root_expression) {
	// bind the main expression
	auto error_msg = Bind(&expr, 0, root_expression);
	if (!error_msg.empty()) {
		// failed to bind: try to bind correlated columns in the expression (if any)
		bool success = BindCorrelatedColumns(expr);
		if (!success) {
			throw BinderException(error_msg);
		}
		auto bound_expr = (BoundExpression *)expr.get();
		ExtractCorrelatedExpressions(binder, *bound_expr->expr);
	}
	D_ASSERT(expr->expression_class == ExpressionClass::BOUND_EXPRESSION);
	auto bound_expr = (BoundExpression *)expr.get();
	unique_ptr<Expression> result = std::move(bound_expr->expr);
	if (target_type.id() != LogicalTypeId::INVALID) {
		// the binder has a specific target type: add a cast to that type
		result = BoundCastExpression::AddCastToType(context, std::move(result), target_type);
	} else {
		if (!binder.can_contain_nulls) {
			// SQL NULL type is only used internally in the binder
			// cast to INTEGER if we encounter it outside of the binder
			if (ContainsNullType(result->return_type)) {
				auto result_type = ExchangeNullType(result->return_type);
				result = BoundCastExpression::AddCastToType(context, std::move(result), result_type);
			}
		}
		if (result->return_type.id() == LogicalTypeId::UNKNOWN) {
			throw ParameterNotResolvedException();
		}
	}
	if (result_type) {
		*result_type = result->return_type;
	}
	return result;
}

string ExpressionBinder::Bind(unique_ptr<ParsedExpression> *expr, idx_t depth, bool root_expression) {
	// bind the node, but only if it has not been bound yet
	auto &expression = **expr;
	auto alias = expression.alias;
	if (expression.GetExpressionClass() == ExpressionClass::BOUND_EXPRESSION) {
		// already bound, don't bind it again
		return string();
	}
	// bind the expression
	BindResult result = BindExpression(expr, depth, root_expression);
	if (result.HasError()) {
		return result.error;
	}
	// successfully bound: replace the node with a BoundExpression
	*expr = make_unique<BoundExpression>(std::move(result.expression));
	auto be = (BoundExpression *)expr->get();
	D_ASSERT(be);
	be->alias = alias;
	if (!alias.empty()) {
		be->expr->alias = alias;
	}
	return string();
}

} // namespace duckdb









namespace duckdb {

void ExpressionIterator::EnumerateChildren(const Expression &expr,
                                           const std::function<void(const Expression &child)> &callback) {
	EnumerateChildren((Expression &)expr, [&](unique_ptr<Expression> &child) { callback(*child); });
}

void ExpressionIterator::EnumerateChildren(Expression &expr, const std::function<void(Expression &child)> &callback) {
	EnumerateChildren(expr, [&](unique_ptr<Expression> &child) { callback(*child); });
}

void ExpressionIterator::EnumerateChildren(Expression &expr,
                                           const std::function<void(unique_ptr<Expression> &child)> &callback) {
	switch (expr.expression_class) {
	case ExpressionClass::BOUND_AGGREGATE: {
		auto &aggr_expr = (BoundAggregateExpression &)expr;
		for (auto &child : aggr_expr.children) {
			callback(child);
		}
		if (aggr_expr.filter) {
			callback(aggr_expr.filter);
		}
		break;
	}
	case ExpressionClass::BOUND_BETWEEN: {
		auto &between_expr = (BoundBetweenExpression &)expr;
		callback(between_expr.input);
		callback(between_expr.lower);
		callback(between_expr.upper);
		break;
	}
	case ExpressionClass::BOUND_CASE: {
		auto &case_expr = (BoundCaseExpression &)expr;
		for (auto &case_check : case_expr.case_checks) {
			callback(case_check.when_expr);
			callback(case_check.then_expr);
		}
		callback(case_expr.else_expr);
		break;
	}
	case ExpressionClass::BOUND_CAST: {
		auto &cast_expr = (BoundCastExpression &)expr;
		callback(cast_expr.child);
		break;
	}
	case ExpressionClass::BOUND_COMPARISON: {
		auto &comp_expr = (BoundComparisonExpression &)expr;
		callback(comp_expr.left);
		callback(comp_expr.right);
		break;
	}
	case ExpressionClass::BOUND_CONJUNCTION: {
		auto &conj_expr = (BoundConjunctionExpression &)expr;
		for (auto &child : conj_expr.children) {
			callback(child);
		}
		break;
	}
	case ExpressionClass::BOUND_FUNCTION: {
		auto &func_expr = (BoundFunctionExpression &)expr;
		for (auto &child : func_expr.children) {
			callback(child);
		}
		break;
	}
	case ExpressionClass::BOUND_OPERATOR: {
		auto &op_expr = (BoundOperatorExpression &)expr;
		for (auto &child : op_expr.children) {
			callback(child);
		}
		break;
	}
	case ExpressionClass::BOUND_SUBQUERY: {
		auto &subquery_expr = (BoundSubqueryExpression &)expr;
		if (subquery_expr.child) {
			callback(subquery_expr.child);
		}
		break;
	}
	case ExpressionClass::BOUND_WINDOW: {
		auto &window_expr = (BoundWindowExpression &)expr;
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
	case ExpressionClass::BOUND_UNNEST: {
		auto &unnest_expr = (BoundUnnestExpression &)expr;
		callback(unnest_expr.child);
		break;
	}
	case ExpressionClass::BOUND_COLUMN_REF:
	case ExpressionClass::BOUND_LAMBDA_REF:
	case ExpressionClass::BOUND_CONSTANT:
	case ExpressionClass::BOUND_DEFAULT:
	case ExpressionClass::BOUND_PARAMETER:
	case ExpressionClass::BOUND_REF:
		// these node types have no children
		break;
	default:
		throw InternalException("ExpressionIterator used on unbound expression");
	}
}

void ExpressionIterator::EnumerateExpression(unique_ptr<Expression> &expr,
                                             const std::function<void(Expression &child)> &callback) {
	if (!expr) {
		return;
	}
	callback(*expr);
	ExpressionIterator::EnumerateChildren(*expr,
	                                      [&](unique_ptr<Expression> &child) { EnumerateExpression(child, callback); });
}

void ExpressionIterator::EnumerateTableRefChildren(BoundTableRef &ref,
                                                   const std::function<void(Expression &child)> &callback) {
	switch (ref.type) {
	case TableReferenceType::EXPRESSION_LIST: {
		auto &bound_expr_list = (BoundExpressionListRef &)ref;
		for (auto &expr_list : bound_expr_list.values) {
			for (auto &expr : expr_list) {
				EnumerateExpression(expr, callback);
			}
		}
		break;
	}
	case TableReferenceType::JOIN: {
		auto &bound_join = (BoundJoinRef &)ref;
		if (bound_join.condition) {
			EnumerateExpression(bound_join.condition, callback);
		}
		EnumerateTableRefChildren(*bound_join.left, callback);
		EnumerateTableRefChildren(*bound_join.right, callback);
		break;
	}
	case TableReferenceType::SUBQUERY: {
		auto &bound_subquery = (BoundSubqueryRef &)ref;
		EnumerateQueryNodeChildren(*bound_subquery.subquery, callback);
		break;
	}
	case TableReferenceType::TABLE_FUNCTION:
	case TableReferenceType::EMPTY:
	case TableReferenceType::BASE_TABLE:
	case TableReferenceType::CTE:
		break;
	default:
		throw NotImplementedException("Unimplemented table reference type in ExpressionIterator");
	}
}

void ExpressionIterator::EnumerateQueryNodeChildren(BoundQueryNode &node,
                                                    const std::function<void(Expression &child)> &callback) {
	switch (node.type) {
	case QueryNodeType::SET_OPERATION_NODE: {
		auto &bound_setop = (BoundSetOperationNode &)node;
		EnumerateQueryNodeChildren(*bound_setop.left, callback);
		EnumerateQueryNodeChildren(*bound_setop.right, callback);
		break;
	}
	case QueryNodeType::RECURSIVE_CTE_NODE: {
		auto &cte_node = (BoundRecursiveCTENode &)node;
		EnumerateQueryNodeChildren(*cte_node.left, callback);
		EnumerateQueryNodeChildren(*cte_node.right, callback);
		break;
	}
	case QueryNodeType::SELECT_NODE: {
		auto &bound_select = (BoundSelectNode &)node;
		for (idx_t i = 0; i < bound_select.select_list.size(); i++) {
			EnumerateExpression(bound_select.select_list[i], callback);
		}
		EnumerateExpression(bound_select.where_clause, callback);
		for (idx_t i = 0; i < bound_select.groups.group_expressions.size(); i++) {
			EnumerateExpression(bound_select.groups.group_expressions[i], callback);
		}
		EnumerateExpression(bound_select.having, callback);
		for (idx_t i = 0; i < bound_select.aggregates.size(); i++) {
			EnumerateExpression(bound_select.aggregates[i], callback);
		}
		for (idx_t i = 0; i < bound_select.unnests.size(); i++) {
			EnumerateExpression(bound_select.unnests[i], callback);
		}
		for (idx_t i = 0; i < bound_select.windows.size(); i++) {
			EnumerateExpression(bound_select.windows[i], callback);
		}
		if (bound_select.from_table) {
			EnumerateTableRefChildren(*bound_select.from_table, callback);
		}
		break;
	}
	default:
		throw NotImplementedException("Unimplemented query node in ExpressionIterator");
	}
	for (idx_t i = 0; i < node.modifiers.size(); i++) {
		switch (node.modifiers[i]->type) {
		case ResultModifierType::DISTINCT_MODIFIER:
			for (auto &expr : ((BoundDistinctModifier &)*node.modifiers[i]).target_distincts) {
				EnumerateExpression(expr, callback);
			}
			break;
		case ResultModifierType::ORDER_MODIFIER:
			for (auto &order : ((BoundOrderModifier &)*node.modifiers[i]).orders) {
				EnumerateExpression(order.expression, callback);
			}
			break;
		default:
			break;
		}
	}
}

} // namespace duckdb



namespace duckdb {

ConjunctionOrFilter::ConjunctionOrFilter() : ConjunctionFilter(TableFilterType::CONJUNCTION_OR) {
}

FilterPropagateResult ConjunctionOrFilter::CheckStatistics(BaseStatistics &stats) {
	// the OR filter is true if ANY of the children is true
	D_ASSERT(!child_filters.empty());
	for (auto &filter : child_filters) {
		auto prune_result = filter->CheckStatistics(stats);
		if (prune_result == FilterPropagateResult::NO_PRUNING_POSSIBLE) {
			return FilterPropagateResult::NO_PRUNING_POSSIBLE;
		} else if (prune_result == FilterPropagateResult::FILTER_ALWAYS_TRUE) {
			return FilterPropagateResult::FILTER_ALWAYS_TRUE;
		}
	}
	return FilterPropagateResult::FILTER_ALWAYS_FALSE;
}

string ConjunctionOrFilter::ToString(const string &column_name) {
	string result;
	for (idx_t i = 0; i < child_filters.size(); i++) {
		if (i > 0) {
			result += " OR ";
		}
		result += child_filters[i]->ToString(column_name);
	}
	return result;
}

bool ConjunctionOrFilter::Equals(const TableFilter &other_p) const {
	if (!ConjunctionFilter::Equals(other_p)) {
		return false;
	}
	auto &other = (ConjunctionOrFilter &)other_p;
	if (other.child_filters.size() != child_filters.size()) {
		return false;
	}
	for (idx_t i = 0; i < other.child_filters.size(); i++) {
		if (!child_filters[i]->Equals(*other.child_filters[i])) {
			return false;
		}
	}
	return true;
}

void ConjunctionOrFilter::Serialize(FieldWriter &writer) const {
	writer.WriteSerializableList(child_filters);
}

unique_ptr<TableFilter> ConjunctionOrFilter::Deserialize(FieldReader &source) {
	auto res = make_unique<ConjunctionOrFilter>();
	res->child_filters = source.ReadRequiredSerializableList<TableFilter>();
	return std::move(res);
}

ConjunctionAndFilter::ConjunctionAndFilter() : ConjunctionFilter(TableFilterType::CONJUNCTION_AND) {
}

FilterPropagateResult ConjunctionAndFilter::CheckStatistics(BaseStatistics &stats) {
	// the AND filter is true if ALL of the children is true
	D_ASSERT(!child_filters.empty());
	auto result = FilterPropagateResult::FILTER_ALWAYS_TRUE;
	for (auto &filter : child_filters) {
		auto prune_result = filter->CheckStatistics(stats);
		if (prune_result == FilterPropagateResult::FILTER_ALWAYS_FALSE) {
			return FilterPropagateResult::FILTER_ALWAYS_FALSE;
		} else if (prune_result != result) {
			result = FilterPropagateResult::NO_PRUNING_POSSIBLE;
		}
	}
	return result;
}

string ConjunctionAndFilter::ToString(const string &column_name) {
	string result;
	for (idx_t i = 0; i < child_filters.size(); i++) {
		if (i > 0) {
			result += " AND ";
		}
		result += child_filters[i]->ToString(column_name);
	}
	return result;
}

bool ConjunctionAndFilter::Equals(const TableFilter &other_p) const {
	if (!ConjunctionFilter::Equals(other_p)) {
		return false;
	}
	auto &other = (ConjunctionAndFilter &)other_p;
	if (other.child_filters.size() != child_filters.size()) {
		return false;
	}
	for (idx_t i = 0; i < other.child_filters.size(); i++) {
		if (!child_filters[i]->Equals(*other.child_filters[i])) {
			return false;
		}
	}
	return true;
}

void ConjunctionAndFilter::Serialize(FieldWriter &writer) const {
	writer.WriteSerializableList(child_filters);
}

unique_ptr<TableFilter> ConjunctionAndFilter::Deserialize(FieldReader &source) {
	auto res = make_unique<ConjunctionAndFilter>();
	res->child_filters = source.ReadRequiredSerializableList<TableFilter>();
	return std::move(res);
}

} // namespace duckdb






namespace duckdb {

ConstantFilter::ConstantFilter(ExpressionType comparison_type_p, Value constant_p)
    : TableFilter(TableFilterType::CONSTANT_COMPARISON), comparison_type(comparison_type_p),
      constant(std::move(constant_p)) {
}

FilterPropagateResult ConstantFilter::CheckStatistics(BaseStatistics &stats) {
	D_ASSERT(constant.type().id() == stats.type.id());
	switch (constant.type().InternalType()) {
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
		return ((NumericStatistics &)stats).CheckZonemap(comparison_type, constant);
	case PhysicalType::VARCHAR:
		return ((StringStatistics &)stats).CheckZonemap(comparison_type, StringValue::Get(constant));
	default:
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
}

string ConstantFilter::ToString(const string &column_name) {
	return column_name + ExpressionTypeToOperator(comparison_type) + constant.ToString();
}

bool ConstantFilter::Equals(const TableFilter &other_p) const {
	if (!TableFilter::Equals(other_p)) {
		return false;
	}
	auto &other = (ConstantFilter &)other_p;
	return other.comparison_type == comparison_type && other.constant == constant;
}

void ConstantFilter::Serialize(FieldWriter &writer) const {
	writer.WriteField(comparison_type);
	writer.WriteSerializable(constant);
}

unique_ptr<TableFilter> ConstantFilter::Deserialize(FieldReader &source) {
	auto comparision_type = source.ReadRequired<ExpressionType>();
	auto constant = source.ReadRequiredSerializable<Value, Value>();
	return make_unique<ConstantFilter>(comparision_type, constant);
}

} // namespace duckdb



namespace duckdb {

IsNullFilter::IsNullFilter() : TableFilter(TableFilterType::IS_NULL) {
}

FilterPropagateResult IsNullFilter::CheckStatistics(BaseStatistics &stats) {
	if (!stats.CanHaveNull()) {
		// no null values are possible: always false
		return FilterPropagateResult::FILTER_ALWAYS_FALSE;
	}
	if (!stats.CanHaveNoNull()) {
		// no non-null values are possible: always true
		return FilterPropagateResult::FILTER_ALWAYS_TRUE;
	}
	return FilterPropagateResult::NO_PRUNING_POSSIBLE;
}

string IsNullFilter::ToString(const string &column_name) {
	return column_name + "IS NULL";
}

IsNotNullFilter::IsNotNullFilter() : TableFilter(TableFilterType::IS_NOT_NULL) {
}

FilterPropagateResult IsNotNullFilter::CheckStatistics(BaseStatistics &stats) {
	if (!stats.CanHaveNoNull()) {
		// no non-null values are possible: always false
		return FilterPropagateResult::FILTER_ALWAYS_FALSE;
	}
	if (!stats.CanHaveNull()) {
		// no null values are possible: always true
		return FilterPropagateResult::FILTER_ALWAYS_TRUE;
	}
	return FilterPropagateResult::NO_PRUNING_POSSIBLE;
}

string IsNotNullFilter::ToString(const string &column_name) {
	return column_name + " IS NOT NULL";
}

void IsNotNullFilter::Serialize(FieldWriter &writer) const {
}

unique_ptr<TableFilter> IsNotNullFilter::Deserialize(FieldReader &source) {
	return make_unique<IsNotNullFilter>();
}

void IsNullFilter::Serialize(FieldWriter &writer) const {
}

unique_ptr<TableFilter> IsNullFilter::Deserialize(FieldReader &source) {
	return make_unique<IsNullFilter>();
}

} // namespace duckdb









namespace duckdb {

unique_ptr<Expression> JoinCondition::CreateExpression(JoinCondition cond) {
	auto bound_comparison =
	    make_unique<BoundComparisonExpression>(cond.comparison, std::move(cond.left), std::move(cond.right));
	return std::move(bound_comparison);
}

unique_ptr<Expression> JoinCondition::CreateExpression(vector<JoinCondition> conditions) {
	unique_ptr<Expression> result;
	for (auto &cond : conditions) {
		auto expr = CreateExpression(std::move(cond));
		if (!result) {
			result = std::move(expr);
		} else {
			auto conj = make_unique<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND, std::move(expr),
			                                                    std::move(result));
			result = std::move(conj);
		}
	}
	return result;
}

//! Serializes a JoinCondition to a stand-alone binary blob
void JoinCondition::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteOptional(left);
	writer.WriteOptional(right);
	writer.WriteField<ExpressionType>(comparison);
	writer.Finalize();
}

//! Deserializes a blob back into a JoinCondition
JoinCondition JoinCondition::Deserialize(Deserializer &source, PlanDeserializationState &state) {
	auto result = JoinCondition();

	FieldReader reader(source);
	auto left = reader.ReadOptional<Expression>(nullptr, state);
	auto right = reader.ReadOptional<Expression>(nullptr, state);
	result.left = std::move(left);
	result.right = std::move(right);
	result.comparison = reader.ReadRequired<ExpressionType>();
	reader.Finalize();
	return result;
}

JoinSide JoinSide::CombineJoinSide(JoinSide left, JoinSide right) {
	if (left == JoinSide::NONE) {
		return right;
	}
	if (right == JoinSide::NONE) {
		return left;
	}
	if (left != right) {
		return JoinSide::BOTH;
	}
	return left;
}

JoinSide JoinSide::GetJoinSide(idx_t table_binding, const unordered_set<idx_t> &left_bindings,
                               const unordered_set<idx_t> &right_bindings) {
	if (left_bindings.find(table_binding) != left_bindings.end()) {
		// column references table on left side
		D_ASSERT(right_bindings.find(table_binding) == right_bindings.end());
		return JoinSide::LEFT;
	} else {
		// column references table on right side
		D_ASSERT(right_bindings.find(table_binding) != right_bindings.end());
		return JoinSide::RIGHT;
	}
}

JoinSide JoinSide::GetJoinSide(Expression &expression, const unordered_set<idx_t> &left_bindings,
                               const unordered_set<idx_t> &right_bindings) {
	if (expression.type == ExpressionType::BOUND_COLUMN_REF) {
		auto &colref = (BoundColumnRefExpression &)expression;
		if (colref.depth > 0) {
			throw Exception("Non-inner join on correlated columns not supported");
		}
		return GetJoinSide(colref.binding.table_index, left_bindings, right_bindings);
	}
	D_ASSERT(expression.type != ExpressionType::BOUND_REF);
	if (expression.type == ExpressionType::SUBQUERY) {
		D_ASSERT(expression.GetExpressionClass() == ExpressionClass::BOUND_SUBQUERY);
		auto &subquery = (BoundSubqueryExpression &)expression;
		JoinSide side = JoinSide::NONE;
		if (subquery.child) {
			side = GetJoinSide(*subquery.child, left_bindings, right_bindings);
		}
		// correlated subquery, check the side of each of correlated columns in the subquery
		for (auto &corr : subquery.binder->correlated_columns) {
			if (corr.depth > 1) {
				// correlated column has depth > 1
				// it does not refer to any table in the current set of bindings
				return JoinSide::BOTH;
			}
			auto correlated_side = GetJoinSide(corr.binding.table_index, left_bindings, right_bindings);
			side = CombineJoinSide(side, correlated_side);
		}
		return side;
	}
	JoinSide join_side = JoinSide::NONE;
	ExpressionIterator::EnumerateChildren(expression, [&](Expression &child) {
		auto child_side = GetJoinSide(child, left_bindings, right_bindings);
		join_side = CombineJoinSide(child_side, join_side);
	});
	return join_side;
}

JoinSide JoinSide::GetJoinSide(const unordered_set<idx_t> &bindings, const unordered_set<idx_t> &left_bindings,
                               const unordered_set<idx_t> &right_bindings) {
	JoinSide side = JoinSide::NONE;
	for (auto binding : bindings) {
		side = CombineJoinSide(side, GetJoinSide(binding, left_bindings, right_bindings));
	}
	return side;
}

} // namespace duckdb











namespace duckdb {

const uint64_t PLAN_SERIALIZATION_VERSION = 1;

LogicalOperator::LogicalOperator(LogicalOperatorType type)
    : type(type), estimated_cardinality(0), has_estimated_cardinality(false) {
}

LogicalOperator::LogicalOperator(LogicalOperatorType type, vector<unique_ptr<Expression>> expressions)
    : type(type), expressions(std::move(expressions)), estimated_cardinality(0), has_estimated_cardinality(false) {
}

LogicalOperator::~LogicalOperator() {
}

vector<ColumnBinding> LogicalOperator::GetColumnBindings() {
	return {ColumnBinding(0, 0)};
}

string LogicalOperator::GetName() const {
	return LogicalOperatorToString(type);
}

string LogicalOperator::ParamsToString() const {
	string result;
	for (idx_t i = 0; i < expressions.size(); i++) {
		if (i > 0) {
			result += "\n";
		}
		result += expressions[i]->GetName();
	}
	return result;
}

void LogicalOperator::ResolveOperatorTypes() {

	types.clear();
	// first resolve child types
	for (auto &child : children) {
		child->ResolveOperatorTypes();
	}
	// now resolve the types for this operator
	ResolveTypes();
	D_ASSERT(types.size() == GetColumnBindings().size());
}

vector<ColumnBinding> LogicalOperator::GenerateColumnBindings(idx_t table_idx, idx_t column_count) {
	vector<ColumnBinding> result;
	for (idx_t i = 0; i < column_count; i++) {
		result.emplace_back(table_idx, i);
	}
	return result;
}

vector<LogicalType> LogicalOperator::MapTypes(const vector<LogicalType> &types, const vector<idx_t> &projection_map) {
	if (projection_map.empty()) {
		return types;
	} else {
		vector<LogicalType> result_types;
		result_types.reserve(projection_map.size());
		for (auto index : projection_map) {
			result_types.push_back(types[index]);
		}
		return result_types;
	}
}

vector<ColumnBinding> LogicalOperator::MapBindings(const vector<ColumnBinding> &bindings,
                                                   const vector<idx_t> &projection_map) {
	if (projection_map.empty()) {
		return bindings;
	} else {
		vector<ColumnBinding> result_bindings;
		result_bindings.reserve(projection_map.size());
		for (auto index : projection_map) {
			result_bindings.push_back(bindings[index]);
		}
		return result_bindings;
	}
}

string LogicalOperator::ToString() const {
	TreeRenderer renderer;
	return renderer.ToString(*this);
}

void LogicalOperator::Verify(ClientContext &context) {
#ifdef DEBUG
	// verify expressions
	for (idx_t expr_idx = 0; expr_idx < expressions.size(); expr_idx++) {
		auto str = expressions[expr_idx]->ToString();
		// verify that we can (correctly) copy this expression
		auto copy = expressions[expr_idx]->Copy();
		auto original_hash = expressions[expr_idx]->Hash();
		auto copy_hash = copy->Hash();
		// copy should be identical to original
		D_ASSERT(expressions[expr_idx]->ToString() == copy->ToString());
		D_ASSERT(original_hash == copy_hash);
		D_ASSERT(Expression::Equals(expressions[expr_idx].get(), copy.get()));

		D_ASSERT(!Expression::Equals(expressions[expr_idx].get(), nullptr));
		for (idx_t other_idx = 0; other_idx < expr_idx; other_idx++) {
			// comparison with other expressions
			auto other_hash = expressions[other_idx]->Hash();
			bool expr_equal = Expression::Equals(expressions[expr_idx].get(), expressions[other_idx].get());
			if (original_hash != other_hash) {
				// if the hashes are not equal the expressions should not be equal either
				D_ASSERT(!expr_equal);
			}
		}
		D_ASSERT(!str.empty());

		// verify that serialization + deserialization round-trips correctly
		if (expressions[expr_idx]->HasParameter()) {
			continue;
		}
		BufferedSerializer serializer;
		try {
			expressions[expr_idx]->Serialize(serializer);
		} catch (NotImplementedException &ex) {
			// ignore for now (FIXME)
			return;
		}

		auto data = serializer.GetData();
		auto deserializer = BufferedDeserializer(data.data.get(), data.size);

		PlanDeserializationState state(context);
		auto deserialized_expression = Expression::Deserialize(deserializer, state);
		// FIXME: expressions might not be equal yet because of statistics propagation
		continue;
		D_ASSERT(Expression::Equals(expressions[expr_idx].get(), deserialized_expression.get()));
		D_ASSERT(expressions[expr_idx]->Hash() == deserialized_expression->Hash());
	}
	D_ASSERT(!ToString().empty());
	for (auto &child : children) {
		child->Verify(context);
	}
#endif
}

void LogicalOperator::AddChild(unique_ptr<LogicalOperator> child) {
	D_ASSERT(child);
	children.push_back(std::move(child));
}

idx_t LogicalOperator::EstimateCardinality(ClientContext &context) {
	// simple estimator, just take the max of the children
	if (has_estimated_cardinality) {
		return estimated_cardinality;
	}
	idx_t max_cardinality = 0;
	for (auto &child : children) {
		max_cardinality = MaxValue(child->EstimateCardinality(context), max_cardinality);
	}
	has_estimated_cardinality = true;
	return max_cardinality;
}

void LogicalOperator::Print() {
	Printer::Print(ToString());
}

void LogicalOperator::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteField<LogicalOperatorType>(type);
	writer.WriteSerializableList(children);

	Serialize(writer);
	writer.Finalize();
}

unique_ptr<LogicalOperator> LogicalOperator::Deserialize(Deserializer &deserializer, PlanDeserializationState &gstate) {
	unique_ptr<LogicalOperator> result;

	FieldReader reader(deserializer);
	auto type = reader.ReadRequired<LogicalOperatorType>();
	auto children = reader.ReadRequiredSerializableList<LogicalOperator>(gstate);

	LogicalDeserializationState state(gstate, type, children);
	switch (type) {
	case LogicalOperatorType::LOGICAL_PROJECTION:
		result = LogicalProjection::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_FILTER:
		result = LogicalFilter::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		result = LogicalAggregate::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_WINDOW:
		result = LogicalWindow::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_UNNEST:
		result = LogicalUnnest::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_LIMIT:
		result = LogicalLimit::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_ORDER_BY:
		result = LogicalOrder::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_TOP_N:
		result = LogicalTopN::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_COPY_TO_FILE:
		result = LogicalCopyToFile::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_DISTINCT:
		result = LogicalDistinct::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_SAMPLE:
		result = LogicalSample::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_LIMIT_PERCENT:
		result = LogicalLimitPercent::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_GET:
		result = LogicalGet::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_CHUNK_GET:
		result = LogicalColumnDataGet::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_DELIM_GET:
		result = LogicalDelimGet::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_EXPRESSION_GET:
		result = LogicalExpressionGet::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_DUMMY_SCAN:
		result = LogicalDummyScan::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_EMPTY_RESULT:
		result = LogicalEmptyResult::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_CTE_REF:
		result = LogicalCTERef::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_JOIN:
		throw InternalException("LogicalJoin deserialize not supported");
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		result = LogicalDelimJoin::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
		result = LogicalComparisonJoin::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
		result = LogicalAnyJoin::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
		result = LogicalCrossProduct::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_POSITIONAL_JOIN:
		result = LogicalPositionalJoin::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_UNION:
		result = LogicalSetOperation::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_EXCEPT:
		result = LogicalSetOperation::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_INTERSECT:
		result = LogicalSetOperation::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_RECURSIVE_CTE:
		result = LogicalRecursiveCTE::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_INSERT:
		result = LogicalInsert::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_DELETE:
		result = LogicalDelete::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_UPDATE:
		result = LogicalUpdate::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_CREATE_TABLE:
		result = LogicalCreateTable::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_CREATE_INDEX:
		result = LogicalCreateIndex::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_CREATE_SEQUENCE:
		result = LogicalCreate::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_CREATE_VIEW:
		result = LogicalCreate::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_CREATE_SCHEMA:
		result = LogicalCreate::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_CREATE_MACRO:
		result = LogicalCreate::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_PRAGMA:
		result = LogicalPragma::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_CREATE_TYPE:
		result = LogicalCreate::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_EXPLAIN:
		result = LogicalExplain::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_SHOW:
		result = LogicalShow::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_PREPARE:
		result = LogicalPrepare::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_EXECUTE:
		result = LogicalExecute::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_EXPORT:
		result = LogicalExport::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_SET:
		result = LogicalSet::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_RESET:
		result = LogicalReset::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_ALTER:
	case LogicalOperatorType::LOGICAL_VACUUM:
	case LogicalOperatorType::LOGICAL_LOAD:
	case LogicalOperatorType::LOGICAL_ATTACH:
	case LogicalOperatorType::LOGICAL_TRANSACTION:
	case LogicalOperatorType::LOGICAL_DROP:
		result = LogicalSimple::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR:
		result = LogicalExtensionOperator::Deserialize(state, reader);
		break;
	case LogicalOperatorType::LOGICAL_INVALID:
		/* no default here to trigger a warning if we forget to implement deserialize for a new operator */
		throw SerializationException("Invalid type for operator deserialization");
	}

	reader.Finalize();
	result->children = std::move(children);

	return result;
}

vector<idx_t> LogicalOperator::GetTableIndex() const {
	return vector<idx_t> {};
}

unique_ptr<LogicalOperator> LogicalOperator::Copy(ClientContext &context) const {
	BufferedSerializer logical_op_serializer;
	try {
		this->Serialize(logical_op_serializer);
	} catch (NotImplementedException &ex) {
		throw NotImplementedException("Logical Operator Copy requires the logical operator and all of its children to "
		                              "be serializable: " +
		                              std::string(ex.what()));
	}
	auto data = logical_op_serializer.GetData();
	auto logical_op_deserializer = BufferedDeserializer(data.data.get(), data.size);
	PlanDeserializationState state(context);
	auto op_copy = LogicalOperator::Deserialize(logical_op_deserializer, state);
	return op_copy;
}

} // namespace duckdb






namespace duckdb {

void LogicalOperatorVisitor::VisitOperator(LogicalOperator &op) {
	VisitOperatorChildren(op);
	VisitOperatorExpressions(op);
}

void LogicalOperatorVisitor::VisitOperatorChildren(LogicalOperator &op) {
	for (auto &child : op.children) {
		VisitOperator(*child);
	}
}

void LogicalOperatorVisitor::EnumerateExpressions(LogicalOperator &op,
                                                  const std::function<void(unique_ptr<Expression> *child)> &callback) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_EXPRESSION_GET: {
		auto &get = (LogicalExpressionGet &)op;
		for (auto &expr_list : get.expressions) {
			for (auto &expr : expr_list) {
				callback(&expr);
			}
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_ORDER_BY: {
		auto &order = (LogicalOrder &)op;
		for (auto &node : order.orders) {
			callback(&node.expression);
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_TOP_N: {
		auto &order = (LogicalTopN &)op;
		for (auto &node : order.orders) {
			callback(&node.expression);
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_DISTINCT: {
		auto &distinct = (LogicalDistinct &)op;
		for (auto &target : distinct.distinct_targets) {
			callback(&target);
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_INSERT: {
		auto &insert = (LogicalInsert &)op;
		if (insert.on_conflict_condition) {
			callback(&insert.on_conflict_condition);
		}
		if (insert.do_update_condition) {
			callback(&insert.do_update_condition);
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
		if (op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
			auto &delim_join = (LogicalDelimJoin &)op;
			for (auto &expr : delim_join.duplicate_eliminated_columns) {
				callback(&expr);
			}
		}
		auto &join = (LogicalComparisonJoin &)op;
		for (auto &cond : join.conditions) {
			callback(&cond.left);
			callback(&cond.right);
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_ANY_JOIN: {
		auto &join = (LogicalAnyJoin &)op;
		callback(&join.condition);
		break;
	}
	case LogicalOperatorType::LOGICAL_LIMIT: {
		auto &limit = (LogicalLimit &)op;
		if (limit.limit) {
			callback(&limit.limit);
		}
		if (limit.offset) {
			callback(&limit.offset);
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_LIMIT_PERCENT: {
		auto &limit = (LogicalLimitPercent &)op;
		if (limit.limit) {
			callback(&limit.limit);
		}
		if (limit.offset) {
			callback(&limit.offset);
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
		auto &aggr = (LogicalAggregate &)op;
		for (auto &group : aggr.groups) {
			callback(&group);
		}
		break;
	}
	default:
		break;
	}
	for (auto &expression : op.expressions) {
		callback(&expression);
	}
}

void LogicalOperatorVisitor::VisitOperatorExpressions(LogicalOperator &op) {
	LogicalOperatorVisitor::EnumerateExpressions(op, [&](unique_ptr<Expression> *child) { VisitExpression(child); });
}

void LogicalOperatorVisitor::VisitExpression(unique_ptr<Expression> *expression) {
	auto &expr = **expression;
	unique_ptr<Expression> result;
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::BOUND_AGGREGATE:
		result = VisitReplace((BoundAggregateExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_BETWEEN:
		result = VisitReplace((BoundBetweenExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_CASE:
		result = VisitReplace((BoundCaseExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_CAST:
		result = VisitReplace((BoundCastExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_COLUMN_REF:
		result = VisitReplace((BoundColumnRefExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_COMPARISON:
		result = VisitReplace((BoundComparisonExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_CONJUNCTION:
		result = VisitReplace((BoundConjunctionExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_CONSTANT:
		result = VisitReplace((BoundConstantExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_FUNCTION:
		result = VisitReplace((BoundFunctionExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_SUBQUERY:
		result = VisitReplace((BoundSubqueryExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_OPERATOR:
		result = VisitReplace((BoundOperatorExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_PARAMETER:
		result = VisitReplace((BoundParameterExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_REF:
		result = VisitReplace((BoundReferenceExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_DEFAULT:
		result = VisitReplace((BoundDefaultExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_WINDOW:
		result = VisitReplace((BoundWindowExpression &)expr, expression);
		break;
	case ExpressionClass::BOUND_UNNEST:
		result = VisitReplace((BoundUnnestExpression &)expr, expression);
		break;
	default:
		throw InternalException("Unrecognized expression type in logical operator visitor");
	}
	if (result) {
		*expression = std::move(result);
	} else {
		// visit the children of this node
		VisitExpressionChildren(expr);
	}
}

void LogicalOperatorVisitor::VisitExpressionChildren(Expression &expr) {
	ExpressionIterator::EnumerateChildren(expr, [&](unique_ptr<Expression> &expr) { VisitExpression(&expr); });
}

// these are all default methods that can be overriden
// we don't care about coverage here
// LCOV_EXCL_START
unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundAggregateExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundBetweenExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundCaseExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundCastExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundColumnRefExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundComparisonExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundConjunctionExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundConstantExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundDefaultExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundFunctionExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundOperatorExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundParameterExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundReferenceExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundSubqueryExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundWindowExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

unique_ptr<Expression> LogicalOperatorVisitor::VisitReplace(BoundUnnestExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	return nullptr;
}

// LCOV_EXCL_STOP

} // namespace duckdb




namespace duckdb {

LogicalAggregate::LogicalAggregate(idx_t group_index, idx_t aggregate_index, vector<unique_ptr<Expression>> select_list)
    : LogicalOperator(LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY, std::move(select_list)),
      group_index(group_index), aggregate_index(aggregate_index), groupings_index(DConstants::INVALID_INDEX) {
}

void LogicalAggregate::ResolveTypes() {
	D_ASSERT(groupings_index != DConstants::INVALID_INDEX || grouping_functions.empty());
	for (auto &expr : groups) {
		types.push_back(expr->return_type);
	}
	// get the chunk types from the projection list
	for (auto &expr : expressions) {
		types.push_back(expr->return_type);
	}
	for (idx_t i = 0; i < grouping_functions.size(); i++) {
		types.emplace_back(LogicalType::BIGINT);
	}
}

vector<ColumnBinding> LogicalAggregate::GetColumnBindings() {
	D_ASSERT(groupings_index != DConstants::INVALID_INDEX || grouping_functions.empty());
	vector<ColumnBinding> result;
	result.reserve(groups.size() + expressions.size() + grouping_functions.size());
	for (idx_t i = 0; i < groups.size(); i++) {
		result.emplace_back(group_index, i);
	}
	for (idx_t i = 0; i < expressions.size(); i++) {
		result.emplace_back(aggregate_index, i);
	}
	for (idx_t i = 0; i < grouping_functions.size(); i++) {
		result.emplace_back(groupings_index, i);
	}
	return result;
}

string LogicalAggregate::ParamsToString() const {
	string result;
	for (idx_t i = 0; i < groups.size(); i++) {
		if (i > 0) {
			result += "\n";
		}
		result += groups[i]->GetName();
	}
	for (idx_t i = 0; i < expressions.size(); i++) {
		if (i > 0 || !groups.empty()) {
			result += "\n";
		}
		result += expressions[i]->GetName();
	}
	return result;
}

void LogicalAggregate::Serialize(FieldWriter &writer) const {
	writer.WriteSerializableList(expressions);

	writer.WriteField(group_index);
	writer.WriteField(aggregate_index);
	writer.WriteField(groupings_index);
	writer.WriteSerializableList(groups);
	writer.WriteField<idx_t>(grouping_sets.size());
	for (auto &entry : grouping_sets) {
		writer.WriteList<idx_t>(entry);
	}
	writer.WriteField<idx_t>(grouping_functions.size());
	for (auto &entry : grouping_functions) {
		writer.WriteList<idx_t>(entry);
	}

	// TODO statistics
}

unique_ptr<LogicalOperator> LogicalAggregate::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto expressions = reader.ReadRequiredSerializableList<Expression>(state.gstate);

	auto group_index = reader.ReadRequired<idx_t>();
	auto aggregate_index = reader.ReadRequired<idx_t>();
	auto groupings_index = reader.ReadRequired<idx_t>();
	auto groups = reader.ReadRequiredSerializableList<Expression>(state.gstate);
	auto grouping_sets_size = reader.ReadRequired<idx_t>();
	vector<GroupingSet> grouping_sets;
	for (idx_t i = 0; i < grouping_sets_size; i++) {
		grouping_sets.push_back(reader.ReadRequiredSet<idx_t>());
	}
	vector<vector<idx_t>> grouping_functions;
	auto grouping_functions_size = reader.ReadRequired<idx_t>();
	for (idx_t i = 0; i < grouping_functions_size; i++) {
		grouping_functions.push_back(reader.ReadRequiredList<idx_t>());
	}
	auto result = make_unique<LogicalAggregate>(group_index, aggregate_index, std::move(expressions));
	result->groupings_index = groupings_index;
	result->groups = std::move(groups);
	result->grouping_functions = std::move(grouping_functions);
	result->grouping_sets = std::move(grouping_sets);

	return std::move(result);
}

idx_t LogicalAggregate::EstimateCardinality(ClientContext &context) {
	if (groups.empty()) {
		// ungrouped aggregate
		return 1;
	}
	return LogicalOperator::EstimateCardinality(context);
}

vector<idx_t> LogicalAggregate::GetTableIndex() const {
	vector<idx_t> result {group_index, aggregate_index};
	if (groupings_index != DConstants::INVALID_INDEX) {
		result.push_back(groupings_index);
	}
	return result;
}

} // namespace duckdb



namespace duckdb {

LogicalAnyJoin::LogicalAnyJoin(JoinType type) : LogicalJoin(type, LogicalOperatorType::LOGICAL_ANY_JOIN) {
}

string LogicalAnyJoin::ParamsToString() const {
	return condition->ToString();
}

void LogicalAnyJoin::Serialize(FieldWriter &writer) const {
	writer.WriteField(join_type);
	writer.WriteOptional(condition);
}

unique_ptr<LogicalOperator> LogicalAnyJoin::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto join_type = reader.ReadRequired<JoinType>();
	auto condition = reader.ReadOptional<Expression>(nullptr, state.gstate);
	auto result = make_unique<LogicalAnyJoin>(join_type);
	result->condition = std::move(condition);
	return std::move(result);
}

} // namespace duckdb




namespace duckdb {

LogicalColumnDataGet::LogicalColumnDataGet(idx_t table_index, vector<LogicalType> types,
                                           unique_ptr<ColumnDataCollection> collection)
    : LogicalOperator(LogicalOperatorType::LOGICAL_CHUNK_GET), table_index(table_index),
      collection(std::move(collection)) {
	D_ASSERT(types.size() > 0);
	chunk_types = std::move(types);
}

vector<ColumnBinding> LogicalColumnDataGet::GetColumnBindings() {
	return GenerateColumnBindings(table_index, chunk_types.size());
}

void LogicalColumnDataGet::Serialize(FieldWriter &writer) const {
	writer.WriteField(table_index);
	writer.WriteRegularSerializableList(chunk_types);
	writer.WriteField(collection->ChunkCount());
	for (auto &chunk : collection->Chunks()) {
		chunk.Serialize(writer.GetSerializer());
	}
}

unique_ptr<LogicalOperator> LogicalColumnDataGet::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto table_index = reader.ReadRequired<idx_t>();
	auto chunk_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
	auto chunk_count = reader.ReadRequired<idx_t>();
	auto collection = make_unique<ColumnDataCollection>(state.gstate.context, chunk_types);
	for (idx_t i = 0; i < chunk_count; i++) {
		DataChunk chunk;
		chunk.Deserialize(reader.GetSource());
		collection->Append(chunk);
	}
	return make_unique<LogicalColumnDataGet>(table_index, std::move(chunk_types), std::move(collection));
}

vector<idx_t> LogicalColumnDataGet::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

} // namespace duckdb





namespace duckdb {

LogicalComparisonJoin::LogicalComparisonJoin(JoinType join_type, LogicalOperatorType logical_type)
    : LogicalJoin(join_type, logical_type) {
}

string LogicalComparisonJoin::ParamsToString() const {
	string result = JoinTypeToString(join_type);
	for (auto &condition : conditions) {
		result += "\n";
		auto expr = make_unique<BoundComparisonExpression>(condition.comparison, condition.left->Copy(),
		                                                   condition.right->Copy());
		result += expr->ToString();
	}

	return result;
}

void LogicalComparisonJoin::Serialize(FieldWriter &writer) const {
	LogicalJoin::Serialize(writer);
	writer.WriteRegularSerializableList(conditions);
	writer.WriteRegularSerializableList(delim_types);
}

void LogicalComparisonJoin::Deserialize(LogicalComparisonJoin &comparison_join, LogicalDeserializationState &state,
                                        FieldReader &reader) {
	LogicalJoin::Deserialize(comparison_join, state, reader);
	comparison_join.conditions = reader.ReadRequiredSerializableList<JoinCondition, JoinCondition>(state.gstate);
	comparison_join.delim_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
}

unique_ptr<LogicalOperator> LogicalComparisonJoin::Deserialize(LogicalDeserializationState &state,
                                                               FieldReader &reader) {
	auto result = make_unique<LogicalComparisonJoin>(JoinType::INVALID, state.type);
	LogicalComparisonJoin::Deserialize(*result, state, reader);
	return std::move(result);
}

} // namespace duckdb






namespace duckdb {

// Warning: some fields here were added while this code appears untested
// -> copy test in test/api/test_plan_serialization.cpp was commented out as WIP
void LogicalCopyToFile::Serialize(FieldWriter &writer) const {
	writer.WriteString(file_path);
	writer.WriteField(use_tmp_file);
	writer.WriteField(allow_overwrite);
	writer.WriteField(per_thread_output);
	writer.WriteList<idx_t>(partition_columns);

	D_ASSERT(!function.name.empty());
	writer.WriteString(function.name);

	writer.WriteField(bind_data != nullptr);
	if (bind_data && !function.serialize) {
		throw InvalidInputException("Can't serialize copy function %s", function.name);
	}

	function.serialize(writer, *bind_data, function);
}

unique_ptr<LogicalOperator> LogicalCopyToFile::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto file_path = reader.ReadRequired<string>();
	auto use_tmp_file = reader.ReadRequired<bool>();
	auto allow_overwrite = reader.ReadRequired<bool>();
	auto per_thread_output = reader.ReadRequired<bool>();
	auto partition_columns = reader.ReadRequiredList<idx_t>();

	auto copy_func_name = reader.ReadRequired<string>();

	auto has_bind_data = reader.ReadRequired<bool>();

	auto &context = state.gstate.context;
	auto copy_func_catalog_entry =
	    Catalog::GetEntry<CopyFunctionCatalogEntry>(context, INVALID_CATALOG, DEFAULT_SCHEMA, copy_func_name);
	if (!copy_func_catalog_entry) {
		throw InternalException("Cant find catalog entry for function %s", copy_func_name);
	}
	CopyFunction copy_func = copy_func_catalog_entry->function;

	unique_ptr<FunctionData> bind_data;
	if (has_bind_data) {
		if (!copy_func.deserialize) {
			throw SerializationException("Have bind info but no deserialization function for %s", copy_func.name);
		}
		bind_data = copy_func.deserialize(context, reader, copy_func);
	}

	auto result = make_unique<LogicalCopyToFile>(copy_func, std::move(bind_data));
	result->file_path = file_path;
	result->use_tmp_file = use_tmp_file;
	result->allow_overwrite = allow_overwrite;
	result->per_thread_output = per_thread_output;
	result->partition_columns = std::move(partition_columns);
	return std::move(result);
}

idx_t LogicalCopyToFile::EstimateCardinality(ClientContext &context) {
	return 1;
}

} // namespace duckdb


namespace duckdb {

void LogicalCreate::Serialize(FieldWriter &writer) const {
	info->Serialize(writer.GetSerializer());
}

unique_ptr<LogicalOperator> LogicalCreate::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto &context = state.gstate.context;
	auto info = CreateInfo::Deserialize(reader.GetSource());

	auto schema_catalog_entry = Catalog::GetSchema(context, INVALID_CATALOG, info->schema, true);
	return make_unique<LogicalCreate>(state.type, std::move(info), schema_catalog_entry);
}

idx_t LogicalCreate::EstimateCardinality(ClientContext &context) {
	return 1;
}

} // namespace duckdb






namespace duckdb {

void LogicalCreateIndex::Serialize(FieldWriter &writer) const {

	writer.WriteOptional(info);
	table.Serialize(writer.GetSerializer());
	FunctionSerializer::SerializeBase<TableFunction>(writer, function, bind_data.get());
	writer.WriteSerializableList(unbound_expressions);

	writer.Finalize();
}

unique_ptr<LogicalOperator> LogicalCreateIndex::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {

	auto &context = state.gstate.context;
	auto catalog_info = TableCatalogEntry::Deserialize(reader.GetSource(), context);

	auto table =
	    Catalog::GetEntry<TableCatalogEntry>(context, INVALID_CATALOG, catalog_info->schema, catalog_info->table);
	auto unbound_expressions = reader.ReadRequiredSerializableList<Expression>(state.gstate);

	auto create_info = reader.ReadOptional<CreateInfo>(nullptr);
	if (create_info->type != CatalogType::INDEX_ENTRY) {
		throw InternalException("Unexpected type: '%s', expected '%s'", CatalogTypeToString(create_info->type),
		                        CatalogTypeToString(CatalogType::INDEX_ENTRY));
	}

	CreateInfo *raw_create_info_ptr = create_info.release();
	CreateIndexInfo *raw_create_index_info_ptr = static_cast<CreateIndexInfo *>(raw_create_info_ptr);
	unique_ptr<CreateIndexInfo> uptr_create_index_info = unique_ptr<CreateIndexInfo> {raw_create_index_info_ptr};
	auto info = unique_ptr<CreateIndexInfo> {static_cast<CreateIndexInfo *>(create_info.release())};

	unique_ptr<FunctionData> bind_data;
	bool has_deserialize;
	auto function = FunctionSerializer::DeserializeBaseInternal<TableFunction, TableFunctionCatalogEntry>(
	    reader, state.gstate, CatalogType::TABLE_FUNCTION_ENTRY, bind_data, has_deserialize);

	reader.Finalize();
	return make_unique<LogicalCreateIndex>(std::move(bind_data), std::move(info), std::move(unbound_expressions),
	                                       *table, std::move(function));
}

} // namespace duckdb


namespace duckdb {

void LogicalCreateTable::Serialize(FieldWriter &writer) const {
	writer.WriteSerializable(*info);
}

unique_ptr<LogicalOperator> LogicalCreateTable::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto info = reader.ReadRequiredSerializable<BoundCreateTableInfo>(state.gstate);
	auto schema = info->schema;
	return make_unique<LogicalCreateTable>(schema, std::move(info));
}

idx_t LogicalCreateTable::EstimateCardinality(ClientContext &context) {
	return 1;
}

} // namespace duckdb


namespace duckdb {

LogicalCrossProduct::LogicalCrossProduct(unique_ptr<LogicalOperator> left, unique_ptr<LogicalOperator> right)
    : LogicalUnconditionalJoin(LogicalOperatorType::LOGICAL_CROSS_PRODUCT, std::move(left), std::move(right)) {
}

unique_ptr<LogicalOperator> LogicalCrossProduct::Create(unique_ptr<LogicalOperator> left,
                                                        unique_ptr<LogicalOperator> right) {
	if (left->type == LogicalOperatorType::LOGICAL_DUMMY_SCAN) {
		return right;
	}
	if (right->type == LogicalOperatorType::LOGICAL_DUMMY_SCAN) {
		return left;
	}
	return make_unique<LogicalCrossProduct>(std::move(left), std::move(right));
}

void LogicalCrossProduct::Serialize(FieldWriter &writer) const {
}

unique_ptr<LogicalOperator> LogicalCrossProduct::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	// TODO(stephwang): review if unique_ptr<LogicalOperator> plan is needed
	auto result = unique_ptr<LogicalCrossProduct>(new LogicalCrossProduct());
	return std::move(result);
}

} // namespace duckdb



namespace duckdb {

void LogicalCTERef::Serialize(FieldWriter &writer) const {
	writer.WriteField(table_index);
	writer.WriteField(cte_index);
	writer.WriteRegularSerializableList(chunk_types);
	writer.WriteList<string>(bound_columns);
}

unique_ptr<LogicalOperator> LogicalCTERef::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto table_index = reader.ReadRequired<idx_t>();
	auto cte_index = reader.ReadRequired<idx_t>();
	auto chunk_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
	auto bound_columns = reader.ReadRequiredList<string>();
	return make_unique<LogicalCTERef>(table_index, cte_index, chunk_types, bound_columns);
}

vector<idx_t> LogicalCTERef::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

} // namespace duckdb




namespace duckdb {

LogicalDelete::LogicalDelete(TableCatalogEntry *table, idx_t table_index)
    : LogicalOperator(LogicalOperatorType::LOGICAL_DELETE), table(table), table_index(table_index),
      return_chunk(false) {
}

void LogicalDelete::Serialize(FieldWriter &writer) const {
	table->Serialize(writer.GetSerializer());
	writer.WriteField(table_index);
	writer.WriteField(return_chunk);
}

unique_ptr<LogicalOperator> LogicalDelete::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto &context = state.gstate.context;
	auto info = TableCatalogEntry::Deserialize(reader.GetSource(), context);

	auto table_catalog_entry =
	    Catalog::GetEntry<TableCatalogEntry>(context, INVALID_CATALOG, info->schema, info->table);

	auto table_index = reader.ReadRequired<idx_t>();
	auto result = make_unique<LogicalDelete>(table_catalog_entry, table_index);
	result->return_chunk = reader.ReadRequired<bool>();
	return std::move(result);
}

idx_t LogicalDelete::EstimateCardinality(ClientContext &context) {
	return return_chunk ? LogicalOperator::EstimateCardinality(context) : 1;
}

vector<idx_t> LogicalDelete::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

vector<ColumnBinding> LogicalDelete::GetColumnBindings() {
	if (return_chunk) {
		return GenerateColumnBindings(table_index, table->GetTypes().size());
	}
	return {ColumnBinding(0, 0)};
}

void LogicalDelete::ResolveTypes() {
	if (return_chunk) {
		types = table->GetTypes();
	} else {
		types.emplace_back(LogicalType::BIGINT);
	}
}

} // namespace duckdb



namespace duckdb {

void LogicalDelimGet::Serialize(FieldWriter &writer) const {
	writer.WriteField(table_index);
	writer.WriteRegularSerializableList(chunk_types);
}

unique_ptr<LogicalOperator> LogicalDelimGet::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto table_index = reader.ReadRequired<idx_t>();
	auto chunk_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
	return make_unique<LogicalDelimGet>(table_index, chunk_types);
}

vector<idx_t> LogicalDelimGet::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

} // namespace duckdb



namespace duckdb {

LogicalDelimJoin::LogicalDelimJoin(JoinType type)
    : LogicalComparisonJoin(type, LogicalOperatorType::LOGICAL_DELIM_JOIN) {
}

void LogicalDelimJoin::Serialize(FieldWriter &writer) const {
	LogicalComparisonJoin::Serialize(writer);
	if (type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		D_ASSERT(duplicate_eliminated_columns.empty());
		// if the delim join has no delim columns anymore it is turned into a regular comparison join
		return;
	}
	writer.WriteSerializableList(duplicate_eliminated_columns);
}

unique_ptr<LogicalOperator> LogicalDelimJoin::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto result = make_unique<LogicalDelimJoin>(JoinType::INVALID);
	LogicalComparisonJoin::Deserialize(*result, state, reader);
	result->duplicate_eliminated_columns = reader.ReadRequiredSerializableList<Expression>(state.gstate);
	return std::move(result);
}

} // namespace duckdb




namespace duckdb {

string LogicalDistinct::ParamsToString() const {
	string result = LogicalOperator::ParamsToString();
	if (!distinct_targets.empty()) {
		result += StringUtil::Join(distinct_targets, distinct_targets.size(), "\n",
		                           [](const unique_ptr<Expression> &child) { return child->GetName(); });
	}

	return result;
}
void LogicalDistinct::Serialize(FieldWriter &writer) const {
	writer.WriteSerializableList(distinct_targets);
}

unique_ptr<LogicalOperator> LogicalDistinct::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto distinct_targets = reader.ReadRequiredSerializableList<Expression>(state.gstate);
	return make_unique<LogicalDistinct>(std::move(distinct_targets));
}

} // namespace duckdb



namespace duckdb {

void LogicalDummyScan::Serialize(FieldWriter &writer) const {
	writer.WriteField(table_index);
}

unique_ptr<LogicalOperator> LogicalDummyScan::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto table_index = reader.ReadRequired<idx_t>();
	return make_unique<LogicalDummyScan>(table_index);
}

vector<idx_t> LogicalDummyScan::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

} // namespace duckdb



namespace duckdb {

LogicalEmptyResult::LogicalEmptyResult(unique_ptr<LogicalOperator> op)
    : LogicalOperator(LogicalOperatorType::LOGICAL_EMPTY_RESULT) {

	this->bindings = op->GetColumnBindings();

	op->ResolveOperatorTypes();
	this->return_types = op->types;
}

LogicalEmptyResult::LogicalEmptyResult() : LogicalOperator(LogicalOperatorType::LOGICAL_EMPTY_RESULT) {
}

void LogicalEmptyResult::Serialize(FieldWriter &writer) const {
	writer.WriteRegularSerializableList(return_types);
	writer.WriteList<ColumnBinding>(bindings);
}

unique_ptr<LogicalOperator> LogicalEmptyResult::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto return_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
	auto bindings = reader.ReadRequiredList<ColumnBinding>();
	auto result = unique_ptr<LogicalEmptyResult>(new LogicalEmptyResult());
	result->return_types = return_types;
	result->bindings = bindings;
	return std::move(result);
}

} // namespace duckdb


namespace duckdb {

void LogicalExecute::Serialize(FieldWriter &writer) const {
	throw NotImplementedException(LogicalOperatorToString(type));
}

unique_ptr<LogicalOperator> LogicalExecute::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	throw NotImplementedException(LogicalOperatorToString(state.type));
}
} // namespace duckdb



namespace duckdb {

void LogicalExplain::Serialize(FieldWriter &writer) const {
	writer.WriteField(explain_type);
	writer.WriteString(physical_plan);
	writer.WriteString(logical_plan_unopt);
	writer.WriteString(logical_plan_opt);
}

unique_ptr<LogicalOperator> LogicalExplain::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto explain_type = reader.ReadRequired<ExplainType>();
	// TODO(stephwang) review if unique_ptr<LogicalOperator> plan is needed
	auto result = unique_ptr<LogicalExplain>(new LogicalExplain(explain_type));
	result->physical_plan = reader.ReadRequired<string>();
	result->logical_plan_unopt = reader.ReadRequired<string>();
	result->logical_plan_opt = reader.ReadRequired<string>();
	return std::move(result);
}
} // namespace duckdb


namespace duckdb {

void LogicalExport::Serialize(FieldWriter &writer) const {
	throw NotImplementedException(LogicalOperatorToString(type));
}

unique_ptr<LogicalOperator> LogicalExport::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	throw NotImplementedException(LogicalOperatorToString(state.type));
}

} // namespace duckdb



namespace duckdb {

void LogicalExpressionGet::Serialize(FieldWriter &writer) const {
	writer.WriteField(table_index);
	writer.WriteRegularSerializableList(expr_types);

	writer.WriteField<idx_t>(expressions.size());
	for (auto &entry : expressions) {
		writer.WriteSerializableList(entry);
	}
}

unique_ptr<LogicalOperator> LogicalExpressionGet::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto table_index = reader.ReadRequired<idx_t>();
	auto expr_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();

	auto expressions_size = reader.ReadRequired<idx_t>();
	vector<vector<unique_ptr<Expression>>> expressions;
	for (idx_t i = 0; i < expressions_size; i++) {
		expressions.push_back(reader.ReadRequiredSerializableList<Expression>(state.gstate));
	}

	return make_unique<LogicalExpressionGet>(table_index, expr_types, std::move(expressions));
}

vector<idx_t> LogicalExpressionGet::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

} // namespace duckdb



namespace duckdb {
unique_ptr<LogicalExtensionOperator> LogicalExtensionOperator::Deserialize(LogicalDeserializationState &state,
                                                                           FieldReader &reader) {
	auto &config = DBConfig::GetConfig(state.gstate.context);

	auto extension_name = reader.ReadRequired<std::string>();
	for (auto &extension : config.operator_extensions) {
		if (extension->GetName() == extension_name) {
			return extension->Deserialize(state, reader);
		}
	}

	throw SerializationException("No serialization method exists for extension: " + extension_name);
}
} // namespace duckdb




namespace duckdb {

LogicalFilter::LogicalFilter(unique_ptr<Expression> expression) : LogicalOperator(LogicalOperatorType::LOGICAL_FILTER) {
	expressions.push_back(std::move(expression));
	SplitPredicates(expressions);
}

LogicalFilter::LogicalFilter() : LogicalOperator(LogicalOperatorType::LOGICAL_FILTER) {
}

void LogicalFilter::ResolveTypes() {
	types = MapTypes(children[0]->types, projection_map);
}

vector<ColumnBinding> LogicalFilter::GetColumnBindings() {
	return MapBindings(children[0]->GetColumnBindings(), projection_map);
}

// Split the predicates separated by AND statements
// These are the predicates that are safe to push down because all of them MUST
// be true
bool LogicalFilter::SplitPredicates(vector<unique_ptr<Expression>> &expressions) {
	bool found_conjunction = false;
	for (idx_t i = 0; i < expressions.size(); i++) {
		if (expressions[i]->type == ExpressionType::CONJUNCTION_AND) {
			auto &conjunction = (BoundConjunctionExpression &)*expressions[i];
			found_conjunction = true;
			// AND expression, append the other children
			for (idx_t k = 1; k < conjunction.children.size(); k++) {
				expressions.push_back(std::move(conjunction.children[k]));
			}
			// replace this expression with the first child of the conjunction
			expressions[i] = std::move(conjunction.children[0]);
			// we move back by one so the right child is checked again
			// in case it is an AND expression as well
			i--;
		}
	}
	return found_conjunction;
}

void LogicalFilter::Serialize(FieldWriter &writer) const {
	writer.WriteSerializableList<Expression>(expressions);
	writer.WriteList<idx_t>(projection_map);
}

unique_ptr<LogicalOperator> LogicalFilter::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto expressions = reader.ReadRequiredSerializableList<Expression>(state.gstate);
	auto projection_map = reader.ReadRequiredList<idx_t>();
	auto result = make_unique<LogicalFilter>();
	result->expressions = std::move(expressions);
	result->projection_map = std::move(projection_map);
	return std::move(result);
}

} // namespace duckdb










namespace duckdb {

LogicalGet::LogicalGet(idx_t table_index, TableFunction function, unique_ptr<FunctionData> bind_data,
                       vector<LogicalType> returned_types, vector<string> returned_names)
    : LogicalOperator(LogicalOperatorType::LOGICAL_GET), table_index(table_index), function(std::move(function)),
      bind_data(std::move(bind_data)), returned_types(std::move(returned_types)), names(std::move(returned_names)) {
}

string LogicalGet::GetName() const {
	return StringUtil::Upper(function.name);
}

TableCatalogEntry *LogicalGet::GetTable() const {
	return TableScanFunction::GetTableEntry(function, bind_data.get());
}

string LogicalGet::ParamsToString() const {
	string result;
	for (auto &kv : table_filters.filters) {
		auto &column_index = kv.first;
		auto &filter = kv.second;
		if (column_index < names.size()) {
			result += filter->ToString(names[column_index]);
		}
		result += "\n";
	}
	if (!function.to_string) {
		return string();
	}
	return function.to_string(bind_data.get());
}

vector<ColumnBinding> LogicalGet::GetColumnBindings() {
	if (column_ids.empty()) {
		return {ColumnBinding(table_index, 0)};
	}
	vector<ColumnBinding> result;
	if (projection_ids.empty()) {
		for (idx_t col_idx = 0; col_idx < column_ids.size(); col_idx++) {
			result.emplace_back(table_index, col_idx);
		}
	} else {
		for (auto proj_id : projection_ids) {
			result.emplace_back(table_index, proj_id);
		}
	}
	if (!projected_input.empty()) {
		if (children.size() != 1) {
			throw InternalException("LogicalGet::project_input can only be set for table-in-out functions");
		}
		auto child_bindings = children[0]->GetColumnBindings();
		for (auto entry : projected_input) {
			D_ASSERT(entry < child_bindings.size());
			result.emplace_back(child_bindings[entry]);
		}
	}
	return result;
}

void LogicalGet::ResolveTypes() {
	if (column_ids.empty()) {
		column_ids.push_back(COLUMN_IDENTIFIER_ROW_ID);
	}

	if (projection_ids.empty()) {
		for (auto &index : column_ids) {
			if (index == COLUMN_IDENTIFIER_ROW_ID) {
				types.emplace_back(LogicalType::ROW_TYPE);
			} else {
				types.push_back(returned_types[index]);
			}
		}
	} else {
		for (auto &proj_index : projection_ids) {
			auto &index = column_ids[proj_index];
			if (index == COLUMN_IDENTIFIER_ROW_ID) {
				types.emplace_back(LogicalType::ROW_TYPE);
			} else {
				types.push_back(returned_types[index]);
			}
		}
	}
	if (!projected_input.empty()) {
		if (children.size() != 1) {
			throw InternalException("LogicalGet::project_input can only be set for table-in-out functions");
		}
		for (auto entry : projected_input) {
			D_ASSERT(entry < children[0]->types.size());
			types.push_back(children[0]->types[entry]);
		}
	}
}

idx_t LogicalGet::EstimateCardinality(ClientContext &context) {
	if (function.cardinality) {
		auto node_stats = function.cardinality(context, bind_data.get());
		if (node_stats && node_stats->has_estimated_cardinality) {
			return node_stats->estimated_cardinality;
		}
	}
	return 1;
}

void LogicalGet::Serialize(FieldWriter &writer) const {
	writer.WriteField(table_index);
	writer.WriteRegularSerializableList(returned_types);
	writer.WriteList<string>(names);
	writer.WriteList<column_t>(column_ids);
	writer.WriteList<column_t>(projection_ids);
	writer.WriteSerializable(table_filters);

	FunctionSerializer::SerializeBase<TableFunction>(writer, function, bind_data.get());
	if (!function.serialize) {
		D_ASSERT(!function.deserialize);
		// no serialize method: serialize input values and named_parameters for rebinding purposes
		writer.WriteRegularSerializableList(parameters);
		writer.WriteField<idx_t>(named_parameters.size());
		for (auto &pair : named_parameters) {
			writer.WriteString(pair.first);
			writer.WriteSerializable(pair.second);
		}
		writer.WriteRegularSerializableList(input_table_types);
		writer.WriteList<string>(input_table_names);
	}
	writer.WriteList<column_t>(projected_input);
}

unique_ptr<LogicalOperator> LogicalGet::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto table_index = reader.ReadRequired<idx_t>();
	auto returned_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
	auto returned_names = reader.ReadRequiredList<string>();
	auto column_ids = reader.ReadRequiredList<column_t>();
	auto projection_ids = reader.ReadRequiredList<column_t>();
	auto table_filters = reader.ReadRequiredSerializable<TableFilterSet>();

	unique_ptr<FunctionData> bind_data;
	bool has_deserialize;
	auto function = FunctionSerializer::DeserializeBaseInternal<TableFunction, TableFunctionCatalogEntry>(
	    reader, state.gstate, CatalogType::TABLE_FUNCTION_ENTRY, bind_data, has_deserialize);

	vector<Value> parameters;
	named_parameter_map_t named_parameters;
	vector<LogicalType> input_table_types;
	vector<string> input_table_names;
	if (!has_deserialize) {
		D_ASSERT(!bind_data);
		parameters = reader.ReadRequiredSerializableList<Value, Value>();

		auto named_parameters_size = reader.ReadRequired<idx_t>();
		for (idx_t i = 0; i < named_parameters_size; i++) {
			auto first = reader.ReadRequired<string>();
			auto second = reader.ReadRequiredSerializable<Value, Value>();
			auto pair = make_pair(first, second);
			named_parameters.insert(pair);
		}

		input_table_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
		input_table_names = reader.ReadRequiredList<string>();
		TableFunctionBindInput input(parameters, named_parameters, input_table_types, input_table_names,
		                             function.function_info.get());

		vector<LogicalType> bind_return_types;
		vector<string> bind_names;
		bind_data = function.bind(state.gstate.context, input, bind_return_types, bind_names);
		if (returned_types != bind_return_types) {
			throw SerializationException(
			    "Table function deserialization failure - bind returned different return types than were serialized");
		}
		// names can actually be different because of aliases - only the sizes cannot be different
		if (returned_names.size() != bind_names.size()) {
			throw SerializationException(
			    "Table function deserialization failure - bind returned different returned names than were serialized");
		}
	}
	vector<column_t> projected_input;
	reader.ReadList<column_t>(projected_input);

	auto result = make_unique<LogicalGet>(table_index, function, std::move(bind_data), returned_types, returned_names);
	result->column_ids = std::move(column_ids);
	result->projection_ids = std::move(projection_ids);
	result->table_filters = std::move(*table_filters);
	result->parameters = std::move(parameters);
	result->named_parameters = std::move(named_parameters);
	result->input_table_types = input_table_types;
	result->input_table_names = input_table_names;
	result->projected_input = std::move(projected_input);
	return std::move(result);
}

vector<idx_t> LogicalGet::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

} // namespace duckdb





namespace duckdb {

LogicalInsert::LogicalInsert(TableCatalogEntry *table, idx_t table_index)
    : LogicalOperator(LogicalOperatorType::LOGICAL_INSERT), table(table), table_index(table_index), return_chunk(false),
      action_type(OnConflictAction::THROW) {
}

void LogicalInsert::Serialize(FieldWriter &writer) const {
	writer.WriteField<idx_t>(insert_values.size());
	for (auto &entry : insert_values) {
		writer.WriteSerializableList(entry);
	}

	writer.WriteList<idx_t>(column_index_map);
	writer.WriteRegularSerializableList(expected_types);
	table->Serialize(writer.GetSerializer());
	writer.WriteField(table_index);
	writer.WriteField(return_chunk);
	writer.WriteSerializableList(bound_defaults);
	writer.WriteField(action_type);
}

unique_ptr<LogicalOperator> LogicalInsert::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto &context = state.gstate.context;
	auto insert_values_size = reader.ReadRequired<idx_t>();
	vector<vector<unique_ptr<Expression>>> insert_values;
	for (idx_t i = 0; i < insert_values_size; ++i) {
		insert_values.push_back(reader.ReadRequiredSerializableList<Expression>(state.gstate));
	}

	auto column_index_map = reader.ReadRequiredList<idx_t, physical_index_vector_t<idx_t>>();
	auto expected_types = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
	auto info = TableCatalogEntry::Deserialize(reader.GetSource(), context);
	auto table_index = reader.ReadRequired<idx_t>();
	auto return_chunk = reader.ReadRequired<bool>();
	auto bound_defaults = reader.ReadRequiredSerializableList<Expression>(state.gstate);
	auto action_type = reader.ReadRequired<OnConflictAction>();

	auto &catalog = Catalog::GetCatalog(context, INVALID_CATALOG);

	TableCatalogEntry *table_catalog_entry = catalog.GetEntry<TableCatalogEntry>(context, info->schema, info->table);

	if (!table_catalog_entry) {
		throw InternalException("Cant find catalog entry for table %s", info->table);
	}

	auto result = make_unique<LogicalInsert>(table_catalog_entry, table_index);
	result->type = state.type;
	result->table = table_catalog_entry;
	result->return_chunk = return_chunk;
	result->insert_values = std::move(insert_values);
	result->column_index_map = column_index_map;
	result->expected_types = expected_types;
	result->bound_defaults = std::move(bound_defaults);
	result->action_type = action_type;
	return std::move(result);
}

idx_t LogicalInsert::EstimateCardinality(ClientContext &context) {
	return return_chunk ? LogicalOperator::EstimateCardinality(context) : 1;
}

vector<idx_t> LogicalInsert::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

vector<ColumnBinding> LogicalInsert::GetColumnBindings() {
	if (return_chunk) {
		return GenerateColumnBindings(table_index, table->GetTypes().size());
	}
	return {ColumnBinding(0, 0)};
}

void LogicalInsert::ResolveTypes() {
	if (return_chunk) {
		types = table->GetTypes();
	} else {
		types.emplace_back(LogicalType::BIGINT);
	}
}

} // namespace duckdb






namespace duckdb {

LogicalJoin::LogicalJoin(JoinType join_type, LogicalOperatorType logical_type)
    : LogicalOperator(logical_type), join_type(join_type) {
}

vector<ColumnBinding> LogicalJoin::GetColumnBindings() {
	auto left_bindings = MapBindings(children[0]->GetColumnBindings(), left_projection_map);
	if (join_type == JoinType::SEMI || join_type == JoinType::ANTI) {
		// for SEMI and ANTI join we only project the left hand side
		return left_bindings;
	}
	if (join_type == JoinType::MARK) {
		// for MARK join we project the left hand side plus the MARK column
		left_bindings.emplace_back(mark_index, 0);
		return left_bindings;
	}
	// for other join types we project both the LHS and the RHS
	auto right_bindings = MapBindings(children[1]->GetColumnBindings(), right_projection_map);
	left_bindings.insert(left_bindings.end(), right_bindings.begin(), right_bindings.end());
	return left_bindings;
}

void LogicalJoin::ResolveTypes() {
	types = MapTypes(children[0]->types, left_projection_map);
	if (join_type == JoinType::SEMI || join_type == JoinType::ANTI) {
		// for SEMI and ANTI join we only project the left hand side
		return;
	}
	if (join_type == JoinType::MARK) {
		// for MARK join we project the left hand side, plus a BOOLEAN column indicating the MARK
		types.emplace_back(LogicalType::BOOLEAN);
		return;
	}
	// for any other join we project both sides
	auto right_types = MapTypes(children[1]->types, right_projection_map);
	types.insert(types.end(), right_types.begin(), right_types.end());
}

void LogicalJoin::GetTableReferences(LogicalOperator &op, unordered_set<idx_t> &bindings) {
	auto column_bindings = op.GetColumnBindings();
	for (auto binding : column_bindings) {
		bindings.insert(binding.table_index);
	}
}

void LogicalJoin::GetExpressionBindings(Expression &expr, unordered_set<idx_t> &bindings) {
	if (expr.type == ExpressionType::BOUND_COLUMN_REF) {
		auto &colref = (BoundColumnRefExpression &)expr;
		D_ASSERT(colref.depth == 0);
		bindings.insert(colref.binding.table_index);
	}
	ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) { GetExpressionBindings(child, bindings); });
}

void LogicalJoin::Serialize(FieldWriter &writer) const {
	writer.WriteField<JoinType>(join_type);
	writer.WriteField<idx_t>(mark_index);
	writer.WriteList<idx_t>(left_projection_map);
	writer.WriteList<idx_t>(right_projection_map);
	//	writer.WriteSerializableList(join_stats);
}

void LogicalJoin::Deserialize(LogicalJoin &join, LogicalDeserializationState &state, FieldReader &reader) {
	join.join_type = reader.ReadRequired<JoinType>();
	join.mark_index = reader.ReadRequired<idx_t>();
	join.left_projection_map = reader.ReadRequiredList<idx_t>();
	join.right_projection_map = reader.ReadRequiredList<idx_t>();
	//	join.join_stats = reader.ReadRequiredSerializableList<BaseStatistics>(reader.GetSource());
}

} // namespace duckdb



namespace duckdb {

LogicalLimit::LogicalLimit(int64_t limit_val, int64_t offset_val, unique_ptr<Expression> limit,
                           unique_ptr<Expression> offset)
    : LogicalOperator(LogicalOperatorType::LOGICAL_LIMIT), limit_val(limit_val), offset_val(offset_val),
      limit(std::move(limit)), offset(std::move(offset)) {
}

vector<ColumnBinding> LogicalLimit::GetColumnBindings() {
	return children[0]->GetColumnBindings();
}

idx_t LogicalLimit::EstimateCardinality(ClientContext &context) {
	auto child_cardinality = children[0]->EstimateCardinality(context);
	if (limit_val >= 0 && idx_t(limit_val) < child_cardinality) {
		child_cardinality = limit_val;
	}
	return child_cardinality;
}

void LogicalLimit::ResolveTypes() {
	types = children[0]->types;
}

void LogicalLimit::Serialize(FieldWriter &writer) const {
	writer.WriteField(limit_val);
	writer.WriteField(offset_val);
	writer.WriteOptional(limit);
	writer.WriteOptional(offset);
}

unique_ptr<LogicalOperator> LogicalLimit::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto limit_val = reader.ReadRequired<int64_t>();
	auto offset_val = reader.ReadRequired<int64_t>();
	auto limit = reader.ReadOptional<Expression>(nullptr, state.gstate);
	auto offset = reader.ReadOptional<Expression>(nullptr, state.gstate);
	return make_unique<LogicalLimit>(limit_val, offset_val, std::move(limit), std::move(offset));
}

} // namespace duckdb


#include <cmath>

namespace duckdb {

void LogicalLimitPercent::Serialize(FieldWriter &writer) const {
	writer.WriteField(limit_percent);
	writer.WriteField(offset_val);
	writer.WriteOptional(limit);
	writer.WriteOptional(offset);
}

unique_ptr<LogicalOperator> LogicalLimitPercent::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto limit_percent = reader.ReadRequired<double>();
	auto offset_val = reader.ReadRequired<int64_t>();
	auto limit = reader.ReadOptional<Expression>(nullptr, state.gstate);
	auto offset = reader.ReadOptional<Expression>(nullptr, state.gstate);
	return make_unique<LogicalLimitPercent>(limit_percent, offset_val, std::move(limit), std::move(offset));
}

idx_t LogicalLimitPercent::EstimateCardinality(ClientContext &context) {
	auto child_cardinality = LogicalOperator::EstimateCardinality(context);
	if ((limit_percent < 0 || limit_percent > 100) || std::isnan(limit_percent)) {
		return child_cardinality;
	}
	return idx_t(child_cardinality * (limit_percent / 100.0));
}

} // namespace duckdb




namespace duckdb {

void LogicalOrder::Serialize(FieldWriter &writer) const {
	writer.WriteRegularSerializableList(orders);
	writer.WriteList<idx_t>(projections);
}

unique_ptr<LogicalOperator> LogicalOrder::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto orders = reader.ReadRequiredSerializableList<BoundOrderByNode, BoundOrderByNode>(state.gstate);
	auto projections = reader.ReadRequiredList<idx_t>();
	auto result = make_unique<LogicalOrder>(std::move(orders));
	result->projections = std::move(projections);
	return std::move(result);
}

} // namespace duckdb


namespace duckdb {

LogicalPositionalJoin::LogicalPositionalJoin(unique_ptr<LogicalOperator> left, unique_ptr<LogicalOperator> right)
    : LogicalUnconditionalJoin(LogicalOperatorType::LOGICAL_POSITIONAL_JOIN, std::move(left), std::move(right)) {
}

unique_ptr<LogicalOperator> LogicalPositionalJoin::Create(unique_ptr<LogicalOperator> left,
                                                          unique_ptr<LogicalOperator> right) {
	if (left->type == LogicalOperatorType::LOGICAL_DUMMY_SCAN) {
		return right;
	}
	if (right->type == LogicalOperatorType::LOGICAL_DUMMY_SCAN) {
		return left;
	}
	return make_unique<LogicalPositionalJoin>(std::move(left), std::move(right));
}

void LogicalPositionalJoin::Serialize(FieldWriter &writer) const {
}

unique_ptr<LogicalOperator> LogicalPositionalJoin::Deserialize(LogicalDeserializationState &state,
                                                               FieldReader &reader) {
	// TODO(stephwang): review if unique_ptr<LogicalOperator> plan is needed
	auto result = unique_ptr<LogicalPositionalJoin>(new LogicalPositionalJoin());
	return std::move(result);
}

} // namespace duckdb


namespace duckdb {

void LogicalPragma::Serialize(FieldWriter &writer) const {
	throw NotImplementedException(LogicalOperatorToString(type));
}

unique_ptr<LogicalOperator> LogicalPragma::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	throw NotImplementedException(LogicalOperatorToString(state.type));
}

idx_t LogicalPragma::EstimateCardinality(ClientContext &context) {
	return 1;
}

} // namespace duckdb


namespace duckdb {

void LogicalPrepare::Serialize(FieldWriter &writer) const {
	throw NotImplementedException(LogicalOperatorToString(type));
}

unique_ptr<LogicalOperator> LogicalPrepare::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	throw NotImplementedException(LogicalOperatorToString(state.type));
}

idx_t LogicalPrepare::EstimateCardinality(ClientContext &context) {
	return 1;
}

} // namespace duckdb



namespace duckdb {

LogicalProjection::LogicalProjection(idx_t table_index, vector<unique_ptr<Expression>> select_list)
    : LogicalOperator(LogicalOperatorType::LOGICAL_PROJECTION, std::move(select_list)), table_index(table_index) {
}

vector<ColumnBinding> LogicalProjection::GetColumnBindings() {
	return GenerateColumnBindings(table_index, expressions.size());
}

void LogicalProjection::ResolveTypes() {
	for (auto &expr : expressions) {
		types.push_back(expr->return_type);
	}
}

void LogicalProjection::Serialize(FieldWriter &writer) const {
	writer.WriteField(table_index);
	writer.WriteSerializableList<Expression>(expressions);
}

unique_ptr<LogicalOperator> LogicalProjection::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto table_index = reader.ReadRequired<idx_t>();
	auto expressions = reader.ReadRequiredSerializableList<Expression>(state.gstate);
	return make_unique<LogicalProjection>(table_index, std::move(expressions));
}

vector<idx_t> LogicalProjection::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

} // namespace duckdb



namespace duckdb {

void LogicalRecursiveCTE::Serialize(FieldWriter &writer) const {
	writer.WriteField(table_index);
	writer.WriteField(column_count);
	writer.WriteField(union_all);
}

unique_ptr<LogicalOperator> LogicalRecursiveCTE::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto table_index = reader.ReadRequired<idx_t>();
	auto column_count = reader.ReadRequired<idx_t>();
	auto union_all = reader.ReadRequired<bool>();
	// TODO(stephwang): review if unique_ptr<LogicalOperator> plan is needed
	return unique_ptr<LogicalRecursiveCTE>(new LogicalRecursiveCTE(table_index, column_count, union_all, state.type));
}

vector<idx_t> LogicalRecursiveCTE::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

} // namespace duckdb



namespace duckdb {

void LogicalReset::Serialize(FieldWriter &writer) const {
	writer.WriteString(name);
	writer.WriteField(scope);
}

unique_ptr<LogicalOperator> LogicalReset::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto name = reader.ReadRequired<std::string>();
	auto scope = reader.ReadRequired<SetScope>();
	return make_unique<LogicalReset>(name, scope);
}

idx_t LogicalReset::EstimateCardinality(ClientContext &context) {
	return 1;
}

} // namespace duckdb



namespace duckdb {

LogicalSample::LogicalSample(unique_ptr<SampleOptions> sample_options_p, unique_ptr<LogicalOperator> child)
    : LogicalOperator(LogicalOperatorType::LOGICAL_SAMPLE), sample_options(std::move(sample_options_p)) {
	children.push_back(std::move(child));
}

vector<ColumnBinding> LogicalSample::GetColumnBindings() {
	return children[0]->GetColumnBindings();
}

idx_t LogicalSample::EstimateCardinality(ClientContext &context) {
	auto child_cardinality = children[0]->EstimateCardinality(context);
	if (sample_options->is_percentage) {
		double sample_cardinality =
		    double(child_cardinality) * (sample_options->sample_size.GetValue<double>() / 100.0);
		if (sample_cardinality > double(child_cardinality)) {
			return child_cardinality;
		}
		return idx_t(sample_cardinality);
	} else {
		auto sample_size = sample_options->sample_size.GetValue<uint64_t>();
		if (sample_size < child_cardinality) {
			return sample_size;
		}
	}
	return child_cardinality;
}

void LogicalSample::ResolveTypes() {
	types = children[0]->types;
}

void LogicalSample::Serialize(FieldWriter &writer) const {
	sample_options->Serialize(writer.GetSerializer());
}

unique_ptr<LogicalOperator> LogicalSample::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto sample_options = SampleOptions::Deserialize(reader.GetSource());
	// TODO(stephwang): review how to pass child LogicalOperator
	auto result = make_unique<LogicalSample>(std::move(sample_options), nullptr);
	return std::move(result);
}
} // namespace duckdb



namespace duckdb {

void LogicalSet::Serialize(FieldWriter &writer) const {
	writer.WriteString(name);
	value.Serialize(writer.GetSerializer());
	writer.WriteField(scope);
}

unique_ptr<LogicalOperator> LogicalSet::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto name = reader.ReadRequired<std::string>();
	auto value = Value::Deserialize(reader.GetSource());
	auto scope = reader.ReadRequired<SetScope>();
	return make_unique<LogicalSet>(name, value, scope);
}

idx_t LogicalSet::EstimateCardinality(ClientContext &context) {
	return 1;
}

} // namespace duckdb



namespace duckdb {

void LogicalSetOperation::Serialize(FieldWriter &writer) const {
	writer.WriteField(table_index);
	writer.WriteField(column_count);
}

unique_ptr<LogicalOperator> LogicalSetOperation::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto table_index = reader.ReadRequired<idx_t>();
	auto column_count = reader.ReadRequired<idx_t>();
	// TODO(stephwang): review if unique_ptr<LogicalOperator> plan is needed
	return unique_ptr<LogicalSetOperation>(new LogicalSetOperation(table_index, column_count, state.type));
}

vector<idx_t> LogicalSetOperation::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

} // namespace duckdb



namespace duckdb {

void LogicalShow::Serialize(FieldWriter &writer) const {
	writer.WriteRegularSerializableList(types_select);
	writer.WriteList<string>(aliases);
}

unique_ptr<LogicalOperator> LogicalShow::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto types_select = reader.ReadRequiredSerializableList<LogicalType, LogicalType>();
	auto aliases = reader.ReadRequiredList<string>();

	// TODO(stephwang): review if we need to pass unique_ptr<LogicalOperator> plan
	auto result = unique_ptr<LogicalShow>(new LogicalShow());
	result->types_select = types_select;
	result->aliases = aliases;
	return std::move(result);
}
} // namespace duckdb





namespace duckdb {

void LogicalSimple::Serialize(FieldWriter &writer) const {
	writer.WriteField<LogicalOperatorType>(type);
	switch (type) {
	case LogicalOperatorType::LOGICAL_ALTER:
		static_cast<const AlterInfo &>(*info).Serialize(writer.GetSerializer());
		break;
	case LogicalOperatorType::LOGICAL_DROP:
		static_cast<const DropInfo &>(*info).Serialize(writer.GetSerializer());
		break;
	case LogicalOperatorType::LOGICAL_LOAD:
		static_cast<const LoadInfo &>(*info).Serialize(writer.GetSerializer());
		break;
	default:
		throw NotImplementedException(LogicalOperatorToString(type));
	}
}

unique_ptr<LogicalOperator> LogicalSimple::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto type = reader.ReadRequired<LogicalOperatorType>();
	unique_ptr<ParseInfo> parse_info;
	switch (type) {
	case LogicalOperatorType::LOGICAL_ALTER:
		parse_info = AlterInfo::Deserialize(reader.GetSource());
		break;
	case LogicalOperatorType::LOGICAL_DROP:
		parse_info = DropInfo::Deserialize(reader.GetSource());
		break;
	case LogicalOperatorType::LOGICAL_LOAD:
		parse_info = LoadInfo::Deserialize(reader.GetSource());
		break;
	default:
		throw NotImplementedException(LogicalOperatorToString(state.type));
	}
	return make_unique<LogicalSimple>(type, std::move(parse_info));
}

idx_t LogicalSimple::EstimateCardinality(ClientContext &context) {
	return 1;
}

} // namespace duckdb



namespace duckdb {

void LogicalTopN::Serialize(FieldWriter &writer) const {
	writer.WriteRegularSerializableList(orders);
	writer.WriteField(offset);
	writer.WriteField(limit);
}

unique_ptr<LogicalOperator> LogicalTopN::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto orders = reader.ReadRequiredSerializableList<BoundOrderByNode, BoundOrderByNode>(state.gstate);
	auto offset = reader.ReadRequired<idx_t>();
	auto limit = reader.ReadRequired<idx_t>();
	return make_unique<LogicalTopN>(std::move(orders), limit, offset);
}

idx_t LogicalTopN::EstimateCardinality(ClientContext &context) {
	auto child_cardinality = LogicalOperator::EstimateCardinality(context);
	if (limit >= 0 && child_cardinality < idx_t(limit)) {
		return limit;
	}
	return child_cardinality;
}

} // namespace duckdb


namespace duckdb {

LogicalUnconditionalJoin::LogicalUnconditionalJoin(LogicalOperatorType logical_type, unique_ptr<LogicalOperator> left,
                                                   unique_ptr<LogicalOperator> right)
    : LogicalOperator(logical_type) {
	D_ASSERT(left);
	D_ASSERT(right);
	children.push_back(std::move(left));
	children.push_back(std::move(right));
}

vector<ColumnBinding> LogicalUnconditionalJoin::GetColumnBindings() {
	auto left_bindings = children[0]->GetColumnBindings();
	auto right_bindings = children[1]->GetColumnBindings();
	left_bindings.insert(left_bindings.end(), right_bindings.begin(), right_bindings.end());
	return left_bindings;
}

void LogicalUnconditionalJoin::ResolveTypes() {
	types.insert(types.end(), children[0]->types.begin(), children[0]->types.end());
	types.insert(types.end(), children[1]->types.begin(), children[1]->types.end());
}

} // namespace duckdb



namespace duckdb {

vector<ColumnBinding> LogicalUnnest::GetColumnBindings() {
	auto child_bindings = children[0]->GetColumnBindings();
	for (idx_t i = 0; i < expressions.size(); i++) {
		child_bindings.emplace_back(unnest_index, i);
	}
	return child_bindings;
}

void LogicalUnnest::ResolveTypes() {
	types.insert(types.end(), children[0]->types.begin(), children[0]->types.end());
	for (auto &expr : expressions) {
		types.push_back(expr->return_type);
	}
}

void LogicalUnnest::Serialize(FieldWriter &writer) const {
	writer.WriteField(unnest_index);
	writer.WriteSerializableList<Expression>(expressions);
}

unique_ptr<LogicalOperator> LogicalUnnest::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto unnest_index = reader.ReadRequired<idx_t>();
	auto expressions = reader.ReadRequiredSerializableList<Expression>(state.gstate);
	auto result = make_unique<LogicalUnnest>(unnest_index);
	result->expressions = std::move(expressions);
	return std::move(result);
}

vector<idx_t> LogicalUnnest::GetTableIndex() const {
	return vector<idx_t> {unnest_index};
}

} // namespace duckdb





namespace duckdb {

LogicalUpdate::LogicalUpdate(TableCatalogEntry *table)
    : LogicalOperator(LogicalOperatorType::LOGICAL_UPDATE), table(table), table_index(0), return_chunk(false) {
}

void LogicalUpdate::Serialize(FieldWriter &writer) const {
	table->Serialize(writer.GetSerializer());
	writer.WriteField(table_index);
	writer.WriteField(return_chunk);
	writer.WriteIndexList<PhysicalIndex>(columns);
	writer.WriteSerializableList(bound_defaults);
	writer.WriteField(update_is_del_and_insert);
}

unique_ptr<LogicalOperator> LogicalUpdate::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto &context = state.gstate.context;
	auto info = TableCatalogEntry::Deserialize(reader.GetSource(), context);
	auto &catalog = Catalog::GetCatalog(context, INVALID_CATALOG);

	TableCatalogEntry *table_catalog_entry = catalog.GetEntry<TableCatalogEntry>(context, info->schema, info->table);

	if (!table_catalog_entry) {
		throw InternalException("Cant find catalog entry for table %s", info->table);
	}

	auto result = make_unique<LogicalUpdate>(table_catalog_entry);
	result->table_index = reader.ReadRequired<idx_t>();
	result->return_chunk = reader.ReadRequired<bool>();
	result->columns = reader.ReadRequiredIndexList<PhysicalIndex>();
	result->bound_defaults = reader.ReadRequiredSerializableList<Expression>(state.gstate);
	result->update_is_del_and_insert = reader.ReadRequired<bool>();
	return std::move(result);
}

idx_t LogicalUpdate::EstimateCardinality(ClientContext &context) {
	return return_chunk ? LogicalOperator::EstimateCardinality(context) : 1;
}

vector<ColumnBinding> LogicalUpdate::GetColumnBindings() {
	if (return_chunk) {
		return GenerateColumnBindings(table_index, table->GetTypes().size());
	}
	return {ColumnBinding(0, 0)};
}

void LogicalUpdate::ResolveTypes() {
	if (return_chunk) {
		types = table->GetTypes();
	} else {
		types.emplace_back(LogicalType::BIGINT);
	}
}

} // namespace duckdb



namespace duckdb {

vector<ColumnBinding> LogicalWindow::GetColumnBindings() {
	auto child_bindings = children[0]->GetColumnBindings();
	for (idx_t i = 0; i < expressions.size(); i++) {
		child_bindings.emplace_back(window_index, i);
	}
	return child_bindings;
}

void LogicalWindow::ResolveTypes() {
	types.insert(types.end(), children[0]->types.begin(), children[0]->types.end());
	for (auto &expr : expressions) {
		types.push_back(expr->return_type);
	}
}

void LogicalWindow::Serialize(FieldWriter &writer) const {
	writer.WriteField(window_index);
	writer.WriteSerializableList<Expression>(expressions);
}

unique_ptr<LogicalOperator> LogicalWindow::Deserialize(LogicalDeserializationState &state, FieldReader &reader) {
	auto window_index = reader.ReadRequired<idx_t>();
	auto result = make_unique<LogicalWindow>(window_index);
	result->expressions = reader.ReadRequiredSerializableList<Expression>(state.gstate);
	return std::move(result);
}

vector<idx_t> LogicalWindow::GetTableIndex() const {
	return vector<idx_t> {window_index};
}

} // namespace duckdb





namespace duckdb {
void BoundCreateTableInfo::Serialize(Serializer &serializer) const {
	D_ASSERT(schema);
	schema->Serialize(serializer);
	serializer.WriteOptional(base);

	// TODO[YLM]: Review if we want/need to serialize more of the fields.
	//! The map of column names -> column index, used during binding
	// case_insensitive_map_t<column_t> name_map;

	//! Column dependency manager of the table
	// ColumnDependencyManager column_dependency_manager;

	serializer.WriteList(constraints);
	serializer.WriteList(bound_constraints);
	serializer.WriteList(bound_defaults);

	//! Dependents of the table (in e.g. default values)
	// unordered_set<CatalogEntry *> dependencies;

	//! The existing table data on disk (if any)
	// unique_ptr<PersistentTableData> data;

	//! CREATE TABLE from QUERY
	serializer.WriteOptional(query);

	//! Indexes created by this table <Block_ID, Offset>
	// vector<BlockPointer> indexes;
}

unique_ptr<BoundCreateTableInfo> BoundCreateTableInfo::Deserialize(Deserializer &source,
                                                                   PlanDeserializationState &state) {
	auto create_info = SchemaCatalogEntry::Deserialize(source);
	auto schema_name = create_info->schema;
	auto result = make_unique<BoundCreateTableInfo>(std::move(create_info));
	auto &context = state.context;
	result->schema = Catalog::GetSchema(context, INVALID_CATALOG, schema_name);
	result->base = source.ReadOptional<CreateInfo>();

	source.ReadList<Constraint>(result->constraints);
	source.ReadList<BoundConstraint>(result->bound_constraints);
	source.ReadList<Expression>(result->bound_defaults, state);

	result->query = source.ReadOptional<LogicalOperator>(state);
	return result;
}
} // namespace duckdb



namespace duckdb {

PlanDeserializationState::PlanDeserializationState(ClientContext &context) : context(context) {
}
PlanDeserializationState::~PlanDeserializationState() {
}

LogicalDeserializationState::LogicalDeserializationState(PlanDeserializationState &gstate, LogicalOperatorType type,
                                                         vector<unique_ptr<LogicalOperator>> &children)
    : gstate(gstate), type(type), children(children) {
}

ExpressionDeserializationState::ExpressionDeserializationState(PlanDeserializationState &gstate, ExpressionType type)
    : gstate(gstate), type(type) {
}

} // namespace duckdb













namespace duckdb {

Planner::Planner(ClientContext &context) : binder(Binder::CreateBinder(context)), context(context) {
}

static void CheckTreeDepth(const LogicalOperator &op, idx_t max_depth, idx_t depth = 0) {
	if (depth >= max_depth) {
		throw ParserException("Maximum tree depth of %lld exceeded in logical planner", max_depth);
	}
	for (auto &child : op.children) {
		CheckTreeDepth(*child, max_depth, depth + 1);
	}
}

void Planner::CreatePlan(SQLStatement &statement) {
	auto &profiler = QueryProfiler::Get(context);
	auto parameter_count = statement.n_param;

	BoundParameterMap bound_parameters(parameter_data);

	// first bind the tables and columns to the catalog
	bool parameters_resolved = true;
	try {
		profiler.StartPhase("binder");
		binder->parameters = &bound_parameters;
		auto bound_statement = binder->Bind(statement);
		profiler.EndPhase();

		this->names = bound_statement.names;
		this->types = bound_statement.types;
		this->plan = std::move(bound_statement.plan);

		auto max_tree_depth = ClientConfig::GetConfig(context).max_expression_depth;
		CheckTreeDepth(*plan, max_tree_depth);
	} catch (const ParameterNotResolvedException &ex) {
		// parameter types could not be resolved
		this->names = {"unknown"};
		this->types = {LogicalTypeId::UNKNOWN};
		this->plan = nullptr;
		parameters_resolved = false;
	} catch (const Exception &ex) {
		auto &config = DBConfig::GetConfig(context);

		this->plan = nullptr;
		for (auto &extension_op : config.operator_extensions) {
			auto bound_statement =
			    extension_op->Bind(context, *this->binder, extension_op->operator_info.get(), statement);
			if (bound_statement.plan != nullptr) {
				this->names = bound_statement.names;
				this->types = bound_statement.types;
				this->plan = std::move(bound_statement.plan);
				break;
			}
		}

		if (!this->plan) {
			throw;
		}
	} catch (std::exception &ex) {
		throw;
	}
	this->properties = binder->properties;
	this->properties.parameter_count = parameter_count;
	properties.bound_all_parameters = parameters_resolved;

	Planner::VerifyPlan(context, plan, &bound_parameters.parameters);

	// set up a map of parameter number -> value entries
	for (auto &kv : bound_parameters.parameters) {
		auto parameter_index = kv.first;
		auto &parameter_data = kv.second;
		// check if the type of the parameter could be resolved
		if (!parameter_data->return_type.IsValid()) {
			properties.bound_all_parameters = false;
			continue;
		}
		parameter_data->value = Value(parameter_data->return_type);
		value_map[parameter_index] = parameter_data;
	}
}

shared_ptr<PreparedStatementData> Planner::PrepareSQLStatement(unique_ptr<SQLStatement> statement) {
	auto copied_statement = statement->Copy();
	// create a plan of the underlying statement
	CreatePlan(std::move(statement));
	// now create the logical prepare
	auto prepared_data = make_shared<PreparedStatementData>(copied_statement->type);
	prepared_data->unbound_statement = std::move(copied_statement);
	prepared_data->names = names;
	prepared_data->types = types;
	prepared_data->value_map = std::move(value_map);
	prepared_data->properties = properties;
	prepared_data->catalog_version = MetaTransaction::Get(context).catalog_version;
	return prepared_data;
}

void Planner::CreatePlan(unique_ptr<SQLStatement> statement) {
	D_ASSERT(statement);
	switch (statement->type) {
	case StatementType::SELECT_STATEMENT:
	case StatementType::INSERT_STATEMENT:
	case StatementType::COPY_STATEMENT:
	case StatementType::DELETE_STATEMENT:
	case StatementType::UPDATE_STATEMENT:
	case StatementType::CREATE_STATEMENT:
	case StatementType::DROP_STATEMENT:
	case StatementType::ALTER_STATEMENT:
	case StatementType::TRANSACTION_STATEMENT:
	case StatementType::EXPLAIN_STATEMENT:
	case StatementType::VACUUM_STATEMENT:
	case StatementType::RELATION_STATEMENT:
	case StatementType::CALL_STATEMENT:
	case StatementType::EXPORT_STATEMENT:
	case StatementType::PRAGMA_STATEMENT:
	case StatementType::SHOW_STATEMENT:
	case StatementType::SET_STATEMENT:
	case StatementType::LOAD_STATEMENT:
	case StatementType::EXTENSION_STATEMENT:
	case StatementType::PREPARE_STATEMENT:
	case StatementType::EXECUTE_STATEMENT:
	case StatementType::LOGICAL_PLAN_STATEMENT:
	case StatementType::ATTACH_STATEMENT:
		CreatePlan(*statement);
		break;
	default:
		throw NotImplementedException("Cannot plan statement of type %s!", StatementTypeToString(statement->type));
	}
}

static bool OperatorSupportsSerialization(LogicalOperator &op) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_INSERT:
	case LogicalOperatorType::LOGICAL_UPDATE:
	case LogicalOperatorType::LOGICAL_DELETE:
	case LogicalOperatorType::LOGICAL_PREPARE:
	case LogicalOperatorType::LOGICAL_EXECUTE:
	case LogicalOperatorType::LOGICAL_ALTER:
	case LogicalOperatorType::LOGICAL_CREATE_TABLE:
	case LogicalOperatorType::LOGICAL_CREATE_INDEX:
	case LogicalOperatorType::LOGICAL_CREATE_SEQUENCE:
	case LogicalOperatorType::LOGICAL_CREATE_VIEW:
	case LogicalOperatorType::LOGICAL_CREATE_SCHEMA:
	case LogicalOperatorType::LOGICAL_CREATE_MACRO:
	case LogicalOperatorType::LOGICAL_PRAGMA:
	case LogicalOperatorType::LOGICAL_TRANSACTION:
	case LogicalOperatorType::LOGICAL_CREATE_TYPE:
	case LogicalOperatorType::LOGICAL_EXPLAIN:
	case LogicalOperatorType::LOGICAL_COPY_TO_FILE:
	case LogicalOperatorType::LOGICAL_VACUUM:
		// unsupported (for now)
		return false;
	default:
		break;
	}
	for (auto &child : op.children) {
		if (!OperatorSupportsSerialization(*child)) {
			return false;
		}
	}
	return true;
}

void Planner::VerifyPlan(ClientContext &context, unique_ptr<LogicalOperator> &op, bound_parameter_map_t *map) {
	if (!op || !ClientConfig::GetConfig(context).verify_serializer) {
		return;
	}
	//! SELECT only for now
	if (!OperatorSupportsSerialization(*op)) {
		return;
	}

	BufferedSerializer serializer;
	try {
		op->Serialize(serializer);
	} catch (NotImplementedException &ex) {
		// ignore for now (FIXME)
		return;
	}
	auto data = serializer.GetData();
	auto deserializer = BufferedDeserializer(data.data.get(), data.size);

	PlanDeserializationState state(context);
	auto new_plan = LogicalOperator::Deserialize(deserializer, state);
	if (map) {
		*map = std::move(state.parameter_data);
	}
	op = std::move(new_plan);
}

} // namespace duckdb
















namespace duckdb {

PragmaHandler::PragmaHandler(ClientContext &context) : context(context) {
}

void PragmaHandler::HandlePragmaStatementsInternal(vector<unique_ptr<SQLStatement>> &statements) {
	vector<unique_ptr<SQLStatement>> new_statements;
	for (idx_t i = 0; i < statements.size(); i++) {
		if (statements[i]->type == StatementType::PRAGMA_STATEMENT) {
			// PRAGMA statement: check if we need to replace it by a new set of statements
			PragmaHandler handler(context);
			auto new_query = handler.HandlePragma(statements[i].get());
			if (!new_query.empty()) {
				// this PRAGMA statement gets replaced by a new query string
				// push the new query string through the parser again and add it to the transformer
				Parser parser(context.GetParserOptions());
				parser.ParseQuery(new_query);
				// insert the new statements and remove the old statement
				for (idx_t j = 0; j < parser.statements.size(); j++) {
					new_statements.push_back(std::move(parser.statements[j]));
				}
				continue;
			}
		}
		new_statements.push_back(std::move(statements[i]));
	}
	statements = std::move(new_statements);
}

void PragmaHandler::HandlePragmaStatements(ClientContextLock &lock, vector<unique_ptr<SQLStatement>> &statements) {
	// first check if there are any pragma statements
	bool found_pragma = false;
	for (idx_t i = 0; i < statements.size(); i++) {
		if (statements[i]->type == StatementType::PRAGMA_STATEMENT) {
			found_pragma = true;
			break;
		}
	}
	if (!found_pragma) {
		// no pragmas: skip this step
		return;
	}
	context.RunFunctionInTransactionInternal(lock, [&]() { HandlePragmaStatementsInternal(statements); });
}

string PragmaHandler::HandlePragma(SQLStatement *statement) { // PragmaInfo &info
	auto info = *((PragmaStatement &)*statement).info;
	auto entry =
	    Catalog::GetEntry<PragmaFunctionCatalogEntry>(context, INVALID_CATALOG, DEFAULT_SCHEMA, info.name, false);
	string error;

	FunctionBinder function_binder(context);
	idx_t bound_idx = function_binder.BindFunction(entry->name, entry->functions, info, error);
	if (bound_idx == DConstants::INVALID_INDEX) {
		throw BinderException(error);
	}
	auto bound_function = entry->functions.GetFunctionByOffset(bound_idx);
	if (bound_function.query) {
		QueryErrorContext error_context(statement, statement->stmt_location);
		Binder::BindNamedParameters(bound_function.named_parameters, info.named_parameters, error_context,
		                            bound_function.name);
		FunctionParameters parameters {info.parameters, info.named_parameters};
		return bound_function.query(context, parameters);
	}
	return string();
}

} // namespace duckdb













namespace duckdb {

FlattenDependentJoins::FlattenDependentJoins(Binder &binder, const vector<CorrelatedColumnInfo> &correlated,
                                             bool perform_delim, bool any_join)
    : binder(binder), delim_offset(DConstants::INVALID_INDEX), correlated_columns(correlated),
      perform_delim(perform_delim), any_join(any_join) {
	for (idx_t i = 0; i < correlated_columns.size(); i++) {
		auto &col = correlated_columns[i];
		correlated_map[col.binding] = i;
		delim_types.push_back(col.type);
	}
}

bool FlattenDependentJoins::DetectCorrelatedExpressions(LogicalOperator *op, bool lateral) {
	D_ASSERT(op);
	// check if this entry has correlated expressions
	HasCorrelatedExpressions visitor(correlated_columns, lateral);
	visitor.VisitOperator(*op);
	bool has_correlation = visitor.has_correlated_expressions;
	// now visit the children of this entry and check if they have correlated expressions
	for (auto &child : op->children) {
		// we OR the property with its children such that has_correlation is true if either
		// (1) this node has a correlated expression or
		// (2) one of its children has a correlated expression
		if (DetectCorrelatedExpressions(child.get(), lateral)) {
			has_correlation = true;
		}
	}
	// set the entry in the map
	has_correlated_expressions[op] = has_correlation;
	return has_correlation;
}

unique_ptr<LogicalOperator> FlattenDependentJoins::PushDownDependentJoin(unique_ptr<LogicalOperator> plan) {
	bool propagate_null_values = true;
	auto result = PushDownDependentJoinInternal(std::move(plan), propagate_null_values);
	if (!replacement_map.empty()) {
		// check if we have to replace any COUNT aggregates into "CASE WHEN X IS NULL THEN 0 ELSE COUNT END"
		RewriteCountAggregates aggr(replacement_map);
		aggr.VisitOperator(*result);
	}
	return result;
}

bool SubqueryDependentFilter(Expression *expr) {
	if (expr->expression_class == ExpressionClass::BOUND_CONJUNCTION &&
	    expr->GetExpressionType() == ExpressionType::CONJUNCTION_AND) {
		auto bound_conjuction = (BoundConjunctionExpression *)expr;
		for (auto &child : bound_conjuction->children) {
			if (SubqueryDependentFilter(child.get())) {
				return true;
			}
		}
	}
	if (expr->expression_class == ExpressionClass::BOUND_SUBQUERY) {
		return true;
	}
	return false;
}
unique_ptr<LogicalOperator> FlattenDependentJoins::PushDownDependentJoinInternal(unique_ptr<LogicalOperator> plan,
                                                                                 bool &parent_propagate_null_values) {
	// first check if the logical operator has correlated expressions
	auto entry = has_correlated_expressions.find(plan.get());
	D_ASSERT(entry != has_correlated_expressions.end());
	if (!entry->second) {
		// we reached a node without correlated expressions
		// we can eliminate the dependent join now and create a simple cross product
		// now create the duplicate eliminated scan for this node
		auto left_columns = plan->GetColumnBindings().size();
		auto delim_index = binder.GenerateTableIndex();
		this->base_binding = ColumnBinding(delim_index, 0);
		this->delim_offset = left_columns;
		this->data_offset = 0;
		auto delim_scan = make_unique<LogicalDelimGet>(delim_index, delim_types);
		return LogicalCrossProduct::Create(std::move(plan), std::move(delim_scan));
	}
	switch (plan->type) {
	case LogicalOperatorType::LOGICAL_UNNEST:
	case LogicalOperatorType::LOGICAL_FILTER: {
		// filter
		// first we flatten the dependent join in the child of the filter
		for (auto &expr : plan->expressions) {
			any_join |= SubqueryDependentFilter(expr.get());
		}
		plan->children[0] = PushDownDependentJoinInternal(std::move(plan->children[0]), parent_propagate_null_values);

		// then we replace any correlated expressions with the corresponding entry in the correlated_map
		RewriteCorrelatedExpressions rewriter(base_binding, correlated_map);
		rewriter.VisitOperator(*plan);
		return plan;
	}
	case LogicalOperatorType::LOGICAL_PROJECTION: {
		// projection
		// first we flatten the dependent join in the child of the projection
		for (auto &expr : plan->expressions) {
			parent_propagate_null_values &= expr->PropagatesNullValues();
		}
		plan->children[0] = PushDownDependentJoinInternal(std::move(plan->children[0]), parent_propagate_null_values);

		// then we replace any correlated expressions with the corresponding entry in the correlated_map
		RewriteCorrelatedExpressions rewriter(base_binding, correlated_map);
		rewriter.VisitOperator(*plan);
		// now we add all the columns of the delim_scan to the projection list
		auto proj = (LogicalProjection *)plan.get();
		for (idx_t i = 0; i < correlated_columns.size(); i++) {
			auto &col = correlated_columns[i];
			auto colref = make_unique<BoundColumnRefExpression>(
			    col.name, col.type, ColumnBinding(base_binding.table_index, base_binding.column_index + i));
			plan->expressions.push_back(std::move(colref));
		}

		base_binding.table_index = proj->table_index;
		this->delim_offset = base_binding.column_index = plan->expressions.size() - correlated_columns.size();
		this->data_offset = 0;
		return plan;
	}
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
		auto &aggr = (LogicalAggregate &)*plan;
		// aggregate and group by
		// first we flatten the dependent join in the child of the projection
		for (auto &expr : plan->expressions) {
			parent_propagate_null_values &= expr->PropagatesNullValues();
		}
		plan->children[0] = PushDownDependentJoinInternal(std::move(plan->children[0]), parent_propagate_null_values);
		// then we replace any correlated expressions with the corresponding entry in the correlated_map
		RewriteCorrelatedExpressions rewriter(base_binding, correlated_map);
		rewriter.VisitOperator(*plan);
		// now we add all the columns of the delim_scan to the grouping operators AND the projection list
		idx_t delim_table_index;
		idx_t delim_column_offset;
		idx_t delim_data_offset;
		auto new_group_count = perform_delim ? correlated_columns.size() : 1;
		for (idx_t i = 0; i < new_group_count; i++) {
			auto &col = correlated_columns[i];
			auto colref = make_unique<BoundColumnRefExpression>(
			    col.name, col.type, ColumnBinding(base_binding.table_index, base_binding.column_index + i));
			for (auto &set : aggr.grouping_sets) {
				set.insert(aggr.groups.size());
			}
			aggr.groups.push_back(std::move(colref));
		}
		if (!perform_delim) {
			// if we are not performing the duplicate elimination, we have only added the row_id column to the grouping
			// operators in this case, we push a FIRST aggregate for each of the remaining expressions
			delim_table_index = aggr.aggregate_index;
			delim_column_offset = aggr.expressions.size();
			delim_data_offset = aggr.groups.size();
			for (idx_t i = 0; i < correlated_columns.size(); i++) {
				auto &col = correlated_columns[i];
				auto first_aggregate = FirstFun::GetFunction(col.type);
				auto colref = make_unique<BoundColumnRefExpression>(
				    col.name, col.type, ColumnBinding(base_binding.table_index, base_binding.column_index + i));
				vector<unique_ptr<Expression>> aggr_children;
				aggr_children.push_back(std::move(colref));
				auto first_fun =
				    make_unique<BoundAggregateExpression>(std::move(first_aggregate), std::move(aggr_children), nullptr,
				                                          nullptr, AggregateType::NON_DISTINCT);
				aggr.expressions.push_back(std::move(first_fun));
			}
		} else {
			delim_table_index = aggr.group_index;
			delim_column_offset = aggr.groups.size() - correlated_columns.size();
			delim_data_offset = aggr.groups.size();
		}
		if (aggr.groups.size() == new_group_count) {
			// we have to perform a LEFT OUTER JOIN between the result of this aggregate and the delim scan
			// FIXME: this does not always have to be a LEFT OUTER JOIN, depending on whether aggr.expressions return
			// NULL or a value
			unique_ptr<LogicalComparisonJoin> join = make_unique<LogicalComparisonJoin>(JoinType::INNER);
			for (auto &aggr_exp : aggr.expressions) {
				auto b_aggr_exp = (BoundAggregateExpression *)aggr_exp.get();
				if (!b_aggr_exp->PropagatesNullValues() || any_join || !parent_propagate_null_values) {
					join = make_unique<LogicalComparisonJoin>(JoinType::LEFT);
					break;
				}
			}
			auto left_index = binder.GenerateTableIndex();
			auto delim_scan = make_unique<LogicalDelimGet>(left_index, delim_types);
			join->children.push_back(std::move(delim_scan));
			join->children.push_back(std::move(plan));
			for (idx_t i = 0; i < new_group_count; i++) {
				auto &col = correlated_columns[i];
				JoinCondition cond;
				cond.left = make_unique<BoundColumnRefExpression>(col.name, col.type, ColumnBinding(left_index, i));
				cond.right = make_unique<BoundColumnRefExpression>(
				    correlated_columns[i].type, ColumnBinding(delim_table_index, delim_column_offset + i));
				cond.comparison = ExpressionType::COMPARE_NOT_DISTINCT_FROM;
				join->conditions.push_back(std::move(cond));
			}
			// for any COUNT aggregate we replace references to the column with: CASE WHEN COUNT(*) IS NULL THEN 0
			// ELSE COUNT(*) END
			for (idx_t i = 0; i < aggr.expressions.size(); i++) {
				D_ASSERT(aggr.expressions[i]->GetExpressionClass() == ExpressionClass::BOUND_AGGREGATE);
				auto bound = (BoundAggregateExpression *)&*aggr.expressions[i];
				vector<LogicalType> arguments;
				if (bound->function == CountFun::GetFunction() || bound->function == CountStarFun::GetFunction()) {
					// have to replace this ColumnBinding with the CASE expression
					replacement_map[ColumnBinding(aggr.aggregate_index, i)] = i;
				}
			}
			// now we update the delim_index
			base_binding.table_index = left_index;
			this->delim_offset = base_binding.column_index = 0;
			this->data_offset = 0;
			return std::move(join);
		} else {
			// update the delim_index
			base_binding.table_index = delim_table_index;
			this->delim_offset = base_binding.column_index = delim_column_offset;
			this->data_offset = delim_data_offset;
			return plan;
		}
	}
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT: {
		// cross product
		// push into both sides of the plan
		bool left_has_correlation = has_correlated_expressions.find(plan->children[0].get())->second;
		bool right_has_correlation = has_correlated_expressions.find(plan->children[1].get())->second;
		if (!right_has_correlation) {
			// only left has correlation: push into left
			plan->children[0] =
			    PushDownDependentJoinInternal(std::move(plan->children[0]), parent_propagate_null_values);
			return plan;
		}
		if (!left_has_correlation) {
			// only right has correlation: push into right
			plan->children[1] =
			    PushDownDependentJoinInternal(std::move(plan->children[1]), parent_propagate_null_values);
			return plan;
		}
		// both sides have correlation
		// turn into an inner join
		auto join = make_unique<LogicalComparisonJoin>(JoinType::INNER);
		plan->children[0] = PushDownDependentJoinInternal(std::move(plan->children[0]), parent_propagate_null_values);
		auto left_binding = this->base_binding;
		plan->children[1] = PushDownDependentJoinInternal(std::move(plan->children[1]), parent_propagate_null_values);
		// add the correlated columns to the join conditions
		for (idx_t i = 0; i < correlated_columns.size(); i++) {
			JoinCondition cond;
			cond.left = make_unique<BoundColumnRefExpression>(
			    correlated_columns[i].type, ColumnBinding(left_binding.table_index, left_binding.column_index + i));
			cond.right = make_unique<BoundColumnRefExpression>(
			    correlated_columns[i].type, ColumnBinding(base_binding.table_index, base_binding.column_index + i));
			cond.comparison = ExpressionType::COMPARE_NOT_DISTINCT_FROM;
			join->conditions.push_back(std::move(cond));
		}
		join->children.push_back(std::move(plan->children[0]));
		join->children.push_back(std::move(plan->children[1]));
		return std::move(join);
	}
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
		auto &join = (LogicalJoin &)*plan;
		D_ASSERT(plan->children.size() == 2);
		// check the correlated expressions in the children of the join
		bool left_has_correlation = has_correlated_expressions.find(plan->children[0].get())->second;
		bool right_has_correlation = has_correlated_expressions.find(plan->children[1].get())->second;

		if (join.join_type == JoinType::INNER) {
			// inner join
			if (!right_has_correlation) {
				// only left has correlation: push into left
				plan->children[0] =
				    PushDownDependentJoinInternal(std::move(plan->children[0]), parent_propagate_null_values);
				return plan;
			}
			if (!left_has_correlation) {
				// only right has correlation: push into right
				plan->children[1] =
				    PushDownDependentJoinInternal(std::move(plan->children[1]), parent_propagate_null_values);
				return plan;
			}
		} else if (join.join_type == JoinType::LEFT) {
			// left outer join
			if (!right_has_correlation) {
				// only left has correlation: push into left
				plan->children[0] =
				    PushDownDependentJoinInternal(std::move(plan->children[0]), parent_propagate_null_values);
				return plan;
			}
		} else if (join.join_type == JoinType::RIGHT) {
			// left outer join
			if (!left_has_correlation) {
				// only right has correlation: push into right
				plan->children[1] =
				    PushDownDependentJoinInternal(std::move(plan->children[1]), parent_propagate_null_values);
				return plan;
			}
		} else if (join.join_type == JoinType::MARK) {
			if (right_has_correlation) {
				throw Exception("MARK join with correlation in RHS not supported");
			}
			// push the child into the LHS
			plan->children[0] =
			    PushDownDependentJoinInternal(std::move(plan->children[0]), parent_propagate_null_values);
			// rewrite expressions in the join conditions
			RewriteCorrelatedExpressions rewriter(base_binding, correlated_map);
			rewriter.VisitOperator(*plan);
			return plan;
		} else {
			throw Exception("Unsupported join type for flattening correlated subquery");
		}
		// both sides have correlation
		// push into both sides
		plan->children[0] = PushDownDependentJoinInternal(std::move(plan->children[0]), parent_propagate_null_values);
		auto left_binding = this->base_binding;
		plan->children[1] = PushDownDependentJoinInternal(std::move(plan->children[1]), parent_propagate_null_values);
		auto right_binding = this->base_binding;
		// NOTE: for OUTER JOINS it matters what the BASE BINDING is after the join
		// for the LEFT OUTER JOIN, we want the LEFT side to be the base binding after we push
		// because the RIGHT binding might contain NULL values
		if (join.join_type == JoinType::LEFT) {
			this->base_binding = left_binding;
		} else if (join.join_type == JoinType::RIGHT) {
			this->base_binding = right_binding;
		}
		// add the correlated columns to the join conditions
		for (idx_t i = 0; i < correlated_columns.size(); i++) {
			auto left = make_unique<BoundColumnRefExpression>(
			    correlated_columns[i].type, ColumnBinding(left_binding.table_index, left_binding.column_index + i));
			auto right = make_unique<BoundColumnRefExpression>(
			    correlated_columns[i].type, ColumnBinding(right_binding.table_index, right_binding.column_index + i));

			if (join.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
				JoinCondition cond;
				cond.left = std::move(left);
				cond.right = std::move(right);
				cond.comparison = ExpressionType::COMPARE_NOT_DISTINCT_FROM;

				auto &comparison_join = (LogicalComparisonJoin &)join;
				comparison_join.conditions.push_back(std::move(cond));
			} else {
				auto &any_join = (LogicalAnyJoin &)join;
				auto comparison = make_unique<BoundComparisonExpression>(ExpressionType::COMPARE_NOT_DISTINCT_FROM,
				                                                         std::move(left), std::move(right));
				auto conjunction = make_unique<BoundConjunctionExpression>(
				    ExpressionType::CONJUNCTION_AND, std::move(comparison), std::move(any_join.condition));
				any_join.condition = std::move(conjunction);
			}
		}
		// then we replace any correlated expressions with the corresponding entry in the correlated_map
		RewriteCorrelatedExpressions rewriter(right_binding, correlated_map);
		rewriter.VisitOperator(*plan);
		return plan;
	}
	case LogicalOperatorType::LOGICAL_LIMIT: {
		auto &limit = (LogicalLimit &)*plan;
		if (limit.limit || limit.offset) {
			throw ParserException("Non-constant limit or offset not supported in correlated subquery");
		}
		auto rownum_alias = "limit_rownum";
		unique_ptr<LogicalOperator> child;
		unique_ptr<LogicalOrder> order_by;

		// check if the direct child of this LIMIT node is an ORDER BY node, if so, keep it separate
		// this is done for an optimization to avoid having to compute the total order
		if (plan->children[0]->type == LogicalOperatorType::LOGICAL_ORDER_BY) {
			order_by = unique_ptr_cast<LogicalOperator, LogicalOrder>(std::move(plan->children[0]));
			child = PushDownDependentJoinInternal(std::move(order_by->children[0]), parent_propagate_null_values);
		} else {
			child = PushDownDependentJoinInternal(std::move(plan->children[0]), parent_propagate_null_values);
		}
		auto child_column_count = child->GetColumnBindings().size();
		// we push a row_number() OVER (PARTITION BY [correlated columns])
		auto window_index = binder.GenerateTableIndex();
		auto window = make_unique<LogicalWindow>(window_index);
		auto row_number = make_unique<BoundWindowExpression>(ExpressionType::WINDOW_ROW_NUMBER, LogicalType::BIGINT,
		                                                     nullptr, nullptr);
		auto partition_count = perform_delim ? correlated_columns.size() : 1;
		for (idx_t i = 0; i < partition_count; i++) {
			auto &col = correlated_columns[i];
			auto colref = make_unique<BoundColumnRefExpression>(
			    col.name, col.type, ColumnBinding(base_binding.table_index, base_binding.column_index + i));
			row_number->partitions.push_back(std::move(colref));
		}
		if (order_by) {
			// optimization: if there is an ORDER BY node followed by a LIMIT
			// rather than computing the entire order, we push the ORDER BY expressions into the row_num computation
			// this way, the order only needs to be computed per partition
			row_number->orders = std::move(order_by->orders);
		}
		row_number->start = WindowBoundary::UNBOUNDED_PRECEDING;
		row_number->end = WindowBoundary::CURRENT_ROW_ROWS;
		window->expressions.push_back(std::move(row_number));
		window->children.push_back(std::move(child));

		// add a filter based on the row_number
		// the filter we add is "row_number > offset AND row_number <= offset + limit"
		auto filter = make_unique<LogicalFilter>();
		unique_ptr<Expression> condition;
		auto row_num_ref =
		    make_unique<BoundColumnRefExpression>(rownum_alias, LogicalType::BIGINT, ColumnBinding(window_index, 0));

		int64_t upper_bound_limit = NumericLimits<int64_t>::Maximum();
		TryAddOperator::Operation(limit.offset_val, limit.limit_val, upper_bound_limit);
		auto upper_bound = make_unique<BoundConstantExpression>(Value::BIGINT(upper_bound_limit));
		condition = make_unique<BoundComparisonExpression>(ExpressionType::COMPARE_LESSTHANOREQUALTO,
		                                                   row_num_ref->Copy(), std::move(upper_bound));
		// we only need to add "row_number >= offset + 1" if offset is bigger than 0
		if (limit.offset_val > 0) {
			auto lower_bound = make_unique<BoundConstantExpression>(Value::BIGINT(limit.offset_val));
			auto lower_comp = make_unique<BoundComparisonExpression>(ExpressionType::COMPARE_GREATERTHAN,
			                                                         row_num_ref->Copy(), std::move(lower_bound));
			auto conj = make_unique<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND, std::move(lower_comp),
			                                                    std::move(condition));
			condition = std::move(conj);
		}
		filter->expressions.push_back(std::move(condition));
		filter->children.push_back(std::move(window));
		// we prune away the row_number after the filter clause using the projection map
		for (idx_t i = 0; i < child_column_count; i++) {
			filter->projection_map.push_back(i);
		}
		return std::move(filter);
	}
	case LogicalOperatorType::LOGICAL_LIMIT_PERCENT: {
		// NOTE: limit percent could be supported in a manner similar to the LIMIT above
		// but instead of filtering by an exact number of rows, the limit should be expressed as
		// COUNT computed over the partition multiplied by the percentage
		throw ParserException("Limit percent operator not supported in correlated subquery");
	}
	case LogicalOperatorType::LOGICAL_WINDOW: {
		auto &window = (LogicalWindow &)*plan;
		// push into children
		plan->children[0] = PushDownDependentJoinInternal(std::move(plan->children[0]), parent_propagate_null_values);
		// add the correlated columns to the PARTITION BY clauses in the Window
		for (auto &expr : window.expressions) {
			D_ASSERT(expr->GetExpressionClass() == ExpressionClass::BOUND_WINDOW);
			auto &w = (BoundWindowExpression &)*expr;
			for (idx_t i = 0; i < correlated_columns.size(); i++) {
				w.partitions.push_back(make_unique<BoundColumnRefExpression>(
				    correlated_columns[i].type,
				    ColumnBinding(base_binding.table_index, base_binding.column_index + i)));
			}
		}
		return plan;
	}
	case LogicalOperatorType::LOGICAL_EXCEPT:
	case LogicalOperatorType::LOGICAL_INTERSECT:
	case LogicalOperatorType::LOGICAL_UNION: {
		auto &setop = (LogicalSetOperation &)*plan;
		// set operator, push into both children
#ifdef DEBUG
		plan->children[0]->ResolveOperatorTypes();
		plan->children[1]->ResolveOperatorTypes();
		D_ASSERT(plan->children[0]->types == plan->children[1]->types);
#endif
		plan->children[0] = PushDownDependentJoin(std::move(plan->children[0]));
		plan->children[1] = PushDownDependentJoin(std::move(plan->children[1]));
#ifdef DEBUG
		D_ASSERT(plan->children[0]->GetColumnBindings().size() == plan->children[1]->GetColumnBindings().size());
		plan->children[0]->ResolveOperatorTypes();
		plan->children[1]->ResolveOperatorTypes();
		D_ASSERT(plan->children[0]->types == plan->children[1]->types);
#endif
		// we have to refer to the setop index now
		base_binding.table_index = setop.table_index;
		base_binding.column_index = setop.column_count;
		setop.column_count += correlated_columns.size();
		return plan;
	}
	case LogicalOperatorType::LOGICAL_DISTINCT: {
		auto &distinct = (LogicalDistinct &)*plan;
		// push down into child
		distinct.children[0] = PushDownDependentJoin(std::move(distinct.children[0]));
		// add all correlated columns to the distinct targets
		for (idx_t i = 0; i < correlated_columns.size(); i++) {
			distinct.distinct_targets.push_back(make_unique<BoundColumnRefExpression>(
			    correlated_columns[i].type, ColumnBinding(base_binding.table_index, base_binding.column_index + i)));
		}
		return plan;
	}
	case LogicalOperatorType::LOGICAL_EXPRESSION_GET: {
		// expression get
		// first we flatten the dependent join in the child
		plan->children[0] = PushDownDependentJoinInternal(std::move(plan->children[0]), parent_propagate_null_values);
		// then we replace any correlated expressions with the corresponding entry in the correlated_map
		RewriteCorrelatedExpressions rewriter(base_binding, correlated_map);
		rewriter.VisitOperator(*plan);
		// now we add all the correlated columns to each of the expressions of the expression scan
		auto expr_get = (LogicalExpressionGet *)plan.get();
		for (idx_t i = 0; i < correlated_columns.size(); i++) {
			for (auto &expr_list : expr_get->expressions) {
				auto colref = make_unique<BoundColumnRefExpression>(
				    correlated_columns[i].type, ColumnBinding(base_binding.table_index, base_binding.column_index + i));
				expr_list.push_back(std::move(colref));
			}
			expr_get->expr_types.push_back(correlated_columns[i].type);
		}

		base_binding.table_index = expr_get->table_index;
		this->delim_offset = base_binding.column_index = expr_get->expr_types.size() - correlated_columns.size();
		this->data_offset = 0;
		return plan;
	}
	case LogicalOperatorType::LOGICAL_ORDER_BY:
		plan->children[0] = PushDownDependentJoin(std::move(plan->children[0]));
		return plan;
	case LogicalOperatorType::LOGICAL_GET: {
		auto &get = (LogicalGet &)*plan;
		if (get.children.size() != 1) {
			throw InternalException("Flatten dependent joins - logical get encountered without children");
		}
		plan->children[0] = PushDownDependentJoin(std::move(plan->children[0]));
		for (idx_t i = 0; i < (perform_delim ? correlated_columns.size() : 1); i++) {
			get.projected_input.push_back(this->delim_offset + i);
		}
		this->delim_offset = get.returned_types.size();
		this->data_offset = 0;
		return plan;
	}
	case LogicalOperatorType::LOGICAL_RECURSIVE_CTE: {
		throw BinderException("Recursive CTEs not supported in correlated subquery");
	}
	case LogicalOperatorType::LOGICAL_DELIM_JOIN: {
		throw BinderException("Nested lateral joins or lateral joins in correlated subqueries are not (yet) supported");
	}
	case LogicalOperatorType::LOGICAL_SAMPLE:
		throw BinderException("Sampling in correlated subqueries is not (yet) supported");
	default:
		throw InternalException("Logical operator type \"%s\" for dependent join", LogicalOperatorToString(plan->type));
	}
}

} // namespace duckdb





#include <algorithm>

namespace duckdb {

HasCorrelatedExpressions::HasCorrelatedExpressions(const vector<CorrelatedColumnInfo> &correlated, bool lateral)
    : has_correlated_expressions(false), lateral(lateral), correlated_columns(correlated) {
}

void HasCorrelatedExpressions::VisitOperator(LogicalOperator &op) {
	VisitOperatorExpressions(op);
}

unique_ptr<Expression> HasCorrelatedExpressions::VisitReplace(BoundColumnRefExpression &expr,
                                                              unique_ptr<Expression> *expr_ptr) {
	if (expr.depth == 0) {
		return nullptr;
	}
	if (expr.depth > 1) {
		if (lateral) {
			throw BinderException("Nested lateral joins are not (yet) supported");
		}
		throw InternalException("Expression with depth > 1 detected in non-lateral join");
	}
	// correlated column reference
	D_ASSERT(expr.depth == 1);
	has_correlated_expressions = true;
	return nullptr;
}

unique_ptr<Expression> HasCorrelatedExpressions::VisitReplace(BoundSubqueryExpression &expr,
                                                              unique_ptr<Expression> *expr_ptr) {
	if (!expr.IsCorrelated()) {
		return nullptr;
	}
	// check if the subquery contains any of the correlated expressions that we are concerned about in this node
	for (idx_t i = 0; i < correlated_columns.size(); i++) {
		if (std::find(expr.binder->correlated_columns.begin(), expr.binder->correlated_columns.end(),
		              correlated_columns[i]) != expr.binder->correlated_columns.end()) {
			has_correlated_expressions = true;
			break;
		}
	}
	return nullptr;
}

} // namespace duckdb









namespace duckdb {

RewriteCorrelatedExpressions::RewriteCorrelatedExpressions(ColumnBinding base_binding,
                                                           column_binding_map_t<idx_t> &correlated_map)
    : base_binding(base_binding), correlated_map(correlated_map) {
}

void RewriteCorrelatedExpressions::VisitOperator(LogicalOperator &op) {
	VisitOperatorExpressions(op);
}

unique_ptr<Expression> RewriteCorrelatedExpressions::VisitReplace(BoundColumnRefExpression &expr,
                                                                  unique_ptr<Expression> *expr_ptr) {
	if (expr.depth == 0) {
		return nullptr;
	}
	// correlated column reference
	// replace with the entry referring to the duplicate eliminated scan
	// if this assertion occurs it generally means the correlated expressions were not propagated correctly
	// through different binders
	D_ASSERT(expr.depth == 1);
	auto entry = correlated_map.find(expr.binding);
	D_ASSERT(entry != correlated_map.end());

	expr.binding = ColumnBinding(base_binding.table_index, base_binding.column_index + entry->second);
	expr.depth = 0;
	return nullptr;
}

unique_ptr<Expression> RewriteCorrelatedExpressions::VisitReplace(BoundSubqueryExpression &expr,
                                                                  unique_ptr<Expression> *expr_ptr) {
	if (!expr.IsCorrelated()) {
		return nullptr;
	}
	// subquery detected within this subquery
	// recursively rewrite it using the RewriteCorrelatedRecursive class
	RewriteCorrelatedRecursive rewrite(expr, base_binding, correlated_map);
	rewrite.RewriteCorrelatedSubquery(expr);
	return nullptr;
}

RewriteCorrelatedExpressions::RewriteCorrelatedRecursive::RewriteCorrelatedRecursive(
    BoundSubqueryExpression &parent, ColumnBinding base_binding, column_binding_map_t<idx_t> &correlated_map)
    : parent(parent), base_binding(base_binding), correlated_map(correlated_map) {
}

void RewriteCorrelatedExpressions::RewriteCorrelatedRecursive::RewriteCorrelatedSubquery(
    BoundSubqueryExpression &expr) {
	// rewrite the binding in the correlated list of the subquery)
	for (auto &corr : expr.binder->correlated_columns) {
		auto entry = correlated_map.find(corr.binding);
		if (entry != correlated_map.end()) {
			corr.binding = ColumnBinding(base_binding.table_index, base_binding.column_index + entry->second);
		}
	}
	// now rewrite any correlated BoundColumnRef expressions inside the subquery
	ExpressionIterator::EnumerateQueryNodeChildren(*expr.subquery,
	                                               [&](Expression &child) { RewriteCorrelatedExpressions(child); });
}

void RewriteCorrelatedExpressions::RewriteCorrelatedRecursive::RewriteCorrelatedExpressions(Expression &child) {
	if (child.type == ExpressionType::BOUND_COLUMN_REF) {
		// bound column reference
		auto &bound_colref = (BoundColumnRefExpression &)child;
		if (bound_colref.depth == 0) {
			// not a correlated column, ignore
			return;
		}
		// correlated column
		// check the correlated map
		auto entry = correlated_map.find(bound_colref.binding);
		if (entry != correlated_map.end()) {
			// we found the column in the correlated map!
			// update the binding and reduce the depth by 1
			bound_colref.binding = ColumnBinding(base_binding.table_index, base_binding.column_index + entry->second);
			bound_colref.depth--;
		}
	} else if (child.type == ExpressionType::SUBQUERY) {
		// we encountered another subquery: rewrite recursively
		D_ASSERT(child.GetExpressionClass() == ExpressionClass::BOUND_SUBQUERY);
		auto &bound_subquery = (BoundSubqueryExpression &)child;
		RewriteCorrelatedRecursive rewrite(bound_subquery, base_binding, correlated_map);
		rewrite.RewriteCorrelatedSubquery(bound_subquery);
	}
}

RewriteCountAggregates::RewriteCountAggregates(column_binding_map_t<idx_t> &replacement_map)
    : replacement_map(replacement_map) {
}

unique_ptr<Expression> RewriteCountAggregates::VisitReplace(BoundColumnRefExpression &expr,
                                                            unique_ptr<Expression> *expr_ptr) {
	auto entry = replacement_map.find(expr.binding);
	if (entry != replacement_map.end()) {
		// reference to a COUNT(*) aggregate
		// replace this with CASE WHEN COUNT(*) IS NULL THEN 0 ELSE COUNT(*) END
		auto is_null = make_unique<BoundOperatorExpression>(ExpressionType::OPERATOR_IS_NULL, LogicalType::BOOLEAN);
		is_null->children.push_back(expr.Copy());
		auto check = std::move(is_null);
		auto result_if_true = make_unique<BoundConstantExpression>(Value::Numeric(expr.return_type, 0));
		auto result_if_false = std::move(*expr_ptr);
		return make_unique<BoundCaseExpression>(std::move(check), std::move(result_if_true),
		                                        std::move(result_if_false));
	}
	return nullptr;
}

} // namespace duckdb













#include <algorithm>

namespace duckdb {

Binding::Binding(BindingType binding_type, const string &alias, vector<LogicalType> coltypes, vector<string> colnames,
                 idx_t index)
    : binding_type(binding_type), alias(alias), index(index), types(std::move(coltypes)), names(std::move(colnames)) {
	D_ASSERT(types.size() == names.size());
	for (idx_t i = 0; i < names.size(); i++) {
		auto &name = names[i];
		D_ASSERT(!name.empty());
		if (name_map.find(name) != name_map.end()) {
			throw BinderException("table \"%s\" has duplicate column name \"%s\"", alias, name);
		}
		name_map[name] = i;
	}
}

bool Binding::TryGetBindingIndex(const string &column_name, column_t &result) {
	auto entry = name_map.find(column_name);
	if (entry == name_map.end()) {
		return false;
	}
	auto column_info = entry->second;
	result = column_info;
	return true;
}

column_t Binding::GetBindingIndex(const string &column_name) {
	column_t result;
	if (!TryGetBindingIndex(column_name, result)) {
		throw InternalException("Binding index for column \"%s\" not found", column_name);
	}
	return result;
}

bool Binding::HasMatchingBinding(const string &column_name) {
	column_t result;
	return TryGetBindingIndex(column_name, result);
}

string Binding::ColumnNotFoundError(const string &column_name) const {
	return StringUtil::Format("Values list \"%s\" does not have a column named \"%s\"", alias, column_name);
}

BindResult Binding::Bind(ColumnRefExpression &colref, idx_t depth) {
	column_t column_index;
	bool success = false;
	success = TryGetBindingIndex(colref.GetColumnName(), column_index);
	if (!success) {
		return BindResult(ColumnNotFoundError(colref.GetColumnName()));
	}
	ColumnBinding binding;
	binding.table_index = index;
	binding.column_index = column_index;
	LogicalType sql_type = types[column_index];
	if (colref.alias.empty()) {
		colref.alias = names[column_index];
	}
	return BindResult(make_unique<BoundColumnRefExpression>(colref.GetName(), sql_type, binding, depth));
}

StandardEntry *Binding::GetStandardEntry() {
	return nullptr;
}

EntryBinding::EntryBinding(const string &alias, vector<LogicalType> types_p, vector<string> names_p, idx_t index,
                           StandardEntry &entry)
    : Binding(BindingType::CATALOG_ENTRY, alias, std::move(types_p), std::move(names_p), index), entry(entry) {
}

StandardEntry *EntryBinding::GetStandardEntry() {
	return &this->entry;
}

TableBinding::TableBinding(const string &alias, vector<LogicalType> types_p, vector<string> names_p,
                           vector<column_t> &bound_column_ids, StandardEntry *entry, idx_t index, bool add_row_id)
    : Binding(BindingType::TABLE, alias, std::move(types_p), std::move(names_p), index),
      bound_column_ids(bound_column_ids), entry(entry) {
	if (add_row_id) {
		if (name_map.find("rowid") == name_map.end()) {
			name_map["rowid"] = COLUMN_IDENTIFIER_ROW_ID;
		}
	}
}

static void ReplaceAliases(ParsedExpression &expr, const ColumnList &list,
                           const unordered_map<idx_t, string> &alias_map) {
	if (expr.type == ExpressionType::COLUMN_REF) {
		auto &colref = (ColumnRefExpression &)expr;
		D_ASSERT(!colref.IsQualified());
		auto &col_names = colref.column_names;
		D_ASSERT(col_names.size() == 1);
		auto idx_entry = list.GetColumnIndex(col_names[0]);
		auto &alias = alias_map.at(idx_entry.index);
		col_names = {alias};
	}
	ParsedExpressionIterator::EnumerateChildren(
	    expr, [&](const ParsedExpression &child) { ReplaceAliases((ParsedExpression &)child, list, alias_map); });
}

static void BakeTableName(ParsedExpression &expr, const string &table_name) {
	if (expr.type == ExpressionType::COLUMN_REF) {
		auto &colref = (ColumnRefExpression &)expr;
		D_ASSERT(!colref.IsQualified());
		auto &col_names = colref.column_names;
		col_names.insert(col_names.begin(), table_name);
	}
	ParsedExpressionIterator::EnumerateChildren(
	    expr, [&](const ParsedExpression &child) { BakeTableName((ParsedExpression &)child, table_name); });
}

unique_ptr<ParsedExpression> TableBinding::ExpandGeneratedColumn(const string &column_name) {
	auto catalog_entry = GetStandardEntry();
	D_ASSERT(catalog_entry); // Should only be called on a TableBinding

	D_ASSERT(catalog_entry->type == CatalogType::TABLE_ENTRY);
	auto table_entry = (TableCatalogEntry *)catalog_entry;

	// Get the index of the generated column
	auto column_index = GetBindingIndex(column_name);
	D_ASSERT(table_entry->GetColumn(LogicalIndex(column_index)).Generated());
	// Get a copy of the generated column
	auto expression = table_entry->GetColumn(LogicalIndex(column_index)).GeneratedExpression().Copy();
	unordered_map<idx_t, string> alias_map;
	for (auto &entry : name_map) {
		alias_map[entry.second] = entry.first;
	}
	ReplaceAliases(*expression, table_entry->GetColumns(), alias_map);
	BakeTableName(*expression, alias);
	return (expression);
}

const vector<column_t> &TableBinding::GetBoundColumnIds() const {
#ifdef DEBUG
	unordered_set<column_t> column_ids;
	for (auto &id : bound_column_ids) {
		auto result = column_ids.insert(id);
		// assert that all entries in the bound_column_ids are unique
		D_ASSERT(result.second);
		auto it = std::find_if(name_map.begin(), name_map.end(),
		                       [&](const std::pair<const string, column_t> &it) { return it.second == id; });
		// assert that every id appears in the name_map
		D_ASSERT(it != name_map.end());
		// the order that they appear in is not guaranteed to be sequential
	}
#endif
	return bound_column_ids;
}

ColumnBinding TableBinding::GetColumnBinding(column_t column_index) {
	auto &column_ids = bound_column_ids;
	ColumnBinding binding;

	// Locate the column_id that matches the 'column_index'
	auto it = std::find_if(column_ids.begin(), column_ids.end(),
	                       [&](const column_t &id) -> bool { return id == column_index; });
	// Get the index of it
	binding.column_index = std::distance(column_ids.begin(), it);
	// If it wasn't found, add it
	if (it == column_ids.end()) {
		column_ids.push_back(column_index);
	}

	binding.table_index = index;
	return binding;
}

BindResult TableBinding::Bind(ColumnRefExpression &colref, idx_t depth) {
	auto &column_name = colref.GetColumnName();
	column_t column_index;
	bool success = false;
	success = TryGetBindingIndex(column_name, column_index);
	if (!success) {
		return BindResult(ColumnNotFoundError(column_name));
	}
	auto entry = GetStandardEntry();
	if (entry && column_index != COLUMN_IDENTIFIER_ROW_ID) {
		D_ASSERT(entry->type == CatalogType::TABLE_ENTRY);
		// Either there is no table, or the columns category has to be standard
		auto table_entry = (TableCatalogEntry *)entry;
		auto &column_entry = table_entry->GetColumn(LogicalIndex(column_index));
		(void)table_entry;
		(void)column_entry;
		D_ASSERT(column_entry.Category() == TableColumnType::STANDARD);
	}
	// fetch the type of the column
	LogicalType col_type;
	if (column_index == COLUMN_IDENTIFIER_ROW_ID) {
		// row id: BIGINT type
		col_type = LogicalType::BIGINT;
	} else {
		// normal column: fetch type from base column
		col_type = types[column_index];
		if (colref.alias.empty()) {
			colref.alias = names[column_index];
		}
	}
	ColumnBinding binding = GetColumnBinding(column_index);
	return BindResult(make_unique<BoundColumnRefExpression>(colref.GetName(), col_type, binding, depth));
}

StandardEntry *TableBinding::GetStandardEntry() {
	return entry;
}

string TableBinding::ColumnNotFoundError(const string &column_name) const {
	return StringUtil::Format("Table \"%s\" does not have a column named \"%s\"", alias, column_name);
}

DummyBinding::DummyBinding(vector<LogicalType> types_p, vector<string> names_p, string dummy_name_p)
    : Binding(BindingType::DUMMY, DummyBinding::DUMMY_NAME + dummy_name_p, std::move(types_p), std::move(names_p),
              DConstants::INVALID_INDEX),
      dummy_name(std::move(dummy_name_p)) {
}

BindResult DummyBinding::Bind(ColumnRefExpression &colref, idx_t depth) {
	column_t column_index;
	if (!TryGetBindingIndex(colref.GetColumnName(), column_index)) {
		throw InternalException("Column %s not found in bindings", colref.GetColumnName());
	}
	ColumnBinding binding(index, column_index);

	// we are binding a parameter to create the dummy binding, no arguments are supplied
	return BindResult(make_unique<BoundColumnRefExpression>(colref.GetName(), types[column_index], binding, depth));
}

BindResult DummyBinding::Bind(ColumnRefExpression &colref, idx_t lambda_index, idx_t depth) {
	column_t column_index;
	if (!TryGetBindingIndex(colref.GetColumnName(), column_index)) {
		throw InternalException("Column %s not found in bindings", colref.GetColumnName());
	}
	ColumnBinding binding(index, column_index);
	return BindResult(
	    make_unique<BoundLambdaRefExpression>(colref.GetName(), types[column_index], binding, lambda_index, depth));
}

unique_ptr<ParsedExpression> DummyBinding::ParamToArg(ColumnRefExpression &colref) {
	column_t column_index;
	if (!TryGetBindingIndex(colref.GetColumnName(), column_index)) {
		throw InternalException("Column %s not found in macro", colref.GetColumnName());
	}
	auto arg = (*arguments)[column_index]->Copy();
	arg->alias = colref.alias;
	return arg;
}

} // namespace duckdb






namespace duckdb {

void TableFilterSet::PushFilter(idx_t column_index, unique_ptr<TableFilter> filter) {
	auto entry = filters.find(column_index);
	if (entry == filters.end()) {
		// no filter yet: push the filter directly
		filters[column_index] = std::move(filter);
	} else {
		// there is already a filter: AND it together
		if (entry->second->filter_type == TableFilterType::CONJUNCTION_AND) {
			auto &and_filter = (ConjunctionAndFilter &)*entry->second;
			and_filter.child_filters.push_back(std::move(filter));
		} else {
			auto and_filter = make_unique<ConjunctionAndFilter>();
			and_filter->child_filters.push_back(std::move(entry->second));
			and_filter->child_filters.push_back(std::move(filter));
			filters[column_index] = std::move(and_filter);
		}
	}
}

//! Serializes a LogicalType to a stand-alone binary blob
void TableFilterSet::Serialize(Serializer &serializer) const {
	serializer.Write<idx_t>(filters.size());
	for (auto &entry : filters) {
		serializer.Write<idx_t>(entry.first);
		entry.second->Serialize(serializer);
	}
}

//! Deserializes a blob back into an LogicalType
unique_ptr<TableFilterSet> TableFilterSet::Deserialize(Deserializer &source) {
	auto len = source.Read<idx_t>();
	auto res = make_unique<TableFilterSet>();
	for (idx_t i = 0; i < len; i++) {
		auto key = source.Read<idx_t>();
		auto value = TableFilter::Deserialize(source);
		res->filters[key] = std::move(value);
	}
	return res;
}

//! Serializes a LogicalType to a stand-alone binary blob
void TableFilter::Serialize(Serializer &serializer) const {
	FieldWriter writer(serializer);
	writer.WriteField<TableFilterType>(filter_type);
	Serialize(writer);
	writer.Finalize();
}

//! Deserializes a blob back into an LogicalType
unique_ptr<TableFilter> TableFilter::Deserialize(Deserializer &source) {
	unique_ptr<TableFilter> result;

	FieldReader reader(source);
	auto filter_type = reader.ReadRequired<TableFilterType>();
	switch (filter_type) {
	case TableFilterType::CONSTANT_COMPARISON:
		result = ConstantFilter::Deserialize(reader);
		break;
	case TableFilterType::CONJUNCTION_AND:
		result = ConjunctionAndFilter::Deserialize(reader);
		break;
	case TableFilterType::CONJUNCTION_OR:
		result = ConjunctionOrFilter::Deserialize(reader);
		break;
	case TableFilterType::IS_NOT_NULL:
		result = IsNotNullFilter::Deserialize(reader);
		break;
	case TableFilterType::IS_NULL:
		result = IsNullFilter::Deserialize(reader);
		break;
	default:
		throw NotImplementedException("Unsupported table filter type for deserialization");
	}
	reader.Finalize();
	return result;
}

} // namespace duckdb




namespace duckdb {

ArenaChunk::ArenaChunk(Allocator &allocator, idx_t size) : current_position(0), maximum_size(size), prev(nullptr) {
	D_ASSERT(size > 0);
	data = allocator.Allocate(size);
}
ArenaChunk::~ArenaChunk() {
	if (next) {
		auto current_next = std::move(next);
		while (current_next) {
			current_next = std::move(current_next->next);
		}
	}
}

ArenaAllocator::ArenaAllocator(Allocator &allocator, idx_t initial_capacity) : allocator(allocator) {
	head = nullptr;
	tail = nullptr;
	current_capacity = initial_capacity;
}

ArenaAllocator::~ArenaAllocator() {
}

data_ptr_t ArenaAllocator::Allocate(idx_t len) {
	D_ASSERT(!head || head->current_position <= head->maximum_size);
	if (!head || head->current_position + len > head->maximum_size) {
		do {
			current_capacity *= 2;
		} while (current_capacity < len);
		auto new_chunk = make_unique<ArenaChunk>(allocator, current_capacity);
		if (head) {
			head->prev = new_chunk.get();
			new_chunk->next = std::move(head);
		} else {
			tail = new_chunk.get();
		}
		head = std::move(new_chunk);
	}
	D_ASSERT(head->current_position + len <= head->maximum_size);
	auto result = head->data.get() + head->current_position;
	head->current_position += len;
	return result;
}

data_ptr_t ArenaAllocator::Reallocate(data_ptr_t pointer, idx_t old_size, idx_t size) {
	D_ASSERT(head);
	if (old_size == size) {
		// nothing to do
		return pointer;
	}

	auto head_ptr = head->data.get() + head->current_position;
	int64_t diff = size - old_size;
	if (pointer == head_ptr && (size < old_size || head->current_position + diff <= head->maximum_size)) {
		// passed pointer is the head pointer, and the diff fits on the current chunk
		head->current_position += diff;
		return pointer;
	} else {
		// allocate new memory
		auto result = Allocate(size);
		memcpy(result, pointer, old_size);
		return result;
	}
}

data_ptr_t ArenaAllocator::AllocateAligned(idx_t size) {
	return Allocate(AlignValue<idx_t>(size));
}

data_ptr_t ArenaAllocator::ReallocateAligned(data_ptr_t pointer, idx_t old_size, idx_t size) {
	return Reallocate(pointer, old_size, AlignValue<idx_t>(size));
}

void ArenaAllocator::Reset() {

	if (head) {
		// destroy all chunks except the current one
		if (head->next) {
			auto current_next = std::move(head->next);
			while (current_next) {
				current_next = std::move(current_next->next);
			}
		}
		tail = head.get();

		// reset the head
		head->current_position = 0;
		head->prev = nullptr;
	}
}

void ArenaAllocator::Destroy() {
	head = nullptr;
	tail = nullptr;
	current_capacity = ARENA_ALLOCATOR_INITIAL_CAPACITY;
}

void ArenaAllocator::Move(ArenaAllocator &other) {
	D_ASSERT(!other.head);
	other.tail = tail;
	other.head = std::move(head);
	other.current_capacity = current_capacity;
	Destroy();
}

ArenaChunk *ArenaAllocator::GetHead() {
	return head.get();
}

ArenaChunk *ArenaAllocator::GetTail() {
	return tail;
}

bool ArenaAllocator::IsEmpty() {
	return head == nullptr;
}

} // namespace duckdb



namespace duckdb {

Block::Block(Allocator &allocator, block_id_t id)
    : FileBuffer(allocator, FileBufferType::BLOCK, Storage::BLOCK_SIZE), id(id) {
}

Block::Block(Allocator &allocator, block_id_t id, uint32_t internal_size)
    : FileBuffer(allocator, FileBufferType::BLOCK, internal_size), id(id) {
	D_ASSERT((AllocSize() & (Storage::SECTOR_SIZE - 1)) == 0);
}

Block::Block(FileBuffer &source, block_id_t id) : FileBuffer(source, FileBufferType::BLOCK), id(id) {
	D_ASSERT((AllocSize() & (Storage::SECTOR_SIZE - 1)) == 0);
}

} // namespace duckdb



namespace duckdb {

BufferHandle::BufferHandle() : handle(nullptr), node(nullptr) {
}

BufferHandle::BufferHandle(shared_ptr<BlockHandle> handle, FileBuffer *node) : handle(std::move(handle)), node(node) {
}

BufferHandle::BufferHandle(BufferHandle &&other) noexcept {
	std::swap(node, other.node);
	std::swap(handle, other.handle);
}

BufferHandle &BufferHandle::operator=(BufferHandle &&other) noexcept {
	std::swap(node, other.node);
	std::swap(handle, other.handle);
	return *this;
}

BufferHandle::~BufferHandle() {
	Destroy();
}

bool BufferHandle::IsValid() const {
	return node != nullptr;
}

data_ptr_t BufferHandle::Ptr() const {
	D_ASSERT(IsValid());
	return node->buffer;
}

data_ptr_t BufferHandle::Ptr() {
	D_ASSERT(IsValid());
	return node->buffer;
}

void BufferHandle::Destroy() {
	if (!handle || !IsValid()) {
		return;
	}
	handle->block_manager.buffer_manager.Unpin(handle);
	handle.reset();
	node = nullptr;
}

FileBuffer &BufferHandle::GetFileBuffer() {
	D_ASSERT(node);
	return *node;
}

} // namespace duckdb










namespace duckdb {

BufferPoolReservation::BufferPoolReservation(BufferPoolReservation &&src) noexcept {
	size = src.size;
	src.size = 0;
}

BufferPoolReservation &BufferPoolReservation::operator=(BufferPoolReservation &&src) noexcept {
	size = src.size;
	src.size = 0;
	return *this;
}

BufferPoolReservation::~BufferPoolReservation() {
	D_ASSERT(size == 0);
}

void BufferPoolReservation::Resize(atomic<idx_t> &counter, idx_t new_size) {
	int64_t delta = (int64_t)new_size - size;
	D_ASSERT(delta > 0 || (int64_t)counter >= -delta);
	counter += delta;
	size = new_size;
}

void BufferPoolReservation::Merge(BufferPoolReservation &&src) {
	size += src.size;
	src.size = 0;
}

struct BufferAllocatorData : PrivateAllocatorData {
	explicit BufferAllocatorData(BufferManager &manager) : manager(manager) {
	}

	BufferManager &manager;
};

BlockHandle::BlockHandle(BlockManager &block_manager, block_id_t block_id_p)
    : block_manager(block_manager), readers(0), block_id(block_id_p), buffer(nullptr), eviction_timestamp(0),
      can_destroy(false), unswizzled(nullptr) {
	eviction_timestamp = 0;
	state = BlockState::BLOCK_UNLOADED;
	memory_usage = Storage::BLOCK_ALLOC_SIZE;
}

BlockHandle::BlockHandle(BlockManager &block_manager, block_id_t block_id_p, unique_ptr<FileBuffer> buffer_p,
                         bool can_destroy_p, idx_t block_size, BufferPoolReservation &&reservation)
    : block_manager(block_manager), readers(0), block_id(block_id_p), eviction_timestamp(0), can_destroy(can_destroy_p),
      unswizzled(nullptr) {
	buffer = std::move(buffer_p);
	state = BlockState::BLOCK_LOADED;
	memory_usage = block_size;
	memory_charge = std::move(reservation);
}

BlockHandle::~BlockHandle() {
	// being destroyed, so any unswizzled pointers are just binary junk now.
	unswizzled = nullptr;
	auto &buffer_manager = block_manager.buffer_manager;
	// no references remain to this block: erase
	if (buffer && state == BlockState::BLOCK_LOADED) {
		D_ASSERT(memory_charge.size > 0);
		// the block is still loaded in memory: erase it
		buffer.reset();
		memory_charge.Resize(buffer_manager.current_memory, 0);
	} else {
		D_ASSERT(memory_charge.size == 0);
	}
	buffer_manager.PurgeQueue();
	block_manager.UnregisterBlock(block_id, can_destroy);
}

unique_ptr<Block> AllocateBlock(BlockManager &block_manager, unique_ptr<FileBuffer> reusable_buffer,
                                block_id_t block_id) {
	if (reusable_buffer) {
		// re-usable buffer: re-use it
		if (reusable_buffer->type == FileBufferType::BLOCK) {
			// we can reuse the buffer entirely
			auto &block = (Block &)*reusable_buffer;
			block.id = block_id;
			return unique_ptr_cast<FileBuffer, Block>(std::move(reusable_buffer));
		}
		auto block = block_manager.CreateBlock(block_id, reusable_buffer.get());
		reusable_buffer.reset();
		return block;
	} else {
		// no re-usable buffer: allocate a new block
		return block_manager.CreateBlock(block_id, nullptr);
	}
}

idx_t GetAllocSize(idx_t size) {
	return AlignValue<idx_t, Storage::SECTOR_SIZE>(size + Storage::BLOCK_HEADER_SIZE);
}

unique_ptr<FileBuffer> BufferManager::ConstructManagedBuffer(idx_t size, unique_ptr<FileBuffer> &&source,
                                                             FileBufferType type) {
	if (source) {
		auto tmp = std::move(source);
		D_ASSERT(tmp->AllocSize() == BufferManager::GetAllocSize(size));
		return make_unique<FileBuffer>(*tmp, type);
	} else {
		// no re-usable buffer: allocate a new buffer
		return make_unique<FileBuffer>(Allocator::Get(db), type, size);
	}
}

BufferHandle BlockHandle::Load(shared_ptr<BlockHandle> &handle, unique_ptr<FileBuffer> reusable_buffer) {
	if (handle->state == BlockState::BLOCK_LOADED) {
		// already loaded
		D_ASSERT(handle->buffer);
		return BufferHandle(handle, handle->buffer.get());
	}

	auto &block_manager = handle->block_manager;
	if (handle->block_id < MAXIMUM_BLOCK) {
		auto block = AllocateBlock(block_manager, std::move(reusable_buffer), handle->block_id);
		block_manager.Read(*block);
		handle->buffer = std::move(block);
	} else {
		if (handle->can_destroy) {
			return BufferHandle();
		} else {
			handle->buffer =
			    block_manager.buffer_manager.ReadTemporaryBuffer(handle->block_id, std::move(reusable_buffer));
		}
	}
	handle->state = BlockState::BLOCK_LOADED;
	return BufferHandle(handle, handle->buffer.get());
}

unique_ptr<FileBuffer> BlockHandle::UnloadAndTakeBlock() {
	if (state == BlockState::BLOCK_UNLOADED) {
		// already unloaded: nothing to do
		return nullptr;
	}
	D_ASSERT(!unswizzled);
	D_ASSERT(CanUnload());

	if (block_id >= MAXIMUM_BLOCK && !can_destroy) {
		// temporary block that cannot be destroyed: write to temporary file
		block_manager.buffer_manager.WriteTemporaryBuffer(block_id, *buffer);
	}
	memory_charge.Resize(block_manager.buffer_manager.current_memory, 0);
	state = BlockState::BLOCK_UNLOADED;
	return std::move(buffer);
}

void BlockHandle::Unload() {
	auto block = UnloadAndTakeBlock();
	block.reset();
}

bool BlockHandle::CanUnload() {
	if (state == BlockState::BLOCK_UNLOADED) {
		// already unloaded
		return false;
	}
	if (readers > 0) {
		// there are active readers
		return false;
	}
	if (block_id >= MAXIMUM_BLOCK && !can_destroy && block_manager.buffer_manager.temp_directory.empty()) {
		// in order to unload this block we need to write it to a temporary buffer
		// however, no temporary directory is specified!
		// hence we cannot unload the block
		return false;
	}
	return true;
}

struct BufferEvictionNode {
	BufferEvictionNode() {
	}
	BufferEvictionNode(weak_ptr<BlockHandle> handle_p, idx_t timestamp_p)
	    : handle(std::move(handle_p)), timestamp(timestamp_p) {
		D_ASSERT(!handle.expired());
	}

	weak_ptr<BlockHandle> handle;
	idx_t timestamp;

	bool CanUnload(BlockHandle &handle_p) {
		if (timestamp != handle_p.eviction_timestamp) {
			// handle was used in between
			return false;
		}
		return handle_p.CanUnload();
	}

	shared_ptr<BlockHandle> TryGetBlockHandle() {
		auto handle_p = handle.lock();
		if (!handle_p) {
			// BlockHandle has been destroyed
			return nullptr;
		}
		if (!CanUnload(*handle_p)) {
			// handle was used in between
			return nullptr;
		}
		// this is the latest node in the queue with this handle
		return handle_p;
	}
};

typedef duckdb_moodycamel::ConcurrentQueue<BufferEvictionNode> eviction_queue_t;

struct EvictionQueue {
	eviction_queue_t q;
};

class TemporaryFileManager;

class TemporaryDirectoryHandle {
public:
	TemporaryDirectoryHandle(DatabaseInstance &db, string path_p);
	~TemporaryDirectoryHandle();

	TemporaryFileManager &GetTempFile();

private:
	DatabaseInstance &db;
	string temp_directory;
	unique_ptr<TemporaryFileManager> temp_file;
};

void BufferManager::SetTemporaryDirectory(string new_dir) {
	if (temp_directory_handle) {
		throw NotImplementedException("Cannot switch temporary directory after the current one has been used");
	}
	this->temp_directory = std::move(new_dir);
}

BufferManager::BufferManager(DatabaseInstance &db, string tmp, idx_t maximum_memory)
    : db(db), current_memory(0), maximum_memory(maximum_memory), temp_directory(std::move(tmp)),
      queue(make_unique<EvictionQueue>()), temporary_id(MAXIMUM_BLOCK), queue_insertions(0),
      buffer_allocator(BufferAllocatorAllocate, BufferAllocatorFree, BufferAllocatorRealloc,
                       make_unique<BufferAllocatorData>(*this)) {
	temp_block_manager = make_unique<InMemoryBlockManager>(*this);
}

BufferManager::~BufferManager() {
}

shared_ptr<BlockHandle> BlockManager::RegisterBlock(block_id_t block_id, bool is_meta_block) {
	lock_guard<mutex> lock(blocks_lock);
	// check if the block already exists
	auto entry = blocks.find(block_id);
	if (entry != blocks.end()) {
		// already exists: check if it hasn't expired yet
		auto existing_ptr = entry->second.lock();
		if (existing_ptr) {
			//! it hasn't! return it
			return existing_ptr;
		}
	}
	// create a new block pointer for this block
	auto result = make_shared<BlockHandle>(*this, block_id);
	// for meta block, cache the handle in meta_blocks
	if (is_meta_block) {
		meta_blocks[block_id] = result;
	}
	// register the block pointer in the set of blocks as a weak pointer
	blocks[block_id] = weak_ptr<BlockHandle>(result);
	return result;
}

void BlockManager::ClearMetaBlockHandles() {
	meta_blocks.clear();
}

shared_ptr<BlockHandle> BlockManager::ConvertToPersistent(block_id_t block_id, shared_ptr<BlockHandle> old_block) {

	// pin the old block to ensure we have it loaded in memory
	auto old_handle = buffer_manager.Pin(old_block);
	D_ASSERT(old_block->state == BlockState::BLOCK_LOADED);
	D_ASSERT(old_block->buffer);

	// Temp buffers can be larger than the storage block size. But persistent buffers
	// cannot.
	D_ASSERT(old_block->buffer->AllocSize() <= Storage::BLOCK_ALLOC_SIZE);

	// register a block with the new block id
	auto new_block = RegisterBlock(block_id);
	D_ASSERT(new_block->state == BlockState::BLOCK_UNLOADED);
	D_ASSERT(new_block->readers == 0);

	// move the data from the old block into data for the new block
	new_block->state = BlockState::BLOCK_LOADED;
	new_block->buffer = CreateBlock(block_id, old_block->buffer.get());
	new_block->memory_usage = old_block->memory_usage;
	new_block->memory_charge = std::move(old_block->memory_charge);

	// clear the old buffer and unload it
	old_block->buffer.reset();
	old_block->state = BlockState::BLOCK_UNLOADED;
	old_block->memory_usage = 0;
	old_handle.Destroy();
	old_block.reset();

	// persist the new block to disk
	Write(*new_block->buffer, block_id);

	buffer_manager.AddToEvictionQueue(new_block);

	return new_block;
}

template <typename... ARGS>
TempBufferPoolReservation BufferManager::EvictBlocksOrThrow(idx_t memory_delta, idx_t limit,
                                                            unique_ptr<FileBuffer> *buffer, ARGS... args) {
	auto r = EvictBlocks(memory_delta, limit, buffer);
	if (!r.success) {
		throw OutOfMemoryException(args..., InMemoryWarning());
	}
	return std::move(r.reservation);
}

shared_ptr<BlockHandle> BufferManager::RegisterSmallMemory(idx_t block_size) {
	D_ASSERT(block_size < Storage::BLOCK_SIZE);
	auto res = EvictBlocksOrThrow(block_size, maximum_memory, nullptr,
	                              "could not allocate block of %lld bytes (%lld/%lld used) %s", block_size,
	                              GetUsedMemory(), GetMaxMemory());

	auto buffer = ConstructManagedBuffer(block_size, nullptr, FileBufferType::TINY_BUFFER);

	// create a new block pointer for this block
	return make_shared<BlockHandle>(*temp_block_manager, ++temporary_id, std::move(buffer), false, block_size,
	                                std::move(res));
}

shared_ptr<BlockHandle> BufferManager::RegisterMemory(idx_t block_size, bool can_destroy) {
	D_ASSERT(block_size >= Storage::BLOCK_SIZE);
	auto alloc_size = GetAllocSize(block_size);
	// first evict blocks until we have enough memory to store this buffer
	unique_ptr<FileBuffer> reusable_buffer;
	auto res = EvictBlocksOrThrow(alloc_size, maximum_memory, &reusable_buffer,
	                              "could not allocate block of %lld bytes (%lld/%lld used) %s", alloc_size,
	                              GetUsedMemory(), GetMaxMemory());

	auto buffer = ConstructManagedBuffer(block_size, std::move(reusable_buffer));

	// create a new block pointer for this block
	return make_shared<BlockHandle>(*temp_block_manager, ++temporary_id, std::move(buffer), can_destroy, alloc_size,
	                                std::move(res));
}

BufferHandle BufferManager::Allocate(idx_t block_size, bool can_destroy, shared_ptr<BlockHandle> *block) {
	shared_ptr<BlockHandle> local_block;
	auto block_ptr = block ? block : &local_block;
	*block_ptr = RegisterMemory(block_size, can_destroy);
	return Pin(*block_ptr);
}

void BufferManager::ReAllocate(shared_ptr<BlockHandle> &handle, idx_t block_size) {
	D_ASSERT(block_size >= Storage::BLOCK_SIZE);
	lock_guard<mutex> lock(handle->lock);
	D_ASSERT(handle->state == BlockState::BLOCK_LOADED);
	D_ASSERT(handle->memory_usage == handle->buffer->AllocSize());
	D_ASSERT(handle->memory_usage == handle->memory_charge.size);

	auto req = handle->buffer->CalculateMemory(block_size);
	int64_t memory_delta = (int64_t)req.alloc_size - handle->memory_usage;

	if (memory_delta == 0) {
		return;
	} else if (memory_delta > 0) {
		// evict blocks until we have space to resize this block
		auto reservation =
		    EvictBlocksOrThrow(memory_delta, maximum_memory, nullptr, "failed to resize block from %lld to %lld%s",
		                       handle->memory_usage, req.alloc_size);
		// EvictBlocks decrements 'current_memory' for us.
		handle->memory_charge.Merge(std::move(reservation));
	} else {
		// no need to evict blocks, but we do need to decrement 'current_memory'.
		handle->memory_charge.Resize(current_memory, req.alloc_size);
	}

	// resize and adjust current memory
	handle->buffer->Resize(block_size);
	handle->memory_usage += memory_delta;
	D_ASSERT(handle->memory_usage == handle->buffer->AllocSize());
}

BufferHandle BufferManager::Pin(shared_ptr<BlockHandle> &handle) {
	idx_t required_memory;
	{
		// lock the block
		lock_guard<mutex> lock(handle->lock);
		// check if the block is already loaded
		if (handle->state == BlockState::BLOCK_LOADED) {
			// the block is loaded, increment the reader count and return a pointer to the handle
			handle->readers++;
			return handle->Load(handle);
		}
		required_memory = handle->memory_usage;
	}
	// evict blocks until we have space for the current block
	unique_ptr<FileBuffer> reusable_buffer;
	auto reservation = EvictBlocksOrThrow(required_memory, maximum_memory, &reusable_buffer,
	                                      "failed to pin block of size %lld%s", required_memory);
	// lock the handle again and repeat the check (in case anybody loaded in the mean time)
	lock_guard<mutex> lock(handle->lock);
	// check if the block is already loaded
	if (handle->state == BlockState::BLOCK_LOADED) {
		// the block is loaded, increment the reader count and return a pointer to the handle
		handle->readers++;
		reservation.Resize(current_memory, 0);
		return handle->Load(handle);
	}
	// now we can actually load the current block
	D_ASSERT(handle->readers == 0);
	handle->readers = 1;
	auto buf = handle->Load(handle, std::move(reusable_buffer));
	handle->memory_charge = std::move(reservation);
	// In the case of a variable sized block, the buffer may be smaller than a full block.
	int64_t delta = handle->buffer->AllocSize() - handle->memory_usage;
	if (delta) {
		D_ASSERT(delta < 0);
		handle->memory_usage += delta;
		handle->memory_charge.Resize(current_memory, handle->memory_usage);
	}
	D_ASSERT(handle->memory_usage == handle->buffer->AllocSize());
	return buf;
}

void BufferManager::AddToEvictionQueue(shared_ptr<BlockHandle> &handle) {
	constexpr int INSERT_INTERVAL = 1024;

	D_ASSERT(handle->readers == 0);
	handle->eviction_timestamp++;
	// After each 1024 insertions, run through the queue and purge.
	if ((++queue_insertions % INSERT_INTERVAL) == 0) {
		PurgeQueue();
	}
	queue->q.enqueue(BufferEvictionNode(weak_ptr<BlockHandle>(handle), handle->eviction_timestamp));
}

void BufferManager::VerifyZeroReaders(shared_ptr<BlockHandle> &handle) {
#ifdef DUCKDB_DEBUG_DESTROY_BLOCKS
	auto replacement_buffer = make_unique<FileBuffer>(Allocator::Get(db), handle->buffer->type,
	                                                  handle->memory_usage - Storage::BLOCK_HEADER_SIZE);
	memcpy(replacement_buffer->buffer, handle->buffer->buffer, handle->buffer->size);
	memset(handle->buffer->buffer, 165, handle->buffer->size); // 165 is default memory in debug mode
	handle->buffer = std::move(replacement_buffer);
#endif
}

void BufferManager::Unpin(shared_ptr<BlockHandle> &handle) {
	lock_guard<mutex> lock(handle->lock);
	if (!handle->buffer || handle->buffer->type == FileBufferType::TINY_BUFFER) {
		return;
	}
	D_ASSERT(handle->readers > 0);
	handle->readers--;
	if (handle->readers == 0) {
		VerifyZeroReaders(handle);
		AddToEvictionQueue(handle);
	}
}

BufferManager::EvictionResult BufferManager::EvictBlocks(idx_t extra_memory, idx_t memory_limit,
                                                         unique_ptr<FileBuffer> *buffer) {
	BufferEvictionNode node;
	TempBufferPoolReservation r(current_memory, extra_memory);
	while (current_memory > memory_limit) {
		// get a block to unpin from the queue
		if (!queue->q.try_dequeue(node)) {
			// Failed to reserve. Adjust size of temp reservation to 0.
			r.Resize(current_memory, 0);
			return {false, std::move(r)};
		}
		// get a reference to the underlying block pointer
		auto handle = node.TryGetBlockHandle();
		if (!handle) {
			continue;
		}
		// we might be able to free this block: grab the mutex and check if we can free it
		lock_guard<mutex> lock(handle->lock);
		if (!node.CanUnload(*handle)) {
			// something changed in the mean-time, bail out
			continue;
		}
		// hooray, we can unload the block
		if (buffer && handle->buffer->AllocSize() == extra_memory) {
			// we can actually re-use the memory directly!
			*buffer = handle->UnloadAndTakeBlock();
			return {true, std::move(r)};
		} else {
			// release the memory and mark the block as unloaded
			handle->Unload();
		}
	}
	return {true, std::move(r)};
}

void BufferManager::PurgeQueue() {
	BufferEvictionNode node;
	while (true) {
		if (!queue->q.try_dequeue(node)) {
			break;
		}
		auto handle = node.TryGetBlockHandle();
		if (!handle) {
			continue;
		} else {
			queue->q.enqueue(std::move(node));
			break;
		}
	}
}

void BlockManager::UnregisterBlock(block_id_t block_id, bool can_destroy) {
	if (block_id >= MAXIMUM_BLOCK) {
		// in-memory buffer: destroy the buffer
		if (!can_destroy) {
			// buffer could have been offloaded to disk: remove the file
			buffer_manager.DeleteTemporaryFile(block_id);
		}
	} else {
		lock_guard<mutex> lock(blocks_lock);
		// on-disk block: erase from list of blocks in manager
		blocks.erase(block_id);
	}
}

void BufferManager::SetLimit(idx_t limit) {
	lock_guard<mutex> l_lock(limit_lock);
	// try to evict until the limit is reached
	if (!EvictBlocks(0, limit).success) {
		throw OutOfMemoryException(
		    "Failed to change memory limit to %lld: could not free up enough memory for the new limit%s", limit,
		    InMemoryWarning());
	}
	idx_t old_limit = maximum_memory;
	// set the global maximum memory to the new limit if successful
	maximum_memory = limit;
	// evict again
	if (!EvictBlocks(0, limit).success) {
		// failed: go back to old limit
		maximum_memory = old_limit;
		throw OutOfMemoryException(
		    "Failed to change memory limit to %lld: could not free up enough memory for the new limit%s", limit,
		    InMemoryWarning());
	}
}

void BufferManager::IncreaseUsedMemory(idx_t size) {
	if (current_memory + size > maximum_memory) {
		throw OutOfMemoryException("Failed to allocate data of size %lld%s", size, InMemoryWarning());
	}
	current_memory += size;
}

void BufferManager::DecreaseUsedMemory(idx_t size) {
	D_ASSERT(current_memory >= size);
	current_memory -= size;
}

//===--------------------------------------------------------------------===//
// Temporary File Management
//===--------------------------------------------------------------------===//
unique_ptr<FileBuffer> ReadTemporaryBufferInternal(BufferManager &buffer_manager, FileHandle &handle, idx_t position,
                                                   idx_t size, block_id_t id, unique_ptr<FileBuffer> reusable_buffer) {
	auto buffer = buffer_manager.ConstructManagedBuffer(size, std::move(reusable_buffer));
	buffer->Read(handle, position);
	return buffer;
}

struct TemporaryFileIndex {
	explicit TemporaryFileIndex(idx_t file_index = DConstants::INVALID_INDEX,
	                            idx_t block_index = DConstants::INVALID_INDEX)
	    : file_index(file_index), block_index(block_index) {
	}

	idx_t file_index;
	idx_t block_index;

public:
	bool IsValid() {
		return block_index != DConstants::INVALID_INDEX;
	}
};

struct BlockIndexManager {
	BlockIndexManager() : max_index(0) {
	}

public:
	//! Obtains a new block index from the index manager
	idx_t GetNewBlockIndex() {
		auto index = GetNewBlockIndexInternal();
		indexes_in_use.insert(index);
		return index;
	}

	//! Removes an index from the block manager
	//! Returns true if the max_index has been altered
	bool RemoveIndex(idx_t index) {
		// remove this block from the set of blocks
		indexes_in_use.erase(index);
		free_indexes.insert(index);
		// check if we can truncate the file

		// get the max_index in use right now
		auto max_index_in_use = indexes_in_use.empty() ? 0 : *indexes_in_use.rbegin();
		if (max_index_in_use < max_index) {
			// max index in use is lower than the max_index
			// reduce the max_index
			max_index = max_index_in_use + 1;
			// we can remove any free_indexes that are larger than the current max_index
			while (!free_indexes.empty()) {
				auto max_entry = *free_indexes.rbegin();
				if (max_entry < max_index) {
					break;
				}
				free_indexes.erase(max_entry);
			}
			return true;
		}
		return false;
	}

	idx_t GetMaxIndex() {
		return max_index;
	}

	bool HasFreeBlocks() {
		return !free_indexes.empty();
	}

private:
	idx_t GetNewBlockIndexInternal() {
		if (free_indexes.empty()) {
			return max_index++;
		}
		auto entry = free_indexes.begin();
		auto index = *entry;
		free_indexes.erase(entry);
		return index;
	}

	idx_t max_index;
	set<idx_t> free_indexes;
	set<idx_t> indexes_in_use;
};

class TemporaryFileHandle {
	constexpr static idx_t MAX_ALLOWED_INDEX = 4000;

public:
	TemporaryFileHandle(DatabaseInstance &db, const string &temp_directory, idx_t index)
	    : db(db), file_index(index), path(FileSystem::GetFileSystem(db).JoinPath(
	                                     temp_directory, "duckdb_temp_storage-" + to_string(index) + ".tmp")) {
	}

public:
	struct TemporaryFileLock {
		explicit TemporaryFileLock(mutex &mutex) : lock(mutex) {
		}

		lock_guard<mutex> lock;
	};

public:
	TemporaryFileIndex TryGetBlockIndex() {
		TemporaryFileLock lock(file_lock);
		if (index_manager.GetMaxIndex() >= MAX_ALLOWED_INDEX && index_manager.HasFreeBlocks()) {
			// file is at capacity
			return TemporaryFileIndex();
		}
		// open the file handle if it does not yet exist
		CreateFileIfNotExists(lock);
		// fetch a new block index to write to
		auto block_index = index_manager.GetNewBlockIndex();
		return TemporaryFileIndex(file_index, block_index);
	}

	void WriteTemporaryFile(FileBuffer &buffer, TemporaryFileIndex index) {
		D_ASSERT(buffer.size == Storage::BLOCK_SIZE);
		buffer.Write(*handle, GetPositionInFile(index.block_index));
	}

	unique_ptr<FileBuffer> ReadTemporaryBuffer(block_id_t id, idx_t block_index,
	                                           unique_ptr<FileBuffer> reusable_buffer) {
		auto buffer =
		    ReadTemporaryBufferInternal(BufferManager::GetBufferManager(db), *handle, GetPositionInFile(block_index),
		                                Storage::BLOCK_SIZE, id, std::move(reusable_buffer));
		{
			// remove the block (and potentially truncate the temp file)
			TemporaryFileLock lock(file_lock);
			D_ASSERT(handle);
			RemoveTempBlockIndex(lock, block_index);
		}
		return buffer;
	}

	bool DeleteIfEmpty() {
		TemporaryFileLock lock(file_lock);
		if (index_manager.GetMaxIndex() > 0) {
			// there are still blocks in this file
			return false;
		}
		// the file is empty: delete it
		handle.reset();
		auto &fs = FileSystem::GetFileSystem(db);
		fs.RemoveFile(path);
		return true;
	}

private:
	void CreateFileIfNotExists(TemporaryFileLock &) {
		if (handle) {
			return;
		}
		auto &fs = FileSystem::GetFileSystem(db);
		handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_READ | FileFlags::FILE_FLAGS_WRITE |
		                               FileFlags::FILE_FLAGS_FILE_CREATE);
	}

	void RemoveTempBlockIndex(TemporaryFileLock &, idx_t index) {
		// remove the block index from the index manager
		if (index_manager.RemoveIndex(index)) {
			// the max_index that is currently in use has decreased
			// as a result we can truncate the file
#ifndef WIN32 // this ended up causing issues when sorting
			auto max_index = index_manager.GetMaxIndex();
			auto &fs = FileSystem::GetFileSystem(db);
			fs.Truncate(*handle, GetPositionInFile(max_index + 1));
#endif
		}
	}

	idx_t GetPositionInFile(idx_t index) {
		return index * Storage::BLOCK_ALLOC_SIZE;
	}

private:
	DatabaseInstance &db;
	unique_ptr<FileHandle> handle;
	idx_t file_index;
	string path;
	mutex file_lock;
	BlockIndexManager index_manager;
};

class TemporaryFileManager {
public:
	TemporaryFileManager(DatabaseInstance &db, const string &temp_directory_p)
	    : db(db), temp_directory(temp_directory_p) {
	}

public:
	struct TemporaryManagerLock {
		explicit TemporaryManagerLock(mutex &mutex) : lock(mutex) {
		}

		lock_guard<mutex> lock;
	};

	void WriteTemporaryBuffer(block_id_t block_id, FileBuffer &buffer) {
		D_ASSERT(buffer.size == Storage::BLOCK_SIZE);
		TemporaryFileIndex index;
		TemporaryFileHandle *handle = nullptr;

		{
			TemporaryManagerLock lock(manager_lock);
			// first check if we can write to an open existing file
			for (auto &entry : files) {
				auto &temp_file = entry.second;
				index = temp_file->TryGetBlockIndex();
				if (index.IsValid()) {
					handle = entry.second.get();
					break;
				}
			}
			if (!handle) {
				// no existing handle to write to; we need to create & open a new file
				auto new_file_index = index_manager.GetNewBlockIndex();
				auto new_file = make_unique<TemporaryFileHandle>(db, temp_directory, new_file_index);
				handle = new_file.get();
				files[new_file_index] = std::move(new_file);

				index = handle->TryGetBlockIndex();
			}
			D_ASSERT(used_blocks.find(block_id) == used_blocks.end());
			used_blocks[block_id] = index;
		}
		D_ASSERT(handle);
		D_ASSERT(index.IsValid());
		handle->WriteTemporaryFile(buffer, index);
	}

	bool HasTemporaryBuffer(block_id_t block_id) {
		lock_guard<mutex> lock(manager_lock);
		return used_blocks.find(block_id) != used_blocks.end();
	}

	unique_ptr<FileBuffer> ReadTemporaryBuffer(block_id_t id, unique_ptr<FileBuffer> reusable_buffer) {
		TemporaryFileIndex index;
		TemporaryFileHandle *handle;
		{
			TemporaryManagerLock lock(manager_lock);
			index = GetTempBlockIndex(lock, id);
			handle = GetFileHandle(lock, index.file_index);
		}
		auto buffer = handle->ReadTemporaryBuffer(id, index.block_index, std::move(reusable_buffer));
		{
			// remove the block (and potentially erase the temp file)
			TemporaryManagerLock lock(manager_lock);
			EraseUsedBlock(lock, id, handle, index.file_index);
		}
		return buffer;
	}

	void DeleteTemporaryBuffer(block_id_t id) {
		TemporaryManagerLock lock(manager_lock);
		auto index = GetTempBlockIndex(lock, id);
		auto handle = GetFileHandle(lock, index.file_index);
		EraseUsedBlock(lock, id, handle, index.file_index);
	}

private:
	void EraseUsedBlock(TemporaryManagerLock &lock, block_id_t id, TemporaryFileHandle *handle, idx_t file_index) {
		used_blocks.erase(id);
		if (handle->DeleteIfEmpty()) {
			EraseFileHandle(lock, file_index);
		}
	}

	TemporaryFileHandle *GetFileHandle(TemporaryManagerLock &, idx_t index) {
		return files[index].get();
	}

	TemporaryFileIndex GetTempBlockIndex(TemporaryManagerLock &, block_id_t id) {
		D_ASSERT(used_blocks.find(id) != used_blocks.end());
		return used_blocks[id];
	}

	void EraseFileHandle(TemporaryManagerLock &, idx_t file_index) {
		files.erase(file_index);
		index_manager.RemoveIndex(file_index);
	}

private:
	DatabaseInstance &db;
	mutex manager_lock;
	//! The temporary directory
	string temp_directory;
	//! The set of active temporary file handles
	unordered_map<idx_t, unique_ptr<TemporaryFileHandle>> files;
	//! map of block_id -> temporary file position
	unordered_map<block_id_t, TemporaryFileIndex> used_blocks;
	//! Manager of in-use temporary file indexes
	BlockIndexManager index_manager;
};

TemporaryDirectoryHandle::TemporaryDirectoryHandle(DatabaseInstance &db, string path_p)
    : db(db), temp_directory(std::move(path_p)), temp_file(make_unique<TemporaryFileManager>(db, temp_directory)) {
	auto &fs = FileSystem::GetFileSystem(db);
	if (!temp_directory.empty()) {
		fs.CreateDirectory(temp_directory);
	}
}
TemporaryDirectoryHandle::~TemporaryDirectoryHandle() {
	// first release any temporary files
	temp_file.reset();
	// then delete the temporary file directory
	auto &fs = FileSystem::GetFileSystem(db);
	if (!temp_directory.empty()) {
		fs.RemoveDirectory(temp_directory);
	}
}

TemporaryFileManager &TemporaryDirectoryHandle::GetTempFile() {
	return *temp_file;
}

string BufferManager::GetTemporaryPath(block_id_t id) {
	auto &fs = FileSystem::GetFileSystem(db);
	return fs.JoinPath(temp_directory, to_string(id) + ".block");
}

void BufferManager::RequireTemporaryDirectory() {
	if (temp_directory.empty()) {
		throw Exception(
		    "Out-of-memory: cannot write buffer because no temporary directory is specified!\nTo enable "
		    "temporary buffer eviction set a temporary directory using PRAGMA temp_directory='/path/to/tmp.tmp'");
	}
	lock_guard<mutex> temp_handle_guard(temp_handle_lock);
	if (!temp_directory_handle) {
		// temp directory has not been created yet: initialize it
		temp_directory_handle = make_unique<TemporaryDirectoryHandle>(db, temp_directory);
	}
}

void BufferManager::WriteTemporaryBuffer(block_id_t block_id, FileBuffer &buffer) {
	RequireTemporaryDirectory();
	if (buffer.size == Storage::BLOCK_SIZE) {
		temp_directory_handle->GetTempFile().WriteTemporaryBuffer(block_id, buffer);
		return;
	}
	// get the path to write to
	auto path = GetTemporaryPath(block_id);
	D_ASSERT(buffer.size > Storage::BLOCK_SIZE);
	// create the file and write the size followed by the buffer contents
	auto &fs = FileSystem::GetFileSystem(db);
	auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_FILE_CREATE);
	handle->Write(&buffer.size, sizeof(idx_t), 0);
	buffer.Write(*handle, sizeof(idx_t));
}

unique_ptr<FileBuffer> BufferManager::ReadTemporaryBuffer(block_id_t id, unique_ptr<FileBuffer> reusable_buffer) {
	D_ASSERT(!temp_directory.empty());
	D_ASSERT(temp_directory_handle.get());
	if (temp_directory_handle->GetTempFile().HasTemporaryBuffer(id)) {
		return temp_directory_handle->GetTempFile().ReadTemporaryBuffer(id, std::move(reusable_buffer));
	}
	idx_t block_size;
	// open the temporary file and read the size
	auto path = GetTemporaryPath(id);
	auto &fs = FileSystem::GetFileSystem(db);
	auto handle = fs.OpenFile(path, FileFlags::FILE_FLAGS_READ);
	handle->Read(&block_size, sizeof(idx_t), 0);

	// now allocate a buffer of this size and read the data into that buffer
	auto buffer =
	    ReadTemporaryBufferInternal(*this, *handle, sizeof(idx_t), block_size, id, std::move(reusable_buffer));

	handle.reset();
	DeleteTemporaryFile(id);
	return buffer;
}

void BufferManager::DeleteTemporaryFile(block_id_t id) {
	if (temp_directory.empty()) {
		// no temporary directory specified: nothing to delete
		return;
	}
	{
		lock_guard<mutex> temp_handle_guard(temp_handle_lock);
		if (!temp_directory_handle) {
			// temporary directory was not initialized yet: nothing to delete
			return;
		}
	}
	// check if we should delete the file from the shared pool of files, or from the general file system
	if (temp_directory_handle->GetTempFile().HasTemporaryBuffer(id)) {
		temp_directory_handle->GetTempFile().DeleteTemporaryBuffer(id);
		return;
	}
	auto &fs = FileSystem::GetFileSystem(db);
	auto path = GetTemporaryPath(id);
	if (fs.FileExists(path)) {
		fs.RemoveFile(path);
	}
}

string BufferManager::InMemoryWarning() {
	if (!temp_directory.empty()) {
		return "";
	}
	return "\nDatabase is launched in in-memory mode and no temporary directory is specified."
	       "\nUnused blocks cannot be offloaded to disk."
	       "\n\nLaunch the database with a persistent storage back-end"
	       "\nOr set PRAGMA temp_directory='/path/to/tmp.tmp'";
}

void BufferManager::ReserveMemory(idx_t size) {
	if (size == 0) {
		return;
	}
	auto reservation =
	    EvictBlocksOrThrow(size, maximum_memory, nullptr, "failed to reserve memory data of size %lld%s", size);
	reservation.size = 0;
}

void BufferManager::FreeReservedMemory(idx_t size) {
	if (size == 0) {
		return;
	}
	current_memory -= size;
}

//===--------------------------------------------------------------------===//
// Buffer Allocator
//===--------------------------------------------------------------------===//
data_ptr_t BufferManager::BufferAllocatorAllocate(PrivateAllocatorData *private_data, idx_t size) {
	auto &data = (BufferAllocatorData &)*private_data;
	auto reservation = data.manager.EvictBlocksOrThrow(size, data.manager.maximum_memory, nullptr,
	                                                   "failed to allocate data of size %lld%s", size);
	// We rely on manual tracking of this one. :(
	reservation.size = 0;
	return Allocator::Get(data.manager.db).AllocateData(size);
}

void BufferManager::BufferAllocatorFree(PrivateAllocatorData *private_data, data_ptr_t pointer, idx_t size) {
	auto &data = (BufferAllocatorData &)*private_data;
	BufferPoolReservation r;
	r.size = size;
	r.Resize(data.manager.current_memory, 0);
	return Allocator::Get(data.manager.db).FreeData(pointer, size);
}

data_ptr_t BufferManager::BufferAllocatorRealloc(PrivateAllocatorData *private_data, data_ptr_t pointer, idx_t old_size,
                                                 idx_t size) {
	if (old_size == size) {
		return pointer;
	}
	auto &data = (BufferAllocatorData &)*private_data;
	BufferPoolReservation r;
	r.size = old_size;
	r.Resize(data.manager.current_memory, size);
	r.size = 0;
	return Allocator::Get(data.manager.db).ReallocateData(pointer, old_size, size);
}

Allocator &BufferAllocator::Get(ClientContext &context) {
	auto &manager = BufferManager::GetBufferManager(context);
	return manager.GetBufferAllocator();
}

Allocator &BufferAllocator::Get(DatabaseInstance &db) {
	return BufferManager::GetBufferManager(db).GetBufferAllocator();
}

Allocator &BufferAllocator::Get(AttachedDatabase &db) {
	return BufferAllocator::Get(db.GetDatabase());
}

Allocator &BufferManager::GetBufferAllocator() {
	return buffer_allocator;
}

} // namespace duckdb







namespace duckdb {

CompressionType RowGroupWriter::GetColumnCompressionType(idx_t i) {
	return table.GetColumn(LogicalIndex(i)).CompressionType();
}

void RowGroupWriter::RegisterPartialBlock(PartialBlockAllocation &&allocation) {
	partial_block_manager.RegisterPartialBlock(std::move(allocation));
}

PartialBlockAllocation RowGroupWriter::GetBlockAllocation(uint32_t segment_size) {
	return partial_block_manager.GetBlockAllocation(segment_size);
}

void SingleFileRowGroupWriter::WriteColumnDataPointers(ColumnCheckpointState &column_checkpoint_state) {
	auto &meta_writer = table_data_writer;
	const auto &data_pointers = column_checkpoint_state.data_pointers;

	meta_writer.Write<idx_t>(data_pointers.size());
	// then write the data pointers themselves
	for (idx_t k = 0; k < data_pointers.size(); k++) {
		auto &data_pointer = data_pointers[k];
		meta_writer.Write<idx_t>(data_pointer.row_start);
		meta_writer.Write<idx_t>(data_pointer.tuple_count);
		meta_writer.Write<block_id_t>(data_pointer.block_pointer.block_id);
		meta_writer.Write<uint32_t>(data_pointer.block_pointer.offset);
		meta_writer.Write<CompressionType>(data_pointer.compression_type);
		data_pointer.statistics->Serialize(meta_writer);
	}
}

MetaBlockWriter &SingleFileRowGroupWriter::GetPayloadWriter() {
	return table_data_writer;
}

} // namespace duckdb















namespace duckdb {

TableDataReader::TableDataReader(MetaBlockReader &reader, BoundCreateTableInfo &info) : reader(reader), info(info) {
	info.data = make_unique<PersistentTableData>(info.Base().columns.LogicalColumnCount());
}

void TableDataReader::ReadTableData() {
	auto &columns = info.Base().columns;
	D_ASSERT(!columns.empty());

	// deserialize the total table statistics
	info.data->column_stats.reserve(columns.PhysicalColumnCount());
	for (auto &col : columns.Physical()) {
		info.data->column_stats.push_back(BaseStatistics::Deserialize(reader, col.Type()));
	}

	// deserialize each of the individual row groups
	auto row_group_count = reader.Read<uint64_t>();
	info.data->row_groups.reserve(row_group_count);
	for (idx_t i = 0; i < row_group_count; i++) {
		auto row_group_pointer = RowGroup::Deserialize(reader, columns);
		info.data->row_groups.push_back(std::move(row_group_pointer));
	}
}

} // namespace duckdb








namespace duckdb {

TableDataWriter::TableDataWriter(TableCatalogEntry &table_p) : table((DuckTableEntry &)table_p) {
	D_ASSERT(table_p.IsDuckTable());
}

TableDataWriter::~TableDataWriter() {
}

void TableDataWriter::WriteTableData() {
	// start scanning the table and append the data to the uncompressed segments
	table.GetStorage().Checkpoint(*this);
}

CompressionType TableDataWriter::GetColumnCompressionType(idx_t i) {
	return table.GetColumn(LogicalIndex(i)).CompressionType();
}

void TableDataWriter::AddRowGroup(RowGroupPointer &&row_group_pointer, unique_ptr<RowGroupWriter> &&writer) {
	row_group_pointers.push_back(std::move(row_group_pointer));
	writer.reset();
}

SingleFileTableDataWriter::SingleFileTableDataWriter(SingleFileCheckpointWriter &checkpoint_manager,
                                                     TableCatalogEntry &table, MetaBlockWriter &table_data_writer,
                                                     MetaBlockWriter &meta_data_writer)
    : TableDataWriter(table), checkpoint_manager(checkpoint_manager), table_data_writer(table_data_writer),
      meta_data_writer(meta_data_writer) {
}

unique_ptr<RowGroupWriter> SingleFileTableDataWriter::GetRowGroupWriter(RowGroup &row_group) {
	return make_unique<SingleFileRowGroupWriter>(table, checkpoint_manager.partial_block_manager, table_data_writer);
}

void SingleFileTableDataWriter::FinalizeTable(vector<unique_ptr<BaseStatistics>> &&global_stats, DataTableInfo *info) {
	// store the current position in the metadata writer
	// this is where the row groups for this table start
	auto pointer = table_data_writer.GetBlockPointer();

	for (auto &stats : global_stats) {
		stats->Serialize(table_data_writer);
	}
	// now start writing the row group pointers to disk
	table_data_writer.Write<uint64_t>(row_group_pointers.size());
	for (auto &row_group_pointer : row_group_pointers) {
		RowGroup::Serialize(row_group_pointer, table_data_writer);
	}

	// Pointer to the table itself goes to the metadata stream.
	meta_data_writer.Write<block_id_t>(pointer.block_id);
	meta_data_writer.Write<uint64_t>(pointer.offset);

	// Now we serialize indexes in the table_metadata_writer
	std::vector<BlockPointer> index_pointers = info->indexes.SerializeIndexes(table_data_writer);

	// Write-off to metadata block ids and offsets of indexes
	meta_data_writer.Write<idx_t>(index_pointers.size());
	for (auto &block_info : index_pointers) {
		meta_data_writer.Write<idx_t>(block_info.block_id);
		meta_data_writer.Write<idx_t>(block_info.offset);
	}
}

} // namespace duckdb





namespace duckdb {

WriteOverflowStringsToDisk::WriteOverflowStringsToDisk(BlockManager &block_manager)
    : block_manager(block_manager), block_id(INVALID_BLOCK), offset(0) {
}

WriteOverflowStringsToDisk::~WriteOverflowStringsToDisk() {
	if (offset > 0) {
		block_manager.Write(handle.GetFileBuffer(), block_id);
	}
}

void WriteOverflowStringsToDisk::WriteString(string_t string, block_id_t &result_block, int32_t &result_offset) {
	auto &buffer_manager = block_manager.buffer_manager;
	if (!handle.IsValid()) {
		handle = buffer_manager.Allocate(Storage::BLOCK_SIZE);
	}
	// first write the length of the string
	if (block_id == INVALID_BLOCK || offset + 2 * sizeof(uint32_t) >= STRING_SPACE) {
		AllocateNewBlock(block_manager.GetFreeBlockId());
	}
	result_block = block_id;
	result_offset = offset;

	// GZIP the string
	auto uncompressed_size = string.GetSize();
	MiniZStream s;
	size_t compressed_size = 0;
	compressed_size = s.MaxCompressedLength(uncompressed_size);
	auto compressed_buf = unique_ptr<data_t[]>(new data_t[compressed_size]);
	s.Compress((const char *)string.GetDataUnsafe(), uncompressed_size, (char *)compressed_buf.get(), &compressed_size);
	string_t compressed_string((const char *)compressed_buf.get(), compressed_size);

	// store sizes
	auto data_ptr = handle.Ptr();
	Store<uint32_t>(compressed_size, data_ptr + offset);
	Store<uint32_t>(uncompressed_size, data_ptr + offset + sizeof(uint32_t));

	// now write the remainder of the string
	offset += 2 * sizeof(uint32_t);
	auto strptr = compressed_string.GetDataUnsafe();
	uint32_t remaining = compressed_size;
	while (remaining > 0) {
		uint32_t to_write = MinValue<uint32_t>(remaining, STRING_SPACE - offset);
		if (to_write > 0) {
			memcpy(data_ptr + offset, strptr, to_write);

			remaining -= to_write;
			offset += to_write;
			strptr += to_write;
		}
		if (remaining > 0) {
			// there is still remaining stuff to write
			// first get the new block id and write it to the end of the previous block
			auto new_block_id = block_manager.GetFreeBlockId();
			Store<block_id_t>(new_block_id, data_ptr + offset);
			// now write the current block to disk and allocate a new block
			AllocateNewBlock(new_block_id);
		}
	}
}

void WriteOverflowStringsToDisk::AllocateNewBlock(block_id_t new_block_id) {
	if (block_id != INVALID_BLOCK) {
		// there is an old block, write it first
		block_manager.Write(handle.GetFileBuffer(), block_id);
	}
	offset = 0;
	block_id = new_block_id;
}

} // namespace duckdb


































namespace duckdb {

void ReorderTableEntries(vector<TableCatalogEntry *> &tables);

SingleFileCheckpointWriter::SingleFileCheckpointWriter(AttachedDatabase &db, BlockManager &block_manager)
    : CheckpointWriter(db), partial_block_manager(block_manager) {
}

BlockManager &SingleFileCheckpointWriter::GetBlockManager() {
	auto &storage_manager = (SingleFileStorageManager &)db.GetStorageManager();
	return *storage_manager.block_manager;
}

MetaBlockWriter &SingleFileCheckpointWriter::GetMetaBlockWriter() {
	return *metadata_writer;
}

unique_ptr<TableDataWriter> SingleFileCheckpointWriter::GetTableDataWriter(TableCatalogEntry &table) {
	return make_unique<SingleFileTableDataWriter>(*this, table, *table_metadata_writer, GetMetaBlockWriter());
}

void SingleFileCheckpointWriter::CreateCheckpoint() {
	auto &config = DBConfig::Get(db);
	auto &storage_manager = (SingleFileStorageManager &)db.GetStorageManager();
	if (storage_manager.InMemory()) {
		return;
	}
	// assert that the checkpoint manager hasn't been used before
	D_ASSERT(!metadata_writer);

	auto &block_manager = GetBlockManager();

	//! Set up the writers for the checkpoints
	metadata_writer = make_unique<MetaBlockWriter>(block_manager);
	table_metadata_writer = make_unique<MetaBlockWriter>(block_manager);

	// get the id of the first meta block
	block_id_t meta_block = metadata_writer->GetBlockPointer().block_id;

	vector<SchemaCatalogEntry *> schemas;
	// we scan the set of committed schemas
	auto &catalog = (DuckCatalog &)Catalog::GetCatalog(db);
	catalog.ScanSchemas([&](CatalogEntry *entry) { schemas.push_back((SchemaCatalogEntry *)entry); });
	// write the actual data into the database
	// write the amount of schemas
	metadata_writer->Write<uint32_t>(schemas.size());
	for (auto &schema : schemas) {
		WriteSchema(*schema);
	}
	partial_block_manager.FlushPartialBlocks();
	// flush the meta data to disk
	metadata_writer->Flush();
	table_metadata_writer->Flush();

	// write a checkpoint flag to the WAL
	// this protects against the rare event that the database crashes AFTER writing the file, but BEFORE truncating the
	// WAL we write an entry CHECKPOINT "meta_block_id" into the WAL upon loading, if we see there is an entry
	// CHECKPOINT "meta_block_id", and the id MATCHES the head idin the file we know that the database was successfully
	// checkpointed, so we know that we should avoid replaying the WAL to avoid duplicating data
	auto wal = storage_manager.GetWriteAheadLog();
	wal->WriteCheckpoint(meta_block);
	wal->Flush();

	if (config.options.checkpoint_abort == CheckpointAbort::DEBUG_ABORT_BEFORE_HEADER) {
		throw FatalException("Checkpoint aborted before header write because of PRAGMA checkpoint_abort flag");
	}

	// finally write the updated header
	DatabaseHeader header;
	header.meta_block = meta_block;
	block_manager.WriteHeader(header);

	if (config.options.checkpoint_abort == CheckpointAbort::DEBUG_ABORT_BEFORE_TRUNCATE) {
		throw FatalException("Checkpoint aborted before truncate because of PRAGMA checkpoint_abort flag");
	}

	// truncate the WAL
	wal->Truncate(0);

	// mark all blocks written as part of the metadata as modified
	metadata_writer->MarkWrittenBlocks();
	table_metadata_writer->MarkWrittenBlocks();
}

void SingleFileCheckpointReader::LoadFromStorage() {
	auto &block_manager = *storage.block_manager;
	block_id_t meta_block = block_manager.GetMetaBlock();
	if (meta_block < 0) {
		// storage is empty
		return;
	}

	Connection con(storage.GetDatabase());
	con.BeginTransaction();
	// create the MetaBlockReader to read from the storage
	MetaBlockReader reader(block_manager, meta_block);
	LoadCheckpoint(*con.context, reader);
	con.Commit();
}

void CheckpointReader::LoadCheckpoint(ClientContext &context, MetaBlockReader &reader) {
	uint32_t schema_count = reader.Read<uint32_t>();
	for (uint32_t i = 0; i < schema_count; i++) {
		ReadSchema(context, reader);
	}
}

//===--------------------------------------------------------------------===//
// Schema
//===--------------------------------------------------------------------===//
void CheckpointWriter::WriteSchema(SchemaCatalogEntry &schema) {
	// write the schema data
	schema.Serialize(GetMetaBlockWriter());
	// then, we fetch the tables/views/sequences information
	vector<TableCatalogEntry *> tables;
	vector<ViewCatalogEntry *> views;
	schema.Scan(CatalogType::TABLE_ENTRY, [&](CatalogEntry *entry) {
		if (entry->internal) {
			return;
		}
		if (entry->type == CatalogType::TABLE_ENTRY) {
			tables.push_back((TableCatalogEntry *)entry);
		} else if (entry->type == CatalogType::VIEW_ENTRY) {
			views.push_back((ViewCatalogEntry *)entry);
		} else {
			throw NotImplementedException("Catalog type for entries");
		}
	});
	vector<SequenceCatalogEntry *> sequences;
	schema.Scan(CatalogType::SEQUENCE_ENTRY, [&](CatalogEntry *entry) {
		if (entry->internal) {
			return;
		}
		sequences.push_back((SequenceCatalogEntry *)entry);
	});

	vector<TypeCatalogEntry *> custom_types;
	schema.Scan(CatalogType::TYPE_ENTRY, [&](CatalogEntry *entry) {
		if (entry->internal) {
			return;
		}
		custom_types.push_back((TypeCatalogEntry *)entry);
	});

	vector<ScalarMacroCatalogEntry *> macros;
	schema.Scan(CatalogType::SCALAR_FUNCTION_ENTRY, [&](CatalogEntry *entry) {
		if (entry->internal) {
			return;
		}
		if (entry->type == CatalogType::MACRO_ENTRY) {
			macros.push_back((ScalarMacroCatalogEntry *)entry);
		}
	});

	vector<TableMacroCatalogEntry *> table_macros;
	schema.Scan(CatalogType::TABLE_FUNCTION_ENTRY, [&](CatalogEntry *entry) {
		if (entry->internal) {
			return;
		}
		if (entry->type == CatalogType::TABLE_MACRO_ENTRY) {
			table_macros.push_back((TableMacroCatalogEntry *)entry);
		}
	});

	vector<IndexCatalogEntry *> indexes;
	schema.Scan(CatalogType::INDEX_ENTRY, [&](CatalogEntry *entry) {
		D_ASSERT(!entry->internal);
		indexes.push_back((IndexCatalogEntry *)entry);
	});

	FieldWriter writer(GetMetaBlockWriter());
	writer.WriteField<uint32_t>(custom_types.size());
	writer.WriteField<uint32_t>(sequences.size());
	writer.WriteField<uint32_t>(tables.size());
	writer.WriteField<uint32_t>(views.size());
	writer.WriteField<uint32_t>(macros.size());
	writer.WriteField<uint32_t>(table_macros.size());
	writer.WriteField<uint32_t>(indexes.size());
	writer.Finalize();

	// write the custom_types
	for (auto &custom_type : custom_types) {
		WriteType(*custom_type);
	}

	// write the sequences
	for (auto &seq : sequences) {
		WriteSequence(*seq);
	}
	// reorder tables because of foreign key constraint
	ReorderTableEntries(tables);
	// Write the tables
	for (auto &table : tables) {
		WriteTable(*table);
	}
	// Write the views
	for (auto &view : views) {
		WriteView(*view);
	}

	// Write the macros
	for (auto &macro : macros) {
		WriteMacro(*macro);
	}

	// Write the table's macros
	for (auto &macro : table_macros) {
		WriteTableMacro(*macro);
	}
	// Write the indexes
	for (auto &index : indexes) {
		WriteIndex(*index);
	}
}

void CheckpointReader::ReadSchema(ClientContext &context, MetaBlockReader &reader) {
	// read the schema and create it in the catalog
	auto info = SchemaCatalogEntry::Deserialize(reader);
	// we set create conflict to ignore to ignore the failure of recreating the main schema
	info->on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;
	catalog.CreateSchema(context, info.get());

	// first read all the counts
	FieldReader field_reader(reader);
	uint32_t enum_count = field_reader.ReadRequired<uint32_t>();
	uint32_t seq_count = field_reader.ReadRequired<uint32_t>();
	uint32_t table_count = field_reader.ReadRequired<uint32_t>();
	uint32_t view_count = field_reader.ReadRequired<uint32_t>();
	uint32_t macro_count = field_reader.ReadRequired<uint32_t>();
	uint32_t table_macro_count = field_reader.ReadRequired<uint32_t>();
	uint32_t table_index_count = field_reader.ReadRequired<uint32_t>();
	field_reader.Finalize();

	// now read the enums
	for (uint32_t i = 0; i < enum_count; i++) {
		ReadType(context, reader);
	}

	// read the sequences
	for (uint32_t i = 0; i < seq_count; i++) {
		ReadSequence(context, reader);
	}
	// read the table count and recreate the tables
	for (uint32_t i = 0; i < table_count; i++) {
		ReadTable(context, reader);
	}
	// now read the views
	for (uint32_t i = 0; i < view_count; i++) {
		ReadView(context, reader);
	}

	// finally read the macro's
	for (uint32_t i = 0; i < macro_count; i++) {
		ReadMacro(context, reader);
	}

	for (uint32_t i = 0; i < table_macro_count; i++) {
		ReadTableMacro(context, reader);
	}
	for (uint32_t i = 0; i < table_index_count; i++) {
		ReadIndex(context, reader);
	}
}

//===--------------------------------------------------------------------===//
// Views
//===--------------------------------------------------------------------===//
void CheckpointWriter::WriteView(ViewCatalogEntry &view) {
	view.Serialize(GetMetaBlockWriter());
}

void CheckpointReader::ReadView(ClientContext &context, MetaBlockReader &reader) {
	auto info = ViewCatalogEntry::Deserialize(reader, context);
	catalog.CreateView(context, info.get());
}

//===--------------------------------------------------------------------===//
// Sequences
//===--------------------------------------------------------------------===//
void CheckpointWriter::WriteSequence(SequenceCatalogEntry &seq) {
	seq.Serialize(GetMetaBlockWriter());
}

void CheckpointReader::ReadSequence(ClientContext &context, MetaBlockReader &reader) {
	auto info = SequenceCatalogEntry::Deserialize(reader);
	catalog.CreateSequence(context, info.get());
}

//===--------------------------------------------------------------------===//
// Indexes
//===--------------------------------------------------------------------===//
void CheckpointWriter::WriteIndex(IndexCatalogEntry &index_catalog) {
	// The index data should already have been written as part of WriteTableData.
	// Here, we need only serialize the pointer to that data.
	auto root_offset = index_catalog.index->GetSerializedDataPointer();
	auto &metadata_writer = GetMetaBlockWriter();
	index_catalog.Serialize(metadata_writer);
	// Serialize the Block id and offset of root node
	metadata_writer.Write(root_offset.block_id);
	metadata_writer.Write(root_offset.offset);
}

void CheckpointReader::ReadIndex(ClientContext &context, MetaBlockReader &reader) {

	// Deserialize the index meta data
	auto info = IndexCatalogEntry::Deserialize(reader, context);

	// Create index in the catalog
	auto schema_catalog = catalog.GetSchema(context, info->schema);
	auto table_catalog =
	    (DuckTableEntry *)catalog.GetEntry(context, CatalogType::TABLE_ENTRY, info->schema, info->table->table_name);
	auto index_catalog = (DuckIndexEntry *)schema_catalog->CreateIndex(context, info.get(), table_catalog);
	index_catalog->info = table_catalog->GetStorage().info;
	// Here we just gotta read the root node
	auto root_block_id = reader.Read<block_id_t>();
	auto root_offset = reader.Read<uint32_t>();

	// create an adaptive radix tree around the expressions
	vector<unique_ptr<Expression>> unbound_expressions;
	vector<unique_ptr<ParsedExpression>> parsed_expressions;

	for (auto &p_exp : info->parsed_expressions) {
		parsed_expressions.push_back(p_exp->Copy());
	}

	auto binder = Binder::CreateBinder(context);
	auto table_ref = (TableRef *)info->table.get();
	auto bound_table = binder->Bind(*table_ref);
	D_ASSERT(bound_table->type == TableReferenceType::BASE_TABLE);
	IndexBinder idx_binder(*binder, context);
	unbound_expressions.reserve(parsed_expressions.size());
	for (auto &expr : parsed_expressions) {
		unbound_expressions.push_back(idx_binder.Bind(expr));
	}

	if (parsed_expressions.empty()) {
		// If no parsed_expressions are present, this means this is a PK/FK index, so we create the necessary bound
		// column refs
		unbound_expressions.reserve(info->column_ids.size());
		for (idx_t key_nr = 0; key_nr < info->column_ids.size(); key_nr++) {
			auto &col = table_catalog->GetColumn(LogicalIndex(info->column_ids[key_nr]));
			unbound_expressions.push_back(
			    make_unique<BoundColumnRefExpression>(col.GetName(), col.GetType(), ColumnBinding(0, key_nr)));
		}
	}

	switch (info->index_type) {
	case IndexType::ART: {
		auto &storage = table_catalog->GetStorage();
		auto art = make_unique<ART>(info->column_ids, TableIOManager::Get(storage), std::move(unbound_expressions),
		                            info->constraint_type, storage.db, true, root_block_id, root_offset);
		index_catalog->index = art.get();
		storage.info->indexes.AddIndex(std::move(art));
		break;
	}
	default:
		throw InternalException("Can't read this index type");
	}
}

//===--------------------------------------------------------------------===//
// Custom Types
//===--------------------------------------------------------------------===//
void CheckpointWriter::WriteType(TypeCatalogEntry &table) {
	table.Serialize(GetMetaBlockWriter());
}

void CheckpointReader::ReadType(ClientContext &context, MetaBlockReader &reader) {
	auto info = TypeCatalogEntry::Deserialize(reader);
	catalog.CreateType(context, info.get());
}

//===--------------------------------------------------------------------===//
// Macro's
//===--------------------------------------------------------------------===//
void CheckpointWriter::WriteMacro(ScalarMacroCatalogEntry &macro) {
	macro.Serialize(GetMetaBlockWriter());
}

void CheckpointReader::ReadMacro(ClientContext &context, MetaBlockReader &reader) {
	auto info = ScalarMacroCatalogEntry::Deserialize(reader, context);
	catalog.CreateFunction(context, info.get());
}

void CheckpointWriter::WriteTableMacro(TableMacroCatalogEntry &macro) {
	macro.Serialize(GetMetaBlockWriter());
}

void CheckpointReader::ReadTableMacro(ClientContext &context, MetaBlockReader &reader) {
	auto info = TableMacroCatalogEntry::Deserialize(reader, context);
	catalog.CreateFunction(context, info.get());
}

//===--------------------------------------------------------------------===//
// Table Metadata
//===--------------------------------------------------------------------===//
void CheckpointWriter::WriteTable(TableCatalogEntry &table) {
	// write the table meta data
	table.Serialize(GetMetaBlockWriter());
	// now we need to write the table data.
	if (auto writer = GetTableDataWriter(table)) {
		writer->WriteTableData();
	}
}

void CheckpointReader::ReadTable(ClientContext &context, MetaBlockReader &reader) {
	// deserialize the table meta data
	auto info = TableCatalogEntry::Deserialize(reader, context);
	// bind the info
	auto binder = Binder::CreateBinder(context);
	auto schema = catalog.GetSchema(context, info->schema);
	auto bound_info = binder->BindCreateTableInfo(std::move(info), schema);

	// now read the actual table data and place it into the create table info
	ReadTableData(context, reader, *bound_info);

	// finally create the table in the catalog
	catalog.CreateTable(context, bound_info.get());
}

void CheckpointReader::ReadTableData(ClientContext &context, MetaBlockReader &reader,
                                     BoundCreateTableInfo &bound_info) {
	auto block_id = reader.Read<block_id_t>();
	auto offset = reader.Read<uint64_t>();

	MetaBlockReader table_data_reader(reader.block_manager, block_id);
	table_data_reader.offset = offset;
	TableDataReader data_reader(table_data_reader, bound_info);

	data_reader.ReadTableData();

	// Get any indexes block info
	idx_t num_indexes = reader.Read<idx_t>();
	for (idx_t i = 0; i < num_indexes; i++) {
		auto idx_block_id = reader.Read<idx_t>();
		auto idx_offset = reader.Read<idx_t>();
		bound_info.indexes.emplace_back(idx_block_id, idx_offset);
	}
}

} // namespace duckdb














#include <functional>

namespace duckdb {

static constexpr const idx_t BITPACKING_METADATA_GROUP_SIZE = STANDARD_VECTOR_SIZE > 512 ? STANDARD_VECTOR_SIZE : 2048;

BitpackingMode BitpackingModeFromString(const string &str) {
	auto mode = StringUtil::Lower(str);

	if (mode == "auto") {
		return BitpackingMode::AUTO;
	} else if (mode == "constant") {
		return BitpackingMode::CONSTANT;
	} else if (mode == "constant_delta") {
		return BitpackingMode::CONSTANT_DELTA;
	} else if (mode == "delta_for") {
		return BitpackingMode::DELTA_FOR;
	} else if (mode == "for") {
		return BitpackingMode::FOR;
	} else {
		return BitpackingMode::AUTO;
	}
}

string BitpackingModeToString(const BitpackingMode &mode) {
	switch (mode) {
	case (BitpackingMode::AUTO):
		return "auto";
	case (BitpackingMode::CONSTANT):
		return "constant";
	case (BitpackingMode::CONSTANT_DELTA):
		return "constant_delta";
	case (BitpackingMode::DELTA_FOR):
		return "delta_for";
	case (BitpackingMode::FOR):
		return "for";
	default:
		throw NotImplementedException("Unknown bitpacking mode: " + to_string((uint8_t)mode) + "\n");
	}
}

typedef struct {
	BitpackingMode mode;
	uint32_t offset;
} bitpacking_metadata_t;

typedef uint32_t bitpacking_metadata_encoded_t;

static bitpacking_metadata_encoded_t EncodeMeta(bitpacking_metadata_t metadata) {
	D_ASSERT(metadata.offset <= 16777215); // max uint24_t
	bitpacking_metadata_encoded_t encoded_value = metadata.offset;
	encoded_value |= (uint8_t)metadata.mode << 24;
	return encoded_value;
}
static bitpacking_metadata_t DecodeMeta(bitpacking_metadata_encoded_t *metadata_encoded) {
	bitpacking_metadata_t metadata;
	metadata.mode = Load<BitpackingMode>((data_ptr_t)(metadata_encoded) + 3);
	metadata.offset = *metadata_encoded & 0x00FFFFFF;
	return metadata;
}

struct EmptyBitpackingWriter {
	template <class T>
	static void WriteConstant(T constant, idx_t count, void *data_ptr, bool all_invalid) {
	}
	template <class T, class T_S = typename std::make_signed<T>::type>
	static void WriteConstantDelta(T_S constant, T frame_of_reference, idx_t count, T *values, bool *validity,
	                               void *data_ptr) {
	}
	template <class T, class T_S = typename std::make_signed<T>::type>
	static void WriteDeltaFor(T *values, bool *validity, bitpacking_width_t width, T frame_of_reference,
	                          T_S delta_offset, T *original_values, idx_t count, void *data_ptr) {
	}
	template <class T>
	static void WriteFor(T *values, bool *validity, bitpacking_width_t width, T frame_of_reference, idx_t count,
	                     void *data_ptr) {
	}
};

template <class T, class T_U = typename std::make_unsigned<T>::type, class T_S = typename std::make_signed<T>::type>
struct BitpackingState {
public:
	BitpackingState() : compression_buffer_idx(0), total_size(0), data_ptr(nullptr) {
		compression_buffer_internal[0] = (T)0;
		compression_buffer = &compression_buffer_internal[1];
		Reset();
	}

	// Extra val for delta encoding
	T compression_buffer_internal[BITPACKING_METADATA_GROUP_SIZE + 1];
	T *compression_buffer;
	T_S delta_buffer[BITPACKING_METADATA_GROUP_SIZE];
	bool compression_buffer_validity[BITPACKING_METADATA_GROUP_SIZE];
	idx_t compression_buffer_idx;
	idx_t total_size;

	// Used to pass CompressionState ptr through the Bitpacking writer
	void *data_ptr;

	// Stats on current compression buffer
	T minimum;
	T maximum;
	T min_max_diff;
	T_S minimum_delta;
	T_S maximum_delta;
	T_S min_max_delta_diff;
	T_S delta_offset;
	bool all_valid;
	bool all_invalid;

	bool can_do_delta;
	bool can_do_for;

	// Used to force a specific mode, useful in testing
	BitpackingMode mode = BitpackingMode::AUTO;

public:
	void Reset() {
		minimum = NumericLimits<T>::Maximum();
		minimum_delta = NumericLimits<T_S>::Maximum();
		maximum = NumericLimits<T>::Minimum();
		maximum_delta = NumericLimits<T_S>::Minimum();
		delta_offset = 0;
		all_valid = true;
		all_invalid = true;
		can_do_delta = false;
		can_do_for = false;
		compression_buffer_idx = 0;
		min_max_diff = 0;
		min_max_delta_diff = 0;
	}

	void CalculateFORStats() {
		can_do_for = TrySubtractOperator::Operation(maximum, minimum, min_max_diff);
	}

	void CalculateDeltaStats() {
		// TODO: currently we dont support delta compression of values above NumericLimits<T_S>::Maximum(),
		// 		 we could support this with some clever substract trickery?
		if (maximum > (T)NumericLimits<T_S>::Maximum()) {
			return;
		}

		// Don't delta encoding 1 value makes no sense
		if (compression_buffer_idx < 2) {
			return;
		};

		// TODO: handle NULLS here?
		// Currently we cannot handle nulls because we would need an additional step of patching for this.
		// we could for example copy the last value on a null insert. This would help a bit, but not be optimal for
		// large deltas since theres suddenly a zero then. Ideally we would insert a value that leads to a delta within
		// the current domain of deltas however we dont know that domain here yet
		if (!all_valid) {
			return;
		}

		// Note: since we dont allow any values over NumericLimits<T_S>::Maximum(), all subtractions for unsigned types
		// are guaranteed not to overflow
		bool can_do_all = true;
		if (std::is_signed<T>()) {
			T_S bogus;
			can_do_all = TrySubtractOperator::Operation((T_S)(minimum), (T_S)(maximum), bogus) &&
			             TrySubtractOperator::Operation((T_S)(maximum), (T_S)(minimum), bogus);
		}

		// Calculate delta's
		if (can_do_all) {
			for (int64_t i = 0; i < (int64_t)compression_buffer_idx; i++) {
				delta_buffer[i] = (T_S)compression_buffer[i] - (T_S)compression_buffer[i - 1];
			}
		} else {
			for (int64_t i = 0; i < (int64_t)compression_buffer_idx; i++) {
				auto success = TrySubtractOperator::Operation((T_S)(compression_buffer[i]),
				                                              (T_S)(compression_buffer[i - 1]), delta_buffer[i]);
				if (!success) {
					return;
				}
			}
		}

		can_do_delta = true;

		for (int64_t i = 1; i < (int64_t)compression_buffer_idx; i++) {
			maximum_delta = MaxValue<T_S>(maximum_delta, delta_buffer[i]);
			minimum_delta = MinValue<T_S>(minimum_delta, delta_buffer[i]);
		}

		// Since we can set the first value arbitrarily, we want to pick one from the current domain, note that
		// we will store the original first value - this offset as the  delta_offset to be able to decode this again.
		delta_buffer[0] = minimum_delta;

		can_do_delta = can_do_delta && TrySubtractOperator::Operation(maximum_delta, minimum_delta, min_max_delta_diff);
		can_do_delta =
		    can_do_delta && TrySubtractOperator::Operation((T_S)(compression_buffer[0]), minimum_delta, delta_offset);
	}

	template <class T_INNER>
	void SubtractFrameOfReference(T_INNER *buffer, T_INNER frame_of_reference) {
		static_assert(std::is_integral<T_INNER>::value, "Integral type required.");
		for (idx_t i = 0; i < compression_buffer_idx; i++) {
			buffer[i] -= uint64_t(frame_of_reference);
		}
	}

	template <class OP>
	bool Flush() {
		if (compression_buffer_idx == 0) {
			return true;
		}

		if ((all_invalid || maximum == minimum) && (mode == BitpackingMode::AUTO || mode == BitpackingMode::CONSTANT)) {
			OP::WriteConstant(maximum, compression_buffer_idx, data_ptr, all_invalid);
			total_size += sizeof(T) + sizeof(bitpacking_metadata_encoded_t);
			return true;
		}

		CalculateFORStats();
		CalculateDeltaStats();

		if (can_do_delta) {
			if (maximum_delta == minimum_delta && mode != BitpackingMode::FOR && mode != BitpackingMode::DELTA_FOR) {
				idx_t frame_of_reference = compression_buffer[0];
				OP::WriteConstantDelta((T_S)maximum_delta, (T)frame_of_reference, compression_buffer_idx,
				                       (T *)compression_buffer, (bool *)compression_buffer_validity, data_ptr);
				total_size += sizeof(T) + sizeof(T) + sizeof(bitpacking_metadata_encoded_t);
				return true;
			}

			// Check if delta has benefit
			auto delta_required_bitwidth = BitpackingPrimitives::MinimumBitWidth<T_U>(min_max_delta_diff);
			auto regular_required_bitwidth = BitpackingPrimitives::MinimumBitWidth(min_max_diff);

			if (delta_required_bitwidth < regular_required_bitwidth && mode != BitpackingMode::FOR) {
				SubtractFrameOfReference(delta_buffer, minimum_delta);

				OP::WriteDeltaFor((T *)delta_buffer, compression_buffer_validity, delta_required_bitwidth,
				                  (T)minimum_delta, delta_offset, (T *)compression_buffer, compression_buffer_idx,
				                  data_ptr);

				total_size += BitpackingPrimitives::GetRequiredSize(compression_buffer_idx, delta_required_bitwidth);
				total_size += sizeof(T);                              // FOR value
				total_size += sizeof(T);                              // Delta offset value
				total_size += AlignValue(sizeof(bitpacking_width_t)); // FOR value

				return true;
			}
		}

		if (can_do_for) {
			auto width = BitpackingPrimitives::MinimumBitWidth<T_U>(min_max_diff);
			SubtractFrameOfReference(compression_buffer, minimum);
			OP::WriteFor(compression_buffer, compression_buffer_validity, width, minimum, compression_buffer_idx,
			             data_ptr);

			total_size += BitpackingPrimitives::GetRequiredSize(compression_buffer_idx, width);
			total_size += sizeof(T); // FOR value
			total_size += AlignValue(sizeof(bitpacking_width_t));

			return true;
		}

		return false;
	}

	template <class OP = EmptyBitpackingWriter>
	bool Update(T value, bool is_valid) {
		compression_buffer_validity[compression_buffer_idx] = is_valid;
		all_valid = all_valid && is_valid;
		all_invalid = all_invalid && !is_valid;

		if (is_valid) {
			compression_buffer[compression_buffer_idx] = value;
			minimum = MinValue<T>(minimum, value);
			maximum = MaxValue<T>(maximum, value);
		}

		compression_buffer_idx++;

		if (compression_buffer_idx == BITPACKING_METADATA_GROUP_SIZE) {
			bool success = Flush<OP>();
			Reset();
			return success;
		}
		return true;
	}
};

//===--------------------------------------------------------------------===//
// Analyze
//===--------------------------------------------------------------------===//
template <class T>
struct BitpackingAnalyzeState : public AnalyzeState {
	BitpackingState<T> state;
};

template <class T>
unique_ptr<AnalyzeState> BitpackingInitAnalyze(ColumnData &col_data, PhysicalType type) {
	auto &config = DBConfig::GetConfig(col_data.GetDatabase());

	auto state = make_unique<BitpackingAnalyzeState<T>>();
	state->state.mode = config.options.force_bitpacking_mode;

	return std::move(state);
}

template <class T>
bool BitpackingAnalyze(AnalyzeState &state, Vector &input, idx_t count) {
	auto &analyze_state = (BitpackingAnalyzeState<T> &)state;
	UnifiedVectorFormat vdata;
	input.ToUnifiedFormat(count, vdata);

	auto data = (T *)vdata.data;
	for (idx_t i = 0; i < count; i++) {
		auto idx = vdata.sel->get_index(i);
		if (!analyze_state.state.template Update<EmptyBitpackingWriter>(data[idx], vdata.validity.RowIsValid(idx))) {
			return false;
		}
	}
	return true;
}

template <class T>
idx_t BitpackingFinalAnalyze(AnalyzeState &state) {
	auto &bitpacking_state = (BitpackingAnalyzeState<T> &)state;
	auto flush_result = bitpacking_state.state.template Flush<EmptyBitpackingWriter>();
	if (!flush_result) {
		return DConstants::INVALID_INDEX;
	}
	return bitpacking_state.state.total_size;
}

//===--------------------------------------------------------------------===//
// Compress
//===--------------------------------------------------------------------===//
template <class T, class T_S = typename std::make_signed<T>::type>
struct BitpackingCompressState : public CompressionState {
public:
	explicit BitpackingCompressState(ColumnDataCheckpointer &checkpointer) : checkpointer(checkpointer) {
		auto &db = checkpointer.GetDatabase();
		auto &type = checkpointer.GetType();
		auto &config = DBConfig::GetConfig(db);
		function = config.GetCompressionFunction(CompressionType::COMPRESSION_BITPACKING, type.InternalType());
		CreateEmptySegment(checkpointer.GetRowGroup().start);

		state.data_ptr = (void *)this;

		state.mode = config.options.force_bitpacking_mode;
	}

	ColumnDataCheckpointer &checkpointer;
	CompressionFunction *function;
	unique_ptr<ColumnSegment> current_segment;
	BufferHandle handle;

	// Ptr to next free spot in segment;
	data_ptr_t data_ptr;
	// Ptr to next free spot for storing bitwidths and frame-of-references (growing downwards).
	data_ptr_t metadata_ptr;

	BitpackingState<T> state;

public:
	struct BitpackingWriter {
		static void WriteConstant(T constant, idx_t count, void *data_ptr, bool all_invalid) {
			auto state = (BitpackingCompressState<T> *)data_ptr;

			ReserveSpace(state, sizeof(T));
			WriteMetaData(state, BitpackingMode::CONSTANT);
			WriteData(state->data_ptr, constant);

			UpdateStats(state, count);
		}

		static void WriteConstantDelta(T_S constant, T frame_of_reference, idx_t count, T *values, bool *validity,
		                               void *data_ptr) {
			auto state = (BitpackingCompressState<T> *)data_ptr;

			ReserveSpace(state, 2 * sizeof(T));
			WriteMetaData(state, BitpackingMode::CONSTANT_DELTA);
			WriteData(state->data_ptr, frame_of_reference);
			WriteData(state->data_ptr, constant);

			UpdateStats(state, count);
		}

		static void WriteDeltaFor(T *values, bool *validity, bitpacking_width_t width, T frame_of_reference,
		                          T_S delta_offset, T *original_values, idx_t count, void *data_ptr) {
			auto state = (BitpackingCompressState<T> *)data_ptr;

			auto bp_size = BitpackingPrimitives::GetRequiredSize(count, width);
			ReserveSpace(state, bp_size + 3 * sizeof(T));

			WriteMetaData(state, BitpackingMode::DELTA_FOR);
			WriteData(state->data_ptr, frame_of_reference);
			WriteData(state->data_ptr, (T)width);
			WriteData(state->data_ptr, delta_offset);

			BitpackingPrimitives::PackBuffer<T, false>(state->data_ptr, values, count, width);
			state->data_ptr += bp_size;

			UpdateStats(state, count);
		}

		static void WriteFor(T *values, bool *validity, bitpacking_width_t width, T frame_of_reference, idx_t count,
		                     void *data_ptr) {
			auto state = (BitpackingCompressState<T> *)data_ptr;

			auto bp_size = BitpackingPrimitives::GetRequiredSize(count, width);
			ReserveSpace(state, bp_size + 2 * sizeof(T));

			WriteMetaData(state, BitpackingMode::FOR);
			WriteData(state->data_ptr, frame_of_reference);
			WriteData(state->data_ptr, (T)width);

			BitpackingPrimitives::PackBuffer<T, false>(state->data_ptr, values, count, width);
			state->data_ptr += bp_size;

			UpdateStats(state, count);
		}

		template <class T_OUT>
		static void WriteData(data_ptr_t &ptr, T_OUT val) {
			*((T_OUT *)ptr) = val;
			ptr += sizeof(T_OUT);
		}

		static void WriteMetaData(BitpackingCompressState<T> *state, BitpackingMode mode) {
			bitpacking_metadata_t metadata {mode, (uint32_t)(state->data_ptr - state->handle.Ptr())};
			state->metadata_ptr -= sizeof(bitpacking_metadata_encoded_t);
			Store<bitpacking_metadata_encoded_t>(EncodeMeta(metadata), state->metadata_ptr);
		}

		static void ReserveSpace(BitpackingCompressState<T> *state, idx_t data_bytes) {
			idx_t meta_bytes = sizeof(bitpacking_metadata_encoded_t);
			state->FlushAndCreateSegmentIfFull(data_bytes + meta_bytes);
			D_ASSERT(data_bytes + meta_bytes <= state->RemainingSize());
		}

		static void UpdateStats(BitpackingCompressState<T> *state, idx_t count) {
			state->current_segment->count += count;

			if (!state->state.all_invalid) {
				NumericStatistics::Update<T>(state->current_segment->stats, state->state.minimum);
				NumericStatistics::Update<T>(state->current_segment->stats, state->state.maximum);
			}
		}
	};

	// Space remaining between the metadata_ptr growing down and data ptr growing up
	idx_t RemainingSize() {
		return metadata_ptr - data_ptr;
	}

	void CreateEmptySegment(idx_t row_start) {
		auto &db = checkpointer.GetDatabase();
		auto &type = checkpointer.GetType();
		auto compressed_segment = ColumnSegment::CreateTransientSegment(db, type, row_start);
		compressed_segment->function = function;
		current_segment = std::move(compressed_segment);
		auto &buffer_manager = BufferManager::GetBufferManager(db);
		handle = buffer_manager.Pin(current_segment->block);

		data_ptr = handle.Ptr() + BitpackingPrimitives::BITPACKING_HEADER_SIZE;
		metadata_ptr = handle.Ptr() + Storage::BLOCK_SIZE;
	}

	void Append(UnifiedVectorFormat &vdata, idx_t count) {
		auto data = (T *)vdata.data;

		for (idx_t i = 0; i < count; i++) {
			auto idx = vdata.sel->get_index(i);
			state.template Update<BitpackingCompressState<T, T_S>::BitpackingWriter>(data[idx],
			                                                                         vdata.validity.RowIsValid(idx));
		}
	}

	void FlushAndCreateSegmentIfFull(idx_t required_space) {
		if (RemainingSize() < required_space) {
			auto row_start = current_segment->start + current_segment->count;
			FlushSegment();
			CreateEmptySegment(row_start);
		}
	}

	void FlushSegment() {
		auto &state = checkpointer.GetCheckpointState();
		auto base_ptr = handle.Ptr();

		// Compact the segment by moving the metadata next to the data.
		idx_t metadata_offset = AlignValue(data_ptr - base_ptr);
		idx_t metadata_size = base_ptr + Storage::BLOCK_SIZE - metadata_ptr;
		idx_t total_segment_size = metadata_offset + metadata_size;
		memmove(base_ptr + metadata_offset, metadata_ptr, metadata_size);

		// Store the offset of the metadata of the first group (which is at the highest address).
		Store<idx_t>(metadata_offset + metadata_size, base_ptr);
		handle.Destroy();

		state.FlushSegment(std::move(current_segment), total_segment_size);
	}

	void Finalize() {
		state.template Flush<BitpackingCompressState<T, T_S>::BitpackingWriter>();
		FlushSegment();
		current_segment.reset();
	}
};

template <class T>
unique_ptr<CompressionState> BitpackingInitCompression(ColumnDataCheckpointer &checkpointer,
                                                       unique_ptr<AnalyzeState> state) {
	return make_unique<BitpackingCompressState<T>>(checkpointer);
}

template <class T>
void BitpackingCompress(CompressionState &state_p, Vector &scan_vector, idx_t count) {
	auto &state = (BitpackingCompressState<T> &)state_p;
	UnifiedVectorFormat vdata;
	scan_vector.ToUnifiedFormat(count, vdata);
	state.Append(vdata, count);
}

template <class T>
void BitpackingFinalizeCompress(CompressionState &state_p) {
	auto &state = (BitpackingCompressState<T> &)state_p;
	state.Finalize();
}

//===--------------------------------------------------------------------===//
// Scan
//===--------------------------------------------------------------------===//
template <class T>
static void ApplyFrameOfReference(T *dst, T frame_of_reference, idx_t size) {
	if (!frame_of_reference) {
		return;
	}
	for (idx_t i = 0; i < size; i++) {
		dst[i] += frame_of_reference;
	}
}

// Based on https://github.com/lemire/FastPFor (Apache License 2.0)
template <class T>
static T DeltaDecode(T *data, T previous_value, const size_t size) {
	D_ASSERT(size >= 1);

	data[0] += previous_value;

	const size_t UnrollQty = 4;
	const size_t sz0 = (size / UnrollQty) * UnrollQty; // equal to 0, if size < UnrollQty
	size_t i = 1;
	if (sz0 >= UnrollQty) {
		T a = data[0];
		for (; i < sz0 - UnrollQty; i += UnrollQty) {
			a = data[i] += a;
			a = data[i + 1] += a;
			a = data[i + 2] += a;
			a = data[i + 3] += a;
		}
	}
	for (; i != size; ++i) {
		data[i] += data[i - 1];
	}

	return data[size - 1];
}

template <class T, class T_S = typename std::make_signed<T>::type>
struct BitpackingScanState : public SegmentScanState {
public:
	explicit BitpackingScanState(ColumnSegment &segment) : current_segment(segment) {
		auto &buffer_manager = BufferManager::GetBufferManager(segment.db);
		handle = buffer_manager.Pin(segment.block);
		auto dataptr = handle.Ptr();

		// load offset to bitpacking widths pointer
		auto bitpacking_metadata_offset = Load<idx_t>(dataptr + segment.GetBlockOffset());
		bitpacking_metadata_ptr =
		    dataptr + segment.GetBlockOffset() + bitpacking_metadata_offset - sizeof(bitpacking_metadata_encoded_t);

		// load the first group
		LoadNextGroup();
	}

	BufferHandle handle;
	ColumnSegment &current_segment;

	T decompression_buffer[BITPACKING_METADATA_GROUP_SIZE];

	bitpacking_metadata_t current_group;

	bitpacking_width_t current_width;
	T current_frame_of_reference;
	T current_constant;
	T current_delta_offset;

	idx_t current_group_offset = 0;
	data_ptr_t current_group_ptr;
	data_ptr_t bitpacking_metadata_ptr;

public:
	//! Loads the metadata for the current metadata group. This will set bitpacking_metadata_ptr to the next group.
	//! this will also load any metadata that is at the start of a compressed buffer (e.g. the width, for, or constant
	//! value) depending on the bitpacking mode for that group
	void LoadNextGroup() {
		D_ASSERT(bitpacking_metadata_ptr > handle.Ptr() &&
		         bitpacking_metadata_ptr < handle.Ptr() + Storage::BLOCK_SIZE);
		current_group_offset = 0;
		current_group = DecodeMeta((bitpacking_metadata_encoded_t *)bitpacking_metadata_ptr);

		bitpacking_metadata_ptr -= sizeof(bitpacking_metadata_encoded_t);
		current_group_ptr = GetPtr(current_group);

		// Read first value
		switch (current_group.mode) {
		case BitpackingMode::CONSTANT:
			current_constant = *(T *)(current_group_ptr);
			current_group_ptr += sizeof(T);
			break;
		case BitpackingMode::FOR:
		case BitpackingMode::CONSTANT_DELTA:
		case BitpackingMode::DELTA_FOR:
			current_frame_of_reference = *(T *)(current_group_ptr);
			current_group_ptr += sizeof(T);
			break;
		default:
			throw InternalException("Invalid bitpacking mode");
		}

		// Read second value
		switch (current_group.mode) {
		case BitpackingMode::CONSTANT_DELTA:
			current_constant = *(T *)(current_group_ptr);
			current_group_ptr += sizeof(T);
			break;
		case BitpackingMode::FOR:
		case BitpackingMode::DELTA_FOR:
			current_width = (bitpacking_width_t) * (T *)(current_group_ptr);
			current_group_ptr += MaxValue(sizeof(T), sizeof(bitpacking_width_t));
			break;
		case BitpackingMode::CONSTANT:
			break;
		default:
			throw InternalException("Invalid bitpacking mode");
		}

		// Read third value
		if (current_group.mode == BitpackingMode::DELTA_FOR) {
			current_delta_offset = *(T *)(current_group_ptr);
			current_group_ptr += sizeof(T);
		}
	}

	void Skip(ColumnSegment &segment, idx_t skip_count) {
		while (skip_count > 0) {
			if (current_group_offset + skip_count < BITPACKING_METADATA_GROUP_SIZE) {
				// Skipping Delta FOR requires a bit of decoding to figure out the new delta
				if (current_group.mode == BitpackingMode::DELTA_FOR) {
					// if current_group_offset points into the middle of a
					// BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE, we need to scan a few
					// values before current_group_offset to align with the algorithm groups
					idx_t extra_count = current_group_offset % BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE;

					// Calculate total offset and count to bitunpack
					idx_t base_decompress_count = BitpackingPrimitives::RoundUpToAlgorithmGroupSize(skip_count);
					idx_t decompress_count = base_decompress_count + extra_count;
					idx_t decompress_offset = current_group_offset - extra_count;
					bool skip_sign_extension = true;

					BitpackingPrimitives::UnPackBuffer<T>((data_ptr_t)decompression_buffer,
					                                      current_group_ptr + decompress_offset, decompress_count,
					                                      current_width, skip_sign_extension);

					ApplyFrameOfReference<T_S>((T_S *)&decompression_buffer[extra_count], current_frame_of_reference,
					                           skip_count);
					DeltaDecode<T_S>((T_S *)&decompression_buffer[extra_count], (T_S)current_delta_offset,
					                 (idx_t)skip_count);
					current_delta_offset = decompression_buffer[extra_count + skip_count - 1];

					current_group_offset += skip_count;
				} else {
					current_group_offset += skip_count;
				}
				break;
			} else {
				auto left_in_this_group = BITPACKING_METADATA_GROUP_SIZE - current_group_offset;
				auto number_of_groups_to_skip = (skip_count - left_in_this_group) / BITPACKING_METADATA_GROUP_SIZE;

				current_group_offset = 0;
				bitpacking_metadata_ptr -= number_of_groups_to_skip * sizeof(bitpacking_metadata_encoded_t);

				LoadNextGroup();

				skip_count -= left_in_this_group;
				skip_count -= number_of_groups_to_skip * BITPACKING_METADATA_GROUP_SIZE;
			}
		}
	}

	data_ptr_t GetPtr(bitpacking_metadata_t group) {
		return handle.Ptr() + current_segment.GetBlockOffset() + group.offset;
	}
};

template <class T>
unique_ptr<SegmentScanState> BitpackingInitScan(ColumnSegment &segment) {
	auto result = make_unique<BitpackingScanState<T>>(segment);
	return std::move(result);
}

//===--------------------------------------------------------------------===//
// Scan base data
//===--------------------------------------------------------------------===//
template <class T, class T_S = typename std::make_signed<T>::type>
void BitpackingScanPartial(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count, Vector &result,
                           idx_t result_offset) {
	auto &scan_state = (BitpackingScanState<T> &)*state.scan_state;

	T *result_data = FlatVector::GetData<T>(result);
	result.SetVectorType(VectorType::FLAT_VECTOR);

	//! Because FOR offsets all our values to be 0 or above, we can always skip sign extension here
	bool skip_sign_extend = true;

	idx_t scanned = 0;

	while (scanned < scan_count) {
		// Exhausted this metadata group, move pointers to next group and load metadata for next group.
		if (scan_state.current_group_offset >= BITPACKING_METADATA_GROUP_SIZE) {
			scan_state.LoadNextGroup();
		}

		idx_t offset_in_compression_group =
		    scan_state.current_group_offset % BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE;

		if (scan_state.current_group.mode == BitpackingMode::CONSTANT) {
			idx_t remaining = scan_count - scanned;
			idx_t to_scan = MinValue(remaining, BITPACKING_METADATA_GROUP_SIZE - scan_state.current_group_offset);
			T *begin = result_data + result_offset + scanned;
			T *end = begin + remaining;
			std::fill(begin, end, scan_state.current_constant);
			scanned += to_scan;
			scan_state.current_group_offset += to_scan;
			continue;
		}
		if (scan_state.current_group.mode == BitpackingMode::CONSTANT_DELTA) {
			idx_t remaining = scan_count - scanned;
			idx_t to_scan = MinValue(remaining, BITPACKING_METADATA_GROUP_SIZE - scan_state.current_group_offset);
			T *target_ptr = result_data + result_offset + scanned;

			for (idx_t i = 0; i < to_scan; i++) {
				target_ptr[i] = ((scan_state.current_group_offset + i) * scan_state.current_constant) +
				                scan_state.current_frame_of_reference;
			}

			scanned += to_scan;
			scan_state.current_group_offset += to_scan;
			continue;
		}
		D_ASSERT(scan_state.current_group.mode == BitpackingMode::FOR ||
		         scan_state.current_group.mode == BitpackingMode::DELTA_FOR);

		idx_t to_scan = MinValue<idx_t>(scan_count - scanned, BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE -
		                                                          offset_in_compression_group);
		// Calculate start of compression algorithm group
		data_ptr_t current_position_ptr =
		    scan_state.current_group_ptr + scan_state.current_group_offset * scan_state.current_width / 8;
		data_ptr_t decompression_group_start_pointer =
		    current_position_ptr - offset_in_compression_group * scan_state.current_width / 8;

		T *current_result_ptr = result_data + result_offset + scanned;

		if (to_scan == BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE && offset_in_compression_group == 0) {
			// Decompress directly into result vector
			BitpackingPrimitives::UnPackBlock<T>((data_ptr_t)current_result_ptr, decompression_group_start_pointer,
			                                     scan_state.current_width, skip_sign_extend);
		} else {
			// Decompress compression algorithm to buffer
			BitpackingPrimitives::UnPackBlock<T>((data_ptr_t)scan_state.decompression_buffer,
			                                     decompression_group_start_pointer, scan_state.current_width,
			                                     skip_sign_extend);

			memcpy(current_result_ptr, scan_state.decompression_buffer + offset_in_compression_group,
			       to_scan * sizeof(T));
		}

		if (scan_state.current_group.mode == BitpackingMode::DELTA_FOR) {
			ApplyFrameOfReference<T_S>((T_S *)current_result_ptr, (T_S)scan_state.current_frame_of_reference, to_scan);
			DeltaDecode<T_S>((T_S *)current_result_ptr, (T_S)scan_state.current_delta_offset, to_scan);
			scan_state.current_delta_offset = ((T *)current_result_ptr)[to_scan - 1];
		} else {
			ApplyFrameOfReference<T>(current_result_ptr, scan_state.current_frame_of_reference, to_scan);
		}

		scanned += to_scan;
		scan_state.current_group_offset += to_scan;
	}
}

template <class T>
void BitpackingScan(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count, Vector &result) {
	BitpackingScanPartial<T>(segment, state, scan_count, result, 0);
}

//===--------------------------------------------------------------------===//
// Fetch
//===--------------------------------------------------------------------===//
template <class T>
void BitpackingFetchRow(ColumnSegment &segment, ColumnFetchState &state, row_t row_id, Vector &result,
                        idx_t result_idx) {
	BitpackingScanState<T> scan_state(segment);
	scan_state.Skip(segment, row_id);
	auto result_data = FlatVector::GetData<T>(result);
	T *current_result_ptr = result_data + result_idx;

	idx_t offset_in_compression_group =
	    scan_state.current_group_offset % BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE;

	data_ptr_t decompression_group_start_pointer =
	    scan_state.current_group_ptr +
	    (scan_state.current_group_offset - offset_in_compression_group) * scan_state.current_width / 8;

	//! Because FOR offsets all our values to be 0 or above, we can always skip sign extension here
	bool skip_sign_extend = true;

	if (scan_state.current_group.mode == BitpackingMode::CONSTANT) {
		*current_result_ptr = scan_state.current_constant;
		return;
	}

	if (scan_state.current_group.mode == BitpackingMode::CONSTANT_DELTA) {
		*current_result_ptr =
		    ((scan_state.current_group_offset) * scan_state.current_constant) + scan_state.current_frame_of_reference;
		return;
	}

	D_ASSERT(scan_state.current_group.mode == BitpackingMode::FOR ||
	         scan_state.current_group.mode == BitpackingMode::DELTA_FOR);

	BitpackingPrimitives::UnPackBlock<T>((data_ptr_t)scan_state.decompression_buffer, decompression_group_start_pointer,
	                                     scan_state.current_width, skip_sign_extend);

	*current_result_ptr = *(T *)(scan_state.decompression_buffer + offset_in_compression_group);
	*current_result_ptr += scan_state.current_frame_of_reference;

	if (scan_state.current_group.mode == BitpackingMode::DELTA_FOR) {
		*current_result_ptr += scan_state.current_delta_offset;
	}
}
template <class T>
void BitpackingSkip(ColumnSegment &segment, ColumnScanState &state, idx_t skip_count) {
	auto &scan_state = (BitpackingScanState<T> &)*state.scan_state;
	scan_state.Skip(segment, skip_count);
}

//===--------------------------------------------------------------------===//
// Get Function
//===--------------------------------------------------------------------===//
template <class T>
CompressionFunction GetBitpackingFunction(PhysicalType data_type) {
	return CompressionFunction(CompressionType::COMPRESSION_BITPACKING, data_type, BitpackingInitAnalyze<T>,
	                           BitpackingAnalyze<T>, BitpackingFinalAnalyze<T>, BitpackingInitCompression<T>,
	                           BitpackingCompress<T>, BitpackingFinalizeCompress<T>, BitpackingInitScan<T>,
	                           BitpackingScan<T>, BitpackingScanPartial<T>, BitpackingFetchRow<T>, BitpackingSkip<T>);
}

CompressionFunction BitpackingFun::GetFunction(PhysicalType type) {
	switch (type) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		return GetBitpackingFunction<int8_t>(type);
	case PhysicalType::INT16:
		return GetBitpackingFunction<int16_t>(type);
	case PhysicalType::INT32:
		return GetBitpackingFunction<int32_t>(type);
	case PhysicalType::INT64:
		return GetBitpackingFunction<int64_t>(type);
	case PhysicalType::UINT8:
		return GetBitpackingFunction<uint8_t>(type);
	case PhysicalType::UINT16:
		return GetBitpackingFunction<uint16_t>(type);
	case PhysicalType::UINT32:
		return GetBitpackingFunction<uint32_t>(type);
	case PhysicalType::UINT64:
		return GetBitpackingFunction<uint64_t>(type);
	default:
		throw InternalException("Unsupported type for Bitpacking");
	}
}

bool BitpackingFun::TypeIsSupported(PhysicalType type) {
	switch (type) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
	case PhysicalType::INT16:
	case PhysicalType::INT32:
	case PhysicalType::INT64:
	case PhysicalType::UINT8:
	case PhysicalType::UINT16:
	case PhysicalType::UINT32:
	case PhysicalType::UINT64:
		return true;
	default:
		return false;
	}
}

} // namespace duckdb


namespace duckdb {

constexpr uint8_t BitReader::REMAINDER_MASKS[];
constexpr uint8_t BitReader::MASKS[];

} // namespace duckdb









namespace duckdb {

template <class T>
CompressionFunction GetChimpFunction(PhysicalType data_type) {
	return CompressionFunction(CompressionType::COMPRESSION_CHIMP, data_type, ChimpInitAnalyze<T>, ChimpAnalyze<T>,
	                           ChimpFinalAnalyze<T>, ChimpInitCompression<T>, ChimpCompress<T>,
	                           ChimpFinalizeCompress<T>, ChimpInitScan<T>, ChimpScan<T>, ChimpScanPartial<T>,
	                           ChimpFetchRow<T>, ChimpSkip<T>);
}

CompressionFunction ChimpCompressionFun::GetFunction(PhysicalType type) {
	switch (type) {
	case PhysicalType::FLOAT:
		return GetChimpFunction<float>(type);
	case PhysicalType::DOUBLE:
		return GetChimpFunction<double>(type);
	default:
		throw InternalException("Unsupported type for Chimp");
	}
}

bool ChimpCompressionFun::TypeIsSupported(PhysicalType type) {
	switch (type) {
	case PhysicalType::FLOAT:
	case PhysicalType::DOUBLE:
		return true;
	default:
		return false;
	}
}

} // namespace duckdb


namespace duckdb {

constexpr uint8_t ChimpConstants::Compression::LEADING_ROUND[];
constexpr uint8_t ChimpConstants::Compression::LEADING_REPRESENTATION[];

constexpr uint8_t ChimpConstants::Decompression::LEADING_REPRESENTATION[];

} // namespace duckdb


namespace duckdb {

constexpr uint8_t FlagBufferConstants::MASKS[];
constexpr uint8_t FlagBufferConstants::SHIFTS[];

} // namespace duckdb


namespace duckdb {

constexpr uint32_t LeadingZeroBufferConstants::MASKS[];
constexpr uint8_t LeadingZeroBufferConstants::SHIFTS[];

} // namespace duckdb












namespace duckdb {

// Abstract class for keeping compression state either for compression or size analysis
class DictionaryCompressionState : public CompressionState {
public:
	bool UpdateState(Vector &scan_vector, idx_t count) {
		UnifiedVectorFormat vdata;
		scan_vector.ToUnifiedFormat(count, vdata);
		auto data = (string_t *)vdata.data;
		Verify();

		for (idx_t i = 0; i < count; i++) {
			auto idx = vdata.sel->get_index(i);
			size_t string_size = 0;
			bool new_string = false;
			auto row_is_valid = vdata.validity.RowIsValid(idx);

			if (row_is_valid) {
				string_size = data[idx].GetSize();
				if (string_size >= StringUncompressed::STRING_BLOCK_LIMIT) {
					// Big strings not implemented for dictionary compression
					return false;
				}
				new_string = !LookupString(data[idx]);
			}

			bool fits = CalculateSpaceRequirements(new_string, string_size);
			if (!fits) {
				Flush();
				new_string = true;

				fits = CalculateSpaceRequirements(new_string, string_size);
				if (!fits) {
					throw InternalException("Dictionary compression could not write to new segment");
				}
			}

			if (!row_is_valid) {
				AddNull();
			} else if (new_string) {
				AddNewString(data[idx]);
			} else {
				AddLastLookup();
			}

			Verify();
		}

		return true;
	}

protected:
	// Should verify the State
	virtual void Verify() = 0;
	// Performs a lookup of str, storing the result internally
	virtual bool LookupString(string_t str) = 0;
	// Add the most recently looked up str to compression state
	virtual void AddLastLookup() = 0;
	// Add string to the state that is known to not be seen yet
	virtual void AddNewString(string_t str) = 0;
	// Add a null value to the compression state
	virtual void AddNull() = 0;
	// Needs to be called before adding a value. Will return false if a flush is required first.
	virtual bool CalculateSpaceRequirements(bool new_string, size_t string_size) = 0;
	// Flush the segment to disk if compressing or reset the counters if analyzing
	virtual void Flush(bool final = false) = 0;
};

typedef struct {
	uint32_t dict_size;
	uint32_t dict_end;
	uint32_t index_buffer_offset;
	uint32_t index_buffer_count;
	uint32_t bitpacking_width;
} dictionary_compression_header_t;

struct DictionaryCompressionStorage {
	static constexpr float MINIMUM_COMPRESSION_RATIO = 1.2;
	static constexpr uint16_t DICTIONARY_HEADER_SIZE = sizeof(dictionary_compression_header_t);
	static constexpr size_t COMPACTION_FLUSH_LIMIT = (size_t)Storage::BLOCK_SIZE / 5 * 4;

	static unique_ptr<AnalyzeState> StringInitAnalyze(ColumnData &col_data, PhysicalType type);
	static bool StringAnalyze(AnalyzeState &state_p, Vector &input, idx_t count);
	static idx_t StringFinalAnalyze(AnalyzeState &state_p);

	static unique_ptr<CompressionState> InitCompression(ColumnDataCheckpointer &checkpointer,
	                                                    unique_ptr<AnalyzeState> state);
	static void Compress(CompressionState &state_p, Vector &scan_vector, idx_t count);
	static void FinalizeCompress(CompressionState &state_p);

	static unique_ptr<SegmentScanState> StringInitScan(ColumnSegment &segment);
	template <bool ALLOW_DICT_VECTORS>
	static void StringScanPartial(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count, Vector &result,
	                              idx_t result_offset);
	static void StringScan(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count, Vector &result);
	static void StringFetchRow(ColumnSegment &segment, ColumnFetchState &state, row_t row_id, Vector &result,
	                           idx_t result_idx);

	static bool HasEnoughSpace(idx_t current_count, idx_t index_count, idx_t dict_size,
	                           bitpacking_width_t packing_width);
	static idx_t RequiredSpace(idx_t current_count, idx_t index_count, idx_t dict_size,
	                           bitpacking_width_t packing_width);

	static StringDictionaryContainer GetDictionary(ColumnSegment &segment, BufferHandle &handle);
	static void SetDictionary(ColumnSegment &segment, BufferHandle &handle, StringDictionaryContainer container);
	static string_t FetchStringFromDict(ColumnSegment &segment, StringDictionaryContainer dict, data_ptr_t baseptr,
	                                    int32_t dict_offset, uint16_t string_len);
	static uint16_t GetStringLength(uint32_t *index_buffer_ptr, sel_t index);
};

// Dictionary compression uses a combination of bitpacking and a dictionary to compress string segments. The data is
// stored across three buffers: the index buffer, the selection buffer and the dictionary. Firstly the Index buffer
// contains the offsets into the dictionary which are also used to determine the string lengths. Each value in the
// dictionary gets a single unique index in the index buffer. Secondly, the selection buffer maps the tuples to an index
// in the index buffer. The selection buffer is compressed with bitpacking. Finally, the dictionary contains simply all
// the unique strings without lenghts or null termination as we can deduce the lengths from the index buffer. The
// addition of the selection buffer is done for two reasons: firstly, to allow the scan to emit dictionary vectors by
// scanning the whole dictionary at once and then scanning the selection buffer for each emitted vector. Secondly, it
// allows for efficient bitpacking compression as the selection values should remain relatively small.
struct DictionaryCompressionCompressState : public DictionaryCompressionState {
	explicit DictionaryCompressionCompressState(ColumnDataCheckpointer &checkpointer)
	    : checkpointer(checkpointer), heap(BufferAllocator::Get(checkpointer.GetDatabase())) {
		auto &db = checkpointer.GetDatabase();
		auto &config = DBConfig::GetConfig(db);
		function = config.GetCompressionFunction(CompressionType::COMPRESSION_DICTIONARY, PhysicalType::VARCHAR);
		CreateEmptySegment(checkpointer.GetRowGroup().start);
	}

	ColumnDataCheckpointer &checkpointer;
	CompressionFunction *function;

	// State regarding current segment
	unique_ptr<ColumnSegment> current_segment;
	BufferHandle current_handle;
	StringDictionaryContainer current_dictionary;
	data_ptr_t current_end_ptr;

	// Buffers and map for current segment
	StringHeap heap;
	string_map_t<uint32_t> current_string_map;
	std::vector<uint32_t> index_buffer;
	std::vector<uint32_t> selection_buffer;

	bitpacking_width_t current_width = 0;
	bitpacking_width_t next_width = 0;

	// Result of latest LookupString call
	uint32_t latest_lookup_result;

public:
	void CreateEmptySegment(idx_t row_start) {
		auto &db = checkpointer.GetDatabase();
		auto &type = checkpointer.GetType();
		auto compressed_segment = ColumnSegment::CreateTransientSegment(db, type, row_start);
		current_segment = std::move(compressed_segment);

		current_segment->function = function;

		// Reset the buffers and string map
		current_string_map.clear();
		index_buffer.clear();
		index_buffer.push_back(0); // Reserve index 0 for null strings
		selection_buffer.clear();

		current_width = 0;
		next_width = 0;

		// Reset the pointers into the current segment
		auto &buffer_manager = BufferManager::GetBufferManager(checkpointer.GetDatabase());
		current_handle = buffer_manager.Pin(current_segment->block);
		current_dictionary = DictionaryCompressionStorage::GetDictionary(*current_segment, current_handle);
		current_end_ptr = current_handle.Ptr() + current_dictionary.end;
	}

	void Verify() override {
		current_dictionary.Verify();
		D_ASSERT(current_segment->count == selection_buffer.size());
		D_ASSERT(DictionaryCompressionStorage::HasEnoughSpace(current_segment->count.load(), index_buffer.size(),
		                                                      current_dictionary.size, current_width));
		D_ASSERT(current_dictionary.end == Storage::BLOCK_SIZE);
		D_ASSERT(index_buffer.size() == current_string_map.size() + 1); // +1 is for null value
	}

	bool LookupString(string_t str) override {
		auto search = current_string_map.find(str);
		auto has_result = search != current_string_map.end();

		if (has_result) {
			latest_lookup_result = search->second;
		}
		return has_result;
	}

	void AddNewString(string_t str) override {
		UncompressedStringStorage::UpdateStringStats(current_segment->stats, str);

		// Copy string to dict
		current_dictionary.size += str.GetSize();
		auto dict_pos = current_end_ptr - current_dictionary.size;
		memcpy(dict_pos, str.GetDataUnsafe(), str.GetSize());
		current_dictionary.Verify();
		D_ASSERT(current_dictionary.end == Storage::BLOCK_SIZE);

		// Update buffers and map
		index_buffer.push_back(current_dictionary.size);
		selection_buffer.push_back(index_buffer.size() - 1);
		if (str.IsInlined()) {
			current_string_map.insert({str, index_buffer.size() - 1});
		} else {
			current_string_map.insert({heap.AddBlob(str), index_buffer.size() - 1});
		}
		DictionaryCompressionStorage::SetDictionary(*current_segment, current_handle, current_dictionary);

		current_width = next_width;
		current_segment->count++;
	}

	void AddNull() override {
		selection_buffer.push_back(0);
		current_segment->count++;
	}

	void AddLastLookup() override {
		selection_buffer.push_back(latest_lookup_result);
		current_segment->count++;
	}

	bool CalculateSpaceRequirements(bool new_string, size_t string_size) override {
		if (new_string) {
			next_width = BitpackingPrimitives::MinimumBitWidth(index_buffer.size() - 1 + new_string);
			return DictionaryCompressionStorage::HasEnoughSpace(current_segment->count.load() + 1,
			                                                    index_buffer.size() + 1,
			                                                    current_dictionary.size + string_size, next_width);
		} else {
			return DictionaryCompressionStorage::HasEnoughSpace(current_segment->count.load() + 1, index_buffer.size(),
			                                                    current_dictionary.size, current_width);
		}
	}

	void Flush(bool final = false) override {
		auto next_start = current_segment->start + current_segment->count;

		auto segment_size = Finalize();
		auto &state = checkpointer.GetCheckpointState();
		state.FlushSegment(std::move(current_segment), segment_size);

		if (!final) {
			CreateEmptySegment(next_start);
		}
	}

	idx_t Finalize() {
		auto &buffer_manager = BufferManager::GetBufferManager(checkpointer.GetDatabase());
		auto handle = buffer_manager.Pin(current_segment->block);
		D_ASSERT(current_dictionary.end == Storage::BLOCK_SIZE);

		// calculate sizes
		auto compressed_selection_buffer_size =
		    BitpackingPrimitives::GetRequiredSize(current_segment->count, current_width);
		auto index_buffer_size = index_buffer.size() * sizeof(uint32_t);
		auto total_size = DictionaryCompressionStorage::DICTIONARY_HEADER_SIZE + compressed_selection_buffer_size +
		                  index_buffer_size + current_dictionary.size;

		// calculate ptr and offsets
		auto base_ptr = handle.Ptr();
		auto header_ptr = (dictionary_compression_header_t *)base_ptr;
		auto compressed_selection_buffer_offset = DictionaryCompressionStorage::DICTIONARY_HEADER_SIZE;
		auto index_buffer_offset = compressed_selection_buffer_offset + compressed_selection_buffer_size;

		// Write compressed selection buffer
		BitpackingPrimitives::PackBuffer<sel_t, false>(base_ptr + compressed_selection_buffer_offset,
		                                               (sel_t *)(selection_buffer.data()), current_segment->count,
		                                               current_width);

		// Write the index buffer
		memcpy(base_ptr + index_buffer_offset, index_buffer.data(), index_buffer_size);

		// Store sizes and offsets in segment header
		Store<uint32_t>(index_buffer_offset, (data_ptr_t)&header_ptr->index_buffer_offset);
		Store<uint32_t>(index_buffer.size(), (data_ptr_t)&header_ptr->index_buffer_count);
		Store<uint32_t>((uint32_t)current_width, (data_ptr_t)&header_ptr->bitpacking_width);

		D_ASSERT(current_width == BitpackingPrimitives::MinimumBitWidth(index_buffer.size() - 1));
		D_ASSERT(DictionaryCompressionStorage::HasEnoughSpace(current_segment->count, index_buffer.size(),
		                                                      current_dictionary.size, current_width));
		D_ASSERT((uint64_t)*max_element(std::begin(selection_buffer), std::end(selection_buffer)) ==
		         index_buffer.size() - 1);

		if (total_size >= DictionaryCompressionStorage::COMPACTION_FLUSH_LIMIT) {
			// the block is full enough, don't bother moving around the dictionary
			return Storage::BLOCK_SIZE;
		}
		// the block has space left: figure out how much space we can save
		auto move_amount = Storage::BLOCK_SIZE - total_size;
		// move the dictionary so it lines up exactly with the offsets
		auto new_dictionary_offset = index_buffer_offset + index_buffer_size;
		memmove(base_ptr + new_dictionary_offset, base_ptr + current_dictionary.end - current_dictionary.size,
		        current_dictionary.size);
		current_dictionary.end -= move_amount;
		D_ASSERT(current_dictionary.end == total_size);
		// write the new dictionary (with the updated "end")
		DictionaryCompressionStorage::SetDictionary(*current_segment, handle, current_dictionary);
		return total_size;
	}
};

//===--------------------------------------------------------------------===//
// Analyze
//===--------------------------------------------------------------------===//
struct DictionaryAnalyzeState : public DictionaryCompressionState {
	DictionaryAnalyzeState()
	    : segment_count(0), current_tuple_count(0), current_unique_count(0), current_dict_size(0), current_width(0),
	      next_width(0) {
	}

	size_t segment_count;
	idx_t current_tuple_count;
	idx_t current_unique_count;
	size_t current_dict_size;
	StringHeap heap;
	string_set_t current_set;
	bitpacking_width_t current_width;
	bitpacking_width_t next_width;

	bool LookupString(string_t str) override {
		return current_set.count(str);
	}

	void AddNewString(string_t str) override {
		current_tuple_count++;
		current_unique_count++;
		current_dict_size += str.GetSize();
		if (str.IsInlined()) {
			current_set.insert(str);
		} else {
			current_set.insert(heap.AddBlob(str));
		}
		current_width = next_width;
	}

	void AddLastLookup() override {
		current_tuple_count++;
	}

	void AddNull() override {
		current_tuple_count++;
	}

	bool CalculateSpaceRequirements(bool new_string, size_t string_size) override {
		if (new_string) {
			next_width =
			    BitpackingPrimitives::MinimumBitWidth(current_unique_count + 2); // 1 for null, one for new string
			return DictionaryCompressionStorage::HasEnoughSpace(current_tuple_count + 1, current_unique_count + 1,
			                                                    current_dict_size + string_size, next_width);
		} else {
			return DictionaryCompressionStorage::HasEnoughSpace(current_tuple_count + 1, current_unique_count,
			                                                    current_dict_size, current_width);
		}
	}

	void Flush(bool final = false) override {
		segment_count++;
		current_tuple_count = 0;
		current_unique_count = 0;
		current_dict_size = 0;
		current_set.clear();
	}
	void Verify() override {};
};

struct DictionaryCompressionAnalyzeState : public AnalyzeState {
	DictionaryCompressionAnalyzeState() : analyze_state(make_unique<DictionaryAnalyzeState>()) {
	}

	unique_ptr<DictionaryAnalyzeState> analyze_state;
};

unique_ptr<AnalyzeState> DictionaryCompressionStorage::StringInitAnalyze(ColumnData &col_data, PhysicalType type) {
	return make_unique<DictionaryCompressionAnalyzeState>();
}

bool DictionaryCompressionStorage::StringAnalyze(AnalyzeState &state_p, Vector &input, idx_t count) {
	auto &state = (DictionaryCompressionAnalyzeState &)state_p;
	return state.analyze_state->UpdateState(input, count);
}

idx_t DictionaryCompressionStorage::StringFinalAnalyze(AnalyzeState &state_p) {
	auto &analyze_state = (DictionaryCompressionAnalyzeState &)state_p;
	auto &state = *analyze_state.analyze_state;

	auto width = BitpackingPrimitives::MinimumBitWidth(state.current_unique_count + 1);
	auto req_space =
	    RequiredSpace(state.current_tuple_count, state.current_unique_count, state.current_dict_size, width);

	return MINIMUM_COMPRESSION_RATIO * (state.segment_count * Storage::BLOCK_SIZE + req_space);
}

//===--------------------------------------------------------------------===//
// Compress
//===--------------------------------------------------------------------===//
unique_ptr<CompressionState> DictionaryCompressionStorage::InitCompression(ColumnDataCheckpointer &checkpointer,
                                                                           unique_ptr<AnalyzeState> state) {
	return make_unique<DictionaryCompressionCompressState>(checkpointer);
}

void DictionaryCompressionStorage::Compress(CompressionState &state_p, Vector &scan_vector, idx_t count) {
	auto &state = (DictionaryCompressionCompressState &)state_p;
	state.UpdateState(scan_vector, count);
}

void DictionaryCompressionStorage::FinalizeCompress(CompressionState &state_p) {
	auto &state = (DictionaryCompressionCompressState &)state_p;
	state.Flush(true);
}

//===--------------------------------------------------------------------===//
// Scan
//===--------------------------------------------------------------------===//
struct CompressedStringScanState : public StringScanState {
	BufferHandle handle;
	buffer_ptr<Vector> dictionary;
	bitpacking_width_t current_width;
	buffer_ptr<SelectionVector> sel_vec;
	idx_t sel_vec_size = 0;
};

unique_ptr<SegmentScanState> DictionaryCompressionStorage::StringInitScan(ColumnSegment &segment) {
	auto state = make_unique<CompressedStringScanState>();
	auto &buffer_manager = BufferManager::GetBufferManager(segment.db);
	state->handle = buffer_manager.Pin(segment.block);

	auto baseptr = state->handle.Ptr() + segment.GetBlockOffset();

	// Load header values
	auto dict = DictionaryCompressionStorage::GetDictionary(segment, state->handle);
	auto header_ptr = (dictionary_compression_header_t *)baseptr;
	auto index_buffer_offset = Load<uint32_t>((data_ptr_t)&header_ptr->index_buffer_offset);
	auto index_buffer_count = Load<uint32_t>((data_ptr_t)&header_ptr->index_buffer_count);
	state->current_width = (bitpacking_width_t)(Load<uint32_t>((data_ptr_t)&header_ptr->bitpacking_width));

	auto index_buffer_ptr = (uint32_t *)(baseptr + index_buffer_offset);

	state->dictionary = make_buffer<Vector>(segment.type, index_buffer_count);
	auto dict_child_data = FlatVector::GetData<string_t>(*(state->dictionary));

	for (uint32_t i = 0; i < index_buffer_count; i++) {
		// NOTE: the passing of dict_child_vector, will not be used, its for big strings
		uint16_t str_len = GetStringLength(index_buffer_ptr, i);
		dict_child_data[i] = FetchStringFromDict(segment, dict, baseptr, index_buffer_ptr[i], str_len);
	}

	return std::move(state);
}

//===--------------------------------------------------------------------===//
// Scan base data
//===--------------------------------------------------------------------===//
template <bool ALLOW_DICT_VECTORS>
void DictionaryCompressionStorage::StringScanPartial(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count,
                                                     Vector &result, idx_t result_offset) {
	// clear any previously locked buffers and get the primary buffer handle
	auto &scan_state = (CompressedStringScanState &)*state.scan_state;
	auto start = segment.GetRelativeIndex(state.row_index);

	auto baseptr = scan_state.handle.Ptr() + segment.GetBlockOffset();
	auto dict = DictionaryCompressionStorage::GetDictionary(segment, scan_state.handle);

	auto header_ptr = (dictionary_compression_header_t *)baseptr;
	auto index_buffer_offset = Load<uint32_t>((data_ptr_t)&header_ptr->index_buffer_offset);
	auto index_buffer_ptr = (uint32_t *)(baseptr + index_buffer_offset);

	auto base_data = (data_ptr_t)(baseptr + DICTIONARY_HEADER_SIZE);
	auto result_data = FlatVector::GetData<string_t>(result);

	if (!ALLOW_DICT_VECTORS || scan_count != STANDARD_VECTOR_SIZE ||
	    start % BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE != 0) {
		// Emit regular vector

		// Handling non-bitpacking-group-aligned start values;
		idx_t start_offset = start % BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE;

		// We will scan in blocks of BITPACKING_ALGORITHM_GROUP_SIZE, so we may scan some extra values.
		idx_t decompress_count = BitpackingPrimitives::RoundUpToAlgorithmGroupSize(scan_count + start_offset);

		// Create a decompression buffer of sufficient size if we don't already have one.
		if (!scan_state.sel_vec || scan_state.sel_vec_size < decompress_count) {
			scan_state.sel_vec_size = decompress_count;
			scan_state.sel_vec = make_buffer<SelectionVector>(decompress_count);
		}

		data_ptr_t src = &base_data[((start - start_offset) * scan_state.current_width) / 8];
		sel_t *sel_vec_ptr = scan_state.sel_vec->data();

		BitpackingPrimitives::UnPackBuffer<sel_t>((data_ptr_t)sel_vec_ptr, src, decompress_count,
		                                          scan_state.current_width);

		for (idx_t i = 0; i < scan_count; i++) {
			// Lookup dict offset in index buffer
			auto string_number = scan_state.sel_vec->get_index(i + start_offset);
			auto dict_offset = index_buffer_ptr[string_number];
			uint16_t str_len = GetStringLength(index_buffer_ptr, string_number);
			result_data[result_offset + i] = FetchStringFromDict(segment, dict, baseptr, dict_offset, str_len);
		}

	} else {
		D_ASSERT(start % BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE == 0);
		D_ASSERT(scan_count == STANDARD_VECTOR_SIZE);
		D_ASSERT(result_offset == 0);

		idx_t decompress_count = BitpackingPrimitives::RoundUpToAlgorithmGroupSize(scan_count);

		// Create a selection vector of sufficient size if we don't already have one.
		if (!scan_state.sel_vec || scan_state.sel_vec_size < decompress_count) {
			scan_state.sel_vec_size = decompress_count;
			scan_state.sel_vec = make_buffer<SelectionVector>(decompress_count);
		}

		// Scanning 1024 values, emitting a dict vector
		data_ptr_t dst = (data_ptr_t)(scan_state.sel_vec->data());
		data_ptr_t src = (data_ptr_t)&base_data[(start * scan_state.current_width) / 8];

		BitpackingPrimitives::UnPackBuffer<sel_t>(dst, src, scan_count, scan_state.current_width);

		result.Slice(*(scan_state.dictionary), *scan_state.sel_vec, scan_count);
	}
}

void DictionaryCompressionStorage::StringScan(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count,
                                              Vector &result) {
	StringScanPartial<true>(segment, state, scan_count, result, 0);
}

//===--------------------------------------------------------------------===//
// Fetch
//===--------------------------------------------------------------------===//
void DictionaryCompressionStorage::StringFetchRow(ColumnSegment &segment, ColumnFetchState &state, row_t row_id,
                                                  Vector &result, idx_t result_idx) {
	// fetch a single row from the string segment
	// first pin the main buffer if it is not already pinned
	auto &handle = state.GetOrInsertHandle(segment);

	auto baseptr = handle.Ptr() + segment.GetBlockOffset();
	auto header_ptr = (dictionary_compression_header_t *)baseptr;
	auto dict = DictionaryCompressionStorage::GetDictionary(segment, handle);
	auto index_buffer_offset = Load<uint32_t>((data_ptr_t)&header_ptr->index_buffer_offset);
	auto width = (bitpacking_width_t)(Load<uint32_t>((data_ptr_t)&header_ptr->bitpacking_width));
	auto index_buffer_ptr = (uint32_t *)(baseptr + index_buffer_offset);
	auto base_data = (data_ptr_t)(baseptr + DICTIONARY_HEADER_SIZE);
	auto result_data = FlatVector::GetData<string_t>(result);

	// Handling non-bitpacking-group-aligned start values;
	idx_t start_offset = row_id % BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE;

	// Decompress part of selection buffer we need for this value.
	sel_t decompression_buffer[BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE];
	data_ptr_t src = (data_ptr_t)&base_data[((row_id - start_offset) * width) / 8];
	BitpackingPrimitives::UnPackBuffer<sel_t>((data_ptr_t)decompression_buffer, src,
	                                          BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE, width);

	auto selection_value = decompression_buffer[start_offset];
	auto dict_offset = index_buffer_ptr[selection_value];
	uint16_t str_len = GetStringLength(index_buffer_ptr, selection_value);

	result_data[result_idx] = FetchStringFromDict(segment, dict, baseptr, dict_offset, str_len);
}

//===--------------------------------------------------------------------===//
// Helper Functions
//===--------------------------------------------------------------------===//
bool DictionaryCompressionStorage::HasEnoughSpace(idx_t current_count, idx_t index_count, idx_t dict_size,
                                                  bitpacking_width_t packing_width) {
	return RequiredSpace(current_count, index_count, dict_size, packing_width) <= Storage::BLOCK_SIZE;
}

idx_t DictionaryCompressionStorage::RequiredSpace(idx_t current_count, idx_t index_count, idx_t dict_size,
                                                  bitpacking_width_t packing_width) {
	idx_t base_space = DICTIONARY_HEADER_SIZE + dict_size;
	idx_t string_number_space = BitpackingPrimitives::GetRequiredSize(current_count, packing_width);
	idx_t index_space = index_count * sizeof(uint32_t);

	idx_t used_space = base_space + index_space + string_number_space;

	return used_space;
}

StringDictionaryContainer DictionaryCompressionStorage::GetDictionary(ColumnSegment &segment, BufferHandle &handle) {
	auto header_ptr = (dictionary_compression_header_t *)(handle.Ptr() + segment.GetBlockOffset());
	StringDictionaryContainer container;
	container.size = Load<uint32_t>((data_ptr_t)&header_ptr->dict_size);
	container.end = Load<uint32_t>((data_ptr_t)&header_ptr->dict_end);
	return container;
}

void DictionaryCompressionStorage::SetDictionary(ColumnSegment &segment, BufferHandle &handle,
                                                 StringDictionaryContainer container) {
	auto header_ptr = (dictionary_compression_header_t *)(handle.Ptr() + segment.GetBlockOffset());
	Store<uint32_t>(container.size, (data_ptr_t)&header_ptr->dict_size);
	Store<uint32_t>(container.end, (data_ptr_t)&header_ptr->dict_end);
}

string_t DictionaryCompressionStorage::FetchStringFromDict(ColumnSegment &segment, StringDictionaryContainer dict,
                                                           data_ptr_t baseptr, int32_t dict_offset,
                                                           uint16_t string_len) {
	D_ASSERT(dict_offset >= 0 && dict_offset <= Storage::BLOCK_SIZE);

	if (dict_offset == 0) {
		return string_t(nullptr, 0);
	}
	// normal string: read string from this block
	auto dict_end = baseptr + dict.end;
	auto dict_pos = dict_end - dict_offset;

	auto str_ptr = (char *)(dict_pos);
	return string_t(str_ptr, string_len);
}

uint16_t DictionaryCompressionStorage::GetStringLength(uint32_t *index_buffer_ptr, sel_t index) {
	if (index == 0) {
		return 0;
	} else {
		return index_buffer_ptr[index] - index_buffer_ptr[index - 1];
	}
}

//===--------------------------------------------------------------------===//
// Get Function
//===--------------------------------------------------------------------===//
CompressionFunction DictionaryCompressionFun::GetFunction(PhysicalType data_type) {
	return CompressionFunction(
	    CompressionType::COMPRESSION_DICTIONARY, data_type, DictionaryCompressionStorage ::StringInitAnalyze,
	    DictionaryCompressionStorage::StringAnalyze, DictionaryCompressionStorage::StringFinalAnalyze,
	    DictionaryCompressionStorage::InitCompression, DictionaryCompressionStorage::Compress,
	    DictionaryCompressionStorage::FinalizeCompress, DictionaryCompressionStorage::StringInitScan,
	    DictionaryCompressionStorage::StringScan, DictionaryCompressionStorage::StringScanPartial<false>,
	    DictionaryCompressionStorage::StringFetchRow, UncompressedFunctions::EmptySkip);
}

bool DictionaryCompressionFun::TypeIsSupported(PhysicalType type) {
	return type == PhysicalType::VARCHAR;
}
} // namespace duckdb












namespace duckdb {

//===--------------------------------------------------------------------===//
// Analyze
//===--------------------------------------------------------------------===//
struct FixedSizeAnalyzeState : public AnalyzeState {
	FixedSizeAnalyzeState() : count(0) {
	}

	idx_t count;
};

unique_ptr<AnalyzeState> FixedSizeInitAnalyze(ColumnData &col_data, PhysicalType type) {
	return make_unique<FixedSizeAnalyzeState>();
}

bool FixedSizeAnalyze(AnalyzeState &state_p, Vector &input, idx_t count) {
	auto &state = (FixedSizeAnalyzeState &)state_p;
	state.count += count;
	return true;
}

template <class T>
idx_t FixedSizeFinalAnalyze(AnalyzeState &state_p) {
	auto &state = (FixedSizeAnalyzeState &)state_p;
	return sizeof(T) * state.count;
}

//===--------------------------------------------------------------------===//
// Compress
//===--------------------------------------------------------------------===//
struct UncompressedCompressState : public CompressionState {
	explicit UncompressedCompressState(ColumnDataCheckpointer &checkpointer);

	ColumnDataCheckpointer &checkpointer;
	unique_ptr<ColumnSegment> current_segment;
	ColumnAppendState append_state;

	virtual void CreateEmptySegment(idx_t row_start);
	void FlushSegment(idx_t segment_size);
	void Finalize(idx_t segment_size);
};

UncompressedCompressState::UncompressedCompressState(ColumnDataCheckpointer &checkpointer)
    : checkpointer(checkpointer) {
	UncompressedCompressState::CreateEmptySegment(checkpointer.GetRowGroup().start);
}

void UncompressedCompressState::CreateEmptySegment(idx_t row_start) {
	auto &db = checkpointer.GetDatabase();
	auto &type = checkpointer.GetType();
	auto compressed_segment = ColumnSegment::CreateTransientSegment(db, type, row_start);
	if (type.InternalType() == PhysicalType::VARCHAR) {
		auto &state = (UncompressedStringSegmentState &)*compressed_segment->GetSegmentState();
		state.overflow_writer = make_unique<WriteOverflowStringsToDisk>(checkpointer.GetColumnData().block_manager);
	}
	current_segment = std::move(compressed_segment);
	current_segment->InitializeAppend(append_state);
}

void UncompressedCompressState::FlushSegment(idx_t segment_size) {
	auto &state = checkpointer.GetCheckpointState();
	state.FlushSegment(std::move(current_segment), segment_size);
}

void UncompressedCompressState::Finalize(idx_t segment_size) {
	FlushSegment(segment_size);
	current_segment.reset();
}

unique_ptr<CompressionState> UncompressedFunctions::InitCompression(ColumnDataCheckpointer &checkpointer,
                                                                    unique_ptr<AnalyzeState> state) {
	return make_unique<UncompressedCompressState>(checkpointer);
}

void UncompressedFunctions::Compress(CompressionState &state_p, Vector &data, idx_t count) {
	auto &state = (UncompressedCompressState &)state_p;
	UnifiedVectorFormat vdata;
	data.ToUnifiedFormat(count, vdata);

	idx_t offset = 0;
	while (count > 0) {
		idx_t appended = state.current_segment->Append(state.append_state, vdata, offset, count);
		if (appended == count) {
			// appended everything: finished
			return;
		}
		auto next_start = state.current_segment->start + state.current_segment->count;
		// the segment is full: flush it to disk
		state.FlushSegment(state.current_segment->FinalizeAppend(state.append_state));

		// now create a new segment and continue appending
		state.CreateEmptySegment(next_start);
		offset += appended;
		count -= appended;
	}
}

void UncompressedFunctions::FinalizeCompress(CompressionState &state_p) {
	auto &state = (UncompressedCompressState &)state_p;
	state.Finalize(state.current_segment->FinalizeAppend(state.append_state));
}

//===--------------------------------------------------------------------===//
// Scan
//===--------------------------------------------------------------------===//
struct FixedSizeScanState : public SegmentScanState {
	BufferHandle handle;
};

unique_ptr<SegmentScanState> FixedSizeInitScan(ColumnSegment &segment) {
	auto result = make_unique<FixedSizeScanState>();
	auto &buffer_manager = BufferManager::GetBufferManager(segment.db);
	result->handle = buffer_manager.Pin(segment.block);
	return std::move(result);
}

//===--------------------------------------------------------------------===//
// Scan base data
//===--------------------------------------------------------------------===//
template <class T>
void FixedSizeScanPartial(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count, Vector &result,
                          idx_t result_offset) {
	auto &scan_state = (FixedSizeScanState &)*state.scan_state;
	auto start = segment.GetRelativeIndex(state.row_index);

	auto data = scan_state.handle.Ptr() + segment.GetBlockOffset();
	auto source_data = data + start * sizeof(T);

	// copy the data from the base table
	result.SetVectorType(VectorType::FLAT_VECTOR);
	memcpy(FlatVector::GetData(result) + result_offset * sizeof(T), source_data, scan_count * sizeof(T));
}

template <class T>
void FixedSizeScan(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count, Vector &result) {
	auto &scan_state = (FixedSizeScanState &)*state.scan_state;
	auto start = segment.GetRelativeIndex(state.row_index);

	auto data = scan_state.handle.Ptr() + segment.GetBlockOffset();
	auto source_data = data + start * sizeof(T);

	result.SetVectorType(VectorType::FLAT_VECTOR);
	if (std::is_same<T, list_entry_t>()) {
		// list columns are modified in-place during the scans to correct the offsets
		// so we can't do a zero-copy there
		memcpy(FlatVector::GetData(result), source_data, scan_count * sizeof(T));
	} else {
		FlatVector::SetData(result, source_data);
	}
}

//===--------------------------------------------------------------------===//
// Fetch
//===--------------------------------------------------------------------===//
template <class T>
void FixedSizeFetchRow(ColumnSegment &segment, ColumnFetchState &state, row_t row_id, Vector &result,
                       idx_t result_idx) {
	auto &buffer_manager = BufferManager::GetBufferManager(segment.db);
	auto handle = buffer_manager.Pin(segment.block);

	// first fetch the data from the base table
	auto data_ptr = handle.Ptr() + segment.GetBlockOffset() + row_id * sizeof(T);

	memcpy(FlatVector::GetData(result) + result_idx * sizeof(T), data_ptr, sizeof(T));
}

//===--------------------------------------------------------------------===//
// Append
//===--------------------------------------------------------------------===//
static unique_ptr<CompressionAppendState> FixedSizeInitAppend(ColumnSegment &segment) {
	auto &buffer_manager = BufferManager::GetBufferManager(segment.db);
	auto handle = buffer_manager.Pin(segment.block);
	return make_unique<CompressionAppendState>(std::move(handle));
}

template <class T>
static void AppendLoop(SegmentStatistics &stats, data_ptr_t target, idx_t target_offset, UnifiedVectorFormat &adata,
                       idx_t offset, idx_t count) {
	auto sdata = (T *)adata.data;
	auto tdata = (T *)target;
	if (!adata.validity.AllValid()) {
		for (idx_t i = 0; i < count; i++) {
			auto source_idx = adata.sel->get_index(offset + i);
			auto target_idx = target_offset + i;
			bool is_null = !adata.validity.RowIsValid(source_idx);
			if (!is_null) {
				NumericStatistics::Update<T>(stats, sdata[source_idx]);
				tdata[target_idx] = sdata[source_idx];
			} else {
				// we insert a NullValue<T> in the null gap for debuggability
				// this value should never be used or read anywhere
				tdata[target_idx] = NullValue<T>();
			}
		}
	} else {
		for (idx_t i = 0; i < count; i++) {
			auto source_idx = adata.sel->get_index(offset + i);
			auto target_idx = target_offset + i;
			NumericStatistics::Update<T>(stats, sdata[source_idx]);
			tdata[target_idx] = sdata[source_idx];
		}
	}
}

template <>
void AppendLoop<list_entry_t>(SegmentStatistics &stats, data_ptr_t target, idx_t target_offset,
                              UnifiedVectorFormat &adata, idx_t offset, idx_t count) {
	auto sdata = (list_entry_t *)adata.data;
	auto tdata = (list_entry_t *)target;
	for (idx_t i = 0; i < count; i++) {
		auto source_idx = adata.sel->get_index(offset + i);
		auto target_idx = target_offset + i;
		tdata[target_idx] = sdata[source_idx];
	}
}

template <class T>
idx_t FixedSizeAppend(CompressionAppendState &append_state, ColumnSegment &segment, SegmentStatistics &stats,
                      UnifiedVectorFormat &data, idx_t offset, idx_t count) {
	D_ASSERT(segment.GetBlockOffset() == 0);

	auto target_ptr = append_state.handle.Ptr();
	idx_t max_tuple_count = segment.SegmentSize() / sizeof(T);
	idx_t copy_count = MinValue<idx_t>(count, max_tuple_count - segment.count);

	AppendLoop<T>(stats, target_ptr, segment.count, data, offset, copy_count);
	segment.count += copy_count;
	return copy_count;
}

template <class T>
idx_t FixedSizeFinalizeAppend(ColumnSegment &segment, SegmentStatistics &stats) {
	return segment.count * sizeof(T);
}

//===--------------------------------------------------------------------===//
// Get Function
//===--------------------------------------------------------------------===//
template <class T>
CompressionFunction FixedSizeGetFunction(PhysicalType data_type) {
	return CompressionFunction(CompressionType::COMPRESSION_UNCOMPRESSED, data_type, FixedSizeInitAnalyze,
	                           FixedSizeAnalyze, FixedSizeFinalAnalyze<T>, UncompressedFunctions::InitCompression,
	                           UncompressedFunctions::Compress, UncompressedFunctions::FinalizeCompress,
	                           FixedSizeInitScan, FixedSizeScan<T>, FixedSizeScanPartial<T>, FixedSizeFetchRow<T>,
	                           UncompressedFunctions::EmptySkip, nullptr, FixedSizeInitAppend, FixedSizeAppend<T>,
	                           FixedSizeFinalizeAppend<T>, nullptr);
}

CompressionFunction FixedSizeUncompressed::GetFunction(PhysicalType data_type) {
	switch (data_type) {
	case PhysicalType::BOOL:
	case PhysicalType::INT8:
		return FixedSizeGetFunction<int8_t>(data_type);
	case PhysicalType::INT16:
		return FixedSizeGetFunction<int16_t>(data_type);
	case PhysicalType::INT32:
		return FixedSizeGetFunction<int32_t>(data_type);
	case PhysicalType::INT64:
		return FixedSizeGetFunction<int64_t>(data_type);
	case PhysicalType::UINT8:
		return FixedSizeGetFunction<uint8_t>(data_type);
	case PhysicalType::UINT16:
		return FixedSizeGetFunction<uint16_t>(data_type);
	case PhysicalType::UINT32:
		return FixedSizeGetFunction<uint32_t>(data_type);
	case PhysicalType::UINT64:
		return FixedSizeGetFunction<uint64_t>(data_type);
	case PhysicalType::INT128:
		return FixedSizeGetFunction<hugeint_t>(data_type);
	case PhysicalType::FLOAT:
		return FixedSizeGetFunction<float>(data_type);
	case PhysicalType::DOUBLE:
		return FixedSizeGetFunction<double>(data_type);
	case PhysicalType::INTERVAL:
		return FixedSizeGetFunction<interval_t>(data_type);
	case PhysicalType::LIST:
		return FixedSizeGetFunction<list_entry_t>(data_type);
	default:
		throw InternalException("Unsupported type for FixedSizeUncompressed::GetFunction");
	}
}

} // namespace duckdb
