import records
import re
from babel.numbers import parse_decimal, NumberFormatError
from lib.query import Query, agg_ops, cond_ops
from difflib import SequenceMatcher
import numpy as np

schema_re = re.compile(r'\((.+)\)')
num_re = re.compile(r'[-+]?\d*\.\d+|\d+')


class DBEngine:

    def __init__(self, fdb):
        self.db = records.Database('sqlite:///{}'.format(fdb))

    def execute_query(self, table_id, query, *args, **kwargs):
        return self.execute(table_id, query.sel_index, query.agg_index, query.conditions, *args, **kwargs)

    def execute(self, table_id, select_index, aggregation_index, conditions, lower=True, verbose = True):
        if not table_id.startswith('table'):
            table_id = 'table_{}'.format(table_id.replace('-', '_'))
        table_info = self.db.query('SELECT sql from sqlite_master WHERE tbl_name = :name', name=table_id).all()[0].sql
        schema_str = schema_re.findall(table_info)[0]
        schema = {}
        for tup in schema_str.split(', '):
            c, t = tup.split()
            schema[c] = t
        select = 'col{}'.format(select_index)
        agg = agg_ops[aggregation_index]
        if agg:
            select = '{}({})'.format(agg, select)
        where_clause = []
        where_map = {}
        for col_index, op, val in conditions:
            if lower and isinstance(val, str):
                val = val.lower()
                if verbose: print('\n val: ', val, '\n')
                val = self.best_matched_val(col_index, table_id, val)
                if verbose: print('\n best matched val: ', val,'\n')
            if schema['col{}'.format(col_index)] == 'real' and not isinstance(val, (int, float)):
                try:
                    val = float(parse_decimal(val))
                except NumberFormatError as e:
                    val = float(num_re.findall(val)[0])
            where_clause.append('col{} {} :col{}'.format(col_index, cond_ops[op], col_index))
            where_map['col{}'.format(col_index)] = val
        where_str = ''
        if where_clause:
            where_str = 'WHERE ' + ' AND '.join(where_clause)
        if verbose:
            print('\n where_str: ',where_str,'\n')
            print('\n where_map: ',where_map,'\n')
        query = 'SELECT {} AS result FROM {} {}'.format(select, table_id, where_str)
        if verbose:
            print('\n query: ', query,'\n')
        out = self.db.query(query, **where_map)
        return [o.result for o in out]

    def get_text_score(self, phrase1, phrase2):
        return SequenceMatcher(a= phrase1, b= phrase2).ratio()

    def best_matched_val(self,col_index,table_id, val):
        query = 'SELECT col{} AS result FROM {}'.format(col_index, table_id)
        out = self.db.query(query)
        vals_in_col = [o.result for o in out]
        scores = list(map(self.get_text_score, [val]*len(vals_in_col), vals_in_col))
        max_idx = np.argmax(scores)
        matched_val = vals_in_col[max_idx]
        return matched_val
