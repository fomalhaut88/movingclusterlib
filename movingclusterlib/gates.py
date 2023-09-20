import re
import random


def _drop_leading_spaces(s):
    leading_spaces_counts = [
        len(line) - len(line.lstrip())
        for line in s.split('\n')
        if line.strip()
    ]
    if leading_spaces_counts:
        offset = min(leading_spaces_counts)
        s = "\n".join(
            line[offset:]
            for line in s.split('\n')
        )
    return s


class Expr:
    def __init__(self, expr, preamble=""):
        self.expr = expr
        self.preamble = preamble

    def full_code(self):
        return _drop_leading_spaces(f"""
            {self.preamble}

            __device__ int edge(const Item* item1, const Item* item2) {{
                return ({self.expr});
            }}
        """).strip()


class Func(Expr):
    def __init__(self, code):
        rand_num = random.randint(1, 1000000000)
        func_name = f"func_gate_{rand_num}"

        code = re.sub(
            r'__device__\s+int\s+main\s*\(', 
            f'__device__ int {func_name}(', 
            code, 
            flags=re.DOTALL
        )

        expr = f"{func_name}(item1, item2)"
        super().__init__(expr, preamble=code)


class And(Expr):
    def __init__(self, *args):
        expr_list = []
        preamble_list = []
        for arg in args:
            expr_list.append("(" + arg.expr + ")")
            preamble_list.append(arg.preamble)
        expr = " && ".join(expr_list)
        preamble = "\n\n".join(preamble_list)
        super().__init__(expr, preamble)


class Or(Expr):
    def __init__(self, *args):
        expr_list = []
        preamble_list = []
        for arg in args:
            expr_list.append("(" + arg.expr + ")")
            preamble_list.append(arg.preamble)
        expr = " || ".join(expr_list)
        preamble = "\n\n".join(preamble_list)
        super().__init__(expr, preamble)


class Equal(Expr):
    def __init__(self, key1, key2=None, exclude_zero=False):
        if key2 is None:
            key2 = key1
        if exclude_zero:
            expr = f"(item1->{key1} == item2->{key2}) && (item1->{key1} != 0)"
        else:
            expr = f"item1->{key1} == item2->{key2}"
        super().__init__(expr=expr)


class ArrayIntersect(Or):
    def __init__(self, arr1, arr2=None, exclude_zero=False):
        if arr2 is None:
            arr2 = arr1
        if not hasattr(arr1, '__iter__'):
            arr1 = [arr1]
        if not hasattr(arr2, '__iter__'):
            arr2 = [arr2]

        gates = []
        for key1 in arr1:
            for key2 in arr2:
                gate = Equal(key1, key2, exclude_zero=exclude_zero)
                gates.append(gate)

        super().__init__(*gates)


class TimeWindow(Expr):
    def __init__(self, window, ts_field="ts"):
        expr = f"item1->{ts_field} - item2->{ts_field} < {window}"
        super().__init__(expr)
