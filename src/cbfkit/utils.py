from sympy import symbols, Matrix


def strList2SympyMatrix(str_list):
    sympy_symbols_lst = []
    for istr in str_list:
        sympy_symbol = symbols(istr)
        sympy_symbols_lst.append(sympy_symbol)
    sympy_matrix = Matrix(sympy_symbols_lst)
    return sympy_matrix
