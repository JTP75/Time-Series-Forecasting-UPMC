function th = normalEqn(x,y)
th = pinv(x.' * x) * x.' * y;       % return normal eqn value