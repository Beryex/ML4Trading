# Named feature sets

`nonlinear` trains on the columns of the set named by `model.feature_set`. `all` (the default) is
built in and means every column the build produces; any other name is the file
`feature_sets/<name>.txt` in this directory (the name is the file name without `.txt`, and `all`
is reserved). A set file lists one column name per line, exactly as `build_features` spells it;
blank lines and lines starting with `#` are ignored, as is the whitespace around a name. The order
of the lines is the order of the booster's columns. A set must not be empty or list a column
twice, and every column must be one the features have, or the fit refuses it by name. The stock
code `stock_id` is a column like any other: with `model.stock_id: categorical` (the default) the
set has to list it, and with `model.stock_id: drop` it is left out even if the set lists it.
This file is Markdown on purpose: a `.txt` file here would itself be a set.
