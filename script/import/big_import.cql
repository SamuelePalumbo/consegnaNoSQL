// query 1
LOAD CSV WITH HEADERS FROM "file:///big_customers.csv" AS cust
MERGE(:Customer {
  customerId: toInteger(cust.customerId), 
  x_geo: toFloat(cust.x_geo), 
  y_geo: toFloat(cust.y_geo), 
  meanAmount: toFloat(cust.meanAmount), 
  std_amount: toFloat(cust.std_amount), 
  mean_tx_per_day: toFloat(cust.mean_tx_per_day)
})

// query 2
LOAD CSV WITH HEADERS FROM "file:///big_terminals.csv" AS terms
MERGE(b:Terminal {
  terminalId: toInteger(terms.terminalId), 
  x_geo: toFloat(terms.x_geo), 
  y_geo: toFloat(terms.y_geo)
})

// query 3
:auto LOAD CSV WITH HEADERS FROM "file:///med_txs1.csv" AS txs
CALL {
    WITH txs
    MATCH (c:Customer {customerId: txs.customerId})
    MATCH (te:Terminal {terminalId: txs.terminalId})
    MERGE ((c) -[:TX {txId: toInteger(txs.txId), amount: toFloat(txs.amount), date: date(txs.date), time: time(txs.time), semester: toInteger(txs.semester), isFraud: toBoolean(txs.isFraud)}] -> (te))
} IN TRANSACTIONS OF 1500 ROWS