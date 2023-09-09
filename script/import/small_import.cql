// INSERIRE SEPARATAMENTE
// Le sottostanti query sono per l'import da files csv. Inserire in ordine:
//  - query 1 per i nodi Customer
//  - query 2 per i nodi Terminal
//  - query 3 per le transazioni

// query 1
LOAD CSV WITH HEADERS FROM "file:///small_customers.csv" AS cust
MERGE(a:Customer {customerId: cust.customerId, x_geo: cust.x_geo, y_geo: cust.y_geo, meanAmount: cust.meanAmount, std_amount: cust.std_amount, mean_tx_per_day: cust.mean_tx_per_day})

// query 2
LOAD CSV WITH HEADERS FROM "file:///small_terminals.csv" AS terms
MERGE(b:Terminal {terminalId: terms.terminalId, x_geo: terms.x_geo, y_geo: terms.y_geo})

// query 3
:auto LOAD CSV WITH HEADERS FROM "file:///small_txs1.csv" AS txs
CALL {
  WITH txs
  MATCH (c:Customer {customerId: txs.customerId})
  MATCH (te:Terminal {terminalId: txs.terminalId})
  MERGE ((c) -[:TX {txId: toInteger(txs.txId), amount: toFloat(txs.amount), date: date(txs.date), time: time(txs.time), semester: toInteger(txs.semester), isFraud: txs.isFraud}] -> (te))
    
} IN TRANSACTIONS OF 1500 ROWS