CREATE INDEX cust_ind
FOR (c:Customer) ON (c.customerId);

CREATE INDEX term_ind
FOR (t:Terminal) ON (t.terminalId);

CREATE INDEX txIndex
FOR ()-[r:TX]-()
ON (r.txId);