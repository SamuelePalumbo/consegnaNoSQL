// Customers that make more than three transactions related to the same types of products from the same terminal should be connected as “buying_friends”.

CREATE INDEX txKindIndex
FOR ()-[r:TX]-()
ON (r.kind);

:auto UNWIND ["high-tech","food","clothing","consumable","other"] as category
CALL{
    with category
    match (c:Customer)-[r:TX {kind:category}]-(t:Terminal)
    with c.customerId as customer, t.terminalId as terminal, count(r) as numR
    where numR > 3
    with terminal, collect( distinct customer) as customers
    UNWIND apoc.coll.combinations(customers, 2) as couple
    CALL{
        with couple
        WITH couple[0] as aId, couple[1] as bId
        match (a:Customer {customerId:aId}), (b:Customer {customerId:bId})
        MERGE (a)-[:BUY_FRIEND]-(b)
    }
} IN TRANSACTIONS OF 16000 ROWS