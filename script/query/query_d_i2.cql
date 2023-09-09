// The kind of products that have been bought through the transaction {high-tech, food, clothing, consumable, other}

:auto match (n)-[r:TX]-(m)
CALL {
    with r
    with toInteger(rand()*5) as X, r
    with 
        CASE
            WHEN X = 0 THEN "high-tech"
            WHEN X = 1 THEN "food"
            WHEN X = 2 THEN "clothing"
            WHEN X = 3 THEN "consumable"
            WHEN X = 4 THEN "other"
        END AS type, r
    SET r += {kind: type}
} IN TRANSACTIONS