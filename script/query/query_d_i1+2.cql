// d_i1 + d_i2
// 

:auto match (n)-[r:TX]-(m)
CALL {
    with r
    with r.time.hour as H, r
    with 
        CASE
            WHEN H >= 7 and H <= 12  THEN "morning"
            WHEN H >= 13 and H <= 18  THEN "afternoon"
            WHEN H >= 19 or H = 0 THEN "evening"
            WHEN H >= 1 and H <= 6  THEN "night"
        END AS dayPeriod, r
    with apoc.text.random(1, '01234') as X, dayPeriod, r
    with 
        CASE
            WHEN X = '0' THEN "high-tech"
            WHEN X = '1' THEN "food"
            WHEN X = '2' THEN "clothing"
            WHEN X = '3' THEN "consumable"
            WHEN X = '4' THEN "other"
        END AS type, r, dayPeriod
    SET r += {kind: type, period: dayPeriod}
} IN TRANSACTIONS OF 16000 ROWS

// remove props
:auto match()-[r:TX]->()
CALL{
    with r
    remove r.kind, r.period
} IN TRANSACTIONS OF 30000 ROWS
