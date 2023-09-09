// Each transaction should be extended with the period of the day {morning (7-12), afternoon(13-18), evening(19-00), night(1-6)} in which the transaction has been executed

:auto match (n)-[r:TX]-(m)
CALL {
    with r
    with r, r.time.hour as H
    with 
        CASE
            WHEN H >= 7 and H <= 12  THEN "morning"
            WHEN H >= 13 and H <= 18  THEN "afternoon"
            WHEN H >= 19 or H = 0 THEN "evening"
            WHEN H >= 1 and H <= 6  THEN "night"
        END AS dayPeriod, r
    SET r += {period: dayPeriod}
} IN TRANSACTIONS OF 16000 ROWS