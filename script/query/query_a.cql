// For each customer identifies the amount that he/she has spent for every week of the current semester
with date().month as currentMonth
with
    CASE
        // we are in July-December, it's second semester
        WHEN currentMonth >= 7  THEN 2
        // we are in January-June, it's first semester
        WHEN currentMonth <= 6 THEN 1
    END AS currentSemester
match(c:Customer)-[r:TX]->(n)
where r.semester = currentSemester
return c.customerId as customer, date.truncate("week", r.date) as startingDayOfWeek, sum(r.amount) as totalOfTheWeek
ORDER BY startingDayOfWeek