// For each terminal identify the possible fraudulent transactions. The fraudulent transactions are those whose import is higher or lower than 10% of the average import of the transactions executed on the same terminal in the previous semester
with date().month as currentMonth
with
     CASE
          // we are in July-December, it's second semester
          WHEN currentMonth >= 7  THEN [1,2]
          // we are in January-June, it's first semester
          WHEN currentMonth <= 6 THEN [2,1]
     END AS semesters    // [previusSemester, currentSemester]
match (n)-[r:TX{semester:semesters[0]}]->(te:Terminal)
with te, avg(r.amount) as avg, semesters
with avg+avg/10 as upBound, avg-avg/10 as downBound, te, avg, semesters
match (c:Customer)-[tr:TX]->(te2:Terminal)
where tr.semester = semesters[1] and te2.terminalId = te.terminalId and (tr.amount > upBound or tr.amount < downBound)
set tr.isFraud = true
return te.terminalId as Terminal, collect(tr.txId) as fraudolentTXS

// reset di isFraud
match (n)-[r:TX{isFraud:true}]->(m)
set r.isFraud = false
return count(r)