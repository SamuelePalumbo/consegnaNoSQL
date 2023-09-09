// Given a user u, determine the “co-customer-relationships CC of degree k”. A user u’ is a co-customer of u if you can determine a chain “u1-t1-u2-t2-...tk-1-uk“ such that u1=u, uk=u’, and for each 1<=i,j<=k, ui <> uj, and t1,..tk-1 are the terminals on which a transaction has been executed. Therefore, CCk(u)={u’| a chain exists between u and u’ of degree k}.
match 
    (c1:Customer)-[r1:TX]-(t:Terminal),
    (c2:Customer)-[r2:TX]-(t)
where
    c1.customerId < c2.customerId
MERGE (c1)-[:CC]-(c2)

match (c1:Customer)-[:CC*2]-(c2:Customer)
where c1.customerId < c2.customerId
return distinct c1.customerId as c1, c2.customerId as c2 order by c1, c2