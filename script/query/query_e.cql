// Identify the buying-friend of degree K

match (c1:Customer)-[:BUY_FRIEND*3]-(c2:Customer)
where c1.customerId < c2.customerId
return distinct [c1.customerId, c2.customerId] as couple
order by couple[0], couple[1];