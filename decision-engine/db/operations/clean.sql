-- this file is for clean the db data
-- i already did in db it just reference
begin transaction;
--
-- first creating a duplicate copy of the original (not use in production just for developement)

create table if not exists orders_back as select * from orders;

--
-- removing ducplicate rowsn
create or replace table orders as -- this have to done because i use duckdb and it not works simply
select * from orders qualify row_number() over ( partition by order_id ) = 1;

-- removing rows having no pincode
delete from orders
where pincode IS NULL
   OR length(CAST(pincode AS VARCHAR)) != 6;



-- update all RTO
update orders
set final_status = case
    when final_status like 'RTO' THEN 'RTO'
    when final_status like 'DEL%' then 'DELIVERED'
    else final_status
end
where final_status='rto' or final_status='delivered';

commit;
