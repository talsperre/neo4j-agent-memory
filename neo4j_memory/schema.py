async def ensure_schema(driver):
    statements = [
        "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
        "CREATE CONSTRAINT session_user_id IF NOT EXISTS "
        "FOR (s:Session) REQUIRE (s.user_id, s.id) IS UNIQUE",
        "CREATE CONSTRAINT fact_id IF NOT EXISTS FOR (f:Fact) REQUIRE f.id IS UNIQUE",
        "CREATE CONSTRAINT entity_user_name IF NOT EXISTS "
        "FOR (e:Entity) REQUIRE (e.user_id, e.name) IS UNIQUE",
        "CREATE INDEX fact_user_id IF NOT EXISTS FOR (f:Fact) ON (f.user_id)",
        "CREATE FULLTEXT INDEX entity_search IF NOT EXISTS "
        "FOR (e:Entity) ON EACH [e.name]",
        "CREATE VECTOR INDEX fact_embedding IF NOT EXISTS "
        "FOR (f:Fact) ON f.embedding "
        "OPTIONS {indexConfig: {`vector.dimensions`: 1536, "
        "`vector.similarity_function`: 'cosine'}}",
    ]
    async with driver.session() as session:
        for statement in statements:
            result = await session.run(statement)
            await result.consume()
