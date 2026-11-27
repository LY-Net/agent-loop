
import asyncio

from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import settings


async def main():

    engine = create_async_engine(
        settings.database_url,
        echo=True
    )

    try:
        async with engine.connect() as connection:

            result = await connection.exec_driver_sql(
                "SELECT version()"
            )

            print("PostgreSQL connection success!")
            print(result.scalar())

    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
