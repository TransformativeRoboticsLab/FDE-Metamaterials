import sys

from loguru import logger
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, OperationFailure

# Configure logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add("mark_duplicates.log", rotation="10 MB", retention="1 week")


def mark_duplicates(mongo_uri: str = 'mongodb://localhost:27017',
                    db_name: str = 'metatop',
                    exp_name: str = 'extremal',
                    filter_tags: list = ['Bad'],
                    nu: float = 0.1) -> None:
    """
    Mark duplicate runs in MongoDB based on specific criteria.
    """
    client = None
    try:
        logger.info(f"Connecting to MongoDB at {mongo_uri}")
        client = MongoClient(mongo_uri)

        # Test connection
        client.admin.command('ping')

        db = client[db_name]
        collection = db['runs']

        # Query filters
        db_query = {"$and": [
            {'experiment.name': exp_name},
            {'status': 'COMPLETED'},
            {'omniboard.tags': {'$nin': filter_tags}},
            {'config.nu': {'$eq': nu}},
            {'config.single_sim': {'$eq': True}},
        ]}

        logger.debug(f"Using query filter: {db_query}")

        # Aggregation pipeline
        pipeline = [
            {"$match": db_query},
            {"$group": {
                "_id": {
                    "config_init_run_idx": "$config.init_run_idx",
                },
                "ids": {"$push": "$_id"},
                "count": {"$sum": 1}
            }},
            {"$match": {"count": {"$gt": 1}}},
            {"$sort": {"_id.config_init_run_idx": 1}}
        ]

        duplicate_groups = list(collection.aggregate(pipeline))
        logger.info(f"Found {len(duplicate_groups)} groups with duplicates")

        total_marked = 0
        for group in duplicate_groups:
            ids = group['ids']
            duplicate_ids = [id for id in ids[1:] if "DUPE" not in collection.find_one(
                {"_id": id})["omniboard"]["tags"]]
            if duplicate_ids:
                try:
                    result = collection.update_many(
                        {"_id": {"$in": duplicate_ids}},
                        {"$addToSet": {"omniboard.tags": "DUPE"}}
                    )
                    total_marked += len(duplicate_ids)
                    logger.info(
                        f"Marked {len(duplicate_ids)} documents as duplicates for group {group['_id']}")
                except OperationFailure as e:
                    logger.error(f"Failed to update documents: {e}")
            else:
                logger.info(
                    f"Ids {[i for i in ids[1:] if i not in duplicate_ids]} already have the DUPE tag")

        logger.success(
            f"Successfully marked {total_marked} documents as duplicates")

    except ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise
    except OperationFailure as e:
        logger.error(f"MongoDB operation failed: {e}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error occurred: {e}")
        raise
    finally:
        if client:
            logger.debug("Closing MongoDB connection")
            client.close()


if __name__ == "__main__":
    try:
        mark_duplicates()
    except Exception as e:
        logger.error("Script failed to complete successfully")
        sys.exit(1)
