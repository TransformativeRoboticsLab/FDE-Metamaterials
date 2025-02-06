import os
import sys

import dotenv
from loguru import logger
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, OperationFailure

# Configure logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG"
)
logger.add("mark_duplicates.log", rotation="10 MB", retention="1 week")

# Load environment variables from .env file
if not dotenv.load_dotenv():
    logger.error("No .env file found!")
    sys.exit(1)

# Retrieve environment variables
MONGO_URI = os.getenv('LOCAL_MONGO_URI')
MONGO_DB_NAME = os.getenv('LOCAL_MONGO_DB_NAME')
MONGO_EXP_NAME = os.getenv('LOCAL_MONGO_EXP_NAME')

if not (MONGO_URI or MONGO_DB_NAME or MONGO_EXP_NAME):
    logger.error("Error loading mongodb environment variables")
    logger.error('MONGO_URI:', MONGO_URI)
    logger.error('MONGO_DB_NAME:', MONGO_DB_NAME)
    logger.error('MONGO_EXP_NAME:', MONGO_EXP_NAME)
    sys.exit(1)


def modify_tags(mongo_uri: str = MONGO_URI,
                db_name: str = MONGO_DB_NAME,
                exp_name: str = MONGO_EXP_NAME,
                tag_map: None = None,
                nu: float = 0.1,
                dry_run: bool = True) -> None:  # Added dry_run parameter

    if not isinstance(tag_map, dict):
        raise ValueError("tag_map must be a dictionary")
    else:
        logger.info(f"Using tag map: {tag_map}")

    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in tag_map.items()):
        raise ValueError("All keys and values in tag_map must be strings")

    client = None

    try:
        logger.info(
            f"{'[DRY RUN] ' if dry_run else ''}Connecting to MongoDB at {mongo_uri}")
        client = MongoClient(mongo_uri)
        client.admin.command('ping')

        logger.info(
            f"{'[DRY RUN] ' if dry_run else ''}Using database: {db_name}"
        )
        db = client[db_name]
        logger.info(
            f"{'[DRY RUN] ' if dry_run else ''}Listing collections in database: {db_name}")
        collections = db.list_collection_names()
        logger.info(
            f"{'[DRY RUN] ' if dry_run else ''}Collections: {collections}")
        collection = db['runs']

        query = {'omniboard.tags': {'$in': list(tag_map.keys())}}

        # First, let's check what would be modified
        cursor = collection.find(query)
        affected_docs = list(cursor)

        logger.info(
            f"{'[DRY RUN] ' if dry_run else ''}Found {len(affected_docs)} documents that would be modified")

        # Show sample of affected documents
        logger.debug("First the docs are:")
        for doc in affected_docs[:3]:  # Show first 3 docs as sample
            logger.debug(
                f"{'[DRY RUN] ' if dry_run else ''}Document {doc['_id']}: "
                f"Current tags: {doc.get('omniboard', {}).get('tags', [])}"
            )

        if not dry_run:
            # Your existing update logic here
            conditions = [
                {'case': {'$eq': ['$$tag', old_tag]}, 'then': new_tag}
                for old_tag, new_tag in tag_map.items()
            ]

            update_operation = [{
                '$set': {
                    'omniboard.tags': {
                        '$map': {
                            'input': '$omniboard.tags',
                            'as': 'tag',
                            'in': {
                                '$switch': {
                                    'branches': conditions,
                                    'default': '$$tag'
                                }
                            }
                        }
                    }
                }
            }]

            result = collection.update_many(query, update_operation)
            logger.info(
                f"Updated {result.modified_count} documents with tag mapping: {tag_map}")
        else:
            logger.info("[DRY RUN] No changes were made to the database")

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


# Update the main block to use dry run
if __name__ == "__main__":
    try:
        tag_map = {'Bad': 'BAD'}
        # First run with dry_run=True to check what would be modified
        # modify_tags(tag_map=tag_map, dry_run=False)

        modify_tags(tag_map=tag_map, dry_run=False)
    except Exception as e:
        logger.error("Script failed to complete successfully")
        sys.exit(1)
