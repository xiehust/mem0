import logging
import os
from typing import List, Dict, Optional, Any, Union
import json
from pydantic import BaseModel

try:
    import boto3
except ImportError:
    raise ImportError("The 'boto3' library is required. Please install it using 'pip install boto3'.")

try:
    from opensearchpy import OpenSearch, RequestsHttpConnection,AWSV4SignerAuth
except ImportError:
    raise ImportError("The 'opensearchpy' library is required. Please install it using 'pip install opensearch-py'.")


from mem0.vector_stores.base import VectorStoreBase
logger = logging.getLogger(__name__)

class OutputData(BaseModel):
    id: Optional[str]
    score: Optional[float]
    payload: Optional[dict]

class AWSOpenSearch(VectorStoreBase):
    def __init__(
        self,
        host,
        collection_name,
        embedding_model_dims,
        use_iam,
        secret_arn,
    ):
        """
        Initialize AWS OpenSearch client.
        
        Args:
            host: OpenSearch domain endpoint (without https://)
            collection_name: Default index name to use
            embedding_model_dims (int): Dimensions of the embedding model.
            use_iam: Whether to use IAM authentication instead of basic auth
            secret_arn: ARN of the secret in AWS Secrets Manager containing credentials
        """
        self.host = host 
        self.region = os.environ.get("AWS_REGION")
        self.index_name = collection_name
        self.use_iam = use_iam
        
        # Get AWS credentials
        self.aws_access_key =  os.environ.get("AWS_ACCESS_KEY")
        self.aws_secret_key =  os.environ.get("AWS_SECRET_ACCESS_KEY")

        if not self.host:
            raise ValueError("OpenSearch host must be provided")
        if not self.region:
            raise ValueError("AWS region must be provided")

        # Create boto3 session with explicit credentials if provided
        if self.aws_access_key and self.aws_secret_key:
            self.session = boto3.Session(
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.region
            )
        else:
            self.session = boto3.Session(region_name=self.region)

        # Retrieve credentials from Secrets Manager
        secrets = self._get_secret(secret_arn)
        self.username = secrets.get('username', 'admin')
        self.password = secrets.get('password')

        
        if self.use_iam:
            # Use IAM authentication
            credentials = self.session.get_credentials()
            auth = AWSV4SignerAuth(credentials, self.region)
        else:
            # Use basic authentication with master user credentials
            if not self.username or not self.password:
                raise ValueError("Username and password must be provided when not using IAM authentication")
            auth = (self.username, self.password)

        # Initialize the OpenSearch client
        self.client = OpenSearch(
            hosts=[{'host': self.host, 'port': 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30
        )

        self.create_col(embedding_model_dims)

    def _get_secret(self, secret_arn: str) -> Dict[str, str]:
        """
        Retrieve secret from AWS Secrets Manager.
        
        Args:
            secret_arn: ARN of the secret
            
        Returns:
            Dictionary containing the secret values
        """
        try:
            client = self.session.client('secretsmanager')
            response = client.get_secret_value(SecretId=secret_arn)
            if 'SecretString' in response:
                return json.loads(response['SecretString'])
            raise ValueError("Secret value is not a string")
        except Exception as e:
            raise Exception(f"Failed to retrieve secret from Secrets Manager: {str(e)}")

    def create_col(self, vector_size: int, distance: str = "cosine"):
        """
        Create a new collection (index) in OpenSearch.
        
        Args:
            vector_size: Dimension of vectors
            distance: Distance metric to use (cosine, dotProduct, l2)
        """
        # Map distance metrics to OpenSearch space_type
        distance_mapping = {
            "cosine": "cosinesimil",
            "l2": "l2",
            "dotProduct": "innerproduct"
        }
        
        space_type = distance_mapping.get(distance, "cosinesimil")
        
        index_body = {
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": vector_size,
                        "method": {
                            "name": "hnsw",
                            "space_type": space_type,
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 16
                            }
                        }
                    },
                    "payload": {
                        "type": "object"
                    }
                }
            },
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 1,
                    "knn": "true",
                    "knn.algo_param.ef_search": 32
                }
            }
        }
        
        self._create_index_safe(index_name=self.index_name, body=index_body)

    def _create_index_safe(self, index_name, body):
        try:
            # Check if index exists first
            if self.client.indices.exists(index=index_name):
                logger.info(f"Index '{index_name}' already exists")
                return None
            else:
                logger.info(f"Creating index '{index_name}'")
                
            response = self.client.indices.create(
                index=index_name,
                body=body
            )
            logger.info(f"Index '{index_name}' created successfully")
            return response

        except Exception as e:
            logger.error(f"Error handling index creation: {str(e)}")
            raise
    def insert(
        self,
        vectors: List[List[float]],
        payloads: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ):
        """
        Insert vectors into a collection.
        
        Args:
            vectors: List of vectors to insert
            payloads: Optional metadata for each vector
            ids: Optional custom IDs for the vectors
        """
        if not payloads:
            payloads = [{} for _ in vectors]
        if not ids:
            ids = [None for _ in vectors]

        actions = []
        for vector, payload, vid in zip(vectors, payloads, ids):
            action = {
                "vector": vector,
                "payload": payload
            }
            
            actions.append({
                "index": {
                    "_index": self.index_name,
                    "_id": vid
                }
            })
            actions.append(action)

        # Use bulk operation with error handling
        response = self.client.bulk(body="\n".join(map(json.dumps, actions)) + "\n")
        if response.get("errors", False):
            errors = [item for item in response["items"] if "error" in item["index"]]
            raise Exception(f"Bulk insert failed with errors: {errors}")

    def search(
        self,
        query: List[float],
        limit: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[OutputData]:
        """
        Search for similar vectors using KNN search.
        
        Args:
            query: Query vector to search for similar vectors
            limit: Maximum number of results to return (default: 5)
            filters: Optional metadata filters to apply
            min_score: Minimum similarity score threshold
            return_vector: Whether to include vector data in results (default: True)
        
        Returns:
            List of search results containing id, score, vector (optional), and payload
            
        Raises:
            ValueError: If input parameters are invalid
            Exception: If search operation fails
        """
        # Construct base KNN query
        knn_query = {
            "vector": {
                "vector": query,
                "k": limit
            }
        }
        # Construct final search query
        search_query = {
            "size": limit
        }
        # Add filters if provided
        if filters:
            search_query["query"] = {
                "bool": {
                    "must": [
                        {"knn": knn_query}
                    ]
                }
            }
            
            # Handle multiple filter conditions
            filter_conditions = []
            for field, value in filters.items():
                filter_conditions.append({"term": {f"payload.{field}":value}})
            search_query["query"]["bool"]["filter"] = filter_conditions
        else:
            search_query["query"] = {"knn": knn_query}
        # Execute search
        response = self.client.search(
            body=search_query,
            index=self.index_name,
        )

        # Process results
        results = []
        for hit in response["hits"]["hits"]:
            score = hit["_score"]
                
            result = OutputData(id=hit["_id"], payload=hit["_source"]["payload"], score=score)       
            results.append(result)

        return results

    def delete(self, vector_id: str):
        """
        Delete a vector by ID.
        
        Args:
            vector_id: ID of the vector to delete
        """
        try:
            self.client.delete(index=self.index_name, id=vector_id)
        except Exception as e:
            raise Exception(f"Delete operation failed: {str(e)}")

    def update(
        self,
        vector_id: str,
        vector: Optional[List[float]] = None,
        payload: Optional[Dict] = None
    ):
        """
        Update a vector and its payload.
        
        Args:
            vector_id: ID of the vector to update
            vector: New vector values (optional)
            payload: New payload (optional)
        """
        doc = {}
        if vector is not None:
            doc["vector"] = vector
        if payload is not None:
            doc["payload"] = payload

        try:
            self.client.update(
                index=self.index_name,
                id=vector_id,
                body={"doc": doc}
            )
        except Exception as e:
            raise Exception(f"Update operation failed: {str(e)}")

    def get(self,  vector_id: str) -> OutputData:
        """
        Retrieve a vector by ID.
        
        Args:
            vector_id: ID of the vector to retrieve
            
        Returns:
            Vector data including payload
        """
        try:
            response = self.client.get(index=self.index_name, id=vector_id)
            return OutputData(id=response["_id"],score=None,payload=response["_source"]["payload"])
        except Exception as e:
            raise Exception(f"Get operation failed: {str(e)}")

    def list_cols(self) -> List[str]:
        """
        List all collections/indices.
        
        Returns:
            List of collection names
        """
        try:
            response = self.client.indices.get_alias()
            return list(response.keys())
        except Exception as e:
            raise Exception(f"Failed to list collections: {str(e)}")

    def delete_col(self):
        """
        Delete a collection/index.
        """
        try:
            self.client.indices.delete(index=self.index_name)
        except Exception as e:
            raise Exception(f"Failed to delete collection: {str(e)}")

    def col_info(self) -> Dict:
        """
        Get information about a collection/index.
        
        Args:
            name: Name of the collection
            
        Returns:
            Collection information
        """
        try:
            return self.client.indices.get(index=self.index_name)
        except Exception as e:
            raise Exception(f"Failed to get collection info: {str(e)}")

    def list(self, filters: Optional[Dict] = None, limit: Optional[int] = None) -> List[List[OutputData]]:
        """
        List all vectors in a collection.
        
        Args:
            filters: Optional filters to apply
            limit: Maximum number of results to return
            
        Returns:
            List of vectors with their payloads
        """
        query = {"query": {"match_all": {}}}
        if filters:
            must = []
            for key, value in filters.items():
                must.append({"term":{f"payload.{key}":value}})
            query["query"] = {"bool": {"must":must}}
        if limit:
            query["size"] = limit
        try:
            response = self.client.search(
                body=query,
                index=self.index_name
            )
            results = []
            for hit in response["hits"]["hits"]:
                results.append(OutputData(id=hit["_id"],score=hit["_score"], payload=hit["_source"]["payload"]))
            return [results]
        except Exception as e:
            raise Exception(f"List operation failed: {str(e)}")
