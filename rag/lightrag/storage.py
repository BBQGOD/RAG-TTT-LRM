# import asyncio
import asyncio
import html
import json
import os
# from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import dataclass
from typing import Any, Union, cast
import networkx as nx
import numpy as np
from rank_bm25 import BM25Okapi
# from nano_vectordb import NanoVectorDB

from .utils import (
    logger,
    load_json,
    write_json,
    compute_mdhash_id,
)

from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
)


@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        left_data = {k: v for k, v in data.items() if k not in self._data}
        self._data.update(left_data)
        return left_data

    async def drop(self):
        self._data = {}


# @dataclass
# class NanoVectorDBStorage(BaseVectorStorage):
#     cosine_better_than_threshold: float = 0.2

#     def __post_init__(self):
#         self._client_file_name = os.path.join(
#             self.global_config["working_dir"], f"vdb_{self.namespace}.json"
#         )
#         self._max_batch_size = self.global_config["embedding_batch_num"]
#         self._client = NanoVectorDB(
#             self.embedding_func.embedding_dim, storage_file=self._client_file_name
#         )
#         self.cosine_better_than_threshold = self.global_config.get(
#             "cosine_better_than_threshold", self.cosine_better_than_threshold
#         )

#     async def upsert(self, data: dict[str, dict]):
#         logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
#         if not len(data):
#             logger.warning("You insert an empty data to vector DB")
#             return []
#         list_data = [
#             {
#                 "__id__": k,
#                 **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
#             }
#             for k, v in data.items()
#         ]
#         contents = [v["content"] for v in data.values()]
#         batches = [
#             contents[i : i + self._max_batch_size]
#             for i in range(0, len(contents), self._max_batch_size)
#         ]

#         async def wrapped_task(batch):
#             result = await self.embedding_func(batch)
#             pbar.update(1)
#             return result

#         embedding_tasks = [wrapped_task(batch) for batch in batches]
#         pbar = tqdm_async(
#             total=len(embedding_tasks), desc="Generating embeddings", unit="batch"
#         )
#         embeddings_list = await asyncio.gather(*embedding_tasks)

#         embeddings = np.concatenate(embeddings_list)
#         if len(embeddings) == len(list_data):
#             for i, d in enumerate(list_data):
#                 d["__vector__"] = embeddings[i]
#             results = self._client.upsert(datas=list_data)
#             return results
#         else:
#             # sometimes the embedding is not returned correctly. just log it.
#             logger.error(
#                 f"embedding is not 1-1 with data, {len(embeddings)} != {len(list_data)}"
#             )

#     async def query(self, query: str, top_k=5):
#         embedding = await self.embedding_func([query])
#         embedding = embedding[0]
#         results = self._client.query(
#             query=embedding,
#             top_k=top_k,
#             better_than_threshold=self.cosine_better_than_threshold,
#         )
#         results = [
#             {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
#         ]
#         return results

#     @property
#     def client_storage(self):
#         return getattr(self._client, "_NanoVectorDB__storage")

#     async def delete_entity(self, entity_name: str):
#         try:
#             entity_id = [compute_mdhash_id(entity_name, prefix="ent-")]

#             if self._client.get(entity_id):
#                 self._client.delete(entity_id)
#                 logger.info(f"Entity {entity_name} have been deleted.")
#             else:
#                 logger.info(f"No entity found with name {entity_name}.")
#         except Exception as e:
#             logger.error(f"Error while deleting entity {entity_name}: {e}")

#     async def delete_relation(self, entity_name: str):
#         try:
#             relations = [
#                 dp
#                 for dp in self.client_storage["data"]
#                 if dp["src_id"] == entity_name or dp["tgt_id"] == entity_name
#             ]
#             ids_to_delete = [relation["__id__"] for relation in relations]

#             if ids_to_delete:
#                 self._client.delete(ids_to_delete)
#                 logger.info(
#                     f"All relations related to entity {entity_name} have been deleted."
#                 )
#             else:
#                 logger.info(f"No relations found for entity {entity_name}.")
#         except Exception as e:
#             logger.error(
#                 f"Error while deleting relations for entity {entity_name}: {e}"
#             )

#     async def index_done_callback(self):
#         self._client.save()


@dataclass
class BM25VectorDBStorage(BaseVectorStorage):
    bm25_threshold: float = 0.0  # Optional threshold for BM25 scores

    def __post_init__(self):
        self._client_file_name = os.path.join(
            self.global_config.get("working_dir", "."), f"bm25_vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config.get("embedding_batch_num", 100)
        self._documents: list[dict] = []
        self._id_to_doc: dict[str, dict] = {}
        self._bm25 = None

        # Load existing data if available
        if os.path.exists(self._client_file_name):
            with open(self._client_file_name, 'r', encoding='utf-8') as f:
                self._documents = json.load(f)
            self._id_to_doc = {doc["__id__"]: doc for doc in self._documents}
            logger.info(f"Loaded {len(self._documents)} documents from {self._client_file_name}")
        else:
            logger.info(f"No existing storage found at {self._client_file_name}, starting fresh.")

    async def _build_bm25_index(self):
        if not self._documents:
            logger.warning("No documents to build BM25 index.")
            return
        
        # 需要等待 embedding_func 的结果
        tokenized_corpus = await self._tokenize_documents(self._documents)
        if tokenized_corpus:
            self._bm25 = BM25Okapi(tokenized_corpus)
            logger.info("BM25 index has been built with existing documents.")

    async def _tokenize_documents(self, documents):
        """处理所有文档的嵌入（异步），返回 tokenized 语料库"""
        batches = [documents[i:i + self._max_batch_size] for i in range(0, len(documents), self._max_batch_size)]
        
        async def process_batch(batch):
            contents = [doc["content"].lower() for doc in batch]
            embeddings = await self.embedding_func(contents)
            return [embedding for embedding in embeddings]

        tasks = [process_batch(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*tasks)

        return [embedding for batch in embeddings_list for embedding in batch]

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} documents to {self.namespace}")
        if not data:
            logger.warning("You are inserting an empty data to vector DB")
            return []

        new_documents = []
        for doc_id, doc_content in data.items():
            if "content" not in doc_content:
                logger.error(f"Document with id {doc_id} is missing 'content' field.")
                continue

            # Prepare the document
            doc = {
                "__id__": doc_id,
                "content": doc_content["content"]
            }

            # Include meta fields
            for field in self.meta_fields:
                if field in doc_content:
                    doc[field] = doc_content[field]

            new_documents.append(doc)

        # Update the storage and ID mapping
        for doc in new_documents:
            self._documents.append(doc)
            self._id_to_doc[doc["__id__"]] = doc

        # Tokenize the updated corpus (use async)
        tokenized_corpus = await self._tokenize_documents(self._documents)

        # Rebuild BM25 index
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index has been rebuilt after upsert.")

        # Save the updated documents to the storage file
        await self._save_storage()

        return [doc["__id__"] for doc in new_documents]

    async def query(self, query: str, top_k: int = 5) -> list[dict]:
        if not self._bm25:
            logger.error("BM25 index is not initialized.")
            return []

        # Tokenize query (use async)
        tokenized_query = await self.embedding_func([query.lower()])
        scores = self._bm25.get_scores(tokenized_query[0])  # We expect a list, so access the first element
        ranked_doc_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        count = 0
        for idx in ranked_doc_indices:
            score = scores[idx]
            if score < self.bm25_threshold:
                continue  # Skip documents below the threshold
            doc = self._documents[idx]
            result = {
                "id": doc["__id__"],
                "content": doc["content"],
                "score": score
            }
            # Include meta fields if any
            for field in self.meta_fields:
                if field in doc:
                    result[field] = doc[field]
            results.append(result)
            count += 1
            if count >= top_k:
                break

        logger.info(f"Query '{query}' returned {len(results)} results.")
        return results

    async def _save_storage(self):
        try:
            with open(self._client_file_name, 'w', encoding='utf-8') as f:
                json.dump(self._documents, f, ensure_ascii=False, indent=2)
            logger.info(f"Storage has been saved to {self._client_file_name}")
        except Exception as e:
            logger.error(f"Failed to save storage: {e}")

    async def delete_entity(self, entity_name: str):
        try:
            initial_count = len(self._documents)
            self._documents = [doc for doc in self._documents if doc["__id__"] != entity_name]
            removed_count = initial_count - len(self._documents)
            if removed_count > 0:
                del self._id_to_doc[entity_name]
                await self._build_bm25_index()
                await self._save_storage()
                logger.info(f"Entity {entity_name} has been deleted.")
            else:
                logger.info(f"No entity found with name {entity_name}.")
        except Exception as e:
            logger.error(f"Error while deleting entity {entity_name}: {e}")

    async def delete_relation(self, entity_name: str):
        try:
            # Assuming relations are stored with 'src_id' and 'tgt_id' fields
            initial_count = len(self._documents)
            self._documents = [
                doc for doc in self._documents
                if doc.get("src_id") != entity_name and doc.get("tgt_id") != entity_name
            ]
            removed_count = initial_count - len(self._documents)
            if removed_count > 0:
                # Rebuild the ID mapping
                self._id_to_doc = {doc["__id__"]: doc for doc in self._documents}
                await self._build_bm25_index()
                await self._save_storage()
                logger.info(f"All relations related to entity {entity_name} have been deleted.")
            else:
                logger.info(f"No relations found for entity {entity_name}.")
        except Exception as e:
            logger.error(f"Error while deleting relations for entity {entity_name}: {e}")

    async def index_done_callback(self):
        await self._save_storage()


# @dataclass
# class NetworkXStorage(BaseGraphStorage):
#     @staticmethod
#     def load_nx_graph(file_name) -> nx.Graph:
#         if os.path.exists(file_name):
#             return nx.read_graphml(file_name)
#         return None

#     @staticmethod
#     def write_nx_graph(graph: nx.Graph, file_name):
#         logger.info(
#             f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
#         )
#         nx.write_graphml(graph, file_name)

#     @staticmethod
#     def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
#         """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
#         Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
#         """
#         from graspologic.utils import largest_connected_component

#         graph = graph.copy()
#         graph = cast(nx.Graph, largest_connected_component(graph))
#         node_mapping = {
#             node: html.unescape(node.upper().strip()) for node in graph.nodes()
#         }  # type: ignore
#         graph = nx.relabel_nodes(graph, node_mapping)
#         return NetworkXStorage._stabilize_graph(graph)

#     @staticmethod
#     def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
#         """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
#         Ensure an undirected graph with the same relationships will always be read the same way.
#         """
#         fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

#         sorted_nodes = graph.nodes(data=True)
#         sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

#         fixed_graph.add_nodes_from(sorted_nodes)
#         edges = list(graph.edges(data=True))

#         if not graph.is_directed():

#             def _sort_source_target(edge):
#                 source, target, edge_data = edge
#                 if source > target:
#                     temp = source
#                     source = target
#                     target = temp
#                 return source, target, edge_data

#             edges = [_sort_source_target(edge) for edge in edges]

#         def _get_edge_key(source: Any, target: Any) -> str:
#             return f"{source} -> {target}"

#         edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

#         fixed_graph.add_edges_from(edges)
#         return fixed_graph

#     def __post_init__(self):
#         self._graphml_xml_file = os.path.join(
#             self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
#         )
#         preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
#         if preloaded_graph is not None:
#             logger.info(
#                 f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
#             )
#         self._graph = preloaded_graph or nx.Graph()
#         self._node_embed_algorithms = {
#             "node2vec": self._node2vec_embed,
#         }

#     async def index_done_callback(self):
#         NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

#     async def has_node(self, node_id: str) -> bool:
#         return self._graph.has_node(node_id)

#     async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
#         return self._graph.has_edge(source_node_id, target_node_id)

#     async def get_node(self, node_id: str) -> Union[dict, None]:
#         return self._graph.nodes.get(node_id)

#     async def node_degree(self, node_id: str) -> int:
#         return self._graph.degree(node_id)

#     async def edge_degree(self, src_id: str, tgt_id: str) -> int:
#         return self._graph.degree(src_id) + self._graph.degree(tgt_id)

#     async def get_edge(
#         self, source_node_id: str, target_node_id: str
#     ) -> Union[dict, None]:
#         return self._graph.edges.get((source_node_id, target_node_id))

#     async def get_node_edges(self, source_node_id: str):
#         if self._graph.has_node(source_node_id):
#             return list(self._graph.edges(source_node_id))
#         return None

#     async def upsert_node(self, node_id: str, node_data: dict[str, str]):
#         self._graph.add_node(node_id, **node_data)

#     async def upsert_edge(
#         self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
#     ):
#         self._graph.add_edge(source_node_id, target_node_id, **edge_data)

#     async def delete_node(self, node_id: str):
#         """
#         Delete a node from the graph based on the specified node_id.

#         :param node_id: The node_id to delete
#         """
#         if self._graph.has_node(node_id):
#             self._graph.remove_node(node_id)
#             logger.info(f"Node {node_id} deleted from the graph.")
#         else:
#             logger.warning(f"Node {node_id} not found in the graph for deletion.")

#     async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
#         if algorithm not in self._node_embed_algorithms:
#             raise ValueError(f"Node embedding algorithm {algorithm} not supported")
#         return await self._node_embed_algorithms[algorithm]()

#     # @TODO: NOT USED
#     async def _node2vec_embed(self):
#         from graspologic import embed

#         embeddings, nodes = embed.node2vec_embed(
#             self._graph,
#             **self.global_config["node2vec_params"],
#         )

#         nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
#         return embeddings, nodes_ids
