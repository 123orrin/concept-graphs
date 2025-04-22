import torch
import torch.nn.functional as F

from rclpy.node import Node
from geometry_msgs.msg import Point
from lsy_interfaces.srv import ConceptGraphQuery

from conceptgraph.slam.slam_classes import DetectionList


class QueryServiceProvider(Node):
    """ Node which hosts a service for querying a list of objects (DetectionList)
    """
    def __init__(self, model, tokenizer):
        super().__init__('query_node')
        self.query_service = self.create_service(ConceptGraphQuery, 'conceptgraph_query_service', self.query_callback)

        self.clip_model = model
        self.clip_tokenizer = tokenizer
        self.objects = None

    def query_callback(self, request, response):
        if not self.objects:
            response.object_center = Point()
            return
        text_query = request.query
        text_queries = [text_query]
        
        text_queries_tokenized = self.clip_tokenizer(text_queries).to("cuda")
        text_query_ft = self.clip_model.encode_text(text_queries_tokenized)
        text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
        text_query_ft = text_query_ft.squeeze()
        
        # similarities = objects.compute_similarities(text_query_ft)
        objects_clip_fts = self.objects.get_stacked_values_torch("clip_ft")
        objects_clip_fts = objects_clip_fts.to("cuda")
        similarities = F.cosine_similarity(
            text_query_ft.unsqueeze(0), objects_clip_fts, dim=-1
        )
        probs = F.softmax(similarities, dim=0)
        max_prob_idx = torch.argmax(probs)

        max_prob_object = self.objects[max_prob_idx]
        center = max_prob_object["bbox"].center
        print(f"Most probable object is at index {max_prob_idx} with class name '{max_prob_object['class_name']}'")
        print(f"location xyz: {center}")

        object_center = Point()
        object_center.x, object_center.y, object_center.z = center
        response.object_center = object_center
        return response
    
    def attach_objects(self, objects : DetectionList):
        self.objects = objects