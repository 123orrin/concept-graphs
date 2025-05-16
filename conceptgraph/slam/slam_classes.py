
from collections.abc import Iterable
from enum import Enum
import copy
import matplotlib
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from scipy.special import gammaln
from scipy.stats import norm, uniform
import ros2_numpy

def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()

def to_tensor(numpy_array, device=None):
    if isinstance(numpy_array, torch.Tensor):
        return numpy_array
    if device is None:
        return torch.from_numpy(numpy_array)
    else:
        return torch.from_numpy(numpy_array).to(device)

class DetectionList(list):
    def get_values(self, key, idx:int=None):
        if idx is None:
            return [detection[key] for detection in self]
        else:
            return [detection[key][idx] for detection in self]
    
    def get_stacked_values_torch(self, key, idx:int=None):
        values = []
        for detection in self:
            v = detection[key]
            if idx is not None:
                v = v[idx]
            if isinstance(v, o3d.geometry.OrientedBoundingBox) or \
                isinstance(v, o3d.geometry.AxisAlignedBoundingBox):
                v = np.asarray(v.get_box_points())
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            values.append(v)
        return torch.stack(values, dim=0) if(len(values) > 0) else torch.empty(0)
    
    def get_stacked_values_numpy(self, key, idx:int=None):
        values = self.get_stacked_values_torch(key, idx)
        return to_numpy(values)
    
    def __add__(self, other):
        new_list = copy.deepcopy(self)
        new_list.extend(other)
        return new_list
    
    def __iadd__(self, other):
        self.extend(other)
        return self
    
    def slice_by_indices(self, index: Iterable[int]):
        '''
        Return a sublist of the current list by indexing
        '''
        new_self = type(self)()
        for i in index:
            new_append(self[i])
        return new_self
    
    def slice_by_mask(self, mask: Iterable[bool]):
        '''
        Return a sublist of the current list by masking
        '''
        new_self = type(self)()
        for i, m in enumerate(mask):
            if m:
                new_append(self[i])
        return new_self
    
    def get_most_common_class(self) -> list[int]:
        classes = []
        for d in self:
            values, counts = np.unique(np.asarray(d['class_id']), return_counts=True)
            most_common_class = values[np.argmax(counts)]
            classes.append(most_common_class)
        return classes
    
    def color_by_most_common_classes(self, obj_classes, color_bbox: bool=True):
        '''
        Color the point cloud of each detection by the most common class
        '''
        classes = self.get_most_common_class()
        for d, c in zip(self, classes):
            # color = obj_classes[str(c)]
            color = obj_classes.get_class_color(int(c))
            d['pcd'].paint_uniform_color(color)
            if color_bbox:
                d['bbox'].color = color
                
    def color_by_instance(self):
        if len(self) == 0:
            # Do nothing
            return
        
        if "inst_color" in self[0]:
            for d in self:
                d['pcd'].paint_uniform_color(d['inst_color'])
                d['bbox'].color = d['inst_color']
        else:
            cmap = matplotlib.colormaps.get_cmap("turbo")
            instance_colors = cmap(np.linspace(0, 1, len(self)))
            instance_colors = instance_colors[:, :3]
            for i in range(len(self)):
                self[i]['pcd'].paint_uniform_color(instance_colors[i])
                self[i]['bbox'].color = instance_colors[i]

    def get_point_cloud_msg(self, frame_id):
        num_points = np.sum(len(obj["pcd"].points) for obj in self if "pcd" in obj)

        structured_array = np.zeros(
            num_points,
            dtype=[
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("rgb", np.float32),
            ],
        )
        offset = 0
        for obj in self:
            if "pcd" not in obj:
                continue
            num_points_obj = len(obj["pcd"].points)
            obj_points = np.asarray(obj["pcd"].points)
            obj_colors = (np.asarray(obj["pcd"].colors) * 255).astype(np.uint32)

            structured_array["x"][offset : offset + num_points_obj] = obj_points[:, 0]
            structured_array["y"][offset : offset + num_points_obj] = obj_points[:, 1]
            structured_array["z"][offset : offset + num_points_obj] = obj_points[:, 2]
            structured_array["rgb"][offset : offset + num_points_obj] = (
                (obj_colors[:, 0] << 16)
                | (obj_colors[:, 1] << 8)
                | (obj_colors[:, 2] << 0)
            )

            offset += num_points_obj

        return ros2_numpy.point_cloud2.array_to_pointcloud2(
            structured_array, frame_id=frame_id
        )


class MapObjectList(DetectionList):
    def compute_similarities(self, new_clip_ft):
        '''
        The input feature should be of shape (D, ), a one-row vector
        This is mostly for backward compatibility
        '''
        # if it is a numpy array, make it a tensor 
        new_clip_ft = to_tensor(new_clip_ft)
        
        # assuming cosine similarity for features
        clip_fts = self.get_stacked_values_torch('clip_ft')

        similarities = F.cosine_similarity(new_clip_ft.unsqueeze(0), clip_fts)
        # return similarities.squeeze()
        return similarities
    
    def to_serializable(self):
        s_obj_list = []
        for obj in self:
            s_obj_dict = copy.deepcopy(obj)
            
            s_obj_dict['clip_ft'] = to_numpy(s_obj_dict['clip_ft'])
            # s_obj_dict['text_ft'] = to_numpy(s_obj_dict['text_ft'])
            
            s_obj_dict['pcd_np'] = np.asarray(s_obj_dict['pcd'].points)
            s_obj_dict['bbox_np'] = np.asarray(s_obj_dict['bbox'].get_box_points())
            s_obj_dict['pcd_color_np'] = np.asarray(s_obj_dict['pcd'].colors)
            
            del s_obj_dict['pcd']
            del s_obj_dict['bbox']
            
            s_obj_list.append(s_obj_dict)
            
        return s_obj_list
    
    def load_serializable(self, s_obj_list):
        assert len(self) == 0, 'MapObjectList should be empty when loading'
        for s_obj_dict in s_obj_list:
            new_obj = copy.deepcopy(s_obj_dict)
            
            new_obj['clip_ft'] = to_tensor(new_obj['clip_ft'])
            # new_obj['text_ft'] = to_tensor(new_obj['text_ft'])
            
            new_obj['pcd'] = o3d.geometry.PointCloud()
            new_obj['pcd'].points = o3d.utility.Vector3dVector(new_obj['pcd_np'])
            new_obj['bbox'] = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(new_obj['bbox_np']))
            new_obj['bbox'].color = new_obj['pcd_color_np'][0]
            new_obj['pcd'].colors = o3d.utility.Vector3dVector(new_obj['pcd_color_np'])
            
            del new_obj['pcd_np']
            del new_obj['bbox_np']
            del new_obj['pcd_color_np']
            
            append(new_obj)

class ProbabilisticMapObjectList(MapObjectList):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def expectedToObserve(self, camera_pose, intrinsics, img_height, img_width, min_depth, max_depth, visibility_threshold=0.35):
        '''
        Compute the expected objects to observe given the pose and field of view.

        Args:
            camera_pose: 4x4 numpy array representing the camera pose
            fov_x: field of view (x-axis) in radians
            fov_y: field of view (y-axis) in radians
            projection_plane [Optional: int = 1]: distance from the camera to the projection plane
        
        Returns:
            expected_object_indices: a list of indices of the expected objects to observe
            expected_object_ids: a list of ids of the expected objects to observe
        '''
        expected_object_indices = []
        expected_object_ids = []        
        
        for idx, obj in enumerate(self):
            points = np.asarray(obj['pcd'].points)
            points = np.linalg.inv(camera_pose) @ np.vstack([points.T, np.ones(points.shape[0])])
            uv = (intrinsics @ points).T
            uv[:, 0] /= uv[:, 2]
            uv[:, 1] /= uv[:, 2]

            total_points = uv.shape[0]
            valid_indices = (uv[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < img_width) & \
                        (uv[:, 1] >= 0) & (uv[:, 1] < img_height) & \
                        (uv[:, 2] >= min_depth) & (uv[:, 2] <= max_depth)
            uv = uv[valid_indices]
            expected_points = uv.shape[0]

            if expected_points / total_points > visibility_threshold:
                expected_object_indices.append(idx)
                expected_object_ids.append(obj['id'])
            
        return expected_object_indices, expected_object_ids
    
    def updateAge(self, time, ids=None):
        '''
        Update the age of the objects in the list

        Args:
            time: current time
        '''
        for obj in self:
            obj['age'] = time - obj['last_observed_time']
    
    def updateLostTime(self, time, ids=None):
        '''
        Update the time since the object was last observed

        Args:
            time: current time
        '''
        for obj in self:
            obj['lost_time'] = time - obj['last_observed_time']

    def updateProbability(self, change_list=None, std_change_list=None, ids=None, cap=10):
        '''
        Update the probability that object is in the same location

        Args:
            change: some measure of change between the observation and previous knowledge
            std_change: the standard deviation of the change
        '''
        if ids is None:
            return

        for obj in self:
            if obj['id'] not in ids:
                continue
            
            idx = ids.index(obj['id'])

            mu = obj['mu']
            sig = obj['sig']
            a = obj['a']
            b = obj['b']
            object_type = obj['type']
            eps = obj['eps']
            inlier = obj['inlier']
            change = change_list[idx]
            std_change = std_change_list[idx]

            s_weight = 1 # Higher weight means greater increase, lower decrease
            if object_type == POCDObjectTypes.DYNAMIC and not inlier:
                s_weight = 3 # Drop fast
            elif object_type == POCDObjectTypes.DYNAMIC and inlier:
                s_weight = 0 # Rise slow
            elif object_type == POCDObjectTypes.STATIC and not inlier:
                s_weight = 0 # drop slow
            elif object_type == POCDObjectTypes.STATIC and inlier:
                s_weight = 3 # rise fast
            elif object_type == POCDObjectTypes.DISSAPEARED and not inlier:
                s_weight = 5 # drop very fast
            elif object_type == POCDObjectTypes.DISSAPEARED and inlier:
                s_weight = 0 # rise slow

            object_type = min(1, object_type.value)
            obj['type'] = POCDObjectTypes(object_type)

            tolerance = 20 * std_change

            s_sq = 1 / (1 / np.square(sig) + 1 / np.square(std_change)) 
            m = s_sq * (mu / np.square(sig) + change / np.square(std_change))

            k1, k2 = self.updateKSingle(obj, s_weight)

            C1 = k1 * max(norm.pdf(change, loc=mu, scale=sig), eps)
            if abs(change) >= (tolerance - eps):
                C2 = k2 * uniform.pdf(tolerance, loc=0, scale=tolerance)
            else:
                C2 = k2 * uniform.pdf(abs(change), loc=0, scale=tolerance)
            C1 = max(eps, C1)
            C2 = max(eps, C2)
            C_norm = C1 + C2
            C1 /= C_norm
            C2 /= C_norm

            inlier = True if C1 >= C2 else False
            obj['inlier'] = inlier

            mu_prime = C1 * m + C2 * mu
            sig = np.sqrt(C1 * (s_sq + np.square(m)) + C2 * (np.square(sig) + np.square(mu)) - np.square(mu_prime))
            obj['sig'] = sig

            gamma = (a + object_type * s_weight + 1) / (a + b + s_weight + 1)
            eta = (a + object_type * s_weight) / (a + b + s_weight + 1)
            theta = C1 * gamma + C2 * eta
            alpha = ((a + object_type * s_weight + 2) * (a + object_type * s_weight + 1)) / ((a + b + s_weight + 1) * (a + b + s_weight + 2))
            beta = ((a + object_type * s_weight + 1) * (a + object_type * s_weight)) / ((a + b + s_weight + 1) * (a + b + s_weight + 2))
            # nu = C1 * alpha+ C2 * beta

            obj['mu'] = mu_prime
            theta_sq = np.square(theta)
            a = (C1*theta*alpha+ C2*beta*theta - theta_sq) / (theta_sq - C1*alpha- C2*beta)
            b = (C1*theta*alpha+ C2*beta*theta - theta_sq) * (1 - theta) / ((theta_sq - C1*alpha- C2*beta) * theta)

            if a > cap or b > cap:
                ratio = max(a, b) / cap
                a /= ratio
                b /= ratio
            obj['a'] = a
            obj['b'] = b

            confidence = a / (a + b)
            obj['pocd_confidence'] = confidence
        
    def updateKSingle(self, obj, k):
        '''
        Beta Distribution calculation in posterior stationarity update rule
        '''
        a = obj['a']
        b = obj['b']
        object_type = obj['type'].value
        eps = obj['eps']

        lk1 = (gammaln(a + b) + gammaln(a + k*object_type + 1) + gammaln(b + k - k*object_type)) \
            - (gammaln(a) + gammaln(b) + gammaln(a+b+k+1))
        lk2 = (gammaln(a + b) + gammaln(a + k*object_type) + gammaln(b + k - k*object_type + 1)) \
            - (gammaln(a) + gammaln(b) + gammaln(a+b+k+1))

        k1 = np.exp(lk1)
        k2 = np.exp(lk2)

        ks = k1 + k2
        k1 /= ks
        k2 /= ks

        k1 = max(eps, k1)
        k2 = max(eps, k2)

        return k1, k2

    def pruneObjectsByProbability(self, upper_threshold: float=1, lower_threshold: float=0):
        '''
        Prune objects by probability

        Args:
            threshold: the threshold for which an object is considered to be moved

        Returns:
            pruned_object_ind: a list of indices for objects that are pruned
            pruned_object_ids: a list of ids for objects that are pruned
        '''
        pruned_objects_inds = []
        pruned_objects_ids = []
        for idx, obj in enumerate(self):
            if lower_threshold < obj['pocd_confidence'] < upper_threshold:
                pruned_objects_inds.append(idx)
                pruned_objects_ids.append(obj['id'])
        return pruned_objects_inds, pruned_objects_ids
    
    def getValidDetections(self, ids=None):
        """
        Get the Valid detections in the based on if the detection is an inlier or outlier

        Args:
            ids: list of ids to filter the objects

        Returns:
            is_valid: list indicating if detection is valid or invalid
        """
        is_valid = []
        for obj in self:
            if (ids is not None) and (obj['id'] not in ids):
                continue
            if obj['inlier']:
                is_valid.append(True)
            else:
                is_valid.append(False)
        return is_valid

    def getChangesForExpectedObjects(self, detection_list, expected_object_indices, match_indices, default_change, default_change_std):
        """
        Get the changes of expected objects. If an object is expected but not observed, the default change is used.
        If an object is observed, the change is computed based on the observation usin gICP.

        Arguments:
            detection_list: list of detections
            expected_object_indices: list of indices of expected objects
            match_indices: list of object indices to which the corresponding detection is matched to
            default_change: default change value
            default_change_std: default change standard deviation
        Returns:
            changes: list of changes
            std_changes: list of standard deviations of changes
            transformations: list of transformations
        """
        change_list = [default_change] * len(expected_object_indices)
        std_change_list = [default_change_std] * len(expected_object_indices)
        transform_list = [np.eye(4)] * len(expected_object_indices)
        for detection_idx, object_idx in enumerate(match_indices):
            if object_idx is None:
                # Object is new. No POCD update
                continue
            if object_idx not in expected_object_indices:
                # Object is not expected. No POCD update
                continue

            index = expected_object_indices.index(object_idx)
            # Object has been observed. Evaluate change magnitude with ICP
            detection_pcd = detection_list[detection_idx]['pcd']
            object_pcd = self[object_idx]['pcd']
            registration_results = self.getICPRegistration(detection_pcd, object_pcd)
            std_change_list[index] = default_change_std
            change_list[index] = np.linalg.norm(registration_results.transformation[:3, 3])
            # std_change_list[index] = registration_results.inlier_rmse
            transform_list[index] = registration_results.transformation
            
        return change_list, std_change_list, transform_list

    def transformObjectsToDetection(self, expected_object_indices, transform_list, inds=None):
        """
        Transform objects based on the given transformations

        Args:
            transform_list: list of transformations
            inds: list of indices of objects to transform
        """
        assert inds is not None, 'Indices must be provided'
        for i in inds:
            if i not in expected_object_indices:
                continue
            tmp_idx = expected_object_indices.index(i)
            transform = transform_list[tmp_idx]
            self[i]['pcd'].transform(transform)
            
            oriented_bbox = self[i]['bbox'].get_oriented_bounding_box()
            oriented_bbox.translate(transform[:3, 3])
            oriented_bbox.rotate(transform[:3, :3])
            axis_bbox = oriented_bbox.get_axis_aligned_bounding_box()
            self[i]['bbox'] = axis_bbox

    def transformObjectsToRecentObjects(self, transform_list: list=[], inds: list=[]):
        """
        Transform objects based on the given transformations

        Args:
            transform_list: list of transformations
            inds: list of indices of objects to transform
        """
        assert inds is not None, 'Indices must be provided'
        assert len(inds) == len(transform_list), 'Transform list must be the same length as the indices list'
        for i, ind in enumerate(inds):
            if ind is None:
                continue
            transform = transform_list[i]
            pcd = self[ind]['pcd']
            pcd.transform(transform)
            self[ind]['pcd'] = pcd

            oriented_bbox = self[ind]['bbox'].get_oriented_bounding_box()
            oriented_bbox.translate(transform[:3, 3])
            oriented_bbox.rotate(transform[:3, :3])
            axis_bbox = oriented_bbox.get_axis_aligned_bounding_box()
            self[ind]['bbox'] = axis_bbox

    def mergeObjectsWithRecentObjects(self, dissapeared_inds: list=[], matched_inds: list=[]):
        """
        Merge important properties of the dissapeared objects and the matched objects

        Args:
            dissapeared_inds: list of indices of dissapeared objects
            matched_inds: list of indices of matched objects
        """
        assert len(dissapeared_inds) == len(matched_inds), 'Dissapeared and matched indices must be the same length'

        extend_attributes = ['image_idx', 'mask_idx', 'color_path', 'class_id', 'mask', 'xyxy', 'conf', 'contain_number', 'centroid_locations', 'bbox_shadow_hull_history']
        add_attributes = ['num_detections', 'num_obj_in_class']
        skip_attributes = ['id', 'class_name', 'is_background', 'new_counter', 'curr_obj_num', 'inst_color']  # 'inst_color' just keeps obj1's
        custom_handled = ['pcd', 'bbox', 'clip_ft', 'text_ft', 'n_points']

        pocd_skip_attributes = ['confidence_history', 'first_observed_time', 'pocd_confidence', 'age', 'lost_time', 'eps', 'inlier', 'type']
        pocd_mean_attributes = ['a', 'b', 'mu', 'sig', 'pocd_confidence']
        pocd_custom_attributes = ['last_observed_time', 'time_of_disappearance']

        skip_attributes += pocd_skip_attributes
        custom_handled += pocd_custom_attributes

        # Check for unhandled keys and throw an error if there are
        if len(self) > 0:
            all_handled_keys = set(extend_attributes + add_attributes + skip_attributes + custom_handled + pocd_mean_attributes)
            unhandled_keys = set(self[0].keys()) - all_handled_keys
            if unhandled_keys:
                raise ValueError(f"Unhandled keys detected in obj2: {unhandled_keys}. Please update the merge function to handle these attributes.")
        
        for d_ind, m_ind in zip(dissapeared_inds, matched_inds):
            if m_ind is None:
                continue
            # Process extend and add attributes
            for attr in extend_attributes:
                if attr in self[d_ind] and attr in self[m_ind]:
                    self[d_ind][attr].extend(self[m_ind][attr])
            
            for attr in add_attributes:
                if attr in self[d_ind] and attr in self[m_ind]:
                    self[d_ind][attr] += self[m_ind][attr]

            # Process custom
            self[d_ind]['pcd'] = self[m_ind]['pcd']
            self[d_ind]['clip_ft'] = self[m_ind]['clip_ft']
            self[d_ind]['bbox'] = self[m_ind]['bbox']
            self[d_ind]['n_points'] = self[m_ind]['n_points']
            
            self[d_ind]['last_observed_time'] = self[m_ind]['last_observed_time']
            self[d_ind]['time_of_disappearance'] = -1

            # Process mean attributes
            for attr in pocd_mean_attributes:
                if attr in self[d_ind] and attr in self[m_ind]:
                    self[d_ind][attr] = (self[d_ind][attr] + self[m_ind][attr]) / 2

        return True

    def removeObjectsByIndex(self, inds: list=[]):
        inds = sorted(inds, reverse=True)
        for i in inds:
            self.pop(i)
        return True

    def matchDissapearedObjectsToRecentObjects(self, look_back_time: int=10, look_forward_time: int=10, inds: list=[], threshold: float=0.7):
        """
        Matches dissapeared objects to the objects that were instatiated near its dissapearence. This is able to match objects that "dissapeared" but were actually just moved nearby AND are still visible in the same frame. 

        Arguments:
            inds: list of indices of objects to match
        Returns:
            matches: list of indices which match the disappeared objects
            transforms: list of transformations from the dissapeared object to the matched object
        """

        matches = []
        transforms = []
        for missing_object_ind in inds:
            potential_match_inds = []
            dissapeared_time = self[missing_object_ind]['time_of_disappearance']

            for i, obj in enumerate(self):
                if i == missing_object_ind:
                    continue
                # Check if an object was instatiated near the time the object dissapeared
                instatiated_time = obj['first_observed_time']
                if instatiated_time < dissapeared_time - look_back_time:
                    # The object was instatiated too long ago
                    continue
                if instatiated_time > dissapeared_time + look_forward_time:
                    # The object was instatiated too long after the object dissapeared
                    continue
                potential_match_inds.append(i)
            
            if len(potential_match_inds) == 0:
                # No potential matches
                matches.append(None)
                transforms.append(np.eye(4))
                continue

            # Compute visual similarity between potential matches
            potential_objects = []
            for i in potential_match_inds:
                potential_objects.append(self[i]['clip_ft'])
            potential_objects = torch.stack(potential_objects)
            query_object = self[missing_object_ind]['clip_ft']
            visual_sim = F.cosine_similarity(potential_objects, query_object)

            # Return object index with highest similarity
            max_ind = visual_sim.argmax().item()
            if visual_sim[max_ind] < threshold:
                matches.append(None)
                transforms.append(np.eye(4))
                continue
            match_ind = potential_match_inds[max_ind]

            # Compute transformation between the two objects
            registration_results = self.getICPRegistration(self[missing_object_ind]['pcd'], self[match_ind]['pcd'], threshold=0.01, max_iteration=100)
            if np.all(registration_results.transformation == np.eye(4)):
                matches.append(None)
                transforms.append(np.eye(4))
                continue

            matches.append(match_ind)
            transforms.append(registration_results.transformation)
    
        return matches, transforms
    
    def getICPRegistration(self, source_pcd, target_pcd, transform_init=np.eye(4), threshold=0.01, max_iteration=30):
        """
        Get the ICP registration between two point clouds

        Arguments:
            source_pcd: source point cloud
            target_pcd: target point cloud
            threshold: threshold for convergence
        Returns:
            registration_results: registration results
        """
        registration_results = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold, transform_init, o3d.pipelines.registration.TransformationEstimationPointToPoint(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
        return registration_results

    def matchRemovedObjectsToRecentObjects(self, removed_object_list, look_back_time: int=10, look_forward_time: int=10, threshold: float=0.7):
        """
        Take in a list of removed objects and match them to recently added objects.
        """
        matches = []
        transforms = []
        for missing_object in removed_object_list:
            potential_match_inds = []
            dissapeared_time = missing_object['time_of_disappearance']
            for i, obj in enumerate(self):
                # Check if an object was instatiated near the time the object dissapeared
                instatiated_time = obj['first_observed_time']
                if instatiated_time < dissapeared_time - look_back_time:
                    # The object was instatiated too long ago
                    continue
                if instatiated_time > dissapeared_time + look_forward_time:
                    # The object was instatiated too long after the object dissapeared
                    continue
                potential_match_inds.append(i)
            
            if len(potential_match_inds) == 0:
                # No potential matches
                matches.append(None)
                transforms.append(np.eye(4))
                continue

            # Compute visual similarity between potential matches
            potential_objects = []
            for i in potential_match_inds:
                potential_objects.append(self[i]['clip_ft'])
            potential_objects = torch.stack(potential_objects)
            query_object = missing_object['clip_ft']
            visual_sim = F.cosine_similarity(potential_objects, query_object)

            # Return object index with highest similarity
            max_ind = visual_sim.argmax().item()
            if visual_sim[max_ind] < threshold:
                matches.append(None)
                transforms.append(np.eye(4))
                continue
            match_ind = potential_match_inds[max_ind]

            # Compute transformation between the two objects
            # registration_results = self.getICPRegistration(missing_object['pcd'], self[match_ind]['pcd'], threshold=0.01, max_iteration=100)
            # if np.all(registration_results.transformation == np.eye(4)):
            #     matches.append(None)
            #     transforms.append(np.eye(4))
            #     continue

            matches.append(match_ind)
            transforms.append(np.eye(4))
            # transforms.append(registration_results.transformation)
    
        return matches, transforms
    
    def reinstateRemovedObjects(self, removed_object_list, matches):
        """
        Reinstate the removed objects based on the matches and transformations
        """
        assert len(removed_object_list) == len(matches), 'Dissapeared and matched indices must be the same length'

        extend_attributes = ['image_idx', 'mask_idx', 'color_path', 'class_id', 'mask', 'xyxy', 'conf', 'contain_number', 'centroid_locations', 'bbox_shadow_hull_history']
        add_attributes = ['num_detections', 'num_obj_in_class']
        skip_attributes = ['id', 'class_name', 'is_background', 'new_counter', 'curr_obj_num', 'inst_color']  # 'inst_color' just keeps obj1's
        custom_handled = ['pcd', 'bbox', 'clip_ft', 'text_ft', 'n_points']

        pocd_skip_attributes = ['confidence_history', 'first_observed_time', 'pocd_confidence', 'age', 'lost_time', 'eps', 'inlier', 'type']
        pocd_mean_attributes = ['a', 'b', 'mu', 'sig', 'pocd_confidence']
        pocd_custom_attributes = ['last_observed_time', 'time_of_disappearance']

        skip_attributes += pocd_skip_attributes
        custom_handled += pocd_custom_attributes

        # Check for unhandled keys and throw an error if there are
        if len(self) > 0:
            all_handled_keys = set(extend_attributes + add_attributes + skip_attributes + custom_handled + pocd_mean_attributes)
            unhandled_keys = set(self[0].keys()) - all_handled_keys
            if unhandled_keys:
                raise ValueError(f"Unhandled keys detected in obj2: {unhandled_keys}. Please update the merge function to handle these attributes.")
        

        for d_ind, m_ind in enumerate(matches):
            if m_ind is None:
                continue
            # Process extend and add attributes
            for attr in extend_attributes:
                if attr in removed_object_list[d_ind] and attr in self[m_ind]:
                    removed_object_list[d_ind][attr].extend(self[m_ind][attr])
            
            for attr in add_attributes:
                if attr in removed_object_list[d_ind] and attr in self[m_ind]:
                    removed_object_list[d_ind][attr] += self[m_ind][attr]

            # Process custom
            removed_object_list[d_ind]['pcd'] = self[m_ind]['pcd']
            removed_object_list[d_ind]['clip_ft'] = self[m_ind]['clip_ft']
            removed_object_list[d_ind]['bbox'] = self[m_ind]['bbox']
            removed_object_list[d_ind]['n_points'] = self[m_ind]['n_points']
            
            removed_object_list[d_ind]['last_observed_time'] = self[m_ind]['last_observed_time']
            removed_object_list[d_ind]['time_of_disappearance'] = -1

            # Process mean attributes
            for attr in pocd_mean_attributes:
                if attr in removed_object_list[d_ind] and attr in self[m_ind]:
                    removed_object_list[d_ind][attr] = (removed_object_list[d_ind][attr] + self[m_ind][attr]) / 2

            self[m_ind] = removed_object_list[d_ind]
        return True
    
        

class POCDObjectTypes(Enum):
    DYNAMIC = 0
    STATIC = 1
    DISSAPEARED = 2

class ObjectLocations(dict):
    def __init__(self, radius: float=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add(self, object_class: str, location: np.ndarray):
        if object_class not in self:
            self[object_class] = []
        self[object_class].append(location)
    

# not sure if I will use this 
class MapEdge():
    def __init__(self, obj1_idx, obj2_idx, rel_type, num_detections=1):
        self.obj1_idx = obj1_idx
        self.obj2_idx = obj2_idx
        self.rel_type = rel_type
        self.num_detections = num_detections
        
    def to_serializable(self):
        return {
            'obj1_idx': self.obj1_idx,
            'obj2_idx': self.obj2_idx,
            'rel_type': self.rel_type,
        }
    
    def load_serializable(self, s_edge_dict):
        self.obj1_idx = s_edge_dict['obj1_idx']
        self.obj2_idx = s_edge_dict['obj2_idx']
        self.rel_type = s_edge_dict['rel_type']
        
    def __str__(self):
        return f"({self.obj1_idx}, {self.rel_type}, {self.obj2_idx}), num_det: {self.num_detections}"
    
    def __repr__(self):
        return str(self)
    
class MapEdgeMapping:
    def __init__(self, objects):
        self.objects = objects  # Reference to the list of existing objects
        self.edges_by_index = {}  # {(obj1_index, obj2_index): MapEdge}
        self.edges_by_uuid = {}  # {(obj1_uuid, obj2_uuid): MapEdge}

    def add_or_update_edge(self, obj1_index, obj2_index, rel_type):
        obj1_uuid, obj2_uuid = self.objects[obj1_index]['id'], self.objects[obj2_index]['id']
        uuid_key = (obj1_uuid, obj2_uuid)
        
        if obj1_index == obj2_index:
            print(f"LOOOPY EDGE DETECTED: {obj1_index} == {obj2_index}")
            pass
        
        if (obj1_index, obj2_index) in self.edges_by_index:
            edge = self.edges_by_index[(obj1_index, obj2_index)]
            edge.num_detections += 1
        else:
            edge = MapEdge(obj1_index, obj2_index, rel_type)
            self.edges_by_index[(obj1_index, obj2_index)] = edge
            self.edges_by_uuid[uuid_key] = edge

    def delete_object_edges(self, obj_index):
        # Remove all edges associated with the object at obj_index
        to_remove = [key for key in self.edges_by_index if obj_index in key]
        for key in to_remove:
            # Remove from both index-based and UUID-based storage
            del self.edges_by_index[key]
            uuid_key = (self.objects[key[0]]['id'], self.objects[key[1]]['id'])
            del self.edges_by_uuid[uuid_key]
            
    def update_indices(self, index_map, new_objects):
        self.objects = new_objects  # Update the objects reference if necessary
        new_edges_by_index = {}
        new_edges_by_uuid = {}

        for (old_obj1_index, old_obj2_index), edge in list(self.edges_by_index.items()):
            new_obj1_index = index_map.get(old_obj1_index)
            new_obj2_index = index_map.get(old_obj2_index)

            if new_obj1_index is not None and new_obj2_index is not None:
                new_key = (new_obj1_index, new_obj2_index)
                new_uuid_key = (self.objects[new_obj1_index]['id'], self.objects[new_obj2_index]['id'])

                if new_key in new_edges_by_index:
                    new_edges_by_index[new_key].num_detections += edge.num_detections
                else:
                    edge.obj1 = new_obj1_index  # Update the edge's internal object index reference
                    edge.obj2 = new_obj2_index
                    new_edges_by_index[new_key] = edge
                    new_edges_by_uuid[new_uuid_key] = edge

        self.edges_by_index = new_edges_by_index
        self.edges_by_uuid = new_edges_by_uuid
    
    def merge_update_indices(self, index_updates):
        """Update all edge indices based on the new mapping after merging objects."""
        updated_edges_by_index = {}
        updated_edges_by_uuid = {}

        # Iterate over current edges to update indices based on index_updates
        for (obj1_index, obj2_index), curr_edge in list(self.edges_by_index.items()):
            new_obj1_index = index_updates[obj1_index]
            new_obj2_index = index_updates[obj2_index]

            # Skip updates if either index is None (meaning the object was merged away)
            if new_obj1_index is None or new_obj2_index is None:
                continue

            # Avoid creating a loop edge where an object points to itself
            if new_obj1_index == new_obj2_index:
                print(f"LOOOPY EDGE DETECTED: {new_obj1_index} == {new_obj2_index}")
                continue

            new_key = (new_obj1_index, new_obj2_index)
            new_obj1_uuid, new_obj2_uuid = self.objects[new_obj1_index]['id'], self.objects[new_obj2_index]['id']
            new_uuid_key = (new_obj1_uuid, new_obj2_uuid)
            
            # If the edge already exists after merge, update num_detections
            if new_key in updated_edges_by_index:
                updated_edges_by_index[new_key].num_detections += curr_edge.num_detections
            else:
                # Update the edge with new indices
                curr_edge.obj1_idx = new_obj1_index
                curr_edge.obj2_idx = new_obj2_index
                updated_edges_by_index[new_key] = curr_edge
                updated_edges_by_uuid[new_uuid_key] = curr_edge

        # Update the class attributes with the modified edges
        self.edges_by_index = updated_edges_by_index
        self.edges_by_uuid = updated_edges_by_uuid
        
    def update_objects_list(self, new_objects):
        self.objects = new_objects

    def merge_objects_edges(self, source_index, destination_index):
        # Update edges for a merged object. source_index object is merged into destination_index object
        updated_edges_by_index = {}
        updated_edges_by_uuid = {}

        for (obj1_index, obj2_index), curr_edge in self.edges_by_index.items():
            # Check if source object is part of the edge and update the edge accordingly
            
            # if not (source_index in (obj1_index, obj2_index)):
            #     continue
            
            new_obj1_index, new_obj2_index = obj1_index, obj2_index
            
            if new_obj1_index == new_obj2_index: # check loop edge
                print(f"LOOOPY EDGE DETECTED: {new_obj1_index} == {new_obj2_index}")
                pass
            
            # check if edge is between source and destination
            if source_index in (new_obj1_index, new_obj2_index) and destination_index in (new_obj1_index, new_obj2_index):
                print(f"Edge between source and destination: {source_index} in {new_obj1_index, new_obj2_index} and {destination_index} in {new_obj1_index, new_obj2_index}")
                pass
                continue
            
            if obj1_index == source_index:
                print(f"obj1_index matches source_index: {obj1_index} == {source_index}")
                new_obj1_index = destination_index
                
            if obj2_index == source_index:
                print(f"obj2_index matches source_index: {obj2_index} == {source_index}")
                new_obj2_index = destination_index
                
            if new_obj1_index == new_obj2_index: # check loop edge
                print(f"LOOOPY EDGE DETECTED: {new_obj1_index} == {new_obj2_index}")
                pass
                continue


            # Generate new edge key and UUID key
            new_key = (new_obj1_index, new_obj2_index)
            new_obj1_uuid, new_obj2_uuid = self.objects[new_obj1_index]['id'], self.objects[new_obj2_index]['id']
            new_uuid_key = (new_obj1_uuid, new_obj2_uuid)
            new_edge = MapEdge(new_obj1_index, new_obj2_index, curr_edge.rel_type, curr_edge.num_detections)

            # Check if the edge already exists after merge, update num_detections if it does
            if new_key in updated_edges_by_index:
                updated_edges_by_index[new_key].num_detections += curr_edge.num_detections
            else:
                curr_edge.obj1_idx = new_obj1_index
                curr_edge.obj2_idx = new_obj2_index
                updated_edges_by_index[new_key] = new_edge
                updated_edges_by_uuid[new_uuid_key] = new_edge

        # Update the class attributes
        self.edges_by_index = updated_edges_by_index
        self.edges_by_uuid = updated_edges_by_uuid
        
    def get_edges_by_curr_obj_num(self):
        map_edges_by_curr_obj_num = []
        for (obj1_idx, obj2_idx), map_edge in self.edges_by_index.items():
            obj1_curr_obj_num = self.objects[obj1_idx]['curr_obj_num']
            obj2_curr_obj_num = self.objects[obj2_idx]['curr_obj_num']
            rel_type = map_edge.rel_type
            map_edges_by_curr_obj_num.append((obj1_curr_obj_num, rel_type, obj2_curr_obj_num))
        return map_edges_by_curr_obj_num
    
    def get_edges_by_curr_obj_num_label(self):
        map_edges_by_curr_obj_num_label = []
        for (obj1_idx, obj2_idx), map_edge in self.edges_by_index.items():
            # Construct the curr_obj_num_label for both objects
            obj1 = self.objects[obj1_idx]
            obj2 = self.objects[obj2_idx]
            obj1_curr_obj_num_label = f"{obj1['curr_obj_num']}_{obj1['class_name']}"
            obj2_curr_obj_num_label = f"{obj2['curr_obj_num']}_{obj2['class_name']}"

            # Append the edge with the formatted labels
            map_edges_by_curr_obj_num_label.append((obj1_curr_obj_num_label, map_edge.rel_type, obj2_curr_obj_num_label))
        return map_edges_by_curr_obj_num_label

    def get_edge_endpoints(self, obj1_index, obj2_index):
        # Check if the edge exists
        if (obj1_index, obj2_index) in self.edges_by_index:
            obj1_center = np.asarray(self.objects[obj1_index]['bbox'].get_center())
            obj2_center = np.asarray(self.objects[obj2_index]['bbox'].get_center())
            return [obj1_center, obj2_center]
        return None

    def __str__(self):
        return '\n'.join([str(edge) for edge in self.edges_by_index.values()])

    def __repr__(self):
        return self.__str__()
    
    def to_serializable(self):
        s_edges = []
        for (obj1_index, obj2_index), edge in self.edges_by_index.items():
            s_edges.append({
                'obj1_index': obj1_index,
                'obj2_index': obj2_index,
                'rel_type': edge.rel_type,
                'num_detections': edge.num_detections
            })
        
        # Serialize the object list using its existing method
        s_objects = self.objects.to_serializable()
        
        return {
            'edges': s_edges,
            'objects': s_objects
        }
        
    def load_serializable(self, s_data):
        assert len(self.edges_by_index) == 0 and len(self.objects) == 0, 'MapEdgeMapping should be empty when loading'
        
        # Deserialize the objects list first
        self.objects.load_serializable(s_data['objects'])
        
        # Rebuild the edges
        for s_edge in s_data['edges']:
            obj1_index = s_edge['obj1_index']
            obj2_index = s_edge['obj2_index']
            rel_type = s_edge['rel_type']
            num_detections = s_edge['num_detections']
            
            # Create a new edge
            edge = MapEdge(obj1_index, obj2_index, rel_type, num_detections)
            self.edges_by_index[(obj1_index, obj2_index)] = edge
            
            # Assuming 'id' attribute exists in the objects for UUID key generation
            obj1_uuid = self.objects[obj1_index]['id']
            obj2_uuid = self.objects[obj2_index]['id']
            self.edges_by_uuid[(obj1_uuid, obj2_uuid)] = edge