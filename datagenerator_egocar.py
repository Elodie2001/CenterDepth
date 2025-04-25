import glob
import queue
import sys
import random
import os
import numpy as np
import cv2
from xml.etree.ElementTree import Element, SubElement, tostring, ElementTree


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def is_bbox_inside(inner_box, outer_box):
    return (inner_box[0] > outer_box[0] and inner_box[1] > outer_box[1] and
            inner_box[2] < outer_box[2] and inner_box[3] < outer_box[3])

def check_occlusion(bboxes):
    occluded_indices = set()
    for i in range(len(bboxes)):
        for j in range(len(bboxes)):
            if i != j and is_bbox_inside(bboxes[i], bboxes[j]):
                occluded_indices.add(i)
    return occluded_indices

def set_clear_weather(world):
    clear_weather = carla.WeatherParameters(
        cloudiness=10.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=0.0,
        sun_azimuth_angle=45.0,  
        sun_altitude_angle=75.0,  
        fog_density=0.0,
        fog_distance=1000.0,  
        wetness=0.0
    )
    world.set_weather(clear_weather)

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):

    point = np.array([loc.x, loc.y, loc.z, 1])

    point_camera = np.dot(w2c, point)

    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    point_img = np.dot(K, point_camera)

    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]

def get_cam_point(loc,w2c):
    point = np.array([loc.x, loc.y, loc.z, 1])

    point_camera = np.dot(w2c, point)

    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    return point_camera

def check_color_in_rectangle(image, x_min, x_max, y_min, y_max,target_color,ratio):
    width, height = 1242,375
    x_min = int(max(0, min(x_min, width - 1)))
    x_max = int(max(0, min(x_max, width - 1)))
    y_min = int(max(0, min(y_min, height - 1)))
    y_max = int(max(0, min(y_max, height - 1)))
    if x_min == x_max or y_min == y_max:
        return False
    region = image[y_min:y_max, x_min:x_max]
    mask = cv2.inRange(region, target_color, target_color)
    total_pixels = np.prod(region.shape[:2])
    target_pixels = cv2.countNonZero(mask)
    target_percentage = target_pixels / total_pixels
    return target_percentage > ratio

def process_semantic(image):
    image.convert(carla.ColorConverter.CityScapesPalette)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = ((array/255.0)*255).astype(np.uint8)
    return array

class Writer:
    def __init__(self, filename):
        self.root = Element('annotation')
        filename_elem = SubElement(self.root, 'filename')
        filename_elem.text = filename

        size_elem = SubElement(self.root, 'size')
        width_elem = SubElement(size_elem, 'width')
        width_elem.text = "1242"
        height_elem = SubElement(size_elem, 'height')
        height_elem.text = "375"

    def addObject(self, name, xmin, ymin, xmax, ymax, dist, p0):
        object_elem = SubElement(self.root, 'object')
        name_elem = SubElement(object_elem, 'name')
        name_elem.text = name
        bndbox_elem = SubElement(object_elem, 'bndbox')
        xmin_elem = SubElement(bndbox_elem, 'xmin')
        xmin_elem.text = str(xmin)
        ymin_elem = SubElement(bndbox_elem, 'ymin')
        ymin_elem.text = str(ymin)
        xmax_elem = SubElement(bndbox_elem, 'xmax')
        xmax_elem.text = str(xmax)
        ymax_elem = SubElement(bndbox_elem, 'ymax')
        ymax_elem.text = str(ymax)
        dist_elem = SubElement(object_elem, 'dist')
        dist_elem.text = str(dist)
        p0_elem = SubElement(object_elem, 'p0')
        x0_elem = SubElement(p0_elem, 'x0')
        x0_elem.text = str(p0[2])
        y0_elem = SubElement(p0_elem, 'y0')
        y0_elem.text = str(p0[0])
        z0_elem = SubElement(p0_elem, 'z0')
        z0_elem.text = str(-p0[1])

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write(tostring(self.root, encoding='unicode'))

def main():
    images_with_nested_bboxes = []
    try:
        client = carla.Client('localhost', 2000)
        client.load_world('Town01')
        num_walkers = 50
        num_vehicle = 50

        percentage_pedestrians_running = 0.1

        world = client.get_world()

        set_clear_weather(world)

        traffic_manager = client.get_trafficmanager()

        spectator = world.get_spectator()

        transform = spectator.get_transform()

        location = transform.location + carla.Location(x=-30, z=20)
        rotation = carla.Rotation(pitch=-20, yaw=-20, roll=0)
        new_transform = carla.Transform(location, rotation)

        spectator.set_transform(new_transform)

        # vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')
        vehicle_blueprints = [bp for bp in world.get_blueprint_library().filter('*vehicle*')
                              if 'carlamotors' not in bp.id and 'bh' not in bp.id and 'kawasaki' not in bp.id and 'gazelle' not in bp.id and 'yamaha' not in bp.id and 'vespa' not in bp.id and 'mitsubishi' not in bp.id and 'diamondback' not in bp.id and 'harley-davidson' not in bp.id and 'micro' not in bp.id and 'cybertruck' not in bp.id and 't2' not in bp.id and 'ambulance' not in bp.id and 't2_2021' not in bp.id and 'sprinter' not in bp.id]
        # for bp in vehicle_blueprints:
        #     print(bp.id)
        # ped_blueprints = world.get_blueprint_library().filter('*pedestrian*')
        ped_blueprints = [bp for bp in world.get_blueprint_library().filter('*walker*')
                          if 'child' not in bp.id and 'bicycle' not in bp.id and 'rider' not in bp.id and 'motorcycle' not in bp.id]

        vehicle_spawn_points = world.get_map().get_spawn_points()

        ped_spawn_points = []
        for i in range(num_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                ped_spawn_points.append(spawn_point)

        for i in range(0, num_vehicle):
            world.try_spawn_actor(random.choice(vehicle_blueprints),
                                  random.choice(vehicle_spawn_points))

        walker_batch = []
        walker_speed = []
        walker_ai_batch = []

        for j in range(num_walkers):
            walker_bp = random.choice(ped_blueprints)
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            if walker_bp.has_attribute('speed'):
                if random.random() > percentage_pedestrians_running:
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            walker_batch.append(world.try_spawn_actor(walker_bp, random.choice(ped_spawn_points)))

        walker_ai_blueprint = world.get_blueprint_library().find('controller.ai.walker')

        for walker in world.get_actors().filter('*pedestrian*'):
            walker_ai_batch.append(world.spawn_actor(walker_ai_blueprint, carla.Transform(), walker))

        for i in range(len(walker_ai_batch)):
            walker_ai_batch[i].start()
            walker_ai_batch[i].go_to_location(world.get_random_location_from_navigation())
            walker_ai_batch[i].set_max_speed(float(walker_speed[i]))

        for vehicle in world.get_actors().filter('*vehicle*'):
            vehicle.set_autopilot()

        ego_spawn_point = random.choice(vehicle_spawn_points)
        ego_bp = world.get_blueprint_library().find('vehicle.mini.cooper_s_2021')
        ego_bp.set_attribute('role_name', 'hero')
        ego_vehicle = world.spawn_actor(ego_bp, ego_spawn_point)
        ego_vehicle.set_autopilot()

        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x", str(1242))
        cam_bp.set_attribute("image_size_y", str(375))
        cam_bp.set_attribute("fov", str(90))

        cam_location = carla.Location(1, 0, 2)
        cam_rotation = carla.Rotation(0, 0, 0)
        cam_transform = carla.Transform(cam_location, cam_rotation)

        camera = world.spawn_actor(cam_bp, cam_transform, attach_to=ego_vehicle,
                                   attachment_type=carla.AttachmentType.Rigid)


        semantic_cam_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        semantic_cam_bp.set_attribute("image_size_x", str(1242))
        semantic_cam_bp.set_attribute("image_size_y", str(375))
        semantic_cam_bp.set_attribute("fov", str(90))


        semantic_cam_location = carla.Location(1, 0, 2)
        semantic_cam_rotation = carla.Rotation(0, 0, 0)
        semantic_cam_transform = carla.Transform(semantic_cam_location, semantic_cam_rotation)


        semantic_camera = world.spawn_actor(semantic_cam_bp, semantic_cam_transform, attach_to=ego_vehicle,
                                            attachment_type=carla.AttachmentType.Rigid)
        # 从相机获取属性
        image_w = cam_bp.get_attribute("image_size_x").as_int()  # img width
        image_h = cam_bp.get_attribute("image_size_y").as_int()  # img height
        fov = cam_bp.get_attribute("fov").as_float()  # fov

        K = build_projection_matrix(image_w, image_h, fov)
        edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]

        setting = world.get_settings()
        origin_setting = setting

        setting.synchronous_mode = True

        setting.fixed_delta_seconds = 0.05

        world.apply_settings(setting)

        traffic_manager.synchronous_mode = True

        traffic_manager.global_percentage_speed_difference(-30)

        image_queue = queue.Queue()

        semantic_image_queue = queue.Queue()

        camera.listen(image_queue.put)
        semantic_camera.listen(semantic_image_queue.put)

        output_path = "../data/Town_new/imgewithbbox"
        output_seg_path = "../data/Town_new/segimg"
        output_rawdata = "../data/Town_new/rawimg"
        xml_output_path = "../data/Town_new/xml"
        frame_num = 0


        while True:

            world.get_spectator().set_transform(camera.get_transform())
            # print(camera.get_transform())
            # print(camera.get_transform().get_matrix())

            if traffic_manager.synchronous_mode:

                world.tick()

                image = image_queue.get()
                semantic_image = semantic_image_queue.get()

                img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
                cv2.imwrite(os.path.join(output_rawdata, '%06d.png' % frame_num), img)
                semantic_img = process_semantic(semantic_image)
                cv2.imwrite(os.path.join(output_seg_path, '%06d.png' % frame_num), semantic_img)
                # semantic_img = np.reshape(np.copy(semantic_image.raw_data), (semantic_image.height, semantic_image.width, 4))


                world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

                xml_writer = Writer(os.path.join(xml_output_path, '%06d.xml' % frame_num))

                bboxes = []

                for human in world.get_actors().filter('*pedestrian*'):
                    bb = human.bounding_box
                    dist = (human.get_transform().location.distance(ego_vehicle.get_transform().location))

                    if dist < 200:
                        forward_vec = ego_vehicle.get_transform().get_forward_vector()
                        ray_vec = human.get_transform().location - ego_vehicle.get_transform().location
                        forward_arr = np.array([forward_vec.x,forward_vec.y,forward_vec.z])
                        ray_arr = np.array([ray_vec.x,ray_vec.y,ray_vec.z])

                        if forward_arr.dot(ray_arr) > 1:
                            p0 = get_cam_point(bb.location, world_2_camera)
                            verts = [v for v in bb.get_world_vertices(human.get_transform())]

                            #3Dbbox
                            # for edge in edges:
                            #     p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                            #     p2 = get_image_point(verts[edge[1]], K, world_2_camera)
                            #     cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0, 255), 1)

                            #2Dbbox
                            min_x, min_y = float('inf'), float('inf')
                            max_x, max_y = float('-inf'), float('-inf')
                            for edge in edges:
                                p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                                x1,y1 = p1
                                p2 = get_image_point(verts[edge[1]], K, world_2_camera)
                                x2,y2 = p2
                                min_x = int(min(min_x, x1, x2));min_x = max(0, min(min_x, image_w - 1))
                                min_y = int(min(min_y, y1, y2));min_y = max(0, min(min_y, image_h - 1))
                                max_x = int(max(max_x, x1, x2));max_x = max(0, min(max_x, image_w - 1))
                                max_y = int(max(max_y, y1, y2));max_y = max(0, min(max_y, image_h - 1))
                            if check_color_in_rectangle(semantic_img, min_x, max_x, min_y, max_y, np.array((60, 20, 220)), 0.1):
                            #   or check_color_in_rectangle(semantic_img, min_x, max_x, min_y, max_y, np.array((255, 0, 0)), 0.1):
                                cv2.line(img, (min_x, min_y), (max_x, min_y), (0, 255, 0, 255), 1)
                                cv2.line(img, (max_x, min_y), (max_x, max_y), (0, 255, 0, 255), 1)
                                cv2.line(img, (max_x, max_y), (min_x, max_y), (0, 255, 0, 255), 1)
                                cv2.line(img, (min_x, min_y), (min_x, max_y), (0, 255, 0, 255), 1)
                                if min_x >=0 and max_x < 1920 and min_y >=0 and max_y < 1080:
                                    xml_writer.addObject('pedestrian', min_x, min_y, max_x, max_y, dist, p0)

                            # if check_color_in_rectangle(semantic_img, min_x, max_x, min_y, max_y,(220,20,60)):
                            #     cv2.line(img, (int(min_x), int(min_y)), (int(max_x), int(min_y)), (0, 255, 0, 255), 1)
                            #     cv2.line(img, (int(max_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0, 255), 1)
                            #     cv2.line(img, (int(max_x), int(max_y)), (int(min_x), int(max_y)), (0, 255, 0, 255), 1)
                            #     cv2.line(img, (int(min_x), int(min_y)), (int(min_x), int(max_y)), (0, 255, 0, 255), 1)

                for car in world.get_actors().filter('*vehicle*'):
                    bb_car = car.bounding_box
                    dist = (car.get_transform().location.distance(ego_vehicle.get_transform().location))

                    if dist < 200:
                        forward_vec = ego_vehicle.get_transform().get_forward_vector()
                        ray_vec = car.get_transform().location - ego_vehicle.get_transform().location
                        forward_arr = np.array([forward_vec.x,forward_vec.y,forward_vec.z])
                        ray_arr = np.array([ray_vec.x,ray_vec.y,ray_vec.z])

                        if forward_arr.dot(ray_arr) > 1:
                            p1 = get_image_point(bb.location, K, world_2_camera)
                            verts = [v for v in bb_car.get_world_vertices(car.get_transform())]

                            #3Dbbox
                            # for edge in edges:
                            #     p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                            #     p2 = get_image_point(verts[edge[1]], K, world_2_camera)
                            #     cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 255, 255), 1)
                            #2Dbbox
                            min_x, min_y = float('inf'), float('inf')
                            max_x, max_y = float('-inf'), float('-inf')
                            for edge in edges:
                                p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                                x1,y1 = p1
                                p2 = get_image_point(verts[edge[1]], K, world_2_camera)
                                x2,y2 = p2
                                min_x = int(min(min_x, x1, x2));min_x = max(0, min(min_x, image_w - 1))
                                min_y = int(min(min_y, y1, y2));min_y = max(0, min(min_y, image_h - 1))
                                max_x = int(max(max_x, x1, x2));max_x = max(0, min(max_x, image_w - 1))
                                max_y = int(max(max_y, y1, y2));max_y = max(0, min(max_y, image_h - 1))

                                # if check_color_in_rectangle(semantic_img, min_x, max_x, min_y, max_y, np.array((142, 0, 0)), 0.2):
                                #     bboxes.append((min_x, min_y, max_x, max_y))
                                # elif check_color_in_rectangle(semantic_img, min_x, max_x, min_y, max_y, np.array((70, 0, 0)), 0.2):
                                #     bboxes.append((min_x, min_y, max_x, max_y))
                                # elif check_color_in_rectangle(semantic_img, min_x, max_x, min_y, max_y, np.array((100, 60, 0)), 0.2):
                                #     bboxes.append((min_x, min_y, max_x, max_y))
                
                # occluded_indices = check_occlusion(bboxes)

                # filtered_bboxes = [bbox for i, bbox in enumerate(bboxes) if i not in occluded_indices]

                # for bbox in filtered_bboxes:
                #     min_x, min_y, max_x, max_y = bbox
                #     cv2.line(img, (min_x, min_y), (max_x, min_y), (0, 255, 0, 255), 1)
                #     cv2.line(img, (max_x, min_y), (max_x, max_y), (0, 255, 0, 255), 1)
                #     cv2.line(img, (max_x, max_y), (min_x, max_y), (0, 255, 0, 255), 1)
                #     cv2.line(img, (min_x, min_y), (min_x, max_y), (0, 255, 0, 255), 1)
                #     if min_x >= 0 and max_x < 1920 and min_y >= 0 and max_y < 1080:
                #         xml_writer.addObject('car', min_x, min_y, max_x, max_y, dist, p0)
                            # if check_color_in_rectangle(semantic_img, min_x, max_x, min_y, max_y, np.array((142, 0, 0)),0.1) \
                            # or check_color_in_rectangle(semantic_img, min_x, max_x, min_y, max_y, np.array((70, 0, 0)),0.1) \
                            # or check_color_in_rectangle(semantic_img, min_x, max_x, min_y, max_y, np.array((100, 60, 0)),0.1):
                            #     cv2.line(img, (min_x, min_y), (max_x, min_y), (0, 255, 0, 255), 1)
                            #     cv2.line(img, (max_x, min_y), (max_x, max_y), (0, 255, 0, 255), 1)
                            #     cv2.line(img, (max_x, max_y), (min_x, max_y), (0, 255, 0, 255), 1)
                            #     cv2.line(img, (min_x, min_y), (min_x, max_y), (0, 255, 0, 255), 1)
                            #     if min_x >=0 and max_x < 1920 and min_y >=0 and max_y < 1080:
                            #        xml_writer.addObject('vehicle', min_x, min_y, max_x, max_y, dist, p0)
                            if check_color_in_rectangle(semantic_img, min_x, max_x, min_y, max_y, np.array((142, 0, 0)),0.2):
                                bbox = (min_x, min_y, max_x, max_y)
                                for other_bbox in bboxes:
                                    if is_bbox_inside(bbox, other_bbox) or is_bbox_inside(other_bbox, bbox):
                                        images_with_nested_bboxes.append('%06d.png' % frame_num)
                                        break
                                bboxes.append(bbox)
                                cv2.line(img, (min_x, min_y), (max_x, min_y), (0, 255, 0, 255), 1)
                                cv2.line(img, (max_x, min_y), (max_x, max_y), (0, 255, 0, 255), 1)
                                cv2.line(img, (max_x, max_y), (min_x, max_y), (0, 255, 0, 255), 1)
                                cv2.line(img, (min_x, min_y), (min_x, max_y), (0, 255, 0, 255), 1)
                                if min_x >=0 and max_x < 1920 and min_y >=0 and max_y < 1080:
                                   xml_writer.addObject('car', min_x, min_y, max_x, max_y, dist, p0)
                            elif check_color_in_rectangle(semantic_img, min_x, max_x, min_y, max_y, np.array((70, 0, 0)),0.2):
                                bbox = (min_x, min_y, max_x, max_y)
                                for other_bbox in bboxes:
                                    if is_bbox_inside(bbox, other_bbox) or is_bbox_inside(other_bbox, bbox):
                                        images_with_nested_bboxes.append('%06d.png' % frame_num)
                                        break
                                bboxes.append(bbox)
                                cv2.line(img, (min_x, min_y), (max_x, min_y), (0, 255, 0, 255), 1)
                                cv2.line(img, (max_x, min_y), (max_x, max_y), (0, 255, 0, 255), 1)
                                cv2.line(img, (max_x, max_y), (min_x, max_y), (0, 255, 0, 255), 1)
                                cv2.line(img, (min_x, min_y), (min_x, max_y), (0, 255, 0, 255), 1)
                                if min_x >=0 and max_x < 1920 and min_y >=0 and max_y < 1080:
                                   xml_writer.addObject('truck', min_x, min_y, max_x, max_y, dist, p0)
                            elif check_color_in_rectangle(semantic_img, min_x, max_x, min_y, max_y, np.array((100, 60, 0)),0.2):
                                bbox = (min_x, min_y, max_x, max_y)
                                for other_bbox in bboxes:
                                    if is_bbox_inside(bbox, other_bbox) or is_bbox_inside(other_bbox, bbox):
                                        images_with_nested_bboxes.append('%06d.png' % frame_num)
                                        break
                                bboxes.append(bbox)
                                cv2.line(img, (min_x, min_y), (max_x, min_y), (0, 255, 0, 255), 1)
                                cv2.line(img, (max_x, min_y), (max_x, max_y), (0, 255, 0, 255), 1)
                                cv2.line(img, (max_x, max_y), (min_x, max_y), (0, 255, 0, 255), 1)
                                cv2.line(img, (min_x, min_y), (min_x, max_y), (0, 255, 0, 255), 1)
                                if min_x >=0 and max_x < 1920 and min_y >=0 and max_y < 1080:
                                   xml_writer.addObject('bus', min_x, min_y, max_x, max_y, dist, p0)

                # img.save_to_disk(output_path % image.frame)
                cv2.imwrite(os.path.join(output_path, '%06d.png' % frame_num), img)
                xml_writer.save(os.path.join(xml_output_path, '%06d.xml' % frame_num))
                frame_num += 1
                
            else:
                
                world.wait_for_tick()


    finally:
        
        for controller in world.get_actors().filter('*controller*'):
            controller.stop()
        
        for vehicle in world.get_actors().filter('*vehicle*'):
            vehicle.destroy()
        
        for walker in world.get_actors().filter('*walker*'):
            walker.destroy()
        camera.destroy()
        semantic_camera.destroy()
        save_images_with_nested_bboxes(images_with_nested_bboxes)


       
        world.apply_settings(origin_setting)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')