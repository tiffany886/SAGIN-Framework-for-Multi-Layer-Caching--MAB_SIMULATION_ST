class Satellite:
    """
    卫星类（Satellite），用于模拟SAGIN（Space-Air-Ground Integrated Network）框架中的卫星节点。
    卫星负责生成特定类别的内容，并将其提供给地面站和其他网络节点。
    主要生成I、II、III类内容（分别对应HDF、图像、传感器数据）。
    """
    def __init__(self, satellite_id):
        """
        初始化卫星对象。

        参数:
        satellite_id (str): 卫星的唯一标识符，如"Satellite1"、"Satellite2"等。

        该方法设置卫星的基本属性，包括内容类别、内容大小、缓存结构和传输速率。
        """
        self.satellite_id = satellite_id # 卫星的唯一标识
        self.content_categories = ['I', 'II', 'III']  # 内容类别：I：HDF大数据，II：图像，III：传感器数据 ；三类数据覆盖了灾后常用信息（广域遥感图、高清图像、环境传感器数据）
        self.content_size = {'I':10, 'II': 1, 'III': 0.01, 'IV': 2}  # 类别对应的内容大小
        self.counter = 0
        self.generated_cache = {
            'I': [],
            'II': [],
            'III': [],  # Add more categories if needed
        }  # 用于存储从卫星和其他无人机接收到内容的缓存
        self.transmission_rate = 1  # Mbps 模拟卫星下行速度，但实际计算延迟时会用到

    def generate_content(self, current_time, no_of_content_each_category, ground_station):
        """
        生成卫星内容并将其添加到地面站的缓存中。
        当卫星与地面站建立连接时，它就会生成新的内容。生成的内容会同时保存在卫星本地缓存和地面站中，方便后续被其他节点获取。
        参数:
        current_time (float): 当前时间戳（秒）。
        no_of_content_each_category (int): 每个类别生成的内容数量。
        ground_station (GroundStation): 地面站对象，用于存储生成的內容。

        该方法为卫星的有效内容类别（I、II、III）生成内容，每个内容包含：
        - 内容类型、目的地、生成时间、跳数、坐标、类别、编号等元数据
        - 内容有效期设置为30分钟
        - 内容大小根据类别确定
        生成的内容会同时存储在卫星本地缓存和地面站的全局缓存中。
        """
        """
        ✅ FIXED: Generate content only for valid satellite categories
        """
        content_validity = 30 * 60  #  # 30分钟有效期

        # ✅ FIXED: Only generate content for valid satellite categories
        valid_categories = ['I', 'II', 'III']  # 卫星只生成这三类

        # Generate content for connected satellites
        for category in valid_categories:  # Changed from self.content_categories
            category_contents = []
            content_size = self.content_size.get(category, 0) # 由类别决定对应的大小
            num_contents = no_of_content_each_category  # 生成数量由传入参数决定
            for n in range(1, num_contents + 1):
                content = {
                    'content_type': 'satellite',  # Changed from 'Satellite' to 'satellite'
                    'destination': None, # 目的地
                    'generation_time': current_time, # 生成时间
                    'hop_count': 0, # 跳数
                    'content_coord': int(self.satellite_id.replace('Satellite', '')),  #  # 源设备的编号/坐标Convert "Satellite1" → 1
                    'content_category': category, # 类别
                    'content_no': n, # 内容在该类别下的序号
                    # 'generation_time': current_time,
                    'content_validity': content_validity, #有效期
                    'content_receive_time': 0, # 接收时间（暂未使用）
                    'size': content_size,  # 内容大小 MB
                    'content_hit': 0, # 命中计数（暂未使用）
                }
                category_contents.append(content)

            self.generated_cache[category] = category_contents
            # Add the generated content to the GroundStation's content_cache for this category
            # 把内容备份到地面站
            ground_station.add_to_cache(self.satellite_id, category, category_contents) 

    def print_cache(self):
        """
        打印卫星当前生成的缓存内容。
        显示每个内容类别的所有内容编号，用于调试和监控。
        """
        print("Generated Cache for Satellite", self.satellite_id)
        for category, contents in self.generated_cache.items():
            print(f"Category: {category}", end=' ')
            for content in contents:
                print(f"Content Number: {content['content_no']}", end=' ')
            print()  # Print a new line after each category



    def cache_clean(self, current_time):
        """
        清理卫星缓存中过期的内容。

        参数:
        current_time (float): 当前时间戳（秒）。

        该方法遍历所有内容类别，移除已过期的内容（超过30分钟有效期）。
        保留生成时间+有效期 > 当前时间的内容，移除过期的。
        保持缓存中只包含有效的内容，提高存储效率。
        """
        for category in self.content_categories:
            cached_content = self.generated_cache[category]
            # Filter out content that has expired
           # for content in cached_content:
                #if current_time >= content['generation_time'] + content['content_validity']:
                    #print(f"Expired Content - Generation Time: {content['generation_time']}, Current Time: {current_time}", end=' ')
            self.generated_cache[category] = [content for content in cached_content if
                                              current_time < content['generation_time'] + content['content_validity']]

    def run(self, satellites, communication,  communication_schedule, current_time, slot, no_of_content_each_category, ground_station):
        """
        卫星运行主循环，负责内容生成和缓存管理。

        参数:
        satellites (dict): 所有卫星对象的字典。
        communication (Communication): 通信模块对象。
        communication_schedule (dict): 通信调度表。
        current_time (float): 当前时间戳（秒）。
        slot (int): 当前时间槽。
        no_of_content_each_category (int): 每个类别生成的内容数量。
        ground_station (GroundStation): 地面站对象。

        该方法检查当前卫星是否在连接列表中，如果是则生成内容，
        然后执行缓存清理操作。这是卫星在每个时间槽的主要执行逻辑。
        """
        #print(slot, current_time, self.satellite_id)
        # communication.get_connected_satellites获得当前时隙slot有哪些卫星与地面站建立连接，
        # 根据通信调度表communication_schedule，如果自己在这个列表里，说明卫星此刻可见，就生成新数据。
        satellite_list = communication.get_connected_satellites(slot, communication_schedule, satellites)
        for satellite in satellite_list:
            #print(satellite.satellite_id)
            if satellite.satellite_id == self.satellite_id:
               # 生成卫星内容并将其添加到地面站的缓存中。
               self.generate_content(current_time, no_of_content_each_category, ground_station)
              # self.print_cache()
        self.cache_clean(current_time)
        #self.print_cache()
