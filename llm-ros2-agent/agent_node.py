import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import google.generativeai as generativeai
import pyttsx3

class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.engine = pyttsx3.init()
        self.api_key = self.get_api_key()
        self.setup_google_api()

    def get_api_key(self):
        return os.getenv('GOOGLE_API_KEY')

    def setup_google_api(self):
        generativeai.configure(api_key=self.api_key)

    def process_command(self, command):
        response = generativeai.chat(
            messages=[{"role": "user", "content": command}]
        )
        action = self.extract_action(response)
        self.execute_action(action)

    def extract_action(self, response):
        # Logic to extract action from the response
        return action

    def execute_action(self, action):
        twist = Twist()
        # Logic to convert action to Twist message
        self.publisher.publish(twist)
        self.speak_action(action)

    def speak_action(self, action):
        self.engine.say(action)
        self.engine.runAndWait()

def main(args=None):
    rclpy.init(args=args)
    agent_node = AgentNode()
    rclpy.spin(agent_node)
    agent_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()