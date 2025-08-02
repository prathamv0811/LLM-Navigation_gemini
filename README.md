# LLM-Powered ROS 2 Robot Agent

This project implements a ROS 2 node that acts as an intelligent agent, translating natural language commands into actions for a robot (e.g., TurtleBot3). It uses the Google Gemini API for language understanding and function calling, and provides Text-to-Speech (TTS) feedback for its actions.

## Features

- **Natural Language Understanding:** Give commands in plain English (e.g., "move forward 1 meter").
- **Closed-Loop Control:** Uses odometry feedback from `/odom` to perform precise distance and angle movements, making it more reliable than simple timed movements.
- **LLM-Powered Tool Calling:** Uses Google's Gemini model to determine which robot function to call and with what parameters.
- **Text-to-Speech (TTS) Feedback:** The robot announces its actions and responses out loud.
- **ROS 2 Integration:** Publishes `geometry_msgs/msg/Twist` messages to `/cmd_vel` to control a robot or simulator.
- **Simple & Extensible:** The tool-based architecture makes it easy to add new capabilities to the robot.

## Prerequisites

- ROS 2 (Humble)
- A simulated or real robot that subscribes to `/cmd_vel` (e.g., TurtleBot3, turtlesim).
- Python 3
- A Google AI API Key.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/prathamv0811/LLM-Navigation_gemini.git
    cd llm-ros2-agent 
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install google-generativeai pyttsx3
    ```

3.  **Install TTS Engine (for Debian/Ubuntu):**
    ```bash
    sudo apt-get update && sudo apt-get install espeak
    ```

## How to Run

1.  **Launch your robot or simulator.** For example, to launch the TurtleBot3 simulation:
    ```bash
    export TURTLEBOT3_MODEL=burger
    ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
    ```

2.  **Run the agent node:**
    ```bash
    export GOOGLE_API_KEY='your-google-api-key-here'
    python3 agent_node.py
    ```

3.  Enter commands in the terminal when prompted.
