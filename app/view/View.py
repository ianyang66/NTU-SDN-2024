'''
Description: 
Version: 1.0
Autor: Ian Yang
Date: 2024-04-25 15:01:22
LastEditors: Ian Yang
LastEditTime: 2024-04-28 20:27:07
'''
import threading
import tkinter
from tkinter import Tk
from matplotlib import pyplot as plt
from queue import Queue
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from app.controllers.DDoSControllerThread import DDoSControllerThread
from app.model.Features import Feature
from app.model.TrafficState import TrafficState
import matplotlib.dates as mdates

class View:
    def __init__(self):
        self.queue = Queue()
        self.timestamps = []
        self.features = [[], [], [], [], []]

    def start_controller(self):
        controller = DDoSControllerThread(queue=self.queue)
        controller_thread = threading.Thread(target=controller.run, daemon=True)
        controller_thread.start()

    def init_GUI(self):
        root = Tk()
        root.title('DDoS protect')
        root.config(background='white')
        root.geometry("1280x720")
        traffic_state_label = tkinter.Label(root, text="", bg="grey", fg="white", font=("monospace", 16), width=20, height=3)
        traffic_state_label.pack(pady=(30, 0))

        fig = Figure()
        ax = fig.add_subplot(1, 1, 1)

        graph = FigureCanvasTkAgg(fig, master=root)
        graph.get_tk_widget().pack(fill=tkinter.BOTH, expand=True)

        threading.Thread(target=self.update_charts, args=(traffic_state_label, ax, graph)).start()
        root.mainloop()

    def update_charts(self, traffic_state_label, ax, graph):
        feature_colors = ['r', 'g', 'b', 'c', 'm']  # Define colors for each feature
        feature_labels = [feature.value for feature in Feature]  # Get feature labels

        while True:
            data = self.queue.get()
            traffic_state = data.get_traffic_state()
            traffic_state_label.config(text=traffic_state.name, bg="green" if traffic_state == TrafficState.NORMAL else "red")
            timestamp = data.get_timestamp()
            self.timestamps.append(timestamp)
            self.timestamps = self.timestamps[-50:]

            ax.cla()
            ax.grid()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.setp(ax.get_xticklabels(), rotation=15)
            plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=1))

            for i, (feature_label, value) in enumerate(data.get_features().get_features_as_array()):
                self.features[i].append(value)
                self.features[i] = self.features[i][-50:]
                ax.plot(self.timestamps, self.features[i], marker='o', color=feature_colors[i], label=feature_label.value)

            ax.legend(feature_labels, loc='upper left')
            graph.draw()

if __name__ == "__main__":
    view = View()
    view.start_controller()
    view.init_GUI()