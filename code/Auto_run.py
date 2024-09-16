import numpy as np
import serial
import time
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import font
from datetime import datetime
from threading import Thread

start_time = time.time()
threshold=20

def send_serial_command(PFC, value, port):
    STX = '!'
    ID_1 = '1'
    ID_0 = '6'
    AI = '0'

    # Separate the tens and ones digits
    tens = PFC // 10
    ones = PFC % 10
    PFC_1 = str(tens)
    PFC_0 = str(ones)

    # Separate the individual digits
    letters = list(value)

    Value_5 = letters[0]
    Value_4 = letters[1]
    Value_3 = letters[2]
    Value_2 = letters[3]
    Value_1 = letters[4]
    Value_0 = letters[5]

    # Convert the ASCII characters to byte values
    data_bytes = np.array([ord(STX), ord(ID_1), ord(ID_0), ord(AI), ord(PFC_1), ord(PFC_0),
                           ord(Value_5), ord(Value_4), ord(Value_3), ord(Value_2), ord(Value_1), ord(Value_0)])

    # Calculate the CRC
    crc = np.fmod(np.sum(data_bytes), 256)

    # Convert the CRC to a 3-byte ASCII representation
    crc_ascii = '{:03d}'.format(crc)

    # Combine the pump input command
    message = "b'" + STX + ID_1 + ID_0 + AI + PFC_1 + PFC_0 + Value_5 + Value_4 + Value_3 + Value_2 + Value_1 + Value_0 + crc_ascii + "\n'"
    port.write(message.encode('utf-8'))


def send_command(PFC, value, port):
    STX = '!'
    ID_1 = '5'
    ID_0 = '1'
    AI = '0'

    # Separate the tens and ones digits
    tens = PFC // 10
    ones = PFC % 10
    PFC_1 = str(tens)
    PFC_0 = str(ones)

    # Separate the individual digits
    letters = list(value)

    Value_5 = letters[0]
    Value_4 = letters[1]
    Value_3 = letters[2]
    Value_2 = letters[3]
    Value_1 = letters[4]
    Value_0 = letters[5]

    # Convert the ASCII characters to byte values
    data_bytes = np.array([ord(STX), ord(ID_1), ord(ID_0), ord(AI), ord(PFC_1), ord(PFC_0),
                           ord(Value_5), ord(Value_4), ord(Value_3), ord(Value_2), ord(Value_1), ord(Value_0)])

    # Calculate the CRC
    crc = np.fmod(np.sum(data_bytes), 256)

    # Convert the CRC to a 3-byte ASCII representation
    crc_ascii = '{:03d}'.format(crc)

    # Combine the pump input command
    message = "b'" + STX + ID_1 + ID_0 + AI + PFC_1 + PFC_0 + Value_5 + Value_4 + Value_3 + Value_2 + Value_1 + Value_0 + crc_ascii + "\n'"
    port.write(message.encode('utf-8'))


def send_sample_command(AI, PFC, value, port):
    STX = '!'
    ID_1 = '7'
    ID_0 = '1'

    AI = str(AI)
    # Separate the tens and ones digits
    tens = PFC // 10
    ones = PFC % 10
    PFC_1 = str(tens)
    PFC_0 = str(ones)

    # Separate the individual digits
    letters = list(value)

    Value_5 = letters[0]
    Value_4 = letters[1]
    Value_3 = letters[2]
    Value_2 = letters[3]
    Value_1 = letters[4]
    Value_0 = letters[5]

    # Convert the ASCII characters to byte values
    data_bytes = np.array([ord(STX), ord(ID_1), ord(ID_0), ord(AI), ord(PFC_1), ord(PFC_0),
                           ord(Value_5), ord(Value_4), ord(Value_3), ord(Value_2), ord(Value_1), ord(Value_0)])

    # Calculate the CRC
    crc = np.fmod(np.sum(data_bytes), 256)

    # Convert the CRC to a 3-byte ASCII representation
    crc_ascii = '{:03d}'.format(crc)

    # Combine the pump input command
    message = "b'" + STX + ID_1 + ID_0 + AI + PFC_1 + PFC_0 + Value_5 + Value_4 + Value_3 + Value_2 + Value_1 + Value_0 + crc_ascii + "\n'"
    port.write(message.encode('utf-8'))
def send_command_to_port1():
    PFC = int(entry_pfc_port1.get())
    value = entry_value_port1.get()
    send_serial_command(PFC, value, port1)

def send_command_to_port2():
    PFC = int(entry_pfc_port2.get())
    value = entry_value_port2.get()
    send_serial_command(PFC, value, port2)

def send_command_to_port3():
    PFC = int(entry_pfc_port3.get())
    value = entry_value_port3.get()
    send_command(PFC, value, port3)

def send_command_to_port4():
    AI = entry_ai_port4.get()
    PFC = int(entry_pfc_port4.get())
    value = entry_value_port4.get()
    send_sample_command(AI, PFC, value, port4)

def read_data(port):
    # Read 16 bytes
    au = port.read(16)
    #print(f"Read data: {au}")
    if len(au) != 16:
        print("Error: Invalid response length")
        return None

    # Check if it's an error response
    if au == b'NACK':
        print("Error: NACK received")
        return None
    return au

def analyze_data_to_stop(data,step_index,flag,time_interval=50,interval_threshold=200):
    print("analyze_data_to_stop called...")
    '''
    :param data: processed AU1
    :param step_index: index of checkpoint
    :param time_interval: time interval of the checkpoint (200 step with 10s)
    :param interval_threshold: threshold for determining whether the peak has occurred
    :return: flag---whether to stop or not
    '''
    seq=data[0:time_interval*step_index].values    #use all data before the checkpoint
    if seq.size == 0:  # Add a check here
        return 0
    max_value=np.max(seq)

    #max_iter is 10000 step with 500s (can be adjusted)
    if time_interval*step_index>10000:
        flag=-1

    #if maximum of the seq<20, continue
    # if np.max(seq)<20:
    #     # if the maximum keeps<20 after 6000 steps, it may not absorb UV (can be adjusted)
    #     if time_interval * step_index > 6000:
    #         flag = False
    if flag>1:
        succeed_seq=data[time_interval*(step_index-1):time_interval*step_index].values
        if np.max(succeed_seq)>0.8:
            flag-=1
    # find the peak
    else:
        if flag == 0:
            core_index = np.where((seq > max_value - 20))[0]
            for core in core_index:
                if len(np.where(seq[0:core] < threshold)[0]) != 0:
                    left_bound = np.where(seq[0:core] < threshold)[0][-1]
                else:
                    left_bound=0
                if len(np.where(seq[core:] < threshold)[0]) != 0:
                    right_bound = np.where(seq[core:] < threshold)[0][0] + core
                else:
                    right_bound=time_interval*step_index

                # if the peak is complete and the length of peak larger than interval_threshold, stop
                if right_bound!=time_interval*step_index and right_bound-left_bound>interval_threshold and left_bound!=0:
                    flag=1
        else:
            flag+=1
    print(f"analyze_data_to_stop returns: {flag}")
    return flag

def start_reading_data():
    port3.write(b'!51017000000063\n')
    port3.write(b'!51018000000065\n')
    port3.write(b'!51090000000064\n')

    # Initialize an empty DataFrame
    global data
    data = pd.DataFrame(columns=['AU', 'Transformed_AU', 'AU1'])

    running = 0
    continue_index=2
    last_analyzed_time = 0
    au_values = []  # Initialize a list to store the au values within each 5 second period
    global stop_time
    stop_time = time.time() + 2100  # Assuming you want to stop after 2100 seconds from the start

    while running not in [-1,continue_index]:
        #print("Starting a new loop iteration...")
        current_time = time.time()
        au = read_data(port3)

        # Collect data every 5 seconds
        if current_time - last_analyzed_time < 5:
            #print("Less than 5 seconds elapsed...")
            if au is not None:
                au_values.append(au)
        else:
            #print("5 seconds elapsed...")
            # Initialize a temporary DataFrame for current 5 seconds
            temp_data = pd.DataFrame(columns=['AU', 'Transformed_AU', 'AU1'])
            last_analyzed_time = current_time  # Reset the last_analyzed_time for the next 5 seconds

            for i in range(len(au_values)):
                # Transform the 'AU' values
                substring1_last, substring2_last = transform_byte_string(au_values[i])
                if i + 1 < len(au_values):
                    substring1, substring2 = transform_byte_string(au_values[i + 1])
                    transformed_AU = (substring2_last + substring1).decode('utf-8')
                    if transformed_AU.startswith('510'):
                    # Check the length of the transformed_AU value
                        if len(transformed_AU) == 14:
                            sliced_value = transformed_AU[5:11]
                            if len(sliced_value) == 6:
                                decimal_value = hex_to_decimal(sliced_value)

                                if decimal_value > 8388608:
                                    decimal_value = -16777215 + decimal_value
                                decimal_value = decimal_value * 1e-6
                                decimal_value = 1 - 10 ** (-decimal_value)
                                decimal_value = decimal_value * 100

                                # Append the valid values to the DataFrame
                                temp_data = pd.concat([temp_data, pd.DataFrame({
                                    'AU': [au_values[i]],
                                    'Transformed_AU': [transformed_AU],
                                    'AU1': [decimal_value]
                                })], ignore_index=True)

            # Append temp_data to the main DataFrame
            data = pd.concat([data, temp_data], ignore_index=True)
            step_index = int((current_time - start_time) / 5)
            # Ensure step_index is at least 10
            if step_index >= 10:
                #print(f"Calling analyze_data_to_stop with step_index = {step_index} and data['AU1'] = {data['AU1'].tolist()}")  # Add this line
                running = analyze_data_to_stop(data['AU1'], step_index,running)
                # Check the stop condition at the end of each 5-second interval

            # Reset the au_values for the next 5 seconds
            au_values = []
        #print("Finished a loop iteration...")
    # Save data to an Excel file
    filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".xlsx"
    data.to_excel(filename, index=False)
    return data

def transform_byte_string(byte_string):
    byte_string = byte_string.replace(b"'", b"")
    byte_string = byte_string.replace(b"b", b"")
    string = byte_string.decode('utf-8')
    backslash_index = byte_string.find(b"\n")
    jth_index = byte_string.find(b"!")

    substring1 = byte_string[0: backslash_index]
    substring2 = byte_string[jth_index + 1:]
    #print(substring1, substring2)
    return substring1, substring2

def hex_to_decimal(hex_string):
    return int(hex_string, 16)

def eluent_flow(proportion):
    if proportion == 'r1':
        send_serial_command(10, '000000', port1)
        send_serial_command(10, '001000', port2)
    elif proportion == 'r2':
        send_serial_command(10, '000000', port1)
        send_serial_command(10, '000500', port2)
    elif proportion == 'r3':
        send_serial_command(10, '000500', port1)
        send_serial_command(10, '000500', port2)
    elif proportion == 'r4':
        send_serial_command(10, '000250', port1)
        send_serial_command(10, '000250', port2)
    elif proportion == 'r5':
        send_serial_command(10, '000667', port1)
        send_serial_command(10, '000333', port2)
    elif proportion == 'r6':
        send_serial_command(10, '000333', port1)
        send_serial_command(10, '000167', port2)
    elif proportion == 'r7':
        send_serial_command(10, '000833', port1)
        send_serial_command(10, '000167', port2)
    elif proportion == 'r8':
        send_serial_command(10, '000417', port1)
        send_serial_command(10, '000083', port2)
    elif proportion == 'r9':
        send_serial_command(10, '000909', port1)
        send_serial_command(10, '000091', port2)
    elif proportion == 'r10':
        send_serial_command(10, '000455', port1)
        send_serial_command(10, '000045', port2)
    elif proportion == 'r11':
        send_serial_command(10, '000952', port1)
        send_serial_command(10, '000048', port2)
    elif proportion == 'r12':
        send_serial_command(10, '000476', port1)
        send_serial_command(10, '000024', port2)
    elif proportion == 'r13':
        send_serial_command(10, '001000', port1)
        send_serial_command(10, '000020', port2)
    elif proportion == 'r14':
        send_serial_command(10, '000500', port1)
        send_serial_command(10, '000010', port2)

def auto_column(condition):
    if condition == 'case1':
        # 发送一个命令
        send_sample_command('0', 10, '010000', port4)
        send_sample_command('1', 10, '010000', port4)
        send_sample_command('0', 25, '000350', port4)
    elif condition == 'case2':
        # 发送另一个命令
        send_sample_command('0', 10, '010001', port4)
        send_sample_command('1', 10, '010001', port4)
        send_sample_command('0', 25, '000400', port4)
    elif condition == 'case3':
        send_sample_command('0', 10, '010002', port4)
        send_sample_command('1', 10, '010002', port4)
        send_sample_command('0', 25, '000450', port4)
    #port4.write(b'!71010010000059\n')#试管初始位置右盘第一行第一列
    #port4.write(b'!71110010000060\n')#试管结束位置右盘第一行第一列
    #port4.write(b'!71025000400068\n')#进样体积400ul
    port4.write(b'!71026000500070\n')#清洗体积500ul
    port4.write(b'!71028000001068\n')#单次进样
    port4.write(b'!71029000100069\n')#预设分析时间1min
    port4.write(b'!71030000000060\n')#启动进样
    time.sleep(4)
    # 在这里调用 start_reading_data 函数
    start_reading_data()


def start_auto_column():
    proportions = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14']
    conditions = ['case1', 'case2', 'case3']
    send_serial_command(15, '000000', port1)
    send_serial_command(15, '000000', port2)
    for proportion in proportions:
        eluent_flow(proportion)
        time.sleep(20)  # 等待20秒
        for condition in conditions:
            auto_column(condition)

if __name__ == '__main__':
    port1 = serial.Serial("COM3", 9600, timeout=None)
    port2 = serial.Serial("COM4", 9600, timeout=None)
    port3 = serial.Serial("COM6", 9600, timeout=None)
    port4 = serial.Serial("COM7", 9600, timeout=None)

    # Create the main window
    root = tk.Tk()
    root.title("Serial Command Sender")

    # Set the global font
    global_font = font.Font(family="Helvetica", size=12)

    # Create the mainframe
    mainframe = ttk.Frame(root, padding="20 20 20 20")
    mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Port1 Entry and Button
    ttk.Label(mainframe, text="Pumb1", font=global_font).grid(column=1, row=1, sticky=tk.W)
    entry_pfc_port1 = ttk.Entry(mainframe, width=7)
    entry_pfc_port1.grid(column=2, row=1, sticky=(tk.W, tk.E))
    entry_value_port1 = ttk.Entry(mainframe, width=7)
    entry_value_port1.grid(column=3, row=1, sticky=(tk.W, tk.E))
    ttk.Button(mainframe, text="Send Command", command=send_command_to_port1).grid(column=4, row=1, sticky=tk.W)

    # Port2 Entry and Button
    ttk.Label(mainframe, text="Pumb2", font=global_font).grid(column=1, row=2, sticky=tk.W)
    entry_pfc_port2 = ttk.Entry(mainframe, width=7)
    entry_pfc_port2.grid(column=2, row=2, sticky=(tk.W, tk.E))
    entry_value_port2 = ttk.Entry(mainframe, width=7)
    entry_value_port2.grid(column=3, row=2, sticky=(tk.W, tk.E))
    ttk.Button(mainframe, text="Send Command", command=send_command_to_port2).grid(column=4, row=2, sticky=tk.W)

    # Port3 Entry and Button
    ttk.Label(mainframe, text="Detector", font=global_font).grid(column=1, row=3, sticky=tk.W)
    entry_pfc_port3 = ttk.Entry(mainframe, width=7)
    entry_pfc_port3.grid(column=2, row=3, sticky=(tk.W, tk.E))
    entry_value_port3 = ttk.Entry(mainframe, width=7)
    entry_value_port3.grid(column=3, row=3, sticky=(tk.W, tk.E))
    ttk.Button(mainframe, text="Send Command", command=send_command_to_port3).grid(column=4, row=3, sticky=tk.W)

    # Port4 Entry and Button
    ttk.Label(mainframe, text="Sample", font=global_font).grid(column=1, row=4, sticky=tk.W)
    entry_ai_port4 = ttk.Entry(mainframe, width=7)
    entry_ai_port4.grid(column=2, row=4, sticky=(tk.W, tk.E))
    entry_pfc_port4 = ttk.Entry(mainframe, width=7)
    entry_pfc_port4.grid(column=3, row=4, sticky=(tk.W, tk.E))
    entry_value_port4 = ttk.Entry(mainframe, width=7)
    entry_value_port4.grid(column=4, row=4, sticky=(tk.W, tk.E))
    ttk.Button(mainframe, text="Send Command", command=send_command_to_port4).grid(column=5, row=4, sticky=tk.W)

    ttk.Button(mainframe, text="Start", command=start_auto_column).grid(column=3, row=5, sticky=tk.W)

    # Set the focus
    entry_pfc_port1.focus()

    # Run the main loop
    root.mainloop()

    # Close the ports
    port1.write(b'!16016000000063\n')
    port1.close()
    port2.write(b'!16016000000063\n')
    port2.close()
    port3.close()
    port4.close()
