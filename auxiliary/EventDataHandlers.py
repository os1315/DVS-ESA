import numpy as np
import parse


def readEventList(str_address: str) -> np.ndarray:
    """
    Reads in an event list file and returns a numpy array with the events.
    Be careful that the file is not corrupt.
    """

    event_file = open(str_address, 'r')

    # First line is not a point
    _str_event = event_file.readline()

    all_packets = []

    while True:
        str_event = event_file.readline()
        packet = parse.parse("x: {:d}; y: {:d}; t: {:d}; p: {:d}", str_event)

        # This is really annoying, but it returns this ".Results" object that I have to unpack
        if packet is not None:
            x = packet[0]
            y = packet[1]
            t = packet[2]
            p = packet[3]
            all_packets.append([x, y, t, p])
        else:
            break

    event_file.close()

    return np.array(all_packets)


def splitByPolarity(event_data: np.ndarray) -> (np.ndarray, np.ndarray):
    pos_polarity = []
    neg_polarity = []

    for n in range(event_data.shape[0]):
        if event_data[n, 3] < 0:
            neg_polarity.append([event_data[n, 0], event_data[n, 1], event_data[n, 2]])
        else:
            pos_polarity.append([event_data[n, 0], event_data[n, 1], event_data[n, 2]])

    return np.array(neg_polarity), np.array(pos_polarity)
