
# Credit to Maochuan Zhang

def get_trace(t_start,t_final):
    ## Reads in a start and end time string in the format '2020-08-14T10:00:00.0Z'
    ## Retrieves waveforms as a Stream object from IRIS DMC using ObsPy """
    ## Saves waveform and station metadata and response data in a dictionary """
    ## Saves that dictionary as a .mat file to be accessed in matlab """
    
    ## Internal specifications:
    ## station codes, station locations, station channels, station names


    from obspy.clients.fdsn import Client
    from obspy.core.utcdatetime import UTCDateTime
    from obspy import read, read_inventory
    from obspy.core import inventory
    import numpy as np
    import matplotlib as plt
    import scipy.io as sio
    import pickle
    client = Client("IRIS")

    # Initialize
    channel = ['HHE','HHN','HHZ','HDH','EHE','EHN','EHZ','HHE','HHN','HHZ','HDH','EHE','EHN','EHZ','EHE','EHN','EHZ','EHE','EHN','EHZ','EHE','EHN','EHZ','HHE','HHN','HHZ','HDH',]
    location = ['','','','','','','','','','','','','','','','','','','','','','','','','','','',]
    stationcode = ['AXCC1','AXCC1','AXCC1','AXCC1','AXEC1','AXEC1','AXEC1','AXEC2','AXEC2','AXEC2','AXEC2','AXEC3','AXEC3','AXEC3','AXAS1','AXAS1','AXAS1','AXAS2','AXAS2','AXAS2','AXID1','AXID1','AXID1','AXBA1','AXBA1','AXBA1','AXBA1']
    #stationcode = ['NCHR','KEMF','KEMF_W3','KEMO','ENWF','ENHR']

    


    t_start = UTCDateTime(t_start)
    t_final = UTCDateTime(t_final)
    
    # AFTER SEPTEMBER 10 2020 (KEMO comes back online with new channels)
    # if t_start > UTCDateTime("2020-09-10T00:00:00.0"):
    #    channel = ['EH*','EH*','CN*','HH*','HH*','HH*',]

    # Loop through stations
    stations = []; sampleRate = []; sampleCount = []; locations = [];
    channels = []; stime = []; etime = []; data = []; networks = [];
    sensitivityFrequency = []; sensitivity = [];
    for i in range(len(channel)):
        try:
            stream = client.get_waveforms(network='OO',station=stationcode[i],location=location[i],channel=channel[i],starttime = t_start,endtime = t_final,attach_response=True)
            for j in range(len(stream)):
                trace = stream[j]
                resp = trace.stats.response._get_overall_sensitivity_and_gain()
                stations.append(trace.stats.station)
                sampleRate.append(trace.stats.sampling_rate)
                sampleCount.append(trace.stats.npts)
                locations.append(trace.stats.location)
                channels.append(trace.stats.channel)
                stime.append(trace.stats.starttime.strftime("%Y-%m-%d:%H:%M:%S.%f"))
                etime.append(trace.stats.endtime.strftime("%Y-%m-%d:%H:%M:%S.%f")) 
                data.append(trace.data)
                networks.append(trace.stats.network)
                sensitivityFrequency.append(resp[0])
                sensitivity.append(resp[1])
                print('Found data for ',stationcode[i],' ',trace.stats.channel)
        except:
           print('No data for station',stationcode[i])
           continue

    # Save traces
    trace_dict = {'network': networks, 'station': 	stations,'location':locations,'channel':channels,'sensitivity':sensitivity,'sensitivityFrequency':sensitivityFrequency,'data':data,'sampleCount':sampleCount,'sampleRate':sampleRate,'startTime':stime,'endTime':etime}
    
    with open("trace_metadata.pkl", "wb") as f:
        pickle.dump(trace_dict, f)

    return(trace_dict)
