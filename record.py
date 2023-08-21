import numpy as np
import os.path
import os
from contextlib import contextmanager
from EDFlib import edfreader, edfwriter

from raw_signal import Signal

class Annotation(edfreader.EDFreader.EDFAnnotationStruct):
    """
    Represents annotation in EDF or BDF file.
    
    Attributes
    ----------
    onset : int | float
        Timestamp contataining the start of an annotation in units of 100 nanoseconds.
    description : str
        Description of an annotation.
    duration: int | float
        Duration of an annotation in units of 100 nanoseconds.
    """
    pass


def read_file(file_path: str | os.PathLike, info: bool = False, # TODO: caching
              extra_info: bool = False, get_annots: bool = False, 
              ignore_acc: bool = True) -> 'Record':
    """
    Read EDF or BDF file and create a Record instance from the data.

    Parameters
    ----------
    file_path : str | os.PathLike
        Path to EDF or BDF file.
    info : bool, optional
        If `True`, prints information about each signal to the console. The 
        default is False.
    extra_info : bool, optional
        If `True`, prints extra information. Does nothing when `info` is set to
        `False`. The default is False.
    get_annots : bool, optional
        If `True`, creates a Record object, containing annotations from the file.
        Else `Record.annotations` will be `None`. The default is False.
    ignore_acc : bool, optional
        If `True`, ignores additional signals, i.e. signals from accelerometers.
        The default is True.

    Returns
    -------
    output: Record
        Record object filled with data from chosen file.

    """
    hdl = edfreader.EDFreader(str(file_path))
    
    signals_amount = hdl.getNumSignals()
    
    # Amount of samples in each signal
    samples = np.zeros(signals_amount, dtype=np.int32)
    
    # Contains numbers of signals to ignore
    skip = set()
    
    if info:
        print("Number of signals in file: %d" % (hdl.getNumSignals()))
        print("Recording duration: %f s" %
              (hdl.getLongDataRecordDuration() / 10000000.0 *
               hdl.getNumDataRecords()))
        print("\nSignal list:")
    
    for i in range(signals_amount):
        if info:
            print("\nSig. %d: %s" % (i, hdl.getSignalLabel(i)))
        
        name = hdl.getSignalLabel(i)
        if (name.find('Accel') != -1 or name.find('Gyro') != -1 or
            name.find('Qual') != -1) and ignore_acc:
            skip.add(i)
            
        samples[i] = hdl.getTotalSamples(i)
        
        if extra_info and info:
            print("Sample frequency: %f Hz" % (hdl.getSampleFrequency(i)))
            print("Physical dimension: %s" % (hdl.getPhysicalDimension(i)))
            print("Physical minimum: %f" % (hdl.getPhysicalMinimum(i)))
            print("Physical maximum: %f" % (hdl.getPhysicalMaximum(i)))
            print("Digital minimum: %d" % (hdl.getDigitalMinimum(i)))
            print("Digital maximum: %d" % (hdl.getDigitalMaximum(i)))
            print("Total samples in file: %d" % (hdl.getTotalSamples(i)))  
    
    
    signals = []
    for i in range(signals_amount):
        if i not in skip:
            sig = np.empty(samples[i], dtype=np.int32)
            hdl.rewind(0)
            hdl.readSamples(i, sig, samples[i])
            
            signals.append(Signal(sig, hdl.getSampleFrequency(i)))
    
    annotations = None
    if get_annots:
        annotations = hdl.annotationslist
        
        if info:
            n = len(hdl.annotationslist)
            print("\nAnnotations in file: %d" %(n))
    
    hdl.close()
    
    return Record(signals, annotations)


class Record:
    """
    Represents EDF or BDF record, contains signals and annotations.
    
    Attributes
    ----------
    signals : list[Signal]
        List with signals represented as Signal objects.
    size : int
        Amount of signals in the record.
    annotations: list[Annotation]
        List with annotations for particular record.
    """
    def __init__(self, raw_signals: list[Signal] = list(),
                 annots: list[Annotation] = list()):
        """
        Initialises Record object.
        Parameters
        ----------
        raw_signals : list[Signal], optional
            List of Signal objects containing data for each signal in the record.
            The default is None.
        annots : list[Annotation], optional
            List of Annotation objects for the record. The default is None.

        Returns
        -------
        None.
        """
        self.signals = raw_signals
        self.size = len(self.signals)
        self.annotations = annots
        return
        
    
    def write_file(self, file_path: str | os.PathLike,
                   current_time: bool = True) -> None:
        """
        Write Record object to BDF file.

        Parameters
        ----------
        file_path : str | os.PathLike
            Desired path to store the file.
        current_time : bool
            If `True`, uses actual system time to write to the resulting file.
            The default is True.

        Raises
        ------
        NotImplementedError
            When selected file extension differs from .bdf.
        RuntimeError
            If there is a problem with internal writing algorithm.

        Returns
        -------
        None.

        """
        if os.path.splitext(file_path)[1] != '.bdf':
            raise NotImplementedError('Unsupported file type. Only .bdf files are' 
                             'currently supported.')
        
        if self.size == 0:
            raise RuntimeError('Cannot write Record without signals to file.')
        
        for i in range(1, self.size):
            if not np.isclose(self.signals[i].fs, self.signals[0].fs):
                raise RuntimeError('Cannot write signals with different '
                                   'samping frequencies.')
        
        fs = int(self.signals[0].fs)
        
        @contextmanager
        def manage_writer(f_path, f_type, signals_amount):
            writer = edfwriter.EDFwriter(f_path, f_type, signals_amount)
            try:
                yield writer
            finally:
                writer.close()
            
        with manage_writer(str(file_path),
                           edfwriter.EDFwriter.EDFLIB_FILETYPE_BDFPLUS,
                           self.size) as hdl:
            phys_max = 0
            phys_min = 0
            for sig in self.signals:
                tmp = np.max(sig)
                if tmp > phys_max:
                    phys_max = tmp
                
                tmp = np.min(sig)
                if tmp < phys_min:
                    phys_min = tmp
            
            for chan in range(0, self.size):
                if (hdl.setPhysicalMaximum(chan, phys_max) != 0 or 
                hdl.setPhysicalMinimum(chan, phys_min) != 0 or
                hdl.setDigitalMaximum(chan, 8388607) != 0 or
                hdl.setDigitalMinimum(chan, -8388608) != 0 or
                hdl.setPhysicalDimension(chan, "uV") != 0 or
                hdl.setSignalLabel(chan, f"Channel {chan+1}") != 0 or
                hdl.setSampleFrequency(chan, fs) != 0):
                    raise RuntimeError('Error: cannot set one of record parameters.')
            
            if not current_time:
                hdl.setStartDateTime(1985, 1, 1, 0, 0, 0, 0)
            
            tmp = 0
            for _ in range(int(np.ceil(self.signals[0].size/fs))):
                for i in range(self.size):
                    err = hdl.writeSamples(self.signals[i][tmp:tmp+fs])
                tmp += fs
            
            if err != 0:
                raise RuntimeError('Cannot write samples to the file,'
                                   f' error code is {err}.')
                
            if self.annotations is not None:
                for ann in self.annotations: 
                    err =  hdl.writeAnnotation(ann.onset/1000, # EDFlib magically requires
                                               ann.duration/1000, # timestamps here to be
                                               ann.description) # in units of 0.0001 second
                    if err != 0:
                        raise RuntimeError('Cannot write annotations to the' 
                                           f'file, error code is {err}.')
                    
        return
        
    
    
    def resample(self, new_freq: float | int,
                 min_freq: float | int = 0) -> None:
        """
        Resamples all signals in the record to the desired frequency.

        Parameters
        ----------
        new_freq : float | int
            Desired sampling frequency.
        min_freq : float | int, optional
            Minimal frequency for operation, if the frequency is lower than 
            this value, then raises RuntimeError. The default is 0.

        Raises
        ------
        RuntimeError
            If one of the signals has sampling frequency lower than min_freq.

        Returns
        -------
        None.
        """
        for i in range(self.size):
            if not np.isclose(self.signals[i].fs, new_freq, atol=1e-3):
                if self.signals[i].fs < min_freq:
                    raise RuntimeError(f'Cannot resample signal {i} to '
                                       'frequency {new_freq:.1f} Hz, as '
                                       'sampling frequency of the signal is '
                                       '{self.signals[i].fs:.1f} Hz, while '
                                       'minimal frequency is {min_freq:.1f} Hz.')
                self.signals[i].resample(new_freq)
        return
        
        


if __name__ == '__main__':
    r = read_file('tests/test_annot.bdf', get_annots=True)
    