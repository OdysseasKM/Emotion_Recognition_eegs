def load_eeg_data(participant_id, file_path_template="eegs/{participant_id}.bdf"):
    """
    Load EEG data for a given participant.
    
    Args:
    - participant_id (str): Participant ID.
    - file_path_template (str): Template for file path.
    
    Returns:
    - raw_eeg (mne.io.Raw): The raw EEG data.
    """
    file_path = file_path_template.format(participant_id=participant_id)
    raw = mne.io.read_raw_bdf(file_path, preload=True)
    
    # Keep only EEG channels
    raw_eeg = raw.pick_types(eeg=True)
    return raw_eeg


def reorder_eeg_channels(raw_eeg, channel_names_geneva):
    """
    Reorder EEG channels to Geneva order.
    
    Args:
    - raw_eeg (mne.io.Raw): The raw EEG data.
    - channel_names_geneva (list): List of channel names in Geneva order.
    
    Returns:
    - raw_reordered (mne.io.Raw): Reordered EEG data.
    """
    raw_reordered = raw_eeg.reorder_channels(channel_names_geneva)
    return raw_reordered


def process_participants(participant_ids, channel_names_geneva):
    """
    Process EEG data for all participants, load and reorder channels.
    
    Args:
    - participant_ids (list): List of participant IDs.
    - channel_names_geneva (list): List of channel names in Geneva order.
    
    Returns:
    - eeg_data_reordered (dict): Dictionary with participant ID as keys and reordered EEG data as values.
    """
    eeg_data_reordered = {}
    
    for participant_id in participant_ids:
        raw_eeg = load_eeg_data(participant_id)
        raw_reordered = reorder_eeg_channels(raw_eeg, channel_names_geneva)
        
        # Store reordered EEG data
        eeg_data_reordered[participant_id] = raw_reordered
    
    return eeg_data_reordered


def print_eeg_channel_info(participant_id, raw, raw_reordered):
    """
    Prints channel information before and after reordering for a participant.
    
    Args:
    - participant_id (str): The ID of the participant.
    - raw (mne.io.Raw): The original raw EEG data.
    - raw_reordered (mne.io.Raw): The reordered EEG data.
    """
    print(f"Participant ID: {participant_id}")
    
    # Print channel names before and after reordering
    print("Channel names before reordering:", raw.info['ch_names'])
    print("Channel names after reordering:", raw_reordered.info['ch_names'])
    
    # Print EEG data shape before and after reordering
    print("EEG data shape before reordering:", raw._data.shape)
    print("EEG data shape after reordering:", raw_reordered._data.shape)


import matplotlib.pyplot as plt

def plot_sensor_locations(raw, raw_reordered, show_plots=True):
    """
    Plots sensor locations before and after reordering.
    
    Args:
    - raw (mne.io.Raw): The original raw EEG data.
    - raw_reordered (mne.io.Raw): The reordered EEG data.
    - show_plots (bool): Whether to display the plots.
    """
    if show_plots:
        # Plot channel locations before reordering
        fig_before = raw.plot_sensors(show_names=True, title="Channel Locations Before Reordering")
        plt.show()
        
        # Plot channel locations after reordering
        fig_after = raw_reordered.plot_sensors(show_names=True, title="Channel Locations After Reordering")
        plt.show()



def process_participant_info(eeg_data_reordered, raw_data, show_plots=False):
    """
    Process and display information for each participant, including channel names, 
    data shapes, and sensor plots.
    
    Args:
    - eeg_data_reordered (dict): Dictionary with participant IDs as keys and reordered EEG data as values.
    - raw_data (dict): Dictionary with participant IDs as keys and original raw EEG data as values.
    - show_plots (bool): Whether to display sensor plots.
    """
    for participant_id, raw_reordered in eeg_data_reordered.items():
        raw = raw_data[participant_id]
        
        # Print EEG channel and data shape information
        print_eeg_channel_info(participant_id, raw, raw_reordered)
        
        # Optionally plot sensor locations
        if show_plots:
            plot_sensor_locations(raw, raw_reordered, show_plots)


import pandas as pd

def load_participant_ratings(file_path):
    """
    Load participant ratings from a CSV file.
    
    Args:
    - file_path (str): Path to the CSV file containing participant ratings.
    
    Returns:
    - DataFrame: A pandas DataFrame containing the participant ratings.
    """
    df = pd.read_csv(file_path)
    return df

def sort_ratings(df):
    """
    Sort the DataFrame by Participant ID and Experiment ID.
    
    Args:
    - df (DataFrame): The DataFrame containing participant ratings.
    
    Returns:
    - DataFrame: The sorted DataFrame.
    """
    df_sorted = df.sort_values(by=['Participant_id', 'Experiment_id'])
    return df_sorted



def extract_trial_data(participant_id_str, participant_data, raw_data):
    """
    Extracts trial data for a specific participant.
    
    Args:
    - participant_id_str (str): The ID of the participant.
    - participant_data (DataFrame): The DataFrame containing trial information for the participant.
    - raw_data (mne.io.Raw): The raw EEG data for the participant.
    
    Returns:
    - List: A list containing trial data for the participant.
    """
    trials = []
    
    # Iterate over each trial for the current participant
    for index, trial in participant_data.iterrows():
        # Extract the start time of the trial (in seconds)
        start_time = trial['Start_time'] / 1e6  # Convert microseconds to seconds
        
        # Define the start and end time of the trial (assuming 1-minute duration)
        end_time = start_time + 60  # 1-minute duration
        
        # Extract the trial data based on the start and end time
        trial_data = raw_data.copy().crop(tmin=start_time, tmax=end_time)
        
        # Store the trial data in the list for the current participant
        trials.append(trial_data)
        
        # Print participant ID and trial information (optional)
        # print(participant_id_str, trial)
    
    return trials

def process_participant_trials(participant_ids, trial_info, eeg_data_reordered):
    """
    Processes trial data for all participants.
    
    Args:
    - participant_ids (list): List of participant IDs.
    - trial_info (DataFrame): The sorted DataFrame containing trial information.
    - eeg_data_reordered (dict): Dictionary with participant IDs as keys and reordered EEG data as values.
    
    Returns:
    - dict: A dictionary containing trial data for each participant.
    """
    participant_trials = {participant_id: [] for participant_id in participant_ids}

    # Iterate over each participant's data
    for participant_id, participant_data in trial_info.groupby('Participant_id'):
        participant_id_str = f"s{participant_id:02d}"  # Convert participant_id to string format
        if participant_id_str not in participant_ids:
            continue
        
        # Get the raw EEG data for the current participant
        raw_data = eeg_data_reordered[participant_id_str]

        # Extract trial data for the current participant
        trials = extract_trial_data(participant_id_str, participant_data, raw_data)
        
        # Store trial data in the dictionary
        participant_trials[participant_id_str] = trials
    
    return participant_trials


def apply_common_average_reference(trials):
    """
    Apply Common Average Reference (CAR) to the given trials.

    Args:
    - trials (list): A list of mne.io.Raw objects representing the trials.

    Returns:
    - list: A list of mne.io.Raw objects with CAR applied.
    """
    trials_with_car = []
    
    for trial_data in trials:
        # Create a copy of the trial data
        trial_with_car = trial_data.copy()
        # Set the EEG reference to average and apply projection
        trial_with_car.set_eeg_reference('average', projection=True)
        trial_with_car.apply_proj()  # Apply the projection
        trials_with_car.append(trial_with_car)
    
    return trials_with_car


def apply_bandpass_filter(trials, low_freq, high_freq):
    """
    Apply band-pass filter to the given trials.

    Args:
    - trials (list): A list of mne.io.Raw objects representing the trials.
    - low_freq (float): Lower cutoff frequency in Hz.
    - high_freq (float): Upper cutoff frequency in Hz.

    Returns:
    - list: A list of mne.io.Raw objects with the band-pass filter applied.
    """
    filtered_trials = []
    
    for trial_data in trials:
        # Create a copy of the trial data
        filtered_trial = trial_data.copy()
        # Apply band-pass filter
        filtered_trial.filter(low_freq, high_freq)
        filtered_trials.append(filtered_trial)
    
    return filtered_trials


def apply_notch_filter(trials, notch_freqs):
    """
    Apply notch filters to the given trials.

    Args:
    - trials (list): A list of mne.io.Raw objects representing the trials.
    - notch_freqs (list): A list of frequencies to apply notch filters.

    Returns:
    - list: A list of mne.io.Raw objects with the notch filters applied.
    """
    notch_filtered_trials = []
    
    for trial_data in trials:
        # Create a copy of the trial data
        notch_filtered_trial = trial_data.copy()
        # Apply notch filter for each frequency in the list
        for freq in notch_freqs:
            notch_filtered_trial.notch_filter(freqs=freq, verbose=True)
        notch_filtered_trials.append(notch_filtered_trial)
    
    return notch_filtered_trials


def apply_resampling(trials, new_sampling_rate):
    """
    Resample the given trials to the new sampling rate.

    Args:
    - trials (list): A list of mne.io.Raw objects representing the trials.
    - new_sampling_rate (int): The desired sampling rate.

    Returns:
    - list: A list of mne.io.Raw objects resampled to the new sampling rate.
    """
    resampled_trials = []
    
    for trial_data in trials:
        # Resample trial data
        resampled_trial = trial_data.copy().resample(new_sampling_rate, npad="auto")
        resampled_trials.append(resampled_trial)
    
    return resampled_trials


from mne.preprocessing import ICA
from mne_icalabel import label_components

def perform_ica(trials, montage, variance_proportion=0.999):
    """
    Apply Independent Component Analysis (ICA) to clean EEG trials.

    Args:
    - trials (list): A list of mne.io.Raw objects representing the trials.
    - montage (mne.channels.DigMontage): The montage to set for the EEG data.
    - variance_proportion (float): Proportion of variance to explain (0.999 by default).

    Returns:
    - cleaned_trials (list): A list of cleaned mne.io.Raw objects.
    - ica_models (list): A list of ICA models fitted to each trial.
    """
    cleaned_trials = []
    ica_models = []

    for i, trial_data in enumerate(trials):
        # Set the montage
        trial_data.set_montage(montage)

        # Fit ICA with a proportion of variance
        ica = ICA(n_components=variance_proportion, random_state=97, max_iter=1000)
        ica.fit(trial_data)

        # Store the fitted ICA model
        ica_models.append(ica)

        # Optionally: Label components using mne_icalabel
        labels = label_components(trial_data, ica, method='iclabel')

        # Print labels for inspection
        print(f'Trial {i + 1} component labels:')
        print(labels)

        # Identify indices of components to exclude (not 'brain' or 'other')
        components_to_exclude = [j for j, label in enumerate(labels['labels']) if label not in ['brain', 'other']]
        ica.exclude = components_to_exclude

        # Apply ICA to the data, removing the unwanted components
        cleaned_data = ica.apply(trial_data, exclude=ica.exclude)

        # Store cleaned trial data
        cleaned_trials.append(cleaned_data)

        # Optionally: Show new labels after cleaning
        new_labels = label_components(cleaned_data, ica, method='iclabel')
        print(f'Trial {i + 1} cleaned component labels:')
        print(new_labels)

    return cleaned_trials, ica_models



def epoch_trials(trial_data_list, epoch_duration=1.0, discard_duration=3.0):
    """
    Epoch the EEG data for each trial in the provided list.

    Args:
    - trial_data_list (list): A list of mne.io.Raw objects representing the trials.
    - epoch_duration (float): Duration of each epoch in seconds (default: 1.0).
    - discard_duration (float): Duration to discard from the start of each trial in seconds (default: 3.0).

    Returns:
    - dict: A dictionary with participant IDs as keys and lists of epoched trials as values.
    """
    epoched_data = {}

    # Loop through each trial
    for trial_data in trial_data_list:
        participant_id = trial_data.info['participant_id']  # Assuming participant_id is stored in the info dict
        epoched_trials = []

        # Generate events for fixed-length epochs after discarding initial seconds
        events = mne.make_fixed_length_events(trial_data, start=discard_duration, duration=epoch_duration)

        # Check if events were successfully created
        if len(events) == 0:
            print(f"No events created for participant {participant_id}'s trial. Skipping.")
            continue

        # Create epochs from the events
        epochs = mne.Epochs(trial_data, events, tmin=0.0, tmax=epoch_duration, baseline=None, preload=True, detrend=1)

        # Append epochs if they contain data
        if epochs.get_data().size > 0:
            epoched_trials.append(epochs)
        else:
            print(f"No data after epoching for participant {participant_id}'s trial. Skipping.")

        epoched_data[participant_id] = epoched_trials

    return epoched_data


def create_epoch_dataframe(df, num_participants=5, epochs_per_trial=57):
    """
    Create a DataFrame containing epoch information for each trial.
    
    Args:
    - df_sorted (pd.DataFrame): Sorted DataFrame containing columns 'Participant_id', 'Experiment_id', 'Valence', 'Arousal'.
    - num_participants (int): Number of participants to include in the output (default: 5).
    - epochs_per_trial (int): Number of epochs per trial (default: 57).
    
    Returns:
    - epoch_df (pd.DataFrame): A DataFrame containing 'Participant_ID', 'Experiment_ID', 'Epoch_ID', 'Valence', 'Arousal'.
    """
    epoch_data = []  # List to store epoch data

    # Get unique participant IDs and limit to the specified number
    unique_participants = df['Participant_id'].unique()[:num_participants]

    # Iterate through each row in the sorted DataFrame
    for _, row in df.iterrows():
        participant_id = row['Participant_id']

        # Check if the participant is in the limited list
        if participant_id not in unique_participants:
            continue  # Skip if the participant is not in the selected range

        experiment_id = row['Experiment_id']
        valence = row['Valence']
        arousal = row['Arousal']

        # For each trial, replicate the valence and arousal for all epochs
        for epoch in range(1, epochs_per_trial + 1):
            epoch_data.append({
                'Participant_ID': participant_id,
                'Experiment_ID': experiment_id,
                'Epoch_ID': epoch,
                'Valence': valence,
                'Arousal': arousal
            })

    # Create a new DataFrame from the epoch data
    epoch_df = pd.DataFrame(epoch_data)
    
    return epoch_df

def compute_differential_entropy(signal, sampling_rate, frequency_band):
    """
    Compute the differential entropy of the given signal in the specified frequency band.

    Args:
    - signal (ndarray): The EEG signal to analyze.
    - sampling_rate (float): The sampling rate of the signal.
    - frequency_band (tuple): The frequency band as (low, high).

    Returns:
    - float: The computed differential entropy.
    """
    # Perform FFT
    fft_result = fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1/sampling_rate)

    # Select the frequencies within the specified band
    band_indices = np.where((freqs >= frequency_band[0]) & (freqs <= frequency_band[1]))[0]
    band_fft = fft_result[band_indices]

    # Calculate Power Spectral Density (PSD)
    band_psd = np.abs(band_fft)**2 / len(signal)
    
    # Compute Probability Density Function (PDF)
    band_pdf = band_psd / np.sum(band_psd)
    
    # Compute differential entropy
    diff_entropy = entropy(band_pdf)
    return diff_entropy


import pandas as pd
import numpy as np

def update_epoch_df_with_entropy(epoched_data, epoch_df, sampling_rate, frequency_bands):
    """
    Update the epoch DataFrame with differential entropy features for each epoch.

    Args:
    - epoched_data (dict): Dictionary with participant IDs as keys and lists of epochs as values.
    - epoch_df (pd.DataFrame): The existing DataFrame containing epoch information.
    - sampling_rate (int): The sampling rate of the EEG data.
    - frequency_bands (dict): A dictionary containing frequency band names as keys and tuples of (low, high) frequencies as values.

    Returns:
    - pd.DataFrame: The updated DataFrame with new columns for differential entropy features.
    """
    # Loop through each participant's epoched data
    for participant_id, epochs in epoched_data.items():
        for epoch_index, epoch in enumerate(epochs):
            epoch_data = epoch.get_data()  # Shape: (n_channels, n_times)
            channel_averages = np.mean(epoch_data, axis=0)  # Average across channels
            
            # Calculate differential entropy for each frequency band
            for band_name, frequency_band in frequency_bands.items():
                diff_entropy = compute_differential_entropy(channel_averages, sampling_rate, frequency_band)
                
                # Create a unique index for merging purposes
                epoch_id = f"{participant_id}_epoch_{epoch_index + 1}"
                
                # Update the DataFrame
                epoch_df.loc[epoch_df['Participant_ID'] == participant_id, f'Differential_Entropy_{band_name}'] = \
                    epoch_df.loc[epoch_df['Participant_ID'] == participant_id, f'Differential_Entropy_{band_name}'].fillna(diff_entropy)

    return epoch_df

