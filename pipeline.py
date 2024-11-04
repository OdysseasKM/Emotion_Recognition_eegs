from functions import *

# Define parameters
channel_names_geneva = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 
                        'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 
                        'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
participant_ids = [f"s{i:02d}" for i in range(1, 6)]  # Assuming 5 participants

# Load the raw EEG data for each participant
raw_data = {participant_id: load_eeg_data(participant_id) for participant_id in participant_ids}

# Process (reorder) the participants' data
eeg_data_reordered = process_participants(participant_ids, channel_names_geneva)

# Print information about the loaded and reordered data
process_participant_info(eeg_data_reordered, raw_data, show_plots=False)  # Toggle `show_plots` to True if you want to see plots

# Load and sort participant ratings
ratings_file_path = "participant_ratings.csv"
df = load_participant_ratings(ratings_file_path)  # Load ratings data
df_sorted = sort_ratings(df)  # Sort the data

# Display the sorted DataFrame
print(df_sorted)

# Process trial data for all participants
participant_trials = process_participant_trials(participant_ids, df_sorted, eeg_data_reordered)

# Example: Print the number of trials for each participant
for participant_id, trials in participant_trials.items():
    print(f"Participant ID: {participant_id}, Number of Trials: {len(trials)}")

# Plot PSD for the first trial data of the first participant
first_participant_trials = participant_trials['s01']
first_trial_data = first_participant_trials[0]

# Plot the first trial data
first_trial_data.plot()
# Plot the power spectral density (PSD)
first_trial_data.compute_psd().plot()

# Apply Common Average Reference (CAR)
car_participant_data = {}
for participant_id, trials in participant_trials.items():
    car_participant_data[participant_id] = apply_common_average_reference(trials)

# Apply band-pass filter
low_freq = 4  # Lower cutoff frequency in Hz
high_freq = 45  # Upper cutoff frequency in Hz
filtered_participant_data = {}
for participant_id, trials in car_participant_data.items():
    filtered_participant_data[participant_id] = apply_bandpass_filter(trials, low_freq, high_freq)

# Apply notch filter
notch_freqs = [50, 60]  # Notch filter frequencies in Hz
notch_filtered_participant_data = {}
for participant_id, trials in filtered_participant_data.items():
    notch_filtered_participant_data[participant_id] = apply_notch_filter(trials, notch_freqs)

# Resample the data
new_sampling_rate = 128  # Desired sampling rate
resampled_participant_data = {}
for participant_id, trials in notch_filtered_participant_data.items():
    resampled_participant_data[participant_id] = apply_resampling(trials, new_sampling_rate)

# Example: Plot the EEG data for the first trial of the first participant after resampling
first_participant_id = 's01'
first_resampled_trial = resampled_participant_data[first_participant_id][0]

# Plot the trial data after resampling
first_resampled_trial.plot(title="EEG Data for Participant s01 - Trial 1 (After Preprocessing and Resampling)")

# Set the montage for the EEG data
montage = mne.channels.make_standard_montage('standard_1020')

# Apply ICA to clean the EEG data for each participant
cleaned_participant_data = {}
for participant_id, trials in resampled_participant_data.items():
    cleaned_trials, ica_models = perform_ica(trials, montage)
    cleaned_participant_data[participant_id] = cleaned_trials

# Example: Plot the cleaned data for the first trial of the first participant
first_participant_id = 's01'
first_cleaned_trial = cleaned_participant_data[first_participant_id][0]

# Plot the cleaned trial data
first_cleaned_trial.plot(title="Cleaned EEG Data for Participant s01 - Trial 1")
first_cleaned_trial.compute_psd().plot()

# Epoch the data for each participant's trials
epoched_participant_data = epoch_trials(resampled_participant_data, epoch_duration=1.0, discard_duration=3.0)

# Create epoch DataFrame for the first few participants
epoch_df = create_epoch_dataframe(df_sorted, num_participants=32, epochs_per_trial=56)

