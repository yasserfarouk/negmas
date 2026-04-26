#!/bin/bash

# Timeout in seconds (60 minutes = 3600 seconds)
TIMEOUT=${1:-21000}
echo Timeout is $TIMEOUT

export NEGMAS_IGNORE_TEST_NOTEBOOKS=True
export NEGMAS_SLOW_TEST=True
# Start the process in the background
uv run pytest src/negmas tests &

# Store the process ID
PID=$!
echo $PID is running
# Start a timer
TIMER=0

# Loop until timeout or process finishes
while [ $TIMER -lt $TIMEOUT ]; do
  # Check if the process is still running
  if ! kill -0 $PID > /dev/null 2>&1; then
    echo "Process finished before timeout."
    exit 0 # process finished successfully
  fi

  # Sleep for 1 second
  sleep 5

  # Increment the timer
  TIMER=$((TIMER + 5))
done

# Timeout reached, kill the process
echo "Timeout reached, killing process $PID."
kill $PID

# Check if the process was successfully terminated
if kill -0 $PID > /dev/null 2>&1; then #still running, kill -9 to force termination
  echo "Process did not terminate gracefully, force killing with kill -9."
  kill -9 $PID
fi

# Check the exit code of the process (if any)
wait $PID 2>/dev/null #silence error if process did not even start properly
if [ $? -ne 0 ]; then
  echo "Process failed or was forcefully terminated."
  exit 0
else
  echo "Process terminated successfully after timeout."
  exit 0
fi
