codename=`lsb_release -sc`

# Make sure we are running a valid Ubuntu distribution
case $codename in
  "trusty" | "utopic" | "vivid" | "wily")
  ;;
  *)
    echo "This script will only work on Ubuntu trusty, utopic, vivid, and wily"
    exit 0
esac

# Add the OSRF repository
if [ ! -e /etc/apt/sources.list.d/gazebo-latest.list ]; then
  sudo sh -c "echo \"deb http://packages.osrfoundation.org/gazebo/ubuntu ${codename} main\" > /etc/apt/sources.list.d/gazebo-latest.list"
fi

# Download the OSRF keys
has_key=`apt-key list | grep "OSRF deb-builder"`

echo "Downloading keys"
if [ -z "$has_key" ]; then
  wget --quiet http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
fi

 
# Update apt
echo "Retrieving packages"
sudo apt-get update -qq
echo "OK"

# Install gazebo
echo "Installing Gazebo"
sudo apt-get install gazebo7 libgazebo7-dev

echo "Complete."
echo "Type gazebo to start the simulator."
