# Definition of events sent by server to the client. */
EVENTS = dict(
##    EV_STATE=0x01, # server has changed the state */
##    EV_CTIME=0x02, # current time of the song has changed */
##    EV_SRV_ERROR=0x04, # an error occurred */
##    EV_BUSY=0x05, # another client is connected to the server */
##    EV_DATA=0x06, # data in response to a request will arrive */
##    EV_BITRATE=0x07, # the bitrate has changed */
##    EV_RATE=0x08, # the rate has changed */
##    EV_CHANNELS=0x09, # the number of channels has changed */
##    EV_EXIT=0x0a, # the server is about to exit */
##    EV_PONG=0x0b, # response for CMD_PING */
##    EV_OPTIONS=0x0c, # the options has changed */
##    EV_SEND_PLIST=0x0d, # request for sending the playlist */
##    EV_TAGS=0x0e, # tags for the current file have changed */
##    EV_STATUS_MSG=0x0f, # followed by a status message */
##    EV_MIXER_CHANGE=0x10, # the mixer channel was changed */
##    EV_FILE_TAGS=0x11, # tags in a response for tags request */
##    EV_AVG_BITRATE=0x12, # average bitrate has changed (new song) */
##    EV_AUDIO_START=0x13, # playing of audio has started */
##    EV_AUDIO_STOP=0x14, # playing of audio has stopped */
##
##    # Events caused by a client that wants to modify the playlist (see
##    #* CMD_CLI_PLIST* commands). */
##    EV_PLIST_ADD=0x50, # add an item, followed by the file name */
##    EV_PLIST_DEL=0x51, # delete an item, followed by the file name */
##    EV_PLIST_MOVE=0x52, # move an item, followed by 2 file names */
##    EV_PLIST_CLEAR=0x53, # clear the playlist */
##
##    # These  events, though similar to the four previous are caused by server
##    # which takes care of clients' queue synchronization. */
##    EV_QUEUE_ADD=0x54,
##    EV_QUEUE_DEL=0x55,
##    EV_QUEUE_MOVE=0x56,
##    EV_QUEUE_CLEAR=0x57,

    # Definition of events sent by server to the client. */
    EV_STATE=0x01, #/* server has changed the state */
    EV_CTIME=0x02, #/* current time of the song has changed */
    EV_SRV_ERROR=0x04, #/* an error occurred */
    EV_BUSY=0x05, #/* another client is connected to the server */
    EV_DATA=0x06, #/* data in response to a request will arrive */
    EV_BITRATE=0x07, #/* the bitrate has changed */
    EV_RATE=0x08, #/* the rate has changed */
    EV_CHANNELS=0x09, #/* the number of channels has changed */
    EV_EXIT=0x0a, #/* the server is about to exit */
    EV_PONG=0x0b, #/* response for CMD_PING */
    EV_OPTIONS=0x0c, #/* the options has changed */
    EV_SEND_PLIST=0x0d, #/* request for sending the playlist */
    EV_TAGS=0x0e, #/* tags for the current file have changed */
    EV_STATUS_MSG=0x0f, #/* followed by a status message */
    EV_MIXER_CHANGE=0x10, #/* the mixer channel was changed */
    EV_FILE_TAGS=0x11, #/* tags in a response for tags request */
    EV_AVG_BITRATE=0x12, #/* average bitrate has changed (new song) */
    EV_AUDIO_START=0x13, #/* playing of audio has started */
    EV_AUDIO_STOP=0x14, #/* playing of audio has stopped */

    # Events caused by a client that wants to modify the playlist (see
    #* CMD_CLI_PLIST* commands). */
    EV_PLIST_ADD=0x50, #/* add an item, followed by the file name */
    EV_PLIST_DEL=0x51, #/* delete an item, followed by the file name */
    EV_PLIST_MOVE=0x52, #/* move an item, followed by 2 file names */
    EV_PLIST_CLEAR=0x53, #/* clear the playlist */

    # These events, though similar to the four previous are caused by server
    #* which takes care of clients' queue synchronization. */
    EV_QUEUE_ADD=0x54,
    EV_QUEUE_DEL=0x55,
    EV_QUEUE_MOVE=0x56,
    EV_QUEUE_CLEAR=0x57,

    # State of the server. */
    STATE_PLAY=0x01,
    STATE_STOP=0x02,
    STATE_PAUSE=0x03,

    # Definition of server commands. */
    CMD_PLAY=0x00, #/* play the first element on the list */
    CMD_LIST_CLEAR=0x01, #/* clear the list */
    CMD_LIST_ADD=0x02, #/* add an item to the list */
    CMD_STOP=0x04, #/* stop playing */
    CMD_PAUSE=0x05, #/* pause */
    CMD_UNPAUSE=0x06, #/* unpause */
    CMD_SET_OPTION=0x07, #/* set an option */
    CMD_GET_OPTION=0x08, #/* get an option */
    CMD_GET_CTIME=0x0d, #/* get the current song time */
    CMD_GET_SNAME=0x0f, #/* get the stream file name */
    CMD_NEXT=0x10, #/* start playing next song if available */
    CMD_QUIT=0x11, #/* shutdown the server */
    CMD_SEEK=0x12, #/* seek in the current stream */
    CMD_GET_STATE=0x13, #/* get the state */
    CMD_DISCONNECT=0x15, #/* disconnect from the server */
    CMD_GET_BITRATE=0x16, #/* get the bitrate */
    CMD_GET_RATE=0x17, #/* get the rate */
    CMD_GET_CHANNELS=0x18, #/* get the number of channels */
    CMD_PING=0x19, #/* request for EV_PONG */
    CMD_GET_MIXER=0x1a, #/* get the volume level */
    CMD_SET_MIXER=0x1b, #/* set the volume level */
    CMD_DELETE=0x1c, #/* delete an item from the playlist */
    CMD_SEND_PLIST_EVENTS=0x1d, #/* request for playlist events */
    CMD_PREV=0x20, #/* start playing previous song if available */
    CMD_SEND_PLIST=0x21, #/* send the playlist to the requesting client */
    CMD_GET_PLIST=0x22, #/* get the playlist from one of the clients */
    CMD_CAN_SEND_PLIST=0x23, #/* mark the client as able to send playlist */
    CMD_CLI_PLIST_ADD=0x24, #/* add an item to the client's playlist */
    CMD_CLI_PLIST_DEL=0x25, #/* delete an item from the client's playlist */
    CMD_CLI_PLIST_CLEAR=0x26, #/* clear the client's playlist */
    CMD_GET_SERIAL=0x27, #/* get an unique serial number */
    CMD_PLIST_SET_SERIAL=0x28, #/* assign a serial number to the server's playlist */
    CMD_LOCK=0x29, #/* acquire a lock */
    CMD_UNLOCK=0x2a, #/* release the lock */
    CMD_PLIST_GET_SERIAL=0x2b, #/* get the serial number of the server's playlist */
    CMD_GET_TAGS=0x2c, #/* get tags for the currently played file */
    CMD_TOGGLE_MIXER_CHANNEL=0x2d, #/* toggle the mixer channel */
    CMD_GET_MIXER_CHANNEL_NAME=0x2e, #/* get the mixer channel's name */
    CMD_GET_FILE_TAGS=0x2f, #/* get tags for the specified file */
    CMD_ABORT_TAGS_REQUESTS=0x30, #/* abort previous CMD_GET_FILE_TAGS requests up to some file */
    CMD_CLI_PLIST_MOVE=0x31, #/* move an item */
    CMD_LIST_MOVE=0x32, #/* move an item */
    CMD_GET_AVG_BITRATE=0x33, #/* get the average bitrate */

    CMD_TOGGLE_SOFTMIXER=0x34, #/* toggle use of softmixer */
    CMD_TOGGLE_EQUALIZER=0x35, #/* toggle use of equalizer */
    CMD_EQUALIZER_REFRESH=0x36, #/* refresh EQ-presets */
    CMD_EQUALIZER_PREV=0x37, #/* select previous eq-preset */
    CMD_EQUALIZER_NEXT=0x38, #/* select next eq-preset */

    CMD_TOGGLE_MAKE_MONO=0x39, #/* toggle mono mixing */
    CMD_JUMP_TO=0x3a, #/* jumps to a some position in the current stream */
    CMD_QUEUE_ADD=0x3b, #/* add an item to the queue */
    CMD_QUEUE_DEL=0x3c, #/* delete an item from the queue */
    CMD_QUEUE_MOVE=0x3d, #/* move an item in the queue */
    CMD_QUEUE_CLEAR=0x3e, #/* clear the queue */
    CMD_GET_QUEUE=0x3f, #/* request the queue from the server */

)
