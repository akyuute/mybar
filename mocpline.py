from events import EVENTS
from os.path import expanduser
import time
import asyncio
import inspect
from asyncio import subprocess as aiosp
import sys
import pathlib
import shlex
import textwrap


FILE = '~/.moc/socket2'
MSG_SIZE = 4
CSI = '\033['
event_lookup = {v: k for k, v in EVENTS.items()}


class MocpSocket:
    EV_STATE=0x01, # server has changed the state */
    EV_CTIME=0x02, # current time of the song has changed */
    EV_SRV_ERROR=0x04, # an error occurred */
    EV_BUSY=0x05, # another client is connected to the server */
    EV_DATA=0x06, # data in response to a request will arrive */
    EV_BITRATE=0x07, # the bitrate has changed */
    EV_RATE=0x08, # the rate has changed */
    EV_CHANNELS=0x09, # the number of channels has changed */
    EV_EXIT=0x0a, # the server is about to exit */
    EV_PONG=0x0b, # response for CMD_PING */
    EV_OPTIONS=0x0c, # the options has changed */
    EV_SEND_PLIST=0x0d, # request for sending the playlist */
    EV_TAGS=0x0e, # tags for the current file have changed */
    EV_STATUS_MSG=0x0f, # followed by a status message */
    EV_MIXER_CHANGE=0x10, # the mixer channel was changed */
    EV_FILE_TAGS=0x11, # tags in a response for tags request */
    EV_AVG_BITRATE=0x12, # average bitrate has changed (new song) */
    EV_AUDIO_START=0x13, # playing of audio has started */
    EV_AUDIO_STOP=0x14, # playing of audio has stopped */

    def __init__(self,
        file: str,

    ):
        self.file = expanduser(file)
        self.reader, self.writer = asyncio.run(self.connect())
    
    async def connect(self, file: str, msg_size: int):
        num_events = 1
        # setattr(self._sock, 'reader', None)
        # setattr(self._sock, 'writer', None)
        # print(dir(self._sock))
        return await asyncio.open_unix_connection(
            path=file,
            limit=msg_size
        )
        pass
        # self.reader = reader
        # self.writer = writer

        # print(reader)
        # return

##        counter = 0
##        try:
##            while True:
##                # before = time.time()
##                data = await reader.read(num_events)
##                if not data:
##                    print("Server no longer running")
##                    break
##                # evs = [event_lookup.get(int.from_bytes(b, BYTE_ORDER)) for b in data]
##                # evs = [ev for b in data if (ev := event_lookup.get(b)) is not None]
##
##                ev = event_lookup.get(int.from_bytes(data, sys.byteorder))
##                # if ev is not None:
##                    # pass
##                    # print(ev)
##                if ev == 'EV_CTIME':
##                    # print(ev)
##                    counter += 1 if counter < 100 else 0
##                    if counter % 2 == 0:
##                        times = await self._get_stats(r'%cs', r'%ts')
##                        if times is None:
##                            await asyncio.sleep(0.1)
##                            continue
##                        await self._fix_possible_time_advance(times)
##                    # else:
##                        # await self._loop.sleep(0.1)
##                    # after = time.time() - before
##                    # print(after)
##                    await self._incr_ctime() #.refresh_output()
##                    await asyncio.sleep(0.1)
##                    # print(f"{self._fields = }")
##                    yield
##                elif ev == 'EV_STATUS_MSG':
##                    await self.update_fields()
##                    # print("Updating fields...")
##                    # print(f"{self._fields = }")
##                    yield
##                elif ev == 'EV_STATE':
##                    # pass
##                    await self._toggle_state()
##                    # print(f"{self._fields['state'] = }")
##                    yield
##                elif ev == 'EV_AUDIO_STOP':
##                    await self._stop_audio()
##                    yield
##                # elif ev in ('EV_BITRATE', 'EV_RATE'):
##                    # pass
##                    # print(ev)
##                # else:
##                    # print(data)
##
##        except FileNotFoundError as e:
##            print(f"Socket file {file} does not exist.\nIs the server running?")
##            raise
##        except KeyboardInterrupt:
##            # pass
##            print("Exiting...")
##        finally:
##            # print("Closing socket...")
##            self._sock_writer.close()


class MocpStatusFields:
    pass


class MocStatusLine:
    _field_names = {
        # 'state',
        'state_icon',
        'playing_icon',
        'paused_icon',
        'stopped_icon',
        'title',
        'remaining_t',
    }

##    _track_names = {
##        'State',
##        'File',
##        'Title',
##        'Artist',
##        'SongTitle',
##        'Album',
##        'TotalTime',
##        'TimeLeft',
##        'TotalSec',
##        'CurrentTime',
##        'CurrentSec',
##        'Bitrate',
##        'AvgBitrate',
##        'Rate',
##    }

    def __init__(self,
        sockfile: str,
        fmt: str = None,
        max_width: int = 40,
        trunc_str='...',
        playing_icon='>',
        paused_icon='#',
        stopped_icon='X',
        title_fmt: str = None,
        time_fmt: str = None,
        **fields
    ):
        self.playing_icon = playing_icon
        self.paused_icon = paused_icon
        self.stopped_icon = stopped_icon
        if fmt is None:
            fmt = "[-{remaining_t}]{state_icon} {title}"
        self._fmt = fmt
        # if title_fmt is None:
            # title_fmt = '{Artist}{SongTitle}'
        self._title_fmt = title_fmt
        self._time_fmt = time_fmt

        self._trunc_str = trunc_str
        self._max_width = max_width

        self._loop = asyncio.new_event_loop()
        self._sock_loop = asyncio.new_event_loop()
        # self._track = asyncio.run(self.get_track())
        # self._track = self._loop.run_until_complete(self.get_track())

        self._fields = {f: '' for f in self._field_names}
        self._fields.update({
            'paused_icon': self.paused_icon,
            'playing_icon': self.playing_icon,
            'stopped_icon': self.playing_icon,
        })
        self._last_result = ""

        self._loop.run_until_complete(self._post_init())
        # self._loop.run_until_complete(self._handle_client(FILE, MSG_SIZE))

    async def _post_init(self):
        # self._fields = await self.update_fields(
            # self._used_fields.keys()
        # )
        self._line_gen = self._make_line_gen()
        await self._line_gen.asend(None)
        # self._track = await self.get_track()
        # print(self._track)
        await self.update_fields()

        # self._prim_fields = {}
        # self._field_coros = {}

    async def _cont_print_line(self, stream=sys.stdout, end='\r', hide_cursor=True):
        print("Starting continuous line...")
        # self._loop.run_until_complete(self._handle_client(FILE, MSG_SIZE))
        # self._sock_reader = self._loop.create_task(self._handle_client(FILE, MSG_SIZE))
        # print(await self._line_gen.asend(None))
        curs = CSI + '?25l' if hide_cursor else ""
        stream.write(curs)
        line = await self._line_gen.asend(None) #+ '\n' #'\r'
        # stream.flush()
        # stream.write('\033[K')
        stream.write('\x1b[2K')  # VT100 escape code to clear line
        stream.write(line + end)
        try:
            while True:
                await self._sock.asend(None)
                # print(await self._line_gen.asend(None))
                line = await self._line_gen.asend(None) #+ '\n' #'\r'
                # stream.flush()
                # stream.write('\033[K')
                stream.write('\x1b[2K')  # VT100 escape code to clear line
                stream.write(line + end)
        except StopAsyncIteration:
            return

    def run(self, stream=sys.stdout, end='\r', hide_cursor=True):
        try:
            self._sock = self._handle_client(FILE, MSG_SIZE)
            self._loop.run_until_complete(self._cont_print_line(stream, end, hide_cursor))
        except KeyboardInterrupt:
            print()
            # print("Caught KeyboardInterrupt")
            print("Exiting...")
        finally:
            self._sock_writer.close()
            stream.write(CSI + '?25h')

    async def get_line(self, **kwargs):
        print("Printing one line...")
        if self._line_gen is None:
            self._line_gen = await self._make_line_gen(**kwargs)
            # self._running = lines
        return await self._line_gen.asend(None)

    async def _make_line_gen(self,
        fmt=None,
        fields=None,
        track=None,
        last=None
    ):
        if fmt is None:
            fmt = self._fmt
        if fields is None:
            fields = self._fields
##        if track is None:
##            track = self._track
        if last is None:
            line = self._last_result
        over = self._trunc_str
        width = self._max_width

        # print(f"{fields = }")
        # print(f"{track = }")
            
        # fields.update(**track)
        fmtd = fmt.format(**fields)
        if width is None:
            line = fmtd
        else:
            line = fmtd[:width] + (over if len(fmtd) > width else '')

        updates = {}

        while True:
            # print(f"{self._track.get('SongTitle') = }")
            # print(f"{self._fields.get('title') = }")
            # print(f"{fmtd = }")
            # print(f"{line = }")

            if updates is not None:
                # last = fmt.format(**track, **fields, **newfields)
                fields.update(**updates)
                fmtd = fmt.format(**fields)
                if width is None:
                    line = fmtd
                else:
                    line = fmtd[:width] + (over if len(fmtd) > width else '')
                # line = fmtd[:width] + (over if len(fmtd) > width else '')
            ##print(f"{changed = }")
            ##print(f"{fields = }")
            ##print(f"{last = }")
            self._last_result = line
            # changed = (yield last)
            updates = (yield line)


    async def get_state_icon(self, state=None):
        # print("Getting new state icon...")
        if state is None:
            state = self._fields['State']
        if state == 'PLAY':
            return self.playing_icon
        elif state == 'PAUSE':
            return self.paused_icon 
        else:
            return self.stopped_icon

    async def get_title(self, track):
        # print("Getting new title...")
        filename_fallback = pathlib.Path(track.get('File', '')).stem
        # print(f"{filename_fallback = }")
        # self._track.get(self._title_fmt) or fallback
        # if self._title_fmt is None:
        title_fmt = await self.get_title_fmt() if self._title_fmt is None else self._title_fmt
        # else:
            # title_fmt = self._title_fmt
#        title = title_fmt.format(**track) or filename_fallback
        if title_fmt is None:
            title = filename_fallback
        else:
            title = title_fmt.format(**self._fields)
# #        title = title_fmt.format(**self._fields) or filename_fallback
        # print(f"{title_fmt = }")
        # self._fields['title'] = title
        return title

    async def get_title_fmt(self):
        if self._fields.get('Artist') and self._fields.get('SongTitle'):
            return '{Artist} - {SongTitle}'
        if self._fields.get('SongTitle'):
            return '{SongTitle}'
        # return '{title}'
        return None

    async def secs_to_struct_t(self, n: int):
        '''A generator that counts down from n seconds, yielding the
        remaining seconds and a struct_time object to represent them.'''
        # print("Making new remaining_t generator...")
        limit = 31708800
        # extra, n = divmod(s, limit)
        # print(f"{extra = }, {n = }, {s = }")
        # for it in range(extra + 1, 0, -1):
            # print(it)
        # n = s
        # if n <= limit:
        if n > limit:
            n = limit
        while n >= 0:
            # if s >= 31708800:
                # n -= 31708800
            mins, secs = divmod(n, 60)
            hours, mins = divmod(mins, 60)
            days, hours = divmod(hours, 24)
            timetuple = (0,0,0,hours,mins,secs,0,days,0)
            yield n, time.struct_time(timetuple)
            n -= 1
        # n = extra % limit
        # extra, n = divmod(n, limit)
        # print(f"{extra = }, {n = }, {s = }")
        while True:
            yield 0, time.struct_time((0,) * 9)


    async def get_rtime_s(self, ctime_s, ttime_s):
        '''Probably unnecessary...'''
        # print("Getting new rtime_s...")
        # return self._fields['ttime_s'] - self._fields['ctime_s'] 
        return ttime_s - ctime_s

    async def get_remaining_t(self, rtime_s=None):
        # print("Getting new remaining_t...")
        n, timetuple = await anext(self._rtime_struct_t_gen)
        # if rtime_s is None:
            # n, timetuple = await anext(self._rtime_struct_t_gen)
        # else:
            # In this case, we need to make a new struct_time generator:
            # n, timetuple = anext(self.secs_to_struct_t(rtime_s))
            # n, timetuple = await anext(self._rtime_struct_t_gen)
        # n, t = await anext(self.secs_to_struct_t(rtime_s))
        # n, t = next(self.secs_to_struct_t(392305))
        if self._time_fmt is not None:
            time_fmt = self.time_fmt
        elif n > 86399:
            time_fmt = "%-jd %H:%M:%S"
        elif n > 3599:
            time_fmt = "%H:%M:%S"
        # if n > 59:
        else:
            time_fmt = "%M:%S"
        return time.strftime(time_fmt, timetuple)
        # return time.strftime(await self._fmt_time(n), t)

    async def _incr_ctime(self):
        # print("Incrementing current time...")
        # updates = dict(
        updates = {
            'ctime_s': self._fields['ctime_s'] + 1,
            'remaining_t': await self.get_remaining_t()
        }
        # self._fields['ctime_s'] += 1
        # self._fields['remaining_t'] = await self.get_remaining_t()
        # await self._line_gen.asend('CHANGED')
        await self._line_gen.asend(updates)
        # print(self._fields['ctime_s'])

    async def get_track(self):
        # print("Getting new track...")
        cmd = "mocp -i"
        proc = await aiosp.create_subprocess_shell(
            cmd,
            stdout=aiosp.PIPE,
            stderr=aiosp.PIPE
        )
        out, err = await proc.communicate()

        if not out:
            print("No out. Here's stderr:")
            print(err)
##            self.ttime_s = int(track.get('TotalSec', 0))
##            self.ctime_s = int(track.get('CurrentSec', 0))
##            self.rtime_s = self.ttime_s - self.ctime_s
##            return {}

        lines = out.decode().strip().splitlines()
        track = {k: v for k, v in (l.split(': ', 1) for l in lines)}
        # print(f"{track['SongTitle'] = }")
        # print(f"{track['File'] = }")
        return track

    async def update_fields(self, *fields):
        # print()
        # print("Updating fields...")
        #if not fields:
            #fields = self._field_names
        # track = await self.get_track()
        # self._track = track
        # stats = await self._get_stats(fields)
        # if not self._track:
        # self._track = await self.get_track()
        # print(f"{self._track = }")
        track = await self.get_track()

        state = track.get('State')
        ttime_s = int(track.get('TotalSec', 0))
        ctime_s = int(track.get('CurrentSec', 0))
        # ttime_s = int(self._track.get('TotalSec', 0))
        # ctime_s = int(self._track.get('CurrentSec', 0))
        rtime_s = ttime_s - ctime_s

        self._rtime_struct_t_gen = self.secs_to_struct_t(rtime_s)

        updates = {
            # 'state': state,
            'state_icon': await self.get_state_icon(state),
            'title': await self.get_title(track),
            'ttime_s': ttime_s,
            'ctime_s': ctime_s,
            'rtime_s': rtime_s,
            # updates['rtime_struct_t_gen'] = self.secs_to_struct_t(rtime_s)
            'remaining_t': await self.get_remaining_t(rtime_s),

            # print(f"{self._track.get('File') = }")
        }
        updates.update(**track)

        # self._fields.update(**updates)
        await self._line_gen.asend(updates)
        # await self._line_gen.asend('CHANGED')

    async def _get_stats(self, *stats):
        # print("Getting individual stats...")
        field_str = '\n'.join(stats)
        quoted = shlex.quote(field_str)
        cmd = f"mocp -Q {quoted}"
        proc = await aiosp.create_subprocess_shell(
            cmd, 
            stdout=aiosp.PIPE,
            stderr=aiosp.PIPE
        )
        out, err = await proc.communicate()
        if out:
            # print(out)
            return out.decode().strip().splitlines()

    async def _stop_audio(self):
        await self._line_gen.asend({'State': 'STOP'})

    async def _toggle_state(self, state=None):
        # print("Toggling state...")
        # if self._track.get('State') != 'PLAY':
        if state is None:
            state = self._fields.get('State')

        if state != 'PLAY':
            newstate = 'PLAY'
        else:
            newstate = 'PAUSE'
        # self._fields['state'] = newstate
        # self._state = newstate
        # self._fields['state_icon'] = await self.get_state_icon(newstate)
        # # self._state_icon = await self.get_state_icon(state)
        # # await self._line_gen.asend('CHANGED')
        updates = {
            'State': newstate,
            'state_icon': await self.get_state_icon(newstate)
            }
        await self._line_gen.asend(updates)


    async def _update_state(self):
        '''Read the state from the `mocp -Q` command.'''
        print("Updating state with `mocp -Q`...")
        state = await self._get_stats('%state')
##        if state == 'STOP':
##            self.State = 
##        return await self._get_stat('%state')
        # self._fields['state'] = state
        # self._state = state
        # self._fields['state_icon'] = await self.get_state_icon(state)
        # self._state_icon = await self.get_state_icon(state)
        # await self._line_gen.asend('CHANGED')
        updates = {
            'State': state,
            'state_icon': await self.get_state_icon(state)
        }
        await self._line_gen.asend(updates)

    # async def _update_ctime(self):
    async def _fix_possible_time_advance(self, times):
        '''Updates all the time-related fields in case the
        current time of a song has advanced.'''
        # times = await self._get_stats(r'%cs', r'%ts')
        # if times is None:
            # return
        # print(f"{times = }")
        current, total = map(int, times)
        rtime_s = total - current
        self._rtime_struct_t_gen = self.secs_to_struct_t(rtime_s)
        remain = await self.get_remaining_t(rtime_s)
        # self._fields.update(
        updates = {
            'ctime_s': current,
            'ttime_s': total,
            'rtime_s': rtime_s,
            'remaining_t': remain
        }
        await self._line_gen.asend(updates)
        

    async def _handle_client(self, file: str, msg_size: int):
        event_size = 1
        # setattr(self._sock, 'reader', None)
        # setattr(self._sock, 'writer', None)
        # print(dir(self._sock))
        reader, writer = await asyncio.open_unix_connection(
            path=file,
            limit=msg_size
        )
        self._sock_reader = reader
        self._sock_writer = writer

        # print(reader)
        # return
        counter = 0
        try:
            while True:
                # before = time.time()
                data = await reader.read(event_size)
                if not data:
                    print("Server no longer running")
                    break
                # evs = [event_lookup.get(int.from_bytes(b, BYTE_ORDER)) for b in data]
                # evs = [ev for b in data if (ev := event_lookup.get(b)) is not None]

                ev = event_lookup.get(int.from_bytes(data, sys.byteorder))
                # if ev is not None:
                    # pass
                    # print(ev)
                if ev == 'EV_CTIME':
                    # print(ev)
                    counter += 1 if counter < 100 else 0
                    if counter % 2 == 0:
                        times = await self._get_stats(r'%cs', r'%ts')
                        if times is None:
                            await asyncio.sleep(0.1)
                            continue
                        await self._fix_possible_time_advance(times)
                    # else:
                        # await self._loop.sleep(0.1)
                    # after = time.time() - before
                    # print(after)
                    await self._incr_ctime() #.refresh_output()
                    await asyncio.sleep(0.1)
                    # print(f"{self._fields = }")
                    yield
                elif ev == 'EV_STATUS_MSG':
                    await self.update_fields()
                    # print("Updating fields...")
                    # print(f"{self._fields = }")
                    yield
                elif ev == 'EV_STATE':
                    # pass
                    await self._toggle_state()
                    # print(f"{self._fields['state'] = }")
                    yield
                elif ev == 'EV_AUDIO_STOP':
                    await self._stop_audio()
                    yield
                # elif ev in ('EV_BITRATE', 'EV_RATE'):
                    # pass
                    # print(ev)
                # else:
                    # print(data)

        except FileNotFoundError as e:
            print(f"Socket file {file} does not exist.\nIs the server running?")
            raise
        except KeyboardInterrupt:
            # pass
            print("Exiting...")
        finally:
            # print("Closing socket...")
            self._sock_writer.close()


l = MocStatusLine(FILE, fmt='[-{remaining_t}]{state_icon} {title}', max_width=None)
# l = MocStatusLine(FILE, fmt='[-{remaining_t}]{state_icon} {title}', max_width=None)


def run_an_async(call):
    return l._loop.run_until_complete(call)


