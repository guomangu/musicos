-- name: Setup Mix Buses & Timbral Glue
-- category: Mixer
-- description: Routes channels to BUS Drums / BUS 808 / BUS Music / BUS Vocal + FX Room with timbral glue and Kick sidechain.

function apply()
  local P = project
  local bus_drums = 10
  local bus_music = 11
  local bus_vocal = 12
  local bus_808   = 13
  local fx_room   = 14

  P.mixer.setName(bus_drums, "BUS Drums")
  P.mixer.setName(bus_music, "BUS Music")
  P.mixer.setName(bus_vocal, "BUS Vocal")
  P.mixer.setName(bus_808,   "BUS 808")
  P.mixer.setName(fx_room,   "FX Room")

  local kick_mixer = nil

  for ch = 1, P.channels.count() do
    local m = P.channels.route(ch)
    if m == 0 then
      m = ch
      P.channels.setRoute(ch, m)
    end
    local cname = string.lower(P.channels.name(ch))
    P.mixer.setName(m, P.channels.name(ch))
    if cname:find("808") or cname:find("sub") or cname:find("bass") then
      P.mixer.routeTo(m, bus_808)
    elseif cname:find("kick") or cname:find("snare") or cname:find("clap") or cname:find("hat") or cname:find("perc") or cname:find("rim") then
      P.mixer.routeTo(m, bus_drums)
      if cname:find("kick") and not kick_mixer then
        kick_mixer = m
      end
      if cname:find("snare") or cname:find("clap") then
        P.mixer.setSend(m, fx_room, 0.05)
      end
    elseif cname:find("vocal") or cname:find("voice") or cname:find("chant") then
      P.mixer.routeTo(m, bus_vocal)
      P.mixer.setSend(m, fx_room, 0.10)
    else
      P.mixer.routeTo(m, bus_music)
      P.mixer.setSend(m, fx_room, 0.10)
    end
  end

  if kick_mixer then
    P.mixer.setSend(kick_mixer, bus_808, 1.0, true)
    P.mixer.setSend(kick_mixer, bus_music, 0.75, true)
  end
end
