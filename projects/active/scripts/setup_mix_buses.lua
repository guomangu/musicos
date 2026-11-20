-- name: Setup Mix Buses
-- category: Mixer
-- description: Routes channels to dedicated mixer inserts and prepares Drum / Music / Vocal buses.

function apply()
  local P = project
  local bus_drums = 10
  local bus_music = 11
  local bus_vocal = 12

  P.mixer.setName(bus_drums, "BUS Drums")
  P.mixer.setName(bus_music, "BUS Music")
  P.mixer.setName(bus_vocal, "BUS Vocal")

  for ch = 1, P.channels.count() do
    local m = P.channels.route(ch)
    if m == 0 then
      m = ch
      P.channels.setRoute(ch, m)
    end
    local cname = string.lower(P.channels.name(ch))
    P.mixer.setName(m, P.channels.name(ch))
    if cname:find("kick") or cname:find("snare") or cname:find("clap") or cname:find("hat") or cname:find("perc") then
      P.mixer.routeTo(m, bus_drums)
    elseif cname:find("vocal") or cname:find("voice") or cname:find("chant") then
      P.mixer.routeTo(m, bus_vocal)
    else
      P.mixer.routeTo(m, bus_music)
    end
  end
end
