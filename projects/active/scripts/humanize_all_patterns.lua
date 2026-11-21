-- name: Humanize All Patterns (Transient-Lock Safe)
-- category: Groove
-- description: Applies subtle timing & velocity humanization while keeping Kick/808/Snare/Clap locked on beat grid.

function form(f)
  f:knobInt("MaxTicks", 5, 1, 16, "Maximum timing jitter in ticks (melodic/hats only)")
  f:knobInt("VelRange", 10, 1, 30, "Velocity variation (+/-)")
end

function apply(p)
  p = p or {}
  local max_ticks = p.MaxTicks or 5
  local vel_range = p.VelRange or 10
  local P = project
  local n_ch = P.channels.count()
  local n_pat = P.patterns.count()

  for pat = 1, n_pat do
    for ch = 1, n_ch do
      local cname = string.lower(P.channels.name(ch))
      -- Preserve exact +/- 0ms composite impact lock on Kick, 808, Snare, Clap
      local is_locked = cname:find("kick") or cname:find("808") or cname:find("sub") or cname:find("snare") or cname:find("clap")
      local notes = P.patterns.notes(pat, ch)
      if notes and #notes > 0 then
        for _, n in ipairs(notes) do
          if not is_locked then
            local dt = math.random(-max_ticks, max_ticks)
            n.time = math.max(0, n.time + dt)
          end
          local dv = math.random(-vel_range, vel_range)
          n.velocity = math.max(20, math.min(127, n.velocity + dv))
        end
        P.patterns.setNotes(pat, ch, notes)
      end
    end
  end
end
