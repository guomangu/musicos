-- name: Humanize All Patterns
-- category: Groove
-- description: Applies subtle timing and velocity humanization across all patterns.

function form(f)
  f:knobInt("MaxTicks", 6, 1, 24, "Maximum timing jitter in ticks")
  f:knobInt("VelRange", 12, 1, 35, "Velocity variation (+/-)")
end

function apply(p)
  p = p or {}
  local max_ticks = p.MaxTicks or 6
  local vel_range = p.VelRange or 12
  local P = project
  local n_ch = P.channels.count()
  local n_pat = P.patterns.count()
  for pat = 1, n_pat do
    for ch = 1, n_ch do
      local notes = P.patterns.notes(pat, ch)
      if notes and #notes > 0 then
        for _, n in ipairs(notes) do
          local dt = math.random(-max_ticks, max_ticks)
          n.time = math.max(0, n.time + dt)
          local dv = math.random(-vel_range, vel_range)
          n.velocity = math.max(15, math.min(127, n.velocity + dv))
        end
        P.patterns.setNotes(pat, ch, notes)
      end
    end
  end
end
