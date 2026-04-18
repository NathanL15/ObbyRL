-- AgentClient.client.lua (client-side, uses RemoteFunction RLStep)
-- Expects:
--   � Workspace/Checkpoints/CP_1, CP_2, ...
--   � ReplicatedStorage/RemoteFunction RLStep  (server calls Python)
--   � ServerScriptService/RLServer.lua (from previous message)

-- ========= CONFIG =========
local STEP_DT       = 0.08
local ACTION_DECISION_DT = 0.25       -- increased to reduce HTTP rate limiting
local CHECK_RADIUS  = 6               -- forgiving early on
local FALL_Y        = -20
local DOWN_RAY      = 50
local FWD_RAY       = 50
local VERBOSE       = true
local LOG_EVERY     = 20              -- Increased from 10 to reduce log spam
local TIMING_LOG_EVERY = 10
-- ==========================
-- Progress shaping / anti-loop config
local BACKTRACK_THRESH = 1.0          -- studs increase in distance considered a backtrack
local BACKTRACK_PENALTY_SCALE = 0.05  -- penalty per stud of backtrack beyond threshold
local STUCK_STEPS = 200               -- Increased to allow longer episodes
local MIN_PROGRESS_EPS = 0.5          -- need at least this much dist improvement to count as progress
local STUCK_PENALTY = 8               -- penalty when flagged stuck (separate from normal -10 death)
local HEADING_SHAPE_SCALE = 0.002     -- reward scale for looking toward target (0 to disable)
local HORIZ_PROGRESS_SCALE = 0.15     -- Increased to reward horizontal progress more
local JUMP_UP_BONUS = 1.0             -- bonus when upward velocity during approach & leaving ground
local IDLE_SPEED_THRESH = 1.0         -- below this horizontal speed counts as idle
local IDLE_PENALTY = 0.02             -- per step penalty while idle
local MAX_TIME_SINCE_JUMP = 5.0       -- cap for observation normalization
local BASE_WALK_SPEED = 16            -- baseline Humanoid WalkSpeed
local FORWARD_JUMP_SPEED = 50         -- Increased for better platform jumps
local JUMP_FORWARD_IMPULSE = 60       -- Increased impulse for forward jumps
local FACE_TARGET_DURING_AIR = false  -- keep facing next checkpoint while airborne to reduce spin (set true to force)
local AIR_STEER_KEEP_VEL = true
-- Hazard config
local HAZARD_TAG = "Hazard"
local HAZARD_NEAR_RADIUS = 15         -- start avoidance shaping within this radius
local HAZARD_AVOID_PENALTY = 0.02     -- per step penalty when moving toward nearby hazard
local SAFE_PROGRESS_IMPROVE = 0.25    -- improvement threshold to log safe platform
local START_SPAWN_NAME = "StartSpawn"
local function getStartSpawn()
	-- deep search so it can live inside a Folder
	return workspace:FindFirstChild(START_SPAWN_NAME, true)
end


local Players           = game:GetService("Players")
local RunService        = game:GetService("RunService")
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local CollectionService = game:GetService("CollectionService")

local RLStep = ReplicatedStorage:WaitForChild("RLStep")

local function now()
	return string.format("%.3f", os.clock())
end
local function v3(v)
	return string.format("(%.2f, %.2f, %.2f)", v.X, v.Y, v.Z)
end
local function log(fmt, ...)
	if VERBOSE then
		print(string.format("[%s] ", now()) .. string.format(fmt, ...))
	end
end
local function warnf(fmt, ...)
	warn(string.format("[%s] ", now()) .. string.format(fmt, ...))
end

local player = Players.LocalPlayer
if not player then
	warnf("No LocalPlayer (use Play, not Run)")
end
local char = player.Character or player.CharacterAdded:Wait()
local hrp  = char:WaitForChild("HumanoidRootPart")
local hum  = char:WaitForChild("Humanoid")

local CHECKPOINT_FOLDER = workspace:FindFirstChild("Checkpoints")
assert(CHECKPOINT_FOLDER, "Workspace/Checkpoints folder is required with CP_1, CP_2, ...")

local function getCP(idx)
	return CHECKPOINT_FOLDER:FindFirstChild("CP_" .. tostring(idx))
end
local function getCPPos(idx)
	local p = getCP(idx)
	return p and p.Position or nil
end

local function rayDist(origin, dir, len)
	local params = RaycastParams.new()
	params.FilterDescendantsInstances = {char}
	params.FilterType = Enum.RaycastFilterType.Exclude
	local result = workspace:Raycast(origin, dir.Unit * len, params)
	if result then
		return (result.Position - origin).Magnitude
	else
		return len
	end
end

-- allow an optional CP_0 (acts as first learning target). If absent we start at CP_1.
local HAS_CP0 = (CHECKPOINT_FOLDER:FindFirstChild("CP_0") ~= nil)
local INITIAL_CP_INDEX = HAS_CP0 and 0 or 1
local MAX_CP_INDEX = (function()
	local maxIdx = INITIAL_CP_INDEX
	for _,child in ipairs(CHECKPOINT_FOLDER:GetChildren()) do
		if child.Name:match("^CP_%d+$") then
			local n = tonumber(child.Name:match("CP_(%d+)$"))
			if n and n > maxIdx then maxIdx = n end
		end
	end
	return maxIdx
end)()
log("Detected max checkpoint index = %d", MAX_CP_INDEX)
local nextCP = INITIAL_CP_INDEX
local lastDist = math.huge
local lastPotential = 0            -- -lastDist after first distance measurement
local died = false
local stepCounter = 0
local episodeCounter = 1
local lastObs = nil
local currentAction = 0               -- cached action applied every frame for smooth motion
local bestDistThisCP = math.huge      -- best (smallest) distance observed since current checkpoint became target
local noProgressSteps = 0             -- counter for stagnation
local milestoneDist = math.huge       -- last distance threshold that yielded milestone reward
local lastDeathType = 0               -- 0 none,1 hazard,2 fall,3 other
local hazardTouchTime = -1
local safePlatformCFrame = nil
local timeSinceJump = 0
local wasGrounded = true

local decisionCallCount = 0
local totalRequestTime = 0
local requestTimes = {}
local maxStoredTimes = 100

-- Convert action id -> desired move vector (world space forward/right relative to HRP)
local function actionToMove(action)
	if action == 1 then
		return hrp.CFrame.LookVector
	elseif action == 2 then
		return -hrp.CFrame.RightVector
	elseif action == 3 then
		return hrp.CFrame.RightVector
	elseif action == 5 then
		return hrp.CFrame.LookVector -- move forward + jump
	elseif action == 6 then
		return -hrp.CFrame.LookVector -- backward
	end
	return Vector3.zero
end

local function faceNextCheckpoint()
	if not FACE_TARGET_DURING_AIR then return end
	local cpPos = getCPPos(nextCP)
	if not cpPos then return end
	local pos = hrp.Position
	local target = Vector3.new(cpPos.X, pos.Y, cpPos.Z)
	local lookDir = target - pos
	if lookDir.Magnitude < 0.001 then return end
	hrp.CFrame = CFrame.new(pos, pos + lookDir.Unit)
end

local function applyForwardJumpBoost()
	local mass = hrp.AssemblyMass
	local forward = hrp.CFrame.LookVector
	local vel = hrp.AssemblyLinearVelocity
	local horiz = Vector3.new(vel.X, 0, vel.Z)
	local desired = forward * FORWARD_JUMP_SPEED
	local add = desired - horiz
	local impulse = Vector3.new(add.X, 0, add.Z) * mass
	hrp:ApplyImpulse(impulse)
end

hum.Died:Connect(function()
	died = true
	if os.clock() - hazardTouchTime < 1.0 then
		lastDeathType = 1
	elseif lastDeathType == 0 then
		lastDeathType = 3
	end
	log("Humanoid.Died fired type=%d", lastDeathType)
end)

local function getNearestHazard()
	local hazards = CollectionService:GetTagged(HAZARD_TAG)
	local nearest, dist = nil, math.huge
	for _,h in ipairs(hazards) do
		if h:IsA("BasePart") then
			local d = (h.Position - hrp.Position).Magnitude
			if d < dist then
				dist = d; nearest = h
			end
		end
	end
	return nearest, dist
end

local function connectHazardTouch(obj)
	if obj:IsA("BasePart") then
		obj.Touched:Connect(function(hit)
			if CollectionService:HasTag(hit, HAZARD_TAG) then
				hazardTouchTime = os.clock()
			end
		end)
	end
end
for _,d in ipairs(char:GetDescendants()) do connectHazardTouch(d) end
char.DescendantAdded:Connect(connectHazardTouch)

hum.WalkSpeed = BASE_WALK_SPEED
hum.AutoRotate = true  -- allow natural rotation toward Move() direction

local function distanceToCP()
	local pos = getCPPos(nextCP)
	if not pos or not hrp or not hrp.Position then return 0 end
	return (pos - hrp.Position).Magnitude
end

local function atCheckpoint()
	local pos = getCPPos(nextCP)
	if not pos then return false end
	return (hrp.Position - pos).Magnitude < CHECK_RADIUS
end

local function getObs()
	local pos = getCPPos(nextCP)
	local dx, dy, dz = 0, 0, 0
	if pos then
		local d = pos - hrp.Position
		dx, dy, dz = d.X, d.Y, d.Z
	end
	local v = hrp.AssemblyLinearVelocity
	local down = rayDist(hrp.Position, Vector3.new(0, -1, 0), DOWN_RAY)
	local forward = rayDist(hrp.Position, hrp.CFrame.LookVector, FWD_RAY)
	local grounded = (down < 5) and 1 or 0
	local cpDir = Vector3.new(dx, dy, dz)
	local look = hrp.CFrame.LookVector
	local angleCos = 0
	if cpDir.Magnitude > 0.001 then
		angleCos = look:Dot(cpDir.Unit)
	end
	local speedH = Vector3.new(v.X, 0, v.Z).Magnitude

	-- Radial horizontal rays (8 directions around agent) for local obstacle / gap sensing
	local radial = {}
	local f = hrp.CFrame.LookVector
	local r = hrp.CFrame.RightVector
	for i = 0,7 do
		local ang = (math.pi * 2) * (i / 8)
		local dir = (f * math.cos(ang) + r * math.sin(ang))
		dir = Vector3.new(dir.X, 0, dir.Z).Unit
		radial[i+1] = rayDist(hrp.Position, dir, FWD_RAY)
	end

	-- Edge downward probes (forward / left / right) a few studs out to detect upcoming gaps
	local EDGE_OFFSET = 3
	local posBase = hrp.Position
	local posF = posBase + f * EDGE_OFFSET
	local posR = posBase + r * EDGE_OFFSET
	local posL = posBase - r * EDGE_OFFSET
	local dropF = rayDist(posF, Vector3.new(0,-1,0), DOWN_RAY)
	local dropR = rayDist(posR, Vector3.new(0,-1,0), DOWN_RAY)
	local dropL = rayDist(posL, Vector3.new(0,-1,0), DOWN_RAY)
	local nearestHazard, hazardDist = getNearestHazard()
	if hazardDist == math.huge then hazardDist = FWD_RAY end
	local hazardDistNorm = math.clamp(hazardDist / FWD_RAY, 0, 1)
	return {
		dx = dx, dy = dy, dz = dz,
		vx = v.X, vy = v.Y, vz = v.Z,
		down = down, forward = forward,
		angle = angleCos,
		grounded = grounded,
		speed = speedH,
		tJump = math.min(timeSinceJump, MAX_TIME_SINCE_JUMP),
		-- radial distances
		r0 = radial[1], r1 = radial[2], r2 = radial[3], r3 = radial[4],
		r4 = radial[5], r5 = radial[6], r6 = radial[7], r7 = radial[8],
		-- edge downward probes
		dropF = dropF, dropR = dropR, dropL = dropL,
		-- hazard features
		hazardDist = hazardDistNorm,
		lastDeathType = lastDeathType,
	}
end

local function resetTo(cpIndex)
	-- Always prefer explicit start spawn if we haven't reached the first active checkpoint yet
	if cpIndex < INITIAL_CP_INDEX then
		local startPart = getStartSpawn()
		if startPart and startPart:IsA("BasePart") then
			hrp.CFrame = CFrame.new(startPart.Position + Vector3.new(0, 5, 0))
		else
			-- fallback: CP_1 if no start part exists
			local cp1 = CHECKPOINT_FOLDER:FindFirstChild("CP_1")
			if cp1 then
				hrp.CFrame = CFrame.new(cp1.Position + Vector3.new(0, 5, 0))
			end
		end
		hrp.AssemblyLinearVelocity = Vector3.zero
		hum:ChangeState(Enum.HumanoidStateType.Landed)
		return
	end

	-- normal case: go to last reached checkpoint
	local back = math.max(INITIAL_CP_INDEX, cpIndex)
	local cp = CHECKPOINT_FOLDER:FindFirstChild("CP_" .. tostring(back))
	if cp then
		hrp.CFrame = CFrame.new(cp.Position + Vector3.new(0, 5, 0))
		hrp.AssemblyLinearVelocity = Vector3.zero
		hum:ChangeState(Enum.HumanoidStateType.Landed)
	end
end


local function resetIfDeadOrFall()
	if hrp.Position.Y < FALL_Y then lastDeathType = 2 end
	if died or hrp.Position.Y < FALL_Y then
		if lastDeathType == 1 and safePlatformCFrame then
			hrP = hrp; hrP.CFrame = safePlatformCFrame + Vector3.new(0,5,0)
			hrP.AssemblyLinearVelocity = Vector3.zero
			hum:ChangeState(Enum.HumanoidStateType.Landed)
		else
			resetTo(nextCP - 1)
		end
		died = false
		return true
	end
	return false
end

log("AgentClient starting. Character at %s", v3(hrp.Position))
if getCP(1) then
	log("Found CP_1 at %s", v3(getCP(1).Position))
else
	warnf("CP_1 not found under Workspace/Checkpoints")
end
lastObs = getObs()
lastDist = distanceToCP()
lastPotential = -lastDist
log("Initial nextCP=%d  dist=%.2f", nextCP, lastDist)
local GAMMA = 0.99  -- discount for potential-based shaping

local accum = 0
local actionAccum = 0                 -- accumulates time until next server decision
local pendingReward = 0              -- batched reward since last decision
RunService.RenderStepped:Connect(function()
	local moveVec = actionToMove(currentAction)
	hum:Move(moveVec, true)
	if FACE_TARGET_DURING_AIR then
		faceNextCheckpoint()
	end
end)

-- Lower-frequency loop: query RL server & update action
RunService.Heartbeat:Connect(function(dt)
	timeSinceJump += dt
	accum += dt
	if accum < STEP_DT then return end
	accum = 0
	stepCounter += 1

	-- Progress reward (difference in distance)
	local dNow = distanceToCP()
	local improvement = lastDist - dNow              -- >0 means closer
	local shaped = improvement
	if shaped < -2 then shaped = -2 end             -- cap regress penalty
	local leapBonus = (improvement >= 1.5) and 2 or 0
	-- Milestone reward: every time bestDist improves by >=0.5 stud beyond previous milestone
	local milestoneBonus = 0
	if dNow + 0.5 < milestoneDist then
		milestoneBonus = 5
		milestoneDist = dNow
	end
	local baseReward = -0.001
	local progressReward = 5.0 * shaped
	local sustainedBonus = 0
	if improvement > 0.3 then
		sustainedBonus = 1.0
	end
	local reward = baseReward + progressReward + leapBonus + milestoneBonus + sustainedBonus
	lastPotential = -dNow
	lastDist = dNow

	if dNow + MIN_PROGRESS_EPS < bestDistThisCP then
		bestDistThisCP = dNow
		noProgressSteps = 0
		if wasGrounded and improvement > SAFE_PROGRESS_IMPROVE then
			safePlatformCFrame = hrp.CFrame
		end
	else
		noProgressSteps += 1
	end
	local forcedStuck = false
	if noProgressSteps >= STUCK_STEPS then
		reward -= STUCK_PENALTY
		forcedStuck = true
		noProgressSteps = 0
	end

	local reached = false
	if atCheckpoint() then
		reward += 50  -- MUCH HIGHER checkpoint incentive (was 20)
		log("Reached CP_%d (dist=%.2f)  +50", nextCP, dNow)
		nextCP += 1
		reached = true
		hrp.AssemblyLinearVelocity = Vector3.zero
		bestDistThisCP = math.huge
		noProgressSteps = 0
		-- If we've surpassed the final checkpoint, treat as terminal episode success
		if nextCP > MAX_CP_INDEX then
			-- Large completion bonus scaled by number of checkpoints
			local completionBonus = 100 + 20 * (MAX_CP_INDEX - INITIAL_CP_INDEX)
			reward += completionBonus
			log("All checkpoints cleared! bonus=%.1f restarting from beginning", completionBonus)
			-- Reset loop
			nextCP = INITIAL_CP_INDEX
			forcedStuck = true
		end
	end

	local done = resetIfDeadOrFall()
	if forcedStuck and not done then
		resetTo(nextCP - 1)
		done = true
		episodeCounter += 1
		lastDist = distanceToCP()
		lastPotential = -lastDist
		log("Episode %d stuck-reset (-%d). New dist=%.2f nextCP=%d dt=%d", episodeCounter, 3, lastDist, nextCP, lastDeathType)
	elseif done then
		if lastDeathType == 1 then
			reward -= 5
		elseif lastDeathType == 2 then
			reward -= 3
		else
			reward -= 4
		end
		episodeCounter += 1
		lastDist = distanceToCP()
		lastPotential = -lastDist
		log("Episode %d reset  (-%.1f). New dist=%.2f  nextCP=%d dt=%d", episodeCounter, -reward, lastDist, nextCP, lastDeathType)
		bestDistThisCP = math.huge
		noProgressSteps = 0
	end

	-- Hazard avoidance shaping
	local hPart, hDist = getNearestHazard()
	if hPart and hDist < HAZARD_NEAR_RADIUS then
		local vel = hrp.AssemblyLinearVelocity
		local velH = Vector3.new(vel.X,0,vel.Z)
		if velH.Magnitude > 1 then
			local toHaz = Vector3.new(hPart.Position.X - hrp.Position.X, 0, hPart.Position.Z - hrp.Position.Z)
			if toHaz.Magnitude > 0 then
				local toward = velH.Unit:Dot(toHaz.Unit)
				if toward > 0.5 then
					reward -= HAZARD_AVOID_PENALTY
				end
			end
		end
	end
	pendingReward += reward
	actionAccum += STEP_DT

	if VERBOSE and (stepCounter % LOG_EVERY == 0 or reached) then
		log("step=%d ep=%d nextCP=%d dist=%.2f r=%.3f [base=%.3f prog=%.3f leap=%d mile=%d sust=%.3f] best=%.2f stuck=%d chk=%s done=%s dt=%d",
			stepCounter, episodeCounter, nextCP, dNow, reward, baseReward, progressReward, leapBonus, milestoneBonus,
			sustainedBonus, bestDistThisCP, noProgressSteps, tostring(reached), tostring(done), lastDeathType)
	end

	if actionAccum >= ACTION_DECISION_DT or done then
		local newObs = getObs()
		local action = 0
		local payload = { obs = newObs, reward = pendingReward, done = done }
		
		local requestStart = os.clock()
		local ok, result = pcall(function()
			return RLStep:InvokeServer(payload)   -- returns an action integer
		end)
		local requestDuration = os.clock() - requestStart
		
		decisionCallCount += 1
		totalRequestTime += requestDuration
		table.insert(requestTimes, requestDuration)
		if #requestTimes > maxStoredTimes then
			table.remove(requestTimes, 1)
		end
		
		if ok then
			action = tonumber(result) or 0
		else
			warnf("RLStep InvokeServer failed: %s", tostring(result))
			action = 0
		end
		
		if VERBOSE and (decisionCallCount % TIMING_LOG_EVERY == 0) then
			local avgTime = totalRequestTime / decisionCallCount
			local recentAvg = 0
			if #requestTimes > 0 then
				local sum = 0
				for _, t in ipairs(requestTimes) do sum += t end
				recentAvg = sum / #requestTimes
			end
			log("Timing stats: call#%d avgAll=%.3fs recent=%.3fs last=%.3fs", 
				decisionCallCount, avgTime, recentAvg, requestDuration)
		end

		-- don't jump while airborne
		local groundedNow = (rayDist(hrp.Position, Vector3.new(0,-1,0), DOWN_RAY) < 5)
		if (action == 4 or action == 5) and not groundedNow then
			action = 1
		end
		currentAction = action
		if VERBOSE and (stepCounter % LOG_EVERY == 0 or reached) then
			log("Decision: action=%d (0=idle,1=fwd,2=left,3=right,4=jump,5=fwdjump,6=back)", currentAction)
		end
		if groundedNow and (action == 4 or action == 5) then
			hum.Jump = true
			if action == 5 then
				applyForwardJumpBoost()
			end
		end

		lastObs = newObs
		lastDeathType = 0
		pendingReward = 0
		actionAccum = 0
	end
end)
