-- RewardConfig.lua
-- Loads reward shaping configuration from HTTP endpoint or falls back to defaults
-- This allows runtime configuration changes without restarting the Roblox client

local HttpService = game:GetService("HttpService")

local RewardConfig = {}

-- Default configuration (fallback if HTTP request fails)
local DEFAULT_CONFIG = {
	progress_rewards = {
		base_reward_per_step = -0.005,
		progress_reward_scale = 3.0,
		progress_reward_cap = -2.0,
		leap_threshold = 2.0,
		leap_bonus = 1.0,
		milestone_threshold = 1.0,
		milestone_bonus = 2.0,
		sustained_threshold = 0.5,
		sustained_bonus = 0.5
	},
	checkpoints = {
		checkpoint_bonus = 20.0,
		completion_base_bonus = 50.0,
		completion_cp_bonus = 10.0
	},
	penalties = {
		death_penalty_hazard = 15.0,
		death_penalty_fall = 8.0,
		death_penalty_other = 10.0,
		stuck_penalty = 8.0
	},
	movement_rewards = {
		heading_shape_scale = 0.002,
		horizontal_progress_scale = 0.15,
		jump_up_bonus = 1.0,
		idle_speed_threshold = 1.0,
		idle_penalty_per_step = 0.02
	},
	hazards = {
		hazard_near_radius = 15.0,
		hazard_avoid_penalty = 0.02,
		safe_progress_threshold = 0.25
	},
	thresholds = {
		backtrack_threshold = 1.0,
		backtrack_penalty_scale = 0.05,
		min_progress_eps = 0.5,
		stuck_steps = 200
	}
}

-- Current loaded configuration
local currentConfig = DEFAULT_CONFIG

-- Load configuration from server
function RewardConfig.loadFromServer(serverUrl)
	serverUrl = serverUrl or "http://127.0.0.1:5000"
	local configUrl = serverUrl .. "/config/reward"
	
	local success, result = pcall(function()
		return HttpService:GetAsync(configUrl, false)
	end)
	
	if success then
		local parseSuccess, config = pcall(function()
			return HttpService:JSONDecode(result)
		end)
		
		if parseSuccess and config then
			currentConfig = config
			print("[RewardConfig] Loaded configuration from server")
			return true
		else
			warn("[RewardConfig] Failed to parse config JSON, using defaults")
		end
	else
		warn("[RewardConfig] Failed to fetch config from server: " .. tostring(result))
	end
	
	return false
end

-- Get a configuration value with path like "progress_rewards.base_reward_per_step"
function RewardConfig.get(path, default)
	local keys = string.split(path, ".")
	local value = currentConfig
	
	for _, key in ipairs(keys) do
		if type(value) == "table" and value[key] ~= nil then
			value = value[key]
		else
			return default
		end
	end
	
	return value
end

-- Get an entire section
function RewardConfig.getSection(sectionName)
	return currentConfig[sectionName] or {}
end

-- Get all configuration
function RewardConfig.getAll()
	return currentConfig
end

-- Reload configuration from server
function RewardConfig.reload(serverUrl)
	return RewardConfig.loadFromServer(serverUrl)
end

-- Initialize with defaults
RewardConfig.currentConfig = currentConfig

return RewardConfig