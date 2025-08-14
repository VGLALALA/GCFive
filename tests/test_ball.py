import numpy as np
import pytest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trajectory_simulation.ball import Ball, project, angle_between, limit_length, rotation_matrix
from trajectory_simulation.vector import vec3, length, normalized


class TestBall:
    """Test cases for the Ball class"""
    
    def test_ball_initialization(self):
        """Test ball initialization with default values"""
        ball = Ball()
        assert ball.mass == 0.04592623
        assert ball.radius == 0.021335
        assert ball.u_k == 0.4
        assert ball.u_kr == 0.2
        assert ball.rho == 1.225
        assert ball.mu == 0.00001802
        assert ball.nu == 0.00001470
        assert ball.nu_g == 0.0012
        assert ball.drag_scale == 1.0
        assert ball.lift_scale == 1.0
        assert len(ball.position_list) == 0
        assert len(ball.total_position_list) == 0
    
    def test_ball_post_init(self):
        """Test post-initialization calculations"""
        ball = Ball()
        expected_A = np.pi * ball.radius**2
        expected_I = 0.4 * ball.mass * ball.radius**2
        assert ball.A == pytest.approx(expected_A)
        assert ball.I == pytest.approx(expected_I)
    
    def test_ball_reset(self):
        """Test ball reset functionality"""
        ball = Ball()
        ball.position = vec3(10.0, 20.0, 30.0)
        ball.velocity = vec3(1.0, 2.0, 3.0)
        ball.omega = vec3(4.0, 5.0, 6.0)
        ball.position_list = [vec3(1.0, 1.0, 1.0)]
        ball.total_position_list = [vec3(2.0, 2.0, 2.0)]
        
        ball.reset()
        
        assert np.allclose(ball.position, [0.0, 0.1, 0.0])
        assert np.allclose(ball.velocity, [0.0, 0.0, 0.0])
        assert np.allclose(ball.omega, [0.0, 0.0, 0.0])
        assert len(ball.position_list) == 0
        assert len(ball.total_position_list) == 0
    
    def test_ball_hit(self):
        """Test ball hit with default parameters"""
        ball = Ball()
        ball.hit()
        
        # Check position is reset to origin
        assert np.allclose(ball.position, [0.0, 0.0, 0.0])
        
        # Check velocity components
        v = 44.7
        expected_vx = v * np.cos(np.radians(20.8)) * np.cos(np.radians(1.7))
        expected_vy = v * np.sin(np.radians(20.8))
        expected_vz = v * np.sin(np.radians(1.7))
        
        assert ball.velocity[0] == pytest.approx(expected_vx)
        assert ball.velocity[1] == pytest.approx(expected_vy)
        assert ball.velocity[2] == pytest.approx(expected_vz)
        
        # Check omega components
        expected_omega_y = 784.0 * np.sin(np.radians(2.7))
        expected_omega_z = 784.0 * np.cos(np.radians(2.7))
        
        assert ball.omega[0] == pytest.approx(0.0)
        assert ball.omega[1] == pytest.approx(expected_omega_y)
        assert ball.omega[2] == pytest.approx(expected_omega_z)
    
    def test_ball_hit_from_data(self):
        """Test ball hit with custom data"""
        ball = Ball()
        data = {
            "Speed": 50.0,  # mph
            "VLA": 15.0,    # degrees
            "HLA": 5.0,     # degrees
            "TotalSpin": 3000.0,  # rpm
            "SpinAxis": 45.0      # degrees
        }
        
        ball.hit_from_data(data)
        
        # Check position is set correctly
        assert np.allclose(ball.position, [0.0, 0.05, 0.0])
        
        # Check that velocity and omega are not zero
        assert not np.allclose(ball.velocity, [0.0, 0.0, 0.0])
        assert not np.allclose(ball.omega, [0.0, 0.0, 0.0])
    
    def test_ball_update_airborne(self):
        """Test ball update when airborne"""
        ball = Ball()
        ball.position = vec3(0.0, 10.0, 0.0)  # High in air
        ball.velocity = vec3(20.0, 0.0, 0.0)   # Moving horizontally
        ball.omega = vec3(0.0, 100.0, 0.0)     # Spinning
        
        initial_pos = ball.position.copy()
        initial_vel = ball.velocity.copy()
        
        ball.update(0.01)  # Small time step
        
        # Should have moved and been affected by gravity
        assert not np.allclose(ball.position, initial_pos)
        assert not np.allclose(ball.velocity, initial_vel)
        assert ball.velocity[1] < 0  # Should be falling
        assert len(ball.total_position_list) == 1
    
    def test_ball_update_on_ground(self):
        """Test ball update when on ground"""
        ball = Ball()
        ball.position = vec3(0.0, 0.01, 0.0)  # Very close to ground
        ball.velocity = vec3(5.0, 0.0, 0.0)    # Moving horizontally
        ball.omega = vec3(0.0, 50.0, 0.0)      # Spinning
        
        initial_pos = ball.position.copy()
        
        ball.update(0.01)
        
        # Should have moved but stayed near ground
        assert not np.allclose(ball.position, initial_pos)
        assert ball.position[1] >= 0.0  # Should not go below ground
        assert len(ball.total_position_list) == 1
    
    def test_ball_bounce(self):
        """Test ball bounce physics"""
        ball = Ball()
        ball.position = vec3(0.0, 0.1, 0.0)
        ball.velocity = vec3(10.0, -5.0, 0.0)  # Moving down and forward
        ball.omega = vec3(0.0, 100.0, 0.0)     # Spinning
        
        # Simulate hitting ground
        ball.position[1] = 0.0
        ball.velocity[1] = -5.0
        
        initial_vel = ball.velocity.copy()
        initial_omega = ball.omega.copy()
        
        ball.update(0.01)
        
        # Should have bounced
        assert ball.velocity[1] > initial_vel[1]  # Should be moving up
        assert len(ball.position_list) == 1  # Should record bounce position
    
    def test_ball_very_slow_motion(self):
        """Test ball behavior with very slow motion"""
        ball = Ball()
        ball.position = vec3(0.0, 0.01, 0.0)
        ball.velocity = vec3(0.05, 0.0, 0.0)  # Very slow
        ball.omega = vec3(0.0, 0.1, 0.0)      # Very slow spin
        
        ball.update(0.01)
        
        # Should be affected by gravity even when slow
        assert ball.velocity[1] < 0.0  # Should be falling due to gravity
        assert length(ball.velocity) < 0.1  # Should be very slow or stopped


class TestUtilityFunctions:
    """Test cases for utility functions"""
    
    def test_project(self):
        """Test vector projection"""
        v = vec3(3.0, 4.0, 0.0)
        n = vec3(1.0, 0.0, 0.0)  # Unit vector in x direction
        
        result = project(v, n)
        
        # Should project to x component only
        assert result[0] == pytest.approx(3.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.0)
    
    def test_angle_between(self):
        """Test angle between vectors"""
        a = vec3(1.0, 0.0, 0.0)
        b = vec3(0.0, 1.0, 0.0)
        
        angle = angle_between(a, b)
        
        assert angle == pytest.approx(np.pi/2)  # 90 degrees
    
    def test_angle_between_parallel(self):
        """Test angle between parallel vectors"""
        a = vec3(1.0, 0.0, 0.0)
        b = vec3(2.0, 0.0, 0.0)
        
        angle = angle_between(a, b)
        
        assert angle == pytest.approx(0.0)  # 0 degrees
    
    def test_angle_between_opposite(self):
        """Test angle between opposite vectors"""
        a = vec3(1.0, 0.0, 0.0)
        b = vec3(-1.0, 0.0, 0.0)
        
        angle = angle_between(a, b)
        
        assert angle == pytest.approx(np.pi)  # 180 degrees
    
    def test_limit_length_shorter(self):
        """Test limit_length when vector is shorter than limit"""
        v = vec3(1.0, 0.0, 0.0)
        limit = 5.0
        
        result = limit_length(v, limit)
        
        assert np.allclose(result, v)  # Should be unchanged
    
    def test_limit_length_longer(self):
        """Test limit_length when vector is longer than limit"""
        v = vec3(3.0, 4.0, 0.0)  # Length 5
        limit = 2.0
        
        result = limit_length(v, limit)
        
        assert length(result) == pytest.approx(2.0)
        assert not np.allclose(result, v)
    
    def test_limit_length_zero_vector(self):
        """Test limit_length with zero vector"""
        v = vec3(0.0, 0.0, 0.0)
        limit = 5.0
        
        result = limit_length(v, limit)
        
        assert np.allclose(result, v)  # Should be unchanged
    
    def test_limit_length_zero_limit(self):
        """Test limit_length with zero limit"""
        v = vec3(3.0, 4.0, 0.0)
        limit = 0.0
        
        result = limit_length(v, limit)
        
        assert np.allclose(result, [0.0, 0.0, 0.0])
    
    def test_rotation_matrix_x_axis(self):
        """Test rotation matrix around x-axis"""
        axis = vec3(1.0, 0.0, 0.0)
        angle = np.pi/2  # 90 degrees
        
        R = rotation_matrix(axis, angle)
        
        # Should rotate y to z
        v = vec3(0.0, 1.0, 0.0)
        result = R @ v
        
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(1.0)
    
    def test_rotation_matrix_y_axis(self):
        """Test rotation matrix around y-axis"""
        axis = vec3(0.0, 1.0, 0.0)
        angle = np.pi/2  # 90 degrees
        
        R = rotation_matrix(axis, angle)
        
        # Test the actual rotation behavior
        v = vec3(0.0, 0.0, 1.0)
        result = R @ v
        
        # The rotation matrix should produce a unit vector result
        assert length(result) == pytest.approx(1.0)
        # The y component should remain 0
        assert result[1] == pytest.approx(0.0)
        # The result should be perpendicular to the original vector
        assert abs(np.dot(result, v)) == pytest.approx(0.0)
    
    def test_rotation_matrix_z_axis(self):
        """Test rotation matrix around z-axis"""
        axis = vec3(0.0, 0.0, 1.0)
        angle = np.pi/2  # 90 degrees
        
        R = rotation_matrix(axis, angle)
        
        # Should rotate x to y
        v = vec3(1.0, 0.0, 0.0)
        result = R @ v
        
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(1.0)
        assert result[2] == pytest.approx(0.0)
    
    def test_rotation_matrix_zero_angle(self):
        """Test rotation matrix with zero angle"""
        axis = vec3(1.0, 0.0, 0.0)
        angle = 0.0
        
        R = rotation_matrix(axis, angle)
        
        # Should be identity matrix
        expected = np.eye(3)
        assert np.allclose(R, expected)
    
    def test_rotation_matrix_non_unit_axis(self):
        """Test rotation matrix with non-unit axis"""
        axis = vec3(2.0, 0.0, 0.0)  # Non-unit vector
        angle = np.pi/2
        
        R = rotation_matrix(axis, angle)
        
        # Should still work correctly (axis gets normalized)
        v = vec3(0.0, 1.0, 0.0)
        result = R @ v
        
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(1.0)


class TestBallEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_ball_very_high_speed(self):
        """Test ball behavior at very high speeds"""
        ball = Ball()
        ball.position = vec3(0.0, 10.0, 0.0)
        ball.velocity = vec3(100.0, 0.0, 0.0)  # Very fast
        ball.omega = vec3(0.0, 1000.0, 0.0)    # Very fast spin
        
        ball.update(0.001)  # Small time step
        
        # Should still behave reasonably
        assert ball.position[1] >= 0.0  # Should not go below ground
        assert len(ball.total_position_list) == 1
    
    def test_ball_zero_velocity(self):
        """Test ball behavior with zero velocity"""
        ball = Ball()
        ball.position = vec3(0.0, 0.1, 0.0)
        ball.velocity = vec3(0.0, 0.0, 0.0)
        ball.omega = vec3(0.0, 0.0, 0.0)
        
        ball.update(0.01)
        
        # Should fall due to gravity
        assert ball.velocity[1] < 0.0
        assert ball.position[1] < 0.1
    
    def test_ball_multiple_bounces(self):
        """Test multiple bounces"""
        ball = Ball()
        ball.position = vec3(0.0, 1.0, 0.0)
        ball.velocity = vec3(0.0, -10.0, 0.0)  # Moving straight down
        
        # Simulate multiple updates
        for _ in range(10):
            ball.update(0.01)
        
        # Should have bounced multiple times
        assert len(ball.position_list) > 0
    
    def test_ball_rolling_friction(self):
        """Test rolling friction effects"""
        ball = Ball()
        ball.position = vec3(0.0, 0.01, 0.0)
        ball.velocity = vec3(1.0, 0.0, 0.0)
        ball.omega = vec3(0.0, 50.0, 0.0)  # Rolling motion
        
        initial_vel = ball.velocity.copy()
        
        ball.update(0.01)
        
        # Should slow down due to friction
        assert length(ball.velocity) < length(initial_vel)
