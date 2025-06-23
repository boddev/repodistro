Refactor the provided Web APIs to enable secure integration with Microsoft Power Platform, implementing the following specifications:

1. Configure authentication to support Power Platform requirements:
   - Use Azure AD authentication
   - Set redirect URI to: https://{environment-name}.flow.microsoft.com/manage/oauthresponse
   - Enable CORS for Power Platform domains (*.flow.microsoft.com)

2. Implement the following security measures:
   - Add OAuth 2.0 authorization flows
   - Include proper scopes for API permissions
   - Implement rate limiting and request validation
   - Add API versioning headers

3. For each API endpoint:
   - Document the complete request/response schema
   - Specify required authentication headers
   - List all supported HTTP methods
   - Include example requests and responses
   - Document error handling and status codes

4. Maintain existing functionality while adding:
   - Standardized response formats
   - Proper HTTP status codes
   - Input validation
   - Error handling middleware
   - Request/response logging

5. Include comprehensive XML documentation for:
   - Method descriptions
   - Parameter definitions
   - Response types
   - Authentication requirements
   - Rate limiting details

Please provide the current API code for refactoring with these specifications.

Agent
Create a GitHub API integration agent with the following specifications:

1. Authentication Requirements:
- Implement secure OAuth or Personal Access Token authentication for GitHub API
- Store credentials securely following best practices
- Handle token refresh and expiration gracefully

2. API Integration:
- Use GitHub REST API v3 or GraphQL API v4
- Implement rate limiting and error handling
- Log all API interactions for debugging
- Follow GitHub API best practices for requests

3. Core Functionality:
- Define specific GitHub operations to perform (e.g., repo management, issue tracking)
- Implement request validation and sanitization
- Structure API responses for consistent data handling
- Include timeout and retry mechanisms

4. User Interaction:
- Specify the format of user inputs and expected responses
- Define clear success/error messages
- Implement input validation and error handling
- Document all available commands and actions

5. Custom Actions:
- List specific actions the agent should support
- Define the trigger conditions for each action
- Specify the expected outcomes and error states
- Include rollback mechanisms for failed operations

Please provide:
- Required GitHub scopes and permissions
- Target GitHub API endpoints
- Expected user interaction flow
- Error handling requirements
- Performance and security constraints