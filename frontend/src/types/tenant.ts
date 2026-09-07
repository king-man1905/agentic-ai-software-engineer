export type Role =
  | 'OWNER'
  | 'ADMIN'
  | 'ENGINEER'
  | 'REVIEWER'
  | 'SECURITY_REVIEWER'
  | 'VIEWER';

export type Permission =
  | 'ORG_MANAGE'
  | 'USER_MANAGE'
  | 'REPO_MANAGE'
  | 'REPO_READ'
  | 'POLICY_MANAGE'
  | 'POLICY_READ'
  | 'RUN_CREATE'
  | 'RUN_READ'
  | 'RUN_APPROVE'
  | 'SECURITY_APPROVE'
  | 'RUN_CANCEL'
  | 'AUDIT_READ';

export interface User {
  id: string;
  email: string;
  name: string;
  key_prefix?: string | null;
}

export interface Organization {
  id: string;
  name: string;
  created_at: string;
  status: string;
}

export interface TenantContext {
  organization_id: string;
  organization_name: string;
  user_id: string;
  user_name: string;
  role: Role;
  permissions: Permission[];
}
